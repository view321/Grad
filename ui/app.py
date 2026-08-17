"""The NiceGUI desktop app: a tiling workspace over twelve windows.

    "The things that make it pleasant -- being able to see a funnel's reasoning,
     a preflight's failing check, a prediction against its outcome -- are the
     same things that make it trustworthy."

The app used to be a row of tabs. It is now a workspace shell: a title bar, a
window opener, a tiling area of resizable panes, and a status bar. What moved
where:

* `ui/tokens.py`   -- the design tokens, and the stylesheet generated from them
* `ui/layout.py`   -- the pane tree and the moves over it, pure and tested
* `ui/models.py`   -- what each window shows, as plain data, pure and tested
* `ui/registry.py` -- the list of windows the whole shell is derived from
* `ui/shell.py`    -- the chrome, and how a window survives a retile
* `ui/windows/`    -- twelve renderers, none of which read a ledger directly

This module keeps only what is genuinely the application's: the SDK session, the
per-client keying, and `run()`.

Two implementation details are the difference between this feeling like a tool
and feeling like a demo, and both survive from the first version:

  * **Buffered flush.** Tokens accumulate in the turn's blocks and a `ui.timer`
    flushes at ~15 Hz, rather than re-rendering a markdown element per token.
  * **Split tail.** The streaming turn lives in its own element, separate from
    the settled transcript above it, so only the tail re-renders -- and inside
    the tail, only the block that moved.

Notebooks render, they do not rebuild: JupyterLab already exists and is better
at editing. Building a notebook editor is the single easiest way to burn a month
on this project.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import sys
from pathlib import Path
from typing import Any

from core import appdata, config as config_mod, effort, instance, migrate, paths, rewind
from ui import desktop, katex, kit, render, sessions, shell, state as state_mod
#: Roles the chat window knows how to draw, and therefore the ones `restore`
#: keeps. `system` is the compaction marker: not something anyone said, but the
#: record of the moment the agent's memory of this transcript was replaced --
#: which is the one event a reader needs in order to interpret everything above
#: it correctly.
ROLES = ("user", "assistant", "system")
STATIC_URL = "/grad-static"

#: The port `run()` bound, so the rest of the app can name its own origin. The
#: embedded Lab scopes its framing headers to one origin and is started by a
#: button in the notebook window, which would otherwise have to assume the
#: default port and be wrong on every `--port`. Set at launch rather than read
#: from the environment because `ui.run` is where the number is decided.
PORT = 8080

#: How long the SDK's own interrupt is given to end the turn before the client
#: is taken down instead. The same shape as `ui/tasks.py:cancel` -- ask the thing
#: that knows how to stop cleanly, then stop it anyway -- and for the same
#: reason: a control that reports "interrupting…" and leaves the session busy
#: forever is worse than no control, because the composer then silently refuses
#: every prompt after it.
INTERRUPT_GRACE_S = 8.0

#: What the CLI says when a truncating resume would discard more than the one
#: turn it was told about. Matched on the text because that is what the SDK
#: documents as the contract -- there is no distinct exception class for it --
#: and `Session._connect` treats it as "resume without the check" rather than as
#: a failure, because by then the transcript has already been rewound.
_REWIND_REFUSED = "Resume rejected by --resume-drops-turn:"
#: How often `_stop_turn` checks whether the turn it asked to stop has settled.
SETTLE_POLL_S = 0.1

# Where anything the transcript must not carry goes instead: this handler is the
# app's own log, not user-visible text and not the persisted session file.
log = logging.getLogger("grad.ui")


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


class Session:
    """Owns the `ClaudeSDKClient` and the turn in flight.

    The UI holds no logic of its own beyond this: everything else it shows is
    read from the ledger or produced by the CLIs.

    One of these per connected client, keyed by `key`: two windows sharing a
    single `ClaudeSDKClient` would interleave their turns into one conversation
    and overwrite each other's transcript file.
    """

    def __init__(self, key: str = "default") -> None:
        self.key = key
        # A live client, so a claim it holds is not stale. Registering here
        # rather than at claim time means `most_recent` can tell "another window
        # is in this session" from "the window that was in it is gone".
        sessions.register(key)
        #: Where a message that has nowhere else to go is put on screen. Set by
        #: `build` to the workspace's status bar; None in tests and on the CLI.
        #: An interrupt that failed used to be swallowed entirely, which is how
        #: pressing STOP twice became the way to stop a turn.
        self.notify: Any = None
        self.client: Any = None
        #: The turn in flight, as `agent.TurnStream` blocks: prose, and the tool
        #: calls between it. The chat window's timer draws from here, so a card
        #: for a running command is on screen while it runs.
        self.blocks: list[dict[str, Any]] = []
        self.settled: list[dict[str, Any]] = []
        self.busy = False
        #: Which named session this is. `sessions.LEGACY_ID` on an upgraded
        #: machine, because the transcript that was already there is one.
        self.session_id: str = sessions.LEGACY_ID
        self.title: str = ""
        self.created_at: str | None = None
        #: The *SDK's* id for this conversation, which is what `resume` takes.
        #: Ours names the file; this one is what makes reopening continue rather
        #: than merely redisplay. See the note in `ui/sessions.py`.
        self.sdk_session_id: str | None = None
        #: Set while no turn is in flight, so an interrupt can wait for the turn
        #: it asked to stop without polling `busy` -- which the *next* turn sets
        #: back to True, and a waiter watching that flag would never wake.
        self._idle = asyncio.Event()
        self._idle.set()
        #: The interrupt in progress, if one is. Held so the next turn can wait
        #: for it: a fire-and-forget interrupt can outlive the turn it was aimed
        #: at and land on the one after it.
        self._stopping: asyncio.Task[None] | None = None
        #: The last reading from `get_context_usage`, or None before the first
        #: one. The statusline draws it; `ui/models.py:context_model` decides
        #: what it means. None is a state the meter renders, not an error.
        self.context: dict[str, Any] | None = None
        #: One reading at a time. The call is a control request over the same
        #: transport a turn is streaming on, and a slow one must not be able to
        #: queue a second behind it every time the poll timer fires.
        self._reading_context = False
        #: Where the next client should stop loading the conversation, set by a
        #: rewind and consumed by the `start()` that follows it. It is state
        #: rather than an argument because the two are separated by whatever
        #: rebuilds the client next -- `rewind_to` drops it and `ask` builds the
        #: replacement a turn later, and a rewind that had to be followed
        #: immediately by a turn would be a rewind you could not think about.
        self._rewind_at: str | None = None
        #: The prompt that rewind means to discard, for the CLI to check the
        #: truncation against. Only set when exactly one turn is going -- the
        #: check is "everything after the anchor belongs to this turn", which a
        #: multi-turn rewind contradicts by design.
        self._rewind_drops: str | None = None
        #: The last transcript entry the SDK named in the turn in flight. Stamped
        #: onto the settled record so a later rewind knows where that turn ended;
        #: see `core/rewind.py`.
        self._turn_uuid: str | None = None
        #: The handover note a compaction left, waiting for somewhere to go. It
        #: rides in front of the next prompt rather than being sent as a turn of
        #: its own, which would spend a round-trip to produce an answer nobody
        #: asked for. Cleared once it has been sent.
        self.pending_seed: str | None = None
        #: The reasoning effort the live client was *built* with. The SDK has no
        #: control request for effort -- only `set_model` and
        #: `set_permission_mode` -- so this is fixed at construction, and a
        #: change means a new client. Recorded here so `apply_effort` can tell a
        #: real change from a click that landed on the level already running.
        self.client_effort: str | None = None
        #: The `research` model the live client was *built* with, for the same
        #: reason and with the same consequence. A project may override it
        #: (`core/budget.py:configure`), so switching to one that does while a
        #: session is live would otherwise leave the previous model answering --
        #: silently, and while the projects window shows the new one.
        self.client_model: str | None = None
        #: Held across `start` and `close`, so one can never run inside the
        #: other. The SDK's `connect` is not safe against a concurrent
        #: `disconnect`: disconnect nulls the client's transport while connect
        #: is still between "spawn the CLI" and "build the query around the
        #: transport", and the query comes out built around None. That is not
        #: hypothetical -- a websocket blip during the first, slowest connect
        #: (a cold `claude.exe` under a fresh install) fired the disconnect
        #: handler mid-spawn, and every first prompt died on an AttributeError
        #: deep in the SDK.
        self._lifecycle = asyncio.Lock()

    async def start(self) -> None:
        if self.client is not None:
            return
        import agent  # noqa: PLC0415 - imported here so the UI can load without the SDK
        from claude_agent_sdk import ClaudeSDKClient  # noqa: PLC0415

        async with self._lifecycle:
            # Re-checked under the lock: a start that waited here waited on
            # another start, and the client it built is the one to use.
            if self.client is not None:
                return
            cfg = config_mod.load()
            agent.preflight_environment()
            await self._connect(agent, ClaudeSDKClient, cfg)
            # Consumed, not kept. A rewind aims at one rebuild; leaving these set
            # would have the *next* one -- an effort change, an interrupt, a
            # model switch -- silently truncate the conversation again, at a
            # point that by then is several turns in the past.
            self._rewind_at = None
            self._rewind_drops = None
            # Recorded after the client exists, so a construction that raised
            # does not leave this claiming a level nothing is running at -- which
            # would have `apply_effort` decide there was nothing to rebuild.
            self.client_effort = effort.current(cfg)
            self.client_model = cfg.model_for("research")

    async def _connect(self, agent: Any, client_cls: Any, cfg: Any) -> None:
        """Build the client and enter it, retrying once past a refused rewind.

        `resume_drops_turn` is a claim about what a truncating resume will
        discard, and the CLI refuses the resume outright when the claim does not
        hold -- a message queued while the turn ran, a wake the session absorbed.
        The SDK documents that refusal as deterministic and says to clear the
        check and resume plainly rather than retry it, so that is what this does:
        the rewind still happens, unvalidated, and the reason is logged.

        Retrying without it rather than failing is the right trade because the
        alternative is a session that cannot be reconnected at all. The
        transcript has already been rewound by this point, so a refusal that
        propagated would leave the window showing turns the agent is not being
        rebuilt to match.
        """
        try:
            await self._enter(agent, client_cls, cfg)
            return
        except Exception as exc:  # noqa: BLE001 - narrowed by the message below
            if not (self._rewind_drops and _REWIND_REFUSED in str(exc)):
                raise
            log.warning("the rewind check was refused; resuming without it", exc_info=exc)
            self.say(
                "more than that one turn had been queued into this conversation — "
                "rewinding anyway, without the check"
            )
        self._rewind_drops = None
        await self._enter(agent, client_cls, cfg)

    async def _enter(self, agent: Any, client_cls: Any, cfg: Any) -> None:
        """One attempt at a live client, leaving nothing behind if it fails.

        `self.client` is cleared on the way out of a failure because `start()`
        returns early when it is set: a half-built client left in place is a
        session that can never connect again and never says why. That was
        reachable before a rewind existed -- any raise from `__aenter__` did it
        -- and the retry above would have made it reachable twice.
        """
        client = client_cls(
            options=agent.build_options(
                cfg,
                resume=self.sdk_session_id,
                resume_at=self._rewind_at,
                drops_turn=self._rewind_drops,
            )
        )
        self.client = client
        try:
            await client.__aenter__()
        except BaseException:
            self.client = None
            raise

    # -- named sessions -----------------------------------------------------
    def adopt(self) -> None:
        """Take a session for this client, and load it.

        The most recent one *no other window is in*, or a new one when they are
        all taken. Two clients on one session is two writers on one file --
        `_persist` writes the whole thing, so the loser's turns disappear -- and
        `Session` has been one-per-client since it was written, for exactly that
        reason. What changed is only that the file is now chosen rather than
        derived from the client's own key, so the claim has to be explicit.

        Opening the most recent rather than a blank one is deliberate: reopening
        the app is not a request to start over.
        """
        chosen = sessions.most_recent(self.key)
        if chosen is None:
            # A fresh workspace, or every existing session already open in
            # another window. Either way this client needs one nobody else can
            # be in, and a **fresh id is the only thing that guarantees that**.
            #
            # This used to reach for `LEGACY_ID` when the listing was empty, on
            # the theory that a first session should be named where the
            # pre-sessions code wrote. That was a race with no upside: `adopt`
            # writes no file, so two clients connecting to an empty workspace
            # both see an empty listing, both pick the same fixed name, and the
            # second one's claim fails -- putting two writers on one transcript
            # before either had said anything. And the theory was empty as well:
            # this branch only runs when the listing *is* empty, which means
            # there is no legacy file to keep writing to. `listing()` would have
            # found it, and `most_recent` would have returned it.
            #
            # A new id cannot collide, so there is no claim to lose here.
            chosen = sessions.new_id()
            sessions.claim(chosen, self.key)
        self.session_id = chosen
        self.restore()

    async def release(self) -> None:
        """Hand the session back and shut the client down, on disconnect."""
        sessions.release(self.key)
        await self.close()

    async def open_session(self, session_id: str) -> str:
        """Switch to a stored session: its transcript, and its conversation.

        The client is dropped rather than reused. `resume` is fixed when the
        client is built, so a client that is already running is already bound to
        another conversation -- keeping it would show one session's transcript
        while the next turn continued a different one, which is the single most
        confusing thing this could do.
        """
        if not sessions.is_id(session_id):
            return f"not a session id: {session_id}"
        if self.busy:
            return "a turn is still running — interrupt it first"
        if session_id == self.session_id:
            return f"already in {self.title or session_id}"
        if not sessions.claim(session_id, self.key):
            # Refused rather than shared. Both windows would write the whole
            # file on every turn, and the loser's turns would vanish with
            # nothing on screen to say they had.
            return "another window has that session open"

        self._persist()
        sessions.release(self.key)
        sessions.claim(session_id, self.key)
        await self.close()
        self.session_id = session_id
        self.blocks = []
        self.settled.clear()
        self.restore()
        if self.sdk_session_id:
            return f"resumed {self.title or session_id}"
        return (
            f"opened {self.title or session_id} — the transcript is here, but the agent "
            "has no memory of it; the next turn starts fresh"
        )

    async def new_session(self, title: str = "") -> str:
        """Start a clean conversation, keeping the one being left.

        The old behaviour -- one file, forever -- meant the only way to start
        clean was to delete the record, and in this project the record is where
        the reasoning behind an expectation lives.
        """
        if self.busy:
            return "a turn is still running — interrupt it first"
        self._persist()
        sessions.release(self.key)
        await self.close()
        self.session_id = sessions.new_id()
        sessions.claim(self.session_id, self.key)
        self.title = title.strip()
        self.created_at = None
        self.sdk_session_id = None
        # For the reason `restore` clears them: this is a different conversation,
        # and a rewind armed against the last one names nothing in it.
        self._rewind_at = None
        self._rewind_drops = None
        self.blocks = []
        self.settled.clear()
        # Written immediately: an empty session that exists is listable, and a
        # new session that vanished because nothing was said in it yet would be
        # a control that silently did nothing.
        self._persist()
        return "new session"

    async def close(self) -> None:
        """Exit the client context entered by `start`.

        The client owns a CLI subprocess and transport tasks. Entering the
        context by hand and never exiting it leaves those alive for the life of
        the app, and leaks a whole set on any later reconnect.

        Serialised against `start` -- see `_lifecycle`. A close that arrives
        while a connect is in flight waits for the connect to finish and then
        closes a whole client, rather than pulling the transport out from under
        the half-built one.
        """
        async with self._lifecycle:
            await self._close_locked()

    async def _close_locked(self) -> None:
        client, self.client = self.client, None
        # Cleared with the client it describes. Left set, a session that closed
        # and reopened at a different level would compare the new selection
        # against the old client's and decide no rebuild was needed.
        self.client_effort = None
        self.client_model = None
        # The reading belongs to the client that answered it. Keeping it across a
        # close would leave the meter reporting the context of a conversation
        # that no longer exists -- and every path that drops a client (a session
        # switch, a workspace rebind, an interrupt) is one where the next context
        # is a different size, usually much smaller. A stale high reading there
        # is exactly the reading that would trigger a needless compaction.
        self.context = None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - shutdown must not raise on the way out
                pass

    async def rebind(self) -> None:
        """Re-point this session at whatever the workspace root is now.

        `path()` is derived from `paths.data_dir()`, so the transcript *file*
        follows a root switch by itself -- but two things do not. `settled` is
        in memory, so the new workspace would open showing the old one's
        conversation; and the SDK client's `cwd` was fixed when it was built, so
        the agent would keep reading and writing in the folder you just left.
        Dropping the client is enough for the second: `ask` starts a new one on
        the next turn, from the environment as it is by then.
        """
        await self.close()
        self.blocks = []
        self.busy = False
        self._idle.set()
        self.settled.clear()
        # Sessions live under the root's data directory, so the one that was
        # open belongs to the folder just left -- and the claim on it does too.
        sessions.release(self.key)
        self.adopt()

    async def ask(self, prompt: str, on_settle: Any) -> None:
        import agent  # noqa: PLC0415

        # Set *before* the first await, not after. `start()` spawns the SDK
        # subprocess and takes seconds on a cold session, and the composer's
        # guard is `if session.busy: return` -- so a second Enter during that
        # window used to pass the guard, and two `ask` coroutines would run
        # `query`/`receive_response` concurrently on one client, interleaving
        # two turns into one block list.
        if self.busy:
            return
        self.busy = True
        self._idle.clear()
        try:
            # Before `start`, and before anything is sent. An interrupt is a
            # control request to the CLI, and one still in flight is aimed at a
            # turn that has already ended -- so without this it lands on the
            # turn about to be issued and kills it the moment it starts, which
            # is a turn that produces nothing and explains nothing.
            await self._stopped()
            # Before `start`, so a level chosen while idle is the level this turn
            # actually runs at -- and after `_stopped`, because it drops the
            # client and doing that under a turn is what `_stop_turn` exists to
            # clean up after.
            await self.apply_effort()
            # And the model, which a project switch can have changed underneath
            # this session since the last turn. Both are lazy for the same
            # reason: a rebuild is seconds, and doing it at the click would make
            # idly switching between two projects cost more than using either.
            await self.apply_model()
            await self.start()
        except Exception:
            self.busy = False
            self._idle.set()
            raise

        # The transcript records what the *user* said; the seed is machinery and
        # goes only to the model. Putting it in `settled` would show the handover
        # note as though the user had typed it, which is both wrong and, given
        # its length, the thing you would then have to scroll past forever.
        seed, self.pending_seed = self.pending_seed, None
        sent = f"{seed}\n\n---\n\n{prompt}" if seed else prompt
        self.settled.append({"role": "user", "text": prompt})
        # The turn lands in the stream's blocks as it arrives; the chat window's
        # ~15 Hz timer is what turns that into something on screen. The same list
        # the stream appends to, not a copy -- a snapshot taken here would never
        # gain a tool card, and the block being written is the block being drawn.
        stream = agent.TurnStream()
        self.blocks = stream.blocks
        # Cleared per turn, so a turn the SDK never named cannot inherit the
        # previous turn's anchor -- which would put a rewind's cut a whole
        # exchange earlier than the one it was aimed at.
        self._turn_uuid = None
        try:
            # The same driver the CLI runs: it checks the token allocation before
            # issuing the turn and records what the turn spent. Doing it here
            # rather than inline is what stops the two surfaces disagreeing about
            # whether the budget applies.
            result = await agent.drive_turn(
                self.client,
                sent,
                stream,
                # Recorded as it arrives rather than read off the return value,
                # which an interrupted turn never reaches. That id is what lets
                # the rebuilt client `resume` this conversation, so losing it on
                # exactly the turns that end in a rebuild cost the whole thread.
                on_session_id=self._remember_sdk_session,
                on_uuid=self._remember_turn_entry,
                session=self.session_id,
            )
            if result.get("sdk_session_id"):
                self.sdk_session_id = result["sdk_session_id"]
        except agent.BudgetRefused as exc:
            # Not an error in the transcript's sense: the system did what it
            # says it does. It still has to be visible, because otherwise the
            # composer just goes quiet.
            self.pending_seed = self.pending_seed or seed
            stream.note(
                f"\n\n**{exc.refusal['message']}**\n\n`{exc.refusal['fix']}`"
            )
        except Exception as exc:  # noqa: BLE001 - the transcript must say why a turn died
            # Otherwise the turn settles as an empty message: the prompt looks
            # unanswered and there is nothing on screen to say why. Only the
            # exception class goes on screen and into the session file -- an SDK
            # message can carry a URL with a token in it, a header, or a path,
            # and both of those destinations are readable long after the fact.
            log.exception("session turn failed")
            # Put the handover note back. It was consumed by a turn that did not
            # complete, and it is the only remaining record of everything the
            # compaction discarded -- dropping it here would mean one failed
            # turn, immediately after a compaction, silently costs the session
            # its entire memory. Re-sending it on the next turn is at worst
            # redundant context; the model may not have read it at all.
            self.pending_seed = self.pending_seed or seed
            # Whatever streamed before the failure is kept: a turn that died
            # half-way is more legible with its half than without it -- and a
            # tool card left mid-flight says which call it died on.
            stream.note(
                f"\n\n**the session failed:** `{type(exc).__name__}` "
                "(details are in the app log)"
            )
        finally:
            self.busy = False
            # Woken here rather than at the end of the block: an interrupt that
            # is waiting for this turn is waiting to stop *doing* things, and
            # persisting and settling the transcript below are not that.
            self._idle.set()
            # The first thing asked is what the session is about, and naming it
            # from the prompt beats leaving every session called "empty session"
            # until someone renames it by hand. An explicit title is never
            # overwritten.
            if not self.title:
                self.title = sessions.title_from(prompt)
            record = {
                "role": "assistant",
                "text": stream.text,
                "blocks": list(stream.blocks),
            }
            # Where this turn ended, and in which conversation. Both, because a
            # uuid only means anything inside the session that issued it and the
            # SDK may name a resumed conversation something new -- see
            # `core/rewind.py:anchor_in`. Written even for a turn that failed:
            # that is the turn most likely to be rewound past, and the entry the
            # rewind resumes at is the last one *before* it.
            if self._turn_uuid:
                record["uuid"] = self._turn_uuid
                record["sdk_session_id"] = self.sdk_session_id
            self.blocks = []
            # `blocks`, not `text`: a turn that only ran commands and said
            # nothing still happened, and dropping it would leave the transcript
            # claiming the prompt went unanswered.
            if record["blocks"]:
                self.settled.append(record)
            self._persist()
            await on_settle(record)

    def interrupt(self) -> str:
        """Stop the turn in flight, and say what was done about it.

        Bound to a button and to Escape, so it fires when nothing is running.

        This used to be one `create_task` around `client.interrupt()` with every
        exception swallowed, and it had three failure modes that all looked the
        same from the composer -- the turn stays busy, so the *next* prompt is
        silently refused and nothing on screen says why. Pressing STOP a second
        time appeared to fix it, which is the bug as reported:

        * the SDK refused the interrupt (not connected, control request
          errored), and nothing said so;
        * the interrupt was accepted but the turn did not end, and the session
          stayed busy for the life of the app;
        * the interrupt arrived late, after the turn had already ended, and
          landed on whatever was issued next.

        So: the failure is reported, the turn is *made* to end, and the pending
        interrupt is something the next turn waits for. The message is returned
        rather than pushed, because the two callers already have somewhere to
        put a line and disagree about where.
        """
        if not self.busy:
            return "nothing is running"
        if self._stopping is not None and not self._stopping.done():
            return "already stopping — the turn is being taken down"
        self._stopping = asyncio.create_task(self._stop_turn())
        return "interrupting the turn…"

    async def _stop_turn(self) -> None:
        """Ask the SDK to stop, then make sure the turn actually stopped.

        The escalation is `ui/tasks.py:cancel`'s, for the same reason: the tool's
        own stop verb is the one that ends things cleanly, and the blunt
        instrument is what happens when it does not work. Here the blunt
        instrument is dropping the client, which ends `receive_response` and lets
        `ask` settle the partial turn the way any other failure settles.

        **The client is dropped either way**, and that is deliberate rather than
        laziness about the happy path. The client owns the CLI subprocess and the
        message stream, and an interrupted turn is precisely the case where what
        is left in that stream is unclear -- a result the aborted turn never
        emitted, or one nobody read. Reusing it made the *next* turn end
        instantly on a message belonging to the last one, with nothing drawn.
        A fresh client cannot have a stale message in it, and `resume` carries
        the conversation across, so what is paid is a restart on a deliberate,
        occasional action.
        """
        # Captured once, and every step below acts on *this* client rather than
        # on `self.client`. There are two awaits in here, each long enough for
        # `ask` to have settled the turn and built a fresh client underneath --
        # and the fresh one belongs to a turn nobody asked to stop. Taking it
        # down, or clearing the busy flag it set, stops a turn that started
        # after the interrupt.
        client = self.client
        if client is not None and hasattr(client, "interrupt"):
            try:
                await client.interrupt()
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                log.warning("the SDK refused the interrupt", exc_info=exc)
                self.say(
                    f"the SDK refused the interrupt ({type(exc).__name__}) — "
                    "stopping the turn the hard way"
                )
        if not await self._idles_within(INTERRUPT_GRACE_S):
            self.say(f"the turn did not stop within {INTERRUPT_GRACE_S:.0f}s — taking the client down")
        # Whether it stopped when asked or had to be taken down, the client goes
        # -- unless it is no longer ours to take down.
        if self.client is not client:
            return
        await self.close()
        if self.client is not client and self.client is not None:
            return
        if self.busy and not await self._idles_within(INTERRUPT_GRACE_S):
            # The turn outlived the client that was feeding it. Whatever it is
            # waiting for is not going to arrive, and a composer that stays
            # locked on the outcome of that is the failure this whole method is
            # about -- so the flag is cleared and the fact is said out loud
            # rather than left to be inferred from a session that never answers.
            self.busy = False
            self._idle.set()
            self.say("the turn is still winding down — the composer is usable again")

    async def _idles_within(self, seconds: float) -> bool:
        """Has the turn settled inside this window? Never raises."""
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return False
        return True

    async def _stopped(self) -> None:
        """Wait for a pending interrupt, so it cannot land on the next turn."""
        pending, self._stopping = self._stopping, None
        if pending is None or pending.done():
            return
        # Bounded: `_stop_turn` bounds itself, and a hang here would be the very
        # thing this method exists to prevent, one layer up.
        await asyncio.wait([pending], timeout=INTERRUPT_GRACE_S * 2)

    def _remember_turn_entry(self, uuid: str) -> None:
        """The latest transcript entry the SDK has named in this turn.

        Overwritten rather than accumulated: a rewind resumes at the *end* of
        the turn it keeps, so the only one worth holding is the last.
        """
        self._turn_uuid = uuid

    async def rewind_to(self, index: int) -> dict[str, Any]:
        """Drop a prompt and everything after it, from the screen and the model.

        The two halves are separable and only one of them is ours, which is the
        whole shape of this: the transcript is a list in memory and a file, and
        rewinding it always works; the conversation belongs to the SDK and comes
        back only if the kept turns recorded an anchor it will accept. Both
        outcomes are real ones and the caller is told which happened -- see
        `core/rewind.py` for why a rewind that silently only cleaned the screen
        would be the worse half of the feature.

        **The client is dropped, and the rewind lands on the next turn.** There
        is no control request for this: `resume_session_at` is fixed when a
        client is built, so putting the model back means building a new one, and
        the same laziness `apply_effort` uses applies for the same reason -- a
        rebuild is seconds, and paying it at the click would make rewinding twice
        to find the right point cost more than living with the bad turns.

        The prompt comes back rather than being re-sent. A rewind is most often
        reached for because the prompt itself wanted changing, and a rewind that
        immediately re-asked would spend a turn reproducing the answer it just
        discarded -- which for this agent can be a job it also just submitted.
        """
        if self.busy:
            return {"ok": False, "message": "a turn is still running — interrupt it first"}

        plan = rewind.plan(self.settled, index, sdk_session_id=self.sdk_session_id)
        if not plan["ok"]:
            return {"ok": False, "message": plan["reason"]}

        anchor = plan["anchor"]
        drops = self._drops_turn(anchor) if anchor and plan["turns"] == 1 else None
        # The client goes before anything is written, so a rewind cannot leave a
        # live conversation running against a transcript that has moved out from
        # under it. `start()` builds the replacement on the next turn.
        await self.close()
        self._rewind_at = anchor
        self._rewind_drops = drops

        self.settled = [
            *plan["keep"],
            rewind.record(dropped=plan["dropped"], resumed=bool(anchor), anchor=anchor),
        ]
        self._persist()

        turns = plan["turns"]
        what = "one exchange" if turns == 1 else f"{turns} exchanges"
        if anchor:
            message = f"rewound {what} — the agent's memory goes back too, on the next turn"
        elif self.sdk_session_id:
            # An anchor from a conversation this session is no longer in. Named
            # rather than folded into the case below, because "there was no live
            # conversation" and "there is one and it cannot be cut here" are
            # different situations with the same symptom.
            message = (
                f"rewound {what} on screen — this conversation was resumed under a new id, "
                "so the agent still remembers them"
            )
        else:
            message = f"rewound {what} on screen — there is no conversation to put back"
        return {"ok": True, "message": message, "prompt": plan["prompt"], "resumed": bool(anchor)}

    def _drops_turn(self, anchor: str) -> str | None:
        """The uuid of the prompt this rewind means to discard, for the CLI's check.

        Read out of the SDK's own transcript rather than captured live, and that
        is the cheaper half of a real trade. Capturing it would mean asking every
        session for `replay-user-messages` and carrying `UserMessage` objects
        through `TurnStream` on every turn of every conversation, to serve an
        optional check on an occasional, deliberate action. Reading the file
        costs one parse at the click.

        Best effort by construction. The check is a refinement of a truncation
        that happens either way, so anything unreadable here -- an SDK without
        the helper, a transcript in a directory this cannot guess, a session that
        never reached disk -- returns None and the rewind runs unvalidated.
        """
        try:
            from claude_agent_sdk import get_session_messages  # noqa: PLC0415

            messages = get_session_messages(self.sdk_session_id, directory=str(paths.root()))
        except Exception:  # noqa: BLE001 - see the docstring
            log.debug("no drops-turn uuid for the rewind", exc_info=True)
            return None
        # The first prompt *after* the anchor: the SDK's rule of thumb is that
        # `resume_session_at` names the last entry kept and `resume_drops_turn`
        # the prompt of the turn immediately following it.
        found = False
        for message in messages:
            if found and getattr(message, "type", None) == "user":
                return getattr(message, "uuid", None)
            if getattr(message, "uuid", None) == anchor:
                found = True
        return None

    def _remember_sdk_session(self, sdk_session_id: str) -> None:
        self.sdk_session_id = sdk_session_id

    async def read_context(self) -> dict[str, Any] | None:
        """Refresh the context reading. Never raises, never blocks a turn.

        `get_context_usage` is a control request, not a model call: it costs no
        tokens and it is answered by the CLI from state it already has. That is
        what makes polling it reasonable at all, and it is why this is a poll
        rather than something folded into `drive_turn` -- the number worth
        watching is the one that climbs *during* a long turn, and a turn that
        runs for forty minutes would otherwise report its context once, at the
        end, when nothing can be done about it.

        Every failure is swallowed to None. An SDK without the method, a client
        mid-restart, a control request that races a shutdown: none of them are
        worth a line in the transcript, and all of them are indistinguishable to
        a reader from "no session yet", which the meter already draws.
        """
        client = self.client
        if client is None or self._reading_context:
            return self.context
        reader = getattr(client, "get_context_usage", None)
        if reader is None:
            return self.context
        self._reading_context = True
        try:
            usage = await reader()
        except Exception as exc:  # noqa: BLE001 - see the docstring
            log.debug("context usage unavailable", exc_info=exc)
            return self.context
        finally:
            self._reading_context = False
        if isinstance(usage, dict):
            self.context = usage
        return self.context

    async def apply_effort(self) -> bool:
        """Rebuild the client if the chosen effort is not the one it is running.

        Returns whether a rebuild happened, which is what lets a caller say so.

        **Lazy on purpose.** There is no control request for effort, so changing
        it means dropping the SDK subprocess and spawning another -- seconds, on
        a cold start. Doing that on every click would make idly comparing two
        levels cost more than using either. Doing it here means the cost lands
        once, immediately before a turn that was going to pay for a connected
        client anyway.

        The conversation survives because `start()` passes `resume=` with the
        SDK's own session id. That is the entire difference between this and
        `compact`, which deliberately clears the id: compacting is *meant* to
        discard the conversation and replace it with a note, and this is meant to
        change one setting and keep everything.
        """
        chosen = effort.current()
        if self.client is None:
            # Nothing is running, so the next `start` picks the level up for
            # free. Recording it here as well would be a lie the moment the
            # selection changed again before that start.
            return False
        if chosen == self.client_effort:
            return False
        if self.sdk_session_id is None:
            # No id to resume, so a rebuild would silently start a new
            # conversation. The level is left to apply at the next natural
            # rebuild rather than paid for with the transcript -- the session has
            # said nothing yet in any case, since the id arrives with the first
            # turn.
            log.info("effort change deferred: no sdk session id to resume yet")
            return False
        await self.close()
        await self.start()
        return True

    async def apply_model(self) -> bool:
        """Rebuild the client if the `research` model is not the one it is
        running. `apply_effort`'s sibling, and every line of its reasoning
        applies unchanged.

        What makes it necessary is Stage 5: a project may override `research`
        (`core/budget.py:configure`), and switching project is a thing that
        happens *while a session is live*. Without this the previous model goes
        on answering while the projects window and the ledger both say
        otherwise -- which is the worst shape for it, because every surface
        agrees on a claim that is false.

        Deliberately not folded into `apply_effort`. The two are checked
        together at the same call site, but a rebuild that could have been
        caused by either is a rebuild whose reason cannot be logged, and the log
        line is what makes a mysterious reconnect explicable.
        """
        if self.client is None:
            return False
        chosen = config_mod.load().model_for("research")
        if chosen == self.client_model:
            return False
        if self.sdk_session_id is None:
            log.info("model change deferred: no sdk session id to resume yet")
            return False
        log.info("rebuilding the client: %s -> %s", self.client_model, chosen)
        await self.close()
        await self.start()
        return True

    async def maybe_compact(self) -> dict[str, Any] | None:
        """Compact if the context has passed the configured threshold.

        Called after a turn settles, which is the only safe moment: this drops
        the client and builds another, and doing that underneath a running
        `receive_response` is the failure `_stop_turn` exists to clean up after.

        The order matters and is the whole method. The note is written *first*,
        while the outgoing session still remembers everything and its cache is
        warm; only then is the client dropped. Reversed, there would be nothing
        left to summarise.

        `pending_seed` rather than an immediate turn: sending the note now would
        cost a whole round-trip to produce an answer nobody asked for. Held, it
        rides along with whatever the user says next and costs nothing.
        """
        from core import compaction  # noqa: PLC0415

        cfg = config_mod.load()
        usage = await self.read_context()
        if not compaction.should_compact(usage, cfg):
            return None
        before = compaction.context_tokens(usage)
        return await self.compact(reason=f"context reached {before:,} tokens")

    async def compact(self, *, reason: str = "") -> dict[str, Any]:
        """Summarise this conversation and start a fresh one holding the summary.

        Separated from `maybe_compact` so it can be asked for directly -- the
        threshold is a default, not the only reason to want this, and a session
        that has wandered is worth compacting at any size.
        """
        import agent  # noqa: PLC0415
        from core import compaction  # noqa: PLC0415

        if self.busy:
            return {"ok": False, "message": "a turn is still running — interrupt it first"}
        if self.client is None:
            return {"ok": False, "message": "nothing to compact — no session is connected"}

        before = compaction.context_tokens(self.context)
        self.busy = True
        self._idle.clear()
        try:
            handoff = await compaction.write_handoff(
                self.client, agent.drive_turn, session=self.session_id
            )
        except agent.BudgetRefused as exc:
            # No carve-out. A compaction is a model call and the allocation
            # applies to it like any other -- exempting it would make "compact"
            # the way to keep spending after the ceiling, and the ceiling is the
            # feature. The conversation is left intact and oversized, which is
            # recoverable; the alternative is not.
            return {"ok": False, "message": exc.refusal["message"], "fix": exc.refusal["fix"]}
        except Exception as exc:  # noqa: BLE001 - a failed compaction keeps the session
            log.exception("could not write the handoff note")
            return {
                "ok": False,
                "message": f"could not summarise the session ({type(exc).__name__}) — nothing was discarded",
            }
        finally:
            self.busy = False
            self._idle.set()

        # The fresh conversation is a *new* SDK session, so the resume id has to
        # go. Leaving it would have `start()` resume the very conversation this
        # is discarding, quietly restoring the context that was just summarised
        # and making the whole operation a cost with no effect.
        await self.close()
        self.sdk_session_id = None
        self.pending_seed = compaction.seed_message(handoff["note"], tokens_before=before)

        entry = compaction.record(
            tokens_before=before,
            tokens_after=0,
            note=handoff["note"],
            cost=handoff.get("quota"),
        )
        if reason:
            entry["reason"] = reason
        self.settled.append(entry)
        self._persist()
        return {"ok": True, "message": f"compacted at {before:,} tokens", "record": entry}

    def say(self, message: str) -> None:
        """A line for the status bar, when there is one to put it in."""
        log.info("%s", message)
        if self.notify is not None:
            try:
                self.notify(message)
            except Exception:  # noqa: BLE001 - a notice must not kill a turn
                log.exception("could not report: %s", message)

    def path(self) -> Path:
        return sessions.path_for(self.session_id)

    def _persist(self) -> None:
        """Closing the window should not be destructive."""
        # The claim is checked at the write, not only at adoption. `write`
        # replaces the whole file, so a client that has lost its claim -- a
        # reload took it over, a workspace switch moved underneath it -- would
        # otherwise overwrite the holder's transcript with its own stale copy,
        # which is the exact data loss the claim exists to prevent.
        #
        # Only *another live* holder blocks the write. An unclaimed session has
        # no one to overwrite, and refusing there would make persistence depend
        # on having gone through `adopt` rather than on the invariant.
        if sessions.held_by_other(self.session_id, self.key):
            log.warning(
                "not persisting session %s: another client holds it", self.session_id
            )
            return
        try:
            sessions.write(
                self.session_id,
                self.settled,
                title=self.title,
                created_at=self.created_at,
                sdk_session_id=self.sdk_session_id,
            )
        except (OSError, ValueError):
            log.exception("could not persist session %s", self.session_id)

    def restore(self) -> None:
        """Read a session back: its metadata, then the records that render.

        The file is on disk between runs, so a record is not necessarily one we
        wrote: the chat window subscripts `role` and `text`, and a line that is a
        bare string or is missing `text` would take the whole page down at build
        time. `blocks` is optional for the same reason in reverse -- transcripts
        written before tool calls were drawn have none, and those still open.

        The `meta` line is skipped by that same filter rather than by a special
        case: its `role` is absent, so it is not a record that renders.
        """
        # Cleared first, and on every path out of here. A pending rewind belongs
        # to one conversation -- it names an entry inside it -- so carrying one
        # across a session switch would resume a *different* SDK session at a
        # uuid that is not in it. Restoring is where the session identity
        # changes, so it is where this has to be true.
        self._rewind_at = None
        self._rewind_drops = None
        path = self.path()
        meta = sessions.read_meta(path)
        self.title = str(meta.get("title") or "")
        self.created_at = meta.get("created_at")
        sdk_id = meta.get("sdk_session_id")
        self.sdk_session_id = sdk_id if isinstance(sdk_id, str) and sdk_id else None
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(record, dict)
                and record.get("role") in ROLES
                and isinstance(record.get("text"), str)
            ):
                kept: dict[str, Any] = {"role": record["role"], "text": record["text"]}
                blocks = _drawable_blocks(record.get("blocks"))
                if blocks:
                    kept["blocks"] = blocks
                # Carried through the filter rather than dropped by it, so a
                # compaction marker still reads as one after a reload instead of
                # degrading into an anonymous message from nobody. Both are
                # copied defensively: this file outlives the version that wrote
                # it, and the renderer subscripts them.
                if isinstance(record.get("kind"), str):
                    kept["kind"] = record["kind"]
                if isinstance(record.get("note"), str):
                    kept["note"] = record["note"]
                # Where this turn ended in the SDK's transcript, so a rewind
                # still has somewhere to cut after the app has been restarted.
                # Dropping these was the difference between a rewind that works
                # on a session you have been in all along and one that quietly
                # stops working the moment you reopen it.
                if isinstance(record.get("uuid"), str):
                    kept["uuid"] = record["uuid"]
                if isinstance(record.get("sdk_session_id"), str):
                    kept["sdk_session_id"] = record["sdk_session_id"]
                # The turns a rewind took out, carried through so the marker
                # still has them to show. Filtered by the same rule as `blocks`:
                # this file outlives the version that wrote it.
                if record.get("dropped") is not None:
                    kept["dropped"] = rewind.dropped_of(record)
                if isinstance(record.get("anchor"), str):
                    kept["anchor"] = record["anchor"]
                self.settled.append(kept)
        # A rewind that had not been applied to a conversation yet when the app
        # was closed. The marker is what says so and the anchor is on it, so this
        # costs no extra state -- see `core/rewind.py:pending_anchor`. The
        # drops-turn check is not restored with it: it is a refinement of a
        # truncation that happens either way, and reconstructing it here would
        # mean reading the SDK's transcript on every session that is opened.
        self._rewind_at = rewind.pending_anchor(self.settled)


def _drawable_blocks(value: Any) -> list[dict[str, Any]]:
    """Blocks off disk that the chat window can actually draw.

    Same reason `restore` filters records: this file outlives the version that
    wrote it, so `blocks` is untrusted input. A block with no `kind` would reach
    the renderer's dispatch and take the page down at build time.
    """
    if not isinstance(value, list):
        return []
    return [b for b in value if isinstance(b, dict) and isinstance(b.get("kind"), str)]


def build() -> None:
    """Register the page. Everything with per-client state is built inside it.

    Constructing the `Session` or the `Workspace` out here would build the
    layout once, at import, and hand every connected client the same
    `ClaudeSDKClient`, the same token buffer, the same transcript file and the
    same focused pane -- a second window would see the first one's stream, race
    it on the way to disk, and fight it over the layout file.
    """
    from nicegui import app as nicegui_app, ui

    _serve_static(nicegui_app)
    # `shared=True` because these are registered at global scope alongside a
    # `@ui.page` route; without it NiceGUI refuses, since it cannot tell whether
    # the markup was meant for one page or all of them. Here it is genuinely all
    # of them: one stylesheet, one tiling module, one KaTeX.
    ui.add_head_html(kit.stylesheet_head(url_prefix=f"{STATIC_URL}/fonts"), shared=True)
    ui.add_body_html(f'<script src="{STATIC_URL}/tiling.js"></script>', shared=True)
    katex.install(nicegui_app)

    @nicegui_app.get("/__grad/show")
    def _show() -> dict[str, bool]:
        """How a second launch hands over to this one.

        The launcher cannot raise another process's window, and on Windows it
        may not even be allowed to try -- foreground rights belong to the
        process that has them. So the running instance raises its own window,
        and the only thing crossing the boundary is the request. Unauthenticated
        like the rest of this port, and harmless: the whole effect is that a
        window the user already owns becomes visible.
        """
        return {"shown": desktop.show_window()}

    @nicegui_app.post("/__grad/wake")
    async def _wake(payload: dict) -> Any:
        """A watcher reporting that something the agent armed has happened.

        **This is the one endpoint on this port that is authenticated, and the
        asymmetry is the point.** `/__grad/show` is unauthenticated because its
        entire effect is that a window the user already owns becomes visible.
        This one *starts a turn for an agent with Bash access*, so anything that
        can open a loopback socket -- which is every process on the machine --
        would otherwise be able to drive it. The token is a mode-600 file in the
        app directory; see `core/wakeups.py:token`.

        `compare_digest` rather than `==`, because a token compared with early
        exit is a token that can be guessed a byte at a time by something already
        able to time it.

        Queued rather than run here. A route has no client in scope and the turn
        has to be drawn into one; `ui/state.py:accept_wake` explains the seam.

        **The body is taken as a plain `dict`, not as an injected `Request`.**
        This module runs under `from __future__ import annotations`, so every
        annotation reaches FastAPI as a *string* which it resolves against the
        module's globals -- and `Request` imported inside this function is not
        one. The result is not an error: FastAPI decides the unresolvable
        parameter must be a query parameter, and every POST is rejected with a
        422 asking for a missing query field called `request`. `dict` is a
        builtin, so it resolves, and FastAPI reads it from the body.
        """
        from core import wakeups as wk  # noqa: PLC0415

        from fastapi.responses import JSONResponse  # noqa: PLC0415

        body = payload if isinstance(payload, dict) else {}
        offered = str(body.get("token") or "")
        if not offered or not secrets.compare_digest(offered, wk.token()):
            log.warning("refused a wake with a bad token")
            return JSONResponse({"error": "bad token"}, status_code=403)

        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return JSONResponse({"error": "nothing to say"}, status_code=400)

        for workspace in list(_WAKE_TARGETS):
            if workspace.accept_wake(prompt):
                return {"queued": True, "wake": body.get("wake")}
        # No window open, or every one of them is already holding a queue. The
        # watcher records this as undelivered and `wakeup list` surfaces it, so
        # the wake is deferred rather than lost.
        return JSONResponse({"queued": False}, status_code=503)

    @nicegui_app.get("/__grad/notebook/{name}")
    def _notebook(name: str) -> Any:
        """One notebook, rendered read-only, for the pane's iframe.

        Served from this app rather than from Lab so the pane has something to
        show whether or not a Lab server is running -- and so what it shows can
        be sandboxed, which Lab cannot be. `ui/render.py` explains the rest; the
        name is validated there, against a directory rather than a pattern.
        """
        from fastapi.responses import HTMLResponse, PlainTextResponse  # noqa: PLC0415

        try:
            body = render.notebook_html(name)
        except render.NotAllowed:
            return PlainTextResponse("no such notebook in this workspace", status_code=404)
        except OSError as exc:
            return PlainTextResponse(f"could not read it: {exc}", status_code=503)
        return HTMLResponse(
            body,
            headers={
                # It is a document built from untrusted stored output and it
                # needs nothing from anywhere: no scripts, no fetches, no
                # framing by anyone but us. The iframe is sandboxed as well --
                # this is the half that holds if the sandbox attribute is ever
                # dropped by an edit that looks unrelated.
                "Content-Security-Policy": (
                    "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; "
                    f"frame-ancestors {desktop.origin(PORT)} http://localhost:{PORT}"
                ),
                "Cache-Control": "no-store",
            },
        )

    @nicegui_app.get("/__grad/figure/{name}")
    def _figure(name: str) -> Any:
        """One figure out of the workspace's `figures/`, for the transcript.

        A route rather than `add_static_files`, for two reasons that both come
        from where the directory is. It is inside `paths.root()`, and the root
        *moves* -- switching workspace folders is a button in the app bar, while
        a static mount binds one absolute path at startup and would go on
        serving the folder the user left. And the name arrives from a markdown
        link the *agent* wrote, which makes it untrusted input to a filesystem
        read: `render.resolve_figure` re-derives the directory per request and
        checks containment after symlinks, exactly as the notebook route does.

        Without this, an inline `![loss](figures/001.png)` resolved against the
        page origin, 404ed, and drew a broken-image icon where the agent had put
        a figure -- while a second copy of the same image was appended at the
        bottom of the message by a different code path.
        """
        from fastapi.responses import FileResponse, PlainTextResponse  # noqa: PLC0415

        try:
            path = render.resolve_figure(name)
        except render.NotAllowed:
            return PlainTextResponse("no such figure in this workspace", status_code=404)
        return FileResponse(
            path,
            headers={
                # The workspace is a working directory: a figure is overwritten
                # by the next run of the cell that drew it, under the same name.
                # A cached copy is the previous experiment's result shown beside
                # this one's numbers.
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; img-src 'self' data:",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @ui.page("/")
    def index() -> None:
        from nicegui import context  # noqa: PLC0415 - page scope, not import scope

        session = Session(_client_key())
        session.adopt()
        workspace = state_mod.Workspace(session, _current_project())
        # The one place both exist. A turn's own failures already reach the
        # transcript; this is for the things that happen *around* a turn -- an
        # interrupt the SDK refused, a client that had to be taken down -- which
        # have no turn to be written into.
        session.notify = workspace.say
        # Where a wake can land. A list rather than one slot because there can
        # be several windows open on one workspace, and the wake belongs to
        # whichever of them can take it -- see `/__grad/wake`.
        _WAKE_TARGETS.append(workspace)
        # Per client, not `app.on_shutdown`: that would accumulate one handler
        # per connection and hold every session's subprocess open until the app
        # itself exits. Graced, not immediate: NiceGUI fires this on any socket
        # drop, including the two-second ping timeout a busy event loop can
        # miss -- and the busiest moment this loop has is spawning the CLI for
        # the session's own first turn. Releasing right then closed the client
        # mid-connect; see `Session._lifecycle` for what that corrupted.
        context.client.on_disconnect(
            lambda: _release_when_gone(context.client, session, workspace)
        )
        shell.build(workspace)


#: How long a dropped websocket is given to come back before its session is
#: closed. Longer than a reconnect takes, shorter than anyone waits before
#: reopening the app -- and generous on purpose: the cost of waiting is a CLI
#: subprocess held open a little longer, and the cost of not waiting was every
#: first prompt on a fresh install dying mid-connect.
RELEASE_GRACE_S = 10.0

#: The grace tasks currently in flight, held so the event loop's weak reference
#: is not the only one. Entries remove themselves when they finish.
_RELEASES: set[Any] = set()

#: Workspaces a wakeup can be delivered to, oldest window first. Module-level
#: because the delivering end is an HTTP route with no client in scope -- see
#: `/__grad/wake` -- and per-process because that is what a wake reaches: one
#: app, holding one single-instance lock, on one published port.
_WAKE_TARGETS: list[Any] = []


def _release_when_gone(client: Any, session: Session, workspace: Any = None) -> None:
    """Hand the session back only if this client's socket stays gone.

    NiceGUI's disconnect handlers run on every socket drop, and a drop is not a
    departure: the page auto-reconnects, and the ping timeout that declares one
    is two seconds on the default settings -- short enough for the event loop
    itself to miss it while it spawns the CLI. So the release waits, and asks
    the client whether anyone came back before taking the session down.
    """

    async def _check() -> None:
        await asyncio.sleep(RELEASE_GRACE_S)
        if getattr(client, "has_socket_connection", False):
            return
        # Withdrawn on the same condition as the session, not on the disconnect:
        # a window that reconnects inside the grace never stopped being somewhere
        # a wake could land, and dropping it early would send a wake that had
        # waited four hours to a 503.
        if workspace is not None and workspace in _WAKE_TARGETS:
            _WAKE_TARGETS.remove(workspace)
        await session.release()

    try:
        task = asyncio.get_running_loop().create_task(_check())
    except RuntimeError:
        # No loop to wait on (tests, teardown). Nothing to grace; the direct
        # release is what the handler did before the grace existed.
        asyncio.run(session.release())
    else:
        # Held for the length of the sleep. The loop keeps only a weak reference
        # to a task, so a `create_task` whose result nobody keeps can be
        # collected before it resumes -- and this one spends ten seconds
        # suspended, which is ten seconds of being the only thing that would
        # ever hand the session back.
        _RELEASES.add(task)
        task.add_done_callback(_RELEASES.discard)


def _serve_static(nicegui_app: Any) -> None:
    """Fonts and `tiling.js`, plus the generated wiki if one exists.

    Both are mounted read-only under fixed prefixes rather than served from the
    workspace root: this app binds an unauthenticated port and accepts prompts
    for an agent with Bash access, and exposing `paths.root()` over HTTP would
    hand anything that reached that port the whole workspace, credentials
    directory included.
    """
    directory = static_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fonts").mkdir(parents=True, exist_ok=True)
    # `max_cache_age` defaults to an hour, which for a desktop app serving its
    # own 16 KB of JavaScript over loopback buys nothing and costs a confusing
    # hour: after an edit to `tiling.js`, a restarted app keeps running the old
    # copy out of the webview's disk cache. The fonts beside it are large but
    # never change, and they are the reason this is not zero.
    nicegui_app.add_static_files(STATIC_URL, str(directory), max_cache_age=60)

    try:
        from tools import wiki as wiki_tool  # noqa: PLC0415 - optional

        wiki_dir = wiki_tool.output_dir()
        if wiki_dir.exists():
            nicegui_app.add_static_files("/grad-wiki", str(wiki_dir))
    except Exception:  # noqa: BLE001 - a missing wiki is not a startup failure
        log.debug("no wiki directory to serve")


def _current_project() -> str | None:
    from core import budget as budget_mod

    try:
        return budget_mod.current_project()
    except Exception:  # noqa: BLE001 - an unreadable project file is not fatal
        return None


def _client_key() -> str:
    """A unique id for this *connection*, used as the session-claim owner.

    Both halves are here on purpose. The browser id is signed into a cookie (see
    `_storage_secret`) and identifies the person; the connection id distinguishes
    one tab from another.

    It used to be the browser id alone, which made two tabs of one browser the
    same owner -- and `claim` returns True when the holder *is* the owner, so
    both tabs adopted the same session and both wrote the whole file on every
    turn. That is the two-writers data loss the claim exists to prevent, reached
    through the claim itself.

    The cost of per-connection ownership is that a page reload arrives as a new
    owner while the old client's `on_disconnect` may not have fired yet, so the
    reloaded page can land on the next session down the list rather than the one
    it was just in. `sessions.claim` takes over a claim whose owner is no longer
    live, which closes that window as soon as the disconnect lands; opening the
    wrong session is recoverable from the sessions menu, and overwriting one is
    not.
    """
    from nicegui import app as nicegui_app, context  # noqa: PLC0415

    try:
        browser = str(nicegui_app.storage.browser.get("id") or "")
    except (RuntimeError, KeyError):  # no storage_secret configured
        browser = ""
    connection = str(getattr(context.client, "id", "")) or "0"
    key = f"{browser or 'anon'}-{connection}"
    return re.sub(r"[^A-Za-z0-9_-]", "", key)[:64] or "default"


def _storage_secret() -> str:
    """The key NiceGUI signs the browser-id cookie with.

    Persisted rather than generated per launch so a restart does not orphan
    every transcript written before it.
    """
    path = appdata.state_dir() / "ui_storage_secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:  # not every filesystem honours it; the file is local either way
            log.debug("could not restrict permissions on %s", path)
    return path.read_text(encoding="utf-8").strip()


def _install_desktop(native: bool) -> None:
    """Startup wiring that only makes sense once the loop and window exist.

    Registered as a NiceGUI startup handler rather than done before `ui.run`,
    because both halves need things that do not exist yet at call time: the
    event loop `desktop.request_quit` dispatches onto, and the pywebview window
    whose close button is being reinterpreted.
    """
    from nicegui import app as nicegui_app  # noqa: PLC0415

    if native:
        # Handed to the *window's* process, because that is the only place a
        # close can be vetoed: what this process holds is a proxy with no
        # events on it, and binding `closing` here silently did nothing for as
        # long as it has existed. `ui/desktop.py:hold_window_open` is the whole
        # story; it travels by pickle, which is why it is a module-level
        # function taking a path rather than a closure over anything here.
        #
        # The flag is cleared first: one left behind by a crashed run would
        # promise a notification area that this run has not built yet.
        desktop.set_tray_flag(False)
        nicegui_app.native.start_args["func"] = desktop.hold_window_open
        nicegui_app.native.start_args["args"] = (str(desktop.tray_flag()),)
        # The taskbar button's icon. Travels to the window's process the same way
        # `func` does, and for the same reason it has to: the form that owns the
        # icon lives there. Without it pywebview falls back to extracting the
        # icon from `sys.executable`, so the app appears on the taskbar as
        # Python. A path only -- `_split_picklable` sends strings through fine.
        #
        # Skipped when it could not be drawn rather than passed as None: pywebview
        # tests `if _state['icon'] and os.path.isfile(...)`, so a None is
        # harmless, but leaving the key out entirely keeps the "we could not draw
        # it" case visible in the start args rather than looking like a choice.
        icon = desktop.icon_path()
        if icon:
            nicegui_app.native.start_args["icon"] = icon
        # Where the window was last time, and how big. Spliced into
        # `create_window`'s arguments *after* NiceGUI's own `width`/`height`, so
        # this is what decides the size and `window_size` below is only the
        # fallback the first launch on a machine gets. `track_window` is what
        # keeps the file current; it has to be registered before `ui.run`,
        # because `ui.run` starts the bridge that delivers the events.
        nicegui_app.native.window_args.update(desktop.window_args())
        desktop.track_window(nicegui_app)

    @nicegui_app.on_connect
    def _drop_splash() -> None:
        """The workspace is on screen, so the loading mark can go.

        On *connect*, not on startup: `on_startup` fires when the server is
        listening, which is several seconds before the webview process has
        rendered anything -- taking the splash down there would put the gap back
        exactly where it was. A connected client is a page that has loaded and
        opened its socket, which is the first moment there is something to look
        at instead.

        Fires per client and `stop` is idempotent, so a reload costs a no-op.
        """
        from ui import splash  # noqa: PLC0415

        splash.stop()

    @nicegui_app.on_startup
    def _wire() -> None:
        desktop.bind_loop(asyncio.get_running_loop())
        if not native:
            # Browser mode has no window to hide and no tray to hide it to; the
            # tab is the affordance and closing it is the user's business.
            return
        desktop.start_tray(on_restart_lab=_restart_lab_here)

    @nicegui_app.on_shutdown
    def _unwire() -> None:
        from core import instance  # noqa: PLC0415

        instance.release()


def _restart_lab_here() -> None:
    """Restart Lab bound to this app's origin. Also the tray's menu entry."""
    from core import spawn  # noqa: PLC0415

    argv = [
        sys.executable, "-m", "tools.lab", "start",
        "--ui-origin", desktop.origin(PORT), "--force",
    ]
    try:
        spawn.run(argv, cwd=str(paths.root()), capture_output=True, text=True, timeout=90)
    except Exception:  # noqa: BLE001 - reported by the window's next poll
        log.exception("could not restart Lab")


def _start_update_check() -> None:
    """Ask the remote whether there is a newer release, once a day, off the loop.

    A thread rather than a NiceGUI timer, and never on a draw. `git fetch`
    crosses the network and blocks for as long as the network makes it; on the
    event loop that is a frozen window, and in `models.update_model` it would be
    a frozen window every time someone opened the project menu. So the check
    writes `update.json` and the UI only ever reads that file.

    Silent by construction. A machine with no network, a checkout with no
    remote, a corporate proxy that refuses -- none of them is something to
    interrupt someone's research to report, and all of them simply leave
    yesterday's answer in place.
    """
    import threading  # noqa: PLC0415

    def _check() -> None:
        from core import update  # noqa: PLC0415

        try:
            if update.check_due():
                update.refresh_cache()
        except Exception:  # noqa: BLE001 - see the docstring
            log.debug("the update check did not complete", exc_info=True)

    # Daemon, so a fetch against a black-holed remote cannot keep the process
    # alive after the window closes; the socket timeout would eventually fire,
    # but "eventually" is not a quit.
    threading.Thread(target=_check, name="grad-update-check", daemon=True).start()


def run(*, native: bool = True, port: int | None = None) -> None:
    """`ui.run(native=True)` gives a real desktop window via pywebview, so the
    packaging question is answered without Electron or Tauri. Browser mode is
    the fallback when pywebview misbehaves on Windows.

    `port=None` means "choose one" -- see `ui/desktop.py:choose_port` for why
    that is a walk-up from 8080 rather than anything random.
    """
    from nicegui import ui

    global PORT

    port = desktop.choose_port(port)
    PORT = port
    appdata.ensure()
    # Here as well as in `agent.py:main`, because this is a public entry point:
    # anything that imports `ui.app` and calls `run` -- a launch config, a test
    # harness, a shortcut written before the CLI existed -- skips `main`
    # entirely, and would then open on an empty app directory while the state it
    # wanted sat unmigrated in the workspace. Idempotent, so running twice costs
    # a stat per entry.
    for name in migrate.run_pending():
        log.info("migrated %s (state now under %s)", name, appdata.app_dir())
    paths.ensure_workspace()
    instance.publish(port)
    _start_update_check()
    build()
    _install_desktop(native)
    # `window_size` is passed *only* in native mode, and that is not a
    # nicety: NiceGUI turns `native` on whenever a window size is given, so
    # passing it unconditionally made `native=False` unreachable and the
    # documented browser fallback impossible to actually take.
    #
    # It is no longer what decides the size. `_install_desktop` puts the
    # remembered geometry in `native.window_args`, which NiceGUI merges over
    # these values -- so this is the size of a window nobody has moved yet, and
    # it is spelled once, in `desktop.DEFAULT_SIZE`.
    extra = {"window_size": desktop.DEFAULT_SIZE} if native else {}
    ui.run(
        native=native,
        title="Grad",
        # The browser tab's mark, and the same file the taskbar button uses. A
        # second `icon_path()` call is a stat once the first has rendered it.
        #
        # This is not what fixes the taskbar in native mode -- a favicon belongs
        # to the page, and the taskbar button belongs to the window -- but the
        # two should not be different marks, and browser mode has no window to
        # carry one.
        favicon=desktop.icon_path(),
        # Explicit localhost: in browser mode NiceGUI would otherwise bind
        # 0.0.0.0, and this is an unauthenticated UI that accepts prompts for an
        # agent with Bash access.
        host="127.0.0.1",
        port=port,
        # Signs the browser-id cookie each client's transcript is keyed by.
        storage_secret=_storage_secret(),
        # The engineio ping timeout is derived from this (0.4x, floor 2s), and
        # two seconds is less than the event loop can be stalled by spawning a
        # cold `claude.exe` -- which is how the app's own first turn used to
        # disconnect its own page. Ten gives a 4s ping timeout and a wider
        # window for a reloaded page to reclaim its session.
        reconnect_timeout=10.0,
        reload=False,
        # The design is a cream paper ground with ink rules. Quasar's dark mode
        # would fight every token in `ui/tokens.py`.
        dark=False,
        show=False,
        **extra,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
