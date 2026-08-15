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

from core import appdata, config as config_mod, instance, paths
from ui import desktop, katex, kit, sessions, shell, state as state_mod
ROLES = ("user", "assistant")
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

    async def start(self) -> None:
        if self.client is not None:
            return
        import agent  # noqa: PLC0415 - imported here so the UI can load without the SDK
        from claude_agent_sdk import ClaudeSDKClient  # noqa: PLC0415

        cfg = config_mod.load()
        agent.preflight_environment()
        options = agent.build_options(cfg, resume=self.sdk_session_id)
        self.client = ClaudeSDKClient(options=options)
        await self.client.__aenter__()

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
        """
        client, self.client = self.client, None
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
            await self.start()
        except Exception:
            self.busy = False
            self._idle.set()
            raise

        self.settled.append({"role": "user", "text": prompt})
        # The turn lands in the stream's blocks as it arrives; the chat window's
        # ~15 Hz timer is what turns that into something on screen. The same list
        # the stream appends to, not a copy -- a snapshot taken here would never
        # gain a tool card, and the block being written is the block being drawn.
        stream = agent.TurnStream()
        self.blocks = stream.blocks
        try:
            # The same driver the CLI runs: it checks the token allocation before
            # issuing the turn and records what the turn spent. Doing it here
            # rather than inline is what stops the two surfaces disagreeing about
            # whether the budget applies.
            result = await agent.drive_turn(
                self.client,
                prompt,
                stream,
                # Recorded as it arrives rather than read off the return value,
                # which an interrupted turn never reaches. That id is what lets
                # the rebuilt client `resume` this conversation, so losing it on
                # exactly the turns that end in a rebuild cost the whole thread.
                on_session_id=self._remember_sdk_session,
                session=self.session_id,
            )
            if result.get("sdk_session_id"):
                self.sdk_session_id = result["sdk_session_id"]
        except agent.BudgetRefused as exc:
            # Not an error in the transcript's sense: the system did what it
            # says it does. It still has to be visible, because otherwise the
            # composer just goes quiet.
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
        # Whether it stopped when asked or had to be taken down, the client goes.
        await self.close()
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

    def _remember_sdk_session(self, sdk_session_id: str) -> None:
        self.sdk_session_id = sdk_session_id

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
                self.settled.append(kept)


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
        # Per client, not `app.on_shutdown`: that would accumulate one handler
        # per connection and hold every session's subprocess open until the app
        # itself exits.
        context.client.on_disconnect(session.release)
        shell.build(workspace)


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

    @nicegui_app.on_startup
    def _wire() -> None:
        desktop.bind_loop(asyncio.get_running_loop())
        if not native:
            # Browser mode has no window to hide and no tray to hide it to; the
            # tab is the affordance and closing it is the user's business.
            return
        desktop.start_tray(on_restart_lab=_restart_lab_here)
        window = getattr(nicegui_app.native, "main_window", None)
        if window is None:
            return

        def _on_closing() -> bool:
            """False cancels the close, which is how pywebview spells "hide"."""
            if desktop.hide_to_tray():
                return False
            desktop.request_quit()
            return False

        try:
            window.events.closing += _on_closing
        except Exception:  # noqa: BLE001 - an un-hookable window just closes
            log.debug("could not intercept the window close", exc_info=True)

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
    for name in appdata.migrate_legacy():
        log.info("moved data/%s into %s", name, appdata.app_dir())
    paths.ensure_workspace()
    instance.publish(port)
    build()
    _install_desktop(native)
    # `window_size` is passed *only* in native mode, and that is not a
    # nicety: NiceGUI turns `native` on whenever a window size is given, so
    # passing it unconditionally made `native=False` unreachable and the
    # documented browser fallback impossible to actually take.
    extra = {"window_size": (1600, 1000)} if native else {}
    ui.run(
        native=native,
        title="Grad",
        # Explicit localhost: in browser mode NiceGUI would otherwise bind
        # 0.0.0.0, and this is an unauthenticated UI that accepts prompts for an
        # agent with Bash access.
        host="127.0.0.1",
        port=port,
        # Signs the browser-id cookie each client's transcript is keyed by.
        storage_secret=_storage_secret(),
        reload=False,
        # The design is a cream paper ground with ink rules. Quasar's dark mode
        # would fight every token in `ui/tokens.py`.
        dark=False,
        show=False,
        **extra,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
