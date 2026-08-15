"""The NiceGUI desktop app: a tiling workspace over eleven windows.

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
* `ui/windows/`    -- eleven renderers, none of which read a ledger directly

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
from pathlib import Path
from typing import Any

from core import config as config_mod, paths
from ui import katex, kit, sessions, shell, state as state_mod
ROLES = ("user", "assistant")
STATIC_URL = "/grad-static"

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
        self._task: asyncio.Task[None] | None = None

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
            # A fresh workspace, or every session already open in another
            # window. Either way this client needs one of its own.
            chosen = sessions.LEGACY_ID if not sessions.listing() else sessions.new_id()
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
        self.settled.clear()
        # Sessions live under the root's data directory, so the one that was
        # open belongs to the folder just left -- and the claim on it does too.
        sessions.release(self.key)
        self.adopt()

    async def ask(self, prompt: str, on_settle: Any) -> None:
        await self.start()
        import agent  # noqa: PLC0415

        self.settled.append({"role": "user", "text": prompt})
        self.busy = True
        # The turn lands in the stream's blocks as it arrives; the chat window's
        # ~15 Hz timer is what turns that into something on screen. The same list
        # the stream appends to, not a copy -- a snapshot taken here would never
        # gain a tool card, and the block being written is the block being drawn.
        stream = agent.TurnStream()
        self.blocks = stream.blocks
        try:
            await self.client.query(prompt)
            async for message in self.client.receive_response():
                # Captured from the stream rather than asked for: the SDK
                # assigns it, and this is the id `resume` takes when the session
                # is reopened. Every message carrying one carries the same one,
                # so the first is enough -- but a resumed conversation can be
                # given a *new* id by the CLI, so the latest wins.
                sdk_id = getattr(message, "session_id", None)
                if isinstance(sdk_id, str) and sdk_id:
                    self.sdk_session_id = sdk_id
                stream.feed(message)
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

    def interrupt(self) -> None:
        """Interrupt the turn in flight, if there is one.

        Bound to a button and to Escape, so it fires when nothing is running.
        The result has to be consumed: a bare `create_task` drops any SDK error
        and Python logs "Task exception was never retrieved".
        """
        if not (self.busy and self.client and hasattr(self.client, "interrupt")):
            return

        async def _interrupt() -> None:
            try:
                await self.client.interrupt()
            except Exception:  # noqa: BLE001 - a failed interrupt must not kill the app
                pass

        self._task = asyncio.create_task(_interrupt())

    def path(self) -> Path:
        return sessions.path_for(self.session_id)

    def _persist(self) -> None:
        """Closing the window should not be destructive."""
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

    @ui.page("/")
    def index() -> None:
        from nicegui import context  # noqa: PLC0415 - page scope, not import scope

        session = Session(_client_key())
        session.adopt()
        workspace = state_mod.Workspace(session, _current_project())
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
    """A filesystem-safe id for this client's transcript.

    The browser id is signed into a cookie (see `_storage_secret`), so a reload
    restores its own transcript rather than someone else's; the connection id is
    the fallback when storage is unavailable, and isolates without persisting.
    """
    from nicegui import app as nicegui_app, context  # noqa: PLC0415

    try:
        key = str(nicegui_app.storage.browser.get("id") or "")
    except (RuntimeError, KeyError):  # no storage_secret configured
        key = ""
    key = key or str(context.client.id)
    return re.sub(r"[^A-Za-z0-9_-]", "", key)[:64] or "default"


def _storage_secret() -> str:
    """The key NiceGUI signs the browser-id cookie with.

    Persisted rather than generated per launch so a restart does not orphan
    every transcript written before it.
    """
    path = paths.data_dir() / "ui_storage_secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:  # not every filesystem honours it; the file is local either way
            log.debug("could not restrict permissions on %s", path)
    return path.read_text(encoding="utf-8").strip()


def run(*, native: bool = True, port: int = 8080) -> None:
    """`ui.run(native=True)` gives a real desktop window via pywebview, so the
    packaging question is answered without Electron or Tauri. Browser mode is
    the fallback when pywebview misbehaves on Windows."""
    from nicegui import ui

    paths.ensure_workspace()
    build()
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
