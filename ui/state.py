"""The workspace's live state: one poll, one snapshot, eleven windows.

The old app gave every panel its own refresh button and its own read of the
ledger. Eleven windows on that pattern is eleven independent pollers doing
eleven full subtree rebuilds, and the design explicitly rules the visible half
of that out: "no easing curves, no fades, no skeleton shimmer. Progress bars
update in place."

So: one `ui.timer`, one pass that rebuilds the models for the windows that are
actually open, and a fingerprint per window so a redraw only happens when that
window's data changed. A workspace with the ledger and the queue open does two
reads per tick, not eleven, and a tick where nothing moved costs one comparison
per open window and touches no DOM.

Layout persists per project, because "which windows are open" is a property of
what you are working on. It is written on settle rather than on every drag
frame -- see `ui/static/tiling.js` for the other half of that.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from core import appdata, jsonl, paths
from ui import layout as layout_mod, models, registry
from ui.tasks import envelope_message, run_tool

log = logging.getLogger("grad.ui")

#: How often the open windows re-read their sources. Two seconds is well under
#: the rate a human notices and well over the rate a JSONL read costs.
POLL_SECONDS = 2.0

#: How often the streaming chat tail flushes. Per token would re-render and
#: reflow the whole element on every token.
FLUSH_HZ = 15


def layout_dir() -> Path:
    """Under the app directory, but keyed to this workspace.

    Both halves are load-bearing. Out of the repo, because an arrangement of
    panes is a property of this machine's screen and not of the research. Keyed
    to the workspace, because it used to live *inside* the root and so a switch
    to a fresh folder opened the default arrangement rather than carrying the
    old one across -- one flat directory would quietly hand every workspace the
    same panes. Per-project within that, which is the next line down.
    """
    return appdata.workspace_state_dir() / "layouts"


def layout_path(project: str | None) -> Path:
    """One layout file per project, named from an untrusted id.

    The project id comes out of a config file, so it is input to a filename
    rather than a constant. Dropping separators alone is not enough: it turns
    `../../etc/passwd` into `....etcpasswd`, which cannot escape the directory
    but is a name nobody can read. Leading dots and doubled dots go too.
    """
    safe = "".join(c for c in str(project or "default") if c.isalnum() or c in "._-")
    while ".." in safe:
        safe = safe.replace("..", ".")
    safe = safe.strip("._-")[:64] or "default"
    return layout_dir() / f"{safe}.json"


def load_layout(project: str | None) -> layout_mod.Layout:
    """The saved arrangement, or the one a workspace with no history opens with.

    Unknown window ids are dropped rather than raised on, so a layout written by
    a version with a window this one does not have still opens.
    """
    data = jsonl.read_json(layout_path(project))
    restored = layout_mod.Layout.from_dict(data, known=registry.ids())
    if restored.columns:
        return restored
    return layout_mod.Layout.default(opening_windows())


def opening_windows() -> tuple[str, ...]:
    """What opens when this workspace has never been arranged.

    The mock's four -- unless the agent has no credentials, in which case those
    four windows are four windows that cannot do anything, and the first thing
    on screen should be the one that fixes it.

    Only reached when there is no saved layout, so this costs a credential-store
    read once per fresh workspace and nothing thereafter. And only the *token* is
    checked (`models.setup_needed`): an unconfigured backend means no remote
    training, which is a real limitation and not a reason to put a wizard in
    front of someone who opened the app to read a ledger.
    """
    try:
        if models.setup_needed():
            return ("setup", *registry.defaults())
    except Exception:  # noqa: BLE001 - never the reason a workspace will not open
        log.debug("could not decide the opening arrangement", exc_info=True)
    return registry.defaults()


def save_layout(project: str | None, value: layout_mod.Layout) -> None:
    try:
        jsonl.write_json(layout_path(project), value.to_dict())
    except OSError:  # a read-only workspace must not break retiling
        log.debug("could not persist layout for %s", project)


# ---------------------------------------------------------------------------
# which model function backs which window
# ---------------------------------------------------------------------------
# `chat` is absent on purpose: its state is the live SDK session, not a file, so
# it redraws from its own stream rather than from the poll.
MODEL_BUILDERS: dict[str, Callable[["Workspace"], Any]] = {
    "setup": lambda w: models.setup_model(),
    "projects": lambda w: models.projects_model(),
    "notebook": lambda w: models.notebook_model(),
    "ledger": lambda w: models.ledger_model(),
    "quota": lambda w: models.quota_model(),
    "wiki": lambda w: models.wiki_model(w.selection.get("wiki.page")),
    "papers": lambda w: models.papers_model(filter_name=w.selection.get("papers.filter", "cited")),
    "evolve": lambda w: models.evolve_model(w.selection.get("evolve.campaign")),
    "editor": lambda w: models.editor_model(w.project),
    "preflight": lambda w: models.preflight_model(),
    "funnel": lambda w: models.funnel_model(w.selection.get("funnel.trace")),
    "queue": lambda w: models.queue_model(),
    # The agent's own calls are read from the live session rather than from a
    # file, so they are passed in: `ui/models.py` never reaches for the session,
    # and the tasks window is the one place that wants both halves of "what is
    # running on this machine right now".
    "tasks": lambda w: models.tasks_model(agent=models.agent_calls_model(w.session)),
}


#: Windows whose model must be built on the event loop. `tasks` reads the live
#: SDK session rather than a file -- no I/O to move off, and a real race if it
#: were read from a worker thread while the loop mutates it.
LOOP_BOUND = frozenset({"tasks"})


def current_project() -> str | None:
    """The selected project, or None. Never raises: an unreadable project file
    means an unnamed workspace, not an app that will not open."""
    from core import budget as budget_mod  # noqa: PLC0415

    try:
        return budget_mod.current_project()
    except Exception:  # noqa: BLE001 - see the docstring
        return None


def _fingerprint(value: Any) -> str:
    """Cheap change detection. Sorted keys so dict order cannot fake a change."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


class Workspace:
    """Everything one connected client's window system knows.

    One per client, like `Session`: two browser windows sharing a `Workspace`
    would share a focused pane and fight over the layout file.
    """

    def __init__(self, session: Any, project: str | None) -> None:
        self.session = session
        self.project = project
        self.agent_state = "idle"
        self.step: int | None = None
        self.layout = load_layout(project)
        self.models: dict[str, Any] = {}
        self.selection: dict[str, Any] = {}
        self.notice: str | None = None
        #: Whether the agent's reasoning is on screen. Kept here rather than in
        #: the chat window's closure so it survives the window being redrawn --
        #: switching session rebuilds that closure, and a display preference that
        #: silently reset itself would read as the toggle having broken.
        self.show_reasoning = False
        #: Set by the chat window once it is built. A gate card in the
        #: transcript needs to send the approval back into the same session, and
        #: routing it through here keeps the gate card from having to reach into
        #: another window's closure.
        self.chat_send: Callable[[str], Any] | None = None
        #: Wakes that have arrived and not yet been turned into a prompt.
        #:
        #: A queue rather than a direct call, because the two ends are in
        #: different worlds: a wake arrives on a FastAPI route with no client in
        #: scope, and `chat_send` draws into elements that belong to one. The
        #: poll already runs in that client's context on a two-second tick, and
        #: two seconds is nothing next to the hours a wake waits.
        self.pending_wakes: list[str] = []
        self._fingerprints: dict[str, str] = {}
        self._redraw: dict[str, Callable[[], None]] = {}
        #: Titlebar redraws, kept apart from the bodies: a titlebar reports live
        #: state (STREAMING, HALTING, an unjudged count) and has to refresh on
        #: every tick, including for windows whose body the poll does not build.
        self._titlebars: dict[str, Callable[[], None]] = {}
        self._chrome: list[Callable[[], None]] = []
        self._retile: Callable[[], None] | None = None
        #: Strong references to in-flight tasks; see `spawn`.
        self._tasks: set[asyncio.Task[Any]] = set()
        #: Drops a poll that arrives while the previous one is still awaiting.
        self._ticking = False
        #: The root this client believes it is on. `GRAD_ROOT` is process-wide,
        #: so another client switching folders moves this one's paths out from
        #: under it; `tick` compares against `paths.root()` and catches up.
        self._root = str(paths.root())

    # -- wiring -------------------------------------------------------------
    def bind_window(self, window_id: str, redraw: Callable[[], None]) -> None:
        self._redraw[window_id] = redraw

    def unbind_window(self, window_id: str) -> None:
        self._redraw.pop(window_id, None)
        self._titlebars.pop(window_id, None)

    def bind_titlebar(self, window_id: str, redraw: Callable[[], None]) -> None:
        self._titlebars[window_id] = redraw

    def bind_chrome(self, redraw: Callable[[], None]) -> None:
        self._chrome.append(redraw)

    def bind_retile(self, retile: Callable[[], None]) -> None:
        self._retile = retile

    # -- models -------------------------------------------------------------
    def model(self, window_id: str) -> Any:
        """This window's data, computing it once if the poll has not yet."""
        if window_id not in self.models and window_id in MODEL_BUILDERS:
            self.rebuild(window_id)
        return self.models.get(window_id)

    def _compute(self, window_id: str) -> Any:
        """Run one window's builder. Safe to call off the event loop: everything
        in `MODEL_BUILDERS` reads files and returns plain data, and none of it
        touches a NiceGUI element."""
        builder = MODEL_BUILDERS.get(window_id)
        if builder is None:
            return None
        try:
            return builder(self)
        except Exception as exc:  # noqa: BLE001 - a window's own failure is its own
            log.exception("model for %s failed", window_id)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _apply(self, window_id: str, value: Any) -> bool:
        mark = _fingerprint(value)
        changed = self._fingerprints.get(window_id) != mark
        self.models[window_id] = value
        self._fingerprints[window_id] = mark
        return changed

    def rebuild(self, window_id: str) -> bool:
        """Recompute one window's model, on this thread. Returns whether it
        changed. The poll uses `rebuild_off_thread`; this is the lazy path taken
        by `model()` when a window renders before its first tick."""
        if window_id not in MODEL_BUILDERS:
            return False
        return self._apply(window_id, self._compute(window_id))

    async def rebuild_off_thread(self, window_id: str) -> bool:
        """`rebuild`, with the file I/O moved off the event loop.

        Every builder here reads files, and one of them opens a socket:
        `tools/lab.py:_listening` probes Lab's port with a 0.4 second timeout,
        on the poll, while the notebook window is open. On the loop that is a
        400ms freeze of the entire app -- every window, the chat stream and the
        pane dragging included -- waiting on a TCP connect. It costs a
        millisecond while Lab is answering, which is why it read as harmless;
        it is a port that has stopped answering that makes the app look hung.

        `tasks` is deliberately not offloaded. It reads the live SDK session
        rather than a file, so there is no I/O to move, and reading that object
        from another thread while the loop mutates it is a race for no gain.
        """
        if window_id not in MODEL_BUILDERS:
            return False
        if window_id in LOOP_BOUND:
            value = self._compute(window_id)
        else:
            value = await asyncio.to_thread(self._compute, window_id)
        return self._apply(window_id, value)

    def invalidate(self, window_id: str) -> None:
        """Force this window to redraw on the next tick even if its file did not
        change -- what a filter chip or a trace selector needs."""
        self._fingerprints.pop(window_id, None)
        self.models.pop(window_id, None)

    def _caught_up_after_move(self) -> bool:
        """Whether the workspace root moved under this client, handling it if so.

        Another client switching folders moves every path the models read, so
        catching up has to happen before any of them are rebuilt -- and the
        rebuild is then pointless, because `reload` has already done it.
        """
        if str(paths.root()) == self._root:
            return False
        self._root = str(paths.root())
        self.project = current_project()
        rebind = getattr(self.session, "rebind", None)
        if rebind is not None:
            self.spawn(rebind(), "workspace rebind")
        self.reload()
        self.say(f"workspace moved to {paths.root()}")
        return True

    def _redraw_chrome(self) -> None:
        """The half of a pass that must happen on the event loop.

        The titlebars are refreshed separately from the bodies. `chat` has no
        entry in `MODEL_BUILDERS` -- its body redraws from its own stream rather
        than from this poll -- so a titlebar drawn only when `rebuild` returned
        True was never redrawn at all for that window: the STREAMING chip and the
        "N messages" subtitle sat at whatever they said when the pane was last
        tiled, which is exactly what `_titlebar`'s docstring says must not
        happen.
        """
        for window_id, redraw in list(self._titlebars.items()):
            if window_id in self.layout.windows:
                _guard(redraw, f"{window_id}.titlebar")
        for redraw in self._chrome:
            _guard(redraw, "chrome")

    def tick(self) -> None:
        """Refresh now, on this thread.

        What a window's own buttons call after they have changed something: the
        user is waiting for the result of a click, and one synchronous pass is
        the shortest path to it. The repeating two-second poll is `poll`, which
        is the one that must not block.
        """
        if self._caught_up_after_move():
            return
        for window_id in list(self.layout.windows):
            if self.rebuild(window_id) and window_id in self.layout.windows:
                redraw = self._redraw.get(window_id)
                if redraw is not None:
                    _guard(redraw, window_id)
        self._redraw_chrome()

    async def poll(self) -> None:
        """`tick`, with the file I/O off the event loop. The timer's callback.

        The redraws stay on the loop, because they build NiceGUI elements and
        those belong to the loop that owns the client. Only the builders move --
        see `rebuild_off_thread` for the 400ms socket probe that motivates it.

        A pass that arrives while the previous one is still awaiting is dropped
        rather than queued. A poll that outruns `POLL_SECONDS` is a slow disk or
        a hung port, and stacking a second pass on top of it turns one slow poll
        into an unbounded pile of them -- each holding a thread, all rebuilding
        the same windows.
        """
        if self._ticking:
            return
        self._ticking = True
        try:
            if self._caught_up_after_move():
                return
            await self._deliver_wakes()
            # Snapshotted, because the rebuilds below await: a retile landing
            # mid-pass would otherwise have this iterating a list that changed
            # under it. A window that opened during the pass is picked up by the
            # next one, two seconds later.
            for window_id in list(self.layout.windows):
                changed = await self.rebuild_off_thread(window_id)
                if changed and window_id in self.layout.windows:
                    redraw = self._redraw.get(window_id)
                    if redraw is not None:
                        _guard(redraw, window_id)
            self._redraw_chrome()
        finally:
            self._ticking = False

    # -- wakeups ------------------------------------------------------------
    #: More than this many wakes waiting is a runaway watcher, not a research
    #: session. They are dropped at the door with a line in the status bar
    #: rather than queued into a conversation nobody asked for.
    MAX_PENDING_WAKES = 8

    def accept_wake(self, prompt: str) -> bool:
        """Take a wake for delivery on the next tick. Returns whether it was kept.

        Called from the HTTP route, so it must do nothing that needs a client:
        no elements, no drawing, no `say`.
        """
        if not prompt.strip():
            return False
        if len(self.pending_wakes) >= self.MAX_PENDING_WAKES:
            log.warning("dropping a wake: %d already queued", len(self.pending_wakes))
            return False
        self.pending_wakes.append(prompt)
        return True

    async def _deliver_wakes(self) -> bool:
        """Turn one queued wake into a turn, if now is a moment that can take it.

        One per tick and never while the session is busy. A wake is a prompt, and
        `Session.ask` refuses a prompt during a turn -- so delivering into a
        running turn would consume the wake and answer nothing, which is exactly
        the silent loss this whole mechanism exists to prevent. Held instead, and
        the next tick tries again.
        """
        if not self.pending_wakes:
            return False
        if getattr(self.session, "busy", False):
            return False
        send = self.chat_send
        if send is None:
            # The chat window is closed, so there is nowhere for a turn to be
            # drawn. Opening it is the honest response: something is trying to
            # wake this session and the window that shows sessions is shut.
            self.open("chat")
            self.say("a wakeup arrived — opening the session window for it")
            return False

        prompt = self.pending_wakes.pop(0)
        self.set_agent_state("running")
        try:
            result = send(prompt)
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - a failed wake must not stop the poll
            log.exception("could not deliver a wakeup")
            self.set_agent_state("idle")
            return False
        return True

    # -- layout moves -------------------------------------------------------
    def toggle(self, window_id: str) -> None:
        self.layout.toggle(window_id)
        self._after_layout_change()

    def open(self, window_id: str) -> None:
        self.layout.open(window_id)
        self._after_layout_change()

    def close(self, window_id: str) -> None:
        self.layout.close(window_id)
        self._after_layout_change()

    def focus(self, window_id: str) -> None:
        if self.layout.focused == window_id:
            return
        self.layout.focus(window_id)
        self._after_layout_change()

    def preset(self, name: str) -> None:
        try:
            self.layout.apply_preset(name)
        except ValueError:
            return
        self._after_layout_change()

    def retile(
        self, window_id: str, column: int, slot: int | None = None, *, new_column: bool = False
    ) -> None:
        self.layout.move(window_id, column, slot, new_column=new_column)
        self._after_layout_change()

    def swap(self, a: str, b: str) -> None:
        self.layout.swap(a, b)
        self._after_layout_change()

    def resize(self, axis: str, fractions: list[float], *, column: int | None = None, total_px: int | None = None) -> None:
        """Settle a drag. No redraw: the browser already moved the panes, and
        rebuilding the tree here would throw away the gesture's own result."""
        if axis == "columns":
            self.layout.resize_columns(fractions, total_px=total_px)
        elif axis == "slots" and column is not None:
            self.layout.resize_slots(column, fractions, total_px=total_px)
        else:
            return
        save_layout(self.project, self.layout)

    def _after_layout_change(self) -> None:
        save_layout(self.project, self.layout)
        if self._retile is not None:
            _guard(self._retile, "retile")
        for redraw in self._chrome:
            _guard(redraw, "chrome")

    # -- the workspace itself -----------------------------------------------
    def reload(self) -> None:
        """Re-read everything derived from the root or the current project.

        Both a folder switch and a project switch land here, because the same
        things are stale either way: the layout is stored per project *under*
        the root, and every window's model is a read of a file beneath it.

        The windows are redrawn explicitly rather than left to the poll. A
        retile reuses live roots -- that is what keeps a drag from wiping the
        transcript -- so without this the panes would be rearranged for the new
        workspace while still showing the old one's contents until something
        happened to change a fingerprint.
        """
        self.project = current_project()
        self.layout = load_layout(self.project)
        self.models.clear()
        self._fingerprints.clear()
        self.selection.clear()
        self.agent_state = "idle"
        self.step = None
        if self._retile is not None:
            _guard(self._retile, "retile")
        # A copy: `retile` unbinds windows the new layout does not have.
        for window_id, redraw in list(self._redraw.items()):
            _guard(redraw, window_id)
        for redraw in self._chrome:
            _guard(redraw, "chrome")

    async def switch_root(self, folder: str, *, create: bool = False) -> None:
        """Point the whole app at another workspace folder.

        Three things have to move together, and missing any one of them leaves
        the app half-switched in a way that is hard to see:

        * **the paths**, via `GRAD_ROOT` -- which also carries to every CLI the
          UI and the agent shell out to, since they inherit this environment;
        * **the config cache**, because `config/grad.toml` moved with the root;
        * **the session**, because its transcript file is derived from the root
          and its SDK client's working directory was fixed when it was built.
          A session left alone would keep the old workspace's conversation on
          screen and keep running the agent's tools in the old directory.

        **This is process-global, and every connected client moves.** `GRAD_ROOT`
        is one environment variable in one process, so a second window cannot
        stay behind: its paths follow the switch while its `Workspace` goes on
        believing otherwise, and its next `_persist` would write its conversation
        into the new root under the old session's name. The other clients catch
        up on their own next `tick`, which compares `paths.root()` against the
        root they were built on -- pull rather than push, so a client that has
        disconnected but not yet been collected is not resurrected to be told.
        """
        from core import config as config_mod, workspace as workspace_mod  # noqa: PLC0415

        try:
            chosen = workspace_mod.select(folder, create=create)
        except Exception as exc:  # noqa: BLE001 - a bad path is a message, not a crash
            log.debug("workspace switch refused", exc_info=exc)
            self.say(getattr(exc, "message", None) or str(exc))
            return

        paths.ensure_workspace()
        config_mod._cache.clear()  # noqa: SLF001 - the config path moved with the root
        rebind = getattr(self.session, "rebind", None)
        if rebind is not None:
            await rebind()
        self._root = str(paths.root())
        self.reload()
        self.say(f"workspace: {chosen}")

    async def create_project(
        self,
        project_id: str,
        title: str,
        *,
        ceilings: dict[str, str] | None = None,
        payer: str = "",
    ) -> None:
        """Create a project and select it, by running the same command the agent
        would (§10) -- so it lands in the same ledger and reads back the same.

        The ceilings go in *here*, on `budget new`, and not as a raise
        afterwards. A raise appends a `project_budget_raised` event carrying a
        `previous` value, so setting the first ceiling through one would record
        that a project's GPU allowance moved from nothing to $50 -- which is not
        what happened, and the history is the reason that module is append-only
        at all. Creating with them is one record saying what the project was
        allowed from the start.

        This is also the fix for the hole the old form left: it created with no
        ceilings and put "set them below once it is selected" in a caption. A
        project with no ceilings bounds nothing and every gate that reads one
        passes silently.
        """
        argv = ["tools.budget", "new", "--id", project_id, "--title", title, "--use"]
        for flag, value in (ceilings or {}).items():
            if (value or "").strip():
                argv += [f"--{flag}", str(value).strip()]
        if (payer or "").strip():
            argv += ["--payer", payer.strip()]
        payload = await run_tool(*argv, "--json")
        self.say(envelope_message(payload))
        if not payload.get("ok"):
            return
        self.reload()
        # The machine half, and only when there is something in it left to
        # answer. A wizard that opened on every project creation would ask a user
        # with six projects for their Claude token six times; one that never
        # opens leaves a fresh install with a project it cannot run.
        try:
            if models.setup_needed():
                self.open("setup")
                self.say("no credentials yet — setup is open beside it")
        except Exception:  # noqa: BLE001 - never the reason a create is reported as failed
            log.debug("could not decide whether to open setup", exc_info=True)

    async def use_project(self, project_id: str) -> None:
        payload = await run_tool("tools.budget", "use", project_id, "--json")
        self.say(envelope_message(payload))
        if payload.get("ok"):
            self.reload()

    async def configure_project(
        self,
        project_id: str,
        *,
        role: str = "",
        model: str = "",
        backend: str = "",
    ) -> None:
        """Set or clear one of this project's overrides.

        An empty `model` for a named role clears it, which is what the ✕ beside
        each role sends -- `--clear` rather than an empty value, because an
        override stored as empty resolves as falsy everywhere and is therefore
        present, wrong and invisible.

        The rebuild this may need is not done here. `Session.apply_model` does it
        lazily, immediately before the next turn, for the reason `apply_effort`
        is lazy: dropping and respawning the SDK subprocess is seconds, and
        paying it at the click would make idly comparing two projects cost more
        than using either.
        """
        argv = ["tools.budget", "configure", "--project", project_id]
        # An explicit flag rather than counting argv. The length sentinel was
        # correct only while the prefix stayed four words long, which is exactly
        # the kind of thing a later flag breaks silently -- and the failure would
        # be a command that runs with nothing to do and reports success.
        changed = False
        if role and model:
            argv += [f"--{role}", model]
            changed = True
        elif role:
            argv += ["--clear", role]
            changed = True
        if backend:
            argv += ["--backend", backend]
            changed = True
        if not changed:
            self.say("nothing to change — pick a model or a backend")
            return
        await self.run_and_reload(*argv, "--json")

    # -- setup ---------------------------------------------------------------
    # Every one of these runs the same command a terminal would (§10), so the
    # window cannot grow a second way to write a setting. They reload rather
    # than invalidate: a model role or a host changes what `config.load()`
    # answers, and that is read by every window and by the agent's own tools.
    async def set_model(self, role: str, model: str) -> None:
        if not (model or "").strip():
            self.say("no model given — paste an id or pick one of the buttons")
            return
        await self.run_and_reload("tools.setup", "models", f"--{role}", model.strip(), "--json")

    async def clear_model(self, role: str) -> None:
        await self.run_and_reload("tools.setup", "models", "--clear", role, "--json")

    async def set_backend(self, name: str) -> None:
        await self.run_and_reload("tools.setup", "backend", "--default", name, "--json")

    async def set_context_budget(self, tokens: str) -> None:
        """Where the agent compacts. Through the CLI, like every other setting
        here, so the window cannot grow a second way to write one."""
        value = (tokens or "").strip().replace(",", "").replace("_", "")
        if not value:
            self.say("no number given — type a token count, or pick one of the buttons")
            return
        await self.run_and_reload("tools.setup", "context", "--compact-at", value, "--json")

    async def clear_context_budget(self) -> None:
        await self.run_and_reload("tools.setup", "context", "--clear", "--json")

    async def add_host(self, name: str, hostname: str, user: str, rate: str) -> None:
        if not name or not hostname:
            self.say("a host needs a name and a hostname")
            return
        await self.run_and_reload(
            "tools.setup", "host", "add",
            "--name", name,
            "--hostname", hostname,
            "--user", user,
            "--rate", rate or "0",
            "--json",
        )

    async def remove_host(self, name: str) -> None:
        await self.run_and_reload("tools.setup", "host", "remove", "--name", name, "--json")

    async def set_kaggle_account(self, username: str) -> None:
        if not (username or "").strip():
            self.say("no username given")
            return
        await self.run_and_reload("tools.kaggle", "account", "--set", username.strip(), "--json")

    async def run_and_reload(self, *argv: str) -> None:
        """Run a CLI, report it, and re-read everything derived from it.

        For the buttons that change what the *whole workspace* is looking at --
        a ceiling moved, a project created -- as opposed to one window's data,
        which `invalidate` covers more cheaply.
        """
        payload = await run_tool(*argv)
        self.say(envelope_message(payload))
        if payload.get("ok"):
            self.reload()

    def workspaces(self) -> dict[str, Any]:
        return models.workspaces_model()

    def projects(self) -> dict[str, Any]:
        """A fresh read for the `project ▾` switcher.

        Not `model("projects")`: that is the poll's cached copy, and the window
        it feeds may not even be open. A menu is redrawn on open precisely
        because what it lists changes because of what it does.
        """
        return models.projects_model()

    def update(self) -> dict[str, Any]:
        return models.update_model()

    async def check_update(self) -> None:
        """Ask the remote now, instead of waiting for the daily check.

        `--json` and the same CLI the terminal runs, like every other button
        here (§10): the answer lands in the same `update.json` the background
        thread writes, so a check from the menu and a check from a shell cannot
        leave the app showing two different things.
        """
        payload = await run_tool("tools.update", "check", "--json")
        self.say(envelope_message(payload))
        self.reload()

    async def apply_update(self) -> None:
        """Apply the update, and say what to do next.

        Bounded by `run_tool`'s timeout, which is the right bound because of
        what `core/update.py` will and will not do while Grad is running: a
        release that changed dependencies is *refused* here, with the message
        that it needs a quit and a terminal. What is left is a fast-forward and
        a migration pass, which are file operations.

        Nothing restarts itself. The new code is on disk and the running process
        is still the old one, and an app that vanished and came back while a
        turn was streaming would be a worse surprise than a line asking for a
        restart.
        """
        payload = await run_tool("tools.update", "apply", "--json", timeout=300.0)
        self.say(envelope_message(payload))
        self.reload()

    def credentials(self) -> dict[str, Any]:
        return models.credentials_model()

    def sessions(self) -> dict[str, Any]:
        return models.sessions_model(getattr(self.session, "session_id", None))

    def rebuild_chat(self) -> None:
        """Redraw the chat window, which the poll deliberately never touches.

        Its state is the live session rather than a file, so a redraw costs the
        transcript's scroll position -- which is exactly why the poll leaves it
        alone. Switching session replaces the transcript wholesale, so here that
        cost is the entire point.
        """
        redraw = self._redraw.get("chat")
        if redraw is not None:
            _guard(redraw, "chat")
        for redraw in self._chrome:
            _guard(redraw, "chrome")

    async def set_credential(self, name: str, value: str) -> None:
        """Store one credential, down a pipe rather than as an argument.

        The same command the README tells you to run, with `--stdin` instead of
        the `getpass` prompt -- because the prompt needs a terminal, and needing
        a terminal for this was the only thing that forced one open beside the
        app on a fresh machine.
        """
        if not value.strip():
            self.say("nothing to store — paste the token first")
            return
        payload = await run_tool(
            "tools.jobs", "credential", "set", name, "--stdin", "--json", stdin=value
        )
        # `envelope_message` and nothing else: the CLI never prints a value, and
        # neither does this, but the notice is worth being explicit about.
        self.say(f"{name}: {envelope_message(payload)}")

    async def delete_credential(self, name: str) -> None:
        payload = await run_tool("tools.jobs", "credential", "delete", name, "--json")
        self.say(f"{name}: {envelope_message(payload)}")

    # -- selections ---------------------------------------------------------
    def select(self, key: str, value: Any, *, window: str | None = None) -> None:
        self.selection[key] = value
        target = window or key.split(".")[0]
        self.invalidate(target)
        self.rebuild(target)
        redraw = self._redraw.get(target)
        if redraw is not None:
            _guard(redraw, target)

    # -- header -------------------------------------------------------------
    def header(self) -> dict[str, Any]:
        return models.header_model(agent_state=self.agent_state, step=self.step)

    def status(self) -> dict[str, Any]:
        return models.status_model()

    def set_agent_state(self, state: str, *, step: int | None = None) -> None:
        self.agent_state = state
        self.step = step
        for redraw in self._chrome:
            _guard(redraw, "chrome")

    def interrupt_turn(self) -> None:
        """Stop the turn in flight. Bound to the appbar's STOP button.

        The button this replaced only changed the header caption, so the app
        could say "AGENT PAUSED" while tokens were still streaming. Interrupting
        is the thing the SDK can actually do, and `Session.interrupt` is already
        the tested path for it -- Escape and the chat window's own button use it.

        What it *says* comes from the session rather than from here, because only
        the session knows which of the three answers applies: nothing was
        running, an interrupt is already in flight, or one has just been asked
        for. The line used to be a fixed "interrupting the turn…" printed
        whatever happened, which is how a refused interrupt looked like a
        successful one.
        """
        session = self.session
        if session is None:
            return
        self.say(session.interrupt())

    def toggle_reasoning(self) -> bool:
        """Show or hide the agent's reasoning. Returns the new state.

        No redraw: the blocks are always in the DOM and a class on the chat root
        decides whether they are drawn. Rebuilding the transcript to hide a block
        would cost its scroll position, which is the same reason the poll never
        touches this window.
        """
        self.show_reasoning = not self.show_reasoning
        return self.show_reasoning

    def cycle_effort(self) -> str:
        """Step the agent's reasoning effort round the ring. Returns the new level.

        Takes effect on the next turn rather than now, and the notice says so.
        The SDK fixes effort when a client is built and offers no request to
        change it, so applying this immediately would mean tearing down the
        conversation and resuming it while the user is only browsing the levels
        -- see `core/effort.py` and `Session.apply_effort`.
        """
        from core import effort  # noqa: PLC0415 - keeps app state off the import path

        level = effort.set_current(effort.cycle())
        session = self.session
        running = session is not None and session.client_effort not in (None, level)
        self.say(
            f"reasoning effort: {level}"
            + (" — takes effect on the next turn" if running else "")
        )
        return level

    def say(self, message: str | None) -> None:
        """A one-line notice in the status bar: what a button just did."""
        self.notice = message
        for redraw in self._chrome:
            _guard(redraw, "chrome")

    # -- async work started from a click ------------------------------------
    def spawn(self, coro: Any, what: str) -> Any:
        """Run a coroutine started by a click handler, and do not lose it.

        Two failure modes this closes, both easy to write by accident and both
        silent:

        * **The task is garbage collected mid-flight.** asyncio keeps only a
          *weak* reference to a running task, so a bare `create_task(...)` whose
          result nobody holds can vanish part-way through. The turn simply
          stops, with nothing on screen and nothing in the log.
        * **The exception is retrieved and thrown away.** `t.exception()` in a
          done-callback silences Python's "Task exception was never retrieved"
          warning by *discarding* the error, which is worse than the warning it
          suppresses -- a gate approval that failed inside the SDK becomes
          invisible.

        So: a strong reference until the task settles, and a failure that
        reaches both the log and the status bar.
        """
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._settle(t, what))
        return task

    def _settle(self, task: Any, what: str) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        # Only the exception class goes on screen: an SDK message can carry a
        # URL with a token in it, and the status bar is not a log.
        log.error("%s failed", what, exc_info=error)
        self.say(f"{what} failed: {type(error).__name__} (details are in the app log)")


def _guard(fn: Callable[[], None], what: str) -> None:
    """A redraw that raises must not stop the other ten windows redrawing."""
    try:
        fn()
    except Exception:  # noqa: BLE001 - deliberately broad; see the docstring
        log.exception("redraw of %s failed", what)


# `run_tool` and `envelope_message` moved to `ui/tasks.py`, next to `start` --
# the two are the same decision made twice ("wait for this command" against
# "watch it"), and having them in one module is what keeps the timeout on the
# waiting one from being applied to something that should never have had one.
