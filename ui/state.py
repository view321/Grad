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

from core import jsonl, paths
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
    return paths.data_dir() / "layouts"


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
    """The saved arrangement, or the mock's opening one.

    Unknown window ids are dropped rather than raised on, so a layout written by
    a version with a window this one does not have still opens.
    """
    data = jsonl.read_json(layout_path(project))
    restored = layout_mod.Layout.from_dict(data, known=registry.ids())
    if restored.columns:
        return restored
    return layout_mod.Layout.default(registry.defaults())


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
    "notebook": lambda w: models.notebook_model(),
    "ledger": lambda w: models.ledger_model(),
    "quota": lambda w: models.quota_model(),
    "wiki": lambda w: models.wiki_model(),
    "papers": lambda w: models.papers_model(filter_name=w.selection.get("papers.filter", "cited")),
    "evolve": lambda w: models.evolve_model(w.selection.get("evolve.campaign")),
    "editor": lambda w: models.editor_model(w.project),
    "preflight": lambda w: models.preflight_model(),
    "funnel": lambda w: models.funnel_model(w.selection.get("funnel.trace")),
    "queue": lambda w: models.queue_model(),
    "tasks": lambda w: models.tasks_model(),
}


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
        #: Set by the chat window once it is built. A gate card in the
        #: transcript needs to send the approval back into the same session, and
        #: routing it through here keeps the gate card from having to reach into
        #: another window's closure.
        self.chat_send: Callable[[str], Any] | None = None
        self._fingerprints: dict[str, str] = {}
        self._redraw: dict[str, Callable[[], None]] = {}
        self._chrome: list[Callable[[], None]] = []
        self._retile: Callable[[], None] | None = None
        #: Strong references to in-flight tasks; see `spawn`.
        self._tasks: set[asyncio.Task[Any]] = set()

    # -- wiring -------------------------------------------------------------
    def bind_window(self, window_id: str, redraw: Callable[[], None]) -> None:
        self._redraw[window_id] = redraw

    def unbind_window(self, window_id: str) -> None:
        self._redraw.pop(window_id, None)

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

    def rebuild(self, window_id: str) -> bool:
        """Recompute one window's model. Returns whether it changed."""
        builder = MODEL_BUILDERS.get(window_id)
        if builder is None:
            return False
        try:
            value = builder(self)
        except Exception as exc:  # noqa: BLE001 - a window's own failure is its own
            log.exception("model for %s failed", window_id)
            value = {"error": f"{type(exc).__name__}: {exc}"}
        mark = _fingerprint(value)
        changed = self._fingerprints.get(window_id) != mark
        self.models[window_id] = value
        self._fingerprints[window_id] = mark
        return changed

    def invalidate(self, window_id: str) -> None:
        """Force this window to redraw on the next tick even if its file did not
        change -- what a filter chip or a trace selector needs."""
        self._fingerprints.pop(window_id, None)
        self.models.pop(window_id, None)

    def tick(self) -> None:
        """One poll: rebuild the open windows, redraw the ones that moved."""
        for window_id in self.layout.windows:
            if self.rebuild(window_id):
                redraw = self._redraw.get(window_id)
                if redraw is not None:
                    _guard(redraw, window_id)
        for redraw in self._chrome:
            _guard(redraw, "chrome")

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
        self.reload()
        self.say(f"workspace: {chosen}")

    async def create_project(self, project_id: str, title: str) -> None:
        """Create a project and select it, by running the same command the agent
        would (§10) -- so it lands in the same ledger and reads back the same."""
        payload = await run_tool(
            "tools.budget", "new", "--id", project_id, "--title", title, "--use", "--json"
        )
        self.say(envelope_message(payload))
        if payload.get("ok"):
            self.reload()

    async def use_project(self, project_id: str) -> None:
        payload = await run_tool("tools.budget", "use", project_id, "--json")
        self.say(envelope_message(payload))
        if payload.get("ok"):
            self.reload()

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
