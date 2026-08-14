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
import sys
from pathlib import Path
from typing import Any, Callable

from core import jsonl, paths
from ui import layout as layout_mod, models, registry

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
}


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

    def retile(self, window_id: str, column: int) -> None:
        self.layout.move(window_id, column)
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


def _guard(fn: Callable[[], None], what: str) -> None:
    """A redraw that raises must not stop the other ten windows redrawing."""
    try:
        fn()
    except Exception:  # noqa: BLE001 - deliberately broad; see the docstring
        log.exception("redraw of %s failed", what)


# ---------------------------------------------------------------------------
# running the CLIs the buttons are bound to
# ---------------------------------------------------------------------------
async def run_tool(*argv: str, timeout: float = 900.0) -> dict[str, Any]:
    """Run one of Grad's own CLIs and parse its JSON envelope.

    Every button in the UI that *does* something does it by shelling out to the
    same command the agent would run, with `--json`. That is deliberate: it
    keeps the UI free of logic (§10), and it means anything the UI can do is
    reproducible from a terminal and shows up in the same ledgers.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        *argv,
        cwd=str(paths.root()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "error": {"message": f"timed out after {timeout:.0f}s"}}

    stdout = (out or b"").decode("utf-8", "replace").strip()
    stderr = (err or b"").decode("utf-8", "replace").strip()
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {
        "ok": False,
        "error": {"message": (stderr or stdout or "the command produced no output")[-2000:]},
    }


def envelope_message(payload: dict[str, Any]) -> str:
    """The one line a status bar should show for a CLI result."""
    if payload.get("ok"):
        data = payload.get("data")
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
        return "done"
    error = payload.get("error") or {}
    message = error.get("message") or "the command failed"
    fix = error.get("fix")
    return f"{message}" + (f" — fix: {fix}" if fix else "")
