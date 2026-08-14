"""The workspace shell: title bar, opener strip, tiling area, status bar.

The one thing worth understanding before changing anything here is how a window
survives a retile.

A window's content lives in a single element -- its *root* -- created once, the
first time the window is opened. Retiling tears down and rebuilds the pane
scaffolding, which would destroy those roots and, with them, the chat
transcript, the funnel's scroll position and the notebook's anchor. So before
the teardown every live root is moved into a hidden `attic`, and afterwards each
is moved back into its new slot. `Element.move()` reparents without rebuilding,
which is the single NiceGUI API that makes this design practical.

The Lab iframe cannot be moved even by that: browsers destroy and recreate an
`<iframe>` on reparent, reloading the document inside it. It therefore lives in
a fixed overlay outside the pane tree, flown to wherever its anchor is by
`ui/static/tiling.js`. That is the one place the window system leaves NiceGUI's
model, and it is contained to two functions.
"""

from __future__ import annotations

from typing import Any

from ui import kit, registry
from ui.state import POLL_SECONDS, Workspace


def build(workspace: Workspace) -> None:
    """Assemble the whole shell for one connected client."""
    from nicegui import ui

    roots: dict[str, Any] = {}

    with kit.el("div", "grad-app"):
        with kit.el("div", "grad-shell"):
            appbar = kit.el("div", "grad-appbar")
            opener = kit.el("div", "grad-opener")
            tiles = kit.el("div", "grad-tiles")
            statusbar = kit.el("div", "grad-statusbar")
        # Detached window roots wait here between tilings. `display: none`
        # rather than removal: a root that was destroyed would take its
        # window's state with it.
        attic = kit.el("div", "", style="display: none")

    palette = _command_palette(ui, workspace)

    def draw_appbar() -> None:
        appbar.clear()
        with appbar:
            _appbar(workspace, palette)

    def draw_opener() -> None:
        opener.clear()
        with opener:
            _opener(workspace)

    def draw_status() -> None:
        statusbar.clear()
        with statusbar:
            _statusbar(workspace)

    def draw_window(window_id: str) -> None:
        """Re-render one window's body in place, leaving its slot alone."""
        root = roots.get(window_id)
        if root is None:
            return
        root.clear()
        with root:
            _render(workspace, window_id)

    def retile() -> None:
        for root in roots.values():
            root.move(attic)
        tiles.clear()
        live = set(workspace.layout.windows)
        for window_id in list(roots):
            if window_id not in live:
                # Closed for real: drop the root, and let the notebook window
                # take its iframe down with it.
                _on_close(ui, window_id)
                roots.pop(window_id).delete()
                workspace.unbind_window(window_id)

        with tiles:
            for column_index, column in enumerate(workspace.layout.columns):
                if column_index:
                    _handle(vertical=False)
                container = kit.el("div", "grad-column", style=f"--grad-fraction: {column.fraction:.6f}")
                container.props(f'data-column-index="{column_index}"')
                with container:
                    for slot_index, slot in enumerate(column.slots):
                        if slot_index:
                            _handle(vertical=True)
                        with kit.el("div", "grad-slot", style=f"--grad-fraction: {slot.fraction:.6f}"):
                            _frame(workspace, slot.window, roots, attic, draw_window)

    workspace.bind_chrome(draw_appbar)
    workspace.bind_chrome(draw_opener)
    workspace.bind_chrome(draw_status)
    workspace.bind_retile(retile)

    draw_appbar()
    draw_opener()
    draw_status()
    retile()

    # One poll for the whole workspace; see the note at the top of ui/state.py.
    ui.timer(POLL_SECONDS, workspace.tick)

    _bind_client_events(ui, workspace)


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------
def _appbar(workspace: Workspace, palette: Any) -> None:
    header = workspace.header()
    session = header["session"]

    with kit.el("div", "grad-appbar-cell brand"):
        kit.text("∇", "grad-mark")
        kit.text("GRAD", "grad-wordmark")

    with kit.el("div", "grad-appbar-cell"):
        kit.text("project", "dim")
        kit.text(header["project"], "", style="font-weight: 700")

    with kit.el("div", "grad-appbar-cell"):
        state = header["agent_state"]
        caption = {
            "idle": "AGENT IDLE",
            "running": f"AGENT RUNNING · step {header['step']}" if header.get("step") else "AGENT RUNNING",
            "awaiting_gate": "AWAITING YOUR CALL",
            "paused": "AGENT PAUSED",
        }[state]
        kit.chip(caption, header["accent"], dot=state == "running")
        kit.button(
            "■ PAUSE" if state == "running" else "▶ RESUME",
            tone="ghost",
            classes="grad-appbar-btn",
            on_click=lambda: workspace.set_agent_state("paused" if state == "running" else "idle"),
        )

    kit.spacer()

    with kit.el("div", "grad-appbar-cell right"):
        kit.text("session", "dim")
        kit.bar(
            [
                (session["chat_fraction"] * _used_share(session), "chat", ""),
                (session["tool_fraction"] * _used_share(session), "tool", ""),
            ],
            thin=True,
        ).style("width: 150px")
        kit.text(session["used_label"], "", style="font-weight: 700")
        kit.text(f"· resets {session['resets_in']}", "dim")

    with kit.el("div", "grad-appbar-cell right"):
        kit.button("⌘K", tone="ghost", classes="grad-appbar-btn", on_click=palette.open)
        kit.button(
            "LAYOUTS ▾",
            tone="ghost",
            classes="grad-appbar-btn",
            title="tile ⌥1 · stack ⌥2 · full ⌥3",
            on_click=lambda: workspace.preset("tile"),
        )


def _used_share(session: dict[str, Any]) -> float:
    ceiling = session.get("ceiling_usd") or 0.0
    if not ceiling:
        return 0.0
    return max(0.0, min(1.0, float(session.get("used_usd", 0.0)) / float(ceiling)))


def _opener(workspace: Workspace) -> None:
    kit.text("OPEN A WINDOW →", "grad-opener-hint")
    open_ids = set(workspace.layout.windows)
    for window in registry.WINDOWS:
        is_open = window.id in open_ids
        cell = kit.text(window.name, f"grad-opener-cell {'open' if is_open else ''}".strip(), tag="button")
        cell.props(f'title="{kit.escape(window.hint)}"')
        cell.on("click", lambda _=None, wid=window.id: workspace.toggle(wid))
    kit.spacer()
    kit.text("tile ⌥1 · stack ⌥2 · full ⌥3", "grad-opener-hint")


def _statusbar(workspace: Workspace) -> None:
    status = workspace.status()
    kit.text(status["cwd"], "dim", tag="span")
    kit.text(status["kernel"], "", tag="span")
    kit.text(f"queue {status['queued']} · gpu {status['gpu']}", "", tag="span")
    if workspace.notice:
        kit.text(workspace.notice, "", tag="span")
    kit.spacer()
    kit.text("⌥drag to retile", "dim", tag="span")
    kit.text(f"{len(workspace.layout.windows)} open", "count", tag="span")


def _command_palette(ui: Any, workspace: Workspace) -> Any:
    """`⌘K`: open a window by name. The opener strip, without the mouse."""
    with ui.dialog() as dialog, kit.el("div", "grad-app"):
        with kit.el("div", "grad-card", style="background: var(--grad-paper); min-width: 420px"):
            kit.text("OPEN A WINDOW", "head ink")
            with kit.el("div", "body"):
                for window in registry.WINDOWS:
                    with kit.row("grad-row"):
                        kit.button(
                            window.name.upper(),
                            tone="neutral",
                            on_click=lambda _=None, wid=window.id: (workspace.open(wid), dialog.close()),
                        )
                        kit.text(window.hint, "grad-caption")
    return dialog


# ---------------------------------------------------------------------------
# panes
# ---------------------------------------------------------------------------
def _handle(*, vertical: bool) -> None:
    """The 8px drag handle: ink, three paper dots, the right cursor."""
    with kit.el("div", f"grad-handle {'row' if vertical else ''}".strip()):
        for _ in range(3):
            kit.el("span")


def _frame(workspace: Workspace, window_id: str, roots: dict[str, Any], attic: Any, draw_window: Any) -> None:
    """One window inside its slot: title bar, then the root moved back in."""
    spec = registry.spec(window_id)
    focused = workspace.layout.focused == window_id
    with kit.el("div", f"grad-window {'focused' if focused else ''}".strip()) as frame:
        bar = kit.row("grad-titlebar")
        bar.props(f'data-window="{window_id}"')
        bar.on("click", lambda _=None, wid=window_id: workspace.focus(wid))
        with bar:
            kit.text(spec.name, "name")
            kit.text(registry.subtitle(window_id, workspace), "subtitle")
            for text, tone in registry.chips(window_id, workspace):
                kit.chip(text, tone)
            kit.spacer()
            kit.button("⇱", tone="ghost", classes="grad-winctl", title="focus this pane",
                       on_click=lambda _=None, wid=window_id: (workspace.focus(wid), workspace.preset("full")))
            kit.button("⇲", tone="ghost", classes="grad-winctl", title="restore the tiled layout",
                       on_click=lambda: workspace.preset("tile"))
            kit.button("✕", tone="ghost", classes="grad-winctl", title="close",
                       on_click=lambda _=None, wid=window_id: workspace.close(wid))

    # Outside the `with`, on purpose: `move` sets the parent explicitly, and
    # doing it inside the block would append to whatever slot is current.
    root = roots.get(window_id)
    if root is None:
        with attic:
            root = kit.scroll_body()
        roots[window_id] = root
        workspace.bind_window(window_id, lambda wid=window_id: draw_window(wid))
        with root:
            _render(workspace, window_id)
    root.move(frame)


def _render(workspace: Workspace, window_id: str) -> None:
    """Draw one window's body, or say why it could not be drawn.

    A window whose module fails to import -- an optional dependency missing, a
    syntax error in a window nobody opened yet -- renders as a card saying so.
    Ten working windows and one broken one is a usable workspace; a traceback at
    page build time is not.
    """
    try:
        registry.renderer(window_id)(workspace)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see the docstring
        import logging

        logging.getLogger("grad.ui").exception("window %s failed to render", window_id)
        kit.empty(f"the {window_id} window failed to render: {type(exc).__name__}", "details are in the app log")


def _on_close(ui: Any, window_id: str) -> None:
    """Let a window drop anything living outside the element tree."""
    if window_id == "notebook":
        kit.run_js("window.gradDropFrame && window.gradDropFrame('grad-anchor-notebook')")


# ---------------------------------------------------------------------------
# events from the browser
# ---------------------------------------------------------------------------
def _bind_client_events(ui: Any, workspace: Workspace) -> None:
    """The four gestures `tiling.js` sends back once they have settled."""

    def on_resize(event: Any) -> None:
        data = getattr(event, "args", {}) or {}
        fractions = [float(f) for f in data.get("fractions") or []]
        workspace.resize(
            str(data.get("axis") or ""),
            fractions,
            column=data.get("column"),
            total_px=data.get("total_px"),
        )

    def on_retile(event: Any) -> None:
        data = getattr(event, "args", {}) or {}
        window_id = data.get("window")
        if isinstance(window_id, str) and window_id in registry.BY_ID:
            workspace.retile(window_id, int(data.get("column") or 0))

    def on_preset(event: Any) -> None:
        data = getattr(event, "args", {}) or {}
        workspace.preset(str(data.get("preset") or "tile"))

    ui.on("grad_resize", on_resize)
    ui.on("grad_retile", on_retile)
    ui.on("grad_preset", on_preset)
    ui.on("grad_palette", lambda _: None)
