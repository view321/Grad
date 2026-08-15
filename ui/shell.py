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
    bars: dict[str, Any] = {}

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
    projects = _project_menu(ui, workspace)

    def draw_appbar() -> None:
        appbar.clear()
        with appbar:
            _appbar(workspace, palette, projects)

    def draw_opener() -> None:
        opener.clear()
        with opener:
            _opener(workspace)

    def draw_status() -> None:
        statusbar.clear()
        with statusbar:
            _statusbar(workspace)

    def draw_window(window_id: str) -> None:
        """Re-render one window's title bar and body, leaving its slot alone."""
        bar = bars.get(window_id)
        if bar is not None:
            bar.clear()
            with bar:
                _titlebar(workspace, window_id)
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
                bars.pop(window_id, None)
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
                            _frame(workspace, slot.window, roots, bars, attic, draw_window)

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
def _appbar(workspace: Workspace, palette: Any, projects: Any) -> None:
    header = workspace.header()
    session = header["session"]

    with kit.el("div", "grad-appbar-cell brand"):
        kit.text("∇", "grad-mark")
        kit.text("GRAD", "grad-wordmark")

    with kit.el("div", "grad-appbar-cell"):
        kit.text("project", "dim")
        kit.button(
            f"{header['project']} ▾",
            tone="ghost",
            classes="grad-appbar-btn",
            title="switch project, or open another workspace folder",
            on_click=projects.open,
        )

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
    kit.text("drag a title bar to move · drop on another to swap", "dim", tag="span")
    kit.text(f"{len(workspace.layout.windows)} open", "count", tag="span")


class _Menu:
    """A dialog whose body is rebuilt each time it opens.

    `ui.dialog` builds its contents once. This one lists projects and recent
    folders, and both change *because of* what the dialog does -- create a
    project and the list it was read from is already stale. Redrawing on open is
    cheaper than binding every row to the poll, and it cannot go stale between
    the click and the dialog appearing.
    """

    def __init__(self, dialog: Any, draw: Any) -> None:
        self._dialog = dialog
        self._draw = draw

    def open(self) -> None:
        self._draw()
        self._dialog.open()

    def close(self) -> None:
        self._dialog.close()


def _project_menu(ui: Any, workspace: Workspace) -> _Menu:
    """The workspace menu: which folder, which project, and how to change both."""
    with ui.dialog() as dialog, kit.el("div", "grad-app"):
        body = kit.el(
            "div", "grad-card", style="background: var(--grad-paper); min-width: 540px"
        )

    menu = _Menu(dialog, lambda: _draw_project_menu(ui, workspace, body, dialog))
    return menu


def _draw_project_menu(ui: Any, workspace: Workspace, body: Any, dialog: Any) -> None:
    model = workspace.workspaces()
    body.clear()

    def act(coro: Any, what: str) -> None:
        """Close first, then run: `reload` redraws the shell underneath, and a
        dialog still open over it would be showing the workspace it just left."""
        dialog.close()
        workspace.spawn(coro, what)

    with body:
        kit.text("WORKSPACE", "head ink")
        with kit.el("div", "body"):
            kit.error_strip(model.get("error"))
            kit.kv([("folder", model["root"]), ("chosen by", model["source"])])

            with kit.row("", gap=6).style("margin-top: 10px"):
                folder = (
                    ui.input(placeholder="path to another workspace folder")
                    .props("borderless dense")
                    .classes("field")
                    .style("flex: 1 1 auto; padding: 0 8px")
                )
                kit.button(
                    "BROWSE…",
                    tone="neutral",
                    title="pick a folder (needs the desktop window)",
                    on_click=lambda: workspace.spawn(_browse(workspace, folder), "folder picker"),
                )
                kit.button(
                    "OPEN",
                    tone="primary",
                    title="switch this app to that folder",
                    on_click=lambda: act(
                        workspace.switch_root(folder.value or "", create=True), "workspace switch"
                    ),
                )
            kit.text(
                "a folder that does not exist yet is created; the agent's tools follow it",
                "grad-caption",
            )

            if model["recent"]:
                kit.text("RECENT", "grad-caption").style("margin-top: 12px")
                for path in model["recent"]:
                    with kit.row("grad-row", gap=6):
                        kit.button(
                            "OPEN",
                            tone="neutral",
                            on_click=lambda _=None, p=path: act(
                                workspace.switch_root(p), "workspace switch"
                            ),
                        )
                        kit.text(path, "grad-caption")

            # -- projects ---------------------------------------------------
            kit.text("PROJECTS IN THIS FOLDER", "grad-caption").style("margin-top: 16px")
            if not model["projects"]:
                kit.text("none yet — the first one is created below", "grad-empty")
            for project in model["projects"]:
                with kit.row("grad-row", gap=6):
                    kit.button(
                        "IN USE" if project["current"] else "USE",
                        tone="active" if project["current"] else "neutral",
                        disabled=project["current"] or project["status"] == "closed",
                        on_click=lambda _=None, pid=project["id"]: act(
                            workspace.use_project(pid), "project switch"
                        ),
                    )
                    kit.text(project["id"], "", style="font-weight: 700")
                    kit.text(project["title"], "grad-caption")
                    kit.spacer()
                    if project["status"] == "closed":
                        kit.chip("CLOSED", "neutral")
                    kit.text(project["spend"], "grad-caption")

            # -- a new one --------------------------------------------------
            kit.text("NEW PROJECT", "grad-caption").style("margin-top: 16px")
            with kit.row("", gap=6):
                project_id = (
                    ui.input(placeholder="id, e.g. proj-scaling-w2")
                    .props("borderless dense")
                    .classes("field")
                    .style("flex: 0 0 220px; padding: 0 8px")
                )
                title = (
                    ui.input(placeholder="what this research is")
                    .props("borderless dense")
                    .classes("field")
                    .style("flex: 1 1 auto; padding: 0 8px")
                )
                kit.button(
                    "CREATE",
                    tone="primary",
                    on_click=lambda: act(
                        workspace.create_project(project_id.value or "", title.value or ""),
                        "project create",
                    ),
                )
            kit.text(
                "created with no ceilings — set them with `tools.budget raise`",
                "grad-caption",
            )


def folder_dialog_type() -> int:
    """`dialog_type` for "pick a folder", as something that survives pickling.

    Native mode runs pywebview in a **separate process** and marshals this call
    over a `multiprocessing` queue, so every argument has to pickle. The obvious
    constant does not: `webview.FOLDER_DIALOG` is a deprecated `proxy_tools.Proxy`
    whose repr is `20` but whose type is a proxy around a function, and pickling
    it fails with *"Can't pickle <function FOLDER_DIALOG>: it's not the same
    object as webview.FOLDER_DIALOG"*.

    That failure is also **uncatchable from here**: it is raised in the queue's
    own feeder thread, so it prints a traceback and leaves the awaited call
    hanging rather than raising where `_browse` could handle it. Sending a value
    that pickles is the only real fix, which is why this returns a plain `int`
    rather than the `FileDialog.FOLDER` enum member -- `create_file_dialog`
    declares the parameter as `int` and passes it straight through, so the
    narrowest thing that can cross a process boundary is the right one to send.
    """
    try:
        import webview  # noqa: PLC0415

        return int(webview.FileDialog.FOLDER)
    except (ImportError, AttributeError):
        # Older pywebview, where the constant existed only as the proxy above --
        # whose value was this same 20.
        return 20


async def _browse(workspace: Workspace, field: Any) -> None:
    """The native folder picker, when there is a native window to hang it on.

    Only `ui.run(native=True)` has one; the documented browser fallback does
    not, and neither does a second tab. So this fills the text field rather than
    switching directly -- the typed path is the mechanism, and the picker is a
    convenience on top of it that is allowed to be unavailable.
    """
    import logging  # noqa: PLC0415

    try:
        from nicegui import app as nicegui_app  # noqa: PLC0415

        window = getattr(getattr(nicegui_app, "native", None), "main_window", None)
        if window is None:
            raise RuntimeError("no native window")
        chosen = await window.create_file_dialog(dialog_type=folder_dialog_type())
    except Exception as exc:  # noqa: BLE001 - an unavailable picker is not an error
        logging.getLogger("grad.ui").debug("folder picker unavailable", exc_info=exc)
        workspace.say("no folder picker here — type the path instead")
        return
    if chosen:
        field.value = chosen[0] if isinstance(chosen, (list, tuple)) else str(chosen)


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


def _frame(
    workspace: Workspace,
    window_id: str,
    roots: dict[str, Any],
    bars: dict[str, Any],
    attic: Any,
    draw_window: Any,
) -> None:
    """One window inside its slot: title bar, then the root moved back in."""
    focused = workspace.layout.focused == window_id
    with kit.el("div", f"grad-window {'focused' if focused else ''}".strip()) as frame:
        bar = kit.row("grad-titlebar")
        bar.props(f'data-window="{window_id}"')
        bar.on("click", lambda _=None, wid=window_id: workspace.focus(wid))
        bars[window_id] = bar
        with bar:
            _titlebar(workspace, window_id)

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


def _titlebar(workspace: Workspace, window_id: str) -> None:
    """The bar's contents, refreshed on the same tick as the window's body.

    Separated from `_frame` because the subtitle and the state chips are read
    from the model, not from the layout: `EVOLVING` becomes `HALTING`, a verify
    turns `NOT CITABLE` into `CITABLE`, an unjudged count moves. Drawing them
    only when the panes are rebuilt left them stale until the next retile --
    a chip that lags the thing it reports is worse than no chip.
    """
    spec = registry.spec(window_id)
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
        if not isinstance(window_id, str) or window_id not in registry.BY_ID:
            return
        slot = data.get("slot")
        workspace.retile(
            window_id,
            int(data.get("column") or 0),
            int(slot) if isinstance(slot, (int, float)) else None,
            new_column=bool(data.get("new_column")),
        )

    def on_swap(event: Any) -> None:
        """Both ids are checked against the registry before either is used: this
        arrives from the browser, and `swap` writes whatever it is handed
        straight into a slot."""
        data = getattr(event, "args", {}) or {}
        a, b = data.get("a"), data.get("b")
        if not (isinstance(a, str) and isinstance(b, str)):
            return
        if a in registry.BY_ID and b in registry.BY_ID:
            workspace.swap(a, b)

    def on_preset(event: Any) -> None:
        data = getattr(event, "args", {}) or {}
        workspace.preset(str(data.get("preset") or "tile"))

    ui.on("grad_resize", on_resize)
    ui.on("grad_retile", on_retile)
    ui.on("grad_swap", on_swap)
    ui.on("grad_preset", on_preset)
    ui.on("grad_palette", lambda _: None)
