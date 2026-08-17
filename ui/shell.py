"""The workspace shell: title bar, tiling area, status bar.

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

from ui import desktop, kit, registry
from ui.state import POLL_SECONDS, Workspace


def build(workspace: Workspace) -> None:
    """Assemble the whole shell for one connected client."""
    from nicegui import ui

    roots: dict[str, Any] = {}
    bars: dict[str, Any] = {}

    with kit.el("div", "grad-app"):
        with kit.el("div", "grad-shell"):
            appbar = kit.el("div", "grad-appbar")
            tiles = kit.el("div", "grad-tiles")
            statusbar = kit.el("div", "grad-statusbar")
        # Detached window roots wait here between tilings. `display: none`
        # rather than removal: a root that was destroyed would take its
        # window's state with it.
        attic = kit.el("div", "", style="display: none")

    windows = _windows_menu(ui, workspace)
    projects = _project_menu(ui, workspace)
    workspaces = _workspace_menu(ui, workspace)

    def draw_appbar() -> None:
        appbar.clear()
        with appbar:
            _appbar(workspace, windows, projects, workspaces)

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

            # Every root above has just been through `Element.move()`, and a
            # moved root comes back as a *new DOM node* on the client -- which
            # silently discards anything a window attached to its own node
            # rather than to a NiceGUI element. The sticky transcript is the one
            # that mattered: it was armed once at chat render and died on the
            # first retile, so opening any window once stopped the chat
            # scrolling for the rest of the session. `gradRearm` is idempotent
            # and knows which ids to re-arm, so the shell does not have to know
            # that `grad-transcript` exists.
            #
            # Inside `with tiles`, and that is not stylistic. This runs from a
            # titlebar button, and by here the retile has deleted that button
            # along with the slot it belonged to -- so at this point there is no
            # current slot, no way to reach the client through one, and
            # `kit.run_js` raises into `_guard` rather than sending anything.
            # `tiles` is built once in `build` and outlives every retile.
            kit.run_js("window.gradRearm && window.gradRearm()")

    workspace.bind_chrome(draw_appbar)
    workspace.bind_chrome(draw_status)
    workspace.bind_retile(retile)

    draw_appbar()
    draw_status()
    retile()

    # One poll for the whole workspace; see the note at the top of ui/state.py.
    ui.timer(POLL_SECONDS, workspace.poll)

    _install_quit_guard(ui, workspace)
    _bind_client_events(ui, workspace, windows)


def _install_quit_guard(ui: Any, workspace: Workspace) -> None:
    """The dialog Quit raises when something is still running.

    Built here, during the page, and only *opened* later: a NiceGUI element
    belongs to the client whose slot context created it, and the tray's Quit
    arrives on another thread entirely with no client in scope. So the dialog is
    constructed while there is one, and `desktop.request_quit` reaches it by
    dispatching onto this loop.

    The last client to build wins, which is right for the app this is: one
    native window, one workspace, enforced by `core/instance.py`. In browser
    mode with two tabs open the prompt appears in one of them -- still a prompt,
    still blocking the quit, which is the property that matters.
    """
    dialog = ui.dialog().props("persistent")
    with dialog:
        card = kit.column("grad-pad", gap=9).style(
            "background: var(--grad-paper); border: var(--grad-border); min-width: 440px"
        )

    async def confirm(report: dict[str, Any]) -> bool:
        card.clear()
        with card:
            kit.text("Quit while work is running?", "grad-label")
            kit.text(desktop.busy_sentence(report))
            kit.note(
                "Quitting ends this app and the commands it started. A Lab kernel keeps "
                "running — Lab is a separate server — but anything Grad launched, including "
                "a verify on a fresh kernel, is stopped where it stands."
            )
            with kit.row("", gap=9):
                kit.button("KEEP RUNNING", tone="primary", on_click=lambda: dialog.submit(False))
                kit.button("QUIT ANYWAY", tone="danger", on_click=lambda: dialog.submit(True))
        dialog.open()
        return bool(await dialog)

    desktop.bind_confirm(confirm)


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------
def _appbar(workspace: Workspace, windows: Any, projects: Any, workspaces: Any) -> None:
    header = workspace.header()
    session = header["session"]

    with kit.el("div", "grad-appbar-cell brand"):
        kit.text("∇", "grad-mark")
        kit.text("GRAD", "grad-wordmark")

    # Two scopes, two controls. One folder holds many projects, and switching
    # folders replaces the ledger, the project list, the notebooks and the config
    # under every open window -- while switching project changes what spend is
    # charged to. They were one dialog, six rows apart, styled identically.
    with kit.el("div", "grad-appbar-cell"):
        kit.text("workspace", "dim")
        kit.button(
            f"{header['root_name']} ▾",
            tone="ghost",
            classes="grad-appbar-btn",
            # The basename is on the button because an absolute path does not fit
            # the cell; the whole one is here, because "which folder is this?"
            # has to be answerable without opening anything.
            title=f"{header['root']} — switch folder, credentials, and this installation",
            on_click=workspaces.open,
        )

    with kit.el("div", "grad-appbar-cell"):
        kit.text("project", "dim")
        kit.button(
            f"{header['project']} ▾",
            tone="ghost",
            classes="grad-appbar-btn",
            title="switch the project runs and tokens are charged to",
            on_click=projects.open,
        )
        # Only when there is something to do about it. A permanent "up to date"
        # in the title bar is the same kind of noise as a permanent "tasks 0" in
        # the status bar, and the menu answers the question for anyone who asks
        # it. `update()` reads a cached file, never the network -- see
        # `ui/models.py:update_model`.
        update = workspace.update()
        if update["available"]:
            kit.button(
                f"↑ {update['target']}",
                tone="ghost",
                classes="grad-appbar-btn",
                title=f"{update['target']} is available — open the workspace menu to install it",
                on_click=workspaces.open,
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
        # STOP, not PAUSE. The button used to flip the header caption to "AGENT
        # PAUSED" and nothing else: the turn kept streaming, tools kept running,
        # and the next settle overwrote the state back to idle. A control that
        # says the agent is paused while it is spending is worse than no control,
        # so this one interrupts the turn -- the thing the SDK can actually do --
        # and is disabled when there is nothing to interrupt.
        kit.button(
            "■ STOP",
            tone="ghost",
            classes="grad-appbar-btn",
            title="interrupt the turn in flight",
            on_click=workspace.interrupt_turn,
            disabled=state != "running",
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
        # One control, not three. There used to be an always-visible strip of
        # eleven window names, a `⌘K` palette that listed the same eleven, and a
        # `LAYOUTS ▾` button whose caret promised a menu it did not have. All
        # three were derived from `registry.WINDOWS`; this is the one that is
        # left, and `⌘K` still opens it.
        kit.button(
            "⋯",
            tone="ghost",
            classes="grad-appbar-btn grad-dots",
            title="windows and arrangement (⌘K)",
            on_click=windows.open,
        )


def _used_share(session: dict[str, Any]) -> float:
    ceiling = session.get("ceiling_usd") or 0.0
    if not ceiling:
        return 0.0
    return max(0.0, min(1.0, float(session.get("used_usd", 0.0)) / float(ceiling)))


def _statusbar(workspace: Workspace) -> None:
    status = workspace.status()
    kit.text(status["cwd"], "dim", tag="span")
    kit.text(status["kernel"], "", tag="span")
    kit.text(f"queue {status['queued']} · gpu {status['gpu']}", "", tag="span")
    if status.get("tasks"):
        # Only when something is actually running: a permanent "tasks 0" is
        # noise in a bar that is read at a glance.
        kit.text(f"tasks {status['tasks']}", "count", tag="span")
    if workspace.notice:
        kit.text(workspace.notice, "", tag="span")
    kit.spacer()
    kit.text("drag a title bar to move · drop on another to swap", "dim", tag="span")
    kit.text(f"{len(workspace.layout.windows)} open", "count", tag="span")


# `_Menu` moved to `kit.Menu`: the chat window's session picker is the fourth of
# these, and a dialog helper that only the shell can reach is what kept that
# picker a Quasar `select`.
_Menu = kit.Menu


def _project_menu(ui: Any, workspace: Workspace) -> kit.Menu:
    """The quick switcher: which project is charged, and nothing else.

    It used to be the whole settings surface — folder, recent folders, projects,
    creation, ceilings, credentials and the updater, in one 540px dialog behind a
    button labelled `project`. Four scopes in one menu, and the two most
    different actions in the app (switch project, switch folder) six rows apart
    and styled the same.

    What is left here is the one thing worth a single click from the title bar.
    Everything a project *is* — its ceilings, its memory, creating and closing —
    is the projects window, which this opens.
    """
    return kit.menu(lambda body, menu: _draw_project_menu(workspace, body, menu), width=460)


def _draw_project_menu(workspace: Workspace, body: Any, menu: Any) -> None:
    model = workspace.projects()
    body.clear()

    def act(coro: Any, what: str) -> None:
        """Close first, then run: `reload` redraws the shell underneath, and a
        dialog still open over it would be showing the workspace it just left."""
        menu.close()
        workspace.spawn(coro, what)

    def open_window() -> None:
        menu.close()
        workspace.open("projects")

    with body:
        kit.text("PROJECT", "head ink")
        with kit.el("div", "body"):
            kit.error_strip(model.get("error"))

            rows = [r for r in model.get("rows") or [] if r["status"] != "closed"]
            if not rows:
                kit.text("none in this folder yet", "grad-empty")
            for project in rows:
                with kit.row("grad-row", gap=6):
                    kit.button(
                        "IN USE" if project["current"] else "USE",
                        tone="active" if project["current"] else "neutral",
                        disabled=project["current"],
                        on_click=lambda _=None, pid=project["id"]: act(
                            workspace.use_project(pid), "project switch"
                        ),
                    )
                    kit.text(project["id"], "", style="font-weight: 700; white-space: nowrap")
                    # Squeezable, but never to less than a word a line: the id,
                    # the badge and the spend own their width, the title takes
                    # what is left and ellipsizes.
                    kit.text(
                        project["title"], "grad-caption",
                        style="flex: 1 1 auto; min-width: 0; overflow: hidden;"
                        " text-overflow: ellipsis; white-space: nowrap",
                    )
                    kit.spacer()
                    if project["unbounded"]:
                        kit.chip("UNBOUNDED", "attention")
                    kit.text(project["spend"], "grad-caption")

            with kit.row("", gap=6).style("margin-top: 12px"):
                kit.button(
                    "OPEN THE PROJECTS WINDOW",
                    tone="primary",
                    title="ceilings, memory, creating and closing — for every project, not just this one",
                    on_click=open_window,
                )


def _workspace_menu(ui: Any, workspace: Workspace) -> kit.Menu:
    """Everything that is not a project: which folder, which credentials, which
    Grad.

    The `Confirm` is built here, during the page, for the reason
    `_install_quit_guard` builds its dialog here: a NiceGUI element belongs to
    the client whose slot context made it, and by the time the answer is wanted
    that context is gone.
    """
    confirm = kit.confirm()
    return kit.menu(
        lambda body, menu: _draw_workspace_menu(ui, workspace, body, menu, confirm), width=540
    )


#: What switching folders actually does, said before it happens. Every window in
#: the app re-reads from the new root, `config/grad.toml` may resolve to a
#: different file, and the agent's own tools follow — which is the whole point,
#: and is also nothing like the project switch two controls away.
SWITCH_NOTE = (
    "The ledger, the project list, the notebooks and the config all come from the folder, "
    "so every open window re-reads from the new one and the agent's tools follow it. "
    "Nothing is deleted, and the folder you are leaving is untouched."
)


def _draw_workspace_menu(
    ui: Any, workspace: Workspace, body: Any, menu: Any, confirm: kit.Confirm
) -> None:
    model = workspace.workspaces()
    body.clear()

    def act(coro: Any, what: str) -> None:
        menu.close()
        workspace.spawn(coro, what)

    def open_setup() -> None:
        menu.close()
        workspace.open("setup")

    async def switch(path: str, *, create: bool = False) -> None:
        if not (path or "").strip():
            workspace.say("no folder given")
            return
        if not await confirm.ask(
            "Switch workspace folder?",
            f"This app moves to {path}.",
            confirm="SWITCH",
            note_text=SWITCH_NOTE,
        ):
            return
        await workspace.switch_root(path, create=create)

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
                    on_click=lambda: act(switch(folder.value or "", create=True), "workspace switch"),
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
                            on_click=lambda _=None, p=path: act(switch(p), "workspace switch"),
                        )
                        kit.text(path, "grad-caption")

            # Credentials and the updater moved to the setup window. They are
            # facts about this machine and this installation, they need more
            # room than a 540px dialog, and neither was ever a thing you do
            # *while* switching folders.
            with kit.row("", gap=6).style("margin-top: 16px"):
                kit.button(
                    "SETUP",
                    tone="primary",
                    title="token, models, backends, credentials and this installation",
                    on_click=open_setup,
                )
                kit.text(
                    _setup_line(workspace), "grad-caption", tag="span",
                    style="flex: 1 1 auto; min-width: 0",
                )


def _updates(workspace: Workspace, menu: _Menu) -> None:
    """Which Grad this is, and the one button that changes it.

    Reads a cached answer -- `ui/models.py:update_model` explains why this must
    never be the thing that talks to the network. The section is always drawn,
    including when there is nothing to install: "you are on v0.2.0, checked an
    hour ago" is the answer to a question people actually ask, and a section
    that appeared only when an update existed would leave them with nowhere to
    look for it.
    """
    model = workspace.update()

    def act(coro: Any, what: str) -> None:
        menu.close()
        workspace.spawn(coro, what)

    kit.text("THIS INSTALLATION", "grad-caption").style("margin-top: 16px")
    kit.kv([("version", model["installed"]), ("last checked", model["checked"])])

    if not model["is_checkout"]:
        kit.note(
            "This copy was not installed from a git checkout, so it cannot update itself. "
            "Reinstall from the repository to get updates."
        )
        return

    for warning in model["warnings"]:
        kit.note(f"{warning['message']} — {warning['fix']}")
    for blocker in model["blockers"]:
        kit.error_strip(f"{blocker['message']} — {blocker['fix']}")

    with kit.row("", gap=6).style("margin-top: 8px"):
        if model["available"]:
            kit.chip(f"{model['target']} AVAILABLE", "attention")
            kit.button(
                "UPDATE",
                tone="primary",
                title=(
                    "quit first: this release changes dependencies"
                    if model["needs_reinstall"]
                    else "fast-forward this installation and migrate its state"
                ),
                on_click=lambda: act(workspace.apply_update(), "update"),
            )
        kit.button(
            "CHECK NOW",
            tone="neutral",
            title="ask the remote whether there is a newer release",
            on_click=lambda: act(workspace.check_update(), "update check"),
        )
        kit.spacer()

    if model["available"] and model["needs_reinstall"]:
        kit.text(
            "this release changes dependencies, so it needs Grad closed — quit, then run "
            "`grad update` in a terminal",
            "grad-caption",
        )
    elif model["available"]:
        kit.text(
            f"{model['behind']} commit(s) behind · restart Grad afterwards to load it",
            "grad-caption",
        )
    if model["dirty"]:
        kit.text(
            "the installation has uncommitted edits; runs submitted from it are stamped "
            "as modified and `report check` will say so",
            "grad-caption",
        )


def _setup_line(workspace: Workspace) -> str:
    """One line saying whether this machine is wired up, beside the button that
    wires it. Caught, because the menu has to open on a machine where nothing
    reads -- that is the machine it exists for."""
    try:
        model = workspace.model("setup") or {}
    except Exception:  # noqa: BLE001 - a caption must never break a menu
        return "credentials, models, backends"
    if not (model.get("token") or {}).get("ready"):
        return "not authenticated — nothing can reach a model yet"
    if not model.get("complete"):
        return "no backend configured — runs cannot leave this machine"
    steps = model.get("steps") or []
    return f"{len([s for s in steps if s['ready']])}/{len(steps)} answered"


# The ceiling controls moved to `ui/windows/projects.py`, and the list of the
# three resources with them (`ui/models.py:CEILINGS`). They lived here because
# the menu was the only project surface there was, and they addressed only the
# *selected* project -- the window draws them for every project, which is the
# reason it exists.


def _credentials(ui: Any, workspace: Workspace, menu: _Menu) -> None:
    """Store the credentials the README's install section lists.

    This is the one thing the workspace genuinely could not do: `credential set`
    prompts with `getpass`, which needs a terminal, so a fresh machine needed a
    shell open beside the app to become usable. The value goes down a pipe
    rather than in an argument -- see `Workspace.set_credential`.

    Values are never shown, and there is nothing here that could show one: the
    CLI does not print them and `credentials.status()` returns booleans.
    """
    model = workspace.credentials()
    kit.text("CREDENTIALS", "grad-caption").style("margin-top: 16px")
    kit.error_strip(model.get("error"))

    for row in model["rows"]:
        # Two lines, not one. On one line the fixed-width pieces -- chip, name,
        # a 200px input, two buttons -- left the purpose text a few dozen
        # pixels in a 540px dialog, and flex squeezed it to its min-content
        # width: one word per line, a column taller than the rest of the row.
        with kit.column("grad-row", gap=6):
            # Full width explicitly: `.grad-row`'s `align-items: flex-start`
            # would otherwise shrink each line to its content and the input
            # with it.
            with kit.row("", gap=6).style("width: 100%"):
                kit.chip(row["state"], row["tone"])
                kit.text(row["name"], "grad-mono", tag="span")
                kit.text(
                    row["purpose"], "grad-caption", tag="span",
                    style="flex: 1 1 auto; min-width: 0",
                )
            with kit.row("", gap=6).style("width: 100%"):
                value = (
                    ui.input(placeholder="paste to set")
                    .props("borderless dense type=password")
                    .classes("field")
                    .style("flex: 1 1 auto; padding: 0 8px")
                )

                def store(_=None, name=row["name"], field=value) -> None:
                    pasted, field.value = field.value or "", ""
                    workspace.spawn(workspace.set_credential(name, pasted), "credential set")
                    menu.redraw()

                def forget(_=None, name=row["name"]) -> None:
                    workspace.spawn(workspace.delete_credential(name), "credential delete")
                    menu.redraw()

                kit.button("SET", tone="neutral", on_click=store)
                kit.button("✕", tone="neutral", disabled=not row["stored"], title="forget it",
                           on_click=forget)

    kit.text(
        "stored in Windows Credential Manager, never in the workspace and never in the "
        "agent's environment — they are fetched at the moment of use",
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


#: The arrangements `apply_preset` knows, with the chord the browser sends for
#: each. Listed here rather than in `layout.py` because the caption and the
#: shortcut are chrome; the moves themselves are the layout's.
PRESET_ROWS = (
    ("tile", "TILE", "⌥1", "a column each, up to three"),
    ("stack", "STACK", "⌥2", "one column, everything stacked"),
    ("full", "FULL", "⌥3", "the focused window, the rest at the edge"),
)


def _windows_menu(ui: Any, workspace: Workspace) -> kit.Menu:
    """`⋯` and `⌘K`: which windows are open, and how they are arranged.

    This is the only opener. It replaced a permanent strip of eleven names,
    which cost 34px of vertical space to show a list that is read once a session
    and a state -- open or closed -- that the mark beside each name carries just
    as well.

    It does not close on a toggle. Opening three windows is three clicks, and a
    menu that dismissed itself after each one would be three trips back to the
    same button; `menu.redraw()` re-reads the layout in place so the marks stay
    honest without the dialog going away.
    """
    return kit.menu(lambda body, menu: _draw_windows_menu(workspace, body, menu))


def _draw_windows_menu(workspace: Workspace, body: Any, menu: kit.Menu) -> None:
    open_ids = set(workspace.layout.windows)
    body.clear()

    with body:
        with kit.row("head ink", gap=9):
            kit.text("WINDOWS", "", tag="span")
            kit.spacer()
            kit.text(f"{len(open_ids)} of {len(registry.WINDOWS)} open", "", tag="span")

        with kit.el("div", "body"):
            for window in registry.WINDOWS:
                is_open = window.id in open_ids
                # A filled square for open, an empty one for closed. The opener
                # strip said the same thing by inverting the whole cell, which is
                # louder than a list of eleven can carry.
                row = kit.menu_row(
                    "■" if is_open else "□", window.name, window.hint,
                    open=is_open, title=window.hint,
                )
                row.on(
                    "click",
                    lambda _=None, wid=window.id: (workspace.toggle(wid), menu.redraw()),
                )

            kit.text("ARRANGEMENT", "grad-caption").style("margin-top: 14px")
            for name, caption, chord, hint in PRESET_ROWS:
                row = kit.menu_row(chord, caption, hint, title=hint)
                row.on("click", lambda _=None, p=name: (workspace.preset(p), menu.close()))

            kit.text(
                "drag a title bar to move a window · drop it on another to swap them",
                "grad-caption",
            ).style("margin-top: 12px")


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

        def redraw_titlebar(b=bar, wid=window_id) -> None:
            b.clear()
            with b:
                _titlebar(workspace, wid)

        redraw_titlebar()
        # Refreshed by the poll, not only by a retile. `chat` has no model
        # builder, so nothing else would ever redraw its bar.
        workspace.bind_titlebar(window_id, redraw_titlebar)

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
def _bind_client_events(ui: Any, workspace: Workspace, windows: _Menu) -> None:
    """The gestures `tiling.js` sends back once they have settled."""

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
    # `⌘K` opened nothing at all before: the browser emitted this and the
    # handler was a no-op, because the palette it was meant to open was bound to
    # its own button and never to the chord it advertised.
    ui.on("grad_palette", lambda _: windows.open())
