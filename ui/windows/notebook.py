"""Window 2 — the notebook: Grad's chrome above a real JupyterLab.

Notebooks render, they do not rebuild. JupyterLab already exists and is better
at editing, and building a notebook editor is the single easiest way to burn a
month on this project. So the interior is Lab in an iframe, themed by
`config/jupyter/custom/custom.css` so the seam is invisible, and everything
Grad owns -- toolbar, verify banner, footer -- stays here in NiceGUI above it.

Two things in this window are deliberate and look like omissions:

**`▶ RUN` and `▶▶ ALL` are disabled.** Lab runs on its own port, so it is a
different origin and the host page cannot reach into it; there is no honest
wiring from this toolbar to that kernel. Faking it by driving Grad's *own*
kernel through `tools/nb.py` would be worse than disabling it, because it would
silently execute the file in a second kernel -- which is precisely the
"works in the kernel that grew it" failure `nb verify` exists to catch. The
buttons keep their place in the group and say why they are dark.

**The verify banner is the only source of citable state.** It goes yellow on any
edit, including an edit made inside Lab that the host never saw, because
staleness is decided by comparing the file's mtime against the verification's.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

from ui import desktop, kit, models
from ui.tasks import CANCELLED, envelope_message, run_tool, start, task_message

ANCHOR_ID = "grad-anchor-notebook"


def _record(workspace: Any, name: str, task: Any) -> None:
    """Fold a finished verify task into the notebook's citable state.

    A verify that was *stopped* proves nothing, and writing it as a failure
    would be a claim about the notebook rather than about the interruption. The
    banner already has the right state for "we do not know" -- unverified since
    the last edit -- so a cancelled task is kept out of the store entirely.
    """
    if task.state == CANCELLED:
        workspace.say(f"{name}: verification stopped — it is still unverified")
    else:
        payload = task.envelope or {"ok": False, "error": {"message": task_message(task)}}
        models.record_verify(name, payload)
        workspace.say(f"{name}: {_verify_line(payload)}")
    workspace.invalidate("notebook")
    workspace.tick()


def _current(workspace: Any) -> str | None:
    model = workspace.model("notebook") or {}
    names = [n["name"] for n in model.get("notebooks", [])]
    chosen = workspace.selection.get("notebook.name")
    if chosen in names:
        return chosen
    return names[0] if names else None


def subtitle(workspace: Any) -> str:
    model = workspace.model("notebook") or {}
    name = _current(workspace)
    if model.get("origin_mismatch"):
        where = " wrong origin"
    elif model.get("lab_running"):
        where = f":{model.get('lab_port')}"
    else:
        where = " stopped"
    return f"{name or 'no notebook'} · lab{where}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("notebook") or {}
    name = _current(workspace)
    for entry in model.get("notebooks", []):
        if entry["name"] == name:
            verify = entry["verify"]
            return [(verify["chip"], verify["accent"])]
    return []


def render(workspace: Any) -> None:
    from nicegui import ui

    model = workspace.model("notebook") or {}
    kit.error_strip(model.get("error"))
    notebooks = model.get("notebooks", [])
    if not notebooks:
        kit.run_js(f"window.gradDropFrame && window.gradDropFrame('{ANCHOR_ID}')")
        kit.empty(
            "No notebooks yet.",
            "python -m tools.nb exec notebooks/<name>.ipynb --code 'print(1)' --json",
        )
        return

    name = _current(workspace)
    entry = next((n for n in notebooks if n["name"] == name), notebooks[0])

    with kit.column("", gap=0).classes("h-full"):
        _toolbar(ui, workspace, model, entry)
        _banner(entry["verify"])
        if entry["verify"]["state"] == "failed":
            _failure(entry["verify"])
        _lab(ui, workspace, model, entry)
        _footer(model, entry)


def _toolbar(ui: Any, workspace: Any, model: dict[str, Any], entry: dict[str, Any]) -> None:
    name = entry["name"]

    def verify() -> None:
        """In the background, and with a way to stop it that the kernel survives.

        This used to be awaited under `run_tool`'s wall clock, which killed it
        at 900 seconds -- below `verify_timeout_s`, which allows 1800 *per cell*.
        A notebook slow enough to need the allowance was the one the UI refused
        to finish.

        `nb stop` rather than a signal, for the reason `_shutdown` exists: the
        verify kernel is spawned detached so it outlives the CLI, so terminating
        the CLI would leave it holding the VRAM the verify was meant to free.
        """
        session = f"verify-{Path(name).stem}"
        start(
            f"verify {name}",
            "tools.nb", "verify", f"notebooks/{name}", "--json",
            halt=("tools.nb", "stop", "--kernel", session, "--json"),
            on_done=lambda task: _record(workspace, name, task),
        )
        workspace.say(f"verifying {name} on a fresh kernel — see the tasks window")
        workspace.invalidate("tasks")
        workspace.tick()

    async def kernel(command: str) -> None:
        payload = await run_tool("tools.nb", command, "--json")
        workspace.say(f"kernel {command}: {envelope_message(payload)}")

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border); flex: 0 0 auto"):
        with kit.el("div", "grad-group"):
            kit.button("▶ RUN", tone="neutral", disabled=True,
                       title="Lab owns this notebook's kernel — run cells in the embedded Lab")
            kit.button("▶▶ ALL", tone="neutral", disabled=True,
                       title="use VERIFY: a run-all that does not start from a fresh kernel proves nothing")
            kit.button("■ STOP", tone="neutral", on_click=lambda: kernel("stop"),
                       title="stop Grad's own kernel (tools.nb)")
            kit.button("↻ RESTART", tone="neutral", on_click=lambda: kernel("restart"),
                       title="restart Grad's own kernel (tools.nb)")
        kit.button("✓ VERIFY — FRESH KERNEL", tone="primary", on_click=verify)
        kit.spacer()
        if len(model.get("notebooks", [])) > 1:
            ui.select(
                [n["name"] for n in model["notebooks"]],
                value=name,
                on_change=lambda e: workspace.select("notebook.name", e.value, window="notebook"),
            ).props("dense borderless").style("min-width: 180px")
        kit.text(f"ruler {model.get('ruler', 88)}", "grad-chip dashed")
        kit.button(
            "↗ OPEN IN LAB",
            tone="neutral",
            on_click=lambda: ui.navigate.to(models.lab_url(model, name), new_tab=True),
            disabled=not model.get("lab_running"),
        )


def _banner(verify: dict[str, Any]) -> None:
    """`NB VERIFY`, the result sentence, and the citable chip."""
    tone = {"clean": "ok", "failed": "broken", "stale": "attention", "unverified": "attention"}[
        verify["state"]
    ]
    background = {
        "ok": "background: var(--grad-verified); color: var(--grad-verified-ink);",
        "broken": "background: var(--grad-broken); color: #fff;",
        "attention": "background: var(--grad-attention); color: var(--grad-ink);",
    }[tone]
    with kit.row("", gap=11).style(
        f"{background} padding: 8px 14px; border-bottom: var(--grad-border); flex: 0 0 auto"
    ):
        kit.text("NB VERIFY", "grad-label", tag="span")
        kit.text(verify["sentence"], "", tag="span")
        kit.spacer()
        chip = kit.text(verify["chip"], "grad-chip", tag="span")
        if verify["state"] == "clean":
            chip.style("background: var(--grad-verified-ink); color: var(--grad-verified); border-color: var(--grad-verified-ink);")


def _failure(verify: dict[str, Any]) -> None:
    """The failing cell, its traceback, and the command that repairs it."""
    with kit.pad():
        with kit.el("div", "grad-card"):
            index = verify.get("cell_index")
            kit.text(
                f"FAILED AT CELL {index}" if index is not None else "VERIFICATION FAILED",
                "head broken",
            )
            with kit.el("div", "body"):
                if verify.get("traceback"):
                    kit.pre(str(verify["traceback"])[-4000:], "broken")
                if verify.get("fix"):
                    with kit.el("div").style(
                        "border: var(--grad-border); background: var(--grad-attention); padding: 9px; margin-top: 9px"
                    ):
                        kit.text("FIX", "grad-label")
                        kit.pre(verify["fix"])


#: What the Lab server is told to allow when the page cannot be asked. The port
#: is `ui/app.py`'s, recorded when it ran, so this is right for a default launch
#: and is only a guess when the browser has gone away mid-click.
def _fallback_origin() -> str:
    from ui import app as app_mod  # noqa: PLC0415 - avoids a cycle at import time

    return f"http://127.0.0.1:{getattr(app_mod, 'PORT', 8080)}"


async def _origin(ui: Any) -> str:
    """The origin this page is actually on, asked of the page itself.

    Lab's framing headers are scoped to one origin and fixed at launch, and
    getting it wrong does not fail loudly: the browser reports a blocked frame
    as *"127.0.0.1 refused to connect"*, which reads as a dead port and sends
    you looking for a server that is running perfectly well.

    There are two ways to be wrong that guessing cannot cover, and the page
    knows the answer to both. `--port` moves the app and `agent.py --ui`'s own
    help already warned that Lab would need telling; and `http://localhost:8080`
    and `http://127.0.0.1:8080` are *different origins* to a browser, so which
    one the window happens to be opened on decides whether the frame loads.
    """
    try:
        origin = await ui.run_javascript("window.location.origin", timeout=3.0)
    except Exception:  # noqa: BLE001 - a page that cannot answer is not an error
        origin = ""
    return str(origin or "").strip() or _fallback_origin()


def _lab(ui: Any, workspace: Any, model: dict[str, Any], entry: dict[str, Any]) -> None:
    if not model.get("lab_running"):
        kit.run_js(f"window.gradDropFrame && window.gradDropFrame('{ANCHOR_ID}')")

        # Awaited rather than backgrounded, and named apart from `tasks.start`:
        # the window needs the port back before it can draw the iframe, so there
        # is nothing useful to do while this runs.
        async def start_lab() -> None:
            workspace.say("starting JupyterLab …")
            payload = await run_tool(
                "tools.lab", "start", "--ui-origin", await _origin(ui), "--json", timeout=90
            )
            workspace.say(envelope_message(payload))
            workspace.invalidate("notebook")
            workspace.tick()

        with kit.pad():
            kit.text("JupyterLab is not running.", "grad-empty")
            kit.button("▶ START LAB", tone="primary", on_click=start_lab)
            kit.note(
                "Anything edited in Lab must pass VERIFY before it is cited in notes/ or "
                "referenced from a ledger entry — Lab and tools/nb.py are two kernel owners "
                "over one notebook."
            )
        return

    if model.get("origin_mismatch"):
        kit.run_js(f"window.gradDropFrame && window.gradDropFrame('{ANCHOR_ID}')")
        _origin_mismatch(ui, workspace, model)
        return

    if desktop.native_available():
        kit.run_js(f"window.gradDropFrame && window.gradDropFrame('{ANCHOR_ID}')")
        _own_window(ui, workspace, model, entry)
        return

    anchor = kit.el("div", "grad-iframe-anchor")
    anchor.props(f'id="{ANCHOR_ID}"')
    url = models.lab_url(model, entry["name"])
    # Not sandboxed, and that is a considered difference from the notebook-output
    # iframe: Lab is a server we started ourselves, on its own port, behind a
    # token we minted, and it cannot function sandboxed.
    kit.run_js(f"window.gradRegisterFrame && window.gradRegisterFrame('{ANCHOR_ID}', {url!r}, false)")


async def _restart_lab(ui: Any, workspace: Any) -> None:
    """Restart Lab bound to the origin this page is actually on.

    `--force` because the whole point is to replace a server that is already
    running: framing headers are decided at launch, so there is no way to change
    the allowed origin of a live one.
    """
    workspace.say("restarting JupyterLab for this window …")
    payload = await run_tool(
        "tools.lab", "start", "--ui-origin", await _origin(ui), "--force", "--json", timeout=90
    )
    workspace.say(envelope_message(payload))
    workspace.invalidate("notebook")
    workspace.tick()


def _origin_mismatch(ui: Any, workspace: Any, model: dict[str, Any]) -> None:
    """The banner for a Lab that is running, healthy, and cannot be embedded.

    Worth a window of its own rather than a line in the log, because the failure
    it replaces is genuinely misleading: the browser renders a blocked frame as
    "refused to connect", so the visible evidence says the server is down while
    the server is fine. Naming both origins is the whole fix -- one glance and
    the cause is obvious.
    """
    with kit.pad():
        kit.text("JupyterLab is running, but it was started for a different window.", "grad-empty")
        with kit.el("div").style(
            "border: var(--grad-border); background: var(--grad-attention); padding: 9px; margin: 9px 0"
        ):
            kit.text("LAB ALLOWS", "grad-label")
            kit.pre(str(model.get("lab_origin") or "unknown"))
            kit.text("THIS WINDOW IS", "grad-label")
            kit.pre(desktop.origin(models.app_port()))
        kit.button("↻ RESTART LAB FOR THIS WINDOW", tone="primary",
                   on_click=lambda: _restart_lab(ui, workspace))
        kit.note(
            "Lab fixes the origins it will be framed by when it starts, so this cannot be "
            "changed on a running server. Restarting keeps your files; unsaved cells in the "
            "old session are lost, so save in Lab first if it is still open elsewhere."
        )


def _own_window(ui: Any, workspace: Any, model: dict[str, Any], entry: dict[str, Any]) -> None:
    """Lab as a second desktop window instead of an iframe in this one.

    See `ui/desktop.py:open_lab_window` for why: a separate webview is a
    separate renderer process, so Lab's rendering and the workspace's stop
    competing for one main thread. The verify banner and the toolbar stay here,
    which is what matters -- they are what decides whether a notebook is
    citable, and that must not be inside the window Lab owns.
    """
    url = models.lab_url(model, entry["name"])

    def open_lab() -> None:
        if desktop.open_lab_window(url):
            workspace.say(f"{entry['name']} — opened in the Lab window")
        else:
            workspace.say("could not open a Lab window; opening in a browser tab instead")
            ui.navigate.to(url, new_tab=True)

    with kit.pad():
        kit.text("JupyterLab runs in its own window.", "grad-empty")
        kit.button("⧉ OPEN LAB WINDOW", tone="primary", on_click=open_lab)
        kit.note(
            "A separate window is a separate renderer, so a busy notebook cannot slow the "
            "workspace down. Everything that decides whether this notebook is citable — the "
            "verify banner above, and VERIFY — stays here."
        )


def _footer(model: dict[str, Any], entry: dict[str, Any]) -> None:
    with kit.row("", gap=14).style(
        "background: var(--grad-ink); color: var(--grad-paper); padding: 6px 14px; "
        "font-family: var(--grad-font-mono); font-size: 11px; flex: 0 0 auto"
    ):
        kit.text("CMD", "", tag="span")
        kit.text(entry["name"], "", tag="span")
        kit.spacer()
        kit.text("kernel owner: lab", "", tag="span")
        kit.chip(entry["verify"]["state"].upper(), entry["verify"]["accent"])


def _verify_line(payload: dict[str, Any]) -> str:
    if payload.get("ok"):
        data = payload.get("data") or {}
        return f"clean — {data.get('cells_executed', '?')} cells on a fresh kernel"
    error = payload.get("error") or {}
    return str(error.get("message") or "verification failed")
