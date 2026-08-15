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

from ui import kit, models
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
    where = f":{model.get('lab_port')}" if model.get("lab_running") else " stopped"
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


def _lab(ui: Any, workspace: Any, model: dict[str, Any], entry: dict[str, Any]) -> None:
    if not model.get("lab_running"):
        kit.run_js(f"window.gradDropFrame && window.gradDropFrame('{ANCHOR_ID}')")

        # Awaited rather than backgrounded, and named apart from `tasks.start`:
        # the window needs the port back before it can draw the iframe, so there
        # is nothing useful to do while this runs.
        async def start_lab() -> None:
            workspace.say("starting JupyterLab …")
            payload = await run_tool("tools.lab", "start", "--json", timeout=90)
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

    anchor = kit.el("div", "grad-iframe-anchor")
    anchor.props(f'id="{ANCHOR_ID}"')
    url = models.lab_url(model, entry["name"])
    # Not sandboxed, and that is a considered difference from the notebook-output
    # iframe: Lab is a server we started ourselves, on its own port, behind a
    # token we minted, and it cannot function sandboxed.
    kit.run_js(f"window.gradRegisterFrame && window.gradRegisterFrame('{ANCHOR_ID}', {url!r}, false)")


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
