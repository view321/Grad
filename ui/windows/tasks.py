"""Window 12 — background tasks.

The commands the workspace started and did not wait for: a notebook verify on a
fresh kernel, a preflight, a wiki rebuild, a PDF build. Separate from the queue
window on purpose -- see `models.tasks_model` for why a local subprocess and a
GPU job do not belong in one table.

Two things here are the window rather than decoration.

**The command is shown in full.** Every button in the workspace runs the same
command the agent would, and printing it is what makes that claim checkable --
and reproducible from a terminal when the workspace is not the right place to
run it.

**STOP says which stop it will use.** A task that carries the tool's own halt
verb is stopped by *asking*; one that does not is signalled. Those are different
enough to be worth different tooltips: `nb verify` spawns its kernel detached,
so signalling the CLI would leave the kernel holding VRAM.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.state import Workspace
from ui.tasks import cancel, clear_finished

#: How much of a task's tail is on screen at once. The rest is scrolled to.
TAIL_STYLE = "max-height: 220px; overflow: auto"


def subtitle(workspace: Any) -> str:
    model = workspace.model("tasks") or {}
    running = model.get("running", 0)
    finished = model.get("finished", 0)
    if not running and not finished:
        return "nothing running"
    return f"{running} running · {finished} finished"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("tasks") or {}
    out: list[tuple[str, str]] = []
    if model.get("running"):
        out.append((f"{model['running']} RUNNING", "ok"))
    failed = len([r for r in model.get("rows") or [] if r["state"] == "failed"])
    if failed:
        out.append((f"{failed} FAILED", "broken"))
    return out


def render(workspace: Workspace) -> None:
    model = workspace.model("tasks") or {}
    kit.error_strip(model.get("error"))
    rows = model.get("rows") or []
    if not rows:
        kit.empty("Nothing has been started here.", model.get("empty_fix"))
        return

    def clear() -> None:
        gone = clear_finished()
        workspace.say(f"cleared {gone} finished task{'' if gone == 1 else 's'}")
        workspace.invalidate("tasks")
        workspace.tick()

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
        kit.button(
            "CLEAR FINISHED",
            tone="neutral",
            disabled=not model.get("finished"),
            title="running tasks are left alone",
            on_click=clear,
        )
        kit.spacer()
        kit.text(f"{len(rows)} shown", "grad-caption", tag="span")

    for row in rows:
        _task(workspace, row)


def _task(workspace: Workspace, row: dict[str, Any]) -> None:
    async def stop() -> None:
        workspace.say(await cancel(row["id"]))
        workspace.invalidate("tasks")
        workspace.tick()

    with kit.el("div", "grad-card tool"):
        with kit.row("head ink", gap=9):
            kit.text("TASK", "", tag="span")
            kit.text(row["label"], "subject", tag="span")
            kit.spacer()
            kit.text(row["elapsed"], "", tag="span")
            kit.chip(row["state"].upper(), row["tone"], dot=row["running"])

        with kit.el("div", "body"):
            kit.pre(row["command"])
            if row["stoppable"]:
                with kit.row("", gap=6).style("margin-top: 8px"):
                    kit.button(
                        "■ STOP",
                        tone="neutral",
                        title=(
                            f"asks the tool to stop first: {row['halt']}"
                            if row["halt"]
                            else "signals the process; anything it started detached survives"
                        ),
                        on_click=stop,
                    )
                    kit.text(row["message"], "grad-caption")
            else:
                kit.text(row["message"], "grad-caption").style("margin-top: 8px")

            tail = row.get("tail") or []
            if tail:
                with kit.el("div", "out"):
                    kit.text("OUTPUT", "grad-label")
                    if row.get("dropped"):
                        # A tail that silently forgets reads as complete output.
                        kit.text(
                            f"… {row['dropped']:,} earlier lines scrolled out of the tail",
                            "grad-caption",
                        )
                    body = kit.el("div", "", style=TAIL_STYLE)
                    with body:
                        for tag, line in tail:
                            kit.pre(line, "broken" if tag == "err" else "neutral")
