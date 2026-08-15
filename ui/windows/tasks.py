"""Window 12 — background work.

Two lists, because there are two ways work ends up running with nobody watching
it, and they are not the same thing:

* **The agent's calls.** Every capability in this project is reached by a Bash
  into `tools/`, so a turn that is "thinking" is usually a command running. The
  transcript shows them, but the transcript scrolls, and once it has, the answer
  to "is that still going" was nowhere. They are listed first and they carry no
  STOP: the only thing that stops one is interrupting the turn.
* **The commands the workspace started and did not wait for**: a notebook verify
  on a fresh kernel, a preflight, a wiki rebuild, a PDF build. These are this
  app's own subprocesses, so these are the ones it can stop.

Separate from the queue window on purpose -- see `models.tasks_model` for why a
local subprocess and a GPU job do not belong in one table, which is the same
argument that keeps the two lists here apart.

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
    calls = model.get("agent_running", 0)
    if not running and not finished and not model.get("agent"):
        return "nothing running"
    parts = [f"{running} running", f"{finished} finished"]
    if calls:
        parts.append(f"{calls} agent")
    return " · ".join(parts)


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("tasks") or {}
    out: list[tuple[str, str]] = []
    if model.get("running"):
        out.append((f"{model['running']} RUNNING", "ok"))
    if model.get("agent_running"):
        out.append((f"{model['agent_running']} AGENT", "ok"))
    failed = len([r for r in model.get("rows") or [] if r["state"] == "failed"])
    if failed:
        out.append((f"{failed} FAILED", "broken"))
    return out


def render(workspace: Workspace) -> None:
    model = workspace.model("tasks") or {}
    kit.error_strip(model.get("error"))
    rows = model.get("rows") or []
    calls = model.get("agent") or []
    if not rows and not calls:
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
        kit.text(f"{len(rows)} started here · {len(calls)} agent", "grad-caption", tag="span")

    if calls:
        # First, and above the workspace's own: this is the section that answers
        # "is it still going", and the agent's calls are the ones with nowhere
        # else to be read once the transcript has scrolled on.
        _section("THE AGENT'S CALLS", "what the agent is running, newest first")
        for call in calls:
            _call(call)

    if rows:
        _section("STARTED HERE", "commands this workspace ran and did not wait for")
        for row in rows:
            _task(workspace, row)


def _section(title: str, hint: str) -> None:
    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-hairline)"):
        kit.label(title)
        kit.text(hint, "grad-caption", tag="span")


def _call(row: dict[str, Any]) -> None:
    """One of the agent's tool calls.

    No STOP button, and the absence is the point. A task here is a subprocess
    this app spawned and holds a handle to; a call is one the agent made inside
    the SDK, and the only thing that stops it is interrupting the turn -- which
    is what the chat window's own control does. A button that looked like the
    one above it and did something else would be worse than no button.
    """
    with kit.el("div", "grad-card tool"):
        with kit.row("head ink", gap=9):
            kit.text("CALL", "", tag="span")
            kit.text(row["name"], "", tag="span")
            if row["subject"]:
                kit.text(row["subject"], "subject", tag="span")
            kit.spacer()
            if row["elapsed"]:
                kit.text(row["elapsed"], "", tag="span")
            kit.chip(row["state"].upper(), row["tone"], dot=row["running"])

        if row["state"] == "unfinished":
            with kit.el("div", "body"):
                kit.note(
                    "the turn ended before this call reported — whatever it started "
                    "is not something the workspace can see or stop"
                )
        elif row["tail"]:
            with kit.el("div", "body"):
                with kit.el("div", "out"):
                    kit.text("OUTPUT", "grad-label")
                    if row["lines"] > len(row["tail"].splitlines()):
                        kit.text(
                            f"the last {len(row['tail'].splitlines())} of {row['lines']:,} lines — "
                            "the call's own card in the transcript has the rest",
                            "grad-caption",
                        )
                    kit.pre(row["tail"], "broken" if row["state"] == "failed" else "neutral")


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
