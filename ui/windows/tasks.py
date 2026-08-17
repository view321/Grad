"""Window 12 — background work.

Three lists, because there are three ways work ends up running with nobody
watching it, and they are not the same thing:

* **The agent's calls.** Every capability in this project is reached by a Bash
  into `tools/`, so a turn that is "thinking" is usually a command running. The
  transcript shows them, but the transcript scrolls, and once it has, the answer
  to "is that still going" was nowhere. They are listed first and they carry no
  STOP: the only thing that stops one is interrupting the turn.
* **The agent's background tasks** — anything it started with `tools.task`, which
  is how it runs a preflight or a collect without blocking its own turn. These
  live in a *file* rather than in this process, which is the whole reason they
  can appear here at all: they were started by the agent, or by a second
  terminal, and until the registry existed the workspace could not see them.
  They stop through `tools.task stop`, which asks the tool's own halt verb first.
* **The commands the workspace started and did not wait for**: a notebook verify
  on a fresh kernel, a preflight, a wiki rebuild, a PDF build. These are this
  app's own subprocesses, held in memory, so these are the ones it can signal
  directly.

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
from ui.tasks import cancel, clear_finished, envelope_message, run_tool

#: How much of a task's tail is on screen at once. The rest is scrolled to.
TAIL_STYLE = "max-height: 220px; overflow: auto"


def subtitle(workspace: Any) -> str:
    model = workspace.model("tasks") or {}
    running = model.get("running", 0) + model.get("background_running", 0)
    finished = model.get("finished", 0)
    calls = model.get("agent_running", 0)
    if not running and not finished and not model.get("agent") and not model.get("background"):
        return "nothing running"
    parts = [f"{running} running", f"{finished} finished"]
    if calls:
        parts.append(f"{calls} agent")
    return " · ".join(parts)


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("tasks") or {}
    out: list[tuple[str, str]] = []
    running = model.get("running", 0) + model.get("background_running", 0)
    if running:
        out.append((f"{running} RUNNING", "ok"))
    if model.get("agent_running"):
        out.append((f"{model['agent_running']} AGENT", "ok"))
    failed = len(
        [r for r in (model.get("rows") or []) + (model.get("background") or [])
         if r["state"] == "failed"]
    )
    if failed:
        out.append((f"{failed} FAILED", "broken"))
    return out


def render(workspace: Workspace) -> None:
    model = workspace.model("tasks") or {}
    kit.error_strip(model.get("error"))
    rows = model.get("rows") or []
    calls = model.get("agent") or []
    background = model.get("background") or []
    if not rows and not calls and not background:
        kit.empty("Nothing has been started here.", model.get("empty_fix"))
        return

    async def clear() -> None:
        gone = clear_finished()
        # Both registries, because the section header says "clear finished" and a
        # button that cleared one of two lists would be a button that looks
        # broken. The agent's tasks are cleared through its own CLI, which is the
        # same command it would run -- §10, and it is also the only thing that
        # knows to delete the logs.
        payload = await run_tool("tools.task", "clear", "--json")
        forgotten = ((payload.get("data") or {}).get("forgotten")) or 0
        workspace.say(f"cleared {gone + forgotten} finished task(s)")
        workspace.invalidate("tasks")
        workspace.tick()

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
        kit.button(
            "CLEAR FINISHED",
            tone="neutral",
            disabled=not model.get("finished")
            and not any(not r["running"] for r in background),
            title="running tasks are left alone",
            on_click=clear,
        )
        kit.spacer()
        kit.text(
            f"{len(rows)} started here · {len(background)} backgrounded · {len(calls)} agent",
            "grad-caption",
            tag="span",
        )

    if calls:
        # First, and above the rest: this is the section that answers "is it
        # still going", and the agent's calls are the ones with nowhere else to
        # be read once the transcript has scrolled on.
        _section("THE AGENT'S CALLS", "what the agent is running, newest first")
        for call in calls:
            _call(call)

    if background:
        _section(
            "BACKGROUNDED BY THE AGENT",
            "started with tools.task; still running after the turn that started them",
        )
        for row in background:
            _background(workspace, row)

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
                # Shortened for the head, whole in the tooltip: a pane is 320px at
                # its floor and the end of a path is the part that identifies it.
                subject = kit.text(kit.shorten_path(row["subject"]), "subject", tag="span")
                subject.props(f'title="{kit.attr(row["subject"])}"')
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
                    kit.sublabel("output")
                    if row["lines"] > len(row["tail"].splitlines()):
                        kit.text(
                            f"the last {len(row['tail'].splitlines())} of {row['lines']:,} lines — "
                            "the call's own card in the transcript has the rest",
                            "grad-caption",
                        )
                    kit.pre(row["tail"], "broken" if row["state"] == "failed" else "neutral")


def _background(workspace: Workspace, row: dict[str, Any]) -> None:
    """One of the agent's background tasks.

    Stopped through `tools.task stop` rather than by signalling anything from
    here, and that is not merely tidiness: this app did not spawn the process and
    holds no handle to it. The CLI does the halt-verb-then-signal dance, and
    running the same command the agent would is the rule the whole workspace is
    built on.
    """

    async def stop() -> None:
        payload = await run_tool("tools.task", "stop", row["id"], "--json")
        workspace.say(envelope_message(payload))
        workspace.invalidate("tasks")
        workspace.tick()

    with kit.el("div", "grad-card tool"):
        with kit.row("head ink", gap=9):
            kit.text("BACKGROUND", "", tag="span")
            kit.text(row["label"], "subject", tag="span")
            kit.spacer()
            if row["exit_code"] is not None:
                kit.text(f"exit {row['exit_code']}", "", tag="span")
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
                            else "python -m tools.task stop " + row["id"]
                        ),
                        on_click=stop,
                    )
                    kit.text(row["id"], "grad-caption")
            elif row["state"] == "lost":
                kit.note(
                    "started, never recorded an exit, and its supervisor is gone — "
                    "usually a reboot. There is no exit code to report."
                ).style("margin-top: 8px")
            if row["error"]:
                kit.text(row["error"], "grad-caption").style("margin-top: 8px")
            for note in row["notes"]:
                kit.text(note, "grad-caption")
            if row["log"]:
                kit.text(f"log: {row['log']}", "grad-caption")


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
                    kit.sublabel("output")
                    if row.get("dropped"):
                        # A tail that silently forgets reads as complete output.
                        kit.text(
                            f"… {row['dropped']:,} earlier lines scrolled out of the tail",
                            "grad-caption",
                        )
                    body = kit.el("div", "tail", style=TAIL_STYLE)
                    with body:
                        for tag, line in tail:
                            kit.pre(line, "broken" if tag == "err" else "neutral")
