"""Window 5 — preflight and gates.

When a submission is blocked, the failing check, its output and its `error.fix`
command are one click away. A gate is only tolerable if it explains itself.

`▶ PROCEED` is disabled while any ✕ row exists, and the remedy button runs the
failing check's own fix and re-runs the checklist in place. Neither of those is
decoration: a checklist that lets you proceed past a red row is a checklist that
teaches you to ignore it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ui import kit
from ui.tasks import start, task_message


def subtitle(workspace: Any) -> str:
    model = workspace.model("preflight") or {}
    current = model.get("current")
    if not current:
        return "nothing submitted yet"
    return f"{current.get('hash', '?')} · {current.get('verified_at', '?')}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("preflight") or {}
    blocking = model.get("blocking", 0)
    if blocking:
        return [(f"{blocking} BLOCKING", "broken")]
    return [("CLEAR", "ok")] if model.get("can_proceed") else []


def render(workspace: Any) -> None:
    model = workspace.model("preflight") or {}
    # Before the empty state, not after: "no records yet" and "the records are
    # there and cannot be read" look identical otherwise, and only one of them
    # means you are clear to submit.
    kit.error_strip(model.get("error"))
    current = model.get("current")
    if not current:
        kit.empty("No preflight records yet.", model.get("empty_fix"))
        return

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
        kit.text(str(current.get("spec") or ""), "grad-mono", tag="span")
        kit.spacer()
        kit.text(f"verified {current.get('verified_at', '?')}", "grad-caption", tag="span")

    for row in current.get("rows") or []:
        _row(workspace, row)

    for warning in current.get("warnings") or []:
        # The known gaps in the submission hash: dynamic imports and files
        # loaded at runtime. Shown, not swallowed.
        with kit.pad():
            kit.note(f"⚠ {warning}")

    _footer(workspace, model)


def _row(workspace: Any, row: dict[str, Any]) -> None:
    background = "attention-bg" if row["state"] == "attention" else ""
    with kit.column(f"grad-row {background}".strip(), gap=0):
        with kit.row("", gap=11):
            kit.status_square(row["state"], row["glyph"])
            kit.text(row["name"], "grad-mono", tag="span")
            kit.text(row["sentence"], "", tag="span")
            kit.spacer()
            kit.text(row["detail"], "grad-caption", tag="span")
        if row["state"] == "broken":
            with kit.column("", gap=6).style("padding-left: 29px; margin-top: 8px"):
                if row.get("output"):
                    kit.pre(str(row["output"])[-3000:], "broken")
                if row.get("fix"):
                    kit.pre(row["fix"])


def _footer(workspace: Any, model: dict[str, Any]) -> None:
    def rerun() -> None:
        """In the background: `tests` and `dry_run` are 900 seconds each, so the
        gate that has to pass before money is spent could outlast any wall clock
        the UI put on it."""
        current = model.get("current") or {}
        spec = current.get("spec")
        if not spec:
            workspace.say("this record does not name the spec it came from")
            return
        start(
            f"preflight {Path(str(spec)).name}",
            "tools.preflight", "run", "--spec", str(spec), "--json",
            on_done=lambda task: _settled(workspace, task),
        )
        workspace.say("re-running preflight — see the tasks window")
        workspace.invalidate("tasks")
        workspace.tick()

    def _settled(workspace: Any, task: Any) -> None:
        workspace.say(task_message(task))
        workspace.invalidate("preflight")
        workspace.tick()

    with kit.row("grad-pad", gap=9).style("border-top: var(--grad-border)"):
        kit.button(
            "▶ PROCEED",
            tone="ok" if model.get("can_proceed") else "neutral",
            disabled=not model.get("can_proceed"),
            title="disabled while any check is failing",
        )
        if model.get("remedy"):
            kit.button("↻ FIX AND RE-CHECK", tone="primary", on_click=rerun,
                       title=str(model["remedy"]))
        kit.spacer()
        blocking = model.get("blocking", 0)
        kit.chip(f"{blocking} BLOCKING" if blocking else "NOTHING BLOCKING",
                 "broken" if blocking else "ok")
