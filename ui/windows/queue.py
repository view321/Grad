"""Window 7 — the run queue and GPU jobs.

Campaign candidates share this table with runs, because they spend the same GPU
dollars against the same ceiling. A queue that showed only `runs.jsonl` would
render a campaign as idle while it burned the budget -- which is the exact
failure the campaign ledger was split out to avoid making invisible.

Cost is `cost_for_ceiling()`: actual once collected, the estimate while in
flight. A job that has not been collected yet is not free.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.state import envelope_message, run_tool

COLUMNS = ("job", "what", "device", "progress", "eta", "cost", "state")


def subtitle(workspace: Any) -> str:
    model = workspace.model("queue") or {}
    return f"{model.get('running', 0)} running · {model.get('failed', 0)} failed"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("queue") or {}
    out: list[tuple[str, str]] = []
    if model.get("running"):
        out.append((f"{model['running']} RUNNING", "ok"))
    if model.get("failed"):
        out.append((f"{model['failed']} FAILED", "broken"))
    return out


def render(workspace: Any) -> None:
    model = workspace.model("queue") or {}
    kit.error_strip(model.get("error"))
    rows = model.get("rows") or []
    if not rows:
        kit.empty("Nothing has been submitted.", model.get("empty_fix"))
        return

    async def refresh() -> None:
        workspace.say("polling job status …")
        payload = await run_tool("tools.jobs", "status", "--json", timeout=120)
        workspace.say(envelope_message(payload))
        workspace.invalidate("queue")
        workspace.tick()

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
        kit.button("↻ POLL", tone="neutral", on_click=refresh)
        kit.spacer()
        kit.text(f"{len(rows)} shown", "grad-caption", tag="span")

    with kit.el("table", "grad-table"):
        with kit.el("thead"):
            with kit.el("tr"):
                for column in COLUMNS:
                    kit.text(column, "", tag="th")
        with kit.el("tbody"):
            for row in rows:
                _row(row)


def _row(row: dict[str, Any]) -> None:
    with kit.el("tr", "running" if row["tone"] == "running" else ""):
        kit.text(row["job"], "", tag="td")
        kit.text(row["what"], "", tag="td")
        kit.text(row["device"], "", tag="td")
        with kit.el("td"):
            kit.progress(row["progress"], row["tone"])
        kit.text(row["eta"], "", tag="td")
        kit.text(row["cost"], "", tag="td")
        with kit.el("td"):
            kit.chip(row["state"], row["accent"])
