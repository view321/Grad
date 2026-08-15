"""Window 6 — the retrieval funnel.

400 → 50 → 15, with stage 3's one-line reason per surviving candidate. This is
the debugging surface for retrieval, and it is what makes the stage-0/3
evaluation interpretable rather than a pair of numbers.

The dropped chunks below the dashed rule are the point of the window. A funnel
that shows only survivors cannot answer the question you actually have when
retrieval goes wrong, which is "why is the obviously relevant paper not in
here" -- and stage 3's per-candidate reason is not decoration either: it is the
provenance that populates a ledger entry's basis.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.tasks import start, task_message


def _search(ui: Any, workspace: Any) -> None:
    """Run the funnel from the window that exists to debug it.

    It was read-only, which is a strange shape for a debugging surface: the
    thing you do after reading a trace is run the search again with one stage
    changed, and that meant a terminal.

    A background task rather than an awaited call, because the funnel is minutes
    at its own rate limits -- and because stage 0 and stage 3 spend quota and
    stage 2 spends credits, so watching it is worth something.
    """

    def settled(task: Any) -> None:
        workspace.say(task_message(task))
        workspace.invalidate("funnel")
        workspace.tick()

    def run() -> None:
        question = (entry.value or "").strip()
        if not question:
            return
        entry.value = ""
        argv = ["tools.paper_search", "search", question, "--json"]
        if skip_rerank.value:
            # The one stage that spends credits rather than quota.
            argv.append("--no-rerank")
        start(f"search {question[:40]}", *argv, on_done=settled)
        workspace.say(f"searching — {question[:60]}")
        workspace.invalidate("tasks")
        workspace.tick()

    with kit.row("grad-pad", gap=6).style("border-bottom: var(--grad-border)"):
        entry = (
            ui.input(placeholder="a research question, in words")
            .props("borderless dense")
            .classes("field")
            .style("flex: 1 1 auto; padding: 0 8px")
        )
        entry.on("keydown.enter.prevent", run)
        skip_rerank = ui.checkbox("no rerank").props("dense")
        skip_rerank.props('title="stage 2 costs credits; the rest of the funnel does not"')
        kit.button("SEARCH ⏎", tone="primary", on_click=run)


def subtitle(workspace: Any) -> str:
    model = workspace.model("funnel") or {}
    trace = model.get("trace")
    if not trace:
        return "no searches yet"
    return f"{trace['name']} · {len(trace.get('survivors') or [])} in context"


def render(workspace: Any) -> None:
    from nicegui import ui

    model = workspace.model("funnel") or {}
    kit.error_strip(model.get("error"))
    trace = model.get("trace")

    _search(ui, workspace)
    if not trace:
        kit.empty("No searches yet.", model.get("empty_fix"))
        return

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
        ui.select(
            model.get("traces") or [],
            value=trace["name"],
            on_change=lambda e: workspace.select("funnel.trace", e.value),
        ).props("dense borderless").style("min-width: 220px")
        kit.spacer()

    with kit.pad():
        kit.text(trace.get("question", ""), "").style("font-size: 15px; margin-bottom: 10px")

        for stage in trace.get("bars") or []:
            tone = {"rerank": "rerank", "context": "context"}.get(stage["tone"], "")
            indent = (1.0 - stage["width"]) * 100 / 2
            bar = kit.text(stage["label"], f"grad-stage {tone}".strip())
            bar.style(f"width: {stage['width'] * 100:.1f}%; margin-left: {indent:.1f}%")

        expansion = trace.get("expansion") or {}
        if expansion.get("queries"):
            kit.hr()
            kit.label("stage 0 — expansion")
            for query in expansion["queries"]:
                kit.text(f"· {query}", "grad-mono")
            kit.text(
                f"HyDE passage: {expansion.get('hyde_words', 0)} words — dense side of the local "
                "index only; a synthetic abstract dilutes a lexical query",
                "grad-caption",
            )

        for warning in trace.get("warnings") or []:
            kit.note(f"⚠ {warning}")

        kit.hr()
        kit.label("survivors, in rank order")
        for survivor in trace.get("survivors") or []:
            with kit.column("grad-row", gap=2):
                with kit.row("", gap=9):
                    kit.chip(str(survivor["rank"]), "solid")
                    kit.text(survivor["title"], "", tag="span")
                    kit.spacer()
                    kit.text(str(survivor.get("score") if survivor.get("score") is not None else "—"),
                             "grad-mono", tag="span")
                if survivor.get("reason"):
                    kit.text(survivor["reason"], "grad-caption")

        dropped = trace.get("dropped") or []
        if dropped:
            kit.hr()
            kit.label("dropped")
            for item in dropped:
                with kit.row("grad-row grad-dropped", gap=9):
                    kit.text(item["title"], "", tag="span")
                    kit.spacer()
                    kit.text(item.get("reason") or "", "grad-caption", tag="span")
