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
