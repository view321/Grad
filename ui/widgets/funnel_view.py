"""Widget 4: the funnel view (HANDOFF §10).

    "400 -> 50 -> 15 with stage-3's one-line reason per surviving candidate on
     hover. This is the debugging surface for retrieval, and it is what makes the
     stage-0/3 evaluation in §5 interpretable rather than a pair of numbers."

Reads the traces `paper_search.py` writes to `notes/funnel/`. A dense sortable
table is exactly what `ui.table` already is, which is part of why NiceGUI won
this decision.
"""

from __future__ import annotations

import json
from typing import Any

from core import paths


def _traces() -> list[str]:
    d = paths.notes_dir() / "funnel"
    if not d.exists():
        return []
    return sorted((p.stem for p in d.glob("*.json")), reverse=True)


def _load(name: str) -> dict[str, Any] | None:
    path = paths.notes_dir() / "funnel" / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def funnel_view() -> None:
    from nicegui import ui

    names = _traces()
    if not names:
        ui.label("No searches yet.").classes("text-sm opacity-60")
        ui.code('python -m tools.paper_search search "..." --json', language="bash")
        return

    container = ui.column().classes("w-full gap-2")

    def show(name: str) -> None:
        container.clear()
        trace = _load(name)
        with container:
            if not trace:
                ui.label("could not read that trace").classes("text-red-400")
                return
            ui.label(trace.get("question", "")).classes("text-base font-semibold")

            stages = trace.get("stages", {})
            counts = [
                ("retrieved", stages.get("1_retrieve", {}).get("candidates", 0)),
                ("reranked", stages.get("2_rerank", {}).get("out", 0)),
                ("kept", stages.get("3_triage", {}).get("returned", len(trace.get("survivors", [])))),
            ]
            with ui.row().classes("items-center gap-3"):
                for i, (label, value) in enumerate(counts):
                    if i:
                        ui.icon("arrow_forward").classes("opacity-40")
                    with ui.column().classes("gap-0 items-center"):
                        ui.label(str(value)).classes("text-2xl font-mono")
                        ui.label(label).classes("text-xs opacity-60")

            expand = stages.get("0_expand", {})
            if expand.get("queries"):
                with ui.expansion("stage 0 — expansion", value=False).classes("w-full"):
                    ui.label("keyword queries (lexical, for Semantic Scholar)").classes("text-xs opacity-60")
                    for q in expand["queries"]:
                        ui.label(f"· {q}").classes("font-mono text-sm")
                    ui.label(
                        f"HyDE passage: {expand.get('hyde_words', 0)} words "
                        "(dense side of the local index only — a synthetic abstract "
                        "dilutes a lexical query)"
                    ).classes("text-xs opacity-60")

            for warning in trace.get("warnings", []) or []:
                ui.label(f"⚠ {warning}").classes("text-xs text-amber-400")

            survivors = trace.get("survivors", [])
            if survivors:
                ui.table(
                    columns=[
                        {"name": "title", "label": "title", "field": "title", "align": "left", "sortable": True},
                        {"name": "year", "label": "year", "field": "year", "sortable": True},
                        {"name": "source", "label": "source", "field": "source", "sortable": True},
                        {"name": "score", "label": "rerank", "field": "score", "sortable": True},
                        {"name": "reason", "label": "why it survived", "field": "reason", "align": "left"},
                    ],
                    rows=[
                        {
                            "title": s.get("title") or s.get("id"),
                            "year": s.get("year"),
                            "source": s.get("source"),
                            "score": round(s["rerank_score"], 4) if isinstance(s.get("rerank_score"), (int, float)) else None,
                            # Stage 3's per-candidate reason is not decoration: it
                            # is the provenance that populates the ledger's basis.
                            "reason": s.get("reason", ""),
                        }
                        for s in survivors
                    ],
                    row_key="title",
                ).classes("w-full")

    ui.select(names, value=names[0], on_change=lambda e: show(e.value)).classes("w-full")
    show(names[0])
