"""Window 9 — cited papers.

Not a file listing. The status chips are what make this a research surface: "3
claims depend on this" is computed by walking every expectation's `basis`, and
"contradicts exp-…" by intersecting that with the falsified set. A paper the
agent queued but never read is shown as such rather than omitted, because the
gap between "we cited this" and "we read this" is exactly the gap a citation
discipline exists to close.

Matching a basis to a directory needs normalising: a basis cites
"arXiv:2001.08361", the directory is `2001.08361`, and the corpus stores
`arxiv_2001.08361`. Comparing them raw yields a claim count of zero, which looks
like working software.
"""

from __future__ import annotations

from typing import Any

from ui import kit

FILTERS = (("cited", "CITED IN PAPER"), ("read", "READ"), ("queued", "QUEUED"))


def subtitle(workspace: Any) -> str:
    model = workspace.model("papers") or {}
    counts = model.get("counts") or {}
    return f"{counts.get('cited', 0)} cited · {counts.get('read', 0)} read · {counts.get('queued', 0)} queued"


def render(workspace: Any) -> None:
    model = workspace.model("papers") or {}
    kit.error_strip(model.get("error"))
    if not model.get("all"):
        kit.empty("No papers ingested yet.", model.get("empty_fix"))
        return

    active = model.get("filter", "cited")
    counts = model.get("counts") or {}
    with kit.row("grad-pad", gap=6).style("border-bottom: var(--grad-border)"):
        for key, caption in FILTERS:
            kit.button(
                f"{caption} {counts.get(key, 0)}",
                tone="active" if key == active else "neutral",
                on_click=lambda _=None, k=key: workspace.select("papers.filter", k),
            )

    rows = model.get("rows") or []
    if not rows:
        kit.empty(f"Nothing matches {active}.")
        return

    selected = workspace.selection.get("papers.selected")
    with kit.row("", gap=0, align="stretch").style("min-height: 0; flex: 1 1 auto"):
        with kit.column("", gap=0).style("flex: 1 1 auto; min-width: 0; overflow-y: auto"):
            for row in rows:
                _row(workspace, row, selected == row["id"])
        _reader(next((r for r in rows if r["id"] == selected), None))


def _row(workspace: Any, row: dict[str, Any], selected: bool) -> None:
    classes = "grad-row" + (" striped selected" if selected else "")
    container = kit.row(classes, gap=11, align="flex-start")
    container.on("click", lambda _=None, pid=row["id"]: workspace.select("papers.selected", pid, window="papers"))
    with container:
        kit.el("div", f"grad-cover {'' if row['read'] else 'unread'}".strip())
        with kit.column("", gap=4).style("min-width: 0; flex: 1 1 auto"):
            kit.text(row["title"], "grad-serif").style("font-size: 21px; line-height: 1.2")
            kit.text(row["authors"], "grad-caption")
            with kit.row("", gap=6).style("flex-wrap: wrap; margin-top: 4px"):
                for chip in row.get("chips") or []:
                    kit.chip(chip["text"], chip["tone"])


def _reader(row: dict[str, Any] | None) -> None:
    with kit.column("grad-pad", gap=9).style(
        "flex: 0 0 520px; background: var(--grad-paper-sunk); "
        "border-left: var(--grad-border); overflow-y: auto"
    ):
        if row is None:
            kit.text("select a paper", "grad-caption")
            return
        kit.label("reader")
        kit.text(row["title"], "grad-serif").style("font-size: 21px")
        kit.text(row["path"], "grad-caption")
        kit.figure_placeholder("page 1", "PDF · not rendered")
        if row.get("claims"):
            with kit.el("div", "grad-card"):
                kit.text("PULLED INTO", "head attention")
                with kit.el("div", "body"):
                    for claim in row["claims"]:
                        kit.text(claim, "grad-mono")
                    kit.text(
                        "these expectations name this paper in their basis — the sentence that "
                        "became a prediction is in the ledger entry's `locator`",
                        "grad-caption",
                    )
        else:
            kit.note("nothing in the ledger cites this paper yet")
