"""Window 3 — the ledger: expectations against outcomes.

The highest-value visual in the app, and the reason it is a band strip rather
than a table: predicted range as a band, observed value as a marker, in-range or
not obvious at a glance. A number in a cell requires the reader to do the
comparison; a tick outside a block does the comparison for them.

Unjudged deviations are flagged rather than left to accumulate quietly, which is
the whole argument of §7. A deviation is unjudged when it is not confirmed in
range *and* carries no verdict -- `in_range` is `False` for a numeric miss and
`None` for the cases no program can settle (a relational prediction, a
non-numeric result), and both need a human.
"""

from __future__ import annotations

from typing import Any

from ui import kit

FILTERS = (("open", "OPEN"), ("met", "MET"), ("broken", "BROKEN"))


def subtitle(workspace: Any) -> str:
    model = workspace.model("ledger") or {}
    counts = model.get("counts") or {}
    return f"{counts.get('open', 0)} open · {counts.get('met', 0)} met · {counts.get('broken', 0)} broken"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("ledger") or {}
    unjudged = len([e for e in model.get("entries", []) if e.get("unjudged")])
    return [(f"{unjudged} UNJUDGED", "attention")] if unjudged else []


def render(workspace: Any) -> None:
    model = workspace.model("ledger") or {}
    kit.error_strip(model.get("error"))
    entries = model.get("entries") or []
    if not entries:
        kit.empty("No expectations registered yet.", model.get("empty_fix"))
        return

    active = workspace.selection.get("ledger.filter", "open")
    counts = model.get("counts") or {}

    with kit.row("grad-pad", gap=6).style("border-bottom: var(--grad-border)"):
        for key, caption in FILTERS:
            kit.button(
                f"{caption} {counts.get(key, 0)}",
                tone="active" if key == active else "neutral",
                on_click=lambda _=None, k=key: workspace.select("ledger.filter", k),
            )
        kit.spacer()
        kit.text(f"{model.get('total', 0)} total", "grad-caption")

    visible = [e for e in entries if e["state"] == active]
    if not visible:
        kit.empty(f"Nothing is {active}.")
        return

    for entry in visible:
        _entry(entry)


def _entry(entry: dict[str, Any]) -> None:
    with kit.column(f"grad-row striped {entry['accent']}", gap=0):
        with kit.row("", gap=9):
            kit.text(entry["id"], "grad-mono", tag="span")
            kit.chip(entry["state"].upper(), entry["accent"])
            if entry.get("unjudged"):
                kit.chip("UNJUDGED", "attention")
            kit.spacer()
            kit.text(entry.get("at") or "", "grad-caption", tag="span")

        kit.text(entry["claim"], "").style("font-size: 13.5px; margin: 6px 0")

        if entry["state"] == "open" or entry.get("band"):
            kit.band_strip(entry.get("band"))

        if entry.get("comparability"):
            kit.note(f"comparability — {entry['comparability']}")

        for basis in entry.get("basis") or []:
            kit.text(
                f"· {basis.get('paper')} — {basis.get('locator')} = {basis.get('value')} "
                f"({basis.get('conditions')})",
                "grad-caption",
            )

        if entry.get("runs"):
            with kit.row("", gap=6).style("margin-top: 6px"):
                kit.text("runs", "grad-caption", tag="span")
                for run_id in entry["runs"][:6]:
                    kit.chip(run_id, "outline")

        if entry.get("unjudged"):
            # The verdict is a ledger write, so it belongs to the CLI. The UI's
            # job is to make sure nobody has to go looking for the command.
            kit.pre(
                f"python -m tools.ledger verdict {(entry.get('runs') or ['<run>'])[-1]} "
                f"--quantity {entry.get('quantity') or '<quantity>'} "
                "--verdict bug|real|inconclusive --note '...' --json"
            )
