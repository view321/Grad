"""Widget 2: expectation vs. outcome (HANDOFF §10).

    "The highest-value visual in the app: predicted range as a band, actual as a
     marker, in-range or not obvious at a glance, with the basis citations and
     the comparability note beside it."

Unjudged deviations are flagged, because §7's whole argument is that they
otherwise accumulate quietly.
"""

from __future__ import annotations

from typing import Any

from core import ledger_store as ls


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in ls.runs():
        for dev in run.get("deviations", []) or []:
            expectation = None
            if dev.get("expectation_id"):
                try:
                    expectation = ls.expectation(dev["expectation_id"])
                except Exception:  # noqa: BLE001 - a dangling ref is reported by `ledger verify`
                    expectation = None
            rows.append({"run": run, "dev": dev, "expectation": expectation})
    return rows


def expectation_panel() -> None:
    from nicegui import ui

    rows = _rows()
    if not rows:
        ui.label("No predictions have met an outcome yet.").classes("text-sm opacity-60")
        ui.code(
            "python -m tools.ledger expect --task <task> --quantity <q> --low <lo> --high <hi> "
            "--basis '<paper>|<locator>|<value>|<conditions>' --comparability '<how ours differs>' --json",
            language="bash",
        )
        return

    unjudged = [r for r in rows if r["dev"].get("in_range") is not True and not r["dev"].get("verdict")]
    if unjudged:
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.icon("gavel").classes("text-amber-400")
            ui.label(f"{len(unjudged)} deviation(s) awaiting a verdict").classes("text-amber-400 text-sm")

    _chart(ui, rows)

    for row in reversed(rows[-25:]):
        _entry(ui, row)


def _chart(ui: Any, rows: list[dict[str, Any]]) -> None:
    """Predicted band and actual marker, per quantity."""
    labels, lows, highs, actuals = [], [], [], []
    for row in rows[-20:]:
        dev = row["dev"]
        expected = dev.get("expected") or {}
        if not isinstance(dev.get("actual"), (int, float)):
            continue
        labels.append(f"{dev.get('quantity', '?')}\n{row['run'].id[-6:]}")
        low = expected.get("low")
        high = expected.get("high")
        lows.append(low if low is not None else None)
        highs.append((high - low) if (low is not None and high is not None) else None)
        actuals.append(dev["actual"])
    if not labels:
        return

    ui.echart(
        {
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 60, "right": 20, "top": 30, "bottom": 60},
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"fontSize": 9}},
            "yAxis": {"type": "value", "scale": True},
            "series": [
                # A stacked transparent base plus the band height is the standard
                # ECharts way to draw a range; the marker sits on top of it.
                {"name": "base", "type": "bar", "stack": "band", "data": lows,
                 "itemStyle": {"color": "transparent"}, "silent": True},
                {"name": "predicted", "type": "bar", "stack": "band", "data": highs,
                 "itemStyle": {"color": "rgba(56,189,248,0.25)"}},
                {"name": "actual", "type": "scatter", "data": actuals, "symbolSize": 12,
                 "itemStyle": {"color": "#f59e0b"}},
            ],
        }
    ).classes("w-full h-64")


def _entry(ui: Any, row: dict[str, Any]) -> None:
    dev, run, expectation = row["dev"], row["run"], row["expectation"]
    in_range = dev.get("in_range")
    badge = "in range" if in_range else ("OUT OF RANGE" if in_range is False else "needs judgement")
    colour = "text-emerald-400" if in_range else ("text-red-400" if in_range is False else "text-amber-400")

    with ui.expansion(f"{dev.get('quantity', '?')} — {badge} ({run.id})", value=in_range is False).classes("w-full"):
        with ui.row().classes("gap-6 items-baseline"):
            ui.label(f"actual: {dev.get('actual')}").classes("font-mono")
            expected = dev.get("expected") or {}
            ui.label(f"predicted: {expected.get('low')} – {expected.get('high')}").classes("font-mono opacity-70")
            if dev.get("ratio") is not None:
                ui.label(f"ratio: {dev['ratio']}").classes("font-mono opacity-70")
        ui.label(badge).classes(f"{colour} text-sm")

        if expectation:
            if expectation.get("claim"):
                ui.label(expectation["claim"]).classes("text-sm italic")
            if expectation.get("comparability"):
                # The field that prevents the whole system from generating
                # confident nonsense: a number from a paper means nothing
                # without matching setup.
                ui.label(f"comparability: {expectation['comparability']}").classes("text-xs opacity-70")
            for basis in expectation.get("basis", []) or []:
                ui.label(
                    f"· {basis.get('paper')} — {basis.get('locator')} = {basis.get('value')} "
                    f"({basis.get('conditions')})"
                ).classes("text-xs opacity-60 font-mono")
            ui.label(f"confidence: {expectation.get('confidence')}").classes("text-xs opacity-60")

        if dev.get("verdict"):
            ui.label(f"verdict: {dev['verdict']} — {dev.get('note', '')}").classes("text-sm")
        else:
            ui.code(
                f"python -m tools.ledger verdict {run.id} --quantity {dev.get('quantity')} "
                "--verdict bug|real|inconclusive --note '...' --json",
                language="bash",
            ).classes("w-full")
