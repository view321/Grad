"""Widget 3: the quota and spend meter (HANDOFF §10).

One honesty note is part of the widget, not a footnote: Anthropic exposes no
remaining-quota API and the Max 5x window (5-hour rolling plus weekly caps) is
opaque, so the token meter is *self-measured usage against an assumed budget*
and is labelled that way on screen. That is exactly what the §5 stage-0/3
decision needs -- relative attribution by stage -- just not a fuel gauge.

GPU spend is different: it is real dollars, counted with in-flight runs at their
estimates, against the ceiling that actually blocks submissions.
"""

from __future__ import annotations

from typing import Any

from core import config as config_mod, ledger_store as ls, quota_log


def _spend() -> dict[str, Any]:
    cfg = config_mod.load()
    window = int(cfg.get("spend", "window_days", 30))
    rolling = ls.rolling_spend(window)
    monthly = float(cfg.get("spend", "monthly_usd", 200.0))
    return {
        "window": window,
        "monthly": monthly,
        "total": rolling["total_usd"],
        "actual": rolling["actual_usd"],
        "in_flight": rolling["in_flight_usd"],
        "fraction": min(1.0, rolling["total_usd"] / monthly) if monthly else 0.0,
        "uncollected": [r.id for r in ls.in_flight()],
        "stale": [r.id for r in ls.stale_runs(cfg=cfg)],
    }


def quota_meter() -> Any:
    """The persistent header strip."""
    from nicegui import ui

    spend = _spend()
    tokens = quota_log.summarise(days=7)

    with ui.row().classes("items-center gap-4 text-xs"):
        with ui.column().classes("gap-0"):
            ui.label(f"${spend['total']:.2f} / ${spend['monthly']:.0f}").classes("font-mono")
            ui.linear_progress(spend["fraction"], show_value=False).classes("w-32 h-1")
            ui.label(
                f"{spend['window']}d GPU · ${spend['in_flight']:.2f} in flight"
            ).classes("opacity-60")
        with ui.column().classes("gap-0"):
            ui.label(f"{tokens['total_tokens']:,} tok (7d)").classes("font-mono")
            ui.label("self-measured, not a fuel gauge").classes("opacity-50")
        if tokens["total_credits_usd"]:
            ui.label(f"${tokens['total_credits_usd']:.2f} credits").classes("font-mono opacity-70")
        if spend["stale"]:
            ui.badge(f"{len(spend['stale'])} stale", color="red").tooltip(
                "submissions are blocked until these are collected"
            )
        elif spend["uncollected"]:
            ui.badge(f"{len(spend['uncollected'])} uncollected", color="amber")


def quota_panel() -> None:
    """The full breakdown: which stage spent what."""
    from nicegui import ui

    spend = _spend()
    tokens = quota_log.summarise()

    with ui.row().classes("gap-8 w-full"):
        with ui.column().classes("gap-1"):
            ui.label("GPU spend").classes("text-sm font-semibold")
            ui.label(f"actual: ${spend['actual']:.2f}").classes("font-mono text-sm")
            ui.label(f"in flight (at estimate): ${spend['in_flight']:.2f}").classes("font-mono text-sm")
            ui.label(f"ceiling: ${spend['monthly']:.2f} / {spend['window']}d").classes("font-mono text-sm opacity-70")
            if spend["stale"]:
                ui.label("submissions blocked: " + ", ".join(spend["stale"])).classes("text-red-400 text-xs")
                ui.code(f"python -m tools.jobs collect {spend['stale'][0]} --json", language="bash")
        with ui.column().classes("gap-1"):
            ui.label("Tokens by stage").classes("text-sm font-semibold")
            ui.label("Self-measured usage against an assumed budget. Anthropic exposes no "
                     "remaining-quota API and the Max 5x window is opaque.").classes("text-xs opacity-60 max-w-md")

    rows = [
        {
            "stage": stage,
            "calls": data["calls"],
            "input": data["input_tokens"],
            "output": data["output_tokens"],
            "cached": data["cache_read_tokens"],
            "credits_usd": data["credits_usd"],
        }
        for stage, data in tokens["by_stage"].items()
    ]
    if rows:
        ui.table(
            columns=[
                {"name": "stage", "label": "stage", "field": "stage", "align": "left", "sortable": True},
                {"name": "calls", "label": "calls", "field": "calls", "sortable": True},
                {"name": "input", "label": "in", "field": "input", "sortable": True},
                {"name": "output", "label": "out", "field": "output", "sortable": True},
                {"name": "cached", "label": "cached", "field": "cached", "sortable": True},
                {"name": "credits_usd", "label": "credits $", "field": "credits_usd", "sortable": True},
            ],
            rows=rows,
            row_key="stage",
        ).classes("w-full")

        by_stage = tokens["by_stage"]
        ui.echart(
            {
                "tooltip": {"trigger": "item"},
                "series": [
                    {
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "data": [
                            {"name": stage, "value": d["input_tokens"] + d["output_tokens"]}
                            for stage, d in by_stage.items()
                            if d["input_tokens"] + d["output_tokens"] > 0
                        ],
                    }
                ],
            }
        ).classes("w-full h-64")
    else:
        ui.label("No usage recorded yet.").classes("text-sm opacity-60")
