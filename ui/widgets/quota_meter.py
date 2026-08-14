"""Widget 3: the quota and spend meter (HANDOFF §10, extended by §15).

One honesty note is part of the widget, not a footnote: Anthropic exposes no
remaining-quota API and the Max 5x window (5-hour rolling plus weekly caps) is
opaque, so the token meter is *self-measured usage against an assumed budget*
and is labelled that way on screen. That is exactly what the §5 stage-0/3
decision needs -- relative attribution by stage -- just not a fuel gauge.

GPU spend is different: it is real dollars, counted with in-flight runs at their
estimates, against the ceiling that actually blocks submissions.

§15 adds the project dimension: a selector, and three bars rather than one --
GPU dollars, credits, and tokens, each against the current project's ceiling.
All of it is read from the ledger; no new logic lives in the UI, per §10.
"""

from __future__ import annotations

from typing import Any

from core import budget as budget_mod, config as config_mod, ledger_store as ls, quota_log


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


def _project_state() -> dict[str, Any] | None:
    current = budget_mod.current_project()
    if not current or not budget_mod.exists(current):
        return None
    return budget_mod.status(current)


# How each resource renders. Kept as data so the header and the panel cannot
# disagree about what a bar means or how a number is formatted.
_RESOURCE_LABELS = {
    "gpu_usd": ("GPU", lambda v: f"${v:,.2f}"),
    "credits_usd": ("credits", lambda v: f"${v:,.2f}"),
    "quota_tokens": ("tokens", lambda v: f"{int(v):,}"),
}


def _bar(ui: Any, resource: str, node: dict[str, Any]) -> None:
    label, fmt = _RESOURCE_LABELS[resource]
    ceiling = node.get("ceiling")
    with ui.column().classes("gap-0"):
        if ceiling is None:
            ui.label(f"{fmt(node['spent'])} {label}").classes("font-mono")
            ui.label("no ceiling set").classes("opacity-50")
            return
        ui.label(f"{fmt(node['spent'])} / {fmt(ceiling)}").classes(
            "font-mono" + (" text-red-400" if node["over"] else "")
        )
        ui.linear_progress(node.get("fraction") or 0.0, show_value=False).classes("w-28 h-1")
        ui.label(label + (" — over" if node["over"] else "")).classes("opacity-60")


def quota_meter() -> Any:
    """The persistent header strip: a project selector and three bars."""
    from nicegui import ui

    spend = _spend()
    tokens = quota_log.summarise(days=7)
    project = _project_state()

    with ui.row().classes("items-center gap-4 text-xs"):
        _project_selector(ui)

        if project:
            for resource in ("gpu_usd", "credits_usd", "quota_tokens"):
                _bar(ui, resource, project["resources"][resource])
        else:
            # No project selected: fall back to the machine-wide view, which is
            # the ceiling that actually blocks submissions either way.
            with ui.column().classes("gap-0"):
                ui.label(f"${spend['total']:.2f} / ${spend['monthly']:.0f}").classes("font-mono")
                ui.linear_progress(spend["fraction"], show_value=False).classes("w-32 h-1")
                ui.label(
                    f"{spend['window']}d GPU · ${spend['in_flight']:.2f} in flight"
                ).classes("opacity-60")
            with ui.column().classes("gap-0"):
                ui.label(f"{tokens['total_tokens']:,} tok (7d)").classes("font-mono")
                ui.label("self-measured, not a fuel gauge").classes("opacity-50")

        if project and project["over_budget"]:
            ui.badge("over budget", color="red").tooltip(
                "cost-bearing commands are denied until the ceiling is raised deliberately"
            )
        if spend["stale"]:
            ui.badge(f"{len(spend['stale'])} stale", color="red").tooltip(
                "submissions are blocked until these are collected"
            )
        elif spend["uncollected"]:
            ui.badge(f"{len(spend['uncollected'])} uncollected", color="amber")


def _project_selector(ui: Any) -> None:
    """Selecting here writes the same file `tools.budget use` writes.

    One selection mechanism, not two: a UI-only notion of "current project"
    would attribute the CLI's spend to the wrong allocation the moment the two
    disagreed.
    """
    projects = budget_mod.projects()
    options = ["(none)"] + [p for p, d in projects.items() if d["status"] == "open"]
    current = budget_mod.current_project() or "(none)"
    if current not in options:
        options.append(current)

    def choose(event: Any) -> None:
        budget_mod.set_current(None if event.value == "(none)" else event.value)

    ui.select(options, value=current, on_change=choose).props("dense outlined").classes(
        "w-44 text-xs"
    ).tooltip("the project every cost-bearing record is charged to")


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
                     "remaining-quota API and the Max 5x window is opaque, so a token "
                     "ceiling is a proxy you control, not a mirror of the real limit.").classes(
                "text-xs opacity-60 max-w-md"
            )

    _project_breakdown(ui, tokens)

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

    _role_table(ui, tokens)


def _project_breakdown(ui: Any, tokens: dict[str, Any]) -> None:
    """Per-project spend against ceilings, and per-project usage by stage (§15)."""
    projects = budget_mod.projects()
    if not projects:
        return

    ui.label("Projects").classes("text-sm font-semibold mt-4")
    rows = []
    for pid, proj in projects.items():
        state = budget_mod.status(pid)
        usage = tokens["by_project"].get(pid, {})
        res = state["resources"]
        rows.append(
            {
                "project": pid,
                "title": proj["title"],
                "status": proj["status"],
                "gpu": _cell(res["gpu_usd"], "${:,.2f}"),
                "credits": _cell(res["credits_usd"], "${:,.2f}"),
                "tokens": _cell(res["quota_tokens"], "{:,.0f}"),
                "calls": usage.get("calls", 0),
                "over": ", ".join(state["over_budget"]) or "—",
            }
        )
    ui.table(
        columns=[
            {"name": "project", "label": "project", "field": "project", "align": "left", "sortable": True},
            {"name": "title", "label": "title", "field": "title", "align": "left"},
            {"name": "gpu", "label": "GPU $", "field": "gpu", "align": "left"},
            {"name": "credits", "label": "credits $", "field": "credits", "align": "left"},
            {"name": "tokens", "label": "tokens", "field": "tokens", "align": "left"},
            {"name": "calls", "label": "calls", "field": "calls", "sortable": True},
            {"name": "over", "label": "over", "field": "over", "align": "left"},
            {"name": "status", "label": "status", "field": "status", "align": "left"},
        ],
        rows=rows,
        row_key="project",
    ).classes("w-full")


def _cell(node: dict[str, Any], fmt: str) -> str:
    """`spent / ceiling`, or bare spend where no ceiling is set.

    "unbounded" and "a very large ceiling" must not look the same, so the
    absence of a ceiling is spelled out rather than rendered as a full bar.
    """
    spent = fmt.format(node["spent"])
    if node["ceiling"] is None:
        return f"{spent} (no ceiling)"
    return f"{spent} / {fmt.format(node['ceiling'])}"


def _role_table(ui: Any, tokens: dict[str, Any]) -> None:
    """What each §16 model role cost.

    This is the question the role tagging exists to answer -- "what did Opus
    cost me this week" -- without inferring a role from a model id.
    """
    by_role = {k: v for k, v in tokens["by_role"].items() if v["calls"]}
    if not by_role:
        return
    ui.label("Tokens by role").classes("text-sm font-semibold mt-4")
    ui.table(
        columns=[
            {"name": "role", "label": "role", "field": "role", "align": "left", "sortable": True},
            {"name": "calls", "label": "calls", "field": "calls", "sortable": True},
            {"name": "input", "label": "in", "field": "input", "sortable": True},
            {"name": "output", "label": "out", "field": "output", "sortable": True},
        ],
        rows=[
            {"role": role, "calls": d["calls"], "input": d["input_tokens"], "output": d["output_tokens"]}
            for role, d in by_role.items()
        ],
        row_key="role",
    ).classes("w-full")
