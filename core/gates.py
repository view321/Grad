"""The four submit gates, and the smoke carve-out (HANDOFF §6).

    "a gate that lives in the system prompt is a gate the model will skip when it
     is three steps into a plan and confident. A gate that lives in the submitter
     is not."

`jobs.py` and `gpu.py` both call `check_submit()` before doing anything that can
cost money, and neither has a flag that turns it off. `--smoke` is not an
exception to that: it is a *different*, hard-capped path, checked by
`check_smoke_caps()` below.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from core import budget as budget_mod, jsonl, ledger_store as ls, paths
from core.config import Config
from core.errors import (
    EXIT_EXPECTATION,
    EXIT_PREFLIGHT,
    EXIT_SPEND,
    EXIT_STALE_RUN,
    GateRefusal,
)
from core.submission import Submission


# ---------------------------------------------------------------------------
# gate 1: a passing preflight record for this exact submission hash
# ---------------------------------------------------------------------------
def preflight_record(submission_hash: str) -> dict[str, Any] | None:
    return jsonl.read_json(paths.preflight_record(submission_hash))


def check_preflight(sub: Submission, cfg: Config, *, required: list[str] | None = None) -> dict[str, Any]:
    h = sub.hash()
    record = preflight_record(h)
    fix = f"python -m tools.preflight run --spec {sub.spec_path} --json"
    if record is None:
        raise GateRefusal(
            "preflight_missing",
            f"no preflight record for submission hash {h}",
            EXIT_PREFLIGHT,
            fix=fix,
            detail={"submission_hash": h, "warnings": sub.warnings},
        )

    required = required if required is not None else list(cfg.get("preflight", "checks", []))
    results = record.get("checks", {})
    missing = [c for c in required if c not in results]
    failing = [c for c in required if results.get(c, {}).get("ok") is False]
    if missing or failing:
        raise GateRefusal(
            "preflight_failing",
            "preflight for this submission is incomplete or failing: "
            + ", ".join(
                [f"{c} missing" for c in missing] + [f"{c} failed" for c in failing]
            ),
            EXIT_PREFLIGHT,
            fix=fix,
            detail={"submission_hash": h, "missing": missing, "failing": failing},
        )
    return record


# ---------------------------------------------------------------------------
# gate 2: an open expectation, bound at submit time
# ---------------------------------------------------------------------------
def check_expectation(expectation_id: str | None, sub: Submission) -> dict[str, Any]:
    fix = (
        "python -m tools.ledger expect --task <task> --quantity <q> "
        "--low <lo> --high <hi> --basis <paper:locator=value> --json"
    )
    if not expectation_id:
        raise GateRefusal(
            "expectation_required",
            "--expect is required: no pre-registration, no submission",
            EXIT_EXPECTATION,
            fix=fix,
        )
    try:
        exp = ls.expectation(expectation_id)
    except Exception:
        raise GateRefusal(
            "expectation_missing",
            f"expectation {expectation_id!r} does not exist",
            EXIT_EXPECTATION,
            fix=fix,
        ) from None
    if expectation_id in ls.bound_expectation_ids():
        raise GateRefusal(
            "expectation_bound",
            f"expectation {expectation_id!r} is already bound to a run; "
            "each prediction covers exactly one run",
            EXIT_EXPECTATION,
            fix="mint a new expectation for this run: " + fix,
        )
    return exp


# ---------------------------------------------------------------------------
# gate 3: per-job and rolling spend ceilings
# ---------------------------------------------------------------------------
def check_spend(estimate_usd: float, cfg: Config, *, now: _dt.datetime | None = None) -> dict[str, Any]:
    per_job = float(cfg.get("spend", "per_job_usd", 25.0))
    monthly = float(cfg.get("spend", "monthly_usd", 200.0))
    window = int(cfg.get("spend", "window_days", 30))

    if estimate_usd > per_job:
        raise GateRefusal(
            "spend_per_job",
            f"estimated ${estimate_usd:.2f} exceeds the per-job ceiling of ${per_job:.2f}",
            EXIT_SPEND,
            fix=(
                "shrink the job, or raise per_job_usd in config/grad.toml deliberately "
                "(the ceiling is the point)"
            ),
            detail={"estimate_usd": estimate_usd, "per_job_usd": per_job},
        )

    rolling = ls.rolling_spend(window, now=now)
    projected = rolling["total_usd"] + estimate_usd
    if projected > monthly:
        raise GateRefusal(
            "spend_monthly",
            (
                f"projected {window}-day spend ${projected:.2f} "
                f"(${rolling['actual_usd']:.2f} actual + ${rolling['in_flight_usd']:.2f} in flight "
                f"+ ${estimate_usd:.2f} this job) exceeds the ceiling of ${monthly:.2f}"
            ),
            EXIT_SPEND,
            fix=(
                "python -m tools.jobs collect <run_id> --json  # collect in-flight runs so their "
                "estimates become actuals, or raise monthly_usd in config/grad.toml"
            ),
            detail={"rolling": rolling, "estimate_usd": estimate_usd, "monthly_usd": monthly},
        )
    return {"rolling": rolling, "projected_usd": round(projected, 4), "monthly_usd": monthly}


# ---------------------------------------------------------------------------
# gate 3b: the project's own allocation (HANDOFF-2 §15)
# ---------------------------------------------------------------------------
def check_project_spend(project_id: str | None, estimate_usd: float) -> dict[str, Any] | None:
    """Alongside the global ceiling, not instead of it.

    Submission is a discrete, gateable event, so GPU dollars are the resource
    this enforces cleanly. It raises with exit **12**, not 6: an organisation's
    budget and the machine's budget are separate allocations, and telling them
    apart is what stops "raise monthly_usd" being the reflex fix for the wrong
    problem.
    """
    return budget_mod.check(project_id, gpu_usd=estimate_usd, what="this job")


# ---------------------------------------------------------------------------
# gate 4: no stale uncollected run
# ---------------------------------------------------------------------------
def check_stale(cfg: Config, *, now: _dt.datetime | None = None) -> None:
    stale = ls.stale_runs(cfg=cfg, now=now)
    if stale:
        ids = ", ".join(r.id for r in stale)
        raise GateRefusal(
            "stale_run",
            (
                f"{len(stale)} run(s) are past their collection window and still uncollected: {ids}. "
                "Spend only becomes actual at collect time; if collection were optional, "
                "the ceiling would be too."
            ),
            EXIT_STALE_RUN,
            fix=f"python -m tools.jobs collect {stale[0].id} --json",
            detail={"stale_run_ids": [r.id for r in stale]},
        )


# ---------------------------------------------------------------------------
# all four, in order
# ---------------------------------------------------------------------------
def check_submit(
    sub: Submission,
    expectation_id: str | None,
    cfg: Config,
    *,
    estimate_usd: float | None = None,
    project: str | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Run every gate. Raises `GateRefusal` on the first that refuses.

    Order is cheapest-and-most-actionable first: a missing preflight is the most
    common refusal and the easiest to fix. The project ceiling runs immediately
    after the global one -- both are spend gates, and a caller that has blown
    both should hear about the machine's ceiling first, since that is the one
    that stops every other project too.
    """
    estimate = sub.estimated_cost_usd() if estimate_usd is None else estimate_usd
    record = check_preflight(sub, cfg)
    expectation = check_expectation(expectation_id, sub)
    spend = check_spend(estimate, cfg, now=now)
    project_state = check_project_spend(project, estimate)
    check_stale(cfg, now=now)
    return {
        "submission_hash": sub.hash(),
        "preflight": {"checks": list(record.get("checks", {})), "verified_at": record.get("verified_at")},
        "expectation": {"id": expectation.get("id"), "quantity": expectation.get("quantity")},
        "spend": spend,
        "project": project,
        # Current state *plus* the projection, because "you have $50 left" and
        # "you will have $48 left after this" answer different questions and the
        # caller is about to spend.
        "project_budget": (
            None
            if project_state is None
            else {
                **project_state["resources"]["gpu_usd"],
                "projected_remaining": (
                    None
                    if project_state["resources"]["gpu_usd"]["remaining"] is None
                    else round(project_state["resources"]["gpu_usd"]["remaining"] - estimate, 4)
                ),
            }
        ),
        "estimate_usd": estimate,
    }


# ---------------------------------------------------------------------------
# the smoke carve-out
# ---------------------------------------------------------------------------
def check_smoke_caps(sub: Submission, cfg: Config, *, requested: dict[str, Any] | None = None) -> dict[str, Any]:
    """Smoke skips the gates above and is hard-capped here instead.

    "The caps are what keep the exemption from becoming the way real jobs escape
     the gate: nothing useful can be trained inside them."

    The caps are applied, not merely validated: whatever the spec asked for, the
    smoke submission is clamped to one step, minutes of wall clock, and cents.
    """
    requested = requested or {}
    max_steps = int(cfg.get("smoke", "max_steps", 1))
    max_wall = int(cfg.get("smoke", "max_wall_clock_s", 600))
    max_cost = float(cfg.get("smoke", "max_cost_usd", 0.50))

    steps = int(requested.get("steps", max_steps))
    wall = int(requested.get("timeout_s", max_wall))
    cost = float(requested.get("cost_usd", sub.estimate.get("smoke_cost_usd", max_cost)))

    clamped = {
        "steps": min(steps, max_steps),
        "timeout_s": min(wall, max_wall),
        "cost_ceiling_usd": min(cost, max_cost),
        "artifact_upload": bool(cfg.get("smoke", "allow_artifact_upload", False)),
    }

    # A spec whose *minimum* possible smoke cost is above the cap cannot be
    # smoked at all, and saying so is better than silently billing more.
    floor_cost = float(sub.estimate.get("smoke_cost_usd", 0.0))
    if floor_cost > max_cost:
        raise GateRefusal(
            "smoke_too_expensive",
            f"the spec's smoke cost estimate ${floor_cost:.2f} exceeds the smoke cap ${max_cost:.2f}",
            EXIT_SPEND,
            fix="use a smaller instance for the smoke step, or lower estimate.smoke_cost_usd",
            detail=clamped,
        )
    return clamped
