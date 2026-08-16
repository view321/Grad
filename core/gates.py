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
import math
from typing import Any

from core import budget as budget_mod, jsonl, ledger_store as ls, paths
from core.config import Config
from core.errors import (
    EXIT_CONCURRENCY,
    EXIT_EXPECTATION,
    EXIT_PREFLIGHT,
    EXIT_SPEND,
    EXIT_STALE_RUN,
    ConfigError,
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
    if not isinstance(results, dict):
        results = {}
    missing = [c for c in required if c not in results]
    # `ok is True`, not `not (ok is False)`. A check whose entry is `{}`, or
    # `{"ok": null}`, or a bare string is not a check that passed -- it is a
    # record a crashed writer or a future check left half-written, and the gate
    # is the part of this system that does not trust its inputs. Enumerating the
    # known-bad states let every unknown state through.
    def _entry(name: str) -> dict[str, Any]:
        value = results.get(name)
        return value if isinstance(value, dict) else {}

    failing = [c for c in required if c in results and _entry(c).get("ok") is False]
    # Anything that is neither a pass nor an explicit failure: `{}`, `{"ok":
    # null}`, a bare string. Reported separately because "this check failed" and
    # "this check recorded no verdict" send you to different places.
    unverified = [
        c
        for c in required
        if c in results and c not in failing and _entry(c).get("ok") is not True
    ]
    if missing or failing or unverified:
        raise GateRefusal(
            "preflight_failing",
            "preflight for this submission is incomplete or failing: "
            + ", ".join(
                [f"{c} missing" for c in missing]
                + [f"{c} failed" for c in failing]
                + [f"{c} recorded no pass/fail verdict" for c in unverified]
            ),
            EXIT_PREFLIGHT,
            fix=fix,
            detail={
                "submission_hash": h,
                "missing": missing,
                "failing": failing,
                "unverified": unverified,
            },
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
    if expectation_id in ls.falsified_ids():
        raise GateRefusal(
            "expectation_falsified",
            f"expectation {expectation_id!r} was retracted; a withdrawn prediction is not "
            "pre-registration, and binding one after the fact is the thing §7 exists to stop",
            EXIT_EXPECTATION,
            fix="mint a new expectation for this run: " + fix,
        )
    # Runs *and* campaigns: `evolve` consumes an expectation the same way a
    # submission does, and checking only one of the two ledgers here let one
    # prediction cover both.
    if expectation_id in ls.consumed_expectation_ids():
        raise GateRefusal(
            "expectation_bound",
            f"expectation {expectation_id!r} is already bound to a run or campaign; "
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
    """Refuse while an uncollected run is past its grace window.

    The fix has to distinguish two populations, because one of them cannot take
    the other's advice. A run that reached a backend has a handle and is
    collectable. A run that never reached one has no handle, and `collect`
    refuses it with `no_handle` -- so pointing every stale run at `collect` sent
    exactly the runs that could not be collected to the one command that would
    not work, and the only remaining exit was editing `runs.jsonl` by hand.
    Those go to `ledger abandon` instead.
    """
    stale = ls.stale_runs(cfg=cfg, now=now)
    if not stale:
        return
    from core import submit as submit_lib  # noqa: PLC0415 - avoids an import cycle

    unreached = [r for r in stale if not r.get("handle")]
    ids = ", ".join(r.id for r in stale)
    raise GateRefusal(
        "stale_run",
        (
            f"{len(stale)} run(s) are past their collection window and still uncollected: {ids}. "
            "Spend only becomes actual at collect time; if collection were optional, "
            "the ceiling would be too."
        ),
        EXIT_STALE_RUN,
        # The first run that can actually take the advice given, so a mixed batch
        # names a command that works rather than the first id in the list.
        fix=(
            f'python -m tools.ledger abandon {unreached[0].id} --reason "..." --json  '
            "# never reached a backend, so there is nothing to collect"
            if unreached
            else submit_lib.collect_command(stale[0])
        ),
        detail={
            "stale_run_ids": [r.id for r in stale],
            # Split out rather than left for the caller to re-derive: which of
            # these is collectable and which is not is the whole of the decision.
            "unreached_run_ids": [r.id for r in unreached],
        },
    )


# ---------------------------------------------------------------------------
# gate 5: not too many in flight at once
# ---------------------------------------------------------------------------
def concurrency_ceiling(sub: Submission | None, cfg: Config) -> int:
    """How many runs may be in flight: the spec's `[execution] max_concurrent`,
    then `[execution] max_concurrent_runs`.

    Per-spec because the right number is a property of the work: two ten-hour
    training jobs is a different proposition from two two-minute evaluations, and
    the spec is where the job describes itself. Bounded below at 1 -- a ceiling of
    zero would refuse every submission, which is a configuration typo that reads
    as the whole system being broken.
    """
    declared = (sub.execution.get("max_concurrent") if sub else None)
    if declared is None:
        declared = cfg.get("execution", "max_concurrent_runs", 2)
    try:
        return max(1, int(declared))
    except (TypeError, ValueError):
        raise ConfigError(
            f"max_concurrent must be a whole number, not {declared!r}",
            fix="fix [execution] max_concurrent in the spec, or max_concurrent_runs in config/grad.toml",
        ) from None


def check_concurrency(cfg: Config, *, sub: Submission | None = None) -> dict[str, Any]:
    """Refuse when too many runs are already in flight.

    **This is the gate that makes parallel submission safe rather than merely
    possible**, and the argument is entirely about `check_stale`. That gate
    refuses *every* later submission while *any* uncollected run is past its
    window -- so each additional job in flight is another collection window open,
    and one wedged backend takes every other job's successor down with it. Without
    a ceiling, "submit things in parallel" and "exit 7 is the normal state" are
    the same sentence.

    It is deliberately not folded into `check_stale`. They refuse for opposite
    reasons: 7 means something has gone wrong with a run and needs clearing, 14
    means nothing has gone wrong at all and you should wait. Giving them one exit
    code would make "abandon it" look like the fix for a healthy system.

    Smoke runs are counted. They are short, but they are real jobs on a real
    backend holding a real collection window, and §6's argument about the smoke
    carve-out not becoming the hole in the ceiling applies unchanged.
    """
    ceiling = concurrency_ceiling(sub, cfg)
    live = ls.in_flight()
    if len(live) < ceiling:
        return {"in_flight": len(live), "ceiling": ceiling}
    from core import submit as submit_lib  # noqa: PLC0415 - avoids an import cycle

    oldest = live[0]
    raise GateRefusal(
        "too_many_in_flight",
        (
            f"{len(live)} run(s) are already in flight and the ceiling is {ceiling}: "
            + ", ".join(r.id for r in live[:4])
            + ". Every run in flight holds a collection window open, and one that goes "
            "stale refuses every later submission (exit 7)."
        ),
        EXIT_CONCURRENCY,
        # The oldest, because it is the one most likely to be finished and is
        # certainly the one closest to going stale.
        fix=submit_lib.collect_command(oldest),
        detail={
            "in_flight": [r.id for r in live],
            "ceiling": ceiling,
            "source": "spec [execution] max_concurrent"
            if sub is not None and sub.execution.get("max_concurrent") is not None
            else "config [execution] max_concurrent_runs",
        },
    )


# ---------------------------------------------------------------------------
# all five, in order
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
    # Last, and after the stale check on purpose. A stale run is *also* in flight,
    # so a caller with one wedged job would otherwise be told "too many in flight,
    # wait" -- advice that never becomes true, for a situation whose actual fix is
    # to collect or abandon the stale one. 7 first, then 14.
    concurrency = check_concurrency(cfg, sub=sub)
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
        "concurrency": concurrency,
    }


# ---------------------------------------------------------------------------
# the smoke carve-out
# ---------------------------------------------------------------------------
# A smoke that cannot run a minute cannot run one step of anything real, so
# clamping below this is refusing with extra steps.
MIN_SMOKE_WALL_S = 60


def _finite(value: Any, what: str) -> float:
    """A float that can actually bound something, or a refusal.

    `nan` and `inf` are valid TOML floats, so both a spec's `[estimate]` and a
    host's rate can carry them, and neither is caught by a sign check. They fail
    in opposite and equally bad ways:

      * **NaN fails every comparison.** `rate < 0` and `rate > 0` are both
        False, so the affordability block below was skipped entirely -- no
        wall-clock clamp, no cost refusal, `projected_cost_usd` recorded as
        `nan`. A rate of NaN was the one input that disabled the cost cap while
        passing the check written to stop exactly that (`rate_usd_per_hour is
        None`).
      * **Infinity converts to nothing.** `int(inf)` raises OverflowError and
        `int(nan)` raises ValueError, so a non-finite cost reached
        `int(affordable_s)` and came out as exit 1, "a bug in the CLI" --
        when it is a bug in a file the user can fix.

    Refused here rather than at the config loader alone, because the spec and
    the rate are not config: one is a file the agent writes, the other is a
    lookup.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise GateRefusal(
            "smoke_value_invalid",
            f"{what} is not a number ({value!r})",
            EXIT_SPEND,
            fix="give it a finite number, or remove it to take the configured default",
        ) from None
    if not math.isfinite(number):
        raise GateRefusal(
            "smoke_value_invalid",
            f"{what} is {number}, which cannot bound anything -- a cap that is not a "
            "finite number is not a cap",
            EXIT_SPEND,
            fix=(
                "give it a finite number. `nan` and `inf` are valid TOML floats and neither "
                "can be compared against a spend"
            ),
        )
    return number


def check_smoke_caps(
    sub: Submission,
    cfg: Config,
    *,
    requested: dict[str, Any] | None = None,
    rate_usd_per_hour: float | None = None,
    target_name: str = "this target",
) -> dict[str, Any]:
    """Smoke skips the gates above and is hard-capped here instead.

    "The caps are what keep the exemption from becoming the way real jobs escape
     the gate: nothing useful can be trained inside them."

    The caps are applied, not merely validated: whatever the spec asked for, the
    smoke submission is clamped to one step, minutes of wall clock, and cents.

    `rate_usd_per_hour` is what makes the cost cap real rather than advisory.
    Without it the only cost refusal was against `estimate.smoke_cost_usd` --
    a number the spec declares about itself, defaulting to 0.0 -- so a smoke on
    a $4.13/h flavor could run the full 600 s wall cap and bill $0.69 against a
    $0.50 ceiling. The wall clock is therefore clamped to what the rate affords,
    and a rate we cannot look up is a refusal: an unpriced flavor is exactly the
    one that turns the ceiling into decoration.
    """
    requested = requested or {}
    max_steps = int(cfg.get("smoke", "max_steps", 1))
    max_wall = int(cfg.get("smoke", "max_wall_clock_s", 600))
    max_cost = _finite(cfg.get("smoke", "max_cost_usd", 0.50), "smoke.max_cost_usd")

    steps = int(requested.get("steps", max_steps))
    wall = int(requested.get("timeout_s", max_wall))
    cost = _finite(
        requested.get("cost_usd", sub.estimate.get("smoke_cost_usd", max_cost)),
        "the smoke cost estimate",
    )

    clamped = {
        "steps": min(steps, max_steps),
        "timeout_s": min(wall, max_wall),
        "cost_ceiling_usd": min(cost, max_cost),
        "artifact_upload": bool(cfg.get("smoke", "allow_artifact_upload", False)),
    }

    # A spec whose *minimum* possible smoke cost is above the cap cannot be
    # smoked at all, and saying so is better than silently billing more.
    floor_cost = _finite(sub.estimate.get("smoke_cost_usd", 0.0), "estimate.smoke_cost_usd")
    if floor_cost > max_cost:
        raise GateRefusal(
            "smoke_too_expensive",
            f"the spec's smoke cost estimate ${floor_cost:.2f} exceeds the smoke cap ${max_cost:.2f}",
            EXIT_SPEND,
            fix="use a smaller instance for the smoke step, or lower estimate.smoke_cost_usd",
            detail=clamped,
        )

    if rate_usd_per_hour is None:
        raise GateRefusal(
            "smoke_rate_unknown",
            f"no hourly rate is known for {target_name}, so the smoke cost cap of "
            f"${max_cost:.2f} cannot be enforced -- and a cap that cannot be computed "
            "is not a cap",
            EXIT_SPEND,
            fix=(
                "add the flavor to [hf.flavor_rates] in config/grad.toml (or set the host's "
                "rate_usd_per_hour), then re-run"
            ),
            detail={**clamped, "target": target_name},
        )

    rate = _finite(rate_usd_per_hour, f"the hourly rate for {target_name}")
    if rate < 0:
        raise GateRefusal(
            "smoke_rate_invalid",
            f"the hourly rate for {target_name} is negative (${rate:.2f}/h)",
            EXIT_SPEND,
            fix="fix the rate in config/grad.toml; a negative rate would credit spend back",
            detail={**clamped, "target": target_name},
        )

    # A ceiling of zero or less is not "this smoke may cost nothing", it is a
    # spec that declared `smoke_cost_usd = 0` (or asked for `cost_usd: 0`)
    # meaning it expects the step to be free. Taken literally it made
    # `affordable_s` zero and refused every such smoke as too expensive --
    # punishing the spec that claimed the *least* cost. The configured cap is
    # the honest reading, and it is what every other path here compares against.
    if clamped["cost_ceiling_usd"] <= 0:
        clamped["cost_ceiling_usd"] = max_cost

    if rate > 0:
        affordable_s = int((clamped["cost_ceiling_usd"] / rate) * 3600)
        if affordable_s < MIN_SMOKE_WALL_S:
            raise GateRefusal(
                "smoke_too_expensive",
                f"at ${rate:.2f}/h, {target_name} burns the ${clamped['cost_ceiling_usd']:.2f} "
                f"smoke cap in {affordable_s}s -- less than the {MIN_SMOKE_WALL_S}s floor a "
                "one-step run needs",
                EXIT_SPEND,
                fix="use a smaller instance for the smoke step",
                detail={**clamped, "target": target_name, "rate_usd_per_hour": rate},
            )
        clamped["timeout_s"] = min(clamped["timeout_s"], affordable_s)

    clamped["rate_usd_per_hour"] = rate
    clamped["projected_cost_usd"] = round(rate * clamped["timeout_s"] / 3600.0, 4)
    return clamped
