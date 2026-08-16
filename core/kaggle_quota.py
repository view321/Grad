"""The weekly accelerator-hour allowance, and the gate that refuses against it.

Kaggle is the first backend here whose scarce resource is not money. Every
kernel run costs $0.00, so §6's per-job and rolling dollar ceilings pass every
Kaggle submission unconditionally -- they are not wrong, they are simply not
measuring anything on this backend. The README's rule still applies:

    "Anything that spends money, destroys work, or must be true before the fact
     is enforced mechanically, not by prompt."

Running out of GPU hours mid-week destroys work -- Kaggle kills the kernel and
whatever was not checkpointed is gone -- so the allowance is enforced here, in
the same shape as `ledger_store.rolling_spend`: **actual hours for collected
runs, estimated hours for in-flight ones.** Without the second half, N runs
pushed before any is collected would all pass a 30-hour ceiling on 0 hours
counted, which is precisely the failure `cost_for_ceiling` exists to prevent on
the dollar side.

Two ceilings, because they fail differently:

  * the **weekly** allowance, which is a pool shared by every run in the window;
  * the **session** cap, which one run can blow on its own. Kaggle stops a
    kernel at 12 hours (9 on TPU) and returns what it has. A 20-hour training
    run does not get 20 hours, it gets 12 hours and a dead kernel, and the only
    cheap place to say so is before the push.

The numbers are a proxy, not a mirror. Kaggle varies the allowance with demand
and exposes no remaining balance, so `[kaggle.quota]` is a ceiling you control
in the same sense `[quota]`'s token ceiling is -- set it at or under the real
one and it binds first, which is the useful direction to be wrong in.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

from core import ledger_store as ls
from core.config import Config
from core.errors import EXIT_QUOTA, GateRefusal

PLATFORM = "kaggle"

#: What a run record carries so this module can fold it. Written by
#: `tools/kaggle.py` at submit (the estimate) and at collect (the actual).
F_ACCELERATOR = "accelerator"
F_KIND = "accelerator_kind"
F_ESTIMATE = "accelerator_hours_estimate"
F_ACTUAL = "accelerator_hours_actual"

#: `cpu` draws from no pool, so it has no ceiling to check. Named rather than
#: absent: a CPU-only kernel is a real thing to run and must not be refused as
#: "unknown accelerator".
UNMETERED = "cpu"


def _finite_hours(value: Any, what: str) -> float:
    """A number of hours that can actually be compared against a ceiling.

    The same argument `gates._finite` makes, at the other gate. The estimate
    reaches here from a spec's `[estimate]`, which is a file the agent writes,
    and `nan` and `inf` are both valid TOML floats. NaN fails every comparison,
    so `if projected > allowance` waves it through and the ceiling is disabled by
    the one input that looks most like a number; inf raises OverflowError on the
    way into a round(). Neither may reach the comparison.
    """
    try:
        hours = float(value)
    except (TypeError, ValueError):
        raise GateRefusal(
            "quota_value_invalid",
            f"{what} is not a number ({value!r})",
            EXIT_QUOTA,
            fix="give it a finite number of hours in the spec's [estimate] section",
        ) from None
    if not math.isfinite(hours):
        raise GateRefusal(
            "quota_value_invalid",
            f"{what} is {hours}, and a duration that is not a finite number cannot be "
            "counted against an allowance",
            EXIT_QUOTA,
            fix="give it a finite number of hours; nan and inf are valid TOML floats and neither bounds anything",
        )
    if hours < 0:
        raise GateRefusal(
            "quota_value_invalid",
            f"{what} is negative ({hours}h)",
            EXIT_QUOTA,
            fix="a negative duration would credit hours back into the weekly pool",
        )
    return hours


def hours_for_quota(run: ls.Run) -> float:
    """Actual once collected, estimate while in flight.

    Deliberately the same shape as `Run.cost_for_ceiling`, and for the same
    reason it gives: a kernel that has not been collected yet is not free of
    quota. A run whose record predates this backend, or that never recorded an
    estimate, folds to 0.0 -- it cannot be counted, and inventing a number for it
    would be worse than reporting the gap.
    """
    if run.collected and run.get(F_ACTUAL) is not None:
        try:
            return max(0.0, float(run.get(F_ACTUAL)))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, float(run.get(F_ESTIMATE) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def accelerator_hours(
    *, window_days: int = 7, now: _dt.datetime | None = None, kind: str | None = None
) -> dict[str, Any]:
    """Rolling accelerator hours, folded from the runs ledger.

    Mirrors `ledger_store.rolling_spend` down to the key names, because these are
    the same computation over a different unit and a reader who knows one should
    not have to learn the other.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(days=window_days)
    pools: dict[str, dict[str, Any]] = {}
    for r in ls.runs():
        if r.get("platform") != PLATFORM:
            continue
        run_kind = str(r.get(F_KIND) or "")
        if not run_kind or run_kind == UNMETERED:
            continue
        if kind and run_kind != kind:
            continue
        submitted = ls.parse_iso(r.get("submitted_at"))
        if submitted and submitted < cutoff:
            continue
        hours = hours_for_quota(r)
        pool = pools.setdefault(
            run_kind,
            {"kind": run_kind, "actual_hours": 0.0, "in_flight_hours": 0.0, "runs": []},
        )
        if r.collected and r.get(F_ACTUAL) is not None:
            pool["actual_hours"] += hours
            basis = "actual"
        else:
            pool["in_flight_hours"] += hours
            basis = "estimate"
        pool["runs"].append(
            {
                "run_id": r.id,
                "hours": round(hours, 4),
                "basis": basis,
                "accelerator": r.get(F_ACCELERATOR),
                "smoke": r.is_smoke,
            }
        )
    for pool in pools.values():
        pool["total_hours"] = round(pool["actual_hours"] + pool["in_flight_hours"], 4)
        pool["actual_hours"] = round(pool["actual_hours"], 4)
        pool["in_flight_hours"] = round(pool["in_flight_hours"], 4)
    return {"window_days": window_days, "pools": pools}


def allowance(cfg: Config, kind: str) -> float | None:
    """The weekly ceiling for one pool, or None where the pool is unmetered."""
    if kind == UNMETERED:
        return None
    quota = cfg.get("kaggle", "quota", {}) or {}
    key = {"gpu": "gpu_hours_per_week", "tpu": "tpu_hours_per_week"}.get(kind)
    if not key:
        return None
    value = quota.get(key)
    return None if value is None else float(value)


def session_cap(cfg: Config, kind: str) -> float | None:
    """The single-run ceiling. TPU sessions are shorter than GPU ones."""
    quota = cfg.get("kaggle", "quota", {}) or {}
    if kind == "tpu":
        return float(quota.get("max_tpu_session_hours", 9.0))
    if kind == "gpu":
        return float(quota.get("max_session_hours", 12.0))
    return None


def check_session(cfg: Config, kind: str, hours: float, *, accelerator: str) -> dict[str, Any] | None:
    """Refuse a single run that cannot finish inside one Kaggle session.

    This one is not about the pool. Kaggle stops the kernel at the cap and hands
    back whatever it wrote, so a run estimated past it is not a run that costs
    too much -- it is a run that has already been decided to fail, and letting it
    push spends the hours anyway.
    """
    cap = session_cap(cfg, kind)
    if cap is None:
        return None
    hours = _finite_hours(hours, "the spec's estimated duration")
    if hours > cap:
        raise GateRefusal(
            "quota_session",
            (
                f"this run estimates {hours:.2f}h on {accelerator}, past the {cap:.1f}h "
                f"limit Kaggle enforces on a single {kind.upper()} session -- the kernel "
                "would be stopped mid-run and only what it had checkpointed would survive"
            ),
            EXIT_QUOTA,
            fix=(
                "shorten the run, checkpoint and resume across several submissions, or raise "
                "kaggle.quota.max_session_hours in config/grad.toml if Kaggle's limit has moved"
            ),
            detail={"estimate_hours": hours, "session_cap_hours": cap, "kind": kind},
        )
    return {"estimate_hours": hours, "session_cap_hours": cap}


def check_quota(
    cfg: Config, kind: str, hours: float, *, accelerator: str, now: _dt.datetime | None = None
) -> dict[str, Any] | None:
    """Refuse a run that would take the weekly pool past its allowance.

    Returns the projection when it passes, so a caller can report "you will have
    N hours left after this" rather than only "you had N before it".
    """
    ceiling = allowance(cfg, kind)
    if ceiling is None:
        return None
    hours = _finite_hours(hours, "the spec's estimated duration")
    window = int((cfg.get("kaggle", "quota", {}) or {}).get("window_days", 7))
    rolling = accelerator_hours(window_days=window, now=now, kind=kind)
    pool = rolling["pools"].get(kind, {"total_hours": 0.0, "actual_hours": 0.0, "in_flight_hours": 0.0})
    projected = pool["total_hours"] + hours
    if projected > ceiling:
        raise GateRefusal(
            "quota_weekly",
            (
                f"projected {window}-day {kind.upper()} use {projected:.2f}h "
                f"({pool['actual_hours']:.2f}h collected + {pool['in_flight_hours']:.2f}h in flight "
                f"+ {hours:.2f}h this run) exceeds the {ceiling:.1f}h weekly allowance"
            ),
            EXIT_QUOTA,
            fix=(
                f"python -m tools.kaggle quota --json   # see what is holding the hours; collect "
                "in-flight runs so their estimates become actuals, wait for the window to roll, "
                f"or raise kaggle.quota.{kind}_hours_per_week in config/grad.toml"
            ),
            detail={
                "kind": kind,
                "accelerator": accelerator,
                "projected_hours": round(projected, 4),
                "allowance_hours": ceiling,
                "pool": pool,
                "window_days": window,
            },
        )
    return {
        "kind": kind,
        "window_days": window,
        "allowance_hours": ceiling,
        "used_hours": pool["total_hours"],
        "this_run_hours": round(hours, 4),
        "projected_hours": round(projected, 4),
        "projected_remaining_hours": round(ceiling - projected, 4),
    }


def check(
    cfg: Config, kind: str, hours: float, *, accelerator: str, now: _dt.datetime | None = None
) -> dict[str, Any]:
    """Both ceilings, session first.

    Order matters for the message: a 20-hour run on a 30-hour weekly allowance
    fails both, and "Kaggle will stop this at 12h" is the more useful of the two
    things to hear -- it is about this run and it is true regardless of how much
    of the week is left.
    """
    session = check_session(cfg, kind, hours, accelerator=accelerator)
    weekly = check_quota(cfg, kind, hours, accelerator=accelerator, now=now)
    return {
        "accelerator": accelerator,
        "kind": kind,
        "metered": kind != UNMETERED and weekly is not None,
        "session": session,
        "weekly": weekly,
    }


def summary(cfg: Config, *, now: _dt.datetime | None = None) -> dict[str, Any]:
    """Every pool, its allowance and what is left. What `kaggle quota` prints."""
    window = int((cfg.get("kaggle", "quota", {}) or {}).get("window_days", 7))
    rolling = accelerator_hours(window_days=window, now=now)
    out: dict[str, Any] = {"window_days": window, "pools": {}}
    for kind in ("gpu", "tpu"):
        ceiling = allowance(cfg, kind)
        pool = rolling["pools"].get(
            kind, {"kind": kind, "actual_hours": 0.0, "in_flight_hours": 0.0, "total_hours": 0.0, "runs": []}
        )
        out["pools"][kind] = {
            **pool,
            "allowance_hours": ceiling,
            "remaining_hours": None if ceiling is None else round(ceiling - pool["total_hours"], 4),
            "session_cap_hours": session_cap(cfg, kind),
        }
    out["basis"] = (
        "actual hours for collected runs, estimated hours for in-flight ones -- an uncollected "
        "kernel is not free of quota"
    )
    out["caveat"] = (
        "Kaggle varies the real allowance with demand and exposes no remaining balance, so these "
        "are ceilings you control, not a mirror of Kaggle's"
    )
    return out
