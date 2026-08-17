"""Replication: what n runs of one configuration actually say.

The gap this closes is the one the audit called the biggest scientific weakness.
Until now an expectation bound one quantity to one interval and `collect`
compared **one number** to it, so `val_loss = 3.05` against a predicted
`[2.9, 3.2]` was recorded as in-range with identical confidence whether the
run-to-run spread was 0.001 or 0.3. For a system whose stated purpose is that
"a surprise is an alarm", that is the one place the alarm could not be
calibrated -- and it matters most in `tools/evolve.py`, because selecting on a
noisy single-sample metric is how a search learns to prefer a lucky seed.

**Where the samples come from.** A metrics artifact that reports one quantity
more than once *is* a replicated run: three `{"quantity": "val_loss", "value":
...}` records are three seeds. `core/submit.py:parse_metrics` used to keep the
last of them and silently drop the rest, which is why this needed no new
contract -- only for the existing one to stop discarding data. Declaring
`seeds = [...]` in `[config]` is optional and buys a cross-check; it lands in
the submission hash for free, because `config` is hashed and a different seed
list is a different experiment.

**Why a t interval and not a normal one.** Replication here means three to five
seeds, and at n = 3 the normal approximation understates the interval by about
a third. The critical values are a table rather than a computed inverse CDF
because `core/` runs on the standard library plus a file lock -- pulling in
SciPy for thirty numbers would be the heaviest dependency in the project, for a
quantity that is tabulated in every statistics textbook ever printed.

**Why the comparison is interval-to-interval.** A prediction is a range and a
replicated observation is a range, and the honest comparison of two ranges has
three outcomes rather than two: contained, disjoint, and overlapping. That last
is not a failure of the method, it is the answer -- the run neither confirmed
nor refuted the prediction and needs a judgement. `in_range` is already
tri-state for exactly this class of case, and `Run.unjudged_deviations` already
treats anything that is `not True` as needing a verdict, so an overlapping
result lands in the pending list where it belongs.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

#: Two-sided 95% critical values of Student's t, by degrees of freedom.
#: Beyond 30 the normal value is within a per cent and the table stops.
_T95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}
_T95_ASYMPTOTIC = 1.960

#: Outcomes of comparing an observation against a prediction. `OVERLAPPING` is
#: the one that did not exist before replication did.
CONTAINED = "contained"
DISJOINT = "disjoint"
OVERLAPPING = "overlapping"


def t95(df: int) -> float:
    """The two-sided 95% critical value for `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    return _T95.get(df, _T95_ASYMPTOTIC)


def numeric(values: Sequence[Any]) -> list[float]:
    """The usable samples: real numbers, in order.

    Booleans are excluded for the reason `_scalar` excludes them from metrics --
    `True` is an `int` in Python and averaging a flag produces a number that
    looks like a measurement. NaN is excluded because it would poison every
    statistic downstream, and a run that reported one has not measured anything.
    """
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def summarise(values: Sequence[Any]) -> dict[str, Any]:
    """n, mean, spread and a 95% interval for one quantity's samples.

    `sd`, `sem` and the interval are **None at n = 1**, and that is a statement
    rather than a gap: one sample has no spread, and reporting 0.0 would claim
    a precision that was never measured. Callers test for None; nothing here
    invents a number it does not have.
    """
    samples = numeric(values)
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "sem": None,
                "ci95": None, "min": None, "max": None, "samples": []}
    mean = math.fsum(samples) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "sem": None, "ci95": None,
                "min": samples[0], "max": samples[0], "samples": samples}
    # Sample standard deviation: n - 1 in the denominator. The population form
    # would understate the spread of exactly the small n this is written for.
    variance = math.fsum((x - mean) ** 2 for x in samples) / (n - 1)
    sd = math.sqrt(variance)
    sem = sd / math.sqrt(n)
    half = t95(n - 1) * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95": [mean - half, mean + half],
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def observed_interval(summary: dict[str, Any]) -> tuple[float, float] | None:
    """What the run measured, as an interval. None when there is nothing.

    At n = 1 the interval is the point itself, which keeps the comparison below
    a single function rather than two -- a point is a degenerate interval and
    `compare` needs no special case for it. What *does* need saying is that the
    result rests on one sample, and `compare` records `n` so a reader and
    `report check` can both see it.
    """
    if not summary or summary.get("mean") is None:
        return None
    interval = summary.get("ci95")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        return float(interval[0]), float(interval[1])
    mean = float(summary["mean"])
    return mean, mean


def compare(
    summary: dict[str, Any], low: float | None, high: float | None
) -> dict[str, Any]:
    """The observation against the prediction, as a tri-state with its reason.

    Returns `in_range`: True when the observed interval sits inside the
    prediction, False when the two are disjoint, and **None when they overlap**
    -- which no program can settle and which is therefore a verdict's business,
    exactly like a relational prediction or a non-numeric result.

    A one-sided prediction (`--low` alone, or `--high` alone) is handled by the
    open end being infinite, so "at least 0.8" against an interval that starts
    at 0.79 is overlapping rather than a pass.
    """
    interval = observed_interval(summary)
    if interval is None:
        return {"in_range": None, "relation": None,
                "reason": "the run reported no usable numeric samples"}
    obs_low, obs_high = interval
    lo = float("-inf") if low is None else float(low)
    hi = float("inf") if high is None else float(high)
    n = int(summary.get("n") or 0)
    basis = "the mean" if n == 1 else f"the 95% interval over {n} samples"

    if obs_low >= lo and obs_high <= hi:
        return {
            "in_range": True,
            "relation": CONTAINED,
            "reason": f"{basis} lies inside the prediction",
        }
    if obs_high < lo or obs_low > hi:
        return {
            "in_range": False,
            "relation": DISJOINT,
            "reason": f"{basis} does not overlap the prediction at all",
        }
    return {
        "in_range": None,
        "relation": OVERLAPPING,
        "reason": (
            f"{basis} overlaps the prediction without being contained by it -- "
            "the run neither confirms nor refutes it"
        ),
    }


def round_summary(summary: dict[str, Any], places: int = 6) -> dict[str, Any]:
    """The same summary with the floats rounded, for a ledger record.

    Rounded on the way *in* to the record rather than on the way out to a
    reader, so the number a report cites and the number the ledger holds are the
    same one. Six places is far beyond any measurement here and well inside
    float64, so this is cosmetic rather than lossy.
    """
    def _r(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, places)
        if isinstance(value, list):
            return [_r(v) for v in value]
        return value

    return {key: _r(value) for key, value in summary.items()}
