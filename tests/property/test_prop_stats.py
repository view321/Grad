"""Properties of the replication statistics.

`core/stats.py` decides whether a run confirmed its prediction, refuted it, or
settled nothing. That verdict is the input to `Run.unjudged_deviations` and to
the evolutionary search's selection step, so an arithmetic slip here does not
show up as a crash -- it shows up as a search that prefers a lucky seed, months
later, with no way to tell which conclusions were affected.

Statistics is unusually well suited to this kind of test, because the answers
are constrained by identities rather than by examples: a mean lies between the
extremes it was taken over, an interval straddles its centre, and shifting every
sample by a constant shifts the mean by that constant and leaves the spread
alone. None of those need a fixture, and each of them fails loudly if the
arithmetic drifts.
"""

from __future__ import annotations

import math

from hypothesis import assume, given, note
from hypothesis import strategies as st

from core import stats

#: Bounded away from the float extremes on purpose. A metric of 1e308 is not a
#: measurement this system will ever see, and generating one only tests whether
#: `fsum` overflows -- which is a question about CPython, not about this module.
measurements = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
samples = st.lists(measurements, min_size=1, max_size=12)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------
@given(samples)
def test_the_mean_lies_between_the_extremes(values: list[float]) -> None:
    """The identity that catches a summation bug for what it is.

    A `mean` outside `[min, max]` is not a rounding artefact; it is the wrong
    sum or the wrong denominator, and it would otherwise be reported as a
    measurement with a straight face.
    """
    summary = stats.summarise(values)
    note(summary)
    assert summary["min"] <= summary["mean"] <= summary["max"]


@given(samples)
def test_the_interval_is_centred_on_the_mean(values: list[float]) -> None:
    """`ci95` is symmetric about `mean`, or absent.

    Absent at n = 1 by design -- one sample has no spread, and reporting 0.0
    would claim a precision nobody measured.
    """
    summary = stats.summarise(values)
    note(summary)
    if summary["ci95"] is None:
        assert summary["n"] == 1
        return
    low, high = summary["ci95"]
    assert low <= summary["mean"] <= high
    assert math.isclose(
        summary["mean"] - low, high - summary["mean"], rel_tol=1e-9, abs_tol=1e-12
    )


@given(samples, measurements)
def test_shifting_every_sample_shifts_the_mean_and_nothing_else(
    values: list[float], shift: float
) -> None:
    """Translation equivariance: the location moves, the spread does not.

    This is the property that separates a real standard deviation from one
    computed against a fixed origin, and it is invisible to any single example
    because every example has *some* origin.
    """
    before = stats.summarise(values)
    after = stats.summarise([v + shift for v in values])
    note({"before": before, "after": after})
    assert math.isclose(after["mean"], before["mean"] + shift, rel_tol=1e-9, abs_tol=1e-6)
    if before["sd"] is not None:
        assert math.isclose(after["sd"], before["sd"], rel_tol=1e-6, abs_tol=1e-6)


@given(samples)
def test_the_order_of_the_samples_does_not_change_the_summary(
    values: list[float],
) -> None:
    """Reversing the seeds must not move the answer.

    Runs arrive in whatever order `parse_metrics` read them, which is the order
    the artifact happened to be written in. A statistic that depends on it is
    reporting a property of the file.
    """
    forward = stats.summarise(values)
    backward = stats.summarise(list(reversed(values)))
    for key in ("n", "min", "max"):
        assert forward[key] == backward[key]
    assert math.isclose(forward["mean"], backward["mean"], rel_tol=1e-9, abs_tol=1e-9)
    if forward["sd"] is not None:
        assert math.isclose(forward["sd"], backward["sd"], rel_tol=1e-9, abs_tol=1e-9)


@given(st.lists(st.one_of(st.booleans(), st.none(), st.text(max_size=3)), max_size=8))
def test_nothing_numeric_means_nothing_measured(values: list[object]) -> None:
    """Booleans, strings and None are not measurements.

    `True` is an `int` in Python, so a flag averaged into a metric produces a
    number that looks exactly like a result. `numeric` excludes them, and this
    pins the consequence rather than the implementation.
    """
    summary = stats.summarise(values)
    assert summary["n"] == 0
    assert summary["mean"] is None
    assert stats.observed_interval(summary) is None
    assert stats.compare(summary, 0.0, 1.0)["in_range"] is None


@given(samples, st.integers(min_value=0, max_value=4))
def test_a_nan_sample_is_dropped_rather_than_propagated(
    values: list[float], where: int
) -> None:
    """One NaN must not erase the other seeds.

    A run that reported NaN measured nothing, but the seeds beside it did, and
    a summary that returns NaN throughout would make the whole replication
    unreadable -- including its `min` and `max`, which are not in doubt.
    """
    poisoned = list(values)
    poisoned.insert(min(where, len(poisoned)), float("nan"))
    summary = stats.summarise(poisoned)
    note(summary)
    assert summary["n"] == len(values)
    assert not math.isnan(summary["mean"])


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
@given(samples, measurements, measurements)
def test_a_verdict_is_one_of_three_and_agrees_with_its_reason(
    values: list[float], a: float, b: float
) -> None:
    """`in_range` and `relation` are two spellings of one answer.

    They are read by different callers -- `report check` looks at the relation,
    `Run.unjudged_deviations` at the tri-state -- so a disagreement between them
    would put a run in the pending list and out of it at the same time.
    """
    low, high = min(a, b), max(a, b)
    verdict = stats.compare(stats.summarise(values), low, high)
    note(verdict)
    pairs = {
        stats.CONTAINED: True,
        stats.DISJOINT: False,
        stats.OVERLAPPING: None,
    }
    assert verdict["in_range"] is pairs[verdict["relation"]]
    assert verdict["reason"]


@given(samples, measurements, measurements)
def test_containment_and_disjointness_are_what_they_claim(
    values: list[float], a: float, b: float
) -> None:
    """The verdict checked against the interval arithmetic directly.

    `compare` is the one function here whose answer a reader cannot verify by
    eye, because it compares two intervals rather than a number to a range. So
    it is checked against the definition instead of against a table of cases.
    """
    low, high = min(a, b), max(a, b)
    summary = stats.summarise(values)
    observed = stats.observed_interval(summary)
    assume(observed is not None)
    obs_low, obs_high = observed
    verdict = stats.compare(summary, low, high)
    note({"observed": observed, "predicted": [low, high], "verdict": verdict})

    if verdict["relation"] == stats.CONTAINED:
        assert low <= obs_low and obs_high <= high
    elif verdict["relation"] == stats.DISJOINT:
        assert obs_high < low or obs_low > high
    else:
        assert not (low <= obs_low and obs_high <= high)
        assert not (obs_high < low or obs_low > high)


@given(samples, measurements)
def test_a_one_sided_prediction_leaves_the_open_end_open(
    values: list[float], bound: float
) -> None:
    """"At least 0.8" cannot be refuted from above.

    A `None` bound is infinite rather than zero, and the difference is a whole
    class of false alarm: with `low=None` read as 0.0, every negative loss would
    be reported as disjoint from its own prediction.
    """
    summary = stats.summarise(values)
    upper_only = stats.compare(summary, None, bound)
    lower_only = stats.compare(summary, bound, None)
    note({"<= bound": upper_only, ">= bound": lower_only})
    observed = stats.observed_interval(summary)
    assert observed is not None
    if observed[1] <= bound:
        assert upper_only["relation"] == stats.CONTAINED
    if observed[0] >= bound:
        assert lower_only["relation"] == stats.CONTAINED
    assert stats.compare(summary, None, None)["relation"] == stats.CONTAINED


@given(samples)
def test_an_observation_is_contained_by_its_own_interval(values: list[float]) -> None:
    """The reflexive case, which every other verdict is measured against."""
    summary = stats.summarise(values)
    low, high = stats.observed_interval(summary)
    assert stats.compare(summary, low, high)["in_range"] is True


# ---------------------------------------------------------------------------
# rounding, and the table
# ---------------------------------------------------------------------------
@given(samples)
def test_rounding_a_summary_changes_no_structure(values: list[float]) -> None:
    """`round_summary` is cosmetic, and the ledger depends on it staying so.

    The rounded figure is what a report cites *and* what the ledger holds, so a
    key dropped or a None turned into a number here is a discrepancy between a
    published number and its own provenance.
    """
    summary = stats.summarise(values)
    rounded = stats.round_summary(summary)
    assert rounded.keys() == summary.keys()
    for key, value in summary.items():
        if value is None:
            assert rounded[key] is None
        elif isinstance(value, float):
            # One unit in the last place kept, not half of one. Rounding moves a
            # value by at most half a unit, but the *rounded* value then has its
            # own representation error, and the two can add: 0.0234375 rounds to
            # a float that is 5.000000000005e-07 away from it rather than exactly
            # 5e-07. A bound of 5e-7 was a claim about decimal arithmetic in a
            # test of binary floats.
            assert abs(rounded[key] - value) < 1e-6


@given(st.integers(min_value=1, max_value=200))
def test_the_critical_value_falls_towards_the_normal_one(df: int) -> None:
    """t is always above 1.96 and never rises with more evidence.

    A transcription error in the table -- thirty hand-typed numbers -- would
    show up here as a non-monotone step, and nowhere else until an interval came
    out the wrong width.
    """
    assert stats.t95(df) >= 1.959
    if df > 1:
        assert stats.t95(df) <= stats.t95(df - 1)


@given(st.integers(max_value=0))
def test_no_degrees_of_freedom_is_not_a_number(df: int) -> None:
    assert math.isnan(stats.t95(df))
