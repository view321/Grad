"""Properties of the spend and quota ceilings.

These are the gates the project describes as standing between the agent and a
$40 mistake, and the failure they exist to prevent is not "a gate raised the
wrong exception". It is "a sequence of individually reasonable submissions added
up to a number nobody chose". That is a property of a *history*, not of a call,
which is why the tests here build a ledger a record at a time and then ask the
same question after every one.

Three invariants, and each of them has a plausible way to be false that no
single example would show:

  * spend is monotone -- submitting a run never lowers the rolling total, so N
    jobs submitted before any is collected cannot all pass a check that each of
    them individually passes;
  * a collection does not change what a run cost the ceiling by more than the
    difference between its estimate and its actual, so "collect to free up
    headroom" cannot be a way to spend twice;
  * the gate and the report agree -- whatever `status` says is over is exactly
    what `over_budget` names and exactly what `check` refuses.

The ledger is real rather than mocked, which the repository's own conftest goes
out of its way to make possible: "a mock of a gate proves nothing about the
gate".
"""

from __future__ import annotations

import datetime as _dt
import itertools

import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

import hooks
from core import budget, config, gates, kaggle_quota, quota_log
from core import ledger_store as ls
from core.errors import GateRefusal

_ids = itertools.count()

#: Dollar figures at the scale this project actually works in. A run costing
#: 1e300 tests float64, not the ceiling.
usd = st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)
hours = st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)


def _submit(estimate_usd: float, *, project: str = "unassigned", smoke: bool = False) -> str:
    """One run, in flight, in the ledger. Returns its id.

    Real records through the real append path, as §24 asks for: "the budget
    gates deserve the same treatment §6's gates got: tested against a real
    ledger, not mocks, because they are what stands between a loop and a bill."
    """
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "status": "in_flight",
            "submitted_at": ls.now_iso(),
            "project": project,
            "estimate_usd": estimate_usd,
            "smoke": smoke,
            "platform": "hf-jobs",
        }
    )
    return run_id


def _collect(run_id: str, actual_usd: float) -> None:
    ls.append_run_event(
        {
            "type": ls.T_RUN_COLLECTED,
            "id": run_id,
            "status": "completed",
            "collected_at": ls.now_iso(),
            "cost_usd_actual": actual_usd,
            "results": {},
            "deviations": [],
        }
    )


def _submit_kaggle(hours: float) -> None:
    """One metered Kaggle run, in flight. The hours ledger's own unit."""
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": ls.new_id("run"),
            "status": "in_flight",
            "submitted_at": ls.now_iso(),
            "platform": kaggle_quota.PLATFORM,
            kaggle_quota.F_KIND: "gpu",
            kaggle_quota.F_ACCELERATOR: "P100",
            kaggle_quota.F_ESTIMATE: hours,
        }
    )


# ---------------------------------------------------------------------------
# the rolling total
# ---------------------------------------------------------------------------
@given(st.lists(usd, min_size=1, max_size=6))
def test_spend_never_falls_as_runs_are_submitted(
    fresh_workspace, estimates: list[float]
) -> None:
    """The invariant the whole in-flight accounting exists for.

    "A job that has not been collected yet is not free. Without this, N jobs
    submitted before any is collected all pass the ceiling check." That is a
    statement about a sequence, and it is only ever true or false of one.
    """
    fresh_workspace()
    seen = 0.0
    for estimate in estimates:
        _submit(estimate)
        total = ls.rolling_spend(30)["total_usd"]
        note({"added": estimate, "total": total})
        assert total >= seen - 1e-6
        seen = total
    assert seen == pytest.approx(sum(estimates), abs=1e-3)


@given(st.lists(st.tuples(usd, usd), min_size=1, max_size=5))
def test_collecting_a_run_replaces_its_estimate_and_nothing_else(
    fresh_workspace, pairs: list[tuple[float, float]]
) -> None:
    """An actual supersedes its own estimate, not somebody else's.

    A collection that subtracted the estimate but added the actual to a
    different pool -- or that left both -- would make "collect in-flight runs so
    their estimates become actuals", the fix every spend refusal prints, either
    a no-op or a way to double-count.
    """
    fresh_workspace()
    ids = [_submit(estimate) for estimate, _ in pairs]
    assert ls.rolling_spend(30)["total_usd"] == pytest.approx(
        sum(e for e, _ in pairs), abs=1e-3
    )
    for run_id, (_, actual) in zip(ids, pairs):
        _collect(run_id, actual)
    rolling = ls.rolling_spend(30)
    note(rolling)
    assert rolling["total_usd"] == pytest.approx(sum(a for _, a in pairs), abs=1e-3)
    assert rolling["in_flight_usd"] == pytest.approx(0.0, abs=1e-3)
    assert rolling["actual_usd"] == pytest.approx(rolling["total_usd"], abs=1e-3)


@given(st.lists(usd, max_size=5), st.integers(min_value=1, max_value=90))
def test_the_window_is_the_only_thing_that_drops_a_run(
    fresh_workspace, estimates: list[float], window: int
) -> None:
    """Everything inside the window counts, and the split adds up.

    `total_usd`, `actual_usd` and `in_flight_usd` are rounded separately, so
    they are allowed to disagree in the last place and nowhere else -- a gap
    bigger than that is a run counted in the total and in neither half.
    """
    fresh_workspace()
    for estimate in estimates:
        _submit(estimate)
    rolling = ls.rolling_spend(window)
    note(rolling)
    assert rolling["total_usd"] == pytest.approx(
        rolling["actual_usd"] + rolling["in_flight_usd"], abs=1e-3
    )
    assert len(rolling["runs"]) == len(estimates)


@given(st.lists(usd, max_size=4))
def test_a_run_older_than_the_window_is_outside_it(
    fresh_workspace, estimates: list[float]
) -> None:
    """The window is measured from `now`, and `now` is injectable for this test.

    Without the injection this property could only be checked by waiting a
    month, which is why `check_spend` and `rolling_spend` both take it.
    """
    fresh_workspace()
    for estimate in estimates:
        _submit(estimate)
    future = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=400)
    assert ls.rolling_spend(30, now=future)["total_usd"] == 0.0
    assert ls.rolling_spend(1000, now=future)["total_usd"] == pytest.approx(
        sum(estimates), abs=1e-3
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@given(st.lists(usd, max_size=4), usd)
def test_the_monthly_ceiling_is_never_crossed_quietly(
    fresh_workspace, history: list[float], estimate: float
) -> None:
    """Either the projected total is inside the ceiling, or the gate raised.

    Stated as a disjunction on purpose: it does not matter which branch a given
    example takes, only that no example takes neither. That is the shape of
    every ceiling claim in this project and the shape a single example cannot
    have.
    """
    fresh_workspace()
    cfg = config.load(reload=True)
    for spent in history:
        _submit(spent)
    monthly = float(cfg.get("spend", "monthly_usd", 200.0))
    per_job = float(cfg.get("spend", "per_job_usd", 25.0))
    before = ls.rolling_spend(int(cfg.get("spend", "window_days", 30)))["total_usd"]
    note({"before": before, "estimate": estimate, "monthly": monthly})
    try:
        result = gates.check_spend(estimate, cfg)
    except GateRefusal as refusal:
        assert refusal.code in ("spend_per_job", "spend_monthly")
        assert estimate > per_job or before + estimate > monthly
    else:
        assert estimate <= per_job
        assert result["projected_usd"] <= monthly


@given(usd)
def test_a_refusal_names_a_command_that_could_change_the_answer(
    fresh_workspace, estimate: float
) -> None:
    """"Errors carry the next command", from CONTRIBUTING, held to mechanically.

    A ceiling that refuses without a route forward is the kind that gets argued
    around, and the argument is usually correct.
    """
    fresh_workspace()
    cfg = config.load(reload=True)
    assume(estimate > float(cfg.get("spend", "per_job_usd", 25.0)))
    with pytest.raises(GateRefusal) as caught:
        gates.check_spend(estimate, cfg)
    assert caught.value.fix
    assert caught.value.detail


# ---------------------------------------------------------------------------
# the project allocation
# ---------------------------------------------------------------------------
@given(st.lists(usd, max_size=4), st.floats(min_value=1.0, max_value=400.0))
def test_over_budget_names_exactly_what_status_says_is_over(
    fresh_workspace, estimates: list[float], ceiling: float
) -> None:
    """Two readers of one fact, which must not be able to disagree.

    `budget status` prints one and the hook in `hooks.py` acts on the other, so
    a discrepancy is a project the meter calls fine and the agent cannot spend
    in -- or worse, the reverse.
    """
    fresh_workspace()
    budget.create("proj-a", title="a", budget={"gpu_usd": ceiling})
    for estimate in estimates:
        _submit(estimate, project="proj-a")
    state = budget.status("proj-a")
    note(state["resources"]["gpu_usd"])
    expected = [
        name
        for name, node in state["resources"].items()
        if node["ceiling"] is not None and node["spent"] > float(node["ceiling"])
    ]
    assert budget.over_budget("proj-a") == expected


@given(st.lists(usd, max_size=4), st.floats(min_value=1.0, max_value=400.0), usd)
def test_the_project_gate_refuses_exactly_when_the_projection_crosses(
    fresh_workspace, estimates: list[float], ceiling: float, proposed: float
) -> None:
    """`check` and `status` are the same arithmetic, or the gate is decoration.

    Exit 12 rather than 6 is the whole point of this gate existing separately,
    so it has to fire on the project's own numbers rather than on the machine's.
    """
    fresh_workspace()
    budget.create("proj-b", title="b", budget={"gpu_usd": ceiling})
    for estimate in estimates:
        _submit(estimate, project="proj-b")
    spent = budget.status("proj-b")["resources"]["gpu_usd"]["spent"]
    note({"spent": spent, "ceiling": ceiling, "proposed": proposed})
    try:
        budget.check("proj-b", gpu_usd=proposed, what="a job")
    except GateRefusal:
        assert spent + proposed > ceiling
    else:
        assert spent + proposed <= ceiling


@given(usd)
def test_a_project_with_no_ceiling_is_tracked_and_not_bounded(
    fresh_workspace, proposed: float
) -> None:
    """An unbudgeted project is not an overrun.

    The distinction matters because `over_budget` is consulted for *every*
    cost-bearing command, including in a workspace where nobody has set a
    budget at all -- and a gate that refused there would make the feature
    mandatory by accident.
    """
    fresh_workspace()
    budget.create("proj-c", title="c", budget={})
    assert budget.over_budget("proj-c") == []
    assert budget.check("proj-c", gpu_usd=proposed) is not None
    assert budget.over_budget(None) == []
    assert budget.over_budget("no-such-project") == []


# ---------------------------------------------------------------------------
# Kaggle hours, which the dollar ceilings cannot see
# ---------------------------------------------------------------------------
@given(hours, hours)
def test_a_run_is_refused_by_the_session_cap_or_fits_in_one_session(
    fresh_workspace, estimate: float, cap: float
) -> None:
    """Kaggle stops the kernel at the cap, so a longer run has already failed.

    The gate is about that, not about cost: the hours are spent either way and
    only what was checkpointed survives.
    """
    assume(cap > 0)
    root = fresh_workspace()
    # Through the config file rather than by poking the dataclass, because the
    # dataclass is not where the value comes from: `Config.get` reads the
    # overlay and the project overrides ahead of the file, and a test that set
    # the attribute directly would pass while the real precedence was broken.
    path = root / "config" / "grad.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[kaggle.quota]\nmax_session_hours = {cap}\n", encoding="utf-8"
    )
    cfg = config.load(reload=True)
    try:
        result = kaggle_quota.check_session(cfg, "gpu", estimate, accelerator="P100")
    except GateRefusal as refusal:
        assert estimate > cap
        assert refusal.fix
    else:
        assert estimate <= cap
        assert result["estimate_hours"] == pytest.approx(estimate)


@given(st.lists(hours, max_size=4))
def test_accelerator_hours_never_falls_as_runs_are_submitted(
    fresh_workspace, estimates: list[float]
) -> None:
    """The weekly allowance has the same monotonicity claim as the dollars.

    It is a separate implementation over a different unit -- `hours_for_quota`
    beside `cost_for_ceiling` -- so it needs its own statement of the property
    rather than inheriting the one above.
    """
    fresh_workspace()
    seen = 0.0
    for i, estimate in enumerate(estimates):
        _submit_kaggle(estimate)
        pools = kaggle_quota.accelerator_hours(window_days=7)["pools"]
        total = pools.get("gpu", {}).get("total_hours", 0.0)
        note({"i": i, "added": estimate, "total": total})
        assert total >= seen - 1e-6
        seen = total


@given(hours)
def test_hours_are_never_negative_however_the_record_reads(
    fresh_workspace, estimate: float
) -> None:
    """A malformed record folds to zero rather than to a credit.

    A negative estimate in a ledger is damage; treating it as *refunding* the
    weekly allowance would make the damage spendable.
    """
    fresh_workspace()
    for value in (-estimate, "not a number", None, float("nan")):
        run = ls.Run(
            "run-x",
            {
                "platform": kaggle_quota.PLATFORM,
                kaggle_quota.F_KIND: "gpu",
                kaggle_quota.F_ESTIMATE: value,
            },
        )
        folded = kaggle_quota.hours_for_quota(run)
        note({"value": value, "folded": folded})
        assert folded >= 0.0


# ---------------------------------------------------------------------------
# the turn-boundary warning
# ---------------------------------------------------------------------------
# `hooks.budget_warning` is what the Stop hook prints between turns, and it is
# where mutation testing found the largest hole in this file: seventy-eight
# mutants, none of them covered by any test at all. It is not an enforcement
# point -- the Stop hook's `block` forces continuation rather than halting, so
# enforcement lives in `agent.py` and `pre_tool_use` -- which is precisely why
# nothing noticed. A warning nobody tests is a warning that can quietly stop
# appearing, and the thing it warns about is money.
@given(st.floats(min_value=0.0, max_value=2.0), st.floats(min_value=1.0, max_value=100.0))
def test_the_warning_appears_exactly_when_a_threshold_is_crossed(
    fresh_workspace, ratio: float, ceiling: float
) -> None:
    """One rule, checked at every fraction rather than at three chosen ones.

    `WARN_AT` is `(0.75, 0.9, 1.0)`, and the claim is that the warning appears
    if and only if the spend has reached the lowest of them -- so the boundary
    is checked from both sides at every step, which is the part a
    three-example test cannot do.
    """
    fresh_workspace()
    budget.create("warn", title="w", budget={"gpu_usd": ceiling})
    budget.set_current("warn")
    _submit(ratio * ceiling, project="warn")

    # Read off the meter rather than recomputed from `ratio`. `spend` rounds to
    # four places on the way out, so `ratio = 1.0` can arrive as a fraction of
    # 0.99995 -- and a test that reconstructs the arithmetic instead of reading
    # the number under test is asserting its own copy of the code. That the
    # meter and the gate agree is `test_over_budget_names_exactly_what_status_
    # says_is_over`'s job; this one is about the warning agreeing with the meter.
    fraction = budget.status("warn")["resources"]["gpu_usd"]["fraction"]
    warning = hooks.budget_warning()
    note({"ratio": ratio, "fraction": fraction, "warning": warning})

    if fraction >= min(hooks.WARN_AT):
        assert warning is not None
        assert warning["threshold"] == max(t for t in hooks.WARN_AT if fraction >= t)
        assert warning["project"] == "warn"
        # The message is what a person reads at a turn boundary, and the only
        # part of it that is a claim rather than a formatting choice is whether
        # it says the ceiling has been passed.
        assert ("now denied" in warning["message"]) == (warning["fraction"] >= 1.0)
    else:
        assert warning is None


@given(st.floats(min_value=0.8, max_value=3.0), st.floats(min_value=0.0, max_value=0.7))
def test_the_warning_names_the_resource_nearest_its_ceiling(
    fresh_workspace, hot: float, cold: float
) -> None:
    """One line, for the resource that matters.

    "Reports the *highest* threshold crossed rather than one line per resource:
    a turn boundary is a bad place for a wall of text." Which of two resources
    is nearest its ceiling is a comparison, and a comparison written the wrong
    way round still produces a plausible-looking warning about the wrong thing.
    """
    fresh_workspace()
    budget.create("two", title="t", budget={"gpu_usd": 100.0, "credits_usd": 100.0})
    budget.set_current("two")
    _submit(hot * 100.0, project="two")
    quota_log.record({"project": "two", "credits_usd": cold * 100.0})

    warning = hooks.budget_warning()
    note({"hot": hot, "cold": cold, "warning": warning})
    assert warning is not None
    assert warning["resource"] == "gpu_usd"


@given(usd)
def test_no_project_means_no_warning(fresh_workspace, spent: float) -> None:
    """Nothing selected, nothing budgeted, nothing to say.

    The Stop hook runs after every turn in every workspace, including one where
    nobody has made a project -- so this is the common case rather than an edge
    one, and a warning here would be noise on every turn forever.
    """
    fresh_workspace()
    _submit(spent)
    assert hooks.budget_warning() is None
    budget.create("unbudgeted", title="u", budget={})
    budget.set_current("unbudgeted")
    assert hooks.budget_warning() is None
