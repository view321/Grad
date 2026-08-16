"""Abandoning a run that never reached hf/kaggle/gpu (HANDOFF §6, gate 4).

Every submitter writes its in-flight record before the network call and its
handle after it. A submitter killed in between -- Ctrl-C, a dead agent process,
an exception past the handler -- leaves a run that

  * holds its estimate against the monthly ceiling and its accelerator hours
    against the weekly pool,
  * cannot be collected, because there is no job id to poll, and
  * goes stale, at which point gate 4 refuses every later submission on every
    backend.

The three together were a dead end with no exit but hand-editing `runs.jsonl`.
These tests pin the exit, and -- more importantly -- pin how narrow it is: a run
that *did* reach a platform is still spending, and must not be abandonable.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core import gates, kaggle_quota, ledger_store as ls, submit as submit_lib
from core.errors import EXIT_STALE_RUN, EXIT_USAGE, GateRefusal, GradError
from tools import ledger as ledger_tool


def strand(
    run_id: str = "run-stranded",
    *,
    age_days: float = 2.0,
    estimate_usd: float = 4.0,
    platform: str = "hf_jobs",
    handle: dict | None = None,
    **extra,
) -> str:
    """A run recorded at submit time whose submitter never came back."""
    submitted = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=age_days)
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "task": "t1",
            "status": "in_flight",
            "submitted_at": submitted.isoformat(),
            "platform": platform,
            "estimate_usd": estimate_usd,
            "estimated_duration_s": 600,
            **extra,
        }
    )
    if handle:
        ls.append_run_event({"type": "run_handle", "id": run_id, "handle": handle})
    return run_id


# ---------------------------------------------------------------------------
# the block it clears
# ---------------------------------------------------------------------------
def test_a_stranded_run_blocks_every_backend_until_it_is_abandoned(workspace, cfg):
    """The whole point, end to end: refused before, allowed after."""
    run_id = strand()
    with pytest.raises(GateRefusal) as exc:
        gates.check_stale(cfg)
    assert exc.value.exit_code == EXIT_STALE_RUN

    submit_lib.abandon(run_id, reason="the submitter was killed before the job id came back")

    gates.check_stale(cfg)  # does not raise
    assert ls.stale_runs(cfg=cfg) == []
    assert ls.in_flight() == []


def test_abandoning_gives_the_estimate_back_to_the_ceiling(workspace):
    """An in-flight run is counted at its estimate. A run that never started
    must stop being counted at all, or the ceiling shrinks permanently by the
    cost of every submitter that was interrupted."""
    run_id = strand(estimate_usd=40.0)
    assert ls.rolling_spend(30)["total_usd"] == pytest.approx(40.0)

    submit_lib.abandon(run_id, reason="never reached HF")

    spend = ls.rolling_spend(30)
    assert spend["total_usd"] == pytest.approx(0.0)
    assert spend["in_flight_usd"] == pytest.approx(0.0)


def test_abandoning_gives_kaggle_its_accelerator_hours_back(workspace):
    """The weekly pool reads the estimate until an actual is written, so a run
    finalised without one holds its hours for a week -- the same leak as the
    dollar ceiling, in the unit that is actually scarce on Kaggle."""
    run_id = strand(
        platform="kaggle",
        estimate_usd=0.0,
        **{
            kaggle_quota.F_KIND: "gpu",
            kaggle_quota.F_ACCELERATOR: "nvidiaTeslaT4",
            kaggle_quota.F_ESTIMATE: 9.0,
        },
    )
    assert kaggle_quota.accelerator_hours()["pools"]["gpu"]["total_hours"] == pytest.approx(9.0)

    record = submit_lib.abandon(run_id, reason="the push was interrupted")

    assert record[kaggle_quota.F_ACTUAL] == 0.0
    assert kaggle_quota.accelerator_hours()["pools"]["gpu"]["total_hours"] == pytest.approx(0.0)


def test_a_run_with_no_hours_gets_no_invented_actual(workspace):
    """The zeroing keys off what the record is holding, not off the platform.
    An HF run has no hours to give back and must not grow a Kaggle field."""
    record = submit_lib.abandon(strand(), reason="killed")
    assert kaggle_quota.F_ACTUAL not in record


# ---------------------------------------------------------------------------
# how narrow it is
# ---------------------------------------------------------------------------
def test_a_run_that_reached_a_backend_cannot_be_abandoned(workspace):
    """The bypass this must not become.

    A handle means the platform accepted the job. That job may be running and
    billing right now, and letting the agent write it off at $0 to clear its own
    gate is precisely what §6 exists to stop.
    """
    run_id = strand(handle={"job_id": "job-1", "namespace": "myorg"})
    with pytest.raises(GradError) as exc:
        submit_lib.abandon(run_id, reason="I would like to submit something else")
    assert exc.value.code == "run_reached_backend"
    assert exc.value.exit_code == EXIT_USAGE
    # And it names that backend's collector, not a generic pointer.
    assert exc.value.fix == f"python -m tools.jobs collect {run_id} --json"

    assert ls.run(run_id).status == "in_flight"
    assert ls.rolling_spend(30)["total_usd"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("hf_jobs", "python -m tools.jobs collect run-stranded --json"),
        ("kaggle", "python -m tools.kaggle collect run-stranded --json"),
        ("ssh", "python -m tools.gpu collect run-stranded --json"),
        ("something-new", "python -m tools.ledger show run-stranded --json"),
    ],
)
def test_the_refusal_names_the_right_collector(workspace, platform, expected):
    """A run that has to be collected elsewhere is worth one correct command.
    An unknown platform degrades to the ledger rather than to a wrong one."""
    run_id = strand(platform=platform, handle={"job_id": "j"})
    with pytest.raises(GradError) as exc:
        submit_lib.abandon(run_id, reason="x")
    assert exc.value.fix == expected


def test_an_already_collected_run_cannot_be_abandoned(workspace):
    run_id = strand()
    submit_lib.abandon(run_id, reason="killed")
    with pytest.raises(GradError) as exc:
        submit_lib.abandon(run_id, reason="killed again")
    assert exc.value.code == "already_collected"


def test_a_reason_is_required(workspace):
    """Recorded, not printed: a run that leaves the ledger without a result is
    the one record a later session cannot reconstruct from anything else."""
    run_id = strand()
    with pytest.raises(GradError) as exc:
        submit_lib.abandon(run_id, reason="   ")
    assert exc.value.code == "reason_required"
    assert ls.run(run_id).status == "in_flight"


# ---------------------------------------------------------------------------
# what it writes
# ---------------------------------------------------------------------------
def test_the_record_says_it_was_abandoned_why_and_on_what_basis(workspace):
    run_id = strand()
    record = submit_lib.abandon(run_id, reason="agent process was killed mid-submit")

    assert record["status"] == "abandoned"
    assert record["results"] == {}
    assert record["cost_usd_actual"] == 0.0
    assert record["reason"] == "agent process was killed mid-submit"
    # $0 is a choice made without a witness, so the record carries the basis
    # rather than leaving a reader to infer it from the zero.
    assert "no handle was ever recorded" in record["cost_basis"]

    folded = ls.run(run_id)
    assert folded.collected
    assert folded.status == "abandoned"


def test_abandoning_writes_no_deviation_to_judge(workspace):
    """The prediction was never tested. A deviation here would put a row in the
    unjudged list that no verdict can honestly settle."""
    exp_id = ls.append_expectation(
        {
            "id": ls.new_id("exp"),
            "task": "t1",
            "created_at": ls.now_iso(),
            "quantity": "val_loss",
            "predicted": {"low": 2.9, "high": 3.2, "direction": None},
            "basis": [],
            "comparability": "same setup",
            "confidence": "medium",
        }
    )["id"]
    run_id = strand(expectation_id=exp_id)

    submit_lib.abandon(run_id, reason="killed")

    assert ls.run(run_id).unjudged_deviations() == []
    assert ls.pending()["unjudged_deviations"] == []


def test_the_expectation_stays_bound(workspace):
    """§7 has one spelling of "consumed" and every binding site depends on it
    meaning the same thing. A retry mints a new prediction; it does not reuse
    one the ledger already recorded as spent."""
    exp_id = ls.append_expectation(
        {
            "id": ls.new_id("exp"), "task": "t1", "created_at": ls.now_iso(),
            "quantity": "val_loss", "predicted": {"low": 1.0, "high": 2.0, "direction": None},
            "basis": [], "comparability": "same", "confidence": "medium",
        }
    )["id"]
    run_id = strand(expectation_id=exp_id)

    submit_lib.abandon(run_id, reason="killed")

    assert exp_id in ls.consumed_expectation_ids()


def test_an_abandoned_run_does_not_count_as_a_result_for_its_task(workspace):
    """`ledger expect` refuses a task that already has a collected run. An
    abandoned one produced nothing, so it must not close the task off."""
    run_id = strand()
    submit_lib.abandon(run_id, reason="killed")
    assert "t1" not in ls.tasks_with_results()


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------
def test_the_cli_reports_what_is_still_blocking(workspace):
    """One refusal per submission is the failure to avoid: the agent should
    learn about the second stranded run now, not on its next attempt."""
    first = strand("run-a")
    strand("run-b")

    out = ledger_tool.cmd_abandon(_args(first, "killed"))

    assert out["run"]["status"] == "abandoned"
    assert out["stale_runs_remaining"] == ["run-b"]
    assert "abandon run-b" in out["next"]


def test_the_cli_points_at_a_fresh_expectation_once_nothing_is_blocking(workspace):
    out = ledger_tool.cmd_abandon(_args(strand(), "killed"))
    assert out["stale_runs_remaining"] == []
    assert "ledger expect" in out["next"]


def test_the_cli_reports_the_binding_it_did_not_release(workspace):
    exp_id = ls.append_expectation(
        {
            "id": ls.new_id("exp"), "task": "t1", "created_at": ls.now_iso(),
            "quantity": "val_loss", "predicted": {"low": 1.0, "high": 2.0, "direction": None},
            "basis": [], "comparability": "same", "confidence": "medium",
        }
    )["id"]
    out = ledger_tool.cmd_abandon(_args(strand(expectation_id=exp_id), "killed"))
    assert out["expectation_still_bound"] == exp_id


def test_the_cli_refuses_an_unknown_run(workspace):
    with pytest.raises(GradError) as exc:
        ledger_tool.cmd_abandon(_args("run-nope", "killed"))
    assert exc.value.code == "not_found"


def _args(run_id: str, reason: str):
    import argparse

    return argparse.Namespace(run_id=run_id, reason=reason)
