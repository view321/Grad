"""Writing off a Kaggle run whose kernel was deleted through the Kaggle UI.

The dead end this closes had three walls and no door.

  * `kaggle collect` polls the kernel, gets a 404, reports "still running", and
    goes on reporting it -- because an unreadable status is not a terminal one.
  * `ledger abandon` refuses anything carrying a handle, correctly: that check
    is what stops a live job being dropped off the spend ceiling.
  * Meanwhile the run holds its estimate against the monthly ceiling and its
    accelerator hours against the weekly pool, and once past the grace window
    gate 4 refuses *every* later submission on *every* backend.

So the only exit was editing `runs.jsonl` by hand. These tests pin the door --
and, as with `test_abandon.py`, they mostly pin how narrow it is: a kernel that
still exists, or one whose status could not be read at all, must not be
write-off-able, because either may be training right now.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pytest

from core import gates, kaggle_quota, ledger_store as ls
from core.errors import EXIT_RUNNING, EXIT_STALE_RUN, EXIT_USAGE, GateRefusal, GradError
from tools import kaggle as kaggle_tool


def submitted(
    run_id: str = "run-kag",
    *,
    age_days: float = 2.0,
    hours: float = 3.0,
    accelerator: str = "gpu-t4x2",
    kernel_ref: str | None = "someone/grad-run-kag",
    **extra,
) -> str:
    """A Kaggle run recorded at submit time, with its kernel handle attached."""
    at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=age_days)
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "task": "t1",
            "status": "in_flight",
            "submitted_at": at.isoformat(),
            "platform": "kaggle",
            "estimate_usd": 0.0,
            "estimated_duration_s": int(hours * 3600),
            kaggle_quota.F_ACCELERATOR: accelerator,
            kaggle_quota.F_KIND: "gpu",
            kaggle_quota.F_ESTIMATE: hours,
            **extra,
        }
    )
    if kernel_ref:
        ls.append_run_event(
            {"type": "run_handle", "id": run_id, "handle": {"kernel_ref": kernel_ref}}
        )
    return run_id


def _args(run_id: str, reason: str = "deleted in the Kaggle UI", *, assume: bool = False):
    return argparse.Namespace(
        run_id=run_id, reason=reason, assume_deleted=assume, json=True
    )


def _status_is(monkeypatch, payload):
    monkeypatch.setattr(kaggle_tool, "_status", lambda cfg, ref: dict(payload))


# ---------------------------------------------------------------------------
# telling "gone" from "cannot say"
# ---------------------------------------------------------------------------
def test_a_404_from_the_cli_reads_as_missing_not_unknown(workspace, monkeypatch):
    """The whole mechanism rests on this distinction. `_status` used to collapse
    every failed status call into `unknown`, which is what made a deleted kernel
    indistinguishable from an unreachable network."""
    def boom(argv, cfg, timeout=None):
        raise GradError("kaggle_failed", "404 - Not Found: kernels/status", exit_code=8)

    monkeypatch.setattr(kaggle_tool, "_executable", lambda: "kaggle")
    monkeypatch.setattr(kaggle_tool, "_run", boom)
    from core import config as config_mod

    assert kaggle_tool._status(config_mod.load(), "u/k")["status"] == kaggle_tool.MISSING


def test_an_unreachable_kaggle_stays_unknown(workspace, monkeypatch):
    def boom(argv, cfg, timeout=None):
        raise GradError("kaggle_failed", "connection timed out after 120s", exit_code=8)

    monkeypatch.setattr(kaggle_tool, "_executable", lambda: "kaggle")
    monkeypatch.setattr(kaggle_tool, "_run", boom)
    from core import config as config_mod

    assert kaggle_tool._status(config_mod.load(), "u/k")["status"] == "unknown"


def test_a_real_error_status_is_not_mistaken_for_a_missing_kernel(workspace, monkeypatch):
    """A kernel that ran and failed reports `error`, and that is a *result*.
    Reading it as "gone" would write off a run whose logs are downloadable."""
    monkeypatch.setattr(kaggle_tool, "_executable", lambda: "kaggle")
    monkeypatch.setattr(
        kaggle_tool, "_run", lambda argv, cfg, timeout=None: "status: error\nmessage: boom"
    )
    from core import config as config_mod

    assert kaggle_tool._status(config_mod.load(), "u/k")["status"] == "error"


# ---------------------------------------------------------------------------
# how narrow it is
# ---------------------------------------------------------------------------
def test_a_kernel_that_still_exists_is_refused_and_sent_to_collect(workspace, monkeypatch):
    run_id = submitted()
    _status_is(monkeypatch, {"status": "running", "message": ""})

    with pytest.raises(GradError) as exc:
        kaggle_tool.cmd_forget(_args(run_id))

    assert exc.value.code == "kernel_exists"
    assert "collect" in (exc.value.fix or "")
    assert ls.run(run_id).status == "in_flight"


def test_a_complete_kernel_is_refused_because_there_is_a_result_to_fetch(workspace, monkeypatch):
    run_id = submitted()
    _status_is(monkeypatch, {"status": "complete", "message": ""})

    with pytest.raises(GradError) as exc:
        kaggle_tool.cmd_forget(_args(run_id))
    assert exc.value.code == "kernel_exists"


def test_an_unreadable_status_is_refused_because_the_job_may_be_live(workspace, monkeypatch):
    """The dangerous case. "We could not ask" is not evidence of anything, and
    writing the run off on it would drop a training job off the weekly pool."""
    run_id = submitted()
    _status_is(monkeypatch, {"status": "unknown", "message": "connection reset"})

    with pytest.raises(GradError) as exc:
        kaggle_tool.cmd_forget(_args(run_id))

    assert exc.value.code == "kernel_status_unknown"
    assert exc.value.exit_code == EXIT_RUNNING
    # Both ways out are named: wait for Kaggle, or assert the deletion yourself.
    assert "--assume-deleted" in (exc.value.fix or "")
    assert ls.run(run_id).status == "in_flight"


def test_a_run_that_never_reached_kaggle_is_sent_to_abandon(workspace, monkeypatch):
    """Two write-off commands is already one more than ideal. This one covering
    the other's ground would make "which do I use" unanswerable."""
    run_id = submitted(kernel_ref=None)

    with pytest.raises(GradError) as exc:
        kaggle_tool.cmd_forget(_args(run_id))

    assert "ledger abandon" in (exc.value.fix or "")
    assert exc.value.exit_code == EXIT_USAGE


def test_a_reason_is_required(workspace, monkeypatch):
    run_id = submitted()
    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})

    with pytest.raises(GradError):
        kaggle_tool.cmd_forget(_args(run_id, "   "))
    assert ls.run(run_id).status == "in_flight"


def test_a_collected_run_cannot_be_forgotten_twice(workspace, monkeypatch):
    run_id = submitted()
    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})
    kaggle_tool.cmd_forget(_args(run_id))

    with pytest.raises(GradError) as exc:
        kaggle_tool.cmd_forget(_args(run_id))
    assert exc.value.code == "already_collected"


# ---------------------------------------------------------------------------
# what it writes
# ---------------------------------------------------------------------------
def test_the_record_says_it_was_forgotten_why_and_on_what_evidence(workspace, monkeypatch):
    run_id = submitted()
    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404 - Not Found"})

    out = kaggle_tool.cmd_forget(_args(run_id, "deleted through the Kaggle UI"))
    record = out["run"]

    assert record["status"] == "forgotten"
    assert record["results"] == {}
    assert record["reason"] == "deleted through the Kaggle UI"
    assert record["deletion_verified"] is True
    assert "404" in record["forget_basis"]

    folded = ls.run(run_id)
    assert folded.collected
    assert folded.status == "forgotten"


def test_assume_deleted_records_that_nothing_was_verified(workspace, monkeypatch):
    """The escape from the escape. It must be legible as one: a reader has to be
    able to tell a confirmed write-off from an asserted one."""
    run_id = submitted()

    def refuse(cfg, ref):  # pragma: no cover - must not be reached
        raise AssertionError("--assume-deleted must not need the network")

    monkeypatch.setattr(kaggle_tool, "_status", refuse)
    out = kaggle_tool.cmd_forget(_args(run_id, "checked by hand", assume=True))

    assert out["deletion_verified"] is False
    assert "asserted deleted by the operator" in out["run"]["forget_basis"]
    assert ls.run(run_id).status == "forgotten"


# ---------------------------------------------------------------------------
# the block it clears -- the point of the whole thing
# ---------------------------------------------------------------------------
def test_forgetting_releases_the_weekly_accelerator_hours(workspace, monkeypatch):
    """The leak this closes. The pool reads the *estimate* until an actual is
    written, so a run finalised without one holds its hours for a week."""
    from core import config as config_mod

    run_id = submitted(hours=9.0)
    cfg = config_mod.load()
    before = kaggle_quota.summary(cfg)["pools"]["gpu"]
    assert before["in_flight_hours"] == pytest.approx(9.0)

    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})
    kaggle_tool.cmd_forget(_args(run_id))

    after = kaggle_quota.summary(config_mod.load())["pools"]["gpu"]
    assert after["in_flight_hours"] == pytest.approx(0.0)
    assert after["actual_hours"] == pytest.approx(0.0)


def test_forgetting_clears_the_stale_gate_that_was_refusing_everything(workspace, monkeypatch):
    """Gate 4 refuses every later submission on every backend while any
    uncollected run is past its window. That is the block, and this is the
    only command that can lift it for a deleted kernel."""
    from core import config as config_mod

    run_id = submitted(age_days=5.0)
    cfg = config_mod.load()
    with pytest.raises(GateRefusal) as exc:
        gates.check_stale(cfg)
    assert exc.value.exit_code == EXIT_STALE_RUN

    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})
    out = kaggle_tool.cmd_forget(_args(run_id))

    gates.check_stale(config_mod.load())  # no longer refuses
    assert out["stale_runs_remaining"] == []


def test_the_expectation_stays_bound(workspace, monkeypatch):
    """§7 has one spelling of "consumed". A retry mints a fresh prediction."""
    exp_id = ls.append_expectation(
        {
            "id": ls.new_id("exp"), "task": "t1", "created_at": ls.now_iso(),
            "quantity": "val_loss", "predicted": {"low": 1.0, "high": 2.0, "direction": None},
            "basis": [], "comparability": "same", "confidence": "medium",
        }
    )["id"]
    run_id = submitted(expectation_id=exp_id)
    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})

    out = kaggle_tool.cmd_forget(_args(run_id))

    assert out["expectation_still_bound"] == exp_id
    assert exp_id in ls.consumed_expectation_ids()
    assert ls.run(run_id).unjudged_deviations() == []


def test_a_forgotten_run_does_not_count_as_a_result_for_its_task(workspace, monkeypatch):
    run_id = submitted()
    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})
    kaggle_tool.cmd_forget(_args(run_id))
    assert "t1" not in ls.tasks_with_results()


# ---------------------------------------------------------------------------
# collect stops lying about it
# ---------------------------------------------------------------------------
def test_collect_names_forget_instead_of_saying_still_running(workspace, monkeypatch):
    """The refusal that used to be a dead end. `still_running` was advice this
    run could never take -- there is no kernel left to finish."""
    run_id = submitted()
    _status_is(monkeypatch, {"status": kaggle_tool.MISSING, "message": "404"})

    with pytest.raises(GradError) as exc:
        kaggle_tool.cmd_collect(
            argparse.Namespace(
                run_id=run_id, wait=False, timeout=0, delete_kernel=False, json=True
            )
        )

    assert exc.value.code == "kernel_missing"
    assert "kaggle forget" in (exc.value.fix or "")


def test_a_wake_on_a_forgotten_run_does_not_go_on_polling_it(workspace):
    from core import wakeups

    assert "forgotten" in wakeups.TERMINAL_RUN_STATUSES
