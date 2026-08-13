"""The four submit gates and the smoke carve-out (HANDOFF §6, §7).

These are the tests that matter most in this repo. Everything else can be wrong
and cost an afternoon; these can be wrong and cost a GPU bill.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core import gates, jsonl, ledger_store as ls, paths
from core.errors import (
    EXIT_EXPECTATION,
    EXIT_PREFLIGHT,
    EXIT_SPEND,
    EXIT_STALE_RUN,
    GateRefusal,
)
from core.submission import Submission


def make_submission(workspace, *, hours: float = 1.0, rate: float = 2.0) -> Submission:
    d = workspace / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("print('x')\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        f"[estimate]\nhours = {hours}\nrate_usd_per_hour = {rate}\n",
        encoding="utf-8",
    )
    return Submission.load(d / "spec.toml", resolve_digest=False)


def pass_preflight(sub: Submission, checks=("tests", "dry_run", "smoke")) -> None:
    jsonl.write_json(
        paths.preflight_record(sub.hash()),
        {
            "submission_hash": sub.hash(),
            "verified_at": ls.now_iso(),
            "checks": {name: {"ok": True} for name in checks},
        },
    )


def make_expectation(task: str = "t1") -> str:
    record = ls.append_expectation(
        {
            "id": ls.new_id("exp"),
            "task": task,
            "created_at": ls.now_iso(),
            "quantity": "val_loss",
            "predicted": {"low": 2.9, "high": 3.2, "direction": None},
            "basis": [],
            "comparability": "same setup",
            "confidence": "medium",
        }
    )
    return record["id"]


# ---------------------------------------------------------------------------
# gate 1: preflight
# ---------------------------------------------------------------------------
def test_submit_refuses_without_a_preflight_record(workspace, cfg):
    sub = make_submission(workspace)
    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, make_expectation(), cfg)
    assert exc.value.exit_code == EXIT_PREFLIGHT
    assert "preflight" in exc.value.fix


def test_submit_refuses_when_a_check_failed(workspace, cfg):
    sub = make_submission(workspace)
    jsonl.write_json(
        paths.preflight_record(sub.hash()),
        {"submission_hash": sub.hash(), "checks": {"tests": {"ok": True}, "dry_run": {"ok": False}, "smoke": {"ok": True}}},
    )
    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, make_expectation(), cfg)
    assert exc.value.exit_code == EXIT_PREFLIGHT
    assert "dry_run failed" in exc.value.message


def test_preflight_record_does_not_transfer_after_a_config_change(workspace, cfg):
    """No TTL: what invalidates a record is state change, and the hash is what
    notices state change."""
    d = workspace / "pipeline"
    sub = make_submission(workspace)
    pass_preflight(sub)
    gates.check_preflight(sub, cfg)  # passes for the original submission

    changed = Submission.load(d / "spec.toml", overrides={"lr": 0.5}, resolve_digest=False)
    with pytest.raises(GateRefusal):
        gates.check_preflight(changed, cfg)


# ---------------------------------------------------------------------------
# gate 2: expectation
# ---------------------------------------------------------------------------
def test_submit_refuses_without_an_expectation(workspace, cfg):
    sub = make_submission(workspace)
    pass_preflight(sub)
    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, None, cfg)
    assert exc.value.exit_code == EXIT_EXPECTATION


def test_submit_refuses_an_unknown_expectation(workspace, cfg):
    sub = make_submission(workspace)
    pass_preflight(sub)
    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, "exp-does-not-exist", cfg)
    assert exc.value.code == "expectation_missing"


def test_an_expectation_cannot_be_bound_twice(workspace, cfg):
    """Binding at submit time is what makes a retroactive prediction impossible:
    the run record already names the id it was submitted with."""
    sub = make_submission(workspace)
    pass_preflight(sub)
    exp = make_expectation()
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": "run-1",
            "status": "in_flight",
            "submitted_at": ls.now_iso(),
            "expectation_id": exp,
            "estimate_usd": 1.0,
            "estimated_duration_s": 60,
        }
    )
    with pytest.raises(GateRefusal) as exc:
        gates.check_expectation(exp, sub)
    assert exc.value.code == "expectation_bound"


# ---------------------------------------------------------------------------
# gate 3: spend
# ---------------------------------------------------------------------------
def test_per_job_ceiling(workspace, cfg):
    with pytest.raises(GateRefusal) as exc:
        gates.check_spend(10_000.0, cfg)
    assert exc.value.exit_code == EXIT_SPEND
    assert exc.value.code == "spend_per_job"


def test_in_flight_runs_count_at_their_estimates(workspace, cfg):
    """'a job that has not been collected yet is not free. Without this, N jobs
    submitted before any is collected all pass the ceiling check.'"""
    monthly = float(cfg.get("spend", "monthly_usd"))
    per_job = float(cfg.get("spend", "per_job_usd"))
    n = int(monthly // per_job)
    for i in range(n):
        ls.append_run_event(
            {
                "type": ls.T_RUN_SUBMITTED,
                "id": f"run-{i}",
                "status": "in_flight",
                "submitted_at": ls.now_iso(),
                "estimate_usd": per_job,
                "estimated_duration_s": 3600,
            }
        )
    assert ls.rolling_spend(30)["in_flight_usd"] == pytest.approx(monthly)
    with pytest.raises(GateRefusal) as exc:
        gates.check_spend(per_job, cfg)
    assert exc.value.code == "spend_monthly"


def test_collected_runs_count_at_their_actuals(workspace, cfg):
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": "run-a", "status": "in_flight",
            "submitted_at": ls.now_iso(), "estimate_usd": 20.0, "estimated_duration_s": 60,
        }
    )
    ls.append_run_event(
        {
            "type": ls.T_RUN_COLLECTED, "id": "run-a", "status": "completed",
            "collected_at": ls.now_iso(), "cost_usd_actual": 3.0, "results": {}, "deviations": [],
        }
    )
    rolling = ls.rolling_spend(30)
    assert rolling["actual_usd"] == pytest.approx(3.0)
    assert rolling["in_flight_usd"] == pytest.approx(0.0)


def test_spend_outside_the_window_is_not_counted(workspace, cfg):
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90)).isoformat()
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": "run-old", "status": "in_flight",
            "submitted_at": old, "estimate_usd": 500.0, "estimated_duration_s": 60,
        }
    )
    assert ls.rolling_spend(30)["total_usd"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# gate 4: stale uncollected runs
# ---------------------------------------------------------------------------
def test_a_stale_uncollected_run_blocks_new_submissions(workspace, cfg):
    """'Forgetting to collect therefore costs the ability to submit, which is the
    one currency that reliably gets noticed.'"""
    long_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": "run-stale", "status": "in_flight",
            "submitted_at": long_ago, "estimate_usd": 1.0, "estimated_duration_s": 600,
        }
    )
    with pytest.raises(GateRefusal) as exc:
        gates.check_stale(cfg)
    assert exc.value.exit_code == EXIT_STALE_RUN
    assert "collect" in exc.value.fix


def test_a_recent_uncollected_run_does_not_block(workspace, cfg):
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": "run-fresh", "status": "in_flight",
            "submitted_at": ls.now_iso(), "estimate_usd": 1.0, "estimated_duration_s": 600,
        }
    )
    gates.check_stale(cfg)  # does not raise


def test_all_four_gates_pass_together(workspace, cfg):
    sub = make_submission(workspace)
    pass_preflight(sub)
    summary = gates.check_submit(sub, make_expectation(), cfg)
    assert summary["submission_hash"] == sub.hash()
    assert summary["estimate_usd"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# the smoke carve-out
# ---------------------------------------------------------------------------
def test_smoke_caps_are_applied_not_merely_validated(workspace, cfg):
    """'nothing useful can be trained inside them'"""
    sub = make_submission(workspace)
    caps = gates.check_smoke_caps(sub, cfg, requested={"steps": 10_000, "timeout_s": 86_400, "cost_usd": 500.0})
    assert caps["steps"] == 1
    assert caps["timeout_s"] <= 600
    assert caps["cost_ceiling_usd"] <= 0.50
    assert caps["artifact_upload"] is False


def test_smoke_refuses_a_spec_whose_floor_exceeds_the_cap(workspace, cfg):
    d = workspace / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("print('x')\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        "[estimate]\nsmoke_cost_usd = 25.0\n",
        encoding="utf-8",
    )
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    with pytest.raises(GateRefusal) as exc:
        gates.check_smoke_caps(sub, cfg)
    assert exc.value.code == "smoke_too_expensive"
