"""The expectations ledger and the collect contract (HANDOFF §7)."""

from __future__ import annotations

import pytest

from core import ledger_store as ls, submit as submit_lib
from core.errors import EXIT_USAGE
from tools import ledger as ledger_cli


def run_cli(argv: list[str]) -> int:
    return ledger_cli.cli.run(argv)


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def test_absolute_prediction_requires_comparability(workspace, capsys):
    """'A number from a paper means nothing without matching tokenizer, dataset,
    eval protocol, sequence length, and parameter count.'"""
    code = run_cli(
        ["expect", "--task", "t", "--quantity", "val_loss", "--low", "2.9", "--high", "3.2",
         "--basis", "arXiv:1|Table 3|3.05|1.3B", "--json"]
    )
    assert code == EXIT_USAGE
    assert "comparability" in capsys.readouterr().out


def test_absolute_prediction_requires_a_basis(workspace, capsys):
    code = run_cli(
        ["expect", "--task", "t", "--quantity", "val_loss", "--low", "2.9", "--high", "3.2",
         "--comparability", "same eval", "--json"]
    )
    assert code == EXIT_USAGE
    assert "basis" in capsys.readouterr().out


def test_relational_prediction_needs_neither(workspace):
    """'prefer relational expectations over absolute ones' -- they survive setup
    mismatch, so the ledger asks less of them."""
    assert run_cli(["expect", "--task", "t", "--quantity", "val_loss", "--direction", "decrease", "--json"]) == 0
    assert len(ls.expectations()) == 1


def test_expect_refuses_a_task_that_already_has_results(workspace, capsys):
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-1", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 0.0, "estimated_duration_s": 1}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-1", "status": "completed",
         "collected_at": ls.now_iso(), "results": {"val_loss": 3.0}, "deviations": []}
    )
    code = run_cli(["expect", "--task", "t", "--quantity", "val_loss", "--direction", "decrease", "--json"])
    assert code == EXIT_USAGE
    assert "after the fact" in capsys.readouterr().out


def test_low_above_high_is_a_usage_error(workspace):
    assert run_cli(
        ["expect", "--task", "t", "--quantity", "q", "--low", "5", "--high", "1",
         "--comparability", "x", "--basis", "p|l|1|c", "--json"]
    ) == EXIT_USAGE


# ---------------------------------------------------------------------------
# run folding
# ---------------------------------------------------------------------------
def test_a_run_is_the_fold_of_its_events(workspace):
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "r", "status": "in_flight", "submitted_at": ls.now_iso(),
         "estimate_usd": 5.0, "estimated_duration_s": 60}
    )
    assert ls.run("r").status == "in_flight"
    assert ls.run("r").cost_for_ceiling() == pytest.approx(5.0)

    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "r", "status": "completed", "collected_at": ls.now_iso(),
         "cost_usd_actual": 1.25, "results": {"val_loss": 3.0}, "deviations": []}
    )
    run = ls.run("r")
    assert run.collected and run.status == "completed"
    assert run.cost_for_ceiling() == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# deviations: computed mechanically, judged separately
# ---------------------------------------------------------------------------
def test_deviation_is_computed_without_a_verdict(workspace):
    """'the machine records what happened, the model interprets it, and the
    interpretation cannot overwrite the record.'"""
    expectation = {"id": "exp-1", "quantity": "val_loss", "predicted": {"low": 2.9, "high": 3.2}}
    devs = submit_lib.compute_deviations(expectation, {"val_loss": 4.1})
    assert len(devs) == 1
    assert devs[0]["in_range"] is False
    assert devs[0]["ratio"] == pytest.approx(4.1 / 3.05, rel=1e-3)
    assert "verdict" not in devs[0]


def test_in_range_result(workspace):
    expectation = {"id": "exp-1", "quantity": "val_loss", "predicted": {"low": 2.9, "high": 3.2}}
    assert submit_lib.compute_deviations(expectation, {"val_loss": 3.0})[0]["in_range"] is True


def test_missing_quantity_is_flagged_not_ignored(workspace):
    expectation = {"id": "exp-1", "quantity": "val_loss", "predicted": {"low": 1, "high": 2}}
    dev = submit_lib.compute_deviations(expectation, {"other": 1.0})[0]
    assert dev["in_range"] is False
    assert "no value" in dev["reason"]


def test_relational_prediction_needs_a_verdict(workspace):
    expectation = {"id": "exp-1", "quantity": "val_loss", "predicted": {"direction": "decrease"}}
    dev = submit_lib.compute_deviations(expectation, {"val_loss": 3.0})[0]
    assert dev["in_range"] is None


# ---------------------------------------------------------------------------
# verdicts and pending work
# ---------------------------------------------------------------------------
def _collected_run_with_deviation(workspace) -> str:
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-x", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 1.0, "estimated_duration_s": 60}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-x", "status": "completed", "collected_at": ls.now_iso(),
         "cost_usd_actual": 1.0, "results": {"val_loss": 4.1},
         "deviations": [{"quantity": "val_loss", "actual": 4.1, "in_range": False}]}
    )
    return "run-x"


def test_unjudged_deviations_are_surfaced(workspace):
    run_id = _collected_run_with_deviation(workspace)
    pending = ls.pending()
    assert [d["run_id"] for d in pending["unjudged_deviations"]] == [run_id]


def test_relational_results_stay_pending_until_judged(workspace):
    """A relational prediction has no range to test, so `in_range` is None.

    §7 prefers relational expectations precisely because they survive a setup
    mismatch, so they must not fall out of the pending list -- otherwise the
    most-preferred prediction type is the one that quietly stops being judged.
    """
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-rel", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 1.0, "estimated_duration_s": 60}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-rel", "status": "completed", "collected_at": ls.now_iso(),
         "cost_usd_actual": 1.0, "results": {"val_loss": 3.0},
         "deviations": submit_lib.compute_deviations(
             {"id": "exp-r", "quantity": "val_loss", "predicted": {"direction": "decrease"}},
             {"val_loss": 3.0},
         )}
    )
    assert ls.run("run-rel").get("deviations")[0]["in_range"] is None
    assert [d["run_id"] for d in ls.pending()["unjudged_deviations"]] == ["run-rel"]

    assert run_cli(["verdict", "run-rel", "--quantity", "val_loss", "--verdict", "real",
                    "--note", "beat the baseline on the same eval", "--json"]) == 0
    assert ls.pending()["unjudged_deviations"] == []


def test_a_missing_quantity_also_stays_pending(workspace):
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-gap", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 1.0, "estimated_duration_s": 60}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-gap", "status": "completed", "collected_at": ls.now_iso(),
         "cost_usd_actual": 1.0, "results": {"other": 1.0},
         "deviations": submit_lib.compute_deviations(
             {"id": "exp-g", "quantity": "val_loss", "predicted": {"low": 1, "high": 2}},
             {"other": 1.0},
         )}
    )
    assert [d["run_id"] for d in ls.pending()["unjudged_deviations"]] == ["run-gap"]


def test_verdict_attaches_to_the_deviation(workspace):
    run_id = _collected_run_with_deviation(workspace)
    assert run_cli(["verdict", run_id, "--quantity", "val_loss", "--verdict", "bug", "--note", "lr typo", "--json"]) == 0
    dev = ls.run(run_id).get("deviations")[0]
    assert dev["verdict"] == "bug" and dev["note"] == "lr typo"
    assert ls.pending()["unjudged_deviations"] == []


def test_verdict_for_an_unknown_quantity_is_rejected(workspace):
    run_id = _collected_run_with_deviation(workspace)
    assert run_cli(["verdict", run_id, "--quantity", "nope", "--verdict", "real", "--json"]) == 3


def test_falsified_expectations_are_marked_not_deleted(workspace):
    run_cli(["expect", "--task", "t", "--quantity", "q", "--direction", "decrease", "--json"])
    exp_id = ls.expectations()[0]["id"]
    assert run_cli(["falsify", exp_id, "--note", "the baseline was misconfigured", "--json"]) == 0
    assert exp_id in ls.falsified_ids()
    assert len(ls.expectations()) == 1  # still there, still readable


# ---------------------------------------------------------------------------
# derived index
# ---------------------------------------------------------------------------
def test_sqlite_index_is_rebuildable_from_the_jsonl(workspace):
    run_cli(["expect", "--task", "t", "--quantity", "q", "--direction", "decrease", "--json"])
    _collected_run_with_deviation(workspace)
    counts = ls.rebuild_index()
    assert counts == {"expectations": 1, "runs": 1}
    rows = ls.query_index("SELECT quantity, in_range FROM deviations")
    assert rows == [{"quantity": "val_loss", "in_range": 0}]


def test_verify_reports_dangling_references(workspace, capsys):
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-y", "status": "in_flight", "submitted_at": ls.now_iso(),
         "expectation_id": "exp-missing", "estimate_usd": 0.0, "estimated_duration_s": 1}
    )
    assert run_cli(["verify", "--json"]) == 9
    assert "dangling" in capsys.readouterr().out
