"""Replication: `core/stats.py`, and what it changes about a deviation.

Two halves. The arithmetic is checked against values worked by hand, because a
statistics module that is only tested against itself is a module that is
consistent and wrong. The ledger half is checked against a real run record, for
the reason the gate tests are: this is what stands between a lucky seed and a
published number.
"""

from __future__ import annotations

import json
import math

import pytest

from core import ledger_store as ls, report as report_lib, stats, submit


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------
def test_a_single_sample_has_no_spread_and_says_so():
    """0.0 would claim a precision that was never measured."""
    summary = stats.summarise([3.0])
    assert summary["n"] == 1
    assert summary["mean"] == 3.0
    assert summary["sd"] is None
    assert summary["sem"] is None
    assert summary["ci95"] is None


def test_the_summary_matches_values_worked_by_hand():
    # [2, 4, 6]: mean 4, sample sd 2, sem 2/sqrt(3), t(2) = 4.303.
    summary = stats.summarise([2.0, 4.0, 6.0])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(4.0)
    assert summary["sd"] == pytest.approx(2.0)
    assert summary["sem"] == pytest.approx(2.0 / math.sqrt(3))
    half = 4.303 * 2.0 / math.sqrt(3)
    assert summary["ci95"][0] == pytest.approx(4.0 - half, rel=1e-3)
    assert summary["ci95"][1] == pytest.approx(4.0 + half, rel=1e-3)


def test_the_sample_standard_deviation_is_used_not_the_population_one():
    """n - 1, because the population form understates exactly the small n this
    is written for."""
    summary = stats.summarise([1.0, 2.0, 3.0, 4.0])
    assert summary["sd"] == pytest.approx(math.sqrt(5.0 / 3.0))


def test_junk_samples_are_excluded_rather_than_averaged():
    assert stats.numeric([1.0, "x", None, True, float("nan"), 2.0]) == [1.0, 2.0]
    # A quantity whose every sample is unusable has nothing to summarise.
    assert stats.summarise(["a", "b"])["n"] == 0


def test_the_t_table_falls_back_to_the_normal_value_past_thirty():
    assert stats.t95(1) == 12.706
    assert stats.t95(30) == 2.042
    assert stats.t95(31) == pytest.approx(1.96)
    assert stats.t95(500) == pytest.approx(1.96)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def test_an_interval_inside_the_prediction_passes():
    summary = stats.summarise([3.00, 3.02, 3.01])
    verdict = stats.compare(summary, 2.9, 3.2)
    assert verdict["in_range"] is True
    assert verdict["relation"] == stats.CONTAINED


def test_a_disjoint_interval_fails():
    summary = stats.summarise([4.00, 4.02, 4.01])
    verdict = stats.compare(summary, 2.9, 3.2)
    assert verdict["in_range"] is False
    assert verdict["relation"] == stats.DISJOINT


def test_an_overlapping_interval_is_undecidable_rather_than_a_pass_or_a_fail():
    """The outcome that did not exist before replication did.

    A noisy run whose interval straddles the edge of the prediction has neither
    confirmed nor refuted it, and saying either would be the whole problem.
    """
    summary = stats.summarise([3.1, 3.5, 2.9])
    verdict = stats.compare(summary, 2.9, 3.2)
    assert verdict["in_range"] is None
    assert verdict["relation"] == stats.OVERLAPPING
    assert "neither confirms nor refutes" in verdict["reason"]


def test_a_one_sided_prediction_uses_an_open_end():
    summary = stats.summarise([0.90, 0.92, 0.91])
    assert stats.compare(summary, 0.8, None)["in_range"] is True
    assert stats.compare(summary, 0.95, None)["in_range"] is False


def test_a_single_sample_still_compares_as_a_point():
    """n = 1 is a degenerate interval, so the old behaviour is preserved exactly
    -- what changes is that the record now says it was one sample."""
    assert stats.compare(stats.summarise([3.0]), 2.9, 3.2)["in_range"] is True
    assert stats.compare(stats.summarise([3.9]), 2.9, 3.2)["in_range"] is False


# ---------------------------------------------------------------------------
# reading the metrics artifact
# ---------------------------------------------------------------------------
def test_repeated_quantities_are_samples_rather_than_overwrites(workspace):
    """The bug this feature is built on top of.

    A pipeline emitting one record per seed had all but the last silently
    dropped -- the run looked like a single measurement and nothing said that
    two thirds of the evidence was gone.
    """
    path = workspace / "metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"seed": s, "quantity": "val_loss", "value": v})
            for s, v in enumerate([3.0, 3.1, 3.2])
        ),
        encoding="utf-8",
    )
    samples = submit.parse_samples(path)
    assert samples["val_loss"] == [3.0, 3.1, 3.2]
    # `seed` labels the record; it is not a measurement and must not become one.
    assert "seed" not in samples


def test_provenance_keys_are_excluded_from_both_artifact_shapes(workspace):
    """The rule is about the key, not the file format -- one pipeline must not
    measure different things depending on which shape it wrote."""
    as_jsonl = workspace / "a.jsonl"
    as_jsonl.write_text('{"seed": 1, "fold": 2, "val_loss": 3.0}\n', encoding="utf-8")
    as_json = workspace / "a.json"
    as_json.write_text('{"seed": 1, "fold": 2, "val_loss": 3.0}', encoding="utf-8")

    for path in (as_jsonl, as_json):
        assert set(submit.parse_samples(path)) == {"val_loss"}


def test_read_metrics_publishes_the_mean_only_when_there_is_replication(workspace):
    replicated = workspace / "m.jsonl"
    replicated.write_text(
        "\n".join(json.dumps({"quantity": "loss", "value": v}) for v in [2.0, 4.0]),
        encoding="utf-8",
    )
    results, samples = submit.read_metrics(replicated)
    assert results["loss"] == pytest.approx(3.0)
    assert samples["loss"] == [2.0, 4.0]

    # A single reading is recorded exactly as written, int and all: turning it
    # into a float would change the shape of every existing record.
    single = workspace / "one.json"
    single.write_text(json.dumps({"steps": 5, "note": "ok"}), encoding="utf-8")
    results, samples = submit.read_metrics(single)
    assert results == {"steps": 5, "note": "ok"}
    assert samples == {"steps": [5], "note": ["ok"]}


def test_a_non_numeric_quantity_keeps_its_last_value(workspace):
    path = workspace / "m.jsonl"
    path.write_text(
        '{"quantity": "device", "value": "T4"}\n{"quantity": "device", "value": "A100"}\n',
        encoding="utf-8",
    )
    results, _ = submit.read_metrics(path)
    assert results["device"] == "A100"


# ---------------------------------------------------------------------------
# the deviation
# ---------------------------------------------------------------------------
def _expectation(low=2.9, high=3.2):
    return {"id": "exp-1", "quantity": "val_loss", "predicted": {"low": low, "high": high}}


def test_a_deviation_records_the_summary_and_the_relation():
    dev = submit.compute_deviations(
        _expectation(), {"val_loss": 3.01}, {"val_loss": [3.00, 3.02, 3.01]}
    )[0]
    assert dev["in_range"] is True
    assert dev["relation"] == stats.CONTAINED
    assert dev["stats"]["n"] == 3
    assert dev["stats"]["sd"] is not None


def test_an_unreplicated_deviation_is_unchanged_except_for_n():
    """Every existing caller keeps its behaviour and gains only the count."""
    dev = submit.compute_deviations(_expectation(), {"val_loss": 3.0})[0]
    assert dev["in_range"] is True
    assert dev["stats"]["n"] == 1
    assert dev["ratio"] == pytest.approx(3.0 / 3.05, rel=1e-3)


def test_an_overlapping_deviation_lands_in_the_pending_list(workspace):
    """`unjudged_deviations` filters on `is not True`, so the new tri-state
    reaches the verdict queue with no change to the fold."""
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "t", "status": "in_flight",
            "smoke": False, "submitted_at": ls.now_iso(), "project": "p",
            "expectation_id": "exp-1", "estimate_usd": 0.0,
        }
    )
    submit.finish(
        run_id, status="completed", results={"val_loss": 3.17},
        samples={"val_loss": [3.1, 3.5, 2.9]},
        cost_usd_actual=0.0, artifacts_dir=submit.artifacts_dir(run_id),
        expectation=_expectation(),
    )
    run = ls.run(run_id)
    assert len(run.unjudged_deviations()) == 1
    assert run.get("samples") == {"val_loss": [3.1, 3.5, 2.9]}


def test_samples_are_only_recorded_when_there_is_replication(workspace):
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "t", "status": "in_flight",
            "smoke": False, "submitted_at": ls.now_iso(), "project": "p",
            "estimate_usd": 0.0,
        }
    )
    submit.finish(
        run_id, status="completed", results={"val_loss": 3.0},
        samples={"val_loss": [3.0]},
        cost_usd_actual=0.0, artifacts_dir=submit.artifacts_dir(run_id),
        expectation=None,
    )
    # A block restating `results` one list at a time is noise in every record.
    assert "samples" not in ls.run(run_id).data


# ---------------------------------------------------------------------------
# rule 5
# ---------------------------------------------------------------------------
def _cited_run(workspace, *, samples, expectation=_expectation()):
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "t", "status": "in_flight",
            "smoke": False, "submitted_at": ls.now_iso(), "project": "p",
            "expectation_id": "exp-1", "estimate_usd": 0.0,
        }
    )
    submit.finish(
        run_id, status="completed",
        results={"val_loss": stats.summarise(samples)["mean"]},
        samples={"val_loss": samples}, cost_usd_actual=0.0,
        artifacts_dir=submit.artifacts_dir(run_id), expectation=expectation,
    )
    return run_id


def test_report_check_flags_a_number_published_from_one_sample(workspace):
    run_id = _cited_run(workspace, samples=[3.0])
    tex = r"our loss was \gradnum{loss}."
    claims = {"loss": {"run_id": run_id, "quantity": "val_loss"}}

    findings = report_lib.check_replication(tex, claims)
    assert len(findings) == 1
    assert findings[0]["rule"] == "replication"
    assert "single sample" in findings[0]["problem"]


def test_report_check_is_quiet_about_a_replicated_number(workspace):
    run_id = _cited_run(workspace, samples=[3.0, 3.1, 3.05])
    tex = r"our loss was \gradnum{loss}."
    claims = {"loss": {"run_id": run_id, "quantity": "val_loss"}}
    assert report_lib.check_replication(tex, claims) == []


def test_a_record_predating_replication_accounting_is_left_alone(workspace):
    """Reporting every historical claim as under-replicated would make the rule
    noise, and noise is what gets a rule routed around."""
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "t", "status": "in_flight",
            "smoke": False, "submitted_at": ls.now_iso(), "project": "p",
            "estimate_usd": 0.0,
        }
    )
    ls.append_run_event(
        {
            "type": ls.T_RUN_COLLECTED, "id": run_id, "status": "completed",
            "collected_at": ls.now_iso(), "results": {"val_loss": 3.0},
            "cost_usd_actual": 0.0, "artifacts": "",
            # A deviation with no `stats` block, as written before core/stats.py.
            "deviations": [{"quantity": "val_loss", "in_range": True, "actual": 3.0}],
        }
    )
    tex = r"our loss was \gradnum{loss}."
    claims = {"loss": {"run_id": run_id, "quantity": "val_loss"}}
    assert report_lib.check_replication(tex, claims) == []
