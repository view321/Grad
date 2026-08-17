"""The cross-workspace experiment archive (`core/experiments.py`).

The archive is the one store here that outlives a workspace, so the properties
worth testing are the ones that make it trustworthy afterwards: that a run gets
in exactly once per state it reaches, that judgement lands, that an artifact
which changed underneath it is *detectable*, and that a failure to write it
cannot take a collected run down with it.
"""

from __future__ import annotations

import argparse

import pytest

from core import experiments, ledger_store as ls, submit
from core.errors import GradError, NotFound
from core.submission import hash_resolved
from tools import experiments as experiments_cli

RESOLVED = {"schema": 1, "entrypoint": "train.py", "argv": ["--steps", "10"], "config": {}}


def _expectation(quantity="val_loss", low=2.9, high=3.2):
    return ls.append_expectation(
        {
            "id": ls.new_id("exp"), "task": "widths", "created_at": ls.now_iso(),
            "project": "proj-a", "quantity": quantity,
            "claim": "val loss should land in range",
            "predicted": {"low": low, "high": high, "direction": None},
            "basis": [], "comparability": "differs", "confidence": "medium",
        }
    )


def _submit(expectation=None, *, project="proj-a", resolved=RESOLVED, smoke=False):
    run_id = ls.new_id("smoke" if smoke else "run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "widths",
            "status": "in_flight", "smoke": smoke, "submitted_at": ls.now_iso(),
            "project": project, "platform": "kaggle",
            "submission_hash": hash_resolved(resolved), "spec_resolved": resolved,
            "expectation_id": (expectation or {}).get("id"),
            "estimate_usd": 0.0, "estimated_duration_s": 60,
            "code_version": {"version": "0.1.0", "commit": "abc123", "dirty": False},
        }
    )
    return run_id


def _collect(run_id, expectation, results, *, artifacts=None):
    directory = submit.artifacts_dir(run_id)
    for name, text in (artifacts or {}).items():
        (directory / name).write_text(text, encoding="utf-8")
    return submit.finish(
        run_id, status="completed", results=results, cost_usd_actual=0.25,
        artifacts_dir=directory, expectation=expectation,
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def test_collecting_a_run_archives_it(workspace):
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.61})

    rows = list(experiments.experiments().values())
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert row["results"] == {"val_loss": 3.61}
    assert row["cost_usd_actual"] == 0.25
    # The prediction is copied in full, not referenced: an expectation id means
    # nothing once the workspace holding expectations.jsonl is gone.
    assert row["expectation"]["id"] == exp["id"]
    assert row["all_judged"] is False


def test_a_verdict_re_archives_rather_than_patching(workspace):
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.61})
    ls.append_run_event(
        {
            "type": ls.T_VERDICT, "id": run_id, "quantity": "val_loss",
            "verdict": "bug", "note": "lr off by one", "judged_at": ls.now_iso(),
        }
    )
    experiments.archive(run_id)

    # Two snapshots on disk, one experiment after folding: the newest wins, and
    # the history of what was believed when is kept for free.
    assert len(experiments.events()) == 2
    folded = experiments.experiments()
    assert len(folded) == 1
    row = next(iter(folded.values()))
    assert row["all_judged"] is True
    assert row["deviations"][0]["verdict"] == "bug"


def test_abandoning_a_run_archives_it_too(workspace):
    """`abandon` goes through `finish`, which is the one place a run becomes
    terminal -- so a written-off run is a fact the archive keeps."""
    run_id = _submit(None)
    submit.abandon(run_id, reason="submitter killed before the job id came back")
    row = next(iter(experiments.experiments().values()))
    assert row["run_id"] == run_id
    assert row["status"] == "abandoned"


def test_an_unwritable_archive_does_not_fail_the_collection(workspace, monkeypatch):
    """The expensive thing has already happened and the ledger already has it.

    An archive that could take a `collect` down would make the durable copy more
    dangerous than not having one.
    """
    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(experiments, "archive", boom)
    exp = _expectation()
    run_id = _submit(exp)
    record = _collect(run_id, exp, {"val_loss": 3.0})

    assert record["status"] == "completed"
    assert ls.run(run_id).collected is True


# ---------------------------------------------------------------------------
# artifacts and integrity
# ---------------------------------------------------------------------------
def test_artifacts_are_hashed_by_reference_and_drift_is_detected(workspace):
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.0}, artifacts={"metrics.json": '{"val_loss": 3.0}'})

    row = next(iter(experiments.experiments().values()))
    assert [a["relative"] for a in row["artifacts"]] == ["metrics.json"]
    assert row["artifacts"][0]["sha256"]
    assert experiments.verify()["ok"] is True

    # The file the archive points at is edited after the fact. A report citing a
    # figure from this run is no longer showing what was measured.
    (submit.artifacts_dir(run_id) / "metrics.json").write_text('{"val_loss": 1.0}', encoding="utf-8")
    result = experiments.verify()
    assert result["ok"] is False
    assert result["findings"][0]["kind"] == "artifact_changed"

    (submit.artifacts_dir(run_id) / "metrics.json").unlink()
    assert experiments.verify()["findings"][0]["kind"] == "artifact_missing"


def test_a_resolved_spec_that_does_not_hash_to_its_record_is_a_finding(workspace):
    """The archive's one self-contained integrity check.

    Nothing else here can be verified without the workspace; this can, because
    the resolved document is the pre-image of the hash stored beside it.
    """
    exp = _expectation()
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "widths",
            "status": "in_flight", "smoke": False, "submitted_at": ls.now_iso(),
            "project": "proj-a", "platform": "kaggle",
            "submission_hash": "deadbeefdeadbeef",  # not the hash of RESOLVED
            "spec_resolved": RESOLVED, "expectation_id": exp["id"],
            "estimate_usd": 0.0, "estimated_duration_s": 60,
        }
    )
    _collect(run_id, exp, {"val_loss": 3.0})

    result = experiments.verify()
    assert result["ok"] is False
    assert result["findings"][0]["kind"] == "spec_hash_mismatch"


def test_a_large_artifact_is_recorded_without_a_digest(workspace, cfg, monkeypatch):
    """Hashing a checkpoint on the collect path would be a visible stall; the
    size alone still catches truncation."""
    monkeypatch.setattr(experiments, "_hash_limit", lambda _cfg=None: 8)
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.0}, artifacts={"big.bin": "x" * 64})

    entry = next(iter(experiments.experiments().values()))["artifacts"][0]
    assert entry["sha256"] is None
    assert entry["bytes"] == 64
    assert "hash_max_bytes" in entry["skipped"]
    # Nothing to re-hash, so nothing to report.
    assert experiments.verify()["ok"] is True


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def test_search_filters_and_get_resolves_a_run_id(workspace):
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.0})
    smoke_id = _submit(None, smoke=True)
    _collect(smoke_id, None, {})

    assert len(experiments.search()) == 2
    assert len(experiments.search(include_smoke=False)) == 1
    assert len(experiments.search(quantity="val_loss")) == 1
    assert len(experiments.search(project="proj-a")) == 2
    assert len(experiments.search(project="nope")) == 0
    assert experiments.get(run_id)["run_id"] == run_id
    with pytest.raises(NotFound):
        experiments.get("run-does-not-exist")


def test_the_index_is_rebuildable_and_queryable(workspace):
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.61, "note": "text"})

    counts = experiments.rebuild_index()
    assert counts["experiments"] == 1
    assert counts["metrics"] == 2

    rows = experiments.query_index("SELECT quantity, value, text_value FROM metrics ORDER BY quantity")
    assert rows[0] == {"quantity": "note", "value": None, "text_value": "text"}
    assert rows[1] == {"quantity": "val_loss", "value": 3.61, "text_value": None}
    # Tri-state preserved: this deviation is out of range, not undecidable.
    assert experiments.query_index("SELECT in_range FROM deviations")[0]["in_range"] == 0


def test_backfill_archives_terminal_runs_and_skips_in_flight_ones(workspace):
    exp = _expectation()
    collected = _submit(exp)
    _collect(collected, exp, {"val_loss": 3.0})
    in_flight = _submit(None)

    # Wipe the archive so the backfill has something to do.
    experiments.archive_path().unlink()
    payload = experiments_cli.cmd_archive(argparse.Namespace(run_id=[], all=True, json=True))
    assert len(payload["archived"]) == 1
    assert in_flight not in str(payload["archived"])

    payload = experiments_cli.cmd_archive(
        argparse.Namespace(run_id=[in_flight], all=False, json=True)
    )
    assert payload["archived"] == []
    assert payload["skipped"][0]["run_id"] == in_flight


def test_verify_exits_nine_when_something_has_drifted(workspace):
    exp = _expectation()
    run_id = _submit(exp)
    _collect(run_id, exp, {"val_loss": 3.0}, artifacts={"m.json": "{}"})
    (submit.artifacts_dir(run_id) / "m.json").write_text("changed", encoding="utf-8")

    with pytest.raises(GradError) as exc:
        experiments_cli.cmd_verify(argparse.Namespace(identifier=None, json=True))
    assert exc.value.exit_code == 9
