"""Per-project memory (`core/projects.py`, `tools/project.py`).

The generated half is tested against a real ledger in a temp workspace, the way
the gate tests are, and for the same reason: these files are what a later
session believes about this one, and a mock of the fold proves nothing about
whether `RESULTS.md` agrees with `runs.jsonl`.
"""

from __future__ import annotations

import argparse

import pytest

from core import budget, ledger_store as ls, projects, submit
from core.errors import GradError, UsageError
from tools import project as project_cli


def _project(workspace, project_id="proj-a"):
    budget.create(project_id, title="a project", budget={"gpu_usd": 10.0})
    budget.set_current(project_id)
    return project_id


def _expectation(project_id, *, task="widths", quantity="val_loss", low=2.9, high=3.2):
    return ls.append_expectation(
        {
            "id": ls.new_id("exp"),
            "task": task,
            "created_at": ls.now_iso(),
            "project": project_id,
            "quantity": quantity,
            "claim": "val loss should land in range",
            "predicted": {"low": low, "high": high, "direction": None},
            "basis": [{"paper": "arXiv:2001.08361", "locator": "Fig 3", "value": 3.05}],
            "comparability": "our tokenizer differs",
            "confidence": "medium",
        }
    )


def _run(project_id, expectation, *, results, task="widths"):
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": task, "status": "in_flight",
            "smoke": False, "submitted_at": ls.now_iso(), "project": project_id,
            "platform": "kaggle", "submission_hash": "0" * 16,
            "expectation_id": expectation["id"], "estimate_usd": 0.0,
            "estimated_duration_s": 60,
        }
    )
    submit.finish(
        run_id, status="completed", results=results, cost_usd_actual=0.0,
        artifacts_dir=submit.artifacts_dir(run_id), expectation=expectation,
    )
    return run_id


# ---------------------------------------------------------------------------
# the directory
# ---------------------------------------------------------------------------
def test_scaffold_creates_the_authored_files_and_never_overwrites(workspace):
    project_id = _project(workspace)
    first = projects.scaffold(project_id)
    assert set(first["created"]) == set(projects.AUTHORED)

    memory = projects.doc_path(project_id, "MEMORY.md")
    memory.write_text("the tokenizer is ours, not the paper's", encoding="utf-8")
    second = projects.scaffold(project_id)

    assert second["created"] == []
    assert set(second["kept"]) == set(projects.AUTHORED)
    # The whole point: a second `init` must not be able to erase notes.
    assert "tokenizer" in memory.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "  ", "."])
def test_a_project_id_cannot_escape_the_projects_directory(workspace, bad):
    """The id reaches this from `--project`, so it is checked before it is joined."""
    with pytest.raises(UsageError):
        projects.resolve_dir(bad)


# ---------------------------------------------------------------------------
# the generated half
# ---------------------------------------------------------------------------
def test_sync_renders_the_ledger_and_is_idempotent(workspace):
    project_id = _project(workspace)
    exp = _expectation(project_id)
    _run(project_id, exp, results={"val_loss": 3.61, "throughput": 1420.0})

    first = projects.sync(project_id)
    assert set(first["written"]) == set(projects.GENERATED)
    assert first["collected"] == 1
    assert first["awaiting_verdict"] == 1

    # Nobody edited anything, so a second sync must not report an edit. This
    # failed on DONE.md alone when the marker's `source` was unquoted -- it is
    # the one generated file whose provenance contains a space.
    second = projects.sync(project_id)
    assert set(second["written"]) == set(projects.GENERATED)


def test_results_reports_the_deviation_the_ledger_computed(workspace):
    project_id = _project(workspace)
    exp = _expectation(project_id)
    _run(project_id, exp, results={"val_loss": 3.61})
    projects.sync(project_id)

    body = projects.doc_path(project_id, "RESULTS.md").read_text(encoding="utf-8")
    assert "3.61" in body
    assert "**unjudged**" in body
    assert "**no**" in body  # in_range is False and is not softened


def test_done_separates_judged_work_from_the_rest(workspace):
    project_id = _project(workspace)
    exp = _expectation(project_id)
    run_id = _run(project_id, exp, results={"val_loss": 3.61})
    projects.sync(project_id)

    waiting = projects.doc_path(project_id, "DONE.md").read_text(encoding="utf-8")
    assert "**0 done" in waiting
    assert run_id in waiting.split("## Collected, awaiting a verdict")[1]

    ls.append_run_event(
        {
            "type": ls.T_VERDICT, "id": run_id, "quantity": "val_loss",
            "verdict": "bug", "note": "lr off by one", "judged_at": ls.now_iso(),
        }
    )
    projects.sync(project_id)
    done = projects.doc_path(project_id, "DONE.md").read_text(encoding="utf-8")
    assert "**1 done" in done
    assert "lr off by one" in done.split("## Established")[1]


def test_a_hand_edited_generated_file_is_refused_rather_than_overwritten(workspace):
    project_id = _project(workspace)
    projects.sync(project_id)
    path = projects.doc_path(project_id, "RESULTS.md")
    path.write_text(path.read_text(encoding="utf-8") + "\nmy own note\n", encoding="utf-8")

    with pytest.raises(GradError) as exc:
        projects.sync(project_id)
    assert "RESULTS.md" in exc.value.message
    assert "--force" in (exc.value.fix or "")
    # Refused means untouched. A refusal that had already written the file would
    # be the destruction it exists to prevent, with a message about it.
    assert "my own note" in path.read_text(encoding="utf-8")

    projects.sync(project_id, force=True)
    assert "my own note" not in path.read_text(encoding="utf-8")


def test_a_generated_file_survives_a_crlf_rewrite(workspace):
    """An editor that re-saves with Windows line endings has not edited anything.

    The digest is taken over LF-normalised text precisely so that this is not
    reported as a hand edit -- otherwise opening the file in most Windows editors
    would be enough to make `sync` refuse.
    """
    project_id = _project(workspace)
    projects.sync(project_id)
    path = projects.doc_path(project_id, "DONE.md")
    path.write_bytes(path.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    assert projects._hand_edited(path) is False


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def test_an_expectation_is_attributed_by_its_own_project_field(workspace):
    project_id = _project(workspace)
    _project(workspace, "proj-b")
    budget.set_current(project_id)
    _expectation(project_id)

    assert len(projects.state(project_id)["expectations"]) == 1
    assert projects.state("proj-b")["expectations"] == []


def test_an_expectation_without_a_project_falls_back_to_its_binding_run(workspace):
    """Records written before the field existed still belong somewhere.

    The only other evidence of which project an old expectation served is the
    run that bound it, so that is what is used -- and an unbound one from that
    era is genuinely unattributable and is left out rather than guessed.
    """
    project_id = _project(workspace)
    legacy = ls.append_expectation(
        {
            "id": ls.new_id("exp"), "task": "old", "created_at": ls.now_iso(),
            "quantity": "val_loss", "claim": "...",
            "predicted": {"low": 1.0, "high": 2.0, "direction": None},
            "basis": [], "comparability": "", "confidence": "low",
        }
    )
    orphan = ls.append_expectation(
        {
            "id": ls.new_id("exp"), "task": "older", "created_at": ls.now_iso(),
            "quantity": "val_loss", "claim": "...",
            "predicted": {"low": 1.0, "high": 2.0, "direction": None},
            "basis": [], "comparability": "", "confidence": "low",
        }
    )
    _run(project_id, legacy, results={"val_loss": 1.5}, task="old")

    ids = [e["id"] for e in projects.state(project_id)["expectations"]]
    assert legacy["id"] in ids
    assert orphan["id"] not in ids


# ---------------------------------------------------------------------------
# what the agent is given
# ---------------------------------------------------------------------------
def test_memory_is_truncated_at_a_line_boundary_and_says_so(workspace):
    project_id = _project(workspace)
    projects.scaffold(project_id)
    path = projects.doc_path(project_id, "MEMORY.md")
    path.write_text("\n".join(f"fact number {i}" for i in range(2000)), encoding="utf-8")

    text = projects.memory_text(project_id, max_chars=500)
    assert len(text) < 900
    assert "more characters" in text
    # The path, so the agent can go and read the rest rather than assume there
    # is no rest.
    assert "projects/proj-a/MEMORY.md" in text


def test_memory_is_absent_rather_than_fatal_when_there_is_no_project(workspace):
    assert projects.memory_text(None) == ""
    assert projects.prompt_block(None) == ""
    # A project that was never scaffolded is a session without memory, not a
    # session that fails to start.
    assert projects.memory_text("never-created") == ""


def test_the_prompt_block_names_the_other_files_without_including_them(workspace):
    project_id = _project(workspace)
    projects.scaffold(project_id)
    projects.sync(project_id)
    projects.doc_path(project_id, "MEMORY.md").write_text("we use bf16", encoding="utf-8")

    block = projects.prompt_block(project_id)
    assert "we use bf16" in block
    for name in ("PLAN.md", "TODO.md", "RESULTS.md", "DONE.md", "EXPECTATIONS.md"):
        assert name in block
    # Named, not inlined: RESULTS.md's own body must not be in the prompt.
    assert "Generated from `ledger/runs.jsonl`" not in block


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------
def test_init_scaffolds_and_generates_in_one_command(workspace):
    project_id = _project(workspace)
    payload = project_cli.cmd_init(
        argparse.Namespace(project=project_id, json=True, force=False)
    )
    assert set(payload["created"]) == set(projects.AUTHORED)
    assert set(payload["generated"]) == set(projects.GENERATED)
    for name in projects.DOCS:
        assert projects.doc_path(project_id, name).is_file()


def test_the_cli_refuses_a_project_that_does_not_exist(workspace):
    _project(workspace)
    with pytest.raises(GradError):
        project_cli.cmd_sync(argparse.Namespace(project="no-such-project", force=False, json=True))
