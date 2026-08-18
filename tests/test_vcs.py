"""Versioning the workspace: what it refuses, and when it commits.

The ledger is append-only, so the thing this protects is not the run records --
it is `notes/`, a project's `MEMORY.md`, the pipeline code and a report's
`.tex`, all of which are ordinary files that an ordinary mistake can truncate.

Two refusals carry most of the value and both are tested against a real
repository rather than a mock: versioning the *installation* would put research
on the same branch as upstream's releases, and adopting a repository somebody
else made would mean committing to a history that is not ours.
"""

from __future__ import annotations

import subprocess

import pytest

from core import paths, vcs, version


def git_available() -> bool:
    return version.git("--version", cwd=paths.root()) is not None


pytestmark = pytest.mark.skipif(not git_available(), reason="git is not on PATH")


def initialised(workspace):
    result = vcs.initialise()
    assert result["error"] is None, result
    return result


# ---------------------------------------------------------------------------
# what it refuses
# ---------------------------------------------------------------------------
def test_it_refuses_to_version_the_installation(workspace, monkeypatch):
    """The default workspace *is* the checkout, and Grad's source is already
    versioned there. Auto-committing would put a user's notebooks on the same
    branch as upstream's releases -- which is what makes an update a merge, and
    is the whole reason `workspace move` exists."""
    monkeypatch.setattr(paths, "install_dir", lambda: paths.root())

    result = vcs.initialise()
    assert result["created"] is False
    assert "installation folder" in result["error"]
    assert "workspace move" in result["fix"]
    assert not (paths.root() / ".git").exists()


def test_it_refuses_to_adopt_somebody_elses_repository(workspace):
    """A workspace can already be inside somebody's own versioning. Quietly
    taking it over would mean committing to a history that is not ours."""
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=True)

    result = vcs.initialise()
    assert result["created"] is False
    assert "did not create" in result["error"]
    assert vcs.enabled() is False


def test_a_workspace_inside_a_repository_is_not_a_repository(workspace, monkeypatch):
    """`rev-parse --git-dir` answers yes from any subdirectory, so the check has
    to be that the workspace is the *top*. Committing from inside somebody's
    repository would sweep up whatever else it holds."""
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=True)
    inner = workspace / "inner"
    inner.mkdir()
    monkeypatch.setenv("GRAD_ROOT", str(inner))

    assert vcs.is_repository() is False


def test_an_unversioned_workspace_checkpoints_nothing(workspace):
    """Initialisation is deliberate, so everything before it is a no-op rather
    than an implicit `git init` in somebody's folder."""
    result = vcs.checkpoint("a run was collected")
    assert result["committed"] is False
    assert result["error"] is None
    assert not (workspace / ".git").exists()


# ---------------------------------------------------------------------------
# what it does
# ---------------------------------------------------------------------------
def test_initialising_marks_the_repository_and_commits(workspace):
    result = initialised(workspace)

    assert result["created"] is True
    assert vcs.enabled() is True
    assert (workspace / ".gitignore").is_file()
    assert vcs.status()["commits"] == 1


def test_initialising_twice_is_not_an_error(workspace):
    initialised(workspace)
    again = vcs.initialise()
    assert again["already"] is True
    assert again["error"] is None


def test_a_checkpoint_commits_what_changed(workspace):
    initialised(workspace)
    (workspace / "notes").mkdir(exist_ok=True)
    (workspace / "notes" / "finding.md").write_text("the lr schedule was off by one", "utf-8")

    result = vcs.checkpoint("verdict bug on run-1")
    assert result["committed"] is True
    assert result["commit"]
    assert vcs.status()["dirty"] == []

    subjects = [entry["subject"] for entry in vcs.history()]
    assert any("verdict bug on run-1" in s for s in subjects)


def test_a_checkpoint_with_nothing_to_commit_is_quiet(workspace):
    """Called after every collect, so the common case is "nothing changed" and
    it must not produce an empty commit per run."""
    initialised(workspace)
    first = vcs.checkpoint("nothing happened")
    assert first["committed"] is False
    assert first["error"] is None
    assert vcs.status()["commits"] == 1


def test_the_message_names_the_project(workspace):
    """The log is read to answer what an afternoon established, and the project
    is half of that answer."""
    from core import budget

    budget.create("proj-x", title="width vs depth", budget={"gpu_usd": 10.0})
    budget.set_current("proj-x")
    initialised(workspace)
    (workspace / "notes").mkdir(exist_ok=True)
    (workspace / "notes" / "a.md").write_text("x", "utf-8")

    vcs.checkpoint("collected run-9 (succeeded)")
    assert vcs.history()[0]["subject"] == "proj-x: collected run-9 (succeeded)"


# ---------------------------------------------------------------------------
# what it tracks
# ---------------------------------------------------------------------------
def test_the_jsonl_ledgers_are_tracked_and_the_indexes_are_not(workspace):
    """The JSONL ledgers are the source of truth and diff line by line, which is
    the whole reason this repository is worth having. The SQLite files beside
    them are derived from those ledgers and would only ever be conflicts."""
    initialised(workspace)
    (workspace / "ledger" / "runs.jsonl").write_text('{"id": "run-1"}\n', encoding="utf-8")
    (workspace / "ledger" / "ledger.sqlite").write_bytes(b"\x00binary")
    vcs.checkpoint("a run")

    tracked = version.git("ls-files", cwd=workspace) or ""
    assert "ledger/runs.jsonl" in tracked
    assert "ledger.sqlite" not in tracked


@pytest.mark.parametrize(
    "relative",
    [
        "data/papers/2001.08361.tex",
        "figures/001.png",
        "ledger/runs/run-1/checkpoint.pt",
        "data/lab/lab.json",
        ".env",
        "kaggle.json",
    ],
)
def test_the_things_that_must_never_be_committed_are_not(workspace, relative):
    """Two categories in one list: large and re-fetchable, and secret. The second
    matters more than the first even with no remote -- a credential in a commit
    survives the file being deleted."""
    initialised(workspace)
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    vcs.checkpoint("everything")

    tracked = (version.git("ls-files", cwd=workspace) or "").splitlines()
    assert relative not in tracked


def test_a_workspace_that_already_has_a_gitignore_still_excludes_secrets(workspace):
    """The gap the first version left. A workspace can already have a
    `.gitignore` -- somebody's own, or one left by a checkout it used to be --
    and skipping the write there meant `.env` and `kaggle.json` were not
    excluded and the first checkpoint committed them. A credential in a commit
    survives the file being deleted."""
    (workspace / ".gitignore").write_text("# mine\n*.tmp\n", encoding="utf-8")
    initialised(workspace)

    (workspace / ".env").write_text("VOYAGE_KEY=sk-real", encoding="utf-8")
    (workspace / "scratch.tmp").write_text("x", encoding="utf-8")
    (workspace / "notes").mkdir(exist_ok=True)
    (workspace / "notes" / "a.md").write_text("kept", encoding="utf-8")
    vcs.checkpoint("after")

    tracked = (version.git("ls-files", cwd=workspace) or "").splitlines()
    assert ".env" not in tracked
    assert "notes/a.md" in tracked
    # And the rules that were already there still apply.
    assert "scratch.tmp" not in tracked
    assert "# mine" in (workspace / ".gitignore").read_text(encoding="utf-8")


def test_initialising_twice_does_not_write_the_block_twice(workspace):
    initialised(workspace)
    vcs._write_ignore(workspace / ".gitignore")

    body = (workspace / ".gitignore").read_text(encoding="utf-8")
    assert body.count(vcs.IGNORE_BEGIN) == 1


def test_a_git_that_runs_and_refuses_is_not_a_success(workspace, monkeypatch):
    """`is None` means "never ran". A non-zero return code means it ran and said
    no, and conflating the two configured and committed into a directory that
    was not a repository."""
    class Refused:
        returncode = 1
        stdout = ""
        stderr = "fatal: cannot mkdir"

    monkeypatch.setattr(vcs, "_result", lambda *a, **k: Refused())
    out = vcs.initialise()
    assert out["created"] is False
    assert out["error"]


def test_a_failed_add_does_not_commit_whatever_was_staged(workspace, monkeypatch):
    initialised(workspace)

    class Refused:
        returncode = 1
        stdout = ""
        stderr = "fatal: unable to index file"

    monkeypatch.setattr(vcs, "_result", lambda *a, **k: Refused())
    out = vcs.checkpoint("should not land")
    assert out["committed"] is False
    assert "unable to index file" in out["error"]


def test_the_current_project_pointer_is_machine_local(workspace):
    """Which project is selected is one machine's choice, not a fact about the
    research. Two checkouts should not fight over it."""
    initialised(workspace)
    (workspace / "ledger" / ".current_project").write_text("proj-a", encoding="utf-8")
    vcs.checkpoint("selected a project")

    assert ".current_project" not in (version.git("ls-files", cwd=workspace) or "")


# ---------------------------------------------------------------------------
# it may never fail a caller
# ---------------------------------------------------------------------------
def test_a_broken_git_does_not_fail_the_command_that_called_it(workspace, monkeypatch):
    """`checkpoint` runs just after a collect has written the run record. A
    wedged git turning a successful collect into a failed command would be
    strictly worse than an uncommitted workspace."""
    initialised(workspace)

    def explode(*_args, **_kwargs):
        raise OSError("git is on fire")

    monkeypatch.setattr(vcs, "_result", explode)
    result = vcs.checkpoint("collected run-1")

    assert result["committed"] is False
    assert "git is on fire" in result["error"]


def test_the_collect_path_checkpoints_without_being_able_to_fail(workspace, monkeypatch):
    """The wrapper the three call sites share."""
    from core import submit

    seen = []
    monkeypatch.setattr(
        "core.vcs.checkpoint", lambda reason, **_: seen.append(reason) or {"committed": True}
    )
    submit.checkpoint_workspace("collected run-2 (succeeded)")
    assert seen == ["collected run-2 (succeeded)"]

    def explode(*_args, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr("core.vcs.checkpoint", explode)
    submit.checkpoint_workspace("collected run-3")  # must not raise


def test_status_reports_the_refusal_rather_than_pretending(workspace, monkeypatch):
    monkeypatch.setattr(paths, "install_dir", lambda: paths.root())
    state = vcs.status()
    assert state["repository"] is False
    assert "installation folder" in state["error"]
