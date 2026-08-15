"""The updater: identity, migrations, plan, apply, and the report rule.

The plan/apply tests run against a **real git repository** built in a temp
directory, and that is deliberate. Every interesting decision here is a
judgement about what git said -- is this a fast-forward, did `pyproject.toml`
change between these two commits, is the file I have edited one the release also
touches -- and a mock of `git status` proves only that the mock matches the
assumption that wrote it. The repositories are local, the calls are local, and
nothing in this file reaches the network: `plan(do_fetch=False)` everywhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core import migrate, update, version

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


# ---------------------------------------------------------------------------
# a repository to update
# ---------------------------------------------------------------------------
def _outside(tmp_path: Path, name: str) -> Path:
    """A folder that is not inside the workspace.

    `tests/conftest.py` points `GRAD_ROOT` at `tmp_path` itself, so `tmp_path /
    "x"` is *inside* the workspace -- which `tools.workspace move` refuses, and
    rightly: copying a folder into a folder underneath it does not terminate.
    A sibling, the same shape as the app directory that fixture creates.
    """
    return tmp_path.parent / f"{tmp_path.name}-{name}"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str, files: dict[str, str]) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A checkout with two releases, v0.1.0 and v0.2.0, and HEAD on the first.

    `core/version.py` asks `core/workspace.py` where the code is, so pointing
    that at the fixture is enough to redirect every git call in both modules --
    which is also the property the split exists for.
    """
    path = tmp_path / "install"
    path.mkdir()
    _git(path, "init", "--quiet", "--initial-branch=main")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test")

    _commit(
        path,
        "first",
        {
            "pyproject.toml": '[project]\nname = "grad"\nversion = "0.1.0"\n',
            "core/thing.py": "value = 1\n",
            "ledger/.gitkeep": "",
        },
    )
    _git(path, "tag", "v0.1.0")
    first = _git(path, "rev-parse", "HEAD")

    # A release that changes code only. This is the common case, and the one the
    # conditional reinstall is for.
    _commit(path, "second", {"core/thing.py": "value = 2\n"})
    _git(path, "tag", "v0.2.0")

    _git(path, "checkout", "--quiet", "main")
    _git(path, "reset", "--hard", "--quiet", first)

    from core import workspace as workspace_mod

    monkeypatch.setattr(workspace_mod, "code_dir", lambda: path)
    version._cache = None
    yield path
    version._cache = None


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def test_identity_reads_the_checkout(repo):
    data = version.identity(reload=True)
    assert data["source"] == "git"
    assert data["tag"] == "v0.1.0"
    assert data["branch"] == "main"
    assert data["dirty"] is False
    # The file on disk, not the installed metadata: after a fast-forward that
    # changed no dependencies there is no reinstall, so the metadata is stale.
    assert data["version"] == "0.1.0"


def test_identity_degrades_without_git(tmp_path, monkeypatch):
    """A tarball install is not a failure; it is an installation that cannot
    update itself, and it still has to be able to submit a run."""
    from core import workspace as workspace_mod

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "pyproject.toml").write_text('version = "9.9.9"\n', encoding="utf-8")
    monkeypatch.setattr(workspace_mod, "code_dir", lambda: plain)
    version._cache = None

    data = version.identity(reload=True)
    assert data["source"] == "package"
    assert data["commit"] is None
    assert data["version"] == "9.9.9"
    assert version.stamp()["dirty"] is False


def test_research_does_not_make_the_installation_dirty(repo):
    """The whole reason `WORKSPACE_PATHS` exists. On the default layout the
    workspace *is* the checkout, so a `dirty` flag that counted notebooks would
    be permanently true and would describe nothing."""
    (repo / "notebooks").mkdir(exist_ok=True)
    (repo / "notebooks" / "scratch.ipynb").write_text("{}", encoding="utf-8")
    (repo / "ledger" / "runs.jsonl").write_text('{"id": "run-1"}\n', encoding="utf-8")

    assert version.dirty_paths() == []
    assert version.identity(reload=True)["dirty"] is False
    # But they are still visible to the collision check, which asks for every
    # file rather than the collapsed directory the default reports.
    assert "notebooks/scratch.ipynb" in version.status_paths(untracked="all")
    assert "notebooks" in version.status_paths()


def test_edited_code_is_dirty(repo):
    (repo / "core" / "thing.py").write_text("value = 99\n", encoding="utf-8")
    assert version.dirty_paths() == ["core/thing.py"]
    assert version.stamp()["dirty"] is True


@pytest.mark.parametrize(
    "line, expected",
    [
        (" M core/thing.py", "core/thing.py"),
        ("?? notebooks/a.ipynb", "notebooks/a.ipynb"),
        ("R  old.py -> core/new.py", "core/new.py"),
        ('?? "notes/with space.md"', "notes/with space.md"),
    ],
)
def test_porcelain_paths(line, expected):
    """A quoted or renamed path that survived as `"notes/x"` would fail the
    workspace-prefix test and be reported as modified code."""
    assert version._porcelain_path(line) == expected


def test_same_version_prefers_the_commit():
    a = {"version": "0.1.0", "tag": "v0.1.0", "commit": "aaaa", "dirty": False}
    b = {"version": "0.1.0", "tag": "v0.1.0", "commit": "bbbb", "dirty": False}
    assert not version.same_version(a, b)
    assert version.same_version(a, dict(a))
    # Two runs that both know nothing are the same, not a finding.
    assert version.same_version({}, {})
    # Same tree, but one was submitted from an edited checkout.
    assert not version.same_version(a, {**a, "dirty": True})


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def test_plan_sees_the_newer_release(repo):
    plan = update.plan(do_fetch=False)
    assert plan["available"] is True
    assert plan["target"]["tag"] == "v0.2.0"
    assert plan["target"]["direction"] == "forward"
    assert plan["behind"] == 1
    assert plan["blockers"] == []
    # Code only: no reinstall, which is what makes the common update instant.
    assert plan["needs_reinstall"] is False


def test_plan_flags_a_dependency_change(repo):
    _git(repo, "checkout", "--quiet", "v0.2.0")
    _commit(
        repo,
        "third",
        {"pyproject.toml": '[project]\nname = "grad"\nversion = "0.3.0"\ndependencies = ["x"]\n'},
    )
    _git(repo, "tag", "v0.3.0")
    _git(repo, "checkout", "--quiet", "main")

    plan = update.plan(do_fetch=False)
    assert plan["target"]["tag"] == "v0.3.0"
    assert plan["needs_reinstall"] is True


def test_modified_code_blocks(repo):
    (repo / "core" / "thing.py").write_text("local edit\n", encoding="utf-8")
    plan = update.plan(do_fetch=False)
    assert [b["code"] for b in plan["blockers"]] == ["dirty_code"]
    assert "core/thing.py" in plan["blockers"][0]["message"]


def test_research_alone_does_not_block(repo):
    """An edited notebook is not a reason to refuse an update. On the default
    layout it is the normal state of the folder."""
    (repo / "notebooks").mkdir(exist_ok=True)
    (repo / "notebooks" / "mine.ipynb").write_text("{}", encoding="utf-8")
    (repo / "ledger" / "runs.jsonl").write_text('{"id": "run-1"}\n', encoding="utf-8")
    assert update.plan(do_fetch=False)["blockers"] == []


def test_research_the_release_also_changes_does_block(repo):
    """The distinction the updater turns on: a file the incoming release
    changes *and* the user has edited is a real conflict, and the fix for it is
    to get the research out of the checkout rather than to discard either side."""
    # Upstream grows a notebook, on top of the newest release.
    _git(repo, "checkout", "--quiet", "v0.2.0")
    (repo / "notebooks").mkdir(exist_ok=True)
    (repo / "notebooks" / "shared.ipynb").write_text('{"upstream": true}', encoding="utf-8")
    _git(repo, "add", "notebooks/shared.ipynb")
    _git(repo, "commit", "-m", "upstream notebook")
    _git(repo, "tag", "v0.4.0")
    _git(repo, "checkout", "--quiet", "main")

    # And the user has their own copy of exactly that path.
    (repo / "notebooks").mkdir(exist_ok=True)
    (repo / "notebooks" / "shared.ipynb").write_text('{"mine": true}', encoding="utf-8")

    plan = update.plan(do_fetch=False)
    codes = [b["code"] for b in plan["blockers"]]
    assert "workspace_collision" in codes
    blocker = plan["blockers"][codes.index("workspace_collision")]
    assert "notebooks/shared.ipynb" in blocker["message"]
    assert "workspace move" in blocker["fix"]


def test_plan_warns_when_the_workspace_is_the_installation(repo, monkeypatch):
    from core import paths

    monkeypatch.setattr(paths, "root", lambda: repo)
    plan = update.plan(do_fetch=False)
    assert "workspace_in_install" in [w["code"] for w in plan["warnings"]]


def test_plan_on_a_non_checkout_is_an_answer_not_an_exception(tmp_path, monkeypatch):
    from core import workspace as workspace_mod

    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(workspace_mod, "code_dir", lambda: plain)
    version._cache = None

    plan = update.plan(do_fetch=False)
    assert [b["code"] for b in plan["blockers"]] == ["not_a_checkout"]
    assert plan["available"] is False


# ---------------------------------------------------------------------------
# applying it
# ---------------------------------------------------------------------------
def test_apply_fast_forwards_without_reinstalling(repo, monkeypatch):
    called: list[list[str]] = []
    monkeypatch.setattr(update, "_reinstall", lambda chosen: called.append(chosen))

    result = update.apply(do_fetch=False)
    assert result["applied"] is True
    assert result["to"]["tag"] == "v0.2.0"
    assert result["reinstalled"] is False
    assert called == []
    # The tree actually moved, and the branch came with it.
    assert (repo / "core" / "thing.py").read_text(encoding="utf-8") == "value = 2\n"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert version.identity(reload=True)["tag"] == "v0.2.0"


def test_apply_reinstalls_when_the_dependencies_moved(repo, monkeypatch):
    _git(repo, "checkout", "--quiet", "v0.2.0")
    _commit(repo, "deps", {"pyproject.toml": '[project]\nversion = "0.3.0"\n'})
    _git(repo, "tag", "v0.3.0")
    _git(repo, "checkout", "--quiet", "main")

    called: list[list[str]] = []
    monkeypatch.setattr(update, "_reinstall", lambda chosen: called.append(chosen))
    monkeypatch.setattr(update, "instance_running", lambda: False)
    update.write_install_record(extras=["ui", "agent"])

    result = update.apply(do_fetch=False)
    assert result["reinstalled"] is True
    # The extras it was installed with, not a guess.
    assert called == [["ui", "agent"]]


def test_apply_refuses_a_dependency_change_while_grad_is_running(repo, monkeypatch):
    from core.errors import GradError

    _git(repo, "checkout", "--quiet", "v0.2.0")
    _commit(repo, "deps", {"pyproject.toml": '[project]\nversion = "0.3.0"\n'})
    _git(repo, "tag", "v0.3.0")
    _git(repo, "checkout", "--quiet", "main")

    monkeypatch.setattr(update, "instance_running", lambda: True)
    monkeypatch.setattr(update, "_reinstall", lambda chosen: pytest.fail("must not reinstall"))

    with pytest.raises(GradError) as caught:
        update.apply(do_fetch=False)
    assert caught.value.code == "update_running"
    assert "quit" in (caught.value.fix or "").lower()
    # And nothing moved.
    assert version.identity(reload=True)["tag"] == "v0.1.0"


def test_apply_refuses_over_a_blocker(repo):
    from core.errors import GradError

    (repo / "core" / "thing.py").write_text("local edit\n", encoding="utf-8")
    with pytest.raises(GradError) as caught:
        update.apply(do_fetch=False)
    assert caught.value.code == "update_dirty_code"
    assert version.identity(reload=True)["tag"] == "v0.1.0"


def test_a_failed_reinstall_leaves_the_update_resumable(repo, monkeypatch):
    """The recovery this design exists for. The git move is atomic and the pip
    step is not, so a failure between them advances the commit -- after which
    `available` is False and a naive updater says "up to date" forever, with new
    code running against old dependencies."""
    from core.errors import GradError

    _git(repo, "checkout", "--quiet", "v0.2.0")
    _commit(repo, "deps", {"pyproject.toml": '[project]\nversion = "0.3.0"\n'})
    _git(repo, "tag", "v0.3.0")
    _git(repo, "checkout", "--quiet", "main")
    monkeypatch.setattr(update, "instance_running", lambda: False)

    def _explode(chosen):
        raise GradError("update_reinstall_failed", "pip fell over", exit_code=8)

    monkeypatch.setattr(update, "_reinstall", _explode)
    with pytest.raises(GradError):
        update.apply(do_fetch=False)

    # The checkout moved, so nothing derived from the commit can tell that work
    # is outstanding. The marker can.
    assert version.identity(reload=True)["tag"] == "v0.3.0"
    plan = update.plan(do_fetch=False)
    assert plan["available"] is False
    assert plan["incomplete"]["tag"] == "v0.3.0"
    assert "incomplete_update" in [w["code"] for w in plan["warnings"]]

    calls: list[list[str]] = []
    monkeypatch.setattr(update, "_reinstall", lambda chosen: calls.append(chosen))
    result = update.apply(do_fetch=False)
    assert result["applied"] is True
    assert result["resumed"] is True
    assert result["reinstalled"] is True
    assert calls, "the reinstall the first run never completed must run"
    # And once it has, the marker is gone and the next call is a no-op.
    assert update.incomplete_update() is None
    assert update.apply(do_fetch=False)["applied"] is False


def test_a_successful_update_leaves_no_marker(repo, monkeypatch):
    """The marker must not survive the common path, or every subsequent check
    would propose a reinstall that is not needed."""
    monkeypatch.setattr(update, "_reinstall", lambda chosen: None)
    update.apply(do_fetch=False)
    assert update.incomplete_update() is None
    assert update.plan(do_fetch=False)["incomplete"] is None


def test_a_failed_git_move_leaves_no_marker(repo, monkeypatch):
    """Nothing moved, so nothing is half-done: a resume pointing at an update
    that never started would demand a reinstall for no reason."""
    from core.errors import GradError

    monkeypatch.setattr(version, "git_result", lambda *a, **k: None)
    with pytest.raises(GradError):
        update.apply(do_fetch=False)
    assert update.incomplete_update() is None


def test_a_failed_move_does_not_forget_an_earlier_interrupted_update(repo, monkeypatch):
    """The marker this run overwrote is not this run's to discard.

    A release appearing while an earlier update is still outstanding is exactly
    when both are in play: `begin_update` replaces the old marker, and clearing
    it on a failure would lose the reinstall nothing else knows about.
    """
    from core.errors import GradError

    # An earlier update moved the checkout and never finished its reinstall.
    _git(repo, "checkout", "--quiet", "v0.2.0")
    _commit(repo, "deps", {"pyproject.toml": '[project]\nversion = "0.3.0"\n'})
    _git(repo, "tag", "v0.3.0")
    _git(repo, "checkout", "--quiet", "main")
    monkeypatch.setattr(update, "instance_running", lambda: False)
    monkeypatch.setattr(
        update, "_reinstall", lambda chosen: (_ for _ in ()).throw(GradError("x", "pip fell over"))
    )
    with pytest.raises(GradError):
        update.apply(do_fetch=False)
    assert update.incomplete_update()["tag"] == "v0.3.0"

    # A newer release lands, and this run cannot even move the tree.
    _git(repo, "tag", "v0.4.0")
    monkeypatch.setattr(version, "git_result", lambda *a, **k: None)
    with pytest.raises(GradError):
        update.apply(do_fetch=False)

    outstanding = update.incomplete_update()
    assert outstanding is not None, "the earlier update's marker must survive"
    assert outstanding["tag"] == "v0.3.0"
    assert outstanding["needs_reinstall"] is True


def test_apply_is_idempotent(repo, monkeypatch):
    monkeypatch.setattr(update, "_reinstall", lambda chosen: None)
    update.apply(do_fetch=False)
    again = update.apply(do_fetch=False)
    assert again["applied"] is False
    assert "up to date" in again["message"]


def test_pinning_an_older_release_detaches(repo, monkeypatch):
    monkeypatch.setattr(update, "_reinstall", lambda chosen: None)
    update.apply(do_fetch=False)  # now on v0.2.0
    assert version.identity(reload=True)["tag"] == "v0.2.0"

    result = update.apply(to="v0.1.0", do_fetch=False)
    assert result["to"]["direction"] == "checkout"
    assert (repo / "core" / "thing.py").read_text(encoding="utf-8") == "value = 1\n"
    assert update.rollback_target() is None  # nothing older than v0.1.0


def test_rollback_target_is_the_release_before_this_one(repo, monkeypatch):
    monkeypatch.setattr(update, "_reinstall", lambda chosen: None)
    update.apply(do_fetch=False)
    assert update.rollback_target() == "v0.1.0"


def test_unknown_tag_is_named(repo):
    plan = update.plan(do_fetch=False, to="v9.9.9")
    assert [b["code"] for b in plan["blockers"]] == ["unknown_tag"]
    assert "v0.2.0" in plan["blockers"][0]["fix"]


# ---------------------------------------------------------------------------
# extras and the install record
# ---------------------------------------------------------------------------
def test_extras_prefers_the_record_over_detection():
    update.write_install_record(extras=["ui", "lab"])
    assert update.extras() == ["ui", "lab"]


def test_extras_falls_back_to_detection(monkeypatch):
    monkeypatch.setattr(update, "read_install_record", dict)
    monkeypatch.setattr(update, "detect_extras", lambda: ["agent"])
    assert update.extras() == ["agent"]
    monkeypatch.setattr(update, "detect_extras", list)
    assert update.extras() == list(update.DEFAULT_EXTRAS)


def test_parse_extras_refuses_a_typo():
    from core.errors import UsageError

    assert update.parse_extras("ui,agent") == "ui,agent"
    with pytest.raises(UsageError):
        update.parse_extras("ui,noteboook")


# ---------------------------------------------------------------------------
# the cache the app renders
# ---------------------------------------------------------------------------
def test_cache_round_trip_and_staleness(repo):
    assert update.check_due() is True  # never checked
    update.write_cache(update.plan(do_fetch=False))
    assert update.check_due() is False
    assert update.read_cache()["target"]["tag"] == "v0.2.0"
    assert update.cache_age_s() < 60


def test_a_corrupt_cache_is_not_a_crash(repo):
    update.cache_path().parent.mkdir(parents=True, exist_ok=True)
    update.cache_path().write_text("{not json", encoding="utf-8")
    assert update.read_cache() == {}
    assert update.cache_age_s() == float("inf")
    assert update.check_due() is True


# ---------------------------------------------------------------------------
# migrations
# ---------------------------------------------------------------------------
def test_migrations_run_once_and_are_recorded():
    assert migrate.current() == 0
    migrate.run_pending()
    assert migrate.current() == migrate.LATEST
    assert migrate.pending() == []


def test_a_failing_migration_stays_pending(monkeypatch):
    def boom() -> list[str]:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(migrate, "MIGRATIONS", ((1, "ok", list), (2, "boom", boom)))
    assert migrate.run_pending() == []
    # Step 1 completed and was recorded; step 2 did not, and is still pending.
    assert migrate.current() == 1
    assert [number for number, _, _ in migrate.pending()] == [2]


def test_a_migration_that_could_not_write_is_not_recorded(monkeypatch):
    """`write_install_record` swallows OSError, which is right for a caller that
    only wants the record kept current and wrong when the write *is* the
    migration: the number would go up and nothing would ever revisit it."""
    monkeypatch.setattr(update, "detect_extras", lambda: ["ui"])
    monkeypatch.setattr(update, "write_install_record", lambda **kwargs: None)
    update.install_record_path().unlink(missing_ok=True)

    assert migrate.run_pending() == []
    assert migrate.current() == 1
    assert [number for number, _, _ in migrate.pending()] == [2]


def test_migration_records_the_extras_of_an_older_install(monkeypatch):
    monkeypatch.setattr(update, "detect_extras", lambda: ["ui", "notebook"])
    update.install_record_path().unlink(missing_ok=True)

    changed = migrate.run_pending()
    assert any("install.json" in line for line in changed)
    assert update.read_install_record()["extras"] == ["ui", "notebook"]
    assert update.read_install_record()["detected"] is True


# ---------------------------------------------------------------------------
# the workspace split
# ---------------------------------------------------------------------------
def test_shipped_paths_fall_back_to_the_installation(tmp_path, monkeypatch):
    """A workspace outside the checkout has no `prompts/`, and the agent must
    still find its system prompt -- this is what made the split possible."""
    from core import paths, workspace as workspace_mod

    install = tmp_path / "install"
    (install / "prompts").mkdir(parents=True)
    (install / "prompts" / "system.md").write_text("shipped", encoding="utf-8")
    (install / "skills").mkdir()
    monkeypatch.setattr(workspace_mod, "code_dir", lambda: install)

    assert paths.prompt_path() == install / "prompts" / "system.md"
    assert paths.skills_dir() == install / "skills"

    # A workspace copy wins, which is what makes a per-workspace prompt possible.
    local = paths.root() / "prompts"
    local.mkdir(parents=True, exist_ok=True)
    (local / "system.md").write_text("mine", encoding="utf-8")
    assert paths.prompt_path().read_text(encoding="utf-8") == "mine"


def test_system_prompt_says_where_the_skills_are_when_they_are_elsewhere(tmp_path, monkeypatch):
    import agent as agent_mod
    from core import paths, workspace as workspace_mod

    install = tmp_path / "install"
    (install / "prompts").mkdir(parents=True)
    (install / "prompts" / "system.md").write_text("BODY", encoding="utf-8")
    (install / "skills").mkdir()
    monkeypatch.setattr(workspace_mod, "code_dir", lambda: install)

    text = agent_mod.system_prompt()
    assert text.startswith("BODY")
    assert str(paths.root()) in text
    assert str(install / "skills") in text

    # Same folder: byte-for-byte what the file says, with nothing appended.
    monkeypatch.setattr(workspace_mod, "code_dir", lambda: paths.root())
    (paths.root() / "prompts").mkdir(parents=True, exist_ok=True)
    (paths.root() / "prompts" / "system.md").write_text("BODY", encoding="utf-8")
    (paths.root() / "skills").mkdir(parents=True, exist_ok=True)
    assert agent_mod.system_prompt() == "BODY"


def test_workspace_move_copies_and_keeps_the_originals(tmp_path):
    """`move` is not `mv`, and the reason is in the module docstring: these files
    are the only record of someone's experiments."""
    from core import paths
    from tools.workspace import cli as workspace_cli

    source = paths.root()
    (source / "ledger").mkdir(parents=True, exist_ok=True)
    (source / "ledger" / "runs.jsonl").write_text('{"id": "run-1"}\n', encoding="utf-8")
    (source / "notebooks").mkdir(parents=True, exist_ok=True)
    (source / "notebooks" / "a.ipynb").write_text("{}", encoding="utf-8")
    # Regenerable and machine-local: skipped rather than carried to a new machine.
    (source / "data" / "lab").mkdir(parents=True, exist_ok=True)
    (source / "data" / "lab" / "lab.json").write_text('{"token": "secret"}', encoding="utf-8")

    target = _outside(tmp_path, "elsewhere")
    assert workspace_cli.run(["move", str(target), "--keep-pointer", "--json"]) == 0

    assert (target / "ledger" / "runs.jsonl").read_text(encoding="utf-8") == '{"id": "run-1"}\n'
    assert (target / "notebooks" / "a.ipynb").exists()
    assert not (target / "data" / "lab").exists()
    # The originals are still there. Deleting them is a second, explicit step.
    assert (source / "ledger" / "runs.jsonl").exists()


def test_workspace_move_refuses_a_folder_that_already_has_a_ledger(tmp_path):
    from core import paths
    from tools.workspace import cli as workspace_cli

    (paths.root() / "ledger").mkdir(parents=True, exist_ok=True)
    (paths.root() / "ledger" / "runs.jsonl").write_text("{}\n", encoding="utf-8")
    target = _outside(tmp_path, "occupied")
    (target / "ledger").mkdir(parents=True)
    (target / "ledger" / "runs.jsonl").write_text("{}\n", encoding="utf-8")

    assert workspace_cli.run(["move", str(target), "--json"]) != 0


def test_workspace_move_refuses_to_copy_into_itself(tmp_path):
    """The failure that does damage rather than refusing: copying `notebooks/`
    into a folder underneath it descends into the copy it is making."""
    from core import paths
    from tools.workspace import cli as workspace_cli

    (paths.root() / "notebooks").mkdir(parents=True, exist_ok=True)
    (paths.root() / "notebooks" / "a.ipynb").write_text("{}", encoding="utf-8")

    inside = paths.root() / "notebooks" / "new-home"
    assert workspace_cli.run(["move", str(inside), "--json"]) != 0
    assert not (inside / "notebooks").exists()


def test_workspace_move_refuses_to_merge_into_an_occupied_folder(tmp_path):
    """`copytree(dirs_exist_ok=True)` would interleave the two silently, and the
    result would be a folder whose history is neither workspace's."""
    from core import paths
    from tools.workspace import cli as workspace_cli

    (paths.root() / "notebooks").mkdir(parents=True, exist_ok=True)
    (paths.root() / "notebooks" / "mine.ipynb").write_text("{}", encoding="utf-8")
    target = _outside(tmp_path, "someone-elses")
    (target / "notebooks").mkdir(parents=True)
    (target / "notebooks" / "theirs.ipynb").write_text("{}", encoding="utf-8")

    assert workspace_cli.run(["move", str(target), "--json"]) != 0
    assert not (target / "notebooks" / "mine.ipynb").exists()


def test_workspace_move_keeps_the_pointer_when_a_file_did_not_arrive(tmp_path, monkeypatch):
    """A partial copy must not become the workspace: the app would read a
    partial ledger while the complete one sat in the folder it just left."""
    import tools.workspace as workspace_tool
    from core import paths
    from tools.workspace import cli as workspace_cli

    source = paths.root()
    (source / "ledger").mkdir(parents=True, exist_ok=True)
    (source / "ledger" / "runs.jsonl").write_text('{"id": "run-1"}\n', encoding="utf-8")
    (source / "notebooks").mkdir(parents=True, exist_ok=True)
    (source / "notebooks" / "a.ipynb").write_text("{}", encoding="utf-8")

    # The notebooks directory fails to copy, the way a locked file or a full
    # disk would make it fail.
    real = workspace_tool._copy_entry
    monkeypatch.setattr(
        workspace_tool,
        "_copy_entry",
        lambda src, dst, name: False if name == "notebooks" else real(src, dst, name),
    )

    target = _outside(tmp_path, "partial")
    assert workspace_cli.run(["move", str(target), "--json"]) == 0
    # It arrived nowhere, it is reported, and neither the pointer nor the
    # originals moved.
    assert paths.root() == source
    assert (source / "notebooks" / "a.ipynb").exists()


def test_workspace_move_will_not_delete_originals_after_a_partial_copy(tmp_path, monkeypatch):
    import tools.workspace as workspace_tool
    from core import paths
    from tools.workspace import cli as workspace_cli

    source = paths.root()
    (source / "notebooks").mkdir(parents=True, exist_ok=True)
    (source / "notebooks" / "a.ipynb").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workspace_tool, "_copy_entry", lambda src, dst, name: False)

    assert workspace_cli.run(["move", str(_outside(tmp_path, "partial2")), "--remove-originals", "--json"]) != 0
    assert (source / "notebooks" / "a.ipynb").exists()


# ---------------------------------------------------------------------------
# which code produced a number
# ---------------------------------------------------------------------------
def _run_with_version(run_id: str, stamp: dict | None) -> None:
    from core import ledger_store as ls

    record = {
        "type": ls.T_RUN_SUBMITTED,
        "id": run_id,
        "task": "t",
        "status": "in_flight",
        "submitted_at": ls.now_iso(),
    }
    if stamp is not None:
        record["code_version"] = stamp
    ls.append_run_event(record)
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": run_id, "collected_at": ls.now_iso(),
         "results": {"loss": 1.0}, "actual_usd": 0.0}
    )


def test_report_flags_runs_from_two_versions():
    from core import report

    _run_with_version("run-a", {"version": "0.1.0", "tag": "v0.1.0", "commit": "aaaa", "dirty": False})
    _run_with_version("run-b", {"version": "0.2.0", "tag": "v0.2.0", "commit": "bbbb", "dirty": False})

    findings = report.check_code_versions({"run-a", "run-b"})
    assert [f["rule"] for f in findings] == ["version"]
    assert "v0.1.0" in findings[0]["problem"] and "v0.2.0" in findings[0]["problem"]


def test_report_flags_a_run_from_a_modified_checkout():
    from core import report

    _run_with_version("run-c", {"version": "0.1.0", "tag": "v0.1.0", "commit": "aaaa", "dirty": True})
    findings = report.check_code_versions({"run-c"})
    assert len(findings) == 1
    assert "modified" in findings[0]["problem"]


def test_report_is_silent_about_runs_that_predate_the_stamp():
    from core import report

    _run_with_version("run-d", None)
    _run_with_version("run-e", None)
    assert report.check_code_versions({"run-d", "run-e"}) == []


def test_report_accepts_one_version():
    from core import report

    stamp = {"version": "0.1.0", "tag": "v0.1.0", "commit": "aaaa", "dirty": False}
    _run_with_version("run-f", stamp)
    _run_with_version("run-g", dict(stamp))
    assert report.check_code_versions({"run-f", "run-g"}) == []


def test_the_editor_runs_the_same_fourth_rule_as_the_gate(monkeypatch):
    """A badge saying "no findings" over a report `report check` will refuse is
    worse than no badge: the editor is the surface someone watches while
    writing."""
    from core import report as report_mod
    from ui import models

    seen: list[set] = []
    monkeypatch.setattr(
        report_mod,
        "check_code_versions",
        lambda run_ids: seen.append(run_ids)
        or [{"rule": "version", "problem": "two versions", "fix": "re-run"}],
    )
    monkeypatch.setattr(models, "SECTION_RE", models.SECTION_RE)

    from core import paths

    project = "proj-1"
    tex_dir = paths.root() / "reports" / project
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "main.tex").write_text("\\section{One}\n", encoding="utf-8")

    model = models.editor_model(project)
    assert seen, "the editor must ask the ledger-backed rule too"
    assert any(f.get("rule") == "version" for f in model["findings"])


def test_the_stamp_cannot_be_overridden_by_a_backend(monkeypatch):
    """`extra` is backend-specific fields from a submitter. The stamp is what
    `report check` rests on, so it is not a default a caller may replace."""
    from core import submit

    monkeypatch.setattr(
        version, "stamp", lambda: {"version": "1.2.3", "tag": None, "commit": "real", "dirty": False}
    )

    class _Sub:
        spec_path = Path("specs/x/submission.json")
        config = {"task": "t"}
        image = None
        dataset = None
        metrics_file = None

        def hash(self) -> str:
            return "h" * 12

        def estimated_cost_usd(self) -> float:
            return 1.0

        def estimated_duration_s(self) -> float:
            return 60.0

    _, record = submit.record_submission(
        _Sub(),
        expectation_id=None,
        platform="test",
        target={},
        command=["true"],
        extra={"code_version": {"commit": "forged"}},
    )
    assert record["code_version"]["commit"] == "real"


def test_submitted_runs_carry_the_stamp(monkeypatch):
    """The stamp has to be on the record the submitter writes, not added later:
    a run collected on another machine would otherwise carry that machine's."""
    from core import ledger_store as ls, submit

    monkeypatch.setattr(
        version, "stamp", lambda: {"version": "1.2.3", "tag": None, "commit": "cccc", "dirty": False}
    )

    class _Sub:
        spec_path = Path("specs/x/submission.json")
        config = {"task": "t"}
        image = None
        dataset = None
        metrics_file = None

        def hash(self) -> str:
            return "h" * 12

        def estimated_cost_usd(self) -> float:
            return 1.0

        def estimated_duration_s(self) -> float:
            return 60.0

    run_id, record = submit.record_submission(
        _Sub(), expectation_id=None, platform="test", target={}, command=["true"]
    )
    assert record["code_version"]["commit"] == "cccc"
    assert ls.run(run_id).get("code_version")["version"] == "1.2.3"


# ---------------------------------------------------------------------------
# the model the app renders
# ---------------------------------------------------------------------------
def test_update_model_never_claims_up_to_date_without_evidence(repo):
    from ui import models

    model = models.update_model()
    assert model["checked"] == "never"
    assert model["available"] is False
    assert model["is_checkout"] is True

    update.write_cache(update.plan(do_fetch=False))
    model = models.update_model()
    assert model["available"] is True
    assert model["target"] == "v0.2.0"
    assert model["checked"] == "just now"


def test_update_model_hides_an_update_that_is_blocked(repo):
    """A button that cannot work is worse than no button: the blocker is what
    the menu shows instead."""
    from ui import models

    (repo / "core" / "thing.py").write_text("edited\n", encoding="utf-8")
    update.write_cache(update.plan(do_fetch=False))

    model = models.update_model()
    assert model["available"] is False
    assert model["blockers"] and "core/thing.py" in model["blockers"][0]["message"]


def test_tray_entry_is_absent_until_there_is_something_to_install(repo):
    from ui import desktop

    assert desktop._available_tag() is None
    update.write_cache(update.plan(do_fetch=False))
    assert desktop._available_tag() == "v0.2.0"


def test_check_command_caches_what_it_found(repo):
    from tools.update import cli as update_cli

    assert update_cli.run(["check", "--offline", "--json"]) == 0
    cached = json.loads(update.cache_path().read_text(encoding="utf-8"))
    assert cached["target"]["tag"] == "v0.2.0"
