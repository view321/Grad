"""Choosing the workspace folder (`core/workspace.py`, `paths.root`).

The rule this file exists to hold still is the **precedence**: GRAD_ROOT, then
the remembered choice, then the installed directory. Getting it wrong in either
direction is bad in a way that is hard to see -- a remembered folder that beat
the environment would silently redirect the test suite and every explicit
command line, and one that never won would make the app's own folder chooser
forget on every restart.

Every test here redirects the pointer file into a temp directory. The real one
lives beside the code, so a test that forgot would rewrite the developer's own
workspace choice as a side effect of running the suite.
"""

from __future__ import annotations

import json
import os

import pytest

from core import paths, workspace as workspace_mod
from core.errors import ConfigError, UsageError


@pytest.fixture
def pointer(tmp_path, monkeypatch):
    """A pointer file of our own, and a cache that does not leak between tests."""
    path = tmp_path / "pointer" / ".grad-workspace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workspace_mod, "pointer_path", lambda: path)
    monkeypatch.setattr(workspace_mod, "_cache", None, raising=False)
    yield path
    workspace_mod._cache = None


@pytest.fixture
def no_env(monkeypatch):
    """Without this, GRAD_ROOT wins and the pointer is never consulted."""
    monkeypatch.delenv("GRAD_ROOT", raising=False)


@pytest.fixture
def restart(monkeypatch):
    """Simulate relaunching the app.

    `select` sets GRAD_ROOT in this process -- that is how the switch reaches
    the CLIs -- so after a switch the environment legitimately wins and the
    pointer is never consulted. Only a new process asks the pointer anything,
    and a new process is exactly what these tests are about.
    """

    def _restart() -> None:
        monkeypatch.delenv("GRAD_ROOT", raising=False)
        workspace_mod._cache = None

    return _restart


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------
def test_the_environment_beats_a_remembered_folder(tmp_path, pointer, monkeypatch):
    """An explicit override stays explicit. The suite itself relies on this:
    `conftest` sets GRAD_ROOT, and a remembered folder that won would point every
    test at a real workspace."""
    remembered = tmp_path / "remembered"
    remembered.mkdir()
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    workspace_mod.select(remembered)

    monkeypatch.setenv("GRAD_ROOT", str(explicit))
    assert paths.root() == explicit.resolve()
    assert workspace_mod.source() == "environment"


def test_a_remembered_folder_beats_the_installed_directory(tmp_path, pointer, no_env, restart):
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    workspace_mod.select(chosen)
    restart()
    assert paths.root() == chosen.resolve()
    assert workspace_mod.source() == "remembered"


def test_with_nothing_remembered_the_root_is_where_the_code_lives(pointer, no_env):
    assert paths.root() == workspace_mod.code_dir()
    assert workspace_mod.source() == "default"


def test_selecting_sets_the_environment_so_subprocesses_agree(tmp_path, pointer, monkeypatch):
    """The CLIs run as subprocesses and inherit this environment. A switch that
    only updated `paths` would leave the agent's Bash tools reading the folder
    the UI just left."""
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    workspace_mod.select(chosen)
    assert os.environ["GRAD_ROOT"] == str(chosen.resolve())


# ---------------------------------------------------------------------------
# what survives a restart
# ---------------------------------------------------------------------------
def test_the_choice_is_remembered_across_a_fresh_read(tmp_path, pointer, no_env):
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    workspace_mod.select(chosen)

    workspace_mod._cache = None  # a new process
    assert workspace_mod.remembered() == chosen.resolve()


def test_a_remembered_folder_that_no_longer_exists_is_not_returned(
    tmp_path, pointer, no_env, restart
):
    """The folder may have been deleted, renamed, or live on a drive that is not
    mounted today. Returning it would send every ledger read somewhere that
    cannot be created."""
    gone = tmp_path / "gone"
    gone.mkdir()
    workspace_mod.select(gone)
    gone.rmdir()

    restart()
    assert workspace_mod.remembered() is None
    assert paths.root() == workspace_mod.code_dir()


def test_a_corrupt_pointer_is_not_a_startup_failure(pointer, no_env):
    pointer.write_text("{not json", encoding="utf-8")
    workspace_mod._cache = None
    assert workspace_mod.remembered() is None
    assert paths.root() == workspace_mod.code_dir()


@pytest.mark.parametrize("garbage", ["null", '"a string"', "[]", '{"root": 7}', '{"root": ""}'])
def test_a_hand_edited_pointer_yields_no_root_rather_than_a_traceback(pointer, no_env, garbage):
    pointer.write_text(garbage, encoding="utf-8")
    workspace_mod._cache = None
    assert workspace_mod.remembered() is None


def test_a_pointer_that_cannot_be_written_still_switches_this_process(tmp_path, monkeypatch):
    """A system-wide install is read-only. The switch has to apply now even if
    it cannot be remembered for next time."""
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    monkeypatch.setattr(
        workspace_mod, "pointer_path", lambda: tmp_path / "nope" / "deeper" / "p.json"
    )
    monkeypatch.setattr(workspace_mod, "_cache", None, raising=False)
    assert workspace_mod.select(chosen) == chosen.resolve()
    assert os.environ["GRAD_ROOT"] == str(chosen.resolve())
    workspace_mod._cache = None


# ---------------------------------------------------------------------------
# the recent list
# ---------------------------------------------------------------------------
def test_the_folder_you_leave_is_what_you_can_get_back_to(tmp_path, pointer, monkeypatch):
    """Recording only the destination leaves the history empty exactly when it
    matters: after one switch its only entry is where you already are, which the
    menu filters out. Switching back is the whole point of the list."""
    from ui import models

    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    monkeypatch.setenv("GRAD_ROOT", str(here))

    workspace_mod.select(there)
    assert here.resolve() in workspace_mod.recent()
    assert models.workspaces_model()["recent"] == [str(here.resolve())]


def test_recent_folders_are_most_recent_first_and_deduplicated(tmp_path, pointer, no_env):
    made = []
    for name in ("a", "b", "c"):
        folder = tmp_path / name
        folder.mkdir()
        made.append(folder.resolve())
        workspace_mod.select(folder)
    workspace_mod.select(made[0])          # back to the first

    listed = workspace_mod.recent()
    assert listed[:3] == [made[0], made[2], made[1]]
    # `a` was visited twice and appears once. The tail is the folder the first
    # switch left, which is the installed directory here.
    assert listed.count(made[0]) == 1
    assert listed[3:] == [workspace_mod.code_dir()]


def test_the_recent_list_is_capped(tmp_path, pointer, no_env):
    for index in range(workspace_mod.MAX_RECENT + 4):
        folder = tmp_path / f"w{index}"
        folder.mkdir()
        workspace_mod.select(folder)
    assert len(workspace_mod.recent()) <= workspace_mod.MAX_RECENT


def test_a_recent_folder_that_was_deleted_is_dropped_from_the_list(tmp_path, pointer, no_env):
    keep, gone = tmp_path / "keep", tmp_path / "gone"
    keep.mkdir()
    gone.mkdir()
    workspace_mod.select(gone)
    workspace_mod.select(keep)
    assert gone.resolve() in workspace_mod.recent()
    gone.rmdir()
    assert gone.resolve() not in workspace_mod.recent()
    assert keep.resolve() in workspace_mod.recent()


# ---------------------------------------------------------------------------
# validation -- the value comes from a text field
# ---------------------------------------------------------------------------
def test_a_blank_folder_is_refused_with_a_fix():
    with pytest.raises(UsageError) as caught:
        workspace_mod.validate("   ")
    assert caught.value.fix


def test_a_file_is_refused_as_a_workspace(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hi", encoding="utf-8")
    with pytest.raises(UsageError, match="not a folder"):
        workspace_mod.validate(target)


def test_a_missing_folder_is_refused_unless_creating(tmp_path):
    missing = tmp_path / "does" / "not" / "exist"
    with pytest.raises(UsageError, match="does not exist"):
        workspace_mod.validate(missing)
    assert workspace_mod.validate(missing, create=True) == missing.resolve()
    assert missing.is_dir()


def test_surrounding_quotes_and_whitespace_are_tolerated(tmp_path):
    """Copying a path out of a file manager brings the quotes with it."""
    folder = tmp_path / "with space"
    folder.mkdir()
    assert workspace_mod.validate(f'  "{folder}"  ') == folder.resolve()


def test_a_home_relative_path_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # Windows
    monkeypatch.setenv("HOME", str(tmp_path))          # everywhere else
    (tmp_path / "grad").mkdir()
    assert workspace_mod.validate("~/grad") == (tmp_path / "grad").resolve()


def test_validate_does_not_switch_anything(tmp_path, monkeypatch):
    """`validate` is called to check a field; only `select` may move the app."""
    monkeypatch.setenv("GRAD_ROOT", str(tmp_path / "current"))
    (tmp_path / "current").mkdir()
    other = tmp_path / "other"
    other.mkdir()
    workspace_mod.validate(other)
    assert os.environ["GRAD_ROOT"] == str(tmp_path / "current")


# ---------------------------------------------------------------------------
# the menu's model
# ---------------------------------------------------------------------------
def test_the_menu_model_lists_projects_and_marks_the_current_one(workspace, pointer):
    from core import budget as budget_mod
    from ui import models

    budget_mod.create("proj-a", title="Scaling laws", budget={"gpu_usd": 100.0})
    budget_mod.create("proj-b", title="Optimisers", budget={})
    budget_mod.set_current("proj-b")

    model = models.workspaces_model()
    assert model["current_project"] == "proj-b"
    by_id = {p["id"]: p for p in model["projects"]}
    assert by_id["proj-b"]["current"] is True
    assert by_id["proj-a"]["current"] is False
    assert by_id["proj-a"]["spend"].startswith("gpu ")
    # A project with no ceilings says so rather than showing an empty meter.
    assert by_id["proj-b"]["spend"] == "no ceilings"


def test_the_menu_model_opens_on_an_empty_workspace(workspace, pointer):
    """It is the panel that has to render when the workspace is wrong -- that is
    what it is for."""
    from ui import models

    model = models.workspaces_model()
    assert model["projects"] == []
    assert model["root"] == str(workspace)


def test_the_current_folder_is_not_offered_as_somewhere_to_go(tmp_path, pointer, no_env):
    from ui import models

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workspace_mod.select(first)
    workspace_mod.select(second)

    model = models.workspaces_model()
    assert model["root"] == str(second.resolve())
    assert str(second.resolve()) not in model["recent"]
    assert str(first.resolve()) in model["recent"]


# ---------------------------------------------------------------------------
# the install-shape assertion
# ---------------------------------------------------------------------------
def test_a_workspace_inside_site_packages_is_refused(tmp_path, pointer, no_env, monkeypatch):
    """The failure this catches succeeds at everything except being correct.

    `root()`'s last resort is the directory the code sits in. That is right for a
    checkout and wrong for an install: `README.md` documents `pip install -e`,
    and an install recorded as non-editable puts the code under `site-packages`,
    so the fallback resolves there and every ledger write, note and figure lands
    inside the installed package. Nothing raises. The runs just go somewhere
    nobody looks and the next install overwrites, and the only symptom is a
    ledger that keeps coming up empty.
    """
    installed = tmp_path / "venv" / "Lib" / "site-packages" / "grad"
    installed.mkdir(parents=True)
    monkeypatch.setattr(paths, "root", lambda: installed)

    with pytest.raises(ConfigError) as exc:
        paths.ensure_workspace()
    assert "site-packages" in str(exc.value)
    assert "GRAD_ROOT" in (exc.value.fix or "")
    # ...and it refuses *before* creating anything, which is the whole point.
    assert not (installed / "ledger").exists()


def test_dist_packages_counts_too(tmp_path, pointer, no_env, monkeypatch):
    """Debian renames the directory, and the mistake is identical there."""
    installed = tmp_path / "usr" / "lib" / "python3" / "dist-packages" / "grad"
    installed.mkdir(parents=True)
    monkeypatch.setattr(paths, "root", lambda: installed)
    with pytest.raises(ConfigError):
        paths.ensure_workspace()


def test_an_explicit_root_is_a_deliberate_answer_and_is_left_alone(tmp_path, monkeypatch):
    """A check that fires on GRAD_ROOT would break the one escape hatch its own
    error message recommends -- and the test suite, which points GRAD_ROOT at a
    temp directory that could be anywhere."""
    chosen = tmp_path / "site-packages" / "research"
    chosen.mkdir(parents=True)
    monkeypatch.setenv("GRAD_ROOT", str(chosen))
    paths.ensure_workspace()
    assert (chosen / "ledger").is_dir()


def test_an_ordinary_checkout_is_not_refused(workspace):
    """The direction that would cost more if it were wrong: a false refusal here
    is Grad declining to start at all."""
    paths.check_not_installed_copy()
