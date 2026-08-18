"""The first run: what a fresh machine is told it still needs.

The thing under test is mostly a *decision*, not a widget -- whether this
workspace is mid-setup, which is the question that decides what opens on a
machine with no saved layout. It is derived from three conditions rather than
stored as a flag, and the tests below are largely about that choice: a stored
flag is wrong in both directions, and both are asserted here.
"""

from __future__ import annotations

import pytest

from core import budget, credentials
from ui import models, state as state_mod


@pytest.fixture(autouse=True)
def bare_machine(monkeypatch):
    """A machine with nothing in its credential store.

    `conftest.py` isolates the workspace (`GRAD_ROOT`) and the app directory
    (`GRAD_APP_DIR`) and stops there -- the OS keyring is neither, so without
    this every assertion here would be about the *developer's* stored
    credentials. It showed up as "a fresh workspace already has a backend
    configured", which is true of this machine and of no fresh install.
    """
    stored: dict[str, bool] = {}
    monkeypatch.setattr(credentials, "status", lambda: dict(stored))
    monkeypatch.setattr(credentials, "present", lambda name: bool(stored.get(name)))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    return stored


def _no_token(monkeypatch):
    monkeypatch.setattr(models, "setup_needed", lambda: True)


def _token(monkeypatch):
    monkeypatch.setattr(models, "setup_needed", lambda: False)


def _project(name: str = "proj-1") -> None:
    budget.create(name, title="a project", budget={"gpu_usd": 10.0})
    budget.set_current(name)


# ---------------------------------------------------------------------------
# when it is active
# ---------------------------------------------------------------------------
def test_a_fresh_workspace_is_mid_setup(workspace, monkeypatch):
    _no_token(monkeypatch)
    run = models.first_run()

    assert run["active"] is True
    assert run["next"]["id"] == "token"
    assert run["done"] == 0


def test_a_token_alone_is_not_a_configured_workspace(workspace, monkeypatch):
    """The condition this widened. A token and no project opens four windows,
    three of which are empty because there is nothing to file against -- and
    nothing on screen used to say that was the reason."""
    _token(monkeypatch)
    run = models.first_run()

    assert run["active"] is True
    assert run["next"]["id"] == "project"


def test_a_token_and_a_project_are_enough(workspace, monkeypatch):
    _token(monkeypatch)
    _project()
    run = models.first_run()

    assert run["active"] is False
    # The backend step is still *undone*, and that is not the same as unfinished
    # setup -- it is reported, and it does not hold the panel open.
    assert run["next"]["id"] == "backend"
    assert [s["done"] for s in run["steps"]] == [True, True, False]


def test_an_unconfigured_backend_never_holds_the_panel_open(workspace, monkeypatch):
    """Remote training is a real limitation and not a reason to put a wizard in
    front of someone who opened the app to read a ledger."""
    _token(monkeypatch)
    _project()
    backend = next(s for s in models.first_run()["steps"] if s["id"] == "backend")
    assert backend["blocking"] is False


def test_a_selected_project_that_no_longer_exists_does_not_count(workspace, monkeypatch):
    """The selection file is machine-local state and outlives the project it
    names. A dangling pointer is not something for a run to be charged to."""
    _token(monkeypatch)
    budget.set_current("proj-that-was-deleted")

    assert models.first_run()["next"]["id"] == "project"


# ---------------------------------------------------------------------------
# why it is derived and not stored
# ---------------------------------------------------------------------------
def test_it_goes_away_by_being_satisfied_rather_than_dismissed(workspace, monkeypatch):
    """A `first_run_done` flag is wrong in both directions. This is the first:
    dismissing it on a machine that still has no project would hide the panel
    that says so, permanently, with nothing to bring it back."""
    _token(monkeypatch)
    assert models.first_run()["active"] is True

    _project()
    assert models.first_run()["active"] is False


def test_a_second_workspace_gets_its_own_answer(workspace, monkeypatch, tmp_path):
    """And this is the other direction: a flag stored per machine would leave a
    brand-new workspace with no panel because a *different* one once set it."""
    _token(monkeypatch)
    _project()
    assert models.first_run()["active"] is False

    other = tmp_path.parent / f"{tmp_path.name}-second"
    other.mkdir(exist_ok=True)
    monkeypatch.setenv("GRAD_ROOT", str(other))
    from core import config

    config._cache.clear()
    assert models.first_run()["active"] is True, "a fresh workspace is fresh"


def test_nothing_here_is_persisted(workspace, monkeypatch):
    """Stated as an assertion because the tempting fix for every bug above is to
    add a flag, and the flag is the bug."""
    from core import settings

    _token(monkeypatch)
    models.first_run()
    assert "first_run" not in settings.load()


# ---------------------------------------------------------------------------
# what it changes on screen
# ---------------------------------------------------------------------------
def test_a_mid_setup_workspace_opens_the_window_that_fixes_it(workspace, monkeypatch):
    monkeypatch.setattr(models, "first_run_needed", lambda: True)
    assert state_mod.opening_windows()[0] == "setup"


def test_a_configured_workspace_opens_the_ordinary_four(workspace, monkeypatch):
    """`first_run_needed` and not `first_run`: the arrangement asks the cheap
    question, because the full model loads a Config and this path must not --
    see the docstring on `first_run_needed`."""
    from ui import registry

    monkeypatch.setattr(models, "first_run_needed", lambda: False)
    assert state_mod.opening_windows() == registry.defaults()
    assert "setup" not in state_mod.opening_windows()


def test_an_unreadable_machine_still_opens_a_workspace(workspace, monkeypatch):
    """This runs on the path that decides what is on screen at all. A credential
    store that cannot be reached is not a reason to fail to open."""
    def boom():
        raise OSError("no keyring here")

    monkeypatch.setattr(models, "first_run_needed", boom)
    from ui import registry

    assert state_mod.opening_windows() == registry.defaults()


def test_the_model_never_raises_on_a_broken_machine(workspace, monkeypatch):
    """The panel is drawn by the one window whose job is to be usable when
    nothing else is."""
    def boom(*_a, **_k):
        raise RuntimeError("the ledger is on fire")

    monkeypatch.setattr(budget, "current_project", boom)
    _no_token(monkeypatch)

    run = models.first_run()
    assert run["active"] is True
    assert run["steps"][1]["done"] is False


def test_the_setup_model_carries_the_panel(workspace, monkeypatch):
    """One snapshot, not two reads: `authenticate` ticking green above a token
    step that still says missing is the disagreement this avoids."""
    _no_token(monkeypatch)
    model = models.setup_model()

    assert model["first_run"]["active"] is True
    token_step = next(s for s in model["steps"] if s["id"] == "token")
    assert token_step["ready"] is False


# ---------------------------------------------------------------------------
# the backend that was added last
# ---------------------------------------------------------------------------
def test_modal_counts_as_a_backend_once_both_halves_are_stored(workspace, bare_machine):
    """A fifth list of backends, and the one this nearly drifted out of:
    `tools/setup.py:REQUIREMENTS` decides what "a backend is configured" means,
    and a backend missing from it can never satisfy the step."""
    from core import config as config_mod
    from tools import setup as setup_tool

    assert "modal" in setup_tool.REQUIREMENTS

    stored = bare_machine
    ready = {b["backend"]: b for b in setup_tool.readiness(config_mod.load())}
    assert ready["modal"]["ready"] is False
    # Both halves named, so nobody goes round the loop twice.
    assert set(ready["modal"]["missing"]) == {
        credentials.MODAL_TOKEN_ID,
        credentials.MODAL_TOKEN_SECRET,
    }

    stored[credentials.MODAL_TOKEN_ID] = True
    assert setup_tool.readiness(config_mod.load())
    ready = {b["backend"]: b for b in setup_tool.readiness(config_mod.load())}
    assert ready["modal"]["ready"] is False, "half a token pair authenticates nothing"

    stored[credentials.MODAL_TOKEN_SECRET] = True
    ready = {b["backend"]: b for b in setup_tool.readiness(config_mod.load())}
    assert ready["modal"]["ready"] is True


def test_every_backend_that_can_be_chosen_can_be_reported_on(workspace):
    """`settings.BACKENDS` is what the setup window offers and `REQUIREMENTS` is
    what says whether one is ready. A backend in the first and not the second is
    one the window offers and can never mark configured."""
    from core import settings
    from tools import setup as setup_tool

    assert set(settings.BACKENDS) == set(setup_tool.REQUIREMENTS)
