"""The workspace: one poll, one snapshot, layout that persists per project.

The old app gave every panel its own refresh button and its own read of the
ledger. Eleven windows on that pattern is eleven pollers doing eleven full
subtree rebuilds. What replaced it has two properties worth holding still: a
tick only redraws the windows whose data actually changed, and one window's
failure cannot stop the other ten redrawing.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core import ledger_store as ls
from ui import layout as layout_mod, registry, state as state_mod


class FakeSession:
    busy = False
    buffer = ""
    settled: list = []

    def interrupt(self) -> None:
        pass


def workspace_for(project: str | None = "proj") -> state_mod.Workspace:
    return state_mod.Workspace(FakeSession(), project)


# ---------------------------------------------------------------------------
# layout persistence
# ---------------------------------------------------------------------------
def test_a_fresh_workspace_opens_the_default_arrangement(workspace, monkeypatch):
    """On a *configured* machine. A workspace that is still mid-setup opens the
    setup window as well -- see the test below, and `state.opening_windows`."""
    from ui import models as models_mod

    monkeypatch.setattr(models_mod, "first_run_needed", lambda: False)
    space = workspace_for()
    assert set(space.layout.windows) == set(registry.defaults())


def test_a_workspace_that_is_still_mid_setup_opens_the_window_that_fixes_it(
    workspace, monkeypatch
):
    """Widened from "no token" to "no token or no project": a workspace with a
    token and nothing to charge a run to opens four windows, three of them empty
    for a reason nothing on screen was saying."""
    from ui import models as models_mod

    monkeypatch.setattr(models_mod, "first_run_needed", lambda: True)
    space = workspace_for()
    assert "setup" in space.layout.windows


def test_the_layout_persists_per_project(workspace):
    a = workspace_for("alpha")
    a.open("funnel")
    b = workspace_for("beta")
    assert not b.layout.is_open("funnel")
    assert workspace_for("alpha").layout.is_open("funnel")


def test_a_project_id_with_path_separators_cannot_escape_the_layout_directory(workspace):
    """The project id reaches this from a config file, so it is untrusted input
    to a filename."""
    path = state_mod.layout_path("../../etc/passwd")
    assert path.parent == state_mod.layout_dir()
    assert ".." not in path.name


def test_an_unreadable_layout_file_falls_back_to_the_default(workspace, monkeypatch):
    from ui import models as models_mod

    monkeypatch.setattr(models_mod, "first_run_needed", lambda: False)
    state_mod.layout_dir().mkdir(parents=True, exist_ok=True)
    state_mod.layout_path("proj").write_text("{ not json", encoding="utf-8")
    assert set(workspace_for("proj").layout.windows) == set(registry.defaults())


def test_a_saved_layout_survives_a_reconnect(workspace):
    space = workspace_for()
    space.preset("stack")
    space.close("quota")
    reopened = workspace_for()
    assert not reopened.layout.is_open("quota")
    assert len(reopened.layout.columns) == 1


def test_a_resize_is_saved_without_a_redraw(workspace):
    """The browser already moved the panes; rebuilding the tree here would throw
    away the gesture's own result mid-drag."""
    space = workspace_for()
    redraws: list[str] = []
    space.bind_retile(lambda: redraws.append("retile"))
    space.resize("columns", [0.5] * len(space.layout.columns), total_px=1600)
    assert redraws == []
    assert state_mod.layout_path("proj").exists()


# ---------------------------------------------------------------------------
# the poll
# ---------------------------------------------------------------------------
def test_a_tick_redraws_only_what_changed(workspace):
    space = workspace_for()
    space.layout = layout_mod.Layout().open("ledger").open("queue")
    drawn: list[str] = []
    for window in ("ledger", "queue"):
        space.bind_window(window, lambda w=window: drawn.append(w))

    space.tick()          # first pass: both models are new
    drawn.clear()
    space.tick()          # nothing on disk moved
    assert drawn == []

    ls.append_expectation(
        {"id": ls.new_id("exp"), "task": "t", "created_at": ls.now_iso(), "quantity": "q",
         "claim": "c", "predicted": {"low": None, "high": None, "direction": "decrease"},
         "basis": [], "comparability": "", "confidence": "low"}
    )
    space.tick()
    assert drawn == ["ledger"]


def test_a_closed_window_is_not_polled(workspace):
    space = workspace_for()
    space.layout = layout_mod.Layout().open("ledger")
    space.tick()
    assert "ledger" in space.models
    assert "queue" not in space.models


def test_a_window_whose_redraw_raises_does_not_stop_the_others(workspace):
    """A traceback in one window must not leave the other ten frozen."""
    space = workspace_for()
    space.layout = layout_mod.Layout().open("ledger").open("queue")
    drawn: list[str] = []
    space.bind_window("ledger", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    space.bind_window("queue", lambda: drawn.append("queue"))
    space.tick()
    assert drawn == ["queue"]


def test_a_model_builder_that_raises_becomes_an_error_in_the_model(workspace, monkeypatch):
    space = workspace_for()
    space.layout = layout_mod.Layout().open("ledger")
    monkeypatch.setitem(
        state_mod.MODEL_BUILDERS, "ledger", lambda w: (_ for _ in ()).throw(ValueError("nope"))
    )
    space.rebuild("ledger")
    assert "nope" in space.models["ledger"]["error"]


def test_selecting_a_filter_forces_that_window_to_recompute(workspace):
    space = workspace_for()
    space.layout = layout_mod.Layout().open("papers")
    space.tick()
    drawn: list[str] = []
    space.bind_window("papers", lambda: drawn.append("papers"))
    space.select("papers.filter", "queued")
    assert drawn == ["papers"]
    assert space.models["papers"]["filter"] == "queued"


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------
def test_opening_a_window_redraws_the_chrome(workspace):
    space = workspace_for()
    drawn: list[str] = []
    space.bind_chrome(lambda: drawn.append("chrome"))
    space.open("funnel")
    assert drawn


def test_focusing_the_already_focused_window_does_not_rewrite_the_layout(workspace):
    space = workspace_for()
    space.layout.focus("chat")
    calls: list[str] = []
    space.bind_retile(lambda: calls.append("retile"))
    space.focus("chat")
    assert calls == []


def test_an_unknown_preset_is_ignored_rather_than_raising(workspace):
    space = workspace_for()
    before = [c.windows for c in space.layout.columns]
    space.preset("cascade")
    assert [c.windows for c in space.layout.columns] == before


# ---------------------------------------------------------------------------
# the CLI bridge
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_tool_parses_the_json_envelope(workspace):
    payload = await state_mod.run_tool("json.tool", "--help")
    assert isinstance(payload, dict)
    assert "ok" in payload


@pytest.mark.asyncio
async def test_run_tool_reports_a_command_that_produced_nothing_usable(workspace):
    payload = await state_mod.run_tool("this_module_does_not_exist_at_all")
    assert payload["ok"] is False
    assert payload["error"]["message"]


@pytest.mark.asyncio
async def test_spawn_holds_a_reference_until_the_task_settles(workspace):
    """asyncio keeps only a *weak* reference to a running task, so a bare
    `create_task` whose result nobody holds can vanish part-way through."""
    space = workspace_for()
    started = asyncio.Event()
    release = asyncio.Event()

    async def work() -> None:
        started.set()
        await release.wait()

    space.spawn(work(), "unit work")
    await started.wait()
    assert len(space._tasks) == 1

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert space._tasks == set()


@pytest.mark.asyncio
async def test_a_failed_task_reaches_the_log_and_the_status_bar(workspace, caplog):
    """`t.exception()` in a done-callback silences Python's warning by
    *discarding* the error, which is worse than the warning it suppresses: a
    gate approval that failed inside the SDK becomes invisible."""

    async def boom() -> None:
        raise RuntimeError("the SDK said no")

    space = workspace_for()
    with caplog.at_level("ERROR", logger="grad.ui"):
        space.spawn(boom(), "gate answer")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert "gate answer" in (space.notice or "")
    assert "RuntimeError" in space.notice
    # The message itself stays out of the status bar; an SDK message can carry
    # a URL with a token in it.
    assert "the SDK said no" not in space.notice
    assert any("gate answer failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_cancelled_task_is_not_reported_as_a_failure(workspace):
    async def forever() -> None:
        await asyncio.Event().wait()

    space = workspace_for()
    task = space.spawn(forever(), "unit work")
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert space.notice is None


def test_envelope_message_prefers_the_fix():
    assert state_mod.envelope_message({"ok": True}) == "done"
    message = state_mod.envelope_message(
        {"ok": False, "error": {"message": "gate refused", "fix": "run preflight"}}
    )
    assert "gate refused" in message
    assert "run preflight" in message


# ---------------------------------------------------------------------------
# switching the workspace folder
# ---------------------------------------------------------------------------
class RebindableSession(FakeSession):
    """A session that records the two things a folder switch must do to it."""

    def __init__(self) -> None:
        self.settled = [{"role": "user", "text": "from the old workspace"}]
        self.rebound = 0

    async def rebind(self) -> None:
        self.rebound += 1
        self.settled.clear()


@pytest.fixture
def pointer(tmp_path, monkeypatch):
    """Redirect the pointer file. The real one lives beside the code, and a test
    that wrote it would rewrite the developer's own workspace choice."""
    from core import workspace as workspace_mod

    monkeypatch.setattr(
        workspace_mod, "pointer_path", lambda: tmp_path / "pointer" / "p.json"
    )
    (tmp_path / "pointer").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workspace_mod, "_cache", None, raising=False)
    yield
    workspace_mod._cache = None


@pytest.mark.asyncio
async def test_switching_folder_moves_the_paths_the_config_and_the_session(
    workspace, tmp_path, pointer
):
    """The three things that have to move together. Missing any one leaves the
    app half-switched in a way that is hard to see from the screen."""
    from core import config as config_mod, paths

    elsewhere = tmp_path / "another-workspace"
    space = state_mod.Workspace(RebindableSession(), "proj")
    config_mod.load()  # populate the cache the switch has to clear

    await space.switch_root(str(elsewhere), create=True)

    assert paths.root() == elsewhere.resolve()          # the paths
    assert not config_mod._cache                        # the config, which moved with them
    assert space.session.rebound == 1                   # the session and its agent cwd
    assert space.session.settled == []
    # The new folder is a workspace, not just a directory.
    assert paths.ledger_dir().is_dir()
    assert str(elsewhere.resolve()) in (space.notice or "")


@pytest.mark.asyncio
async def test_switching_folder_reloads_the_project_and_its_layout(workspace, tmp_path, pointer):
    from core import budget as budget_mod, paths

    space = state_mod.Workspace(RebindableSession(), None)
    space.preset("stack")
    stacked = [c.windows for c in space.layout.columns]

    elsewhere = tmp_path / "another-workspace"
    await space.switch_root(str(elsewhere), create=True)

    # A fresh folder has no project selected and no saved layout, so both come
    # back to their defaults rather than carrying over from the old workspace.
    assert budget_mod.current_project() is None
    assert space.project is None
    assert [c.windows for c in space.layout.columns] != stacked
    # The layout followed the switch without living in the workspace. Both
    # halves matter: it is keyed to the new root (so the arrangement above came
    # back to its default rather than carrying over), and it resolves under the
    # app directory (so a pane arrangement is never a file somebody commits).
    from core import appdata

    assert state_mod.layout_dir() == appdata.workspace_state_dir() / "layouts"
    assert appdata.app_dir() in state_mod.layout_dir().parents
    assert paths.root() not in state_mod.layout_dir().parents


@pytest.mark.asyncio
async def test_a_folder_that_cannot_be_used_is_a_notice_not_a_crash(workspace, tmp_path, pointer):
    """The path comes from a text field, so a bad one is an everyday event. The
    app must stay where it is and say why."""
    from core import paths

    before = paths.root()
    space = state_mod.Workspace(RebindableSession(), "proj")

    a_file = tmp_path / "notes.txt"
    a_file.write_text("hi", encoding="utf-8")
    await space.switch_root(str(a_file))

    assert paths.root() == before
    assert space.session.rebound == 0        # nothing was torn down
    assert "not a folder" in (space.notice or "")


@pytest.mark.asyncio
async def test_a_blank_folder_does_not_move_the_workspace(workspace, pointer):
    from core import paths

    before = paths.root()
    space = state_mod.Workspace(RebindableSession(), "proj")
    await space.switch_root("   ")
    assert paths.root() == before
    assert space.notice


@pytest.mark.asyncio
async def test_creating_a_project_selects_it_and_reloads(workspace, pointer):
    """Through the CLI, like every other button that does something -- so it
    lands in the same ledger the agent's own `tools.budget new` would write."""
    from core import budget as budget_mod

    space = state_mod.Workspace(RebindableSession(), None)
    await space.create_project("proj-new", "A new piece of research")

    assert budget_mod.current_project() == "proj-new"
    assert budget_mod.projects()["proj-new"]["title"] == "A new piece of research"
    assert space.project == "proj-new"


@pytest.mark.asyncio
async def test_a_refused_project_id_reports_the_clis_own_message(workspace, pointer):
    space = state_mod.Workspace(RebindableSession(), None)
    await space.create_project("not a valid slug!", "whatever")
    assert space.notice and "slug" in space.notice.lower()
    assert space.project is None


@pytest.mark.asyncio
async def test_using_a_project_switches_the_selection(workspace, pointer):
    from core import budget as budget_mod

    budget_mod.create("proj-a", title="A", budget={})
    budget_mod.create("proj-b", title="B", budget={})
    budget_mod.set_current("proj-a")

    space = state_mod.Workspace(RebindableSession(), "proj-a")
    await space.use_project("proj-b")
    assert budget_mod.current_project() == "proj-b"
    assert space.project == "proj-b"
