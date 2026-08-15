"""Being a desktop app: where state lives, one instance, and a poll that yields.

These cover the parts that only show up on a real machine -- a second
double-click of the shortcut, a Lab server left running from a previous port, a
socket that has stopped answering while the window is open. None of them starts
a server or a real process; §24's discipline holds.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from core import appdata, instance, paths
from tools import lab as lab_tool
from ui import desktop, models, render, sessions, state as state_mod


# ---------------------------------------------------------------------------
# where state lives
# ---------------------------------------------------------------------------
def test_app_state_resolves_outside_the_workspace(workspace):
    """The whole point of the split: nothing the app writes for its own
    convenience lands in a folder that gets committed."""
    root = paths.root()
    for path in (
        appdata.state_dir(),
        appdata.logs_dir(),
        appdata.cache_dir(),
        appdata.lock_path(),
        appdata.workspace_state_dir(),
        state_mod.layout_dir(),
    ):
        assert root not in path.parents, f"{path} is inside the workspace"


def test_research_paths_stay_in_the_workspace(workspace):
    """The other half, and the one that matters more. A report's claim traces to
    a run record because the record sits next to it; moving the ledger into
    AppData would break that chain and make the repository not self-describing.
    """
    root = paths.root()
    for path in (
        paths.ledger_dir(),
        paths.runs_path(),
        paths.notebooks_dir(),
        paths.notes_dir(),
        paths.figures_dir(),
        paths.papers_dir(),
    ):
        assert root in path.parents or path == root


def test_two_workspaces_get_different_app_state(workspace, tmp_path, monkeypatch):
    """Transcripts and layouts are per-workspace. One flat directory would hand
    every folder the same conversation and the same panes."""
    first = appdata.workspace_state_dir()
    monkeypatch.setenv("GRAD_ROOT", str(tmp_path / "elsewhere"))
    assert appdata.workspace_state_dir() != first


def test_workspaces_with_the_same_name_do_not_collide(tmp_path):
    """`D:/work/grad` and `C:/old/grad` are different workspaces with one name,
    which a readable-stem-only key would merge."""
    a = appdata.workspace_state_dir(tmp_path / "one" / "grad")
    b = appdata.workspace_state_dir(tmp_path / "two" / "grad")
    assert a != b
    assert a.name.startswith("grad-") and b.name.startswith("grad-")


def test_the_workspace_key_is_stable_across_calls(tmp_path):
    """It names a directory holding transcripts; a key that moved would orphan
    them on every launch."""
    target = tmp_path / "grad"
    assert appdata.workspace_state_dir(target) == appdata.workspace_state_dir(target)


def test_migration_moves_app_state_and_leaves_research_alone(workspace):
    """An existing workspace predates the split. Its layouts should move; its
    papers and datasets are cited or expensive and must not."""
    legacy = workspace / "data"
    (legacy / "layouts").mkdir(parents=True, exist_ok=True)
    (legacy / "layouts" / "proj.json").write_text("{}", encoding="utf-8")
    (legacy / "papers").mkdir(parents=True, exist_ok=True)
    (legacy / "papers" / "a.pdf").write_bytes(b"%PDF-")

    moved = appdata.migrate_legacy(workspace)

    assert "layouts" in moved
    # Where `ui/state.py:layout_dir` actually reads. A migration that lands
    # anywhere else is worse than none: the old copy is gone and the app opens
    # on defaults with no error to explain it.
    assert (appdata.workspace_state_dir() / "layouts" / "proj.json").exists()
    assert not (legacy / "layouts").exists()
    # Untouched, and named in the assertion so deleting it from _LEGACY is a
    # deliberate act rather than a silent one.
    assert (legacy / "papers" / "a.pdf").exists()


def test_every_migration_target_is_where_its_reader_looks(workspace):
    """The bug this guards is silent by construction, so it is asserted against
    the readers rather than against a remembered path."""
    from tools import nb as nb_tool

    (workspace / "data" / "layouts").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "layouts" / "p.json").write_text("{}", encoding="utf-8")
    (workspace / "data" / "kernel").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "kernel" / "default.json").write_text("{}", encoding="utf-8")
    (workspace / "data" / "lab").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "lab" / "lab.json").write_text("{}", encoding="utf-8")
    (workspace / "data" / "ui_session-1.jsonl").write_text("{}\n", encoding="utf-8")

    appdata.migrate_legacy(workspace)

    assert (state_mod.layout_dir() / "p.json").exists()
    assert nb_tool._conn_path("default").exists()
    assert lab_tool._state_path().exists()
    assert (sessions.sessions_dir() / "ui_session-1.jsonl").exists()


def test_the_cache_is_actually_relocated(workspace):
    """`ensure()` used to create the cache directory, so the "destination
    already exists" guard fired on every run and `data/cache` never moved."""
    legacy = workspace / "data" / "cache"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "abc.json").write_text("{}", encoding="utf-8")

    assert "cache" in appdata.migrate_legacy(workspace)
    assert (paths.cache_dir() / "abc.json").exists()
    assert not legacy.exists()


def test_transcripts_are_migrated_with_everything_else(workspace):
    """They sit loose at the top of `data/` rather than in a subdirectory, so a
    pass that only walked directories left the whole conversation history
    behind -- still on disk, still private, no longer anywhere the app looks."""
    (workspace / "data").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "ui_session-20260815-abcd.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace / "data" / "ui_storage_secret").write_text("s3cret", encoding="utf-8")
    # Evidence about the research rather than state about the machine: it stays
    # with the notebooks it describes.
    (workspace / "data" / "nb_verify.json").write_text("{}", encoding="utf-8")

    appdata.migrate_legacy(workspace)

    assert (sessions.sessions_dir() / "ui_session-20260815-abcd.jsonl").exists()
    assert not (workspace / "data" / "ui_session-20260815-abcd.jsonl").exists()
    assert (appdata.state_dir() / "ui_storage_secret").read_text(encoding="utf-8") == "s3cret"
    assert (workspace / "data" / "nb_verify.json").exists()


def test_a_locked_file_is_copied_and_left_rather_than_lost(workspace, monkeypatch):
    """The normal state when this runs: a Lab server is up and holding its own
    log open. A wholesale directory move fails on that handle and its fallback
    can delete some sources after copying them and abort on the locked one --
    leaving files neither here nor there. Copy, promote, then delete.

    The lock is simulated rather than taken. Holding a real handle only blocks
    the unlink on Windows -- POSIX unlinks open files happily -- so a test built
    on one would assert nothing at all on the platform it did not run on.
    """
    legacy = workspace / "data" / "lab"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "lab.json").write_text('{"port": 8889}', encoding="utf-8")
    held = legacy / "lab.log"
    held.write_text("serving\n", encoding="utf-8")

    real_unlink = Path.unlink

    def refuse(self, *args, **kwargs):
        if self.name == "lab.log":
            raise PermissionError(32, "in use by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)

    assert "lab" in appdata.migrate_legacy(workspace)
    # Everything arrived, including the file that could not be removed.
    assert (appdata.state_dir() / "lab" / "lab.json").exists()
    assert (appdata.state_dir() / "lab" / "lab.log").read_text(encoding="utf-8") == "serving\n"
    # And nothing was destroyed on the way: the locked original survives.
    assert held.exists()


def test_a_failed_migration_leaves_the_sources_untouched(workspace, monkeypatch):
    """Half a migration is worse than none. If the copy cannot finish, the
    workspace must look exactly as it did before."""
    legacy = workspace / "data" / "layouts"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "proj.json").write_text('{"a": 1}', encoding="utf-8")

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(appdata.shutil, "copy2", explode)
    assert appdata.migrate_legacy(workspace) == []
    assert (legacy / "proj.json").read_text(encoding="utf-8") == '{"a": 1}'
    assert not (appdata.workspace_state_dir() / "layouts").exists()
    assert not (appdata.workspace_state_dir() / "layouts.incoming").exists()


def test_migration_does_not_overwrite_the_live_location(workspace):
    """A second run, or a workspace opened after the app has already written
    layouts, must not have stale state resurrected over the current state."""
    (appdata.state_dir() / "layouts").mkdir(parents=True, exist_ok=True)
    (appdata.state_dir() / "layouts" / "proj.json").write_text('{"live": 1}', encoding="utf-8")
    legacy = workspace / "data" / "layouts"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "proj.json").write_text('{"stale": 1}', encoding="utf-8")

    appdata.migrate_legacy(workspace)

    assert "live" in (appdata.state_dir() / "layouts" / "proj.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# one instance
# ---------------------------------------------------------------------------
def test_a_second_instance_cannot_take_the_lock(workspace):
    """Two workspaces would fight over the layout file, the transcript directory
    and Lab's recorded origin -- and the second would bind a different port,
    which is exactly the mismatch that stops Lab embedding."""
    first = instance._Lock()
    assert first.acquire() is True
    second = instance._Lock()
    try:
        assert second.acquire() is False
    finally:
        first.release()


def test_the_lock_is_released_for_the_next_launch(workspace):
    """Held by the OS, not by a pid file: a crash must not leave an app that
    refuses to start until someone deletes a file."""
    first = instance._Lock()
    assert first.acquire() is True
    first.release()
    second = instance._Lock()
    assert second.acquire() is True
    second.release()


def test_the_published_port_survives_for_the_handover(workspace):
    """The app takes the first free port at or above 8080, so a second launch
    cannot guess where the first one is listening."""
    instance.publish(8123)
    assert instance.read_state()["port"] == 8123
    instance.clear()
    assert instance.read_state() == {}


def test_a_missing_state_file_is_not_an_error(workspace):
    assert instance.read_state() == {}
    assert instance.show_running({}) is False


# ---------------------------------------------------------------------------
# ports and origins
# ---------------------------------------------------------------------------
def test_an_explicit_port_is_honoured_even_if_it_looks_busy():
    """`--port` is someone overriding the picker on purpose."""
    assert desktop.choose_port(9321) == 9321


def test_the_chosen_port_walks_up_from_the_default():
    chosen = desktop.choose_port(None)
    assert desktop.DEFAULT_PORT <= chosen < desktop.DEFAULT_PORT + desktop.PORT_SPAN


def test_a_lab_on_another_port_is_a_mismatch(monkeypatch):
    """Lab bakes `frame-ancestors` at launch, so an app that has moved ports
    cannot embed it -- and the browser calls that "refused to connect"."""
    monkeypatch.setattr(models, "app_port", lambda: 8081)
    assert models.origin_mismatch({"running": True, "ui_origin": "http://127.0.0.1:8080"}) is True


def test_the_localhost_alias_is_not_a_mismatch(monkeypatch):
    """The Jupyter config allows both spellings on purpose, because a browser
    treats them as different origins and which one the window opened on is not
    Lab's business. Flagging it would put a banner on a working server."""
    monkeypatch.setattr(models, "app_port", lambda: 8080)
    assert models.origin_mismatch({"running": True, "ui_origin": "http://localhost:8080"}) is False
    assert models.origin_mismatch({"running": True, "ui_origin": "http://127.0.0.1:8080"}) is False


def test_a_stopped_lab_is_never_a_mismatch(monkeypatch):
    """There is a different, better message for "not running"; two banners for
    one condition is worse than one."""
    monkeypatch.setattr(models, "app_port", lambda: 9999)
    assert models.origin_mismatch({"running": False, "ui_origin": "http://127.0.0.1:8080"}) is False


# ---------------------------------------------------------------------------
# quitting while work is in flight
# ---------------------------------------------------------------------------
def test_nothing_running_is_not_busy(workspace, monkeypatch):
    monkeypatch.setattr(desktop, "_lab_busy", lambda: [])
    report = desktop.busy_report()
    assert report["busy"] is False
    assert desktop.busy_sentence(report) == "Nothing is running."


def test_a_running_command_makes_quitting_a_question(workspace, monkeypatch):
    """`nb verify` spawns a kernel detached and can hold a GPU for half an hour.
    Losing it to a stray click on Quit is a real cost."""
    monkeypatch.setattr(desktop, "_lab_busy", lambda: [])

    class FakeTask:
        label = "verify 03-optimizers.ipynb"

    monkeypatch.setattr("ui.tasks.running", lambda: [FakeTask()])
    report = desktop.busy_report()
    assert report["busy"] is True
    assert "verify 03-optimizers.ipynb" in desktop.busy_sentence(report)


def test_a_busy_lab_kernel_counts_too(workspace, monkeypatch):
    """Lab's kernels are invisible to `ui/tasks.py` -- different owner, separate
    process -- so they have to be asked about separately."""
    monkeypatch.setattr(desktop, "_lab_busy", lambda: ["python3"])
    monkeypatch.setattr("ui.tasks.running", lambda: [])
    report = desktop.busy_report()
    assert report["busy"] is True
    assert "executing a cell" in desktop.busy_sentence(report)


def test_an_unreachable_lab_does_not_block_the_quit(workspace, monkeypatch):
    """A wedged server must not hang the prompt that asks about it."""
    monkeypatch.setattr(
        "tools.lab.lab_state", lambda: {"running": True, "port": 1, "token": "x"}
    )
    assert desktop._lab_busy() == []


# ---------------------------------------------------------------------------
# the poll
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_poll_builds_models_off_the_event_loop(workspace):
    """`tools/lab.py:_listening` probes a socket with a 0.4s timeout on this
    path, while the notebook window is open. On the loop that is a 400ms freeze
    of every window, the chat stream and the pane dragging."""
    space = state_mod.Workspace(_FakeSession(), "proj")
    seen: dict[str, int] = {}

    def builder(_workspace):
        seen["thread"] = threading.get_ident()
        return {"value": 1}

    space.layout = _only("ledger", space)
    monkey = dict(state_mod.MODEL_BUILDERS)
    monkey["ledger"] = builder
    state_mod.MODEL_BUILDERS.update(monkey)
    try:
        await space.poll()
    finally:
        state_mod.MODEL_BUILDERS.update({"ledger": lambda w: models.ledger_model()})

    assert seen["thread"] != threading.get_ident()


@pytest.mark.asyncio
async def test_a_slow_poll_does_not_stack(workspace):
    """A pass that outruns POLL_SECONDS is a slow disk or a hung port. Queueing
    the next one turns one slow poll into an unbounded pile of them."""
    space = state_mod.Workspace(_FakeSession(), "proj")
    space._ticking = True
    await space.poll()  # returns immediately rather than running a pass
    assert space.models == {}


@pytest.mark.asyncio
async def test_the_loop_bound_window_is_not_offloaded(workspace):
    """`tasks` reads the live SDK session rather than a file. There is no I/O to
    move, and reading it from a worker thread while the loop mutates it is a
    race for no gain."""
    assert "tasks" in state_mod.LOOP_BOUND


# ---------------------------------------------------------------------------
# the read-only render
# ---------------------------------------------------------------------------
def _notebook(workspace, name: str = "n.ipynb", source: str = "print(1)"):
    import nbformat

    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    paths.notebooks_dir().mkdir(parents=True, exist_ok=True)
    target = paths.notebooks_dir() / name
    nbformat.write(nb, target)
    return target


def test_the_render_carries_no_script(workspace):
    """It is stored output from a file that may have been cloned rather than
    written here, shown in a sandboxed frame. The `basic` template is what makes
    that possible; the `lab` one ships the JavaScript that draws it."""
    _notebook(workspace)
    body = render.notebook_html("n.ipynb")
    # Pygments splits the source across spans, so the cell is asserted by its
    # tokens rather than by its text.
    assert "print" in body and "highlight" in body
    assert "<script" not in body.lower()


def test_the_render_refuses_anything_but_a_notebook_here(workspace):
    """The name arrives from an HTTP path on an unauthenticated local port."""
    _notebook(workspace)
    for bad in ("../grad.toml", "sub/n.ipynb", "n.txt", "", "..%2Fx.ipynb"):
        with pytest.raises(render.NotAllowed):
            render.resolve(bad)


def test_the_render_follows_the_file(workspace):
    """An edit in Lab has to show up here. The cache is keyed on the file's
    identity rather than on a clock, so it does -- and an unchanged file is not
    re-rendered on every one of the poll's redraws."""
    target = _notebook(workspace, source="print('first')")
    first = render.notebook_html("n.ipynb")
    assert render.notebook_html("n.ipynb") is first  # unchanged: same object

    import nbformat  # noqa: PLC0415

    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('second')")])
    nbformat.write(nb, target)
    import os  # noqa: PLC0415

    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 10))
    assert "second" in render.notebook_html("n.ipynb")


def test_a_notebook_being_written_is_a_message_not_a_crash(workspace):
    """The agent writes notebooks while the pane is open, so a half-written file
    is an ordinary event rather than an error state."""
    paths.notebooks_dir().mkdir(parents=True, exist_ok=True)
    (paths.notebooks_dir() / "half.ipynb").write_text('{"cells": [', encoding="utf-8")
    body = render.notebook_html("half.ipynb")
    assert "could not be rendered" in body


def test_the_lab_window_gets_a_workspace_of_its_own(workspace):
    """Two Lab clients on one server-side workspace do not cooperate: Lab
    detects the collision and reloads, which drops the kernel connection of
    whichever client was mid-cell. A browser tab keeps the default."""
    state = {"lab_running": True, "lab_port": 8889, "lab_token": "t"}
    app_url = models.lab_url(state, "n.ipynb", lab_workspace=models.APP_LAB_WORKSPACE)
    tab_url = models.lab_url(state, "n.ipynb")
    assert f"/lab/workspaces/{models.APP_LAB_WORKSPACE}/" in app_url
    assert "/workspaces/" not in tab_url
    assert app_url != tab_url


class _FakeSession:
    busy = False
    buffer = ""
    settled: list = []

    def interrupt(self) -> None:
        pass


def _only(window_id: str, space):
    """A layout holding one window, so a poll rebuilds exactly one model."""
    from ui import layout as layout_mod

    return layout_mod.Layout.default([window_id])
