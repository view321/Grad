"""Being a desktop app: where state lives, one instance, and a poll that yields.

These cover the parts that only show up on a real machine -- a second
double-click of the shortcut, a Lab server left running from a previous port, a
socket that has stopped answering while the window is open. None of them starts
a server or a real process; §24's discipline holds.
"""

from __future__ import annotations

import asyncio
import os
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


def test_one_directory_spelled_two_ways_is_one_workspace(tmp_path):
    """The key is a digest of the path's *text*, so an unresolved spelling is a
    different workspace to it. Readers all arrive through `paths.root()`, which
    resolves; `migrate_legacy` takes a root argument and may not -- and a
    migration keyed differently from its reader is the silent kind of loss."""
    target = tmp_path / "grad"
    target.mkdir()
    spellings = [
        target,
        tmp_path / "." / "grad",
        tmp_path / "grad" / "sub" / "..",
        Path(str(target) + os.sep),
    ]
    keys = {appdata.workspace_state_dir(s) for s in spellings}
    assert len(keys) == 1, keys


def test_a_migration_lands_where_an_unresolved_root_reads(tmp_path, monkeypatch):
    """The whole point of resolving: `migrate_legacy` given a scruffy path must
    write where a reader given the tidy one will look."""
    root = tmp_path / "ws"
    (root / "data" / "layouts").mkdir(parents=True)
    (root / "data" / "layouts" / "p.json").write_text("{}", encoding="utf-8")

    appdata.migrate_legacy(tmp_path / "ws" / "sub" / "..")

    monkeypatch.setenv("GRAD_ROOT", str(root))
    assert (state_mod.layout_dir() / "p.json").exists()


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
# closing the window is not quitting
# ---------------------------------------------------------------------------
class _FakeClosingEvent:
    """pywebview's `closing`: a list of handlers, and False means "cancel"."""

    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _FakeWindow:
    def __init__(self, *, hides: bool = True) -> None:
        self.events = type("E", (), {})()
        self.events.closing = _FakeClosingEvent()
        self.hidden = 0
        self._hides = hides

    def hide(self) -> None:
        if not self._hides:
            raise RuntimeError("this window will not hide")
        self.hidden += 1


def _closing_handler(monkeypatch, window):
    """Run `hold_window_open` against a fake pywebview and return its handler.

    The real thing runs in the window's own process, which is the whole reason
    the bug existed: nothing in this process could bind that event, and the
    attempt to raised into a debug log nobody was reading.
    """
    import sys
    import types

    fake = types.ModuleType("webview")
    fake.windows = [window]
    monkeypatch.setitem(sys.modules, "webview", fake)
    desktop.hold_window_open(str(desktop.tray_flag()))
    assert window.events.closing.handlers, "no closing handler was registered"
    return window.events.closing.handlers[0]


def test_closing_the_window_hides_it_when_the_tray_can_bring_it_back(workspace, monkeypatch):
    """The app's second load-bearing decision, and for months it did nothing:
    the close went through, the window process died, and NiceGUI's watchdog took
    the whole app down a second later -- the icon vanishing from the
    notification area is the reported symptom."""
    desktop.set_tray_flag(True)
    window = _FakeWindow()
    handler = _closing_handler(monkeypatch, window)

    assert handler() is False, "the close was not cancelled"
    assert window.hidden == 1


def test_closing_the_window_closes_it_when_there_is_no_way_back(workspace, monkeypatch):
    """A hidden window with no icon is a process holding the port and the
    single-instance lock, unreachable short of Task Manager. Without a tray the
    close is allowed to be a close."""
    desktop.set_tray_flag(False)
    window = _FakeWindow()
    handler = _closing_handler(monkeypatch, window)

    assert handler() is True
    assert window.hidden == 0


def test_a_window_that_will_not_hide_is_allowed_to_close(workspace, monkeypatch):
    """A veto with no hide is a close button that does nothing at all."""
    desktop.set_tray_flag(True)
    window = _FakeWindow(hides=False)
    handler = _closing_handler(monkeypatch, window)

    assert handler() is True


def test_quitting_clears_the_flag_before_the_window_is_destroyed(workspace, monkeypatch):
    """NiceGUI's shutdown calls `destroy()`, pywebview raises `closing` for that
    exactly as for the close button -- so a flag left set would have Quit answer
    by hiding."""
    desktop.set_tray_flag(True)
    monkeypatch.setattr(desktop, "_tray", None)
    desktop._quitting.clear()
    monkeypatch.setattr("core.instance.release", lambda: None)

    desktop.shutdown()
    assert not desktop.tray_flag().exists()
    desktop._quitting.clear()


def test_the_veto_is_actually_handed_to_the_window_process(workspace):
    """The connecting wire, tested because its absence is silent: every other
    test here would pass with `hold_window_open` never reaching a window.

    `_split_picklable` is NiceGUI's own filter -- it drops whatever cannot cross
    the spawn boundary and only warns -- so this asserts the keys survive it
    rather than merely that they were set.
    """
    from nicegui import app as nicegui_app
    from nicegui.native.native_mode import _split_picklable

    from ui import app as grad_app

    nicegui_app.native.start_args.clear()
    grad_app._install_desktop(True)

    kept, dropped = _split_picklable(nicegui_app.native.start_args)
    assert kept.get("func") is desktop.hold_window_open
    assert kept.get("args") == (str(desktop.tray_flag()),)
    assert not dropped
    nicegui_app.native.start_args.clear()


def test_the_window_is_handed_an_icon(workspace):
    """Without one, pywebview falls back to `ExtractIconW(..., sys.executable, 0)`
    and the app sits on the taskbar wearing Python's icon. Measured: the
    fallback rasterises to Python's blue, the icon here to the brand yellow.

    Travels through `_split_picklable` for the reason `func` does -- the form
    that owns the icon lives in the spawned window process, so a value that
    cannot cross the boundary is the same as no icon at all.
    """
    from nicegui import app as nicegui_app
    from nicegui.native.native_mode import _split_picklable

    from ui import app as grad_app

    nicegui_app.native.start_args.clear()
    grad_app._install_desktop(True)

    kept, dropped = _split_picklable(nicegui_app.native.start_args)
    assert Path(kept["icon"]).is_file()
    assert not dropped
    nicegui_app.native.start_args.clear()


def test_the_icon_carries_every_size_windows_asks_for(workspace):
    """Windows picks a size per context -- 16px on the taskbar, 32px on the
    desktop, 256px in the large-icon view -- and an `.ico` carrying one of them
    gets the others by scaling."""
    # Skipped, not errored, without Pillow. It is in the `ui` extra rather than
    # the base install, and `write_icon` already treats its absence as a warning
    # and falls back -- so a suite that errors here reports a missing optional
    # dependency as a broken icon.
    Image = pytest.importorskip("PIL.Image")

    path = desktop.icon_path(refresh=True)
    with Image.open(path) as image:
        assert set(image.ico.sizes()) >= {(16, 16), (32, 32), (48, 48), (256, 256)}
        master = image.ico.getimage((256, 256)).convert("RGB")
        taskbar = image.ico.getimage((32, 32)).convert("RGB")

    # The 256px entry is the drawing itself, so it matches `_icon_image` exactly.
    assert master.getpixel((128, 24)) == (255, 212, 0)
    assert master.getpixel((128, 112)) == (20, 16, 12)

    # 32px is resampled from it, so the colours shift a little. What has to hold
    # is that it is still recognisably this mark and not the interpreter's --
    # Python's icon rasterises to blue, which no tolerance around yellow reaches.
    ground = taskbar.getpixel((16, 3))
    assert ground[0] > 240 and ground[1] > 190 and ground[2] < 40, ground
    glyph = taskbar.getpixel((16, 14))
    assert max(glyph) < 60, glyph


def test_the_shortcut_and_the_window_read_one_file(workspace):
    """`install.ps1` writes `%LOCALAPPDATA%\\Grad\\grad.ico` and the window reads
    it. Two paths would be two marks that drift."""
    from core import appdata

    assert desktop.icon_file() == appdata.app_dir() / "grad.ico"


def test_a_missing_pillow_costs_the_icon_and_not_the_app(workspace, monkeypatch):
    """Pillow is declared by the `ui` extra, and a machine can still be missing
    it -- that is exactly how this file's `start_tray` note came about. Refusing
    to open a window because the icon could not be drawn trades a cosmetic
    problem for a fatal one."""
    monkeypatch.setattr(desktop, "write_icon", _raise_no_pillow)
    target = desktop.icon_file()
    if target.exists():
        target.unlink()

    assert desktop.icon_path() is None

    # And the app still wires up: the key is simply absent.
    from nicegui import app as nicegui_app

    from ui import app as grad_app

    nicegui_app.native.start_args.clear()
    grad_app._install_desktop(True)
    assert "icon" not in nicegui_app.native.start_args
    assert nicegui_app.native.start_args["func"] is desktop.hold_window_open
    nicegui_app.native.start_args.clear()


def _raise_no_pillow(*_args, **_kwargs):
    raise ImportError("No module named 'PIL'")


def test_browser_mode_hands_over_no_window_veto(workspace):
    """There is no window to hide and no tray to hide it to; the tab is the
    affordance and closing it is the user's business."""
    from nicegui import app as nicegui_app

    from ui import app as grad_app

    nicegui_app.native.start_args.clear()
    grad_app._install_desktop(False)
    assert "func" not in nicegui_app.native.start_args


def test_the_close_veto_crosses_the_process_boundary(workspace):
    """It is handed to a *spawned* process, so it travels by pickle: a closure
    over anything here, or a lambda, would fail at window creation -- which is
    the app failing to open at all."""
    import pickle

    payload = {
        "func": desktop.hold_window_open,
        "args": (str(desktop.tray_flag()),),
    }
    assert pickle.loads(pickle.dumps(payload))["func"] is desktop.hold_window_open


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


# ---------------------------------------------------------------------------
# where the window was
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_geometry(monkeypatch):
    """Empty the module's observation of the window between tests.

    `_geometry` is process-wide because there is one window per process, which
    is true in production and would otherwise let one test decide another's
    outcome -- the same reason `clean_process_state` exists in `conftest.py`.
    """
    monkeypatch.setattr(desktop, "_geometry", {})
    monkeypatch.setattr(desktop, "_previous", {})
    monkeypatch.setattr(desktop, "_saved_at", 0.0)
    yield


def _screens(monkeypatch, *rects):
    monkeypatch.setattr(desktop, "_screens", lambda: list(rects))


def test_a_window_nobody_has_moved_opens_at_the_default(workspace, fresh_geometry):
    """No file, no position: the size is the documented default and pywebview is
    left to place it, exactly as before any of this existed."""
    args = desktop.window_args()
    assert (args["width"], args["height"]) == desktop.DEFAULT_SIZE
    assert "x" not in args and "y" not in args


def test_the_window_reopens_where_it_was_left(workspace, fresh_geometry, monkeypatch):
    """The whole feature in one test: move it, and the next launch is told to
    put it back."""
    _screens(monkeypatch, (0, 0, 2560, 1440))
    desktop.remember(x=300, y=120)
    desktop.remember(width=1200, height=800)
    desktop.save_geometry(force=True)

    args = desktop.window_args()
    assert (args["x"], args["y"]) == (300, 120)
    assert (args["width"], args["height"]) == (1200, 800)


def test_a_position_on_a_screen_that_is_gone_is_dropped(workspace, fresh_geometry, monkeypatch):
    """The failure this must not have. A saved position is a promise about a
    monitor arrangement, and undocking breaks the promise -- restoring it puts
    the window where there are no pixels, running and unreachable."""
    _screens(monkeypatch, (0, 0, 2560, 1440))
    desktop.remember(x=3000, y=200, width=1200, height=800)
    desktop.save_geometry(force=True)

    args = desktop.window_args()
    assert "x" not in args and "y" not in args, "restored onto a screen that is not there"
    # The size is still honoured: it was never the part that stopped being true.
    assert (args["width"], args["height"]) == (1200, 800)


def test_a_corner_on_a_screen_is_enough(workspace, fresh_geometry, monkeypatch):
    """A window hanging mostly off the edge is where someone put it, and its
    title bar is still grabbable. Only 'no pixels at all' is the broken case."""
    _screens(monkeypatch, (0, 0, 1920, 1080))
    assert desktop.on_screen(1800, 1000, 1200, 800)
    assert not desktop.on_screen(1900, 1040, 1200, 800)


def test_a_second_screen_left_of_the_first_is_still_a_screen(workspace, fresh_geometry, monkeypatch):
    """Negative coordinates are ordinary on Windows: a monitor placed to the
    left of the primary one has them, and a naive `x >= 0` check would refuse to
    restore onto it."""
    _screens(monkeypatch, (0, 0, 1920, 1080), (-1920, 0, 1920, 1080))
    assert desktop.on_screen(-1500, 100, 1200, 800)


def test_unknown_screens_do_not_veto_a_restore(workspace, fresh_geometry, monkeypatch):
    """A backend that cannot enumerate displays must not turn 'remember the
    position' off; it would fail closed on exactly the machines we cannot
    check."""
    _screens(monkeypatch)
    assert desktop.on_screen(-4000, -4000, 1200, 800)


def test_maximizing_does_not_eat_the_size_to_restore_to(workspace, fresh_geometry, monkeypatch):
    """Maximizing moves and resizes the window, and pywebview reports both as
    ordinary events. Recording them would restore a screen-sized window at 0,0
    on the next launch and lose the size the user actually chose."""
    _screens(monkeypatch, (0, 0, 2560, 1440))
    desktop.remember(x=300, y=120, width=1200, height=800)
    # What the maximize looks like coming over the wire, worst case: the
    # rectangle arrives first and the flag second.
    desktop.remember(x=0, y=0, width=2560, height=1440)
    desktop._note_maximized(True)
    desktop.save_geometry(force=True)

    args = desktop.window_args()
    assert args.get("maximized") is True
    assert (args["width"], args["height"]) == (1200, 800), "the chosen size was overwritten"
    assert (args["x"], args["y"]) == (300, 120)


def test_moving_a_maximized_window_is_not_a_choice_to_remember(workspace, fresh_geometry):
    """Once maximized, the rectangle belongs to the window manager. The one to
    restore to is the one from before."""
    desktop.remember(x=300, y=120, width=1200, height=800)
    desktop._note_maximized(True)
    desktop.remember(x=0, y=0, width=2560, height=1440)

    assert (desktop._geometry["width"], desktop._geometry["height"]) == (1200, 800)


def test_unmaximizing_clears_the_flag(workspace, fresh_geometry, monkeypatch):
    _screens(monkeypatch, (0, 0, 2560, 1440))
    desktop.remember(x=300, y=120, width=1200, height=800)
    desktop._note_maximized(True)
    desktop._note_maximized(False)
    desktop.save_geometry(force=True)

    assert "maximized" not in desktop.window_args()


def test_a_sliver_is_not_restored(workspace, fresh_geometry, monkeypatch):
    """A window dragged down to nothing reopening at nothing looks like an app
    that failed to start."""
    _screens(monkeypatch, (0, 0, 2560, 1440))
    desktop.remember(x=10, y=10, width=12, height=8)
    desktop.save_geometry(force=True)

    args = desktop.window_args()
    assert (args["width"], args["height"]) == desktop.MIN_SIZE


def test_a_corrupt_geometry_file_is_not_a_startup_failure(workspace, fresh_geometry):
    """This file survives upgrades and can be edited by hand, and it is read at
    the one moment where there is no UI to report a problem with."""
    path = desktop.geometry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all", encoding="utf-8")

    assert desktop.read_geometry() == {}
    assert (desktop.window_args()["width"], desktop.window_args()["height"]) == desktop.DEFAULT_SIZE


def test_junk_fields_are_rederived_rather_than_trusted(workspace, fresh_geometry):
    """`create_window` is handed these directly, so a string where an int
    belongs is a crash before there is a window to see it in."""
    from core import jsonl

    path = desktop.geometry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_json(path, {"x": "left", "y": None, "width": 1200, "height": 800, "maximized": "yes"})

    saved = desktop.read_geometry()
    assert saved == {"width": 1200, "height": 800}
    assert "maximized" not in desktop.window_args()


def test_the_save_is_throttled_but_the_last_word_is_not(workspace, fresh_geometry, monkeypatch):
    """`moved` fires per pixel of a drag. The throttle drops the middle of a
    burst; `force` is what makes sure it does not drop the end of one."""
    writes = []
    from core import jsonl

    monkeypatch.setattr(jsonl, "write_json", lambda path, obj: writes.append(dict(obj)))

    desktop.remember(x=1, y=1)
    desktop.remember(x=2, y=2)
    desktop.remember(x=3, y=3)
    assert len(writes) == 1, "the throttle did not hold"

    desktop.save_geometry(force=True)
    assert writes[-1]["x"] == 3


def test_the_geometry_is_actually_handed_to_the_window(workspace, fresh_geometry, monkeypatch):
    """The connecting wire for the geometry, tested for the reason the veto's is:
    every test above passes with `window_args` never reaching `create_window`.

    It asserts the *merge order* as well as the values, because that is the part
    that is easy to get backwards -- NiceGUI splices `native.window_args` in
    after its own `width`/`height`, so these win over `ui.run(window_size=...)`.
    A version that merged the other way would restore the position and silently
    ignore the size.
    """
    from nicegui import app as nicegui_app

    from ui import app as grad_app

    _screens(monkeypatch, (0, 0, 2560, 1440))
    desktop.remember(x=300, y=120, width=1200, height=800)
    desktop.save_geometry(force=True)

    nicegui_app.native.window_args.clear()
    grad_app._install_desktop(True)

    merged = {"width": 800, "height": 600, **nicegui_app.native.window_args}
    assert (merged["x"], merged["y"]) == (300, 120)
    assert (merged["width"], merged["height"]) == (1200, 800)
    nicegui_app.native.window_args.clear()


def test_text_in_the_desktop_app_can_be_selected(workspace, fresh_geometry):
    """pywebview defaults `text_select` to False, and that default is not a
    preference -- it injects `body { user-select: none }` into the page.

    Everything in this workspace is text somebody needs to copy: a run id to
    paste into a command, a traceback to search for, a number out of the ledger.
    None of it could be selected in the desktop app and all of it could in the
    browser, which is exactly why it survived -- the rule is injected at runtime
    and is in no stylesheet to grep.
    """
    assert desktop.window_args()["text_select"] is True


def test_the_stylesheet_only_suppresses_selection_where_a_drag_needs_it(workspace):
    """The other half of the same claim, and the evidence the design always
    assumed selection was on: `user-select: none` appears on the title bar, the
    split handle and the in-flight drag, and nowhere else. A rule that turned it
    off for a *pane* or the app root would put the setting above back."""
    import re

    from ui import tokens

    suppressed = [
        match.group(1).strip()
        for match in re.finditer(r"([^{}]+)\{[^}]*user-select:\s*none", tokens.stylesheet())
    ]
    for selector in suppressed:
        assert any(
            token in selector
            for token in (".grad-handle", ".grad-titlebar", "body.grad-dragging")
        ), f"{selector} makes text unselectable"


def test_it_reaches_the_window_and_not_the_browser(workspace, fresh_geometry, monkeypatch):
    """`window_args` only applies in native mode, which is the mode with the
    problem: in a browser the text was always selectable."""
    from nicegui import app as nicegui_app

    from ui import app as grad_app

    _screens(monkeypatch, (0, 0, 2560, 1440))
    nicegui_app.native.window_args.clear()
    grad_app._install_desktop(True)
    assert nicegui_app.native.window_args.get("text_select") is True
    nicegui_app.native.window_args.clear()

    grad_app._install_desktop(False)
    assert nicegui_app.native.window_args == {}


def test_pywebview_still_takes_the_argument(workspace):
    """Pinned against the installed pywebview rather than assumed. A release that
    renamed or dropped this would leave `create_window` raising a TypeError at
    launch -- or worse, silently ignoring it and putting the rule back."""
    import inspect

    webview = pytest.importorskip("webview", reason="pywebview is not installed")
    assert "text_select" in inspect.signature(webview.create_window).parameters


def test_browser_mode_is_handed_no_window_at_all(workspace, fresh_geometry):
    """`native=False` is the documented fallback and there is no window to place.
    Setting these there would turn native mode back on -- the same trap the
    `window_size` comment in `ui/app.py:run` already records."""
    from nicegui import app as nicegui_app

    from ui import app as grad_app

    nicegui_app.native.window_args.clear()
    grad_app._install_desktop(False)
    assert nicegui_app.native.window_args == {}


def test_the_window_events_are_subscribed_before_the_bridge_starts(workspace, fresh_geometry):
    """`ui.run` is what starts the event manager that delivers these, so a
    registration after it would arrive too late and nothing would ever be saved.
    Asserted against NiceGUI's own registry rather than ours."""
    from nicegui import app as nicegui_app
    from nicegui.native.event_manager import event_manager

    from ui import app as grad_app

    for name in ("moved", "resized", "maximized", "restored", "closed"):
        event_manager._handlers.pop(name, None)
    nicegui_app.native.window_args.clear()
    grad_app._install_desktop(True)

    for name in ("moved", "resized", "maximized", "restored", "closed"):
        assert event_manager._handlers.get(name), f"nothing is listening for {name}"
    nicegui_app.native.window_args.clear()


# ---------------------------------------------------------------------------
# the loading mark
# ---------------------------------------------------------------------------
class _FakePipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    """A splash process that stops when its pipe is closed, as the real one does."""

    def __init__(self, argv, **kwargs) -> None:
        self.argv = list(argv)
        self.kwargs = kwargs
        self.stdin = _FakePipe()
        self.terminated = False
        self.pid = 4242

    def wait(self, timeout=None):
        if not self.stdin.closed:
            import subprocess as sp

            raise sp.TimeoutExpired(self.argv, timeout or 0)
        return 0

    def poll(self):
        return 0 if self.stdin.closed else None

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def fake_splash(monkeypatch):
    """Never spawn a real window in the suite -- §24, and a Tk loop in CI."""
    import subprocess

    from ui import splash

    made: list[_FakeProc] = []

    def popen(argv, **kwargs):
        proc = _FakeProc(argv, **kwargs)
        made.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(splash, "_process", None)
    yield made
    monkeypatch.setattr(splash, "_process", None)


def test_the_loading_mark_is_a_separate_process(workspace, fake_splash):
    """A thread would share the interpreter that is busy importing NiceGUI, and
    a Tk window that does not paint is worse than no window -- Windows greys it
    out and offers to close it."""
    from ui import splash

    splash.start()
    assert len(fake_splash) == 1
    argv = fake_splash[0].argv
    assert argv[1:3] == ["-m", "ui.splash"]
    assert argv[0].endswith(("python.exe", "pythonw.exe", "python", "python3"))


def test_the_pipe_is_what_the_child_watches(workspace, fake_splash):
    """The liveness channel. It has to be a pipe the parent holds open, because
    EOF on it is the one signal that also arrives when the parent is killed."""
    import subprocess

    from ui import splash

    splash.start()
    assert fake_splash[0].kwargs.get("stdin") is subprocess.PIPE
    # And the child is told to watch it. Without the flag it ignores stdin
    # entirely, which is what makes running the module by hand -- where stdin is
    # a console or an already-closed handle -- not exit instantly.
    assert "--watch-stdin" in fake_splash[0].argv


def test_stopping_closes_the_pipe_rather_than_killing(workspace, fake_splash):
    """Closing is the same signal a crash sends, so there is one path to test
    instead of two. `terminate` is only for a child that stopped reading."""
    from ui import splash

    splash.start()
    proc = fake_splash[0]
    splash.stop()

    assert proc.stdin.closed
    assert not proc.terminated
    assert not splash.running()


def test_a_child_that_will_not_go_is_terminated(workspace, fake_splash, monkeypatch):
    from ui import splash

    splash.start()
    proc = fake_splash[0]
    monkeypatch.setattr(proc.stdin, "close", lambda: None)
    splash.stop()

    assert proc.terminated


def test_starting_twice_shows_one_mark(workspace, fake_splash):
    from ui import splash

    splash.start()
    splash.start()
    assert len(fake_splash) == 1


def test_stopping_what_was_never_started_is_fine(workspace, fake_splash):
    """`stop` runs from a connect handler, which fires in browser mode too --
    where nothing ever put a mark up."""
    from ui import splash

    splash.stop()
    assert not splash.running()


def test_a_launch_that_cannot_spawn_still_launches(workspace, monkeypatch):
    """Every reason this fails -- no display, no Tk, no permission to spawn --
    means the app starts exactly as it did before this existed."""
    import subprocess

    from ui import splash

    monkeypatch.setattr(splash, "_process", None)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
    )
    splash.start()
    assert not splash.running()


def test_the_mark_is_the_same_drawing_as_the_icon(workspace):
    """`write_icon`'s rule: there is one drawing of this glyph and everything
    reads it. A splash with its own hand-drawn nabla is the drift it prevents."""
    png = desktop.splash_png(96)
    if png is None:
        pytest.skip("Pillow is not installed")
    assert Path(png).is_file()
    assert Path(png).stat().st_size > 0
    # Cached, not redrawn: the launch path is the one place a couple of hundred
    # milliseconds of Pillow is worth avoiding.
    before = Path(png).stat().st_mtime_ns
    assert desktop.splash_png(96) == png
    assert Path(png).stat().st_mtime_ns == before


def test_the_mark_comes_down_when_a_client_connects(workspace):
    """On connect, not on startup: the server listens several seconds before the
    webview has rendered anything, and taking it down there puts the gap back."""
    from nicegui import app as nicegui_app

    from ui import app as grad_app

    before = len(nicegui_app._connect_handlers)
    grad_app._install_desktop(True)
    assert len(nicegui_app._connect_handlers) > before
    nicegui_app.native.window_args.clear()
