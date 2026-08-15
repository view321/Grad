"""The parts of being a desktop app that are not the workspace itself.

Port selection, the notification-area icon, what the window's close button
means, and the one question that has to be asked before any of it shuts down:
*is something still running?*

Three decisions are load-bearing here.

**The port is chosen, not random.** 8080 if it is free, then 8081, 8082, and so
on. A random port would collide with nothing, which sounds strictly better until
you remember that JupyterLab bakes `frame-ancestors` into its CSP at *launch*
from the app's origin (`config/jupyter/jupyter_server_config.py`). A new random
port every launch is a new origin every launch, so a Lab server left running in
the background would be scoped to the port before last and refuse to embed --
every single time. Walking up from a fixed base means the port is usually the
same one, so the surviving Lab usually still matches. `ui/windows/notebook.py`
handles the case where it does not.

**Closing the window hides it.** The kernels, the Lab server and any running
tool survive, because that is what "background app" means and because closing a
window is not a decision to abandon a running experiment. Quitting is explicit,
from the tray menu, and it is the only path that takes the process down.

**Quit asks when work is in flight.** Two independent sources of "busy", because
there are two kinds of kernel here and neither can see the other: JupyterLab's
own kernels, which only Lab knows about and which are read over its REST API,
and this app's subprocesses -- `nb verify` on a fresh kernel, a preflight, a
report build -- which only `ui/tasks.py` knows about. Losing either to a stray
click on Quit costs real GPU minutes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import urllib.error
import urllib.request
from typing import Any, Callable

log = logging.getLogger("grad.ui")

#: Where port selection starts. Matches the historical default, so an existing
#: Lab server's recorded origin still matches on the common path.
DEFAULT_PORT = 8080
#: How far to walk before giving up. Twenty consecutive busy ports is a machine
#: with a problem, not a machine that needs a twenty-first probe.
PORT_SPAN = 20
#: Lab's REST API is local and already running; anything slower than this is a
#: server that is wedged, and a quit prompt must not hang on it.
_LAB_TIMEOUT_S = 1.5

#: Set by `run` so callbacks arriving on the tray thread can reach the UI loop.
_loop: asyncio.AbstractEventLoop | None = None
#: Set by the shell. Shows the "something is running" dialog on a live client.
_confirm_quit: Callable[[dict[str, Any]], Any] | None = None
_tray: Any = None
_quitting = threading.Event()


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------
def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(preferred: int | None = None) -> int:
    """The first free port at or above `preferred`. See the module docstring.

    An explicitly requested port is honoured even when it looks busy: `--port`
    is someone overriding this function on purpose, and second-guessing them
    would move the app somewhere they did not ask for and did not expect.
    """
    if preferred is not None:
        return preferred
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SPAN):
        if port_is_free(candidate):
            return candidate
    # Nothing free in the span. Hand back the base and let `ui.run` produce the
    # bind error, which names the port and is a better message than ours.
    return DEFAULT_PORT


def origin(port: int) -> str:
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# what is running
# ---------------------------------------------------------------------------
def _lab_busy() -> list[str]:
    """Kernel ids Lab reports as executing. `[]` when Lab is not up.

    Read over Lab's REST API rather than by inspecting kernels directly: Lab
    owns these, their connection files are its business, and `execution_state`
    is exactly the field being asked about.
    """
    from tools import lab as lab_tool  # noqa: PLC0415

    try:
        state = lab_tool.lab_state()
    except Exception:  # noqa: BLE001 - a quit prompt must not fail on this
        return []
    if not state.get("running") or not state.get("port"):
        return []
    url = f"http://127.0.0.1:{int(state['port'])}/api/kernels"
    request = urllib.request.Request(url)  # noqa: S310 - fixed local scheme
    token = state.get("token")
    if token:
        request.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(request, timeout=_LAB_TIMEOUT_S) as response:  # noqa: S310
            kernels = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []
    if not isinstance(kernels, list):
        return []
    return [
        str(k.get("name") or k.get("id"))
        for k in kernels
        if isinstance(k, dict) and k.get("execution_state") == "busy"
    ]


def busy_report() -> dict[str, Any]:
    """Everything that would be interrupted by quitting right now."""
    from ui import tasks as tasks_mod  # noqa: PLC0415

    try:
        local = [t.label for t in tasks_mod.running()]
    except Exception:  # noqa: BLE001
        local = []
    kernels = _lab_busy()
    return {"kernels": kernels, "tasks": local, "busy": bool(kernels or local)}


def busy_sentence(report: dict[str, Any]) -> str:
    """One line naming what is running, for the confirmation dialog."""
    parts: list[str] = []
    kernels = report.get("kernels") or []
    tasks = report.get("tasks") or []
    if kernels:
        noun = "kernel is" if len(kernels) == 1 else "kernels are"
        parts.append(f"{len(kernels)} Lab {noun} executing a cell")
    if tasks:
        noun = "command" if len(tasks) == 1 else "commands"
        parts.append(f"{len(tasks)} {noun} still running ({', '.join(tasks[:3])})")
    return " and ".join(parts) if parts else "Nothing is running."


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
def _window() -> Any:
    """The pywebview window, or None in browser mode."""
    try:
        from nicegui import app as nicegui_app  # noqa: PLC0415

        return getattr(nicegui_app.native, "main_window", None)
    except Exception:  # noqa: BLE001
        return None


def show_window() -> bool:
    window = _window()
    if window is None:
        return False
    try:
        window.show()
        window.restore()
    except Exception:  # noqa: BLE001 - a window that will not raise is not fatal
        log.debug("could not raise the window", exc_info=True)
        return False
    return True


def hide_window() -> None:
    window = _window()
    if window is None:
        return
    try:
        window.hide()
    except Exception:  # noqa: BLE001
        log.debug("could not hide the window", exc_info=True)


#: The separate Lab window, once opened. One at a time; reopening focuses it.
_lab_window: Any = None


def native_available() -> bool:
    """Whether there is a desktop window to open a second one beside."""
    return _window() is not None


def open_lab_window(url: str) -> bool:
    """Show JupyterLab in a window of its own. Returns whether one is up.

    **This is the fix for the workspace feeling slow with Lab open.** As an
    iframe, Lab shares a renderer process with the shell: one main thread laying
    out a whole JupyterLab document *and* the pane tree, so Lab's work and
    Grad's block each other, and the overlay in `ui/static/tiling.js` has to
    measure a layout containing all of it on every reflow. A second webview is
    a second renderer process -- Lab can be as busy as it likes and the
    workspace keeps its frame rate.

    What is given up is the seam: Lab is no longer visually inside the pane. The
    notebook window keeps the chrome that matters -- the verify banner, which is
    the only source of citable state -- so nothing that gates a claim moves.
    """
    global _lab_window

    if _lab_window is not None:
        try:
            # Navigated, not merely raised. Selecting a different notebook and
            # clicking again used to show the window still displaying the old
            # one -- and if that notebook's kernel had since been culled, what
            # you got was a Lab sitting on a dead connection, which reads as
            # "the Lab window loses its kernel" rather than as "this window was
            # never told to move".
            if getattr(_lab_window, "get_current_url", None) and url != _lab_window.get_current_url():
                _lab_window.load_url(url)
            _lab_window.show()
            _lab_window.restore()
            return True
        except Exception:  # noqa: BLE001 - destroyed windows raise; fall through
            _lab_window = None
    try:
        import webview  # noqa: PLC0415
    except ImportError:
        return False
    try:
        window = webview.create_window(
            "Grad — JupyterLab", url, width=1280, height=900, resizable=True
        )
    except Exception:  # noqa: BLE001 - browser mode, or a backend that refuses
        log.exception("could not open the Lab window")
        return False

    def _forget() -> None:
        global _lab_window

        _lab_window = None

    try:
        window.events.closed += _forget
    except Exception:  # noqa: BLE001 - only costs us a stale handle
        log.debug("could not track the Lab window's close", exc_info=True)
    _lab_window = window
    return True


def lab_window_open() -> bool:
    return _lab_window is not None


def bind_confirm(callback: Callable[[dict[str, Any]], Any]) -> None:
    """Registered by the shell: shows the quit confirmation on a live client."""
    global _confirm_quit

    _confirm_quit = callback


# ---------------------------------------------------------------------------
# quitting
# ---------------------------------------------------------------------------
def shutdown() -> None:
    """Take the app down. The only path that ends the process.

    Lab is deliberately *not* stopped here. It is a detached server with its own
    lifetime -- see `core/spawn.py` -- and a user who quit the workspace has not
    necessarily finished with the notebook they left running in it. `tools/lab.py
    stop` is how it ends.
    """
    if _quitting.is_set():
        return
    _quitting.set()
    from core import instance  # noqa: PLC0415

    instance.release()
    if _tray is not None:
        try:
            _tray.stop()
        except Exception:  # noqa: BLE001
            log.debug("tray would not stop", exc_info=True)
    try:
        from nicegui import app as nicegui_app  # noqa: PLC0415

        nicegui_app.shutdown()
    except Exception:  # noqa: BLE001
        log.debug("nicegui shutdown failed", exc_info=True)


def request_quit() -> None:
    """Quit, asking first when something is running.

    Callable from the tray thread, which is why the dialog is dispatched onto
    the UI loop rather than opened here: NiceGUI elements belong to the loop
    that built them, and building one from pystray's thread would either be
    ignored or corrupt the client's element tree.
    """
    report = busy_report()
    if not report["busy"] or _confirm_quit is None or _loop is None:
        shutdown()
        return
    show_window()
    try:
        asyncio.run_coroutine_threadsafe(_ask_then_quit(report), _loop)
    except Exception:  # noqa: BLE001 - if the ask cannot be staged, do not quit
        log.exception("could not raise the quit confirmation")


async def _ask_then_quit(report: dict[str, Any]) -> None:
    if _confirm_quit is None:
        return
    try:
        confirmed = _confirm_quit(report)
        if asyncio.iscoroutine(confirmed):
            confirmed = await confirmed
    except Exception:  # noqa: BLE001
        log.exception("quit confirmation failed")
        return
    if confirmed:
        shutdown()


# ---------------------------------------------------------------------------
# the notification-area icon
# ---------------------------------------------------------------------------
def _icon_image(size: int = 64) -> Any:
    """The tray glyph: the nabla, ink on the brand yellow.

    Drawn rather than loaded so there is no image file to lose track of in a
    packaged install, and because a tray icon is sixteen logical pixels of flat
    colour -- exactly what this design language already is.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    ink, paper = (20, 16, 12), (255, 212, 0)
    image = Image.new("RGBA", (size, size), (*paper, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size - 1, size - 1], outline=ink, width=max(2, size // 16))
    inset = size * 0.26
    draw.polygon(
        [(inset, inset), (size - inset, inset), (size / 2, size - inset)],
        fill=ink,
    )
    return image


def write_icon(path: str) -> str:
    """Save the mark as a multi-resolution `.ico`, for the installer's shortcut.

    Here rather than in the installer so the shortcut and the notification area
    cannot drift apart -- there is one drawing of this glyph and both read it.
    Windows picks a size per context (16px in the taskbar, 32px on the desktop,
    256px in the large-icon view), and an `.ico` carrying only one of them gets
    scaled into the others.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    target = _Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sizes = [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)]
    _icon_image(256).save(target, format="ICO", sizes=sizes)
    return str(target)


def start_tray(*, on_restart_lab: Callable[[], Any] | None = None) -> Any:
    """Put Grad in the notification area. Returns the icon, or None.

    Optional by construction: `pystray` is in the `ui` extra, but a machine
    without a working tray (a bare Windows Server session, most Linux desktops
    without an AppIndicator host) must still get a usable app. The only thing
    lost is the way back from a hidden window, so `hide_to_tray` refuses to hide
    when this returned None.
    """
    global _tray

    try:
        import pystray  # noqa: PLC0415
    except ImportError:
        log.info("pystray is not installed; the app will not show in the notification area")
        return None

    def _menu() -> Any:
        items = [
            pystray.MenuItem("Open Grad", lambda: show_window(), default=True),
        ]
        if on_restart_lab is not None:
            items.append(pystray.MenuItem("Restart Lab for this window", lambda: on_restart_lab()))
        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Quit Grad", lambda: request_quit()))
        return pystray.Menu(*items)

    try:
        icon = pystray.Icon("grad", _icon_image(), "Grad", _menu())
        # `run_detached` would be the tidier call, but it is not implemented on
        # every backend; a daemon thread around `run` works on all of them and
        # dies with the process either way.
        threading.Thread(target=icon.run, name="grad-tray", daemon=True).start()
    except Exception:  # noqa: BLE001 - see the docstring
        log.exception("could not start the tray icon")
        return None
    _tray = icon
    return icon


def has_tray() -> bool:
    return _tray is not None


def hide_to_tray() -> bool:
    """Hide the window, if there is a way back to it.

    Returns whether it hid. Without a tray icon this refuses: a hidden window
    with no icon and no taskbar entry is an app that is running, consuming a
    port, holding the single-instance lock, and unreachable by any means short
    of Task Manager.
    """
    if not has_tray():
        return False
    hide_window()
    return True


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop

    _loop = loop
