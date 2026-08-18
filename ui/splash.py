"""The mark on screen while the workspace is still loading.

Starting Grad from the shortcut is slow in a way nothing on screen admits to.
`pythonw.exe` opens no console, `ui.run(native=True)` shows no window until
NiceGUI has imported, bound a port, built the page and handed a URL to a
pywebview process that then starts an Edge WebView2 host. On a cold start that
is several seconds of a double-click having produced *nothing at all*, which
reads exactly like a shortcut that does not work -- so people click it again,
and the second launch hands over to the first (`core/instance.py`) and still
shows nothing.

Three decisions are load-bearing.

**It is a separate process.** Not a thread: the reason there is nothing on
screen is that this interpreter is busy importing, and importing holds the GIL
for long stretches. A Tk loop sharing that interpreter would put up a window
that does not paint and does not answer, which is worse than no window --
Windows greys out a hung window and offers to close it. A child process is
scheduled independently and is on screen while the parent is still importing.

**It is Tk.** It is in the standard library, it starts in about a tenth of a
second, and it is already installed anywhere `pythonw.exe` came from. Every
other option here -- a second pywebview window, a NiceGUI page, anything with a
browser engine in it -- is the same slow thing whose slowness this exists to
cover.

**It can be pushed out of the way.** A splash screen that insists on staying in
front is a splash screen that stops you doing anything else while an app you
were not waiting for starts. Clicking it drops it behind everything and it
carries on waiting; dragging moves it. Neither cancels the launch, because the
launch is not this process's to cancel.

It closes when the workspace's first client connects -- the page is not merely
built by then, it is *shown* -- and, failing that, when the pipe from the parent
closes or the timeout runs out. See `stop`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any

log = logging.getLogger("grad.ui")

#: How long the window lives if nobody ever tells it to go away. This is the
#: backstop behind the backstop -- the parent closes the pipe, and a parent that
#: died without closing anything is caught by the pipe reaching EOF regardless.
#: What is left is a machine where neither happened, and a mark that sits on
#: screen forever is a bug report about the splash rather than about the start.
MAX_SECONDS = 180.0
#: How often the Tk loop asks whether it should still be here.
POLL_MS = 120
#: Pixels of movement that turn a click into a drag. Below this, a press and
#: release in the same place means "get out of the way" rather than "move here".
DRAG_SLOP_PX = 4

_process: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# the parent's side
# ---------------------------------------------------------------------------
def _theme() -> str:
    """The workspace's palette, or the default. Never raises.

    Called on the launch path before anything else has loaded, so every failure
    -- no app directory yet, a settings file half-written, an import that is not
    there in a stripped install -- has to answer "light" rather than stop a
    launch this module exists to make feel faster.
    """
    try:
        from core import settings  # noqa: PLC0415

        return settings.theme()
    except Exception:  # noqa: BLE001 - see the docstring
        return "light"


def start(*, timeout_s: float = MAX_SECONDS) -> None:
    """Put the mark on screen. Returns immediately; never raises.

    Call this as early in the launch as it can be called and still be right --
    after the single-instance check, because a second launch that hands over to
    the first has nothing to load and should flash nothing, and before the first
    expensive import, because everything after that is the wait being covered.

    Failure here is silent by construction. There is no display, no Tk, no
    permission to spawn: all of them mean the app starts exactly as it did
    before this module existed, and none of them is worth refusing to start over.
    """
    global _process

    if _process is not None:
        return
    argv = [
        sys.executable,
        "-m",
        "ui.splash",
        "--parent-pid",
        str(os.getpid()),
        "--timeout",
        str(float(timeout_s)),
        # Only ever passed here, because only here is stdin known to be a pipe
        # this process holds open. Run by hand it is a console or an
        # already-closed handle, and watching that reads EOF at once -- a splash
        # that vanishes the instant it is started, which is a confusing thing to
        # meet while debugging one.
        "--watch-stdin",
        # Passed down rather than read in the child, and the reason is the whole
        # design of this module: the child exists to be on screen in about a
        # tenth of a second, and reading the setting there would put
        # `core.settings` and `core.appdata` on that path. The parent is already
        # importing them. A splash that flashed cream in front of a dark
        # workspace would be the one frame the whole theme is judged on.
        "--theme",
        _theme(),
    ]
    try:
        from core import paths, spawn  # noqa: PLC0415

        _process = subprocess.Popen(  # noqa: S603 - our own module, our own argv
            argv,
            # The pipe is the liveness channel, not a channel for data: the
            # child watches it for EOF. See `stop` and `_watch_parent`.
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(paths.install_dir()),
            **spawn.quiet(),
        )
    except Exception:  # noqa: BLE001 - see the docstring
        log.debug("could not start the loading window", exc_info=True)
        _process = None


def stop() -> None:
    """Take the mark down. Idempotent, and safe to call from anywhere.

    **Closing the pipe is the signal**, and it is a better one than terminating
    the process, because it is the same signal a crash sends. The child is
    watching one file descriptor for EOF; it gets that whether this process
    closed the handle deliberately, exited, or was killed -- so there is no path
    that leaves a splash screen on a machine whose app is gone. `terminate` is
    only what happens when a child has stopped reading it.
    """
    global _process

    proc, _process = _process, None
    if proc is None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
        except OSError:
            pass
    except Exception:  # noqa: BLE001 - never worth failing a launch over
        log.debug("could not close the loading window", exc_info=True)


def running() -> bool:
    return _process is not None


# ---------------------------------------------------------------------------
# the child's side
# ---------------------------------------------------------------------------
def _watch_parent(gone: threading.Event) -> None:
    """Set `gone` when the pipe from the parent closes.

    `os.read` on descriptor 0 rather than `sys.stdin`, because under
    `pythonw.exe` there is no console and Python may leave `sys.stdin` as None
    even though the descriptor this process was handed is perfectly real.
    """
    while True:
        try:
            if not os.read(0, 1):
                break
        except (OSError, ValueError):
            break
    gone.set()


def _centre(window: Any, width: int, height: int) -> None:
    """Put the window in the middle of the screen it is opening on."""
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = max(0, (screen_w - width) // 2)
    # Slightly above centre. Optical centre sits higher than geometric centre,
    # and a box placed at exactly half the height reads as low.
    y = max(0, int((screen_h - height) * 0.42))
    window.geometry(f"{width}x{height}+{x}+{y}")


def show(
    *, timeout_s: float = MAX_SECONDS, watch_stdin: bool = False, theme: str = "light"
) -> int:
    """The window itself. Runs the Tk loop until something says to stop."""
    import tkinter as tk  # noqa: PLC0415 - the child's whole reason to exist

    from ui import tokens  # noqa: PLC0415

    # Read from the palette, not spelled here. `tests/test_ui_tokens.py` enforces
    # that rule across the package and it applies to a window drawn in Tk exactly
    # as it does to one drawn in CSS -- a splash screen with its own idea of the
    # brand yellow is the first thing to drift when the palette changes. The
    # module is pure Python with no dependencies of its own, so the cost of
    # importing it in this process is a parse.
    #
    # `palette()` rather than `COLOUR` since there are two: an unknown name
    # resolves to the default there, so a `--theme` from a newer version is a
    # cream splash rather than a KeyError in front of a launch.
    colour = tokens.palette(theme)
    # `fill`, not `ink`. The border and the ground behind the card are the
    # emphasis ground -- in the dark palette `ink` is near-white, and a 2px
    # near-white frame around a dark card is the same inversion bug the CSS
    # `fill` token exists to prevent.
    #
    # Three names where there used to be one, for the same reason the stylesheet
    # grew `fill` and `on-attention`: `ink` was doing three jobs here -- the
    # frame around the card, the text on the card, and the nabla on the yellow
    # mark -- and the three move in different directions when the ground
    # inverts. The frame stays dark, the text goes light, and the nabla does not
    # move at all because the yellow under it did not.
    frame, paper, brand = colour["fill"], colour["paper"], colour["attention"]
    ink, on_brand = colour["ink"], colour["on-attention"]
    muted = colour["muted"]
    width, height = 340, 232

    gone = threading.Event()
    if watch_stdin:
        threading.Thread(target=_watch_parent, args=(gone,), daemon=True).start()

    root = tk.Tk()
    root.title("Grad")
    # Borderless: this is a mark, not a window anyone should be asked to manage.
    # It also keeps it out of the taskbar, where a second Grad entry that
    # disappears on its own would be its own small confusion.
    root.overrideredirect(True)
    root.configure(bg=frame)
    root.attributes("-topmost", True)
    _centre(root, width, height)

    # The ink border is the frame's own background showing through a 2px inset,
    # which is this design language's border everywhere else.
    card = tk.Frame(root, bg=paper)
    card.place(x=2, y=2, width=width - 4, height=height - 4)

    mark = tk.Label(card, bg=brand, bd=0)
    image = None
    try:
        from ui import desktop  # noqa: PLC0415

        png = desktop.splash_png(96)
        if png:
            image = tk.PhotoImage(file=png)
            mark.configure(image=image, width=96, height=96)
    except Exception:  # noqa: BLE001 - a wordmark is a fine degraded splash
        image = None
    if image is None:
        # No Pillow, no PNG, or a Tk without the image reader. Deliberately not
        # a second hand-drawn nabla -- see `desktop.splash_png`.
        mark.configure(text="∇", fg=on_brand, font=("Segoe UI", 44, "bold"), width=3, height=1)
    mark.pack(pady=(26, 14))

    tk.Label(
        card, text="GRAD", bg=paper, fg=ink, font=("Segoe UI", 15, "bold")
    ).pack()
    caption = tk.Label(
        card,
        text="starting the workspace…",
        bg=paper,
        fg=ink,
        font=("Segoe UI", 9),
    )
    caption.pack(pady=(6, 0))
    hint = tk.Label(
        card,
        text="click to send it behind · drag to move",
        bg=paper,
        fg=muted,
        font=("Segoe UI", 8),
    )
    hint.pack(pady=(2, 0))

    state: dict[str, Any] = {"press": None, "moved": False, "backgrounded": False}

    def send_behind() -> None:
        if state["backgrounded"]:
            return
        state["backgrounded"] = True
        try:
            root.attributes("-topmost", False)
            root.lower()
        except tk.TclError:
            return
        # Said, because a window that drops behind everything and still says
        # "starting" is indistinguishable from one that gave up.
        caption.configure(text="still starting — this closes itself")
        hint.configure(text="")

    def on_press(event: Any) -> None:
        state["press"] = (event.x_root, event.y_root, root.winfo_x(), root.winfo_y())
        state["moved"] = False

    def on_motion(event: Any) -> None:
        press = state["press"]
        if press is None:
            return
        dx, dy = event.x_root - press[0], event.y_root - press[1]
        if abs(dx) > DRAG_SLOP_PX or abs(dy) > DRAG_SLOP_PX:
            state["moved"] = True
        if state["moved"]:
            root.geometry(f"+{press[2] + dx}+{press[3] + dy}")

    def on_release(_: Any) -> None:
        if state["press"] is not None and not state["moved"]:
            send_behind()
        state["press"] = None

    for widget in (root, card, mark, caption, hint):
        widget.bind("<ButtonPress-1>", on_press)
        widget.bind("<B1-Motion>", on_motion)
        widget.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda _: send_behind())

    deadline = time.monotonic() + max(1.0, float(timeout_s))

    def tick() -> None:
        if gone.is_set() or time.monotonic() >= deadline:
            root.destroy()
            return
        root.after(POLL_MS, tick)

    root.after(POLL_MS, tick)
    try:
        root.mainloop()
    except Exception:  # noqa: BLE001 - a splash must never be the thing that fails
        log.debug("the loading window stopped badly", exc_info=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="ui.splash",
        description="The Grad mark, on screen while the workspace loads.",
    )
    parser.add_argument("--timeout", type=float, default=MAX_SECONDS)
    # Accepted and unused: the pipe is what liveness is actually read from, and
    # a pid in the argument list is worth having when someone is looking at this
    # process in a task manager wondering what started it.
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument(
        "--watch-stdin",
        action="store_true",
        help="close when the pipe from the launching process does",
    )
    # Not validated against a list here: `tokens.palette` resolves an unknown
    # name to the default, which is the behaviour that matters -- a splash is
    # not the place to refuse to start over a theme name.
    parser.add_argument("--theme", default="light", help="which palette to draw in")
    args = parser.parse_args(argv)
    return show(timeout_s=args.timeout, watch_stdin=args.watch_stdin, theme=args.theme)


if __name__ == "__main__":
    raise SystemExit(main())
