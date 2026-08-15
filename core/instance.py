"""One Grad at a time, and a way to reach the one that is already running.

Two mechanisms, because they answer different questions and only one of them
can be trusted.

**The lock decides.** On Windows that is a named mutex, on POSIX an `flock` over
a file in the app directory. Both are held by the kernel for as long as the
process lives and both are released when it dies -- including when it is killed,
which is the case a pid file gets wrong. A pid file left behind by a crash says
"already running" forever, and the fix is always the same undignified thing:
telling a user to go and delete a file before their app will open.

**The state file only describes.** `instance.json` carries the port the running
instance is serving on, so a second launch can hand over to it instead of dying
silently. It is written *after* the lock is taken and it is never consulted to
decide whether to start -- a stale one is an inconvenience, not a lockout.

Why the port has to be discoverable at all: the app picks the first free port at
or above 8080, so the second launch cannot assume 8080 and cannot guess. Without
this file, double-clicking the shortcut while Grad sits in the notification area
would do nothing at all, which reads exactly like a broken shortcut.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from core import appdata

log = logging.getLogger("grad.instance")

#: The mutex name is global to the user's session, not the machine: two people
#: on one Windows box via fast user switching are two installations, and
#: `Local\\` scopes the name to the session. `Global\\` would have one of them
#: refuse to start because the *other* was running.
_MUTEX_NAME = r"Local\GradientAgent.Grad.SingleInstance"

#: How long to wait for the running instance to answer. It is a local process
#: raising a window; if it has not answered in this long it is wedged, and the
#: honest thing is to say so rather than hang the launcher.
_SHOW_TIMEOUT_S = 3.0


class AlreadyRunning(Exception):
    """Raised by `acquire` when another instance holds the lock."""

    def __init__(self, info: dict[str, Any] | None) -> None:
        self.info = info or {}
        port = self.info.get("port")
        where = f" on port {port}" if port else ""
        super().__init__(f"Grad is already running{where}.")


class _Lock:
    """The held lock. One per process; `release` is idempotent."""

    def __init__(self) -> None:
        self._handle: Any = None
        self._fh: Any = None

    def acquire(self) -> bool:
        if os.name == "nt":
            return self._acquire_windows()
        return self._acquire_posix()

    def _acquire_windows(self) -> bool:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        ERROR_ALREADY_EXISTS = 183
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        handle = kernel32.CreateMutexW(None, True, _MUTEX_NAME)
        if not handle:
            # Cannot create the mutex at all. Refusing to start over a failure
            # of the guard itself would be worse than the duplicate it guards
            # against, so this reports "acquired" and lets the app open.
            log.debug("CreateMutexW failed: %s", ctypes.get_last_error())
            return True
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def _acquire_posix(self) -> bool:
        import fcntl  # noqa: PLC0415

        appdata.ensure()
        path = appdata.state_dir() / "instance.lock"
        fh = open(path, "a+b")  # noqa: SIM115 - held for the process lifetime
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._handle is not None:
            import ctypes  # noqa: PLC0415

            ctypes.WinDLL("kernel32").CloseHandle(self._handle)
            self._handle = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None


_held = _Lock()


def read_state() -> dict[str, Any]:
    """What the running instance published, or `{}`. Never raises."""
    try:
        return json.loads(appdata.lock_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def publish(port: int) -> None:
    """Record where this instance is serving, for the next launch to find."""
    appdata.ensure()
    payload = {"pid": os.getpid(), "port": int(port)}
    try:
        appdata.lock_path().write_text(json.dumps(payload), encoding="utf-8")
    except OSError:  # the app works without it; only the handover degrades
        log.debug("could not publish instance state")


def clear() -> None:
    try:
        appdata.lock_path().unlink(missing_ok=True)
    except OSError:
        pass


def acquire() -> None:
    """Take the single-instance lock, or raise `AlreadyRunning`."""
    if not _held.acquire():
        raise AlreadyRunning(read_state())


def release() -> None:
    _held.release()
    clear()


def is_running() -> bool:
    """Whether another Grad holds the lock. Asks, and puts it back.

    The lock rather than `read_state()`, for the reason at the top of this file:
    a pid file left by a crash says "running" forever. Taking it and dropping it
    is safe -- if it can be taken, nothing else holds it.

    **`_held.release()` and not `release()`**, which is the whole reason this
    lives here rather than in the caller. `release` also calls `clear()`, which
    deletes `instance.json`, and a *probe* has no business doing that: on the
    one path where `_acquire_windows` cannot create the mutex it reports success
    by design -- refusing to start over a failure of the guard would be worse
    than the duplicate it guards against -- so a probe that trusted it would
    then delete a genuinely running instance's published port and break the
    handover that file exists for.
    """
    if not _held.acquire():
        return True
    _held.release()
    return False


def show_running(info: dict[str, Any] | None = None) -> bool:
    """Ask the instance that is already up to raise its window.

    Returns whether it answered. A `False` here is the difference between "your
    app is on screen now" and "something is holding the lock but not serving",
    and the launcher says different things for the two.
    """
    state = info if info is not None else read_state()
    port = state.get("port")
    if not port:
        return False
    url = f"http://127.0.0.1:{int(port)}/__grad/show"
    try:
        with urllib.request.urlopen(url, timeout=_SHOW_TIMEOUT_S) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False
