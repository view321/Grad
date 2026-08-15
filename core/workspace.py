"""Which directory the workspace *is*, and remembering the answer.

Every path in the system derives from `paths.root()`, which used to be GRAD_ROOT
or the directory the code sits in. The app can now be pointed at another
workspace from its own menu, and that needs two things this module owns: a
validation step, because the value arrives from a text field, and somewhere to
remember the choice.

**Precedence, highest first:**

1. `GRAD_ROOT` in the environment. An explicit override stays explicit -- the
   test suite sets it, and a remembered choice must never quietly beat someone
   who typed it on the command line.
2. The pointer file, written by the app's folder chooser.
3. The directory the code is installed in, which is what a fresh checkout gets.

**The pointer lives beside the code, not in `data/`.** That is the whole reason
it works: a pointer stored inside the workspace it points away from is
unreadable the moment you leave, so the app could never find its way back.

Switching is applied by setting `GRAD_ROOT` in this process's environment, which
is also how it reaches the CLIs -- they run as subprocesses and inherit it, so
the agent's Bash tools and the UI cannot end up reading different ledgers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.errors import UsageError

log = logging.getLogger("grad.workspace")

#: How many previous roots the chooser offers. Long enough to switch back and
#: forth between a couple of projects, short enough to read at a glance.
MAX_RECENT = 8

_cache: dict[str, Any] | None = None


def code_dir() -> Path:
    """Where Grad itself is installed. Computed here rather than imported from
    `paths`, which would be a cycle: `paths.root()` consults this module."""
    return Path(__file__).resolve().parent.parent


def pointer_path() -> Path:
    return code_dir() / ".grad-workspace.json"


def read_pointer(*, reload: bool = False) -> dict[str, Any]:
    """The pointer file, cached.

    `paths.root()` is called for every path in the system, so an uncached read
    here would be a JSON parse per path lookup. The cache is invalidated by
    `select`, which is the only thing that writes.
    """
    global _cache
    if _cache is not None and not reload:
        return _cache
    data: dict[str, Any] = {}
    path = pointer_path()
    try:
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except (OSError, json.JSONDecodeError):
        # A hand-edited or unreadable pointer must not stop the app starting;
        # falling back to the installed directory is always a valid answer.
        log.debug("could not read %s", path)
    _cache = data
    return data


def remembered() -> Path | None:
    """The remembered root, if it is still a usable directory.

    Checked rather than trusted: the folder may have been deleted, renamed or
    live on a drive that is not mounted today. Returning a path that no longer
    exists would send every ledger read to a directory that cannot be created.
    """
    value = read_pointer().get("root")
    if not isinstance(value, str) or not value:
        return None
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, ValueError):
        return None
    return path if path.is_dir() else None


def recent() -> list[Path]:
    """Previously chosen roots, most recent first, filtered to those that exist."""
    values = read_pointer().get("recent")
    out: list[Path] = []
    if not isinstance(values, list):
        return out
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, ValueError):
            continue
        if path.is_dir() and path not in out:
            out.append(path)
    return out[:MAX_RECENT]


def source() -> str:
    """Which rule decided the current root. Shown in the chooser, because
    "why is it still pointing there?" is otherwise unanswerable from the UI."""
    if os.environ.get("GRAD_ROOT"):
        return "environment"
    return "remembered" if remembered() is not None else "default"


def validate(candidate: str | Path, *, create: bool = False) -> Path:
    """Resolve a candidate root, or refuse it with a fix.

    The value comes from a text field, so every failure it can have is a message
    someone has to act on: blank, a file rather than a directory, a path that
    does not exist, a directory that cannot be written to.
    """
    text = str(candidate or "").strip().strip('"')
    if not text:
        raise UsageError("choose a folder for the workspace", fix="pick one, or type a path")
    try:
        path = Path(text).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise UsageError(f"{text!r} is not a usable path: {exc}") from exc

    if path.exists() and not path.is_dir():
        raise UsageError(
            f"{path} is a file, not a folder",
            fix="choose the directory that contains it",
        )
    if not path.exists():
        if not create:
            raise UsageError(f"{path} does not exist", fix="create it, or choose another folder")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UsageError(f"could not create {path}: {exc}") from exc

    # Checked by asking the filesystem, not by inspecting the mode bits: a
    # network share, a read-only mount and a permission denial all present
    # differently, and all of them mean the same thing here.
    if not os.access(path, os.W_OK):
        raise UsageError(
            f"{path} cannot be written to",
            fix="choose a folder you own, or fix its permissions",
        )
    return path


def select(candidate: str | Path, *, create: bool = False) -> Path:
    """Point this process, and everything it starts, at another workspace.

    Setting `GRAD_ROOT` rather than caching a value is what makes the switch
    total: `paths.root()` reads the environment first, and every CLI the UI or
    the agent shells out to inherits it. The alternative -- a module-level
    override consulted only by `paths` -- would leave subprocesses reading the
    old workspace while the UI showed the new one.
    """
    path = validate(candidate, create=create)
    # Imported lazily in both directions: `paths.root()` consults this module.
    from core import paths  # noqa: PLC0415

    leaving = paths.root()
    os.environ["GRAD_ROOT"] = str(path)
    _write_pointer(path, leaving=leaving)
    return path


def _write_pointer(path: Path, *, leaving: Path | None = None) -> None:
    """Write the pointer, and remember the folder being left.

    Recording only where you are *going* looks right and leaves the history
    empty exactly when it matters: after the first switch the only entry is the
    folder you are now in, which the menu filters out as somewhere you already
    are. Switching back -- the whole reason to keep a list -- would be the one
    thing it could not offer. So the folder being left goes in first.
    """
    history = [p for p in recent() if p != path]
    if leaving is not None and leaving != path and leaving.is_dir():
        history = [leaving, *[p for p in history if p != leaving]]
    previous = [str(p) for p in history]
    payload = {"root": str(path), "recent": [str(path), *previous][:MAX_RECENT]}
    global _cache
    _cache = payload
    try:
        pointer_path().write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        # An installation directory that cannot be written to is a real
        # situation -- a system-wide install, a read-only image. The switch
        # still applies to this process; it just will not be remembered.
        log.debug("could not persist the workspace pointer to %s", pointer_path())
