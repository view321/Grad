"""Where the *app's* state lives, which is not where the *research* lives.

Two roots, and the split between them is the whole point of this module.

`core/paths.py` resolves the **workspace**: the ledger, the notebooks, the notes
and figures a report cites. That is the user's research. It is versioned beside
the code that produced it, and the README's claim that "every number in a report
traces to a run record" is only checkable because the record sits next to the
number. None of it moves here, and a helper that would move it does not belong
in this file.

This module resolves the **installation**: state belonging to this copy of Grad
on this machine. The window layout, the Lab server's port and token, kernel
connection files, the HTTP cache, the cookie-signing secret, chat transcripts,
logs. Every one of them is regenerable, machine-specific, or private, and none
of them describes a result. In a repository they are noise at best -- the Lab
token and the storage secret are a leak.

**Transcripts are the awkward case, and they are why `workspace_state_dir`
exists.** They are private, so they want to be here; but `ui/app.py:rebind`
documents that switching the workspace root switches which conversation is on
screen, and a single flat directory would quietly break that -- one transcript
pile shared by every folder you ever opened. So the per-workspace state is
namespaced by the root it belongs to: readable stem, plus a hash, because two
different folders can share a name and `D:/work/grad` must not collide with
`C:/old/grad`.

`GRAD_APP_DIR` overrides everything, which is what lets the test suite point an
installation at a temp directory the way `GRAD_ROOT` already points a workspace
at one.

On POSIX this is one directory rather than the three XDG would ask for
(`XDG_STATE_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`). That is deliberate: the
Windows install is the one that matters here, it has exactly one
`%LOCALAPPDATA%\\Grad`, and keeping the two platforms the same shape means a
path bug reproduces on either.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("grad.appdata")

#: Subdirectories of the app root, created together by `ensure`.
_SUBDIRS = ("state", "logs", "cache", "workspaces")


def app_dir() -> Path:
    """This installation's private directory.

    `GRAD_APP_DIR` first, then the platform's per-user application data. The
    Windows branch reads `LOCALAPPDATA` rather than joining `~` blindly, because
    a roaming profile moves it and the literal path would be wrong exactly on
    the machines where it matters.
    """
    env = os.environ.get("GRAD_APP_DIR")
    if env:
        return Path(env).resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return (root / "Grad").resolve()
    return (Path.home() / ".local" / "state" / "grad").resolve()


def state_dir() -> Path:
    """Small persistent files: layouts, Lab's port and token, the instance lock."""
    return app_dir() / "state"


def logs_dir() -> Path:
    return app_dir() / "logs"


def cache_dir() -> Path:
    """Regenerable downloads. Safe to delete; nothing cites it."""
    return app_dir() / "cache"


def _slug(value: str) -> str:
    """A readable, filesystem-safe stem. Never the whole name -- see `_key`."""
    keep = "".join(c if c.isalnum() or c in "._-" else "-" for c in value)
    while ".." in keep:
        keep = keep.replace("..", ".")
    return keep.strip("._-")[:32] or "workspace"


def _key(root: Path) -> str:
    """A stable directory name for a workspace root.

    Stem *and* digest. The stem alone collides -- every checkout called `grad`
    would share one directory, which is precisely the "one transcript pile" the
    module docstring rules out. The digest alone is unreadable, and someone will
    eventually have to look in here and work out which folder a directory
    belongs to. Case-folded first because Windows paths are case-insensitive and
    `D:\\Grad` and `d:\\grad` are the same workspace.
    """
    text = str(root).casefold()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(root.name)}-{digest}"


def workspace_state_dir(root: Path | None = None) -> Path:
    """App state that belongs to one workspace, such as its chat transcripts.

    Defaults to the current root. Imported lazily for the same reason
    `core/paths.py` imports `core/workspace.py` lazily: almost everything
    imports this module, and the dependency would otherwise run in a circle.
    """
    if root is None:
        from core import paths  # noqa: PLC0415

        root = paths.root()
    path = app_dir() / "workspaces" / _key(Path(root))
    return path


def lock_path() -> Path:
    """The single-instance lock. See `core/instance.py`."""
    return state_dir() / "instance.json"


def ensure() -> None:
    """Create the app directories. Cheap, idempotent, and safe to call early."""
    for name in _SUBDIRS:
        (app_dir() / name).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------
#: `data/<name>` in a workspace -> where it lives now. Only entries that are
#: unambiguously app state; `data/papers`, `data/corpus.sqlite` and `data/mnist`
#: are cited or downloaded research and are deliberately absent.
_LEGACY: tuple[tuple[str, str], ...] = (
    ("layouts", "state"),
    ("lab", "state"),
    ("kernel", "state"),
    ("cache", "cache"),
)


def _relocate(source: Path, target: Path) -> bool:
    """Copy a directory's contents to `target`, then remove the originals.

    Deliberately *not* `shutil.move`. Moving a directory wholesale is one
    operation with two outcomes on Windows, and the bad one loses files: a
    single open handle inside -- the Lab server holding its own log, which is
    the normal state of affairs when this runs -- makes the rename fail, and the
    copy-then-delete fallback it degrades into can delete some sources after
    copying them and then abort on the locked one. That leaves files neither
    here nor there, which is the one result a migration must never produce.

    So: copy everything first, into a staging directory beside the target;
    promote it only once every file has arrived; and only then remove the
    sources, each independently and never before its copy exists. A locked file
    is left where it is, which is the safe direction -- a duplicate is a tidying
    problem, a deletion is not.
    """
    staging = target.with_name(f"{target.name}.incoming")
    try:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_dir():
                shutil.copytree(item, staging / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, staging / item.name)
    except OSError as exc:
        log.debug("could not stage %s: %s", source, exc)
        shutil.rmtree(staging, ignore_errors=True)
        return False
    try:
        staging.rename(target)
    except OSError as exc:
        log.debug("could not promote %s: %s", staging, exc)
        shutil.rmtree(staging, ignore_errors=True)
        return False
    # Only now, and only what verifiably arrived.
    for item in source.iterdir():
        landed = target / item.name
        if not landed.exists():
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError:  # held open; the copy is safe, so leaving it is fine
            log.debug("left %s in place; it is in use", item)
    try:
        source.rmdir()  # only succeeds if everything above was removed
    except OSError:
        pass
    return True


def migrate_legacy(root: Path | None = None) -> list[str]:
    """Move app state out of a workspace that predates this split.

    Non-fatal by construction: this runs at startup, and a workspace on a
    read-only mount or a directory held open by a running Lab server must not
    stop the app from opening. A destination that already exists is left alone
    rather than merged -- the new location is the live one by then, and merging
    would resurrect stale state over it.

    Returns what moved, for the caller to log.
    """
    from core import paths  # noqa: PLC0415

    base = Path(root) if root is not None else paths.root()
    legacy = base / "data"
    if not legacy.is_dir():
        return []
    ensure()
    moved: list[str] = []
    for name, bucket in _LEGACY:
        source = legacy / name
        if not source.is_dir():
            continue
        target = (app_dir() / bucket / name) if bucket != "cache" else cache_dir()
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if _relocate(source, target):
            moved.append(name)
    return moved
