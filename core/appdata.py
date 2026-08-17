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
#:
#: `cache` is deliberately absent. Its only consumer -- `core/http.py:_cache_path`
#: -- creates it on demand, and creating it here would make `migrate_legacy`'s
#: "the destination already exists, leave it alone" guard fire on every single
#: run, so a legacy `data/cache` would never move at all.
_SUBDIRS = ("state", "logs", "workspaces")


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


def experiments_dir() -> Path:
    """The cross-workspace experiment archive (see `core/experiments.py`).

    Here rather than under a workspace, and that is the whole point of it: a
    workspace's `ledger/runs.jsonl` answers "what did *this* project do", and
    nothing answered "have I run this before" across the two or three workspaces
    a person accumulates. This directory is the one place in Grad that is
    deliberately global to the user.

    It is emphatically *not* a second source of truth. Each workspace's ledger
    stays authoritative for its own runs; this is a durable copy taken at the
    moment a run becomes terminal, so that deleting a workspace loses the
    working files and not the record that the experiment happened.
    """
    return app_dir() / "experiments"


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

    **Resolved first, and that is the load-bearing line.** The digest is taken
    over the path's *text*, so two spellings of one directory are two different
    workspaces to this function: `D:/work/grad` and `D:/work/./grad`, a relative
    path and its absolute form, a symlink and its target. Every reader arrives
    through `paths.root()`, which resolves; a caller that passes a root
    explicitly -- `migrate_legacy` is the one that does -- may not have. Without
    this line that caller writes into a key nothing ever reads, which is the
    same silent failure as a migration landing in the wrong directory: the
    source is gone, the destination is real, and the app opens on defaults with
    nothing to explain it.
    """
    resolved = Path(root).resolve()
    text = str(resolved).casefold()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(resolved.name)}-{digest}"


def workspace_key(root: Path | None = None) -> str:
    """The stable identifier for a workspace, for callers outside this module.

    `core/experiments.py` needs it to name an experiment, because a run id is
    only unique within one ledger and the archive spans several. Public rather
    than reaching for `_key`: a second caller is what turns a private helper
    into an interface, and the resolution rules in `_key` are exactly the part
    an outside caller must not reimplement.
    """
    if root is None:
        from core import paths  # noqa: PLC0415 - see `workspace_state_dir`

        root = paths.root()
    return _key(Path(root))


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
#: `data/<name>` in a workspace -> which directory it belongs in now. Only
#: entries that are unambiguously app state; `data/papers`, `data/corpus.sqlite`
#: and `data/mnist` are cited or downloaded research and are deliberately absent.
#:
#: The bucket is *which* of the three roots, and it has to match what the code
#: that reads the file actually resolves -- a migration that lands somewhere
#: nothing reads is worse than none, because the old copy is gone and the app
#: silently starts from defaults. `layouts` and `kernel` are per-workspace
#: (`ui/state.py:layout_dir`, `tools/nb.py:_conn_path`); `lab` is one server per
#: installation and `cache` is regenerable, so both are installation-wide.
_INSTALL, _WORKSPACE, _CACHE = "install", "workspace", "cache"
_LEGACY: tuple[tuple[str, str], ...] = (
    ("layouts", _WORKSPACE),
    ("kernel", _WORKSPACE),
    ("lab", _INSTALL),
    ("cache", _CACHE),
)

#: Loose files rather than directories, matched by glob under `data/` itself.
#: Transcripts are the conversations, so they are private, and they are keyed to
#: the workspace for the reason `ui/sessions.py:sessions_dir` gives.
#:
#: `data/nb_verify.json` is *not* here on purpose. It records which notebooks
#: verified clean on a fresh kernel, which is what the CITABLE chip and
#: `report check` rest on -- evidence about the research rather than state about
#: the machine -- so it stays in the workspace with the notebooks it describes.
_LEGACY_FILES: tuple[tuple[str, str], ...] = (
    ("ui_session-*.jsonl", _WORKSPACE),
    ("ui_storage_secret", _INSTALL),
)


def _bucket_dir(bucket: str, name: str, base: Path) -> Path:
    """Where a legacy entry lands, which must be exactly where the code that
    reads it resolves. `data/cache` is the whole cache directory rather than a
    child of it, because `core/http.py` reads `cache_dir()` itself."""
    if bucket == _WORKSPACE:
        return workspace_state_dir(base) / name
    if bucket == _CACHE:
        return cache_dir()
    return state_dir() / name


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


def _relocate_files(legacy: Path, pattern: str, bucket: str, base: Path) -> list[str]:
    """Move loose files sitting directly in `data/`, matched by glob.

    A separate pass because these are not directories and the directory entries
    are not globs. The transcripts are the reason it exists: they are the
    conversations themselves, they sit at the top of `data/` rather than in a
    subdirectory of their own, and a migration that moved the layouts but left
    them behind would open a workspace with its whole history apparently gone --
    still on disk, still private, and no longer anywhere the app looks.

    Copy-then-delete, and never the other way, for the reason in `_relocate`.
    """
    target_dir = workspace_state_dir(base) if bucket == _WORKSPACE else state_dir()
    moved: list[str] = []
    for source in sorted(legacy.glob(pattern)):
        if not source.is_file():
            continue
        target = target_dir / source.name
        if target.exists():
            continue
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            log.debug("could not migrate %s: %s", source, exc)
            continue
        try:
            source.unlink()
        except OSError:  # copied safely; a duplicate beats a deletion
            log.debug("left %s in place; it is in use", source)
        moved.append(source.name)
    return moved


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

    # Resolved, so the destinations computed below are keyed exactly as the
    # readers key them. `paths.root()` already resolves; an explicit argument
    # has no such guarantee. See `_key`.
    base = Path(root).resolve() if root is not None else paths.root()
    legacy = base / "data"
    if not legacy.is_dir():
        return []
    ensure()
    moved: list[str] = []
    for pattern, bucket in _LEGACY_FILES:
        moved += _relocate_files(legacy, pattern, bucket, base)
    for name, bucket in _LEGACY:
        source = legacy / name
        if not source.is_dir():
            continue
        target = _bucket_dir(bucket, name, base)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if _relocate(source, target):
            moved.append(name)
    return moved
