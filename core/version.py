"""Which Grad this is, and whether it has been modified.

Two questions, and the second is the one that makes this module worth having.

The README's claim is that "every number in a report traces to a run record".
That is only *complete* if the record also says which code produced it: a run
submitted before an update and one submitted after are two different
experiments, and a ledger that cannot tell them apart will happily let a report
average across them. So `stamp()` goes into every run record at submit time, and
`core/report.py` refuses a report whose cited runs straddle a version.

**The identity is the checkout's, not the workspace's.** `core/paths.py` resolves
where the research lives; this resolves where the *code* lives, which is
`core/workspace.py:code_dir()`. With a workspace pointed elsewhere the two are
different directories, and asking git about the wrong one would stamp every run
with the version of whatever repository the research happens to sit in.

**Dirty means the code was edited, not the research.** With the default layout
the workspace *is* the checkout, so `git status` reports every notebook and every
ledger append as a modification -- and a `dirty` flag that is permanently true on
the standard install describes nothing. `WORKSPACE_PATHS` is the list of
top-level directories that hold research rather than code, and everything under
them is excluded from this judgement. `core/update.py` reuses the same split to
decide what may block a fast-forward.

Every git call degrades rather than raises: no git on PATH, a tarball install
with no `.git`, a repository so broken that `rev-parse` fails. All three are
real, none of them should stop a run from being submitted, and each lands as
`source: "package"` with whatever the installed metadata knows.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from core import spawn, workspace

log = logging.getLogger("grad.version")

#: Top-level entries that belong to the *workspace* rather than to the code.
#: A change under one of these is research, not a modification of Grad, and it
#: must not make the installation look dirty or block an update.
#:
#: `data/` is here because it is downloads and derived indexes; `reports/` and
#: `figures/` because they are outputs. `config/` is deliberately *absent*: it is
#: shipped with the code, a workspace copy overrides it (see
#: `core/paths.py:config_path`), and an edit to the shipped one is a real
#: modification that an update would conflict with.
WORKSPACE_PATHS = (
    "ledger",
    "notebooks",
    "notes",
    "figures",
    "reports",
    "data",
    "evals",
    ".grad-workspace.json",
)

#: Long enough for a cold `git rev-parse` on a spinning disk, short enough that a
#: wedged git cannot hang a submission. Nothing here is on a network path --
#: `core/update.py` owns fetch and sets its own, larger, bound.
_TIMEOUT_S = 10.0

_cache: dict[str, Any] | None = None


def code_dir() -> Path:
    """Where Grad itself is installed. See the module docstring."""
    return workspace.code_dir()


def git(*args: str, cwd: Path | None = None, timeout: float = _TIMEOUT_S) -> str | None:
    """Run git in the checkout and return stdout, or None if it could not.

    None is the answer to every failure -- no binary, not a repository, a
    non-zero exit -- because every caller here has the same fallback and none of
    them can do anything useful with the difference. `core/update.py` needs to
    tell "git is missing" from "git said no", so it calls `git_result`.
    """
    result = git_result(*args, cwd=cwd, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    # `rstrip`, never `strip`. The leading whitespace is data in exactly one
    # place and it is the place that matters most here: `status --porcelain`
    # puts the worktree status in column 2, so an unstaged modification is
    # " M core/thing.py" and stripping the front of the output shifts the first
    # line by one -- which does not fail, it silently reports "ore/thing.py".
    return (result.stdout or "").rstrip()


def git_result(
    *args: str, cwd: Path | None = None, timeout: float = _TIMEOUT_S
) -> subprocess.CompletedProcess | None:
    """`git` with the exit code and stderr intact, or None if it never ran.

    `spawn.run` rather than `subprocess.run`: the desktop app is a GUI process,
    and a bare git call from it flashes a console window on Windows. That is the
    whole reason `core/spawn.py` exists.
    """
    try:
        return spawn.run(
            ["git", *args],
            cwd=str(cwd or code_dir()),
            capture_output=True,
            text=True,
            timeout=timeout,
            # A git that stops to ask for a password would hang the app forever;
            # there is nothing here that should ever need one.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("git %s: %s", " ".join(args), exc)
        return None


def is_checkout() -> bool:
    """Whether the installation is a git checkout we can update in place."""
    return git("rev-parse", "--git-dir") is not None


def _pyproject_version() -> str | None:
    """The version in the checkout's own `pyproject.toml`.

    Preferred over `importlib.metadata`, and the reason is the case this module
    exists for: with an editable install the metadata is written at `pip install`
    time and does not move when the files do. After a fast-forward that changed
    no dependencies -- which `core/update.py` deliberately does not reinstall,
    because with an editable install it does not need to -- the metadata still
    reports the version the user had *yesterday*. The file on disk is the code
    that is actually running.
    """
    path = code_dir() / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # Deliberately not tomllib: this runs on the submission path and a malformed
    # pyproject must degrade to "unknown version", not raise from inside a gate.
    match = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else None


def _metadata_version() -> str | None:
    try:
        from importlib import metadata  # noqa: PLC0415

        return metadata.version("grad")
    except Exception:  # noqa: BLE001 - not installed, or installed under another name
        return None


def status_paths(*, untracked: str = "normal") -> list[str]:
    """Every locally modified or untracked path, sorted. Research included.

    `untracked` is the one knob, and it is a real trade rather than a default to
    ignore. Under `normal` git collapses a wholly-untracked directory into a
    single entry -- `notebooks/` rather than the four hundred files in it --
    which is exactly what `dirty_paths` wants: it only needs to know *which side
    of the split* the change is on, and it is called from `identity()`, which is
    called from `stamp()`, which is on the submission path. Under `all` every
    file is listed, which is what `core/update.py` needs to decide whether an
    incoming commit touches a file the user has edited: a collision cannot be
    detected against a directory name. That walk can be slow over a corpus, and
    it runs only in `plan()`, which is not on anyone's hot path.

    Empty when git cannot answer, which is the same shape as "clean". That is
    the right default here: an installation nobody can inspect should not be
    permanently branded as modified, and `core/update.py` refuses on
    `is_checkout()` being false long before it consults this.
    """
    porcelain = git("status", "--porcelain", f"--untracked-files={untracked}")
    if not porcelain:
        return []
    out: list[str] = []
    for line in porcelain.splitlines():
        path = _porcelain_path(line)
        if path:
            out.append(path)
    return sorted(set(out))


def dirty_paths() -> list[str]:
    """Modified or untracked *code* paths -- the ones that mean "this Grad has
    been edited". The research half is `status_paths()` minus these."""
    return [path for path in status_paths() if not is_workspace_path(path)]


def _porcelain_path(line: str) -> str | None:
    """The path out of one `--porcelain` line.

    Two shapes have to survive this. A rename is `R  old -> new`, and the name
    that matters is the destination. A path with a space, a quote or a non-ASCII
    byte comes back C-quoted (`"a b/c.py"`) unless `core.quotePath` is off, and
    the quotes are not part of the name -- left in, every such file would fail
    the `WORKSPACE_PATHS` prefix test and look like modified code.
    """
    if len(line) < 4:
        return None
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        try:
            path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
        except (UnicodeDecodeError, UnicodeEncodeError):
            path = path[1:-1]
    return path.replace("\\", "/").strip("/") or None


def is_workspace_path(path: str) -> bool:
    """Whether a repository-relative path holds research rather than code."""
    head = path.replace("\\", "/").strip("/").split("/", 1)[0]
    return head in WORKSPACE_PATHS


def identity(*, reload: bool = False) -> dict[str, Any]:
    """What this installation is. Cached for the life of the process.

    Cached because `stamp()` is called on the submission path and shelling out
    to git three times per run record would be three subprocesses in the middle
    of a gate. Nothing here can change without the code changing, and code that
    changes under a running process is not a state this cache owes an answer to
    -- `core/update.py` refuses to run while an instance is up for exactly that
    reason.
    """
    global _cache
    if _cache is not None and not reload:
        return _cache
    version = _pyproject_version() or _metadata_version()
    data: dict[str, Any] = {
        "version": version,
        "tag": None,
        "commit": None,
        "branch": None,
        "dirty": False,
        "source": "package",
        "code_dir": str(code_dir()),
    }
    if is_checkout():
        commit = git("rev-parse", "HEAD")
        # `--exact-match` on purpose: "the release this is" and "the release this
        # came after" are different claims, and `git describe` without it happily
        # answers the second while looking like the first.
        tag = git("describe", "--tags", "--exact-match", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        modified = dirty_paths()
        data.update(
            {
                "tag": tag or None,
                "commit": (commit or "")[:12] or None,
                # Detached HEAD reports the literal string "HEAD", which is not a
                # branch name and must not be recorded as one.
                "branch": None if branch in (None, "HEAD") else branch,
                "dirty": bool(modified),
                "dirty_paths": modified[:20],
                "source": "git",
            }
        )
    _cache = data
    return data


def stamp() -> dict[str, Any]:
    """The compact identity that goes into a run record.

    Four keys and no more: this is written to every run and read by
    `core/report.py`, so it is a schema, and a record that carries the whole of
    `identity()` would carry `code_dir` -- an absolute path from someone's
    machine -- into a ledger meant to be committed and shared.
    """
    data = identity()
    return {
        "version": data.get("version"),
        "tag": data.get("tag"),
        "commit": data.get("commit"),
        "dirty": bool(data.get("dirty")),
    }


def label(data: dict[str, Any] | None = None) -> str:
    """One line for a status bar or a menu: what is installed, and is it clean."""
    data = data or identity()
    name = data.get("tag") or (f"v{data['version']}" if data.get("version") else None)
    commit = data.get("commit")
    if name and commit and not data.get("tag"):
        # Not on a release tag: the commit is the only honest identifier, and
        # saying "v0.1.0" alone would name a release this is merely descended
        # from. The version stays because it is what the file claims.
        name = f"{name}+{commit}"
    elif not name:
        name = commit or "unknown"
    return f"{name}{' (modified)' if data.get('dirty') else ''}"


def same_version(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Whether two run stamps describe the same code.

    Compared on the commit when both have one, because that is the only field
    that actually identifies a tree -- two runs can share a version string and a
    tag while thirty commits apart on a development branch. A stamp with no
    commit (a package install, or a ledger written before this existed) falls
    back to the version string, and two runs that both know nothing are treated
    as the same rather than as a finding: `report check` should refuse a report
    on evidence, not on absence.
    """
    a, b = a or {}, b or {}
    if a.get("commit") and b.get("commit"):
        return a["commit"] == b["commit"] and a.get("dirty") == b.get("dirty")
    return (a.get("version"), a.get("tag")) == (b.get("version"), b.get("tag"))
