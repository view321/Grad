"""The one write path to the append-only ledgers (HANDOFF §7).

    "The ledger files are multi-writer: the CLIs, the funnel stages inside
     paper_search.py, and the Stop hook all append, while the UI reads
     concurrently -- and Windows is less forgiving about concurrent file access
     than POSIX. So there is exactly one write path."

Two properties this module is responsible for:

  * writers take an exclusive lock around each line write, so lines never
    interleave;
  * readers tolerate a torn final line, because a reader may open the file
    between a partial write and its flush.

Windows locks by hand, POSIX through `portalocker` when it is installed and
`fcntl` when it is not. The reason that split is not a preference is spelled out
above `_lock`; the short version is that a Windows lock denies reads, so it has
to be taken somewhere other than over the data. The `fcntl` fallback is real,
not decorative -- the ledger must not depend on an optional package to stay
uncorrupted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.02
# One fixed byte, past any plausible ledger, used purely as a mutex.
_LOCK_OFFSET = 1 << 40

# The OS file lock is what keeps *processes* from interleaving. It does not keep
# *threads* in one process apart -- `msvcrt.locking` is per-process, so two
# threads of the UI and a CLI in-process would both "hold" it. One in-process
# mutex per path closes that half.
_thread_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve() if path.parent.exists() else path)
    with _registry_lock:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[key] = lock
        return lock


# Which backend locks which platform is decided by one asymmetry: **Windows
# byte-range locks are mandatory and POSIX ones are advisory.** A locked region
# on Windows denies reads as well as writes, to every handle including another
# one in this process; `fcntl.flock` denies nothing and only excludes other
# lockers. So on Windows *where* the lock is taken is load-bearing, and on POSIX
# it is not.
#
# On Windows all writers therefore contend on one fixed sentinel byte positioned
# far past any real ledger, never over the data. A lock over live data would make
# concurrent readers -- and a `precondition` that consults the file it is being
# appended to -- fail with PermissionError.
#
# `portalocker` cannot express that, which is why it is not used here: its
# `MsvcrtLocker` normalises the file position to 0 and locks 64 KiB from there,
# which is exactly over the data. Preferring it on Windows is what made
# `campaign.request_halt` and `ledger_store`'s uniqueness check fail against
# their own ledgers, and it denied the UI's two-second poll for the length of
# every append.
if os.name == "nt":  # pragma: no cover - one branch per platform
    import msvcrt

    # msvcrt locks a byte range at the *current file position*, and a handle
    # opened in append mode starts at EOF -- so left alone, every writer would
    # lock a different byte as the file grows and none would exclude any other.
    # The position is set with os.lseek on the descriptor, which is what
    # msvcrt.locking reads; O_APPEND still sends every write to EOF.
    def _lock(fh) -> None:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                os.lseek(fh.fileno(), _LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(_LOCK_POLL_S)

    def _unlock(fh) -> None:
        try:
            os.lseek(fh.fileno(), _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:  # pragma: no cover - one branch per platform
    try:
        import portalocker

        def _lock(fh) -> None:
            portalocker.lock(fh, portalocker.LOCK_EX)

        def _unlock(fh) -> None:
            portalocker.unlock(fh)

    except ImportError:
        import fcntl

        def _lock(fh) -> None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

        def _unlock(fh) -> None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def append(
    path: Path | str,
    record: dict[str, Any],
    *,
    precondition: Any = None,
) -> dict[str, Any]:
    """Append one JSON record as one line, under an exclusive lock.

    No CLI opens a ledger file for writing directly; they all call this.
    Returns the record, so callers can write and use it in one expression.

    `precondition` is an optional callable run *while the lock is held*, before
    the write, and may raise to abort it. That is what lets a uniqueness check
    ("is this expectation already bound?") be atomic with the append rather than
    a check that another process can win the race against.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    if "\n" in line:  # defensive: json.dumps escapes newlines, but the invariant matters
        raise ValueError("record serialised to a multi-line string")

    with _thread_lock(path):
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            _lock(fh)
            try:
                if precondition is not None:
                    precondition()
                # No seek: the handle is open with O_APPEND, so every write
                # lands at EOF regardless of where the descriptor points -- and
                # it points at the lock sentinel.
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                _unlock(fh)
    return record


def read(path: Path | str) -> list[dict[str, Any]]:
    """All well-formed records, oldest first. A torn final line is dropped."""
    return list(iter_records(path))


def iter_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records, tolerating a torn tail.

    A malformed line anywhere other than the end is unexpected and is skipped
    rather than raised: a ledger that cannot be read at all is worse than a
    ledger missing one line, and `grad-ledger verify` reports the damage.
    """
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def damaged_lines(path: Path | str) -> list[int]:
    """1-indexed line numbers that failed to parse. Used by `ledger verify`."""
    path = Path(path)
    bad: list[int] = []
    if not path.exists():
        return bad
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad.append(n)
    return bad


def write_json(path: Path | str, obj: Any) -> None:
    """Atomic whole-file JSON write, for the preflight records in §6.

    These are replaced wholesale, so a temp file plus `os.replace` is the right
    tool rather than the append lock.

    The temp name carries the *thread* as well as the process. The UI persists a
    layout through here, one `Workspace` per connected client in one process --
    so two windows open on the same project are two threads writing the same
    path, and a pid-only name would give them the same temp file to interleave
    into. `os.replace` would then publish whichever half won.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}.{threading.get_ident():x}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path | str) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def update_json(path: Path | str, mutate: Any) -> Any:
    """Read, mutate, and write a JSON file with the whole sequence locked.

    `write_json` is atomic per *file*, which is not the same as atomic per
    *update*: a preflight record is read, one check is inserted, and the result
    is written back, so a submitter folding a smoke result while `preflight run`
    is writing its own checks means one of the two sets of checks is silently
    dropped -- and these records are the input to the gate that decides whether
    code may cost money.

    The lock is taken on a sidecar rather than on the file itself, because
    `write_json` replaces the file and a lock held on the replaced inode
    protects nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _thread_lock(lock_path):
        with open(lock_path, "a+", encoding="utf-8") as fh:
            _lock(fh)
            try:
                current = read_json(path)
                updated = mutate(current)
                write_json(path, updated)
                return updated
            finally:
                _unlock(fh)
