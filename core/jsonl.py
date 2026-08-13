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

`portalocker` is used when installed; otherwise we fall back to `msvcrt` on
Windows and `fcntl` elsewhere. The fallback is real, not decorative -- the
ledger must not depend on an optional package to stay uncorrupted.
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


try:  # pragma: no cover - exercised by whichever branch the machine has
    import portalocker

    def _lock(fh) -> None:
        portalocker.lock(fh, portalocker.LOCK_EX)

    def _unlock(fh) -> None:
        portalocker.unlock(fh)

except ImportError:  # pragma: no cover
    if os.name == "nt":
        import msvcrt

        def _lock(fh) -> None:
            deadline = time.monotonic() + _LOCK_TIMEOUT_S
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(_LOCK_POLL_S)

        def _unlock(fh) -> None:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    else:
        import fcntl

        def _lock(fh) -> None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

        def _unlock(fh) -> None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def append(path: Path | str, record: dict[str, Any]) -> dict[str, Any]:
    """Append one JSON record as one line, under an exclusive lock.

    No CLI opens a ledger file for writing directly; they all call this.
    Returns the record, so callers can write and use it in one expression.
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
                fh.seek(0, os.SEEK_END)
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

    These are single-writer and replaced wholesale, so a temp file plus
    os.replace is the right tool rather than the append lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
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
