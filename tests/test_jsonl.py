"""The single locked write path (HANDOFF §7)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from core import jsonl


def test_append_and_read_roundtrip(workspace):
    path = workspace / "ledger" / "runs.jsonl"
    jsonl.append(path, {"id": "a", "n": 1})
    jsonl.append(path, {"id": "b", "n": 2})
    assert [r["id"] for r in jsonl.read(path)] == ["a", "b"]


def test_torn_final_line_is_tolerated(workspace):
    """A reader may open the file between a partial write and its flush."""
    path = workspace / "ledger" / "runs.jsonl"
    jsonl.append(path, {"id": "a"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"id": "b", "trunc')
    assert [r["id"] for r in jsonl.read(path)] == ["a"]
    assert jsonl.damaged_lines(path) == [2]


def test_concurrent_appends_do_not_interleave(workspace):
    """Windows is less forgiving about concurrent file access than POSIX;
    interleaved partial lines are a real outcome, not theory."""
    path = workspace / "ledger" / "quota.jsonl"
    payload = "x" * 500

    def writer(tag: int) -> None:
        for i in range(25):
            jsonl.append(path, {"tag": tag, "i": i, "pad": payload})

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = jsonl.read(path)
    assert len(records) == 100
    assert jsonl.damaged_lines(path) == []
    assert all(r["pad"] == payload for r in records)


HOLDER = """
import pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from core import jsonl
fh = open(sys.argv[2], "a", encoding="utf-8", newline="\\n")
jsonl._lock(fh)
# Grow the file *while holding the lock*: this is what separates a lock on a
# fixed byte from a lock on "wherever this handle happens to be pointing".
fh.write("x" * 4096 + "\\n")
fh.flush()
pathlib.Path(sys.argv[3]).write_text("ready", encoding="utf-8")
time.sleep(8)
jsonl._unlock(fh)
fh.close()
"""


@pytest.mark.skipif(
    importlib.util.find_spec("portalocker") is not None,
    reason="covers the msvcrt/fcntl fallback; portalocker blocks rather than timing out",
)
def test_the_file_lock_actually_excludes_a_second_process(workspace, monkeypatch):
    """A second process must not be able to take the lock while it is held.

    This is the property the fallback exists for, and the one that is easy to
    lose silently: `msvcrt.locking` locks a byte range at the *current* file
    position, and an append handle opens at EOF -- so as the ledger grows, two
    writers lock two different bytes and neither excludes the other. Nothing
    about the resulting file looks wrong until the day a write is split.
    """
    path = workspace / "ledger" / "held.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    sentinel = workspace / "held.ready"
    repo = Path(__file__).resolve().parent.parent

    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(repo), str(path), str(sentinel)])
    try:
        deadline = time.time() + 30
        while not sentinel.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert sentinel.exists(), "the holder process never acquired the lock"

        monkeypatch.setattr(jsonl, "_LOCK_TIMEOUT_S", 1.0)
        contender = open(path, "a", encoding="utf-8", newline="\n")
        try:
            with pytest.raises(OSError):
                jsonl._lock(contender)
        finally:
            contender.close()
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_concurrent_appends_from_separate_processes(workspace):
    """Separate processes, one file, no in-process mutex to help.

    Note what this does and does not prove: O_APPEND already makes a single
    small write atomic, so this passes even with a broken lock. It is here to
    catch damage from the surrounding logic -- truncation, lost records, a torn
    line from a large payload -- not to prove mutual exclusion. The test above
    proves that.
    """
    path = workspace / "ledger" / "multiproc.jsonl"
    repo = Path(__file__).resolve().parent.parent
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from core import jsonl;"
        "tag = sys.argv[3];"
        "[jsonl.append(sys.argv[2], {'tag': tag, 'i': i, 'pad': 'y' * 400}) for i in range(20)]"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", script, str(repo), str(path), str(tag)])
        for tag in range(4)
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    records = jsonl.read(path)
    assert jsonl.damaged_lines(path) == []
    assert len(records) == 80
    assert all(r["pad"] == "y" * 400 for r in records)
    assert sorted((r["tag"], r["i"]) for r in records) == sorted(
        (str(t), i) for t in range(4) for i in range(20)
    )


def test_write_json_is_atomic_and_readable(workspace):
    path = workspace / "ledger" / "preflight" / "abc.json"
    jsonl.write_json(path, {"ok": True})
    assert jsonl.read_json(path) == {"ok": True}
    assert not list(path.parent.glob("*.tmp*"))
