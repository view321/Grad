"""The single locked write path (HANDOFF §7)."""

from __future__ import annotations

import threading

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


def test_write_json_is_atomic_and_readable(workspace):
    path = workspace / "ledger" / "preflight" / "abc.json"
    jsonl.write_json(path, {"ok": True})
    assert jsonl.read_json(path) == {"ok": True}
    assert not list(path.parent.glob("*.tmp*"))
