"""Properties of the one write path to the ledgers.

`core/jsonl.py` is the only place in the project that appends to a ledger, and
the ledger is the thing every other claim rests on -- "every number in a report
traces to a run record" is false the moment a record does not survive a
round-trip. Two properties the module's own docstring states, and which are
stated here as tests rather than as prose:

  * writers take an exclusive lock around each line write, so lines never
    interleave;
  * readers tolerate a torn final line.

The generated records deliberately include the characters that break line-based
formats -- embedded newlines, carriage returns, tabs, non-ASCII -- because
`append` writes one record per line and `ensure_ascii=False` means the escaping
is doing real work rather than being a formality.

Each example gets its own directory. The autouse `workspace` fixture in
`tests/conftest.py` is function-scoped, so Hypothesis would otherwise hand every
example the same `tmp_path` and let one example read the previous one's ledger.
"""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

from hypothesis import HealthCheck, given, note, settings
from hypothesis import strategies as st

from core import jsonl

#: JSON-representable values, nested. `allow_nan=False` because NaN is not JSON
#: and a ledger that writes `NaN` writes a file no other reader can parse.
scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=40),
)
records = st.dictionaries(
    st.text(min_size=1, max_size=12),
    st.recursive(
        scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4),
        ),
        max_leaves=6,
    ),
    max_size=6,
)


#: Module-level and never reset, which is the whole point. A counter created
#: inside the test body is re-created for every example, so every example gets
#: `ledger-1.jsonl` and reads the previous one's records -- which is exactly the
#: cross-example leak the per-example directory is supposed to prevent, wearing
#: the costume of a fix. It showed up as a round-trip test reading 242 records
#: back from six appends.
_ledgers = itertools.count()


def _fresh(tmp_path: Path) -> Path:
    """A ledger nothing else has written to.

    Hypothesis reuses the function-scoped `tmp_path` across every example of one
    test, so the isolation each example needs has to come from here.
    """
    return tmp_path / f"ledger-{next(_ledgers)}.jsonl"


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------
@given(st.lists(records, max_size=8))
def test_what_was_appended_is_what_is_read(tmp_path: Path, rows: list[dict]) -> None:
    """Every record, in order, unchanged.

    Order is part of the contract and not an accident of the filesystem: the
    ledgers are append-only and read oldest-first, and `rolling_spend` and the
    quota fold both depend on a record's position meaning its time.
    """
    path = _fresh(tmp_path)
    for row in rows:
        jsonl.append(path, row)
    note(path.read_text(encoding="utf-8") if path.exists() else "<no file>")
    assert jsonl.read(path) == rows


@given(records)
def test_a_record_never_becomes_two_lines(tmp_path: Path, row: dict) -> None:
    """One record, one line, whatever is in it.

    A newline inside a value would otherwise split one record into two, and the
    second half would be dropped as damaged by every reader -- silently, because
    the first half parses.
    """
    path = _fresh(tmp_path)
    jsonl.append(path, row)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.count("\n") == 1


@given(st.lists(records, min_size=1, max_size=6), st.text(max_size=20))
def test_a_torn_final_line_costs_only_itself(
    tmp_path: Path, rows: list[dict], tail: str
) -> None:
    """A reader that opens the file mid-write still gets everything before it.

    This is the property the whole lock design exists to make cheap: a partial
    write at the end is normal, and it must cost one record rather than the
    file.
    """
    path = _fresh(tmp_path)
    for row in rows:
        jsonl.append(path, row)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"partial": ' + tail)
    note(path.read_text(encoding="utf-8"))
    read = jsonl.read(path)
    assert read[: len(rows)] == rows


@given(st.lists(st.booleans(), min_size=1, max_size=8))
def test_damaged_lines_names_exactly_the_lines_that_are_damaged(
    tmp_path: Path, damaged: list[bool]
) -> None:
    """`ledger verify` reports the damage, so its line numbers have to be right.

    1-indexed and counted against the file rather than against the records, or
    the number it prints points a human at the wrong line.
    """
    path = _fresh(tmp_path)
    lines = [
        "{not json" if bad else json.dumps({"i": i}) for i, bad in enumerate(damaged)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    expected = [n for n, bad in enumerate(damaged, start=1) if bad]
    assert jsonl.damaged_lines(path) == expected
    assert len(jsonl.read(path)) == damaged.count(False)


@given(st.lists(records, max_size=5))
def test_blank_lines_are_not_damage(tmp_path: Path, rows: list[dict]) -> None:
    """An empty line is skipped by both readers, and consistently.

    `read` and `damaged_lines` are separate loops over the same file, and the
    one thing worse than a reader that disagrees with itself is a `verify` that
    reports damage the reader silently tolerated.
    """
    path = _fresh(tmp_path)
    body = "\n\n".join(json.dumps(r) for r in rows)
    path.write_text(body + "\n\n\n" if rows else "\n\n", encoding="utf-8")
    assert jsonl.damaged_lines(path) == []
    assert jsonl.read(path) == rows


@given(st.lists(st.one_of(records, st.integers(), st.text(max_size=8)), max_size=6))
def test_only_objects_are_records(tmp_path: Path, values: list) -> None:
    """A bare number on a line is valid JSON and is not a record.

    Every consumer of this module indexes into what it reads, so a naked scalar
    surviving the reader would be an AttributeError several frames away from the
    file that caused it.
    """
    path = _fresh(tmp_path)
    path.write_text(
        "".join(json.dumps(v) + "\n" for v in values), encoding="utf-8"
    )
    assert jsonl.read(path) == [v for v in values if isinstance(v, dict)]
    assert jsonl.damaged_lines(path) == []


# ---------------------------------------------------------------------------
# whole-file JSON
# ---------------------------------------------------------------------------
@given(st.recursive(scalars, lambda c: st.lists(c, max_size=4), max_leaves=6))
def test_a_json_file_round_trips(tmp_path: Path, obj: object) -> None:
    path = _fresh(tmp_path)
    jsonl.write_json(path, obj)
    assert jsonl.read_json(path) == obj
    assert not list(path.parent.glob("*.tmp*")), "a temp file survived the write"


@given(st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=6))
def test_an_update_sees_what_the_last_one_wrote(
    tmp_path: Path, additions: list[int]
) -> None:
    """Read-mutate-write is atomic per *update*, not merely per file.

    The records this guards are the preflight ones, which are the input to the
    gate that decides whether code may cost money: a submitter folding a smoke
    result while `preflight run` writes its own checks must not drop either set.
    """
    path = _fresh(tmp_path)
    for value in additions:
        jsonl.update_json(path, lambda cur, v=value: (cur or []) + [v])
    assert jsonl.read_json(path) == additions


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------
@settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(min_value=2, max_value=6), st.integers(min_value=2, max_value=8))
def test_concurrent_appends_never_interleave(
    tmp_path: Path, writers: int, each: int
) -> None:
    """The claim the lock exists for, from more than one thread.

    An OS file lock keeps *processes* apart and does nothing about threads in
    one process -- the UI and an in-process CLI are exactly that case -- so the
    per-path mutex is what closes it. A torn line here is a corrupted ledger,
    which is unrecoverable rather than merely wrong.
    """
    path = _fresh(tmp_path)

    def write(worker: int) -> None:
        for i in range(each):
            jsonl.append(path, {"worker": worker, "i": i, "pad": "x" * 200})

    threads = [threading.Thread(target=write, args=(w,)) for w in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert jsonl.damaged_lines(path) == []
    read = jsonl.read(path)
    assert len(read) == writers * each
    for worker in range(writers):
        mine = [r["i"] for r in read if r["worker"] == worker]
        assert mine == list(range(each)), "one writer's records lost their order"
