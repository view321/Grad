"""Named chat sessions: the file format, the listing, and the two ids.

No NiceGUI and no SDK. What is under test is the part that has rules -- which
file an id names, what a session is called when nobody named it, what survives
an upgrade, and the difference between resuming a conversation and redisplaying
a transcript.

That last one is the property worth stating plainly, because getting it wrong is
invisible: a session reopened without its *SDK* id shows every turn above a
composer whose next turn the agent has no memory of. The transcript and the
conversation are two different things and only one of them is in the file.
"""

from __future__ import annotations

import json

import pytest

from ui import sessions


def write_raw(workspace, session_id: str, lines: list[dict]) -> None:
    path = sessions.path_for(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def user(text: str) -> dict:
    return {"role": "user", "text": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "text": text}


# ---------------------------------------------------------------------------
# ids and paths
# ---------------------------------------------------------------------------
def test_a_new_id_is_sortable_and_unique(workspace):
    first, second = sessions.new_id(), sessions.new_id()
    assert first != second
    assert sessions.is_id(first)
    # Timestamp first, so a plain glob comes back in a sensible order.
    assert first[:8].isdigit()


@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "with/slash", "with\\slash", "", ".hidden", "a" * 200, None, 12],
)
def test_an_id_that_is_not_one_never_reaches_a_filename(workspace, bad):
    """The id comes off disk or out of a click handler, so it is validated
    rather than trusted -- the same reason `state.layout_path` sanitises."""
    assert sessions.is_id(bad) is False
    with pytest.raises(ValueError):
        sessions.path_for(bad)


def test_the_id_round_trips_through_the_filename(workspace):
    session_id = sessions.new_id()
    assert sessions.id_of(sessions.path_for(session_id)) == session_id


# ---------------------------------------------------------------------------
# what a session is called
# ---------------------------------------------------------------------------
def test_a_session_is_named_after_the_first_thing_asked_in_it(workspace):
    write_raw(workspace, "abc", [user("derive the update rule for Adam"), assistant("Sure.")])
    assert sessions.read_meta(sessions.path_for("abc"))["title"] == (
        "derive the update rule for Adam"
    )


def test_an_explicit_title_outranks_the_derived_one(workspace):
    write_raw(
        workspace, "abc",
        [{"type": "meta", "id": "abc", "title": "width vs depth"}, user("something else")],
    )
    assert sessions.read_meta(sessions.path_for("abc"))["title"] == "width vs depth"


def test_a_long_first_message_is_cut_rather_than_filling_the_picker(workspace):
    write_raw(workspace, "abc", [user("x" * 400)])
    title = sessions.read_meta(sessions.path_for("abc"))["title"]
    assert len(title) <= sessions.TITLE_CHARS
    assert title.endswith("…")


def test_a_session_with_nothing_in_it_still_has_a_name(workspace):
    write_raw(workspace, "abc", [{"type": "meta", "id": "abc", "title": ""}])
    assert sessions.read_meta(sessions.path_for("abc"))["title"] == "empty session"


# ---------------------------------------------------------------------------
# the listing
# ---------------------------------------------------------------------------
def test_the_listing_is_most_recently_written_first(workspace):
    import os
    import time

    for index, session_id in enumerate(("aaa", "bbb", "ccc")):
        write_raw(workspace, session_id, [user(f"question {index}")])
        # mtimes, set explicitly: three writes inside one filesystem tick would
        # otherwise be indistinguishable and the order arbitrary.
        stamp = time.time() + index
        os.utime(sessions.path_for(session_id), (stamp, stamp))

    assert [row["id"] for row in sessions.listing()] == ["ccc", "bbb", "aaa"]


def test_the_listing_counts_messages_and_ignores_the_meta_line(workspace):
    write_raw(
        workspace, "abc",
        [{"type": "meta", "id": "abc", "title": "t"}, user("one"), assistant("two")],
    )
    assert sessions.listing()[0]["messages"] == 2


def test_a_damaged_transcript_does_not_make_the_picker_unopenable(workspace):
    path = sessions.path_for("broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all\n" + json.dumps(user("hello")), encoding="utf-8")
    write_raw(workspace, "fine", [user("a good one")])

    listed = {row["id"]: row for row in sessions.listing()}
    assert "fine" in listed
    assert listed["broken"]["title"] == "hello", "the readable lines still count"


def test_files_that_are_not_sessions_are_left_alone(workspace):
    (sessions.sessions_dir() / "notes.jsonl").write_text("{}", encoding="utf-8")
    write_raw(workspace, "abc", [user("hello")])
    assert [row["id"] for row in sessions.listing()] == ["abc"]


# ---------------------------------------------------------------------------
# the upgrade path
# ---------------------------------------------------------------------------
def test_the_transcript_that_existed_before_sessions_is_already_one(workspace):
    """It was `ui_session-default.jsonl` and the scheme here is
    `ui_session-<id>.jsonl`, so there is no migration step -- which is the whole
    reason the naming was left alone."""
    path = sessions.sessions_dir() / "ui_session-default.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(user("the conversation from before")), encoding="utf-8")

    listed = sessions.listing()
    assert [row["id"] for row in listed] == [sessions.LEGACY_ID]
    assert listed[0]["title"] == "the conversation from before"
    assert listed[0]["resumable"] is False, "no SDK id was ever recorded for it"


def test_most_recent_is_none_on_a_workspace_with_no_sessions(workspace):
    assert sessions.most_recent() is None


# ---------------------------------------------------------------------------
# the two ids
# ---------------------------------------------------------------------------
def test_a_written_session_round_trips_with_both_ids(workspace):
    sessions.write(
        "abc", [user("hello"), assistant("hi")],
        title="a title", created_at="2026-08-15T00:00:00Z", sdk_session_id="sdk-1",
    )
    meta = sessions.read_meta(sessions.path_for("abc"))
    assert meta["title"] == "a title"
    assert meta["created_at"] == "2026-08-15T00:00:00Z"
    assert meta["sdk_session_id"] == "sdk-1"
    assert meta["messages"] == 2


def test_resumable_is_exactly_whether_the_sdk_id_is_known(workspace):
    """The difference between reopening a conversation and reopening a
    transcript. A picker that showed both the same way would be lying about the
    more important half."""
    sessions.write("with", [user("hi")], sdk_session_id="sdk-1")
    sessions.write("without", [user("hi")])

    listed = {row["id"]: row for row in sessions.listing()}
    assert listed["with"]["resumable"] is True
    assert listed["without"]["resumable"] is False


def test_the_meta_line_is_not_mistaken_for_a_transcript_record(workspace):
    """`app.Session.restore` keeps only records whose `role` is a real role, so
    the header costs nothing there -- which is why it is a line in the same file
    rather than a second file that could disagree with it."""
    from ui import app as app_mod

    sessions.write("abc", [user("hello")], title="t", sdk_session_id="sdk-1")
    lines = sessions.path_for("abc").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["type"] == "meta"
    assert first.get("role") not in app_mod.ROLES


def test_deleting_a_session_is_idempotent(workspace):
    sessions.write("abc", [user("hello")])
    assert sessions.delete("abc") is True
    assert sessions.delete("abc") is False
    assert sessions.listing() == []


# ---------------------------------------------------------------------------
# listing cost
# ---------------------------------------------------------------------------
def test_reading_a_written_session_stops_at_the_header(workspace, monkeypatch):
    """`listing()` calls `read_meta` once per session, so a scan to the end
    would make listing cost the total size of every transcript in the
    workspace -- and transcripts grow with the conversations most worth coming
    back to."""
    sessions.write("abc", [user("hello"), assistant("hi")], title="a title")

    path = sessions.path_for("abc")
    real_open = open
    read_lines: list[int] = []

    class CountingFile:
        def __init__(self, handle):
            self._handle = handle
            self._count = 0

        def __iter__(self):
            for line in self._handle:
                self._count += 1
                yield line

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            read_lines.append(self._count)
            return self._handle.__exit__(*exc)

    def counting_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if str(file) == str(path):
            return CountingFile(handle.__enter__().__class__ and handle)
        return handle

    monkeypatch.setattr("builtins.open", counting_open)
    meta = sessions.read_meta(path)

    assert meta["messages"] == 2
    assert read_lines == [1], "the header alone answered it"


def test_a_legacy_file_with_no_header_still_falls_back_to_a_full_scan(workspace):
    """One file, once. A transcript written before this format existed has no
    header, so its title is its first user message and its count is its lines."""
    write_raw(workspace, sessions.LEGACY_ID, [user("from before"), assistant("hi")])
    meta = sessions.read_meta(sessions.path_for(sessions.LEGACY_ID))
    assert meta["title"] == "from before"
    assert meta["messages"] == 2


# ---------------------------------------------------------------------------
# one client per session
# ---------------------------------------------------------------------------
def test_a_session_is_claimed_by_one_client_at_a_time(workspace):
    """`_persist` writes the whole file, so two clients on one session is two
    writers and the loser's turns disappear with nothing to say they had."""
    sessions.write("abc", [user("hello")])
    assert sessions.claim("abc", "client-1") is True
    assert sessions.claim("abc", "client-2") is False
    assert sessions.claim("abc", "client-1") is True, "the holder may re-take it"
    assert sessions.holder("abc") == "client-1"

    sessions.release("client-1")
    assert sessions.holder("abc") is None
    assert sessions.claim("abc", "client-2") is True


def test_a_second_window_gets_the_next_session_not_the_same_one(workspace):
    import os
    import time

    for index, session_id in enumerate(("older", "newer")):
        sessions.write(session_id, [user(f"q{index}")])
        stamp = time.time() + index
        os.utime(sessions.path_for(session_id), (stamp, stamp))

    assert sessions.most_recent("client-1") == "newer"
    assert sessions.most_recent("client-2") == "older"
    # Nothing left to hand out.
    assert sessions.most_recent("client-3") is None


def test_most_recent_without_an_owner_claims_nothing(workspace):
    sessions.write("abc", [user("hello")])
    assert sessions.most_recent() == "abc"
    assert sessions.holder("abc") is None


def test_a_claim_is_scoped_to_the_workspace_it_was_made_in(monkeypatch, tmp_path, workspace):
    """Every root has a `default`, so a claim keyed on the bare id would let one
    client switching workspaces find another client's claim on a file it has
    never seen."""
    sessions.write(sessions.LEGACY_ID, [user("in the first workspace")])
    assert sessions.claim(sessions.LEGACY_ID, "client-1") is True

    other = tmp_path / "another-workspace"
    other.mkdir()
    monkeypatch.setenv("GRAD_ROOT", str(other))
    from core import paths

    paths.ensure_workspace()
    sessions.write(sessions.LEGACY_ID, [user("in the second workspace")])

    assert sessions.holder(sessions.LEGACY_ID) is None
    assert sessions.claim(sessions.LEGACY_ID, "client-2") is True
