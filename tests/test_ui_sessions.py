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
