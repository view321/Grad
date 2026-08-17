"""Rewinding a turn: what it drops, what it keeps, and what it admits to.

No NiceGUI and no SDK. What is under test is the part that has rules -- where a
rewind may cut, which SDK entry it resumes at, what the file still holds
afterwards, and the distinction the whole feature turns on: the transcript
always rewinds, the model's memory only sometimes does, and a rewind that
conflated the two would be lying about the expensive half.

That distinction is invisible from the screen, which is exactly why it is
asserted here rather than left to be noticed when the agent answers a question
about a turn nobody can see any more.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core import rewind
from ui import app as app_mod
from ui import sessions

SDK = "sdk-session-1"


def user(text: str) -> dict:
    return {"role": "user", "text": text}


def answer(text: str, uuid: str | None = None, sdk: str | None = SDK) -> dict:
    record: dict = {"role": "assistant", "text": text, "blocks": [{"kind": "text", "text": text}]}
    if uuid:
        record["uuid"] = uuid
        record["sdk_session_id"] = sdk
    return record


def conversation() -> list[dict]:
    """Three exchanges, each with the anchor its turn ended at."""
    return [
        user("what is the scaling law here"),
        answer("roughly N^0.34", "entry-1"),
        user("plot it"),
        answer("plotted", "entry-2"),
        user("Sorry, API errored, continue"),
        answer("the session failed: `APIError`", "entry-3"),
    ]


# ---------------------------------------------------------------------------
# where a rewind may cut
# ---------------------------------------------------------------------------
def test_the_rewind_points_are_the_things_the_user_said(workspace):
    """Anchoring on an answer would keep a question whose reply is gone, which
    reads as an agent that ignored it."""
    assert rewind.points(conversation()) == [0, 2, 4]


def test_a_rewind_onto_an_answer_is_refused(workspace):
    plan = rewind.plan(conversation(), 1)
    assert plan["ok"] is False
    assert "asked" in plan["reason"]


@pytest.mark.parametrize("bad", [-1, 99, "2", None, True])
def test_an_index_that_is_not_one_never_cuts_anything(workspace, bad):
    """The index comes out of a click handler bound to a transcript that may
    have been rebuilt underneath it, so it is validated rather than trusted."""
    assert rewind.plan(conversation(), bad)["ok"] is False


def test_a_plan_splits_the_transcript_and_offers_the_prompt_back(workspace):
    plan = rewind.plan(conversation(), 4)
    assert plan["ok"] is True
    assert len(plan["keep"]) == 4
    assert plan["dropped"] == conversation()[4:]
    assert plan["prompt"] == "Sorry, API errored, continue"
    assert plan["turns"] == 1


def test_a_rewind_past_several_turns_counts_them_all(workspace):
    assert rewind.plan(conversation(), 2)["turns"] == 2


# ---------------------------------------------------------------------------
# the anchor, which is the half that can fail
# ---------------------------------------------------------------------------
def test_the_anchor_is_the_end_of_the_last_turn_being_kept(workspace):
    """`resume_session_at` takes the last transcript entry of the kept turn, so
    a plan that cuts at the third exchange resumes at the second's."""
    assert rewind.plan(conversation(), 4, sdk_session_id=SDK)["anchor"] == "entry-2"


def test_a_uuid_from_another_conversation_is_not_an_anchor(workspace):
    """A uuid names an entry inside the session that issued it. Passing one
    across that boundary would name an entry the resumed conversation does not
    contain -- a rewind that looks like it worked and did nothing."""
    settled = [user("a"), answer("b", "entry-1", sdk="a-different-session"), user("c")]
    assert rewind.plan(settled, 2, sdk_session_id=SDK)["anchor"] is None


def test_turns_without_an_anchor_are_stepped_over_not_stopped_at(workspace):
    """A user's own message never has one, and neither does a turn that died
    before the SDK named anything."""
    settled = [
        user("a"),
        answer("b", "entry-1"),
        user("c"),
        answer("the session failed"),  # no uuid: nothing was ever named
        user("d"),
    ]
    assert rewind.plan(settled, 4, sdk_session_id=SDK)["anchor"] == "entry-1"


def test_a_session_with_no_sdk_id_has_no_anchor_at_all(workspace):
    assert rewind.plan(conversation(), 4, sdk_session_id=None)["anchor"] is None


# ---------------------------------------------------------------------------
# the marker
# ---------------------------------------------------------------------------
def test_the_marker_says_when_the_agent_forgot_too(workspace):
    mark = rewind.record(dropped=conversation()[4:], resumed=True, anchor="entry-2")
    assert mark["role"] == "system", "restore() keeps only roles it knows"
    assert mark["kind"] == rewind.MARK_KIND
    assert "memory" in mark["text"]


def test_the_marker_says_when_only_the_screen_moved(workspace):
    """The failure nobody can see, so it is the one the transcript has to say
    out loud."""
    mark = rewind.record(dropped=conversation()[4:], resumed=False)
    assert "only the transcript moved" in mark["text"]
    assert "still remembers" in mark["text"]


def test_the_marker_carries_the_turns_it_dropped(workspace):
    dropped = conversation()[2:]
    mark = rewind.record(dropped=dropped, resumed=True)
    assert rewind.dropped_of(mark) == dropped
    assert rewind.dropped_of(mark)[1]["blocks"], "tool calls and all"


@pytest.mark.parametrize("junk", [None, "not a list", 12, [1, "two", {}, {"role": "user"}]])
def test_a_marker_from_another_version_cannot_take_the_window_down(workspace, junk):
    """This file outlives the version that wrote it, and the renderer subscripts
    whatever comes back -- the same contract `_drawable_blocks` has."""
    assert all(isinstance(r, dict) for r in rewind.dropped_of({"dropped": junk}))


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------
def session_with(records: list[dict], *, sdk_session_id: str | None = SDK) -> app_mod.Session:
    session = app_mod.Session("client-1")
    session.adopt()
    session.settled = list(records)
    session.sdk_session_id = sdk_session_id
    return session


def test_a_rewind_is_refused_while_a_turn_is_running(workspace):
    """It drops the client, and doing that under a live `receive_response` is
    the failure `_stop_turn` exists to clean up after."""
    session = session_with(conversation())
    session.busy = True
    outcome = asyncio.run(session.rewind_to(4))
    assert outcome["ok"] is False
    assert len(session.settled) == 6, "nothing was touched"


def test_a_rewind_replaces_the_dropped_turns_with_one_marker(workspace):
    session = session_with(conversation())
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["ok"] is True
    assert outcome["prompt"] == "Sorry, API errored, continue"
    assert [r["role"] for r in session.settled] == ["user", "assistant", "user", "assistant", "system"]
    assert session.settled[-1]["kind"] == rewind.MARK_KIND


def test_a_rewind_arms_the_next_client_with_the_anchor(workspace):
    """There is no control request for this: `resume_session_at` is fixed when a
    client is built, so the rewind lands on the rebuild that `ask` does next."""
    session = session_with(conversation())
    asyncio.run(session.rewind_to(4))
    assert session._rewind_at == "entry-2"
    assert session.client is None, "the live conversation was dropped"


def test_an_unanchorable_rewind_still_rewinds_and_says_so(workspace):
    """The transcript is ours and always moves; the conversation is the SDK's and
    only sometimes does. A rewind that reported one number for both would be
    hiding the expensive half."""
    session = session_with(conversation(), sdk_session_id=None)
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["ok"] is True
    assert outcome["resumed"] is False
    assert session._rewind_at is None
    assert "no conversation to put back" in outcome["message"]
    assert len(session.settled) == 5, "the transcript rewound anyway"


def test_a_rewind_against_a_conversation_that_was_renamed_is_named_as_such(workspace):
    """"There is no live conversation" and "there is one and it cannot be cut
    here" have the same symptom and different causes."""
    session = session_with(conversation(), sdk_session_id="some-newer-id")
    outcome = asyncio.run(session.rewind_to(4))
    assert outcome["resumed"] is False
    assert "new id" in outcome["message"]


def test_the_rewind_survives_the_app_being_restarted(workspace):
    """The anchors and the dropped turns are both written, so reopening a
    rewound session is not the moment rewinding quietly stops working."""
    session = session_with(conversation())
    asyncio.run(session.rewind_to(4))
    session_id = session.session_id

    sessions.reset_claims()
    reopened = app_mod.Session("client-2")
    reopened.session_id = session_id
    reopened.restore()

    assert [r["role"] for r in reopened.settled] == [
        "user", "assistant", "user", "assistant", "system",
    ]
    assert reopened.settled[1]["uuid"] == "entry-1"
    assert reopened.settled[1]["sdk_session_id"] == SDK
    marker = reopened.settled[-1]
    assert marker["kind"] == rewind.MARK_KIND
    assert [r["text"] for r in rewind.dropped_of(marker)] == [
        "Sorry, API errored, continue", "the session failed: `APIError`",
    ]
    # And it is still rewindable, from the anchors that came back off disk.
    assert rewind.plan(reopened.settled, 2, sdk_session_id=SDK)["anchor"] == "entry-1"


def test_rewinding_twice_nests_rather_than_losing_the_first_lot(workspace):
    """The point is to take the dead turns out of the conversation, not out of
    the record."""
    session = session_with(conversation())
    asyncio.run(session.rewind_to(4))
    asyncio.run(session.rewind_to(2))

    assert [r["role"] for r in session.settled] == ["user", "assistant", "system"]
    outer = rewind.dropped_of(session.settled[-1])
    assert [r.get("text") for r in outer[:2]] == ["plot it", "plotted"]
    inner = rewind.dropped_of(outer[-1])
    assert [r["text"] for r in inner] == [
        "Sorry, API errored, continue", "the session failed: `APIError`",
    ]


def test_a_rewound_session_is_what_is_on_disk(workspace):
    """`_persist` writes the whole file, so a rewind that only changed memory
    would come back on the next reload."""
    session = session_with(conversation())
    asyncio.run(session.rewind_to(4))

    lines = sessions.path_for(session.session_id).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines[1:]]
    assert len(records) == 5
    assert records[-1]["kind"] == rewind.MARK_KIND
    assert json.loads(lines[0])["messages"] == 5


# ---------------------------------------------------------------------------
# the window between the click and the turn after it
# ---------------------------------------------------------------------------
def test_a_rewind_closed_before_the_next_turn_is_still_armed_when_reopened(workspace):
    """The transcript moves at the click and the conversation only on the next
    client. Closing the app in between used to lose the half that had not
    happened: a short transcript above a session that resumed whole, under a
    marker saying the agent had forgotten."""
    session = session_with(conversation())
    asyncio.run(session.rewind_to(4))
    session_id = session.session_id

    sessions.reset_claims()
    reopened = app_mod.Session("client-2")
    reopened.session_id = session_id
    reopened.restore()

    assert reopened._rewind_at == "entry-2"


def test_a_turn_after_the_rewind_disarms_it(workspace):
    """A trailing marker *is* the state -- a turn that ran would have appended
    below it -- so there is nothing extra to store and nothing to go stale."""
    settled = [
        *conversation()[:4],
        rewind.record(dropped=conversation()[4:], resumed=True, anchor="entry-2"),
        user("try again, more carefully"),
        answer("done", "entry-9"),
    ]
    assert rewind.pending_anchor(settled) is None


def test_a_pending_rewind_does_not_follow_the_client_into_another_session(workspace):
    """An anchor names an entry inside one conversation. Carried across a switch
    it would truncate a different SDK session at a uuid that is not in it."""
    session = session_with(conversation())
    asyncio.run(session.rewind_to(4))
    assert session._rewind_at == "entry-2"

    asyncio.run(session.new_session())
    assert session._rewind_at is None

    sessions.write("elsewhere", [user("a different conversation")], sdk_session_id="sdk-2")
    session._rewind_at = "entry-2"
    session.session_id = "elsewhere"
    session.settled.clear()
    session.restore()
    assert session._rewind_at is None


# ---------------------------------------------------------------------------
# the options a rewind puts on the next client
# ---------------------------------------------------------------------------
def test_no_rewind_means_no_rewind_options(workspace):
    import agent

    assert agent.rewind_option(_sdk_stub(), None, None) == {}
    assert agent.rewind_option(_sdk_stub(), None, "drops-this") == {}


def test_the_anchor_reaches_the_sdk(workspace):
    import agent

    assert agent.rewind_option(_sdk_stub(), "entry-2", None) == {"resume_session_at": "entry-2"}
    assert agent.rewind_option(_sdk_stub(), "entry-2", "prompt-3") == {
        "resume_session_at": "entry-2", "resume_drops_turn": "prompt-3",
    }


def test_an_sdk_without_the_options_gets_a_session_that_ignores_the_rewind(workspace):
    """Feature-detected like `thinking_option`: an older SDK must give a session
    that does not rewind rather than one that cannot be built."""
    import agent

    assert agent.rewind_option(_sdk_stub(fields=("resume",)), "entry-2", "prompt-3") == {}


def _sdk_stub(fields: tuple[str, ...] = ("resume", "resume_session_at", "resume_drops_turn")):
    import dataclasses
    import types

    options = dataclasses.make_dataclass(
        "ClaudeAgentOptions", [(name, str, dataclasses.field(default=None)) for name in fields]
    )
    return types.SimpleNamespace(ClaudeAgentOptions=options)
