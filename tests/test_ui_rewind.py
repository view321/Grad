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

from core import config as config_mod, rewind
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


def test_rewind_support_is_detected_rather_than_assumed(workspace):
    """The other half of the feature detection. `rewind_option` decides what to
    send; this decides what the user is told was going to happen, and the two
    have to agree or the message describes a rewind the SDK is about to ignore."""
    import agent

    assert agent.rewind_supported(_sdk_stub()) is True
    assert agent.rewind_supported(_sdk_stub(fields=("resume",))) is False


def test_an_sdk_that_cannot_resume_is_not_told_the_memory_went_back(workspace, monkeypatch):
    """An anchor is only half the condition. Reporting `resumed` on the strength
    of the anchor alone claimed the agent had forgotten turns it was still being
    charged for -- the one outcome `core/rewind.py` is written to surface."""
    import agent

    monkeypatch.setattr(agent, "rewind_supported", lambda *_: False)
    session = session_with(conversation())
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["ok"] is True
    assert outcome["resumed"] is False
    assert "still remembers" in outcome["message"]
    assert "claude-agent-sdk" in outcome["message"], "the fix is an upgrade; say so"
    # Nothing armed and nothing claimed: a marker carrying an anchor no SDK will
    # honour would survive a restart as a rewind still waiting to be applied.
    assert session._rewind_at is None
    assert session.settled[-1].get("anchor") is None


# ---------------------------------------------------------------------------
# the third half: the files
# ---------------------------------------------------------------------------
class FakeCheckpointClient:
    """A client that records the control requests it was sent, and in what order.

    `closed` is what makes the ordering assertion possible: `rewind_files` is a
    control request and `close` tears down the subprocess that answers it, so
    the only bug worth testing for here is the two in the wrong order.
    """

    def __init__(self, *, fails: bool = False) -> None:
        self.rewound_to: str | None = None
        self.closed = False
        self.rewound_after_close = False
        self._fails = fails

    async def rewind_files(self, user_message_id: str) -> None:
        if self.closed:
            self.rewound_after_close = True
        if self._fails:
            raise RuntimeError("no checkpoint for that message")
        self.rewound_to = user_message_id

    async def __aexit__(self, *_exc) -> None:
        self.closed = True


def checkpointing_session(monkeypatch, *, fails: bool = False, supported: bool = True):
    """A session whose client can checkpoint and whose dropped prompt is known.

    `_drops_turn` reads the SDK's own transcript off disk, which no test has --
    so it is stubbed here. What is under test is what the rewind *does* with the
    uuid, not the parse that finds it, which `_drops_turn` owns.
    """
    import agent

    monkeypatch.setattr(agent, "checkpointing_supported", lambda *_: supported)
    session = session_with(conversation())
    # Mapped rather than constant. A stub that answers "prompt-3" whatever it is
    # asked cannot tell "the earliest dropped prompt" from "some prompt", which
    # is the claim the multi-turn test below is making.
    prompts = {None: "prompt-1", "entry-1": "prompt-2", "entry-2": "prompt-3"}
    monkeypatch.setattr(session, "_drops_turn", lambda anchor: prompts.get(anchor))
    client = FakeCheckpointClient(fails=fails)
    session.client = client
    return session, client


def test_the_files_go_back_to_the_prompt_the_rewind_drops(workspace, monkeypatch):
    """Not to the anchor. The anchor is the *end of the last kept turn* and the
    files should be as they were before the first dropped prompt ran, which is
    the next user message after it."""
    session, client = checkpointing_session(monkeypatch)
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["files"] is True
    assert client.rewound_to == "prompt-3"


def test_the_files_are_rewound_before_the_client_is_closed(workspace, monkeypatch):
    """`rewind_files` is a control request and `rewind_to` closes the client.
    Ordered the other way round this is the one half of a rewind that would
    silently never run -- and it would fail into the same `except` as an SDK
    that cannot checkpoint at all, so nothing would say so."""
    session, client = checkpointing_session(monkeypatch)
    asyncio.run(session.rewind_to(4))

    assert client.closed is True, "the rewind still has to drop the client"
    assert client.rewound_after_close is False


def test_a_multi_turn_rewind_still_restores_the_files(workspace, monkeypatch):
    """`resume_drops_turn` is only sent for a single-turn rewind because the SDK
    validates it. That restriction is about the *conversation*; restoring files
    to the earliest dropped prompt is right for any number of turns."""
    session, client = checkpointing_session(monkeypatch)
    outcome = asyncio.run(session.rewind_to(2))

    assert outcome["files"] is True
    # The prompt at index 2, not the one at index 4: two exchanges go, and the
    # files belong at the point before the *first* of them ran.
    assert client.rewound_to == "prompt-2"
    assert session._rewind_drops is None, "two turns go, so the SDK is not told one does"


def test_rewinding_to_the_very_first_prompt_still_restores_the_files(workspace, monkeypatch):
    """The "start over" rewind, and the one where the work matters most.

    Rewinding to index 0 keeps nothing, so there is no last-entry-of-the-last-
    kept-turn to anchor on -- and `dropped_prompt` was computed only `if anchor`,
    so this case moved the transcript and left every file the session had
    written. A missing anchor means "the first prompt in the conversation", not
    "no prompt at all".
    """
    session, client = checkpointing_session(monkeypatch)
    outcome = session and asyncio.run(session.rewind_to(0))

    assert outcome["ok"] is True
    assert outcome["files"] is True
    assert client.rewound_to == "prompt-1"


def test_an_sdk_that_cannot_checkpoint_rewinds_everything_else(workspace, monkeypatch):
    session, client = checkpointing_session(monkeypatch, supported=False)
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["ok"] is True
    assert outcome["files"] is False
    assert client.rewound_to is None
    assert len(session.settled) == 5, "the transcript still moved"


def test_a_failed_file_rewind_does_not_take_the_rewind_down(workspace, monkeypatch):
    """Every reason this can fail is ordinary -- a prompt with no checkpoint, a
    session already gone -- and none of them is a reason to refuse to rewind the
    transcript."""
    session, client = checkpointing_session(monkeypatch, fails=True)
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["ok"] is True
    assert outcome["files"] is False
    assert session.settled[-1]["kind"] == rewind.MARK_KIND


def test_the_restore_is_only_claimed_when_it_happened(workspace, monkeypatch):
    """Most conversations edit no files, so an absence is the common case and
    reporting it reads as a failure of something never attempted."""
    session, _ = checkpointing_session(monkeypatch, supported=False)
    outcome = asyncio.run(session.rewind_to(4))
    assert "files" not in outcome["message"]

    session, _ = checkpointing_session(monkeypatch)
    outcome = asyncio.run(session.rewind_to(4))
    assert "files it edited are back" in outcome["message"]


def test_the_marker_says_what_the_restore_did_and_did_not_cover(workspace):
    """The agent works mostly through Bash, so "files were restored" on its own
    would be read as a promise the checkpointing does not make."""
    marker = rewind.record(dropped=[user("x")], resumed=True, files=True)
    assert marker["files"] is True
    assert "anything a command wrote was not" in marker["text"]

    quiet = rewind.record(dropped=[user("x")], resumed=True)
    assert quiet["files"] is False
    assert "restored" not in quiet["text"]


def test_checkpointing_support_is_detected_rather_than_assumed(workspace):
    import agent

    assert agent.checkpointing_supported(_sdk_stub(fields=("enable_file_checkpointing",))) is True
    assert agent.checkpointing_supported(_sdk_stub(fields=("resume",))) is False
    assert agent.checkpointing_option(_sdk_stub(fields=("enable_file_checkpointing",))) == {
        "enable_file_checkpointing": True
    }
    assert agent.checkpointing_option(_sdk_stub(fields=("resume",))) == {}


def test_a_turn_that_starts_while_the_client_is_closing_aborts_the_rewind(workspace):
    """`close` tears down a CLI subprocess and the composer is live throughout.
    A prompt sent in that window has already appended to `settled` and started a
    turn, and finishing the rewind would overwrite both -- the prompt gone from
    the screen with its turn still running."""
    session = session_with(conversation())
    original = list(session.settled)

    async def close_and_let_a_turn_in() -> None:
        session.busy = True

    session.close = close_and_let_a_turn_in
    outcome = asyncio.run(session.rewind_to(4))

    assert outcome["ok"] is False
    assert "in flight" in outcome["message"], "say which of the two guards refused"
    assert session.settled == original, "the transcript was not touched"


def test_a_refused_drops_turn_check_reconnects_without_it(workspace):
    """The SDK documents that refusal as deterministic and says to resume plainly
    rather than retry the claim. The transcript has already been rewound by the
    time this runs, so a refusal that propagated would leave a session that
    cannot be reconnected at all, showing turns the agent is not being rebuilt to
    match -- strictly worse than a rewind nobody validated."""
    import agent

    attempts: list[str | None] = []

    class Refuses:
        def __init__(self, options=None):
            self.options = options

        async def __aenter__(self):
            attempts.append(getattr(self.options, "resume_drops_turn", None))
            if len(attempts) == 1:
                raise RuntimeError(f"{app_mod._REWIND_REFUSED} prompt-3")
            return self

    session = session_with(conversation())
    session._rewind_at = "entry-2"
    session._rewind_drops = "prompt-3"
    asyncio.run(session._connect(agent, Refuses, config_mod.load(reload=True)))

    assert len(attempts) == 2, "it retried rather than giving up"
    assert attempts[0] == "prompt-3", "the first attempt carried the check"
    assert attempts[1] is None, "the second dropped it rather than repeating it"
    assert session.client is not None, "the session reconnected"


def test_an_unrelated_connect_failure_is_not_retried(workspace):
    """The retry is scoped to that one refusal. Retrying everything would hide a
    real connection failure behind a second identical one."""
    import agent

    attempts: list[int] = []

    class Broken:
        def __init__(self, options=None):
            pass

        async def __aenter__(self):
            attempts.append(1)
            raise RuntimeError("the CLI is not installed")

    session = session_with(conversation())
    session._rewind_drops = "prompt-3"
    with pytest.raises(RuntimeError):
        asyncio.run(session._connect(agent, Broken, config_mod.load(reload=True)))
    assert len(attempts) == 1
    assert session.client is None, "a half-built client would never reconnect"


# ---------------------------------------------------------------------------
# the draft the rewind leaves behind
# ---------------------------------------------------------------------------
class _Composer:
    """A workspace shrunk to the four things the session-switch path touches."""

    def __init__(self) -> None:
        self.chat_draft = ""
        self.rebuilds = 0
        self.session = self
        self.said: list[str] = []

    def say(self, message: str) -> None:
        self.said.append(message)

    def rebuild_chat(self) -> None:
        self.rebuilds += 1

    async def new_session(self) -> str:
        return "new session"

    async def open_session(self, session_id: str) -> str:
        return f"opened {session_id}"


def test_a_rewound_prompt_does_not_follow_the_user_into_another_session(workspace):
    """`chat_draft` is workspace state because it has to survive the rebuild a
    rewind triggers, and the composer is seeded from it on every draw. Left set
    across a switch, the prompt dropped in one conversation reappears in the
    next one's box, where it reads as something typed there and is one Enter
    away from being asked of the wrong agent."""
    from ui.windows import chat as chat_win

    space = _Composer()
    space.chat_draft = "the prompt session A dropped"
    asyncio.run(chat_win._switch(space, "session-b"))

    assert space.chat_draft == ""
    assert space.rebuilds == 1, "the window still redraws for the new conversation"


def test_a_new_session_starts_with_an_empty_box_too(workspace):
    """Same leak, the other door out of a conversation."""
    from ui.windows import chat as chat_win

    space = _Composer()
    space.chat_draft = "the prompt session A dropped"
    asyncio.run(chat_win._fresh(space))

    assert space.chat_draft == ""
    assert space.rebuilds == 1
