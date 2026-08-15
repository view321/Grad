"""What a compaction does to a session (`ui.app.Session`).

Compaction discards the conversation and keeps a note. That makes the note the
only copy of everything it threw away, and the lifecycle of that one string is
where all the interesting failures are:

  * it must reach the model, but not as something the user appears to have said;
  * it must survive a turn that fails, because the turn immediately after a
    compaction is exactly the one where a failure costs the whole session;
  * the resume id must go with it, or `start()` restores the very conversation
    the compaction just paid to summarise.

None of this needs the SDK. It is about what this class does with a client, so
the client is the same fake the interrupt suite uses.
"""

from __future__ import annotations

import asyncio

import pytest

app = pytest.importorskip("ui.app", reason="the ui extra is not installed")

from tests.test_ui_turns import FakeClient, settled  # noqa: E402

pytestmark = pytest.mark.asyncio

NOTE = "I had just submitted run-3 and was waiting to collect it."


@pytest.fixture
def session(monkeypatch):
    made_session = app.Session("compaction")
    made_session.notify = lambda _msg: None

    async def start() -> None:
        if made_session.client is None:
            made_session.client = FakeClient()

    monkeypatch.setattr(made_session, "start", start)
    return made_session


async def _run(session, prompt: str = "carry on"):
    """One turn, driven to completion against the fake client."""
    task = asyncio.create_task(session.ask(prompt, settled))
    for _ in range(200):
        await asyncio.sleep(0)
        if session.client is not None and session.client.prompts:
            session.client.finish.set()
            break
    await task
    return task


async def test_the_handover_reaches_the_model_but_not_the_transcript(session):
    """It goes in front of the prompt, and the transcript records what the user
    actually typed. Putting it in `settled` would show the note as though they
    had written it -- and, at its length, would be the thing you scroll past
    forever afterwards."""
    session.pending_seed = NOTE
    await _run(session, "what were we doing?")

    sent = session.client.prompts[0]
    assert NOTE in sent
    assert sent.endswith("what were we doing?")
    assert [r["text"] for r in session.settled if r["role"] == "user"] == ["what were we doing?"]


async def test_the_handover_is_sent_once(session):
    session.pending_seed = NOTE
    await _run(session, "first")
    await _run(session, "second")
    assert NOTE in session.client.prompts[0]
    assert NOTE not in session.client.prompts[1]
    assert session.pending_seed is None


async def test_a_failed_turn_gives_the_handover_back(session, monkeypatch):
    """The turn right after a compaction is the one where losing this costs the
    session its entire memory. Re-sending it is at worst redundant context; the
    model may not have read it at all."""
    import agent

    async def explode(*_a, **_k):
        raise RuntimeError("the transport died")

    monkeypatch.setattr(agent, "drive_turn", explode)
    session.pending_seed = NOTE
    await _run(session, "carry on")
    assert session.pending_seed == NOTE


async def test_a_budget_refusal_gives_the_handover_back(session, monkeypatch):
    """A refused turn never reached the model at all, so the note is certainly
    still owed."""
    import agent

    async def refuse(*_a, **_k):
        raise agent.BudgetRefused(
            {"message": "out of allocation", "fix": "python -m tools.budget raise"}
        )

    monkeypatch.setattr(agent, "drive_turn", refuse)
    session.pending_seed = NOTE
    await _run(session, "carry on")
    assert session.pending_seed == NOTE


async def test_compacting_drops_the_resume_id(session, monkeypatch):
    """Otherwise `start()` resumes the conversation the compaction just
    summarised, quietly restoring the context it paid to discard -- a cost with
    no effect, and one nothing on screen would explain."""
    from core import compaction

    async def handoff(*_a, **_k):
        return {"note": NOTE, "quota": None, "sdk_session_id": "old"}

    monkeypatch.setattr(compaction, "write_handoff", handoff)
    await session.start()
    session.sdk_session_id = "sdk-abc"
    session.context = {"totalTokens": 400_000}

    outcome = await session.compact()

    assert outcome["ok"] is True
    assert session.sdk_session_id is None
    assert session.client is None
    assert NOTE in session.pending_seed


async def test_a_compaction_that_cannot_summarise_discards_nothing(session, monkeypatch):
    """An oversized conversation is a cost. A conversation replaced by a failed
    summary is a loss."""
    from core import compaction

    async def explode(*_a, **_k):
        raise RuntimeError("the model refused")

    monkeypatch.setattr(compaction, "write_handoff", explode)
    await session.start()
    session.sdk_session_id = "sdk-abc"
    client = session.client

    outcome = await session.compact()

    assert outcome["ok"] is False
    assert session.sdk_session_id == "sdk-abc"
    assert session.client is client
    assert session.pending_seed is None


async def test_the_marker_lands_in_the_transcript_and_survives_a_reload(session, monkeypatch):
    """`restore` keeps records whose role it knows and drops the rest, so a
    marker it filtered out would leave a transcript reading as one continuous
    conversation beside a model that remembers only the tail."""
    from core import compaction

    async def handoff(*_a, **_k):
        return {"note": NOTE, "quota": None, "sdk_session_id": None}

    monkeypatch.setattr(compaction, "write_handoff", handoff)
    await session.start()
    session.context = {"totalTokens": 310_000}
    await session.compact()

    reopened = app.Session("compaction-reader")
    reopened.session_id = session.session_id
    reopened.restore()
    markers = [r for r in reopened.settled if r.get("kind") == "compaction"]
    assert len(markers) == 1
    assert markers[0]["note"] == NOTE


async def test_the_threshold_is_what_decides_and_a_dropped_client_clears_the_reading(session, monkeypatch):
    """A stale high reading kept across a client swap is exactly the reading that
    would trigger a needless compaction on a conversation that is now empty."""
    from core import compaction

    calls: list[str] = []

    async def handoff(*_a, **_k):
        calls.append("compacted")
        return {"note": NOTE, "quota": None, "sdk_session_id": None}

    monkeypatch.setattr(compaction, "write_handoff", handoff)
    monkeypatch.setattr(compaction, "threshold", lambda _cfg=None: 300_000)
    await session.start()

    session.context = {"totalTokens": 100}
    assert await session.maybe_compact() is None
    assert calls == []

    await session.close()
    assert session.context is None
