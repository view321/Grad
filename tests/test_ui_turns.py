"""Interrupting a turn, and the turn after it (`ui.app.Session`).

The bug this suite exists for, as reported: interrupt the agent, submit another
prompt, and the answer is invisible until you interrupt *again*. It has one
visible symptom and three separate causes, all of which end in the same place --
`Session.busy` is still True, so the composer's guard silently refuses the next
prompt and nothing on screen says why:

  * the SDK refused the interrupt, and every exception was swallowed;
  * the interrupt was accepted and the turn did not end;
  * the interrupt landed *late*, on the turn issued after the one it was aimed
    at, killing it the moment it started.

None of the three needs the SDK to reproduce -- they are all about what this
class does with a client that behaves in a particular way -- so the client here
is a fake that can be told to behave in each of them.
"""

from __future__ import annotations

import asyncio

import pytest

app = pytest.importorskip("ui.app", reason="the ui extra is not installed")

pytestmark = pytest.mark.asyncio


class FakeClient:
    """A `ClaudeSDKClient` shaped just enough to drive one turn.

    `receive_response` yields nothing and waits until `finish` is set, which is
    what an in-flight turn is; `interrupt` sets it, which is what a working
    interrupt does. The variations are what the tests are about.
    """

    def __init__(self, *, refuses: bool = False, ignores: bool = False) -> None:
        self.refuses = refuses
        self.ignores = ignores
        self.finish = asyncio.Event()
        self.prompts: list[str] = []
        self.interrupts = 0
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True
        # Whatever was waiting for a message is not going to get one.
        self.finish.set()
        return False

    async def query(self, prompt: str) -> None:
        self.prompts.append(prompt)
        self.finish.clear()

    async def receive_response(self):
        await self.finish.wait()
        return
        yield  # pragma: no cover - makes this an async generator

    async def interrupt(self) -> None:
        self.interrupts += 1
        if self.refuses:
            raise RuntimeError("not connected")
        if not self.ignores:
            self.finish.set()


@pytest.fixture
def session(monkeypatch):
    """A `Session` whose client is a fake and whose notices are collected."""
    notices: list[str] = []
    made: list[FakeClient] = []
    kind: dict[str, dict] = {"kwargs": {}}

    made_session = app.Session("turns")
    made_session.notify = notices.append

    async def start() -> None:
        if made_session.client is None:
            made_session.client = FakeClient(**kind["kwargs"])
            made.append(made_session.client)

    monkeypatch.setattr(made_session, "start", start)
    made_session.notices = notices  # type: ignore[attr-defined]
    made_session.clients = made  # type: ignore[attr-defined]
    made_session.client_kind = kind  # type: ignore[attr-defined]
    return made_session


async def settled(_record) -> None:
    return None


async def run_turn(session, prompt: str = "hello"):
    """Start a turn and wait until it is genuinely in flight."""
    task = asyncio.create_task(session.ask(prompt, settled))
    for _ in range(200):
        await asyncio.sleep(0)
        if session.client is not None and session.client.prompts:
            return task
    raise AssertionError("the turn never reached the client")


# ---------------------------------------------------------------------------
# the three causes
# ---------------------------------------------------------------------------
async def test_an_interrupt_the_sdk_refuses_is_reported_and_still_ends_the_turn(session, monkeypatch):
    """Every exception used to be swallowed here, which is precisely how
    pressing STOP twice became the way to stop a turn: the first press failed in
    silence and the second one happened to land.

    A refused interrupt is also the case that most needs the escalation: nothing
    asked the turn to stop, so nothing will end it but taking the client down.
    """
    monkeypatch.setattr(app, "INTERRUPT_GRACE_S", 0.05)
    session.client_kind["kwargs"] = {"refuses": True}
    task = await run_turn(session)

    assert session.interrupt() == "interrupting the turn…"
    await asyncio.wait_for(task, timeout=5)

    assert session.busy is False
    assert any("refused the interrupt" in n for n in session.notices)


async def test_a_turn_that_ignores_the_interrupt_is_taken_down_anyway(session, monkeypatch):
    """`ui/tasks.py:cancel`'s escalation, applied to the one control that did
    not have it. A session that stays busy forever refuses every prompt after
    it, and says nothing about either."""
    monkeypatch.setattr(app, "INTERRUPT_GRACE_S", 0.05)
    session.client_kind["kwargs"] = {"ignores": True}
    task = await run_turn(session)
    client = session.client

    session.interrupt()
    await asyncio.wait_for(task, timeout=5)

    assert client.interrupts == 1, "the tool's own stop is asked for first"
    assert client.closed is True
    assert session.busy is False
    assert any("did not stop" in n for n in session.notices)


async def test_the_next_turn_waits_for_a_pending_interrupt(session, monkeypatch):
    """The late interrupt. Fire-and-forget, it outlives the turn it was aimed at
    and lands on the one after it -- which then produces nothing and explains
    nothing, and is the shape of the bug as reported."""
    monkeypatch.setattr(app, "INTERRUPT_GRACE_S", 0.05)
    session.client_kind["kwargs"] = {"ignores": True}
    first = await run_turn(session, "one")

    session.interrupt()
    await asyncio.wait_for(first, timeout=5)

    second = await run_turn(session, "two")
    # The interrupt is finished before the second turn is issued, so it cannot
    # be the thing that ends it.
    assert session._stopping is None or session._stopping.done()  # noqa: SLF001
    assert session.client.interrupts == 0
    session.client.finish.set()
    await asyncio.wait_for(second, timeout=5)
    assert session.client.prompts == ["two"]


# ---------------------------------------------------------------------------
# what the control says
# ---------------------------------------------------------------------------
async def test_interrupting_nothing_says_so_rather_than_pretending(session):
    assert session.interrupt() == "nothing is running"


async def test_a_second_press_does_not_stack_a_second_interrupt(session, monkeypatch):
    monkeypatch.setattr(app, "INTERRUPT_GRACE_S", 0.05)
    session.client_kind["kwargs"] = {"ignores": True}
    task = await run_turn(session)

    session.interrupt()
    assert session.interrupt() == "already stopping — the turn is being taken down"
    await asyncio.wait_for(task, timeout=5)
    assert session.clients[0].interrupts == 1


# ---------------------------------------------------------------------------
# what survives it
# ---------------------------------------------------------------------------
async def test_the_client_is_rebuilt_after_an_interrupt_and_resumes_the_conversation(session):
    """A fresh client cannot hold a message belonging to the turn that was
    stopped -- which is the other way the next turn ended instantly with nothing
    drawn. `resume` is what keeps that from costing the conversation, so the id
    has to have been recorded by then."""
    task = await run_turn(session)
    session.sdk_session_id = "sdk-1"

    session.interrupt()
    await asyncio.wait_for(task, timeout=5)
    # The turn ending and the interrupt finishing are two different events.
    # `interrupt` runs `_stop_turn` as its own task, and that task only reaches
    # `close()` *after* the turn settles -- so the turn's completion is the thing
    # that unblocks the teardown, not evidence it has happened. Waiting for it
    # here is what `ask` does before the next turn (`ui/app.py:450`); without it
    # this read is a race that happened to fall the right way on 3.13 and the
    # wrong way on 3.11.
    if session._stopping is not None:  # noqa: SLF001
        await asyncio.wait_for(session._stopping, timeout=5)  # noqa: SLF001
    assert session.client is None

    second = await run_turn(session, "again")
    assert len(session.clients) == 2, "a new client, not the interrupted one"
    session.client.finish.set()
    await asyncio.wait_for(second, timeout=5)
    assert session.sdk_session_id == "sdk-1"


async def test_the_sdk_session_id_is_recorded_from_a_turn_that_never_finished(session, monkeypatch):
    """`drive_turn`'s return value is not reached on the interrupt path, and
    that id is what the rebuilt client resumes from -- so reading it off the
    return value lost the conversation on exactly the turns that end in a
    rebuild."""
    import agent as agent_mod

    class Message:
        session_id = "sdk-7"

    async def drive_turn(client, prompt, stream, *, on_session_id=None, **_):
        on_session_id(Message.session_id)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(agent_mod, "drive_turn", drive_turn)
    await session.ask("hello", settled)
    assert session.sdk_session_id == "sdk-7"


async def test_a_partly_streamed_turn_keeps_what_it_streamed(session, monkeypatch):
    """An interrupted turn is more legible with its half than without it, and
    the transcript must not claim the prompt went unanswered."""
    import agent as agent_mod

    async def drive_turn(client, prompt, stream, **_):
        stream.feed(_FakeAssistant("half an answer"))
        raise RuntimeError("interrupted")

    monkeypatch.setattr(agent_mod, "drive_turn", drive_turn)
    await session.ask("hello", settled)

    assert session.settled[0] == {"role": "user", "text": "hello"}
    assert "half an answer" in session.settled[1]["text"]
    assert "the session failed" in session.settled[1]["text"]


class _FakeAssistant:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# close racing start
# ---------------------------------------------------------------------------
class _SlowConnectClient:
    """A client whose `__aenter__` takes as long as it is told to.

    The SDK's connect is not safe against a concurrent disconnect -- it comes
    back holding a query built around a transport the disconnect already nulled.
    The bug this models: NiceGUI's disconnect handler fired *during* the first,
    slowest connect (a cold CLI spawn) and closed the client mid-`__aenter__`.
    """

    def __init__(self) -> None:
        self.proceed = asyncio.Event()
        self.entered = asyncio.Event()
        self.exited_after_enter: bool | None = None
        self._enter_done = False

    async def __aenter__(self):
        self.entered.set()
        await self.proceed.wait()
        self._enter_done = True
        return self

    async def __aexit__(self, *_):
        # The property under test: by the time close reaches this client, its
        # connect has finished. Closing a half-connected client is exactly the
        # interleaving that corrupted the SDK's internals.
        self.exited_after_enter = self._enter_done
        return False


async def test_a_close_during_a_slow_connect_waits_for_the_connect(monkeypatch):
    """A disconnect landing mid-spawn must not pull the client apart."""
    import claude_agent_sdk
    import agent as agent_mod

    made: list[_SlowConnectClient] = []

    def make_client(options=None):
        client = _SlowConnectClient()
        made.append(client)
        return client

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", make_client)
    monkeypatch.setattr(agent_mod, "preflight_environment", lambda: {})
    monkeypatch.setattr(agent_mod, "build_options", lambda cfg, **_: object())

    session = app.Session("race")
    starting = asyncio.create_task(session.start())
    for _ in range(200):
        await asyncio.sleep(0)
        if made:
            break
    assert made, "the start task never built a client"
    await asyncio.wait_for(made[0].entered.wait(), timeout=5)

    closing = asyncio.create_task(session.close())
    # Give the close every chance to run early; the lock is what stops it.
    for _ in range(20):
        await asyncio.sleep(0)
    assert made[0].exited_after_enter is None, "close ran inside the connect"

    made[0].proceed.set()
    await asyncio.wait_for(starting, timeout=5)
    await asyncio.wait_for(closing, timeout=5)
    assert made[0].exited_after_enter is True
    assert session.client is None


async def test_a_release_waits_out_a_socket_blip(monkeypatch):
    """`_release_when_gone`: a socket that comes back keeps its session."""
    monkeypatch.setattr(app, "RELEASE_GRACE_S", 0.05)

    class Client:
        has_socket_connection = False

    released: list[bool] = []

    class FakeSession:
        async def release(self) -> None:
            released.append(True)

    client = Client()
    app._release_when_gone(client, FakeSession())
    client.has_socket_connection = True  # the page reconnected inside the grace
    await asyncio.sleep(0.2)
    assert released == [], "a reconnected client lost its session anyway"

    client.has_socket_connection = False
    app._release_when_gone(client, FakeSession())
    await asyncio.sleep(0.2)
    assert released == [True], "a client that stayed gone was never released"
