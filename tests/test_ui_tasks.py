"""The background task registry.

Real subprocesses, not mocks. The whole point of this module is what happens to
a process -- that it is not killed by a wall clock, that stopping it asks the
tool first, that its output arrives while it runs -- and none of those are
properties of a fake.

The commands under test are throwaway scripts written into the temp workspace.
`tasks.start` runs `python -m <argv>` with `cwd=paths.root()`, and `-m` puts the
working directory on `sys.path`, so a file dropped in the workspace root is
importable by name.
"""

from __future__ import annotations

import asyncio
import textwrap
import time

import pytest

from ui import tasks as tasks_mod

SETTLE_TIMEOUT_S = 60.0


def script(workspace, name: str, body: str) -> str:
    (workspace / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return name


async def settled(task, timeout: float = SETTLE_TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while task.running and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert not task.running, f"{task.label} did not settle in {timeout}s"
    return task


def lines(task) -> list[str]:
    return [line for _, line in task.tail]


# ---------------------------------------------------------------------------
# the tail, without a process
# ---------------------------------------------------------------------------
def test_the_envelope_is_the_last_json_object_on_stdout():
    """The §8 envelope, tracked as the lines arrive. A command that printed a
    gigabyte still costs one dict, and the *last* object wins because that is
    what `run_tool` means by the envelope."""
    task = tasks_mod.Task("t", "x", ())
    task.append('{"ok": false, "error": {"message": "early"}}')
    task.append("some progress")
    task.append('{"ok": true, "data": {"message": "done"}}')
    assert task.envelope == {"ok": True, "data": {"message": "done"}}


def test_stderr_does_not_become_the_envelope():
    task = tasks_mod.Task("t", "x", ())
    task.append('{"ok": true}', tag="err")
    assert task.envelope is None


def test_a_tail_that_overflows_says_how_much_it_dropped():
    """A tail that silently forgets reads as complete output."""
    task = tasks_mod.Task("t", "x", ())
    for index in range(tasks_mod.TAIL_LINES + 25):
        task.append(f"line {index}")
    assert len(task.tail) == tasks_mod.TAIL_LINES
    assert task.dropped == 25
    assert lines(task)[0] == "line 25"


def test_one_enormous_line_is_cut_rather_than_kept():
    task = tasks_mod.Task("t", "x", ())
    task.append("x" * (tasks_mod.MAX_LINE_CHARS * 3))
    only = lines(task)[0]
    assert len(only) < tasks_mod.MAX_LINE_CHARS * 2
    assert "characters" in only


# ---------------------------------------------------------------------------
# running one
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_task_streams_its_output_and_settles_ok(workspace):
    name = script(workspace, "grad_ok", """
        import json, sys
        print("working")
        sys.stdout.flush()
        print(json.dumps({"ok": True, "data": {"message": "finished"}}))
    """)
    task = tasks_mod.start("a task", name)
    assert tasks_mod.get(task.id) is task, "registered before the click is over"

    await settled(task)
    assert task.state == tasks_mod.OK
    assert task.exit_code == 0
    assert "working" in lines(task)
    assert tasks_mod.task_message(task) == "a task: finished"


@pytest.mark.asyncio
async def test_a_failing_task_keeps_its_exit_code_and_its_stderr(workspace):
    name = script(workspace, "grad_bad", """
        import sys
        print("about to fail")
        print("the reason", file=sys.stderr)
        sys.exit(9)
    """)
    task = await settled(tasks_mod.start("a failure", name))
    assert task.state == tasks_mod.FAILED
    assert task.exit_code == 9
    assert ("err", "the reason") in list(task.tail)


@pytest.mark.asyncio
async def test_a_command_that_cannot_start_fails_rather_than_hanging(workspace):
    task = await settled(tasks_mod.start("nonsense", "grad_no_such_module_at_all"))
    assert task.state == tasks_mod.FAILED


@pytest.mark.asyncio
async def test_a_line_longer_than_the_stream_readers_limit_survives(workspace):
    """`StreamReader.readline` raises past 64 KiB and a training log's progress
    line can pass it, which is why `_pump` reads chunks."""
    name = script(workspace, "grad_long", """
        print("y" * 200_000)
        print("after")
    """)
    task = await settled(tasks_mod.start("a long line", name))
    assert task.state == tasks_mod.OK
    assert "after" in lines(task)


@pytest.mark.asyncio
async def test_the_completion_callback_runs_once_and_cannot_strand_the_task(workspace):
    calls: list[str] = []
    name = script(workspace, "grad_quiet", "print('hi')")

    def boom(task):
        calls.append(task.state)
        raise RuntimeError("the workspace blew up recording this")

    task = await settled(tasks_mod.start("a task", name, on_done=boom))
    assert calls == [tasks_mod.OK]
    assert task.state == tasks_mod.OK, "a callback that raised must not unsettle the task"
    assert any("could not record" in line for line in lines(task))


# ---------------------------------------------------------------------------
# stopping one
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stopping_a_task_with_no_halt_verb_signals_it(workspace):
    name = script(workspace, "grad_forever", """
        import time
        print("started", flush=True)
        time.sleep(600)
    """)
    task = tasks_mod.start("a long one", name)
    for _ in range(400):                       # let it get as far as printing
        if "started" in lines(task):
            break
        await asyncio.sleep(0.02)

    message = await tasks_mod.cancel(task.id)
    assert "stopped" in message or "killed" in message
    await settled(task, timeout=30.0)
    assert task.state == tasks_mod.CANCELLED, "a stop is not a failure"


@pytest.mark.asyncio
async def test_stopping_asks_the_tool_before_it_signals(workspace):
    """The whole reason `halt` exists. `nb verify` starts its kernel *detached*
    so it outlives the CLI -- so signalling the CLI would leave the kernel
    holding the VRAM the verify was meant to free."""
    flag = workspace / "please-stop"
    name = script(workspace, "grad_pollster", f"""
        import pathlib, time
        stop = pathlib.Path(r"{flag}")
        print("started", flush=True)
        for _ in range(3000):
            if stop.exists():
                print("asked to stop, exiting cleanly", flush=True)
                raise SystemExit(0)
            time.sleep(0.02)
    """)
    halt = script(workspace, "grad_halt", f"""
        import json, pathlib
        pathlib.Path(r"{flag}").write_text("stop")
        print(json.dumps({{"ok": True, "data": {{"message": "asked"}}}}))
    """)

    task = tasks_mod.start("a pollster", name, halt=(halt,))
    for _ in range(400):
        if "started" in lines(task):
            break
        await asyncio.sleep(0.02)

    await tasks_mod.cancel(task.id)
    await settled(task, timeout=30.0)
    assert task.state == tasks_mod.CANCELLED
    assert flag.exists(), "the tool's own stop verb was never run"
    assert any("exiting cleanly" in line for line in lines(task)), (
        "it was signalled rather than asked"
    )
    assert not any("killed" in line for line in lines(task))


@pytest.mark.asyncio
async def test_stopping_something_already_finished_says_so(workspace):
    task = await settled(tasks_mod.start("quick", script(workspace, "grad_quick", "pass")))
    assert "already finished" in await tasks_mod.cancel(task.id)
    assert task.state == tasks_mod.OK, "a late cancel must not rewrite the verdict"


@pytest.mark.asyncio
async def test_cancelling_an_unknown_task_is_a_message_not_a_crash():
    assert "no task" in await tasks_mod.cancel("task-404")


@pytest.mark.asyncio
async def test_stopping_a_task_before_its_process_exists_still_stops_it(workspace):
    """`start` registers the task and spawns on the next tick, so a fast click
    finds `_process` still unset. Settling the task there and walking away would
    leave the process to run to completion with nothing on screen saying so."""
    marker = workspace / "it-ran-to-completion"
    name = script(workspace, "grad_marker", f"""
        import pathlib, time
        time.sleep(3)
        pathlib.Path(r"{marker}").write_text("finished")
    """)
    task = tasks_mod.start("a fast cancel", name)
    assert task._process is None, "the spawn should not have happened yet"  # noqa: SLF001

    await tasks_mod.cancel(task.id)
    assert task.state == tasks_mod.CANCELLED
    await tasks_mod.drained(task)
    assert not marker.exists(), "the process outlived the task that reported it stopped"


@pytest.mark.asyncio
async def test_a_running_tasks_driver_is_held_against_collection(workspace):
    """asyncio keeps only a weak reference to a running task, so a bare
    `create_task` can vanish mid-flight -- leaving a live process with no pump
    and nothing left to settle it."""
    task = tasks_mod.start("held", script(workspace, "grad_held", "print('hi')"))
    assert task._driver is not None  # noqa: SLF001
    await settled(task)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_finished_tasks_are_clearable_and_running_ones_are_not(workspace):
    done = await settled(tasks_mod.start("done", script(workspace, "grad_done", "pass")))
    running = tasks_mod.start(
        "running",
        script(workspace, "grad_slow", "import time; time.sleep(600)"),
    )
    try:
        assert tasks_mod.clear_finished() == 1
        assert tasks_mod.get(done.id) is None
        assert tasks_mod.get(running.id) is running
    finally:
        await tasks_mod.cancel(running.id)
        await tasks_mod.drained(running)


def test_the_newest_task_is_listed_first():
    for index in range(3):
        tasks_mod._register(tasks_mod.Task(f"task-{index}", f"t{index}", ()))  # noqa: SLF001
    assert [t.id for t in tasks_mod.all_tasks()] == ["task-2", "task-1", "task-0"]


def test_the_registry_keeps_a_bounded_history_of_finished_tasks():
    """Enough to look back over a session, not so many that the window becomes
    a history of the machine."""
    for index in range(tasks_mod.KEEP_FINISHED + 12):
        task = tasks_mod.Task(f"task-{index}", "t", ())
        task.state = tasks_mod.OK
        tasks_mod._register(task)  # noqa: SLF001
    assert len(tasks_mod.all_tasks()) == tasks_mod.KEEP_FINISHED


def test_a_running_task_is_never_evicted_by_the_history_bound():
    survivor = tasks_mod._register(tasks_mod.Task("task-keep", "long", ()))  # noqa: SLF001
    for index in range(tasks_mod.KEEP_FINISHED + 12):
        task = tasks_mod.Task(f"task-{index}", "t", ())
        task.state = tasks_mod.OK
        tasks_mod._register(task)  # noqa: SLF001
    assert tasks_mod.get("task-keep") is survivor


# ---------------------------------------------------------------------------
# run_tool, the other half of the same decision
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_tool_returns_the_envelope(workspace):
    name = script(workspace, "grad_env", """
        import json
        print("noise on the way")
        print(json.dumps({"ok": True, "data": {"message": "hello"}}))
    """)
    payload = await tasks_mod.run_tool(name)
    assert payload == {"ok": True, "data": {"message": "hello"}}
    assert tasks_mod.envelope_message(payload) == "hello"


@pytest.mark.asyncio
async def test_run_tool_says_where_a_long_command_belongs_when_it_times_out(workspace):
    """The timeout here is enforced by killing the process, which is exactly why
    the long commands do not go through this function -- `nb verify` allows 1800
    seconds *per cell* and was being killed at 900."""
    name = script(workspace, "grad_sleepy", "import time; time.sleep(600)")
    payload = await tasks_mod.run_tool(name, timeout=1.0)
    assert payload["ok"] is False
    assert "timed out" in payload["error"]["message"]
    assert "background" in payload["error"]["fix"]


@pytest.mark.asyncio
async def test_run_tool_reports_a_command_that_printed_no_envelope(workspace):
    name = script(workspace, "grad_silent", """
        import sys
        print("something went wrong", file=sys.stderr)
        sys.exit(2)
    """)
    payload = await tasks_mod.run_tool(name)
    assert payload["ok"] is False
    assert "something went wrong" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# the agent's own calls
# ---------------------------------------------------------------------------
class FakeSession:
    """A `ui.app.Session` as far as the tasks model is concerned: the turn in
    flight, and the turns that settled."""

    def __init__(self, blocks=None, settled=None) -> None:
        self.blocks = blocks or []
        self.settled = settled or []


def call(cid, name="Bash", title="ls", status="running", result="", started=None):
    import time as _t

    block = {
        "kind": "tool", "id": cid, "name": name, "title": title,
        "status": status, "result": result,
    }
    block["started"] = _t.time() if started is None else started
    return block


def test_the_turn_in_flight_is_what_the_tasks_window_shows_first():
    """Every capability in this project is reached by a Bash into `tools/`, so
    the agent's calls are the other half of "what is running on this machine".
    Until this they were visible only in the transcript, which is the wrong
    place to look once the conversation has scrolled on."""
    from ui import models

    session = FakeSession(
        blocks=[{"kind": "text", "text": "checking"}, call("tu_2", title="pytest -q")],
        settled=[{"role": "assistant", "blocks": [call("tu_1", status="ok", result="one\ntwo")]}],
    )
    rows = models.agent_calls_model(session)
    assert [r["id"] for r in rows] == ["tu_2", "tu_1"]
    assert rows[0]["state"] == "running"
    assert rows[1]["state"] == "ok"


def test_a_call_left_running_by_a_settled_turn_is_not_reported_as_running():
    """The turn died or was interrupted mid-call. Whatever it started is not
    this app's to know about, and saying "running" of something nothing is
    waiting for is the same lie as a tail that silently forgets."""
    from ui import models

    session = FakeSession(settled=[{"role": "assistant", "blocks": [call("tu_1")]}])
    row = models.agent_calls_model(session)[0]
    assert row["state"] == "unfinished"
    assert row["running"] is False
    assert row["elapsed"] == ""


def test_only_a_live_call_is_given_a_clock():
    from ui import models

    session = FakeSession(blocks=[call("tu_1", started=None)])
    assert models.agent_calls_model(session)[0]["elapsed"] != ""


def test_a_call_from_a_transcript_written_before_calls_were_stamped_still_lists():
    """The session file outlives the version that wrote it. The row is worth
    showing; the clock is the part that is not known."""
    from ui import models

    block = call("tu_1")
    del block["started"]
    assert models.agent_calls_model(FakeSession(blocks=[block]))[0]["elapsed"] == ""


def test_the_call_list_is_bounded_so_the_window_is_not_a_second_transcript():
    from ui import models

    settled = [
        {"role": "assistant", "blocks": [call(f"tu_{i}", status="ok")]}
        for i in range(models.AGENT_CALLS * 2)
    ]
    assert len(models.agent_calls_model(FakeSession(settled=settled))) == models.AGENT_CALLS


def test_the_two_lists_are_counted_apart():
    """A task is a process this app started and can stop; a call is one the
    agent made and only the agent can stop. Merging them would imply a STOP
    button that does not exist."""
    from ui import models, tasks

    tasks.start("a wiki rebuild", "tools.wiki", "map")
    model = models.tasks_model(agent=models.agent_calls_model(FakeSession(blocks=[call("tu_1")])))
    assert model["running"] == 1
    assert model["agent_running"] == 1
    assert [r["id"] for r in model["rows"]] != [c["id"] for c in model["agent"]]


def test_the_tasks_model_without_a_session_is_unchanged():
    """`tasks_model` is called from the poll and from tests with no session at
    all; the agent half is additive."""
    from ui import models

    model = models.tasks_model()
    assert model["agent"] == []
    assert model["agent_running"] == 0
