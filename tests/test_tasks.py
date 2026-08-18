"""The background task runner (`core/tasks.py`, `tools/task.py`).

The registry is a *file*, which is the property everything here rests on: a task
started by the agent has to be visible to a second terminal and to the desktop
app, because otherwise "what is this machine doing" has three answers. So the
fold is tested the way the ledger's is -- events in, state out -- and the CLI is
tested for the refusals rather than for the happy path.

Real processes are used where the point is a real process (start, stop, exit
codes), because a mock of a supervisor proves nothing about whether a supervisor
records an exit. They are short.
"""

from __future__ import annotations

import argparse
import sys
import time

import pytest

from core import tasks as tasklib
from core.errors import EXIT_CONCURRENCY, EXIT_RUNNING, GateRefusal, GradError, NotFound, UsageError
from tools import task as task_cli

TICK = "import sys, time\nfor i in range(int(sys.argv[1])):\n    print('tick', i, flush=True)\n    time.sleep(0.2)\n"


def start(workspace, label, ticks=2, **overrides):
    script = workspace / f"{label}.py"
    script.write_text(TICK, encoding="utf-8")
    args = argparse.Namespace(
        label=label, halt=None, json=True,
        command=[sys.executable, str(script), str(ticks)],
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return task_cli.cmd_start(args)


def wait_for(task_id, seconds=30.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        task = tasklib.get(task_id)
        if task and tasklib.finished(task):
            return task
        time.sleep(0.2)
    return tasklib.get(task_id)


def stop_all():
    for task in tasklib.running():
        try:
            task_cli.cmd_stop(argparse.Namespace(task_id=task["id"], json=True))
        except GradError:
            pass


# ---------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------
def test_a_task_with_no_exit_event_is_running(workspace):
    tasklib.record_started(
        "t1", label="x", argv=["a"], pid=None, halt=None, cwd=str(workspace)
    )
    # No pid, so the liveness pass cannot confirm it -- which is `lost`, not
    # `running`, and that is the honest answer.
    assert tasklib.tasks()["t1"]["state"] == tasklib.LOST
    assert tasklib.tasks(check_liveness=False)["t1"]["state"] == tasklib.RUNNING


def test_an_exit_code_decides_ok_or_failed(workspace):
    for tid, code in (("a", 0), ("b", 3)):
        tasklib.record_started(tid, label=tid, argv=[], pid=1, halt=None, cwd=str(workspace))
        tasklib.record_exited(tid, code)
    folded = tasklib.tasks()
    assert folded["a"]["state"] == tasklib.OK
    assert folded["b"]["state"] == tasklib.FAILED


def test_a_stop_request_makes_the_exit_a_stop_rather_than_a_failure(workspace):
    """A task the user deliberately stopped should not read as one that failed:
    the signal's exit code is the wrong verdict on a command that did nothing
    wrong."""
    tasklib.record_started("t", label="x", argv=[], pid=1, halt=None, cwd=str(workspace))
    tasklib.record_stopping("t", note="stop requested")
    tasklib.record_exited("t", -1)
    assert tasklib.tasks()["t"]["state"] == tasklib.STOPPED


def test_finished_is_about_the_record_not_the_process(workspace):
    """A killed supervisor never writes its exit event, so code waiting for "not
    running" is satisfied by a `lost` task, walks away, and leaves a stopped task
    looking like one that vanished."""
    tasklib.record_started("t", label="x", argv=[], pid=None, halt=None, cwd=str(workspace))
    assert tasklib.finished(tasklib.get("t")) is False
    tasklib.record_exited("t", 0)
    assert tasklib.finished(tasklib.get("t")) is True


def test_clear_is_append_only(workspace):
    tasklib.record_started("t", label="x", argv=[], pid=1, halt=None, cwd=str(workspace))
    tasklib.record_exited("t", 0)
    before = len(tasklib.events())
    assert tasklib.forget(["t"]) == 1
    assert "t" not in tasklib.tasks()
    assert len(tasklib.events()) > before, "clear writes an event; it does not rewrite the file"


def test_the_envelope_is_read_back_off_the_log(workspace):
    """The §8 envelope is the last line a well-behaved CLI prints, and it is what
    turns "exit 4" into "no preflight record, run this"."""
    tasklib.record_started("t", label="x", argv=[], pid=1, halt=None, cwd=str(workspace))
    tasklib.log_dir().mkdir(parents=True, exist_ok=True)
    tasklib.log_path("t").write_text(
        'noise\n{"ok": false, "data": null, "error": {"message": "no preflight"}}\n',
        encoding="utf-8",
    )
    assert tasklib.last_envelope("t")["error"]["message"] == "no preflight"
    tasklib.record_exited("t", 4)
    assert tasklib.summarise(tasklib.get("t"))["error"] == "no preflight"


def test_the_log_tail_is_bounded_from_the_end(workspace):
    tasklib.record_started("t", label="x", argv=[], pid=1, halt=None, cwd=str(workspace))
    tasklib.log_dir().mkdir(parents=True, exist_ok=True)
    tasklib.log_path("t").write_text("\n".join(f"line {i}" for i in range(5000)), encoding="utf-8")
    tail = tasklib.read_log("t", tail_bytes=200)
    assert "line 4999" in tail
    assert "line 0\n" not in tail


# ---------------------------------------------------------------------------
# the CLI's refusals
# ---------------------------------------------------------------------------
def test_the_bash_deny_list_applies_here_too(workspace):
    """This is a new way to run a command, so it carries the same speed bump.
    Without it, `task start -- ssh box nvidia-smi` would be the cheapest bypass
    in the system, and it would look like a feature."""
    with pytest.raises(UsageError) as exc:
        task_cli.cmd_start(
            argparse.Namespace(label="x", halt=None, json=True, command=["ssh", "box", "ls"])
        )
    assert "gpu.py" in exc.value.message
    assert "tools.gpu" in (exc.value.fix or "")
    assert tasklib.tasks() == {}


def test_the_deny_list_applies_to_the_halt_command_too(workspace):
    with pytest.raises(UsageError):
        task_cli.cmd_start(
            argparse.Namespace(
                label="x", halt="ssh box pkill train", json=True, command=["python", "-c", "pass"]
            )
        )


def test_a_task_cannot_start_a_task(workspace):
    with pytest.raises(UsageError) as exc:
        task_cli.cmd_start(
            argparse.Namespace(
                label="x", halt=None, json=True,
                command=["python", "-m", "tools.task", "list"],
            )
        )
    assert "supervisor" in str(exc.value)


def test_an_empty_command_is_a_usage_error(workspace):
    with pytest.raises(UsageError):
        task_cli.cmd_start(argparse.Namespace(label="x", halt=None, json=True, command=[]))


def test_an_unknown_task_is_a_not_found(workspace):
    with pytest.raises(NotFound):
        task_cli.cmd_status(argparse.Namespace(task_id="nope", lines=5, json=True))


def test_a_task_id_is_stripped_before_it_is_looked_up(workspace):
    """Ids are copied out of another command's output, and a trailing `\\r` off a
    Windows pipe turns a good id into "no task 'task-x\\r'"."""
    tasklib.record_started("t", label="x", argv=[], pid=1, halt=None, cwd=str(workspace))
    tasklib.record_exited("t", 0)
    assert task_cli._require("t\r\n")["id"] == "t"


def test_the_ceiling_refuses_with_its_own_exit_code(workspace, monkeypatch, cfg):
    """Sixteen background pytest runs is not parallelism, it is a machine that
    has stopped responding -- and making starting things cheap is exactly the
    property that needs a bound."""
    monkeypatch.setattr(task_cli, "max_concurrent", lambda _cfg: 2)
    for i in range(2):
        tasklib.record_started(
            f"t{i}", label="x", argv=[], pid=1000 + i, halt=None, cwd=str(workspace)
        )
    # `alive_pids`, not `pid_alive`: the fold asks once for all of them, which is
    # the whole reason `task list` is O(1) in the number of tasks.
    monkeypatch.setattr(tasklib, "alive_pids", lambda pids, **kw: set(pids))
    with pytest.raises(GateRefusal) as exc:
        task_cli._check_ceiling(cfg)
    assert exc.value.exit_code == EXIT_CONCURRENCY
    assert "wait" in (exc.value.fix or "")


# ---------------------------------------------------------------------------
# a real process, start to finish
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_a_task_runs_records_its_exit_and_keeps_its_output(workspace):
    """The supervisor exists because `start` returns immediately and something
    has to be left holding the child. An exit code is the §8 contract, and "the
    process is gone" reports none of it."""
    started = start(workspace, "short", ticks=3)
    assert started["state"] == tasklib.RUNNING

    task = wait_for(started["task"])
    assert task["state"] == tasklib.OK
    assert task["exit_code"] == 0
    assert "tick 2" in tasklib.read_log(started["task"])


@pytest.mark.slow
def test_wait_reports_exit_ten_rather_than_failing(workspace):
    """10 rather than an error, matching `collect`: a job that has not finished
    is not a failure, and the fix is the same command with a longer timeout."""
    started = start(workspace, "long", ticks=100)
    try:
        with pytest.raises(GradError) as exc:
            task_cli.cmd_wait(
                argparse.Namespace(task_id=started["task"], timeout=1, lines=5, json=True)
            )
        assert exc.value.exit_code == EXIT_RUNNING
    finally:
        stop_all()


@pytest.mark.slow
def test_stopping_a_task_records_a_terminal_event(workspace):
    """A killed supervisor cannot write its own exit event, and a task with no
    terminal record reads as `lost` -- the word for one that vanished rather than
    one that was stopped on purpose."""
    started = start(workspace, "victim", ticks=100)
    payload = task_cli.cmd_stop(argparse.Namespace(task_id=started["task"], json=True))
    assert payload["stopped"] is True
    task = tasklib.get(started["task"])
    assert task["state"] == tasklib.STOPPED
    assert tasklib.finished(task) is True


@pytest.mark.slow
def test_stopping_a_finished_task_says_so_rather_than_signalling(workspace):
    started = start(workspace, "done", ticks=1)
    wait_for(started["task"])
    payload = task_cli.cmd_stop(argparse.Namespace(task_id=started["task"], json=True))
    assert payload["stopped"] is False
    assert "already" in payload["message"]


# ---------------------------------------------------------------------------
# liveness, and the POSIX state Windows has no word for
# ---------------------------------------------------------------------------
# `kill(pid, 0)` succeeds for a process that has exited and not been reaped, so
# a stop used to wait out its whole grace period watching a corpse, report
# `"stopped": false`, and never write the exit event that keeps the task from
# reading as `lost`. Only ever reproducible off Windows, which is why it stood.
#
# The parser is tested rather than the file read, because there is no `/proc` on
# the machine most of this is written on and a test that skips there is a test
# that never runs.
@pytest.mark.parametrize(
    ("line", "zombie", "why"),
    [
        (b"4242 (python3) Z 1 4242 4242 0 -1 4194560 0 0", True, "a plain zombie"),
        (b"4242 (python3) S 1 4242 4242 0 -1 4194560 0 0", False, "sleeping"),
        (b"4242 (python3) R 1 4242 4242 0 -1 4194560 0 0", False, "running"),
        # `comm` is parenthesised and may contain spaces and brackets of its own,
        # so the state is the field after the *last* ')'. Splitting on whitespace
        # and taking the third token reads the wrong field for any process whose
        # name has a space in it.
        (b"77 (a b) c) Z 1 77 77 0 -1 0 0", True, "a name with a space and a bracket"),
        (b"88 (weird)name) S 1 88 88 0 -1 0 0", False, "a name with an embedded bracket"),
        (b"", False, "an empty read, which is not evidence of anything"),
    ],
)
def test_the_process_state_is_read_from_after_the_last_bracket(line, zombie, why):
    assert tasklib._stat_is_zombie(line) is zombie, why  # noqa: SLF001


def test_a_pid_with_no_proc_entry_is_not_called_a_zombie():
    """The read failing is not evidence of anything -- on Windows it always
    fails, and reporting every live process as a corpse there would fold every
    running task to `exited`."""
    import os

    assert tasklib._zombie(os.getpid()) is False  # noqa: SLF001
