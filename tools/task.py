"""grad-task -- run one of Grad's CLIs in the background and come back to it.

    "The kernel is for exploration. Anything long is a job."

The system prompt has always said that, and until now the agent had no way to act
on it for the things that are long *locally*. `preflight run` is tests (900s cap),
then a dry run (900s cap), then a paid remote smoke with a 900s queue grace, and
it is one `Bash` call that owns the turn for all of it. `paper_search` is six
queries against two endpoints. `kaggle collect --wait` polls for an hour. Every
one of them blocks the conversation, and none of them needs to.

So: `start` returns a task id immediately, `status` and `output` read it, `wait`
joins it, and `stop` asks the tool's own halt verb before it reaches for a
signal. The registry is `core/tasks.py`, which is a file, which is why the
desktop app's tasks window can show the agent's work in the same list as your
own.

**Three refusals that are not incidental.**

*The deny list applies here.* This is a new way to run a command, so it carries
the same speed bump `hooks.py` puts on `Bash` -- `evaluate_bash` is a pure
function precisely so a second caller can use it. Without that, `task start --
ssh box nvidia-smi` would be the cheapest bypass in the system, and it would look
like a feature.

*Tasks cannot start tasks.* A supervisor supervising a supervisor is a process
tree `stop` cannot reason about, for no gain: the thing you wanted in the
background is one level down.

*There is a ceiling.* `[execution] max_concurrent_tasks`, refusing with exit 14.
Sixteen background pytest runs is not parallelism, it is a machine that has
stopped responding -- and the point of this tool is to make starting things cheap,
which is exactly the property that needs a bound.

**What this is not.** It is not the SDK's `Task` tool, which stays denied in
`agent.py`. A subagent is a model call `agent.drive_turn` never issued, and the
one thing this codebase has learned three times over is what an unmetered model
call costs. This runs *CLIs*, which are the slow things, and every one of them
lands in the same ledgers it would have landed in from a terminal.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Any

import hooks
from core import config as config_mod, paths, spawn, tasks as tasklib
from core.cli import Cli, main
from core.errors import (
    EXIT_CONCURRENCY,
    EXIT_RUNNING,
    GateRefusal,
    GradError,
    NotFound,
    UsageError,
)

cli = Cli(
    "grad-task",
    "Run a Grad CLI in the background: start it, read its output, stop it.",
    epilog=(
        "  python -m tools.task start --label preflight --json -- \\\n"
        "      python -m tools.preflight run --spec pipeline/spec.toml --json\n"
        "  python -m tools.task list --json\n"
        "  python -m tools.task output task-093042-1f0a --lines 40 --json\n"
        "  python -m tools.task wait task-093042-1f0a --timeout 600 --json\n\n"
        "Everything after `--` is the command, run as given. The same deny list that\n"
        "applies to Bash applies here: this is a way to run a command, not a way around\n"
        "the rules about which commands.\n\n"
        "--halt gives a task the tool's own graceful stop, which `stop` tries before it\n"
        "signals anything:\n\n"
        "  --halt 'python -m tools.evolve halt --campaign camp-... --json'\n\n"
        "That matters for commands that spawn something of their own: `nb verify` starts\n"
        "its kernel detached so it outlives the CLI, and signalling the parent would leave\n"
        "the kernel holding the VRAM the verify was meant to free."
    ),
)

#: How long the tool's own halt verb is given before the signal is used instead.
HALT_GRACE_S = 20.0
#: Between the terminate and the kill.
KILL_GRACE_S = 5.0
#: How long a supervisor waits for its own start record before giving up. It runs
#: nothing until the record exists, which is what lets `start` check the ceiling
#: *inside* the append lock without risking an orphan when the check refuses.
CLAIM_TIMEOUT_S = 15.0

_SUPERVISE = "_supervise"


def max_concurrent(cfg: config_mod.Config) -> int:
    return max(1, int(cfg.get("execution", "max_concurrent_tasks", 4)))


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
def _start_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label", help="what to call this on screen (defaults to the command)")
    p.add_argument(
        "--halt",
        help="the tool's own graceful stop, as a full command. Tried before any signal.",
    )
    p.add_argument("command", nargs="*", help="the command, after --")


@cli.command("start", "run a command in the background and return its task id", setup=_start_args)
def cmd_start(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    paths.ensure_workspace()
    argv = [str(a) for a in (args.command or [])]
    if not argv:
        raise UsageError(
            "no command given",
            fix="python -m tools.task start --label pre --json -- python -m tools.preflight run --spec <spec> --json",
        )

    _refuse_nesting(argv)
    _refuse_denied(argv)

    halt = shlex.split(args.halt) if args.halt else None
    if halt:
        _refuse_denied(halt, what="the --halt command")

    # Checked here for the message and again inside the append lock for the
    # guarantee -- the same shape `core/submit.py:record_submission` uses, and for
    # the same reason: this reads the registry and the record that makes this task
    # visible lands afterwards, so two starts racing would both see room.
    _check_ceiling(cfg)

    task_id = tasklib.new_id()
    label = args.label or " ".join(argv[:6])
    tasklib.log_dir().mkdir(parents=True, exist_ok=True)
    tasklib.log_path(task_id).write_text("", encoding="utf-8")

    supervisor = subprocess.Popen(
        [sys.executable, "-u", "-m", "tools.task", _SUPERVISE, "--task-id", task_id, "--", *argv],
        cwd=str(paths.root()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Detached, so it outlives this CLI -- which exits in a moment and is the
        # whole point. See `core/spawn.py` for why this is not DETACHED_PROCESS.
        **spawn.detached(),
    )

    try:
        tasklib.record_started(
            task_id,
            label=label,
            argv=argv,
            pid=supervisor.pid,
            halt=halt,
            cwd=str(paths.root()),
            precondition=lambda: _check_ceiling(cfg),
        )
    except GradError:
        # The ceiling closed between the check and the write. The supervisor is
        # waiting for a record that will never arrive and would exit on its own
        # inside CLAIM_TIMEOUT_S; killing it now means the refusal is not followed
        # by fifteen seconds of a process doing nothing.
        _kill_tree(supervisor.pid)
        tasklib.log_path(task_id).unlink(missing_ok=True)
        raise

    return {
        "task": task_id,
        "label": label,
        "command": " ".join(argv),
        "pid": supervisor.pid,
        "log": str(tasklib.log_path(task_id)),
        "state": tasklib.RUNNING,
        "next": f"python -m tools.task status {task_id} --json",
        "note": (
            "started; this command does not wait for it. `wait` joins it, `output` reads "
            "what it has printed, `stop` asks it to stop."
        ),
    }


def _refuse_nesting(argv: list[str]) -> None:
    if any(token in ("tools.task", "grad-task") for token in argv):
        raise UsageError(
            "a task cannot start a task: the supervisor would be supervising a supervisor, "
            "and `stop` would be signalling the wrong process",
            fix="start the inner command directly",
        )


def _refuse_denied(argv: list[str], *, what: str = "this command") -> None:
    """The Bash deny list, applied to a command that is not going through Bash.

    `hooks.evaluate_bash` is a pure function for exactly this reason. Joining the
    argv back into a line is the right input for it: the checks it makes are about
    the command's *head* and about which module and verb appear, and both survive
    the round trip. It is a speed bump here as it is there -- the wall is still
    that the credentials are not in this process's environment.
    """
    denial = hooks.evaluate_bash(shlex.join(argv))
    if denial is None:
        return
    raise UsageError(
        f"{what} is denied for the same reason it is denied through Bash: {denial.reason}",
        fix=denial.suggestion,
    )


def _check_ceiling(cfg: config_mod.Config) -> None:
    limit = max_concurrent(cfg)
    live = tasklib.running()
    if len(live) < limit:
        return
    raise GateRefusal(
        "too_many_tasks",
        f"{len(live)} background task(s) are already running and the ceiling is {limit}: "
        + ", ".join(f"{t['id']} ({t['label']})" for t in live[:4]),
        EXIT_CONCURRENCY,
        fix=(
            f"python -m tools.task wait {live[0]['id']} --json   # or `stop` it, or raise "
            "[execution] max_concurrent_tasks in config/grad.toml"
        ),
        detail={"running": [t["id"] for t in live], "ceiling": limit},
    )


# ---------------------------------------------------------------------------
# the supervisor
# ---------------------------------------------------------------------------
def _supervise_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task-id", required=True)
    p.add_argument("command", nargs="*")


@cli.command(_SUPERVISE, None, setup=_supervise_args)
def cmd_supervise(args: argparse.Namespace) -> dict[str, Any]:
    """Run the command, stream it to the log, record how it ended. Not for humans.

    This exists because `start` returns immediately and something has to be left
    holding the child. An exit code is not a detail in this system -- 4, 7, 12 and
    13 are four different refusals with four different fixes -- and "the process
    is gone" reports none of them.
    """
    task_id = args.task_id
    argv = [str(a) for a in (args.command or [])]
    if not _claim(task_id):
        # `start` never wrote the record: its ceiling check refused inside the
        # lock. Running the command anyway would be a task nothing knows about.
        return {"task": task_id, "ran": False, "reason": "no start record appeared"}

    log = tasklib.log_path(task_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
        with open(log, "a", encoding="utf-8", buffering=1) as fh:
            process = subprocess.Popen(
                argv,
                cwd=str(paths.root()),
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    # The child writes straight into a file handle, and Python
                    # block-buffers when stdout is not a terminal -- so a command
                    # that printed progress for ten minutes would deliver all of
                    # it at exit, and `task output` would show an empty log for
                    # the whole run. `ui/tasks.py` passes `-u` for this; the
                    # command here is arbitrary, so the environment variable is
                    # the version that works whatever it turns out to be.
                    "PYTHONUNBUFFERED": "1",
                },
                **spawn.quiet(),
            )
            code = process.wait()
    except OSError as exc:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n[grad-task] could not start: {exc}\n")
        code = 127
    tasklib.record_exited(task_id, code, envelope=tasklib.last_envelope(task_id))
    return {"task": task_id, "ran": True, "exit_code": code}


def _claim(task_id: str) -> bool:
    """Wait for `start` to record this task, so the ceiling can refuse in-lock.

    `check_liveness=False` because this asks whether a *record* exists, and the
    liveness pass would spend a `tasklist` per poll answering a question about
    other people's processes that nothing here reads.
    """
    deadline = time.monotonic() + CLAIM_TIMEOUT_S
    while True:
        if task_id in tasklib.tasks(check_liveness=False):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _list_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--all", action="store_true", help="include tasks that have finished")


@cli.command("list", "what is running, and what has finished", setup=_list_args)
def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    rows = list(tasklib.tasks().values())
    live = [t for t in rows if t["state"] == tasklib.RUNNING]
    shown = rows if args.all else live
    return {
        "running": len(live),
        "ceiling": max_concurrent(cfg),
        "tasks": [tasklib.summarise(t) for t in shown],
        "note": None if args.all else "finished tasks are hidden; --all shows them",
    }


def _require(task_id: str) -> dict[str, Any]:
    # Stripped, because a task id usually arrives having been copied out of
    # another command's output, and a trailing `\r` off a Windows pipe turns a
    # perfectly good id into "no task 'task-110332-c790\r'" -- which reads as the
    # task having vanished rather than as whitespace.
    task = tasklib.get((task_id or "").strip())
    if task is None:
        raise NotFound(
            f"no task {task_id.strip()!r}",
            fix="python -m tools.task list --all --json",
        )
    return task


@cli.command(
    "status",
    "one task's state and the tail of its output",
    setup=lambda p: (
        p.add_argument("task_id"),
        p.add_argument("--lines", type=int, default=20, help="lines of output to include"),
    ),
)
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    task = _require(args.task_id)
    return {
        **tasklib.summarise(task),
        "output": tasklib.read_log(task["id"], lines=args.lines),
        "envelope": task.get("envelope") or tasklib.last_envelope(task["id"]),
    }


@cli.command(
    "output",
    "what a task has printed",
    setup=lambda p: (
        p.add_argument("task_id"),
        p.add_argument("--lines", type=int, help="only the last N lines"),
        p.add_argument(
            "--bytes",
            dest="tail_bytes",
            type=int,
            default=tasklib.DEFAULT_TAIL_BYTES,
            help="read at most this many bytes from the end (0 for the whole log)",
        ),
    ),
)
def cmd_output(args: argparse.Namespace) -> dict[str, Any]:
    task = _require(args.task_id)
    text = tasklib.read_log(task["id"], tail_bytes=args.tail_bytes, lines=args.lines)
    path = tasklib.log_path(task["id"])
    size = path.stat().st_size if path.is_file() else 0
    return {
        "task": task["id"],
        "state": task["state"],
        "log": str(path),
        "bytes": size,
        "output": text,
        # Said out loud rather than left to be noticed. A tail that silently
        # dropped the first half of a traceback is worse than one that says it did.
        "truncated": bool(args.tail_bytes) and size > args.tail_bytes,
    }


@cli.command(
    "wait",
    "block until a task finishes",
    setup=lambda p: (
        p.add_argument("task_id"),
        p.add_argument("--timeout", type=int, default=600, help="seconds before giving up"),
        p.add_argument("--lines", type=int, default=20),
    ),
)
def cmd_wait(args: argparse.Namespace) -> dict[str, Any]:
    """Join a task. Exit 10 if it is still running when the timeout expires.

    10 rather than an error, matching `collect`: a job that has not finished yet
    is not a failure, and the fix is the same command with a longer timeout.
    """
    task = _require(args.task_id)
    deadline = time.monotonic() + max(0, args.timeout)
    while task["state"] == tasklib.RUNNING and time.monotonic() < deadline:
        time.sleep(1.0)
        task = _require(args.task_id)
    if task["state"] == tasklib.RUNNING:
        raise GradError(
            "still_running",
            f"task {task['id']} ({task['label']}) is still running after {args.timeout}s",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.task wait {task['id']} --timeout {args.timeout * 2} --json",
            detail=tasklib.summarise(task),
        )
    return {
        **tasklib.summarise(task),
        "output": tasklib.read_log(task["id"], lines=args.lines),
        "envelope": task.get("envelope") or tasklib.last_envelope(task["id"]),
    }


# ---------------------------------------------------------------------------
# stopping
# ---------------------------------------------------------------------------
@cli.command(
    "stop",
    "ask a task to stop, then signal it if it does not",
    setup=lambda p: p.add_argument("task_id"),
)
def cmd_stop(args: argparse.Namespace) -> dict[str, Any]:
    """The tool's own stop verb first; the signal only if that fails.

    `ui/tasks.py:cancel` makes the argument and it holds here: a signal reaches
    the CLI and nothing it started, and `nb verify` starts its kernel *detached*
    precisely so it survives the CLI exiting. Killing the parent would leave that
    kernel holding the VRAM the verify existed to free.
    """
    task = _require(args.task_id)
    if task["state"] != tasklib.RUNNING:
        return {
            **tasklib.summarise(task),
            "stopped": False,
            "message": f"task {task['id']} already {task['state']}",
        }

    # Written before anything is signalled, because it is what tells a stop apart
    # from a failure when the exit event lands.
    tasklib.record_stopping(task["id"], note="stop requested")

    if task.get("halt"):
        tasklib.record_note(task["id"], f"asking it to stop: {' '.join(task['halt'])}")
        try:
            subprocess.run(
                task["halt"], cwd=str(paths.root()), capture_output=True,
                text=True, timeout=HALT_GRACE_S, check=False, **spawn.quiet(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            tasklib.record_note(task["id"], f"the tool's own stop failed: {exc}")
        # The *record*, not the state. A halt that works exits the command
        # cleanly, so the supervisor is still there to write the exit event with
        # a real code -- which is the whole reason to prefer the halt verb.
        if _await_record(task["id"], HALT_GRACE_S):
            return {
                **tasklib.summarise(_require(task["id"])),
                "stopped": True,
                "how": "the tool's own stop",
            }
        tasklib.record_note(task["id"], "it did not stop when asked; signalling")

    trouble = _kill_tree(task.get("pid"))
    if trouble:
        # Recorded rather than discarded. A `taskkill` that reports "access
        # denied" and a `taskkill` that worked are the same silence otherwise,
        # and the difference is whether the thing you asked to stop is still
        # running.
        tasklib.record_note(task["id"], trouble)
    gone = _await_pid_gone(task.get("pid"), KILL_GRACE_S)
    if _await_record(task["id"], 0.5):
        # It managed to exit on its own between the signal and now.
        return {**tasklib.summarise(_require(task["id"])), "stopped": True, "how": "signal"}
    if gone:
        # A killed supervisor never gets to write its own exit event, so nothing
        # else ever will -- and a task with no terminal record reads as `lost`,
        # which is the word for one that vanished rather than one that was
        # stopped on purpose. -1 is the code, and `task_stopping` above is what
        # makes the fold call it `stopped` rather than `failed`.
        tasklib.record_exited(task["id"], -1)
    else:
        tasklib.record_note(task["id"], f"pid {task.get('pid')} did not go after the signal")
    return {
        **tasklib.summarise(_require(task["id"])),
        "stopped": gone,
        "how": "signal",
    }


def _await_record(task_id: str, seconds: float) -> bool:
    """Wait for a real `task_exited` event -- never for a merely-absent process."""
    deadline = time.monotonic() + seconds
    while True:
        task = tasklib.get(task_id)
        if task is None or tasklib.finished(task):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def _await_pid_gone(pid: int | None, seconds: float) -> bool:
    if not pid:
        return True
    deadline = time.monotonic() + seconds
    while True:
        if not tasklib.pid_alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def _kill_tree(pid: int | None) -> str | None:
    """Kill the supervisor and everything under it. Returns what went wrong, if
    anything.

    The tree, not the process: the supervisor's whole job is holding a child, and
    killing only the supervisor would leave that child running with nothing left
    to record how it ended.

    "Not found" is not a problem -- it means the process went between the decision
    and the signal, which is the outcome being asked for.
    """
    if not pid:
        return None
    if os.name == "nt":
        out = spawn.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            return None
        message = ((out.stdout or "") + (out.stderr or "")).strip().splitlines()
        first = message[0] if message else f"taskkill exited {out.returncode}"
        return None if "not found" in first.lower() else f"taskkill: {first[:200]}"
    import signal  # noqa: PLC0415

    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        return None
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"could not signal pid {pid}: {exc}"
    return None


@cli.command("clear", "forget finished tasks and delete their logs")
def cmd_clear(_: argparse.Namespace) -> dict[str, Any]:
    done = [t["id"] for t in tasklib.tasks().values() if t["state"] != tasklib.RUNNING]
    return {"forgotten": tasklib.forget(done), "tasks": done}


if __name__ == "__main__":
    main(cli)
