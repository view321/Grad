"""grad-wakeup -- arm a condition, end the turn, and be woken when it happens.

    "a two-hour poll inside the agent's only shell is a tool timeout waiting to
     happen"

`tools/task.py` gave the agent a way to *start* something without waiting for it.
This is the way to *stop* waiting for it. Arm a wake, end the turn, and a
detached watcher does the looking -- out of process, at a backing-off interval,
with no shell held and no tokens spent. When the condition happens it wakes the
session with a turn saying so.

What it replaces is a pattern, not a command: `sleep 30`, look, `sleep 60`, look,
`sleep 120`, look. Every one of those looks is a turn and every turn is tokens,
so a four-hour training run produced a conversation that was mostly the agent
waiting -- and the sleeps had to keep growing or the watching cost more than the
job. None of that is necessary. The machine can tell us.

**The conditions are a closed list, and that is the point.** There is no
`--command`. `hooks.py` gates every `Bash` the agent runs and `tools/task.py`
re-uses `evaluate_bash` so that backgrounding a command cannot become the way
around it; a wakeup that ran an arbitrary command on a timer would be the same
bypass with a delay on it. So a wake waits on things this system already knows
how to read: a background task's record, a run's status on its backend, a path,
a clock. `core/wakeups.py` has the rest of the argument.

**The turn a wake issues is metered like any other.** It goes through
`agent.drive_turn`, inside the token allocation, into `ledger/quota.jsonl`. A
wake is a prompt with no human typing it, not an exception to the ceiling.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from core import paths, wakeups as wk
from core.cli import Cli, main
from core.errors import EXIT_RUNNING, GradError, NotFound, UsageError

cli = Cli(
    "grad-wakeup",
    "Wait for something in the background and start a new turn when it happens.",
    epilog=(
        "The shape is always the same: arm, end your turn, get woken.\n\n"
        "  python -m tools.wakeup arm --run run-2026-08-17-a --timeout 14400 \\\n"
        "      --note 'the 4090 sweep; collect it and judge against exp-7' --json\n\n"
        "Then stop. Do not poll it, do not sleep on it, and do not end the turn with a\n"
        "`wait` unless a human is watching -- `wait` holds the shell, which is the thing\n"
        "this tool exists to stop doing.\n\n"
        "A wake fired into a workspace whose app is closed is kept, not lost: `list`\n"
        "shows it as undelivered and `status` prints the turn it would have sent."
    ),
)

_WATCH = "_watch"


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------
def _arm_args(p: argparse.ArgumentParser) -> None:
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--after", type=float, metavar="SECONDS", help="wake after a delay")
    what.add_argument("--task", metavar="TASK_ID", help="wake when a background task finishes")
    what.add_argument("--run", metavar="RUN_ID", help="wake when a run stops running on its backend")
    what.add_argument("--file", metavar="PATH", help="wake when a path appears")
    p.add_argument(
        "--changed",
        action="store_true",
        help="with --file: wake when it changes, not merely when it exists",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=wk.DEFAULT_TIMEOUT_S,
        help=(
            f"seconds to wait before giving up and waking anyway "
            f"(default {int(wk.DEFAULT_TIMEOUT_S)}, ceiling {int(wk.MAX_TIMEOUT_S)}). "
            "Set it to what you actually expect plus a margin: an expired wake is a "
            "fact worth learning, and the state it reports says so."
        ),
    )
    p.add_argument(
        "--note",
        default="",
        help="what you are waiting for and why; it comes back to you in the waking turn",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="record the wake but do not start a turn with it",
    )


@cli.command("arm", "wait for a condition in the background", setup=_arm_args)
def cmd_arm(args: argparse.Namespace) -> dict[str, Any]:
    paths.ensure_workspace()
    timeout = float(args.timeout)
    if timeout <= 0:
        raise UsageError(
            "a timeout of zero would expire before the first look",
            fix="--timeout 3600",
        )
    if timeout > wk.MAX_TIMEOUT_S:
        raise UsageError(
            f"a wake may be armed for at most {int(wk.MAX_TIMEOUT_S)}s "
            f"({wk.MAX_TIMEOUT_S // 3600} hours), and this asked for {int(timeout)}s",
            fix=(
                f"--timeout {int(wk.MAX_TIMEOUT_S)}   # and re-arm if it is genuinely "
                "still running then"
            ),
        )

    condition = _condition(args)
    deadline = time.time() + timeout
    wake_id = wk.new_id()

    # The watcher is started first and the record written second, which is the
    # opposite of `tools/task.py` and deliberate: the watcher's first act is to
    # read its own record, so it is written before the process can look for it.
    # See the retry below -- the alternative was a race between two processes
    # over a file that exists to describe one of them.
    pid = wk.spawn_watcher(wake_id)
    wk.record_armed(
        wake_id,
        condition=condition,
        deadline=deadline,
        note=str(args.note or ""),
        pid=pid,
        resume=not args.no_resume,
    )

    return {
        "wake": wake_id,
        "waiting_for": wk.describe(condition),
        "timeout_s": int(timeout),
        "expires_at": wk.iso_at(deadline),
        "pid": pid,
        "resume": not args.no_resume,
        "note": str(args.note or ""),
        "next": (
            "end your turn. You will be woken with a new turn when this happens."
            if not args.no_resume
            else f"python -m tools.wakeup status {wake_id} --json"
        ),
    }


def _condition(args: argparse.Namespace) -> dict[str, Any]:
    if args.after is not None:
        seconds = float(args.after)
        if seconds < 0:
            raise UsageError("--after cannot be negative", fix="--after 600")
        return {"kind": wk.KIND_AFTER, "seconds": seconds, "fire_at": time.time() + seconds}

    if args.task:
        from core import tasks as tasklib  # noqa: PLC0415

        task = tasklib.get(str(args.task))
        if task is None:
            raise NotFound(
                f"no background task {args.task} in this workspace",
                fix="python -m tools.task list --json",
            )
        if task.get("state") in tasklib.TERMINAL:
            # Refused rather than armed. A wake on something already finished
            # would fire on its first look and spend a turn telling the agent
            # what a `task status` in this same turn would have told it for free.
            raise GradError(
                "already_finished",
                f"task {args.task} has already finished ({task.get('state')})",
                exit_code=EXIT_RUNNING,
                fix=f"python -m tools.task status {args.task} --json",
            )
        return {"kind": wk.KIND_TASK, "task": str(args.task)}

    if args.run:
        from core import ledger_store as ls  # noqa: PLC0415

        try:
            record = ls.run(str(args.run))
        except GradError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NotFound(
                f"could not read run {args.run}: {type(exc).__name__}",
                fix="python -m tools.ledger list --json",
            ) from exc
        if record.collected:
            raise GradError(
                "already_collected",
                f"run {args.run} was collected at {record.get('collected_at')}",
                exit_code=EXIT_RUNNING,
                fix=f"python -m tools.ledger show {args.run} --json",
            )
        return {"kind": wk.KIND_RUN, "run": str(args.run)}

    path = Path(str(args.file))
    if not path.is_absolute():
        path = paths.root() / path
    condition: dict[str, Any] = {
        "kind": wk.KIND_FILE,
        "path": str(path),
        "changed": bool(args.changed),
    }
    if args.changed:
        # The baseline is taken here, at arm time, so "changed" means "changed
        # since you asked" rather than "changed since the watcher got round to
        # looking" -- which would silently miss a write in between.
        try:
            condition["mtime_ns"] = path.stat().st_mtime_ns
        except OSError:
            condition["mtime_ns"] = None
    elif path.exists():
        raise GradError(
            "already_there",
            f"{path} already exists, so this wake would fire immediately",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.wakeup arm --file {path} --changed --json",
        )
    return condition


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _list_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--all", action="store_true", help="include wakes that have already resolved")


@cli.command("list", "what is being waited for, and what has fired", setup=_list_args)
def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    everything = wk.wakeups()
    rows = [
        _summarise(w)
        for w in everything.values()
        if args.all or w["state"] == wk.ARMED or (w["state"] in (wk.FIRED, wk.EXPIRED) and not w["delivered"])
    ]
    rows.sort(key=lambda r: r.get("armed_at") or "")
    undelivered = [r for r in rows if r["state"] in (wk.FIRED, wk.EXPIRED) and not r["delivered"]]
    return {
        "wakeups": rows,
        "armed": sum(1 for r in rows if r["state"] == wk.ARMED),
        "undelivered": len(undelivered),
        "note": (
            f"{len(undelivered)} wake(s) fired while nothing was listening; "
            "`status <id>` prints what they would have said."
            if undelivered
            else ""
        ),
    }


@cli.command(
    "status",
    "one wake in full, including the turn it sent or would send",
    setup=lambda p: p.add_argument("wake_id"),
)
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    wake = _require(args.wake_id)
    out = _summarise(wake)
    if wake["state"] in (wk.FIRED, wk.EXPIRED):
        out["turn"] = wk.prompt_for(wake, wake.get("detail") or {}, expired=wake["state"] == wk.EXPIRED)
    return out


@cli.command(
    "cancel",
    "stop waiting for one; it will not wake you",
    setup=lambda p: (p.add_argument("wake_id"), p.add_argument("--reason", default="")),
)
def cmd_cancel(args: argparse.Namespace) -> dict[str, Any]:
    wake = _require(args.wake_id)
    if wake["state"] != wk.ARMED:
        return {"wake": wake["id"], "state": wake["state"], "note": "it was not waiting for anything"}
    wk.record_cancelled(wake["id"], reason=str(args.reason or ""))
    # The watcher notices on its next look and exits. Not killed: it is sleeping
    # on a bounded interval and reading its own record is the same check it makes
    # every time round, so there is nothing to reach for a signal about.
    return {
        "wake": wake["id"],
        "state": wk.CANCELLED,
        "note": f"the watcher stops within {int(wk.POLL_MAX_S)}s",
    }


def _wait_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("wake_id")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds to block for")


@cli.command(
    "wait",
    "block until one fires -- for a person at a terminal, not for the agent",
    setup=_wait_args,
)
def cmd_wait(args: argparse.Namespace) -> dict[str, Any]:
    """Deliberately the least useful command here.

    It exists because a person driving this from a terminal reasonably wants to
    block on a wake, and because a test needs a synchronous way to observe one.
    The agent should never reach for it: holding the shell is the thing the rest
    of this module exists to stop, and `arm` says so in its own `next`.
    """
    deadline = time.time() + max(0.0, float(args.timeout))
    while True:
        wake = _require(args.wake_id)
        if wake["state"] != wk.ARMED:
            return _summarise(wake)
        if time.time() >= deadline:
            raise GradError(
                "still_waiting",
                f"{args.wake_id} is still armed after {int(args.timeout)}s",
                exit_code=EXIT_RUNNING,
                fix=f"python -m tools.wakeup status {args.wake_id} --json",
            )
        time.sleep(1.0)


@cli.command("clear", "forget wakes that have already resolved")
def cmd_clear(_: argparse.Namespace) -> dict[str, Any]:
    stale = [w["id"] for w in wk.wakeups().values() if w["state"] in wk.TERMINAL]
    return {"forgotten": wk.forget(stale)}


def _require(wake_id: str) -> dict[str, Any]:
    wake = wk.get(wake_id)
    if wake is None:
        raise NotFound(
            f"no wake {wake_id} in this workspace",
            fix="python -m tools.wakeup list --all --json",
        )
    return wake


def _summarise(wake: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": wake["id"],
        "state": wake["state"],
        "waiting_for": wk.describe(wake.get("condition") or {}),
        "note": wake.get("note") or "",
        "armed_at": wake.get("armed_at"),
        "finished_at": wake.get("finished_at"),
        "delivered": wake.get("delivered"),
        "detail": wake.get("detail"),
    }


# ---------------------------------------------------------------------------
# the watcher
# ---------------------------------------------------------------------------
def _watch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wake-id", required=True)


@cli.command(_WATCH, None, setup=_watch_args)
def cmd_watch(args: argparse.Namespace) -> dict[str, Any]:
    """The detached process. Hidden: spawned by name, never typed.

    It waits for its own record to appear before it starts looking. `arm` spawns
    this and writes the record immediately afterwards, and on a cold filesystem
    the process can be running before the append lands -- a watcher that gave up
    there would leave a wake armed forever with nothing watching it.
    """
    wake_id = str(args.wake_id)
    for _ in range(100):
        if wk.get(wake_id) is not None:
            break
        time.sleep(0.1)
    else:
        return {"wake": wake_id, "state": "unclaimed"}
    return wk.watch(wake_id)


if __name__ == "__main__":
    main(cli)
