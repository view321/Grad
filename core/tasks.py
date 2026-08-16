"""Background tasks the agent starts and does not wait for.

`ui/tasks.py` gave the *workspace* this in the form of an in-process asyncio
runner: a click starts a command, output streams into a tail, and a stop asks
the tool's own halt verb before it reaches for a signal. The agent had none of
it. Every capability in this system is reached by a `Bash` into `tools/`, and a
`Bash` blocks the turn until the process exits -- so `preflight run` (tests at
900s, dry_run at 900s, then a *paid remote smoke* with a 900s queue grace) is one
tool call that owns the conversation for as long as it takes. The human could run
four things at once from the workspace and the agent could not run two.

This is the shared half. It is deliberately **not** the SDK's `Task` tool: a
subagent is a model call `agent.drive_turn` never issued, and this project has
now met that failure three times -- the desktop app spending untracked tokens,
ShinkaEvolve's `headless/claude` rail, and `Task` itself. What the agent needs is
concurrency over *its own CLIs*, which are the things that are slow.

**A supervisor, not a bare spawn.** `task start` returns immediately, so nothing
would be left to reap the child or record how it ended -- and an exit code is not
a detail here, it is the §8 contract: 4, 7 and 13 are three different refusals
with three different fixes, and "the process is gone" tells you none of them. So
`start` spawns a supervisor which runs the real command, streams both streams to
one log, and appends the exit event. The supervisor is what a stop signals, and
`taskkill /T` is what reaches the command underneath it.

**Records, not memory.** The registry is a JSONL under the app directory,
appended through `core/jsonl.py` like every other multi-writer file here. That is
what makes a task visible to three processes at once: the agent that started it,
a second terminal, and the desktop app's tasks window. `ui/tasks.py` keeps its
own in-process registry for the UI's own buttons -- those it can stream directly
-- and the window shows both.

Under the *workspace's* app-state directory rather than in the workspace itself:
a running process is machine state, not research, and a `tasks.jsonl` committed
beside the ledger would be noise in every diff. Per-workspace rather than
per-installation for the reason `appdata.workspace_state_dir` gives about
transcripts -- two projects open on two folders are two sets of work.
"""

from __future__ import annotations

import datetime as _dt
import os
import secrets
import time
from pathlib import Path
from typing import Any, Iterable

from core import appdata, jsonl, spawn

T_STARTED = "task_started"
T_EXITED = "task_exited"
T_STOPPING = "task_stopping"
T_NOTE = "task_note"
#: `clear` writes one of these rather than rewriting the file. Append-only is
#: cheaper to reason about than a rewrite that a concurrent append can lose, and
#: it is the discipline every other multi-writer file here already follows.
T_FORGOTTEN = "task_forgotten"

RUNNING = "running"
OK = "ok"
FAILED = "failed"
STOPPED = "stopped"
#: Started, never recorded an exit, and its supervisor is gone. The machine was
#: rebooted, or the supervisor was killed by something other than `task stop`.
#: Reported as its own state rather than as `failed`, because there is no exit
#: code and inventing one would put a number in the ledger nobody measured.
LOST = "lost"

TERMINAL = (OK, FAILED, STOPPED, LOST)

#: Bytes of log returned by `output` when no tail is asked for. A training log is
#: megabytes and a tool result is part of a conversation.
DEFAULT_TAIL_BYTES = 60_000


def registry_path() -> Path:
    return appdata.workspace_state_dir() / "tasks.jsonl"


def log_dir() -> Path:
    return appdata.workspace_state_dir() / "tasks"


def log_path(task_id: str) -> Path:
    return log_dir() / f"{task_id}.log"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    """Short, sortable, and unique across processes.

    A counter would be neither: two terminals starting a task at once would mint
    the same id, and the second `task_started` would silently fold over the
    first -- one task's log attached to another task's record.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%H%M%S")
    return f"task-{stamp}-{secrets.token_hex(2)}"


# ---------------------------------------------------------------------------
# liveness
# ---------------------------------------------------------------------------
#: The last full process snapshot, and when it was taken. See `alive_pids`.
_liveness: tuple[float, set[int]] | None = None


def alive_pids(pids: Iterable[int], *, max_age_s: float = 0.0) -> set[int]:
    """Which of these processes are still running, in one call.

    `max_age_s` lets a caller accept a slightly stale answer from the cache. The
    desktop app is why: its tasks window recomputes its model on a two-second
    poll, and a quarter-second `tasklist` on every one of those is a subprocess
    spawned thirty times a minute for a list that rarely changes. A stop that
    takes two seconds to show as stopped is invisible; the process churn is not.
    Defaults to 0 -- always fresh -- so a CLI asking whether something is alive
    right now gets the true answer, and `_await_pid_gone` is not quietly slowed
    by a cache it never asked for.

    **One call, not one per pid, and that is a fix rather than a flourish.**
    `tasklist` costs about a quarter of a second, so a per-pid check made `task
    list` take 1.2 seconds with four tasks running and would have kept getting
    worse -- on the command whose whole purpose is a quick answer to "what is
    still going". Asking once and filtering is O(1) in the number of tasks.

    Windows goes through `tasklist` for the reason `tools/lab.py:_alive` gives,
    and through `spawn.run` so the check does not open a console window: the
    desktop app is a GUI process with no console to lend.
    """
    global _liveness

    wanted = {int(p) for p in pids if p}
    if not wanted:
        return set()
    if max_age_s > 0 and _liveness is not None:
        taken, snapshot = _liveness
        if time.monotonic() - taken <= max_age_s:
            return wanted & snapshot
    if os.name == "nt":
        out = spawn.run(
            ["tasklist", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, check=False,
        )
        # Every pid on the machine, not only the wanted ones: the snapshot is what
        # gets cached, and one filtered to this call's pids would be useless to
        # the next call with a different set -- which is every other call.
        everything: set[int] = set()
        for line in (out.stdout or "").splitlines():
            # "name","pid","session","#","mem" -- the pid is the second field, and
            # splitting on '","' rather than ',' keeps an image name containing a
            # comma from shifting every column after it.
            fields = line.strip().strip('"').split('","')
            if len(fields) < 2:
                continue
            try:
                everything.add(int(fields[1].strip()))
            except ValueError:
                continue
        if everything:
            _liveness = (time.monotonic(), everything)
        return wanted & everything
    # POSIX: `kill(pid, 0)` is cheap enough that there is nothing to cache, and
    # enumerating every process to build a snapshot would cost more than the
    # checks it saved.
    live = set()
    for pid in wanted:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue
        except OSError:
            continue
        live.add(pid)
    return live


def pid_alive(pid: int | None) -> bool:
    """Is this one process still running? Prefer `alive_pids` for several."""
    return bool(pid) and int(pid) in alive_pids([int(pid)])


# ---------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------
def events() -> list[dict[str, Any]]:
    return jsonl.read(registry_path())


def tasks(*, check_liveness: bool = True, liveness_ttl_s: float = 0.0) -> dict[str, dict[str, Any]]:
    """Every task, folded from its events, oldest first.

    `check_liveness` can be turned off entirely by a caller that only wants to
    know whether a *record* exists -- `_claim` is the one that does.
    `liveness_ttl_s` is the softer version, for the UI's two-second poll: see
    `alive_pids`.
    """
    folded: dict[str, dict[str, Any]] = {}
    forgotten: set[str] = set()
    for rec in events():
        tid = rec.get("id")
        if not tid:
            continue
        kind = rec.get("type")
        if kind == T_FORGOTTEN:
            forgotten.add(tid)
            continue
        if kind == T_STARTED:
            forgotten.discard(tid)
            folded[tid] = {
                "id": tid,
                "label": rec.get("label") or tid,
                "argv": list(rec.get("argv") or []),
                "halt": list(rec.get("halt") or []) or None,
                "pid": rec.get("pid"),
                "started_at": rec.get("at"),
                "log": rec.get("log"),
                "cwd": rec.get("cwd"),
                "state": RUNNING,
                "exit_code": None,
                "finished_at": None,
                "stopping": False,
                "notes": [],
            }
        elif tid in folded and kind == T_STOPPING:
            folded[tid]["stopping"] = True
            folded[tid]["notes"].append(rec.get("note") or "stop requested")
        elif tid in folded and kind == T_NOTE:
            folded[tid]["notes"].append(rec.get("note") or "")
        elif tid in folded and kind == T_EXITED:
            node = folded[tid]
            code = rec.get("exit_code")
            node["exit_code"] = code
            node["finished_at"] = rec.get("at")
            node["state"] = (
                STOPPED if node["stopping"] else (OK if code == 0 else FAILED)
            )
            node["envelope"] = rec.get("envelope")
    for tid in forgotten:
        folded.pop(tid, None)
    if check_liveness:
        apparent = [n for n in folded.values() if n["state"] == RUNNING]
        live = alive_pids(
            [n["pid"] for n in apparent if n.get("pid")], max_age_s=liveness_ttl_s
        )
        for node in apparent:
            if node.get("pid") not in live:
                node["state"] = LOST
    return folded


def finished(task: dict[str, Any]) -> bool:
    """Did this task record an exit, as opposed to merely stopping being alive?

    The distinction is what `stop` turns on. A killed supervisor never gets to
    write its exit event, so the fold marks it `lost` -- and code that waits for
    "not running" is satisfied by that, walks away, and leaves a deliberately
    stopped task looking like one that vanished with no exit code. Waiting on the
    *record* is what makes the difference visible.
    """
    return task.get("finished_at") is not None


def get(task_id: str) -> dict[str, Any] | None:
    return tasks().get(task_id)


def running() -> list[dict[str, Any]]:
    return [t for t in tasks().values() if t["state"] == RUNNING]


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def append(record: dict[str, Any], *, precondition: Any = None) -> dict[str, Any]:
    return jsonl.append(registry_path(), {"at": now_iso(), **record}, precondition=precondition)


def record_started(
    task_id: str,
    *,
    label: str,
    argv: list[str],
    pid: int,
    halt: list[str] | None,
    cwd: str,
    precondition: Any = None,
) -> dict[str, Any]:
    return append(
        {
            "type": T_STARTED,
            "id": task_id,
            "label": label,
            "argv": argv,
            "halt": halt,
            "pid": pid,
            "cwd": cwd,
            "log": str(log_path(task_id)),
        },
        precondition=precondition,
    )


def record_exited(task_id: str, exit_code: int, *, envelope: Any = None) -> dict[str, Any]:
    return append(
        {"type": T_EXITED, "id": task_id, "exit_code": exit_code, "envelope": envelope}
    )


def record_stopping(task_id: str, *, note: str) -> dict[str, Any]:
    return append({"type": T_STOPPING, "id": task_id, "note": note})


def record_note(task_id: str, note: str) -> dict[str, Any]:
    return append({"type": T_NOTE, "id": task_id, "note": note})


def forget(task_ids: list[str]) -> int:
    for tid in task_ids:
        append({"type": T_FORGOTTEN, "id": tid})
        log_path(tid).unlink(missing_ok=True)
    return len(task_ids)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def read_log(task_id: str, *, tail_bytes: int = DEFAULT_TAIL_BYTES, lines: int | None = None) -> str:
    """The task's output, bounded from the end.

    Seeked rather than read whole: a task can be a training log, and reading a
    hundred megabytes to show the last forty lines is the kind of thing that only
    hurts on the day it matters.
    """
    path = log_path(task_id)
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if tail_bytes and size > tail_bytes:
            fh.seek(size - tail_bytes)
            fh.readline()  # drop the partial line the seek landed inside
        text = fh.read().decode("utf-8", "replace")
    if lines:
        text = "\n".join(text.splitlines()[-lines:])
    return text


def last_envelope(task_id: str) -> dict[str, Any] | None:
    """The last §8 envelope the task printed, if it printed one.

    Scanned from the end of the tail rather than the whole file: the envelope is
    the last line of a well-behaved CLI, and a task that printed a gigabyte
    before it should not cost a gigabyte to ask about.
    """
    import json  # noqa: PLC0415 - one call site

    for line in reversed(read_log(task_id).splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return None


def summarise(task: dict[str, Any]) -> dict[str, Any]:
    """One task as the CLI reports it: state, timing, and where to read more."""
    envelope = task.get("envelope")
    if envelope is None and task["state"] in TERMINAL:
        envelope = last_envelope(task["id"])
    error = None
    if isinstance(envelope, dict) and envelope.get("ok") is False:
        error = (envelope.get("error") or {}).get("message")
    return {
        "id": task["id"],
        "label": task["label"],
        "state": task["state"],
        "exit_code": task.get("exit_code"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "command": " ".join(task.get("argv") or []),
        "log": task.get("log"),
        "error": error,
        "notes": task.get("notes") or [],
    }
