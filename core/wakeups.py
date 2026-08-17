"""Conditions the agent asks to be woken on, and the record of them.

    "`collect` is non-blocking by default: a two-hour poll inside the agent's
     only shell is a tool timeout waiting to happen."

That line has been in `tools/jobs.py` since remote jobs existed, and it names a
problem it does not solve. The agent has exactly one shell per turn, so waiting
for anything means either holding that shell -- which is a tool timeout, and a
turn that can do nothing else while it waits -- or coming back to look, over and
over, with a `sleep` between the looks. The second is what actually happened, and
it is worse than it sounds: every look is a turn, every turn is tokens, and the
sleeps have to grow or the polling costs more than the job. A four-hour training
run answered by `sleep 30`, `sleep 60`, `sleep 120` is a conversation whose
content is mostly the agent waiting.

`core/tasks.py` fixed the half of this that is *starting* things without waiting.
This is the other half: **being told**. The agent arms a condition and ends its
turn. A detached watcher polls out of process -- no shell held, no tokens, no
model in the loop -- and when the condition is met it wakes the session with a
turn describing what happened.

**A wake is a model call, so it is metered like one.** The turn a wake issues
goes through `agent.drive_turn` exactly as a typed prompt does, which is what
keeps it inside the token allocation and inside `ledger/quota.jsonl`. This is the
same rule that killed ShinkaEvolve's `headless/claude` rail and keeps `Task`
denied: the thing this project has learned three times is what an unmetered model
call costs. A wake is not an exception to the ceiling; it is a prompt with no
human typing it.

**What the watcher may wait on is a closed list.** There is no `--command`, and
its absence is the security model rather than an omission. `hooks.py` gates every
`Bash` the agent runs, and `tools/task.py` re-uses `evaluate_bash` precisely so
that starting a background command cannot become the cheap way around it. A
wakeup that ran an arbitrary shell command on a timer would be that bypass, with
a delay on it -- so the conditions here are things this system already knows how
to read: a task's record, a run's status, a path, a clock.

The registry is a JSONL under the workspace's app-state directory, for the
reasons `core/tasks.py` gives: three processes can hold an opinion about a wake
at once -- the agent that armed it, the watcher, and the desktop app -- and an
append-only event log is the only shape that survives that.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from core import appdata, jsonl

log = logging.getLogger("grad.wakeups")

T_ARMED = "wake_armed"
T_FIRED = "wake_fired"
T_CANCELLED = "wake_cancelled"
T_DELIVERED = "wake_delivered"
T_FORGOTTEN = "wake_forgotten"

ARMED = "armed"
#: The condition happened.
FIRED = "fired"
#: The condition did not happen inside the agent's own timeout. Its own state,
#: not a kind of `fired`: "the job finished" and "the job has not finished in
#: four hours" are different facts and lead to different next actions.
EXPIRED = "expired"
CANCELLED = "cancelled"
#: Armed, never resolved, and its watcher is gone -- a reboot, or a kill. The
#: same honesty `core/tasks.py:LOST` is after: nothing was observed, so nothing
#: is claimed.
LOST = "lost"

TERMINAL = (FIRED, EXPIRED, CANCELLED, LOST)

#: The conditions a wake can carry. Adding one means teaching `check` to read
#: something this system already records -- not teaching it to run something.
KIND_AFTER = "after"
KIND_TASK = "task"
KIND_RUN = "run"
KIND_FILE = "file"
KINDS = (KIND_AFTER, KIND_TASK, KIND_RUN, KIND_FILE)

#: The longest a wake may be armed for, whatever it asks. A watcher is cheap but
#: it is not free, and a condition nobody will ever meet should become an
#: `expired` record rather than a process that outlives the research.
MAX_TIMEOUT_S = 24 * 60 * 60
#: What a wake waits if the agent names no timeout of its own.
DEFAULT_TIMEOUT_S = 4 * 60 * 60

#: How often the watcher looks, to begin with, and the ceiling it backs off to.
#: The point of the backoff is the same as the point of the whole module: the
#: first minute of a four-hour job is worth watching closely and the third hour
#: is not, and a fixed interval has to choose between being slow to notice and
#: being a process that wakes up ten thousand times.
POLL_START_S = 2.0
POLL_MAX_S = 30.0

#: Remote states that mean a job has stopped, whichever backend reported it.
#: Deliberately a fixed vocabulary rather than "anything that is not running":
#: `tools/kaggle.py:_parse_status` returns `unknown` for output it cannot read,
#: and treating unknown as finished is how you collect a kernel mid-run.
#:
#: `MISSING` is here on the same argument in reverse: `tools/kaggle.py` infers
#: it from a confirmed 404, so it is the opposite of "we could not read this" --
#: a kernel Kaggle has no record of will not acquire one by being polled again,
#: and a wake left armed on it would poll until its own deadline. Firing sends
#: the agent to `collect`, which refuses with `kernel_missing` and names
#: `forget`, which is exactly where that run needs to go.
TERMINAL_REMOTE = {
    "COMPLETE", "COMPLETED", "DONE", "ERROR", "FAILED", "CANCELED", "CANCELLED",
    "KILLED", "CANCELACKNOWLEDGED", "MISSING",
}

#: Which CLI answers `status` for a run, per `platform`. The sibling of
#: `core/submit.py:COLLECTORS`, and kept beside it in spirit for the same reason:
#: an unknown platform degrades to "cannot tell", never to a guess.
STATUS_TOOLS = {
    "hf_jobs": "tools.jobs",
    "kaggle": "tools.kaggle",
    "ssh": "tools.gpu",
}


# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------
def registry_path() -> Path:
    return appdata.workspace_state_dir() / "wakeups.jsonl"


def token_path() -> Path:
    return appdata.state_dir() / "wake.token"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def iso_at(epoch_s: float) -> str:
    """A `time.time()` deadline as an ISO instant, for a record a human reads."""
    return _dt.datetime.fromtimestamp(float(epoch_s), _dt.timezone.utc).isoformat(
        timespec="seconds"
    )


def new_id() -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%H%M%S")
    return f"wake-{stamp}-{secrets.token_hex(2)}"


def token() -> str:
    """The secret a wake must present to start a turn. Created on first use.

    **This is not decoration.** The app binds an unauthenticated loopback port
    on purpose -- `/__grad/show` raises a window, which is harmless -- but the
    endpoint a wake arrives on *starts a turn for an agent with Bash access*, and
    that is the one thing on this port that must not be reachable by anything
    that can open a socket to it. Any process on the machine can; only ours can
    read a mode-600 file in the app directory.

    Persisted rather than generated per launch, because the watcher outlives the
    app: a wake armed before a restart has to still be deliverable after it.
    """
    path = token_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:  # not every filesystem honours it; the file is local either way
            log.debug("could not restrict permissions on %s", path)
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def events() -> list[dict[str, Any]]:
    return jsonl.read(registry_path())


def wakeups(*, check_liveness: bool = True) -> dict[str, dict[str, Any]]:
    """Every wake, folded to its current state."""
    folded: dict[str, dict[str, Any]] = {}
    for record in events():
        wake_id = record.get("id")
        if not wake_id:
            continue
        kind = record.get("type")
        if kind == T_ARMED:
            folded[wake_id] = {
                "id": wake_id,
                "state": ARMED,
                "armed_at": record.get("at"),
                "condition": record.get("condition") or {},
                "note": record.get("note") or "",
                "deadline": record.get("deadline"),
                "pid": record.get("pid"),
                "resume": bool(record.get("resume", True)),
                "detail": None,
                "finished_at": None,
                "delivered": None,
            }
        elif wake_id in folded:
            node = folded[wake_id]
            if kind == T_FIRED:
                node["state"] = EXPIRED if record.get("expired") else FIRED
                node["detail"] = record.get("detail")
                node["finished_at"] = record.get("at")
            elif kind == T_CANCELLED:
                node["state"] = CANCELLED
                node["finished_at"] = record.get("at")
            elif kind == T_DELIVERED:
                node["delivered"] = record.get("delivered")
            elif kind == T_FORGOTTEN:
                folded.pop(wake_id, None)

    if check_liveness:
        from core import tasks as tasklib  # noqa: PLC0415 - shares the pid check

        pending = [n for n in folded.values() if n["state"] == ARMED and n.get("pid")]
        if pending:
            live = tasklib.alive_pids([n["pid"] for n in pending], max_age_s=2.0)
            for node in pending:
                if node["pid"] not in live:
                    node["state"] = LOST
    return folded


def get(wake_id: str) -> dict[str, Any] | None:
    return wakeups().get(wake_id)


def armed() -> list[dict[str, Any]]:
    return [w for w in wakeups().values() if w["state"] == ARMED]


def pending_delivery() -> list[dict[str, Any]]:
    """Wakes that fired and never reached a session.

    What the agent finds when it comes back to a workspace whose app was closed
    while the watcher was still running. Without this a wake that fired into a
    machine with no UI would simply be lost, which is the failure this whole
    module exists to prevent, arrived at by a different road.
    """
    return [
        w
        for w in wakeups().values()
        if w["state"] in (FIRED, EXPIRED) and not w.get("delivered")
    ]


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def append(record: dict[str, Any]) -> dict[str, Any]:
    return jsonl.append(registry_path(), {"at": now_iso(), **record})


def record_armed(
    wake_id: str,
    *,
    condition: dict[str, Any],
    deadline: float,
    note: str,
    pid: int,
    resume: bool,
) -> dict[str, Any]:
    return append(
        {
            "type": T_ARMED,
            "id": wake_id,
            "condition": condition,
            "deadline": deadline,
            "note": note,
            "pid": pid,
            "resume": resume,
        }
    )


def record_fired(wake_id: str, *, detail: dict[str, Any], expired: bool = False) -> dict[str, Any]:
    return append({"type": T_FIRED, "id": wake_id, "detail": detail, "expired": expired})


def record_cancelled(wake_id: str, *, reason: str = "") -> dict[str, Any]:
    return append({"type": T_CANCELLED, "id": wake_id, "reason": reason})


def record_delivered(wake_id: str, *, delivered: str) -> dict[str, Any]:
    return append({"type": T_DELIVERED, "id": wake_id, "delivered": delivered})


def forget(wake_ids: list[str]) -> int:
    for wake_id in wake_ids:
        append({"type": T_FORGOTTEN, "id": wake_id})
    return len(wake_ids)


# ---------------------------------------------------------------------------
# the conditions
# ---------------------------------------------------------------------------
def describe(condition: dict[str, Any]) -> str:
    """One line naming what is being waited for, for a prompt and a listing."""
    kind = condition.get("kind")
    if kind == KIND_AFTER:
        return f"{int(condition.get('seconds') or 0)}s elapse"
    if kind == KIND_TASK:
        return f"background task {condition.get('task')} finishes"
    if kind == KIND_RUN:
        return f"run {condition.get('run')} stops running on its backend"
    if kind == KIND_FILE:
        what = "changes" if condition.get("changed") else "appears"
        return f"{condition.get('path')} {what}"
    return str(kind or "an unknown condition")


def check(condition: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Has it happened? Returns `(fired, detail)` and never raises.

    Never raises because this runs in a loop in a detached process with nowhere
    to report to. A condition that cannot be read right now -- an unreadable
    ledger, a `kaggle` CLI that timed out, a network that is down -- is not a
    condition that has been met, and the honest answer is to look again in a few
    seconds. The timeout is what stops that being forever.
    """
    try:
        return _check(condition)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.debug("could not evaluate %s", condition, exc_info=True)
        return False, {"unreadable": f"{type(exc).__name__}: {exc}"}


def _check(condition: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    kind = condition.get("kind")

    if kind == KIND_AFTER:
        # Held by the watcher's own clock rather than measured here: the
        # deadline is absolute, so a machine that slept through the interval
        # wakes to a condition that is already true.
        fire_at = float(condition.get("fire_at") or 0)
        if time.time() >= fire_at:
            return True, {"elapsed_s": int(condition.get("seconds") or 0)}
        return False, {}

    if kind == KIND_TASK:
        from core import tasks as tasklib  # noqa: PLC0415

        task = tasklib.get(str(condition.get("task") or ""))
        if task is None:
            return False, {"missing": "no such task in the registry"}
        if task.get("state") in tasklib.TERMINAL:
            return True, {
                "task": task.get("id"),
                "state": task.get("state"),
                "exit_code": task.get("exit_code"),
                "label": task.get("label"),
            }
        return False, {"state": task.get("state")}

    if kind == KIND_FILE:
        path = Path(str(condition.get("path") or ""))
        if not path.exists():
            return False, {}
        if not condition.get("changed"):
            return True, {"path": str(path), "appeared": True}
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return False, {}
        baseline = condition.get("mtime_ns")
        if baseline is None or mtime != baseline:
            return True, {"path": str(path), "mtime_ns": mtime}
        return False, {}

    if kind == KIND_RUN:
        return _check_run(str(condition.get("run") or ""))

    return False, {"unknown_kind": str(kind)}


#: Run statuses that mean it is over. Every one of these is written by
#: `core/submit.py:record_collected`, which stamps `collected_at` too -- so in
#: practice the `collected` branch below catches them first, and this is the
#: backstop for a record where the two disagree.
#:
#: An explicit set, and that is the whole point. The test used to be `status and
#: status != "in_flight"`, which treats *everything* unrecognised as finished --
#: including `"unknown"`, which is what `Run.status` returns for a fold with no
#: status in it. `ledger_store.runs()` builds a node from any event carrying an
#: id, and `jsonl.iter_records` skips a malformed line rather than raising, so a
#: torn `run_submitted` line followed by an intact `run_handle` produces exactly
#: that record. The wake then fired immediately, reported that the run had
#: stopped, and spent a metered turn on a claim nothing had checked -- when the
#: honest answer was "ask the backend", which is what falling through does.
#: `forgotten` is here for the same reason `abandoned` is: `tools/kaggle.py
#: forget` writes a terminal record for a kernel Kaggle no longer has, and a
#: wake left armed on it would poll a run that has already stopped.
TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "failed", "submit_failed", "abandoned", "forgotten"}
)


def _check_run(run_id: str) -> tuple[bool, dict[str, Any]]:
    """Has this run stopped running on whatever machine it went to?

    Two sources, cheapest first. The ledger is a local file and knows when a run
    has already been collected; only if it is still in flight is the backend
    asked, and that is a network call priced accordingly by the backoff.

    The backend is reached through its own `status` CLI rather than through its
    client library, which is the same choice `tools/task.py` makes about running
    commands: the CLI owns the credential handling, the timeouts and the §8
    envelope, and a second code path that talked to Hugging Face directly would
    be a second place for the namespace bug to live.
    """
    from core import ledger_store as ls  # noqa: PLC0415

    try:
        record = ls.run(run_id)
    except Exception as exc:  # noqa: BLE001 - a missing run is not a fired wake
        return False, {"unreadable": f"{type(exc).__name__}: {exc}"}

    if record.collected:
        return True, {"run": run_id, "collected": True, "status": record.status}
    if record.status in TERMINAL_RUN_STATUSES:
        return True, {"run": run_id, "status": record.status}

    tool = STATUS_TOOLS.get(str(record.get("platform") or ""))
    if tool is None:
        # An unknown platform cannot be polled, so the wake rests on the ledger
        # alone. Said in the detail rather than silently degrading, because
        # "still in flight" and "nobody can tell you" are different answers.
        return False, {"run": run_id, "unpollable": str(record.get("platform") or "unknown")}

    data = _status_envelope(tool, run_id)
    if data is None:
        return False, {"run": run_id, "unreadable": "the status command did not answer"}
    if data.get("collected"):
        return True, {"run": run_id, "collected": True}
    if _remote_finished(data):
        return True, {"run": run_id, "remote": _remote_state(data)}
    return False, {"run": run_id, "remote": _remote_state(data)}


def _status_envelope(tool: str, run_id: str) -> dict[str, Any] | None:
    """`python -m <tool> status <run_id> --json`, decoded to its `data`."""
    from core import paths, spawn  # noqa: PLC0415

    try:
        proc = spawn.run(
            [sys.executable, "-m", tool, "status", run_id, "--json"],
            cwd=str(paths.root()),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        envelope = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    if not isinstance(envelope, dict):
        return None
    data = envelope.get("data")
    return data if isinstance(data, dict) else None


def _remote_state(data: dict[str, Any]) -> Any:
    """Whatever this backend calls the remote's state, for the record."""
    if isinstance(data.get("remote"), dict):
        return data["remote"] or None
    if isinstance(data.get("kernel"), dict):
        return data["kernel"].get("status")
    return data.get("remote_state")


def _remote_finished(data: dict[str, Any]) -> bool:
    """Does this status payload say the job has stopped?

    Three shapes, because there are three backends and they were never going to
    agree: `gpu.py` reports the marker file it writes when the command exits,
    `jobs.py` a stage string, `kaggle.py` a kernel status. An unrecognised value
    is not finished -- see `TERMINAL_REMOTE`.
    """
    marker = data.get("remote")
    if isinstance(marker, dict) and marker:
        return True
    candidates = [data.get("remote_state")]
    kernel = data.get("kernel")
    if isinstance(kernel, dict):
        candidates.append(kernel.get("status"))
    return any(
        isinstance(value, str) and value.strip().upper() in TERMINAL_REMOTE
        for value in candidates
    )


# ---------------------------------------------------------------------------
# waking the agent
# ---------------------------------------------------------------------------
def prompt_for(wake: dict[str, Any], detail: dict[str, Any], *, expired: bool) -> str:
    """The turn a fired wake issues.

    Written as a report to the agent rather than as an instruction, and the
    difference matters: the agent armed this and knows why, and a prompt that
    told it what to do next would be this module deciding research questions. It
    states what was waited for, what happened, and the note the agent left
    itself.
    """
    what = describe(wake.get("condition") or {})
    head = (
        f"[wakeup {wake['id']}] the condition did not happen within the timeout you set: {what}."
        if expired
        else f"[wakeup {wake['id']}] {what} — this is the wake you armed."
    )
    lines = [head]
    if wake.get("note"):
        lines.append(f"\nWhat you said you were waiting for: {wake['note']}")
    readable = {k: v for k, v in (detail or {}).items() if v not in (None, {}, "")}
    if readable:
        lines.append(f"\nWhat the watcher saw: {json.dumps(readable, default=str)}")
    return "\n".join(lines)


def deliver(wake_id: str, prompt: str) -> bool:
    """Hand a wake to the running app, and say whether it took it.

    The same shape as `core/instance.py:show_running` -- the published port, a
    short timeout, every failure folded to False -- with the one difference that
    matters: this carries the token. See `token`.

    A False here is not an error. It is a workspace whose app is closed, which
    is an ordinary thing for a four-hour job to finish into, and
    `pending_delivery` is what makes it recoverable rather than lost.
    """
    from core import instance  # noqa: PLC0415

    state = instance.read_state()
    port = state.get("port")
    if not port:
        return False
    body = json.dumps({"wake": wake_id, "token": token(), "prompt": prompt}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed local scheme
        f"http://127.0.0.1:{int(port)}/__grad/wake",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def spawn_watcher(wake_id: str) -> int:
    """Start the detached process that does the waiting. Returns its pid.

    Detached for the reason `tools/task.py` spawns its supervisor detached: the
    CLI that armed this is about to exit, and the whole point is that the wait
    outlives it. `core/spawn.py` explains why that is `CREATE_NO_WINDOW` and not
    `DETACHED_PROCESS`.
    """
    from core import paths, spawn  # noqa: PLC0415

    child = subprocess.Popen(  # noqa: S603 - our own module, our own argv
        [sys.executable, "-u", "-m", "tools.wakeup", "_watch", "--wake-id", wake_id],
        cwd=str(paths.root()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **spawn.detached(),
    )
    return int(child.pid)


def watch(wake_id: str) -> dict[str, Any]:
    """Poll one wake until it fires, expires, or is cancelled. **Blocks.**

    The body of the detached watcher, and a plain function so it can be tested
    without spawning anything.
    """
    wake = get(wake_id)
    if wake is None:
        return {"wake": wake_id, "state": "unknown"}
    condition = wake.get("condition") or {}
    deadline = float(wake.get("deadline") or 0)
    interval = POLL_START_S

    while True:
        current = get(wake_id)
        if current is None or current["state"] == CANCELLED:
            # Cancelled out from under us. Nothing to fire and nothing to say.
            return {"wake": wake_id, "state": CANCELLED}

        fired, detail = check(condition)
        expired = not fired and time.time() >= deadline
        if fired or expired:
            record_fired(wake_id, detail=detail, expired=expired)
            if wake.get("resume", True):
                prompt = prompt_for(wake, detail, expired=expired)
                if deliver(wake_id, prompt):
                    record_delivered(wake_id, delivered="session")
            return {
                "wake": wake_id,
                "state": EXPIRED if expired else FIRED,
                "detail": detail,
            }

        time.sleep(min(interval, max(0.0, deadline - time.time()) + 0.1))
        interval = min(POLL_MAX_S, interval * 1.5)
