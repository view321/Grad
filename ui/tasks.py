"""Background tasks: the CLIs the UI starts and does not wait for.

Every button in the workspace that *does* something does it by running the same
command the agent would (§10). Most of those return in under a second. Four do
not -- `nb verify` on a fresh kernel, `preflight run`, `wiki map`, `report
build` -- and the way they used to be run had three problems, all of which this
module exists to close.

**A wall clock that killed working commands.** `run_tool` awaited the process
under a 900-second cap and killed it on expiry. But `verify_timeout_s` is 1800
*per cell*, and preflight's `tests` and `dry_run` are 900 each -- so the cap sat
*below* the runtime the configuration explicitly allows, and the UI's answer to
a slow notebook was to kill it and report a timeout. A background task has no
wall clock: it finishes, or you stop it.

**One status line for every operation.** `Workspace.say` holds a single string,
so two commands in flight overwrote each other and neither left a trace. Tasks
have their own list, their own state and their own output.

**Nothing to watch.** A campaign or a wiki rebuild was opaque from click to
envelope. Output is streamed here as it arrives, in a bounded tail.

**Stopping is asked for, not inflicted.** `terminate()` reaches the CLI and
nothing it spawned -- and `nb verify` spawns its kernel *detached*, precisely so
it outlives the CLI. Killing the parent would leave that kernel holding the VRAM
the verify was meant to free. So a task may carry the tool's own stop verb
(`nb stop`, `evolve halt`), which is tried first and given a grace period; the
signal is the fallback, not the mechanism.

The registry is module-level rather than per-`Workspace` on purpose. A
`Workspace` belongs to one connected client, and a task must survive a browser
reload -- it is a process on this machine, not a view of one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from core import paths

log = logging.getLogger("grad.ui")

#: Lines of output kept per task. The tail is a window onto a running command,
#: not a log: the log is whatever the command writes to the workspace.
TAIL_LINES = 400
#: A single line longer than this is cut. A progress bar that never emits a
#: newline would otherwise buffer without bound.
MAX_LINE_CHARS = 4000
#: Finished tasks kept before the oldest are dropped. Enough to look back over a
#: session; not so many that the window becomes a history.
KEEP_FINISHED = 40
#: How long a tool's own stop verb is given before the signal is used instead.
HALT_GRACE_S = 20.0
#: Between `terminate()` and `kill()`.
KILL_GRACE_S = 5.0

RUNNING = "running"
OK = "ok"
FAILED = "failed"
CANCELLED = "cancelled"

#: `state -> the one accent it is drawn in`, matching the rest of the app: a
#: dashed border while it runs, because an outcome that has not happened yet is
#: not a green one.
STATE_TONE = {RUNNING: "dashed", OK: "ok", FAILED: "broken", CANCELLED: "attention"}


@dataclass
class Task:
    """One local command, running or finished."""

    id: str
    label: str
    argv: tuple[str, ...]
    #: The tool's own graceful stop, if it has one. See the module docstring.
    halt: tuple[str, ...] | None = None
    state: str = RUNNING
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    exit_code: int | None = None
    #: The last JSON object the command printed on stdout -- the §8 envelope,
    #: tracked as the lines arrive rather than by re-reading the output, so a
    #: task that printed a gigabyte still costs one dict.
    envelope: dict[str, Any] | None = None
    tail: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=TAIL_LINES))
    #: Lines that fell out of the tail. Reported rather than silently dropped.
    dropped: int = 0
    #: Called once, when the task settles. See `start`.
    on_done: Any = None
    #: Set by `cancel` before it signals, read by `_run` when the process goes.
    stopping: bool = False
    _process: Any = None
    #: The coroutine driving this task, held so it cannot be collected.
    #: asyncio keeps only a *weak* reference to a running task, so a bare
    #: `create_task` whose result nobody holds can vanish part-way through --
    #: the same failure `Workspace.spawn` exists to close, and here it would
    #: strand a process with no pump and nothing left to settle it.
    _driver: Any = None

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.monotonic()) - self.started_at

    def append(self, line: str, tag: str = "out") -> None:
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + f" … +{len(line) - MAX_LINE_CHARS:,} characters"
        if len(self.tail) == self.tail.maxlen:
            self.dropped += 1
        self.tail.append((tag, line))
        if tag == "out":
            self._remember_envelope(line)

    def note(self, line: str) -> None:
        """A line from the workspace itself -- that a stop was asked for, say."""
        self.append(line, tag="note")

    def _remember_envelope(self, line: str) -> None:
        stripped = line.strip()
        if not stripped.startswith("{"):
            return
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.envelope = payload


_tasks: dict[str, Task] = {}
_counter = 0


def all_tasks() -> list[Task]:
    """Newest first, which is the order the window wants."""
    return list(reversed(_tasks.values()))


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)


def running() -> list[Task]:
    return [t for t in _tasks.values() if t.running]


def clear_finished() -> int:
    """Drop every settled task. Returns how many went."""
    gone = [tid for tid, task in _tasks.items() if not task.running]
    for tid in gone:
        del _tasks[tid]
    return len(gone)


def reset() -> None:
    """Empty the registry. For tests -- module state outlives a fixture."""
    global _counter

    _tasks.clear()
    _counter = 0


def _register(task: Task) -> Task:
    _tasks[task.id] = task
    finished = [tid for tid, other in _tasks.items() if not other.running]
    for tid in finished[: max(0, len(finished) - KEEP_FINISHED)]:
        del _tasks[tid]
    return task


def _next_id() -> str:
    global _counter

    _counter += 1
    return f"task-{_counter}"


# ---------------------------------------------------------------------------
# starting one
# ---------------------------------------------------------------------------
def start(
    label: str,
    *argv: str,
    halt: Iterable[str] | None = None,
    on_done: Callable[[Task], None] | None = None,
) -> Task:
    """Run one of Grad's own CLIs in the background and return its handle.

    Returns as soon as the process is *registered*, not when it is spawned: the
    caller is a click handler and the window that lists tasks reads the registry
    on the next poll, so a task has to exist the moment the click is over --
    otherwise a two-second window opens in which nothing on screen says anything
    happened.

    `on_done` runs once the task settles, however it settled -- a verify has to
    write its record whether it passed, failed or was stopped, because "we do
    not know" is a different notebook state from "it was fine before".
    """
    task = _register(Task(_next_id(), label, tuple(argv), tuple(halt) if halt else None))
    task.on_done = on_done
    task.note(f"$ python -m {' '.join(argv)}")
    task._driver = asyncio.get_event_loop().create_task(_run(task))  # noqa: SLF001 - its own field
    return task


async def drained(task: Task) -> Task:
    """Wait until this task's driver has finished with the process.

    Later than `state`: a stop that lands before the process is even spawned
    settles the task immediately, while `_run` still has to spawn, terminate and
    drain it. Nothing in the UI waits for this -- the poll shows the state -- but
    a caller that needs the process to be *gone* does.
    """
    driver = task._driver  # noqa: SLF001 - its own field
    if driver is not None:
        await asyncio.gather(driver, return_exceptions=True)
    return task


async def _run(task: Task) -> None:
    try:
        task._process = await asyncio.create_subprocess_exec(  # noqa: SLF001 - its own field
            sys.executable,
            "-m",
            *task.argv,
            cwd=str(paths.root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        task.note(f"could not start: {exc}")
        _settle(task, FAILED, None)
        return

    process = task._process  # noqa: SLF001 - its own field
    if task.stopping:
        # Stopped while the spawn was still in flight. `cancel` found no process
        # to signal and settled the task itself, so killing it lands here --
        # without this, a task the workspace reported as stopped runs to
        # completion unattended, with nothing on screen saying it is still going.
        task.note("stopped while it was starting")
        try:
            process.terminate()
        except (ProcessLookupError, OSError):
            pass

    await asyncio.gather(
        _pump(process.stdout, task, tag="out"),
        _pump(process.stderr, task, tag="err"),
    )
    code = await process.wait()
    # The single place a started task settles. `cancel` sets `stopping` and then
    # waits rather than settling the task itself: both coroutines are awake at
    # once, and whichever wrote the state second would win. A stopped task
    # reporting the signal's exit code as a failure is the wrong verdict on a
    # command that did nothing wrong.
    if task.stopping:
        _settle(task, CANCELLED, code)
    else:
        _settle(task, OK if code == 0 else FAILED, code)


async def _pump(stream: Any, task: Task, *, tag: str) -> None:
    """Split a pipe into lines without `readline`.

    `StreamReader.readline` raises once a single line passes its 64 KiB limit,
    and a training log's progress line can. Chunks cannot hit that.
    """
    pending = ""
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        pending += chunk.decode("utf-8", "replace")
        *complete, pending = pending.split("\n")
        for line in complete:
            task.append(line.rstrip("\r"), tag)
        if len(pending) > MAX_LINE_CHARS:
            task.append(pending, tag)
            pending = ""
    if pending:
        task.append(pending.rstrip("\r"), tag)


def _settle(task: Task, state: str, code: int | None) -> None:
    task.state = state
    task.exit_code = code
    task.finished_at = time.monotonic()
    callback, task.on_done = task.on_done, None
    if callback is None:
        return
    try:
        callback(task)
    except Exception as exc:  # noqa: BLE001 - a callback must not strand the task
        log.exception("the completion callback for %s failed", task.id)
        task.note(f"the workspace could not record this result: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# stopping one
# ---------------------------------------------------------------------------
async def cancel(task_id: str) -> str:
    """Stop a task, asking the tool first. Returns the line to put on screen.

    The order matters more here than anywhere else in this module. `terminate()`
    reaches the CLI and nothing it started, and `nb verify` starts its kernel
    detached so it survives the CLI exiting -- so killing the parent would leave
    a kernel holding VRAM with nothing left to shut it down. When a tool has a
    verb for stopping itself, that verb *is* the cancel; the signal is what
    happens when it does not work.
    """
    task = _tasks.get(task_id)
    if task is None:
        return f"no task {task_id}"
    if not task.running:
        return f"{task.label} already finished"

    # Before anything else: `_run` reads this when the process goes, and it is
    # what tells a stop apart from a failure.
    task.stopping = True

    if task.halt:
        task.note(f"stopping: python -m {' '.join(task.halt)}")
        payload = await run_tool(*task.halt, timeout=HALT_GRACE_S)
        if not payload.get("ok"):
            task.note(f"the tool's own stop failed: {envelope_message(payload)}")
        if await _waits_for_exit(task, HALT_GRACE_S):
            return f"{task.label} stopped"
        task.note("it did not stop when asked; signalling")

    process = task._process  # noqa: SLF001 - its own field
    if process is None:
        # Nothing to signal, and nothing that will ever settle it.
        _settle(task, CANCELLED, None)
        return f"{task.label} stopped before it started"

    try:
        process.terminate()
    except (ProcessLookupError, OSError):
        return f"{task.label} was already gone"
    if await _waits_for_exit(task, KILL_GRACE_S):
        return f"{task.label} stopped"
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass
    task.note("killed")
    return f"{task.label} killed"


async def _waits_for_exit(task: Task, seconds: float) -> bool:
    """Poll for the process to go, so a grace period is a grace period.

    `process.wait()` is not awaited here: `_run` is already awaiting it, and a
    second waiter on the same transport is what turns a cancel into a hang.
    """
    process = task._process  # noqa: SLF001 - its own field
    if process is None:
        return True
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.returncode is not None:
            return True
        await asyncio.sleep(0.1)
    return process.returncode is not None


# ---------------------------------------------------------------------------
# running one and waiting for it
# ---------------------------------------------------------------------------
async def run_tool(*argv: str, timeout: float = 120.0, stdin: str | None = None) -> dict[str, Any]:
    """Run one of Grad's own CLIs and parse its JSON envelope.

    For the commands that answer immediately -- selecting a project, a verdict,
    a status poll. Anything that can run for minutes goes through `start`
    instead, because the timeout here is enforced by killing the process.

    Every button in the UI that *does* something does it by running the same
    command the agent would, with `--json`. That is deliberate: it keeps the UI
    free of logic (§10), and it means anything the UI can do is reproducible
    from a terminal and lands in the same ledgers.

    `stdin` exists for exactly one caller: storing a credential. A token passed
    as an argument is visible to anything that can list processes, so it goes
    down a pipe instead and the CLI reads it with `--stdin`.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            *argv,
            cwd=str(paths.root()),
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return {"ok": False, "error": {"message": f"could not run the command: {exc}"}}
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin.encode() if stdin is not None else None), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "ok": False,
            "error": {
                "message": f"timed out after {timeout:.0f}s",
                "fix": "long commands belong in the background — open the tasks window",
            },
        }

    stdout = (out or b"").decode("utf-8", "replace").strip()
    stderr = (err or b"").decode("utf-8", "replace").strip()
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {
        "ok": False,
        "error": {"message": (stderr or stdout or "the command produced no output")[-2000:]},
    }


def envelope_message(payload: dict[str, Any]) -> str:
    """The one line a status bar should show for a CLI result."""
    if payload.get("ok"):
        data = payload.get("data")
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
        return "done"
    error = payload.get("error") or {}
    message = error.get("message") or "the command failed"
    fix = error.get("fix")
    return f"{message}" + (f" — fix: {fix}" if fix else "")


def task_message(task: Task) -> str:
    """The one line a status bar should show for a finished task."""
    if task.running:
        return f"{task.label} running …"
    if task.state == CANCELLED:
        return f"{task.label} stopped"
    if task.envelope is not None:
        return f"{task.label}: {envelope_message(task.envelope)}"
    if task.state == OK:
        return f"{task.label} done"
    return f"{task.label} failed (exit {task.exit_code})"
