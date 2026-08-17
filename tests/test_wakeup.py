"""Being woken instead of waiting (HANDOFF §6's unfinished half).

The thing under test is a *replacement for a habit*: `sleep 30`, look, `sleep
60`, look. So these cover the two properties that make the replacement worth
having -- the agent's shell is not held, and nothing is lost when the condition
happens into a workspace whose window is closed -- and the one that makes it
safe, which is that the endpoint a wake arrives on cannot be driven by anything
that merely knows the port.

Nothing here spawns a watcher. `wk.watch` is a plain function precisely so the
loop can be driven in-process; `arm` is exercised with the spawn stubbed.
"""

from __future__ import annotations

import time

import pytest

from core import tasks as tasklib, wakeups as wk
from core.errors import GradError
from tools import wakeup as wakeup_tool


@pytest.fixture
def no_spawn(monkeypatch):
    """`arm` without the detached process. The watcher is tested separately.

    It reports *this* process as the watcher rather than an invented pid, and
    that is not cosmetic: `wakeups()` folds an armed wake whose pid is gone to
    `lost`, so a made-up number makes every wake in the suite arrive already
    dead -- and `cancel` then correctly declines to cancel it.
    """
    import os

    spawned: list[str] = []

    def fake(wake_id: str) -> int:
        spawned.append(wake_id)
        return os.getpid()

    monkeypatch.setattr(wk, "spawn_watcher", fake)
    return spawned


def _arm(**kwargs):
    args = {
        "after": None, "task": None, "run": None, "file": None,
        "changed": False, "timeout": 3600.0, "note": "", "no_resume": False,
    }
    args.update(kwargs)
    return wakeup_tool.cmd_arm(type("A", (), args)())


# ---------------------------------------------------------------------------
# arming
# ---------------------------------------------------------------------------
def test_arming_returns_at_once_and_tells_the_agent_to_stop(workspace, no_spawn):
    """The whole point. `arm` must not be a thing you wait on -- if it were, it
    would be the `sleep` it replaces with extra steps."""
    started = time.monotonic()
    out = _arm(after=3600)
    assert time.monotonic() - started < 2.0
    assert out["wake"].startswith("wake-")
    assert "end your turn" in out["next"]


def test_the_deadline_is_reported_as_a_deadline(workspace, no_spawn):
    """It said `now` for its first hour of existence, which is the one value
    that is never the answer to 'when does this expire'."""
    out = _arm(after=60, timeout=1800.0)
    assert out["expires_at"] > out["wake"][5:], "not an instant at all"
    armed_at = wk.get(out["wake"])["armed_at"]
    assert out["expires_at"] > armed_at


def test_a_timeout_past_the_ceiling_is_refused(workspace, no_spawn):
    with pytest.raises(GradError) as exc:
        _arm(after=60, timeout=wk.MAX_TIMEOUT_S + 1)
    assert "at most" in exc.value.message
    assert exc.value.fix


def test_waiting_on_a_finished_task_is_refused(workspace, no_spawn):
    """It would fire on its first look and spend a whole turn saying what a
    `task status` in the same turn would have said for nothing."""
    task_id = tasklib.new_id()
    tasklib.record_started(
        task_id, label="done", argv=["python", "-c", "pass"], pid=1, halt=None, cwd="."
    )
    tasklib.record_exited(task_id, 0)

    with pytest.raises(GradError) as exc:
        _arm(task=task_id)
    assert "already finished" in exc.value.message


def test_waiting_on_a_file_that_is_already_there_is_refused(workspace, no_spawn):
    (workspace / "there.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(GradError) as exc:
        _arm(file="there.txt")
    assert "--changed" in (exc.value.fix or "")


def test_a_changed_file_baseline_is_taken_at_arm_time(workspace, no_spawn):
    """'Changed since you asked', not 'changed since the watcher got round to
    looking' -- otherwise a write in between is silently missed."""
    path = workspace / "metrics.json"
    path.write_text("{}", encoding="utf-8")
    out = _arm(file="metrics.json", changed=True)

    condition = wk.get(out["wake"])["condition"]
    assert condition["mtime_ns"] == path.stat().st_mtime_ns
    assert wk.check(condition)[0] is False

    time.sleep(0.01)
    path.write_text('{"loss": 1}', encoding="utf-8")
    fired, detail = wk.check(condition)
    assert fired and detail["path"] == str(path)


def test_an_unknown_task_is_a_named_refusal_not_a_traceback(workspace, no_spawn):
    with pytest.raises(GradError) as exc:
        _arm(task="task-000000-dead")
    assert "no background task" in exc.value.message


# ---------------------------------------------------------------------------
# the conditions
# ---------------------------------------------------------------------------
def test_a_clock_condition_survives_a_sleeping_machine(workspace):
    """The deadline is absolute, so a laptop that slept through the interval
    wakes to a condition that is already true rather than to a fresh countdown."""
    condition = {"kind": wk.KIND_AFTER, "seconds": 5, "fire_at": time.time() - 1}
    assert wk.check(condition)[0] is True


def test_a_task_condition_fires_on_the_record_not_on_the_pid(workspace):
    """`core/tasks.py` distinguishes 'exited' from 'stopped being alive', and a
    wake that could not tell them apart would report a killed supervisor as a
    finished job."""
    task_id = tasklib.new_id()
    tasklib.record_started(
        task_id, label="train", argv=["python", "train.py"], pid=999999, halt=None, cwd="."
    )
    condition = {"kind": wk.KIND_TASK, "task": task_id}

    tasklib.record_exited(task_id, 3)
    fired, detail = wk.check(condition)
    assert fired
    assert detail["exit_code"] == 3


def test_an_unreadable_condition_is_not_a_fired_one(workspace, monkeypatch):
    """This runs in a detached loop with nowhere to report to, so a condition
    that cannot be read is 'look again', never 'it happened'."""
    def explode(_):
        raise RuntimeError("the ledger is on fire")

    monkeypatch.setattr(wk, "_check", explode)
    fired, detail = wk.check({"kind": wk.KIND_AFTER})
    assert fired is False
    assert "unreadable" in detail


def test_an_unrecognised_remote_state_is_not_finished(workspace):
    """`tools/kaggle.py:_parse_status` returns `unknown` for output it cannot
    read, and treating unknown as finished is how you collect a kernel mid-run."""
    assert wk._remote_finished({"kernel": {"status": "unknown"}}) is False
    assert wk._remote_finished({"remote_state": "RUNNING"}) is False
    assert wk._remote_finished({"kernel": {"status": "complete"}}) is True
    assert wk._remote_finished({"remote_state": "COMPLETED"}) is True
    # The ssh backend writes a marker file only when the command exits, so the
    # marker's presence *is* the terminal signal.
    assert wk._remote_finished({"remote": {"exit_code": 0}}) is True
    assert wk._remote_finished({"remote": {}}) is False


def test_a_run_with_no_status_is_not_a_finished_run(workspace, monkeypatch):
    """The same mistake as the line above, one layer down.

    The ledger short-circuit was `status and status != "in_flight"`, which treats
    everything unrecognised as finished -- including `"unknown"`, which is what
    `Run.status` returns for a fold with no status in it. `runs()` builds a node
    from any event carrying an id and `jsonl.iter_records` *skips* a malformed
    line rather than raising, so a torn `run_submitted` followed by an intact
    `run_handle` produces exactly that record. The wake then fired at once and
    spent a metered turn reporting something nothing had checked.
    """
    from core import ledger_store as ls

    # The record that damage leaves behind: an id, a platform, no status.
    ls.append_run_event({"type": "run_handle", "id": "run-torn", "platform": "hf_jobs"})
    assert ls.run("run-torn").status == "unknown"

    polled: list[str] = []

    def _never_answers(tool, run_id):
        polled.append(run_id)
        return None

    monkeypatch.setattr(wk, "_status_envelope", _never_answers)

    fired, detail = wk._check_run("run-torn")
    assert fired is False, "unknown is not finished"
    assert polled == ["run-torn"], "it has to fall through to the backend, which does know"
    assert "unreadable" in detail


def test_a_terminal_status_still_fires_without_a_poll(workspace, monkeypatch):
    """The short-circuit is still a short-circuit for the statuses that mean it."""
    from core import ledger_store as ls

    def _never(tool, run_id):
        raise AssertionError("the backend must not be asked about a finished run")

    monkeypatch.setattr(wk, "_status_envelope", _never)

    for index, status in enumerate(sorted(wk.TERMINAL_RUN_STATUSES)):
        run_id = f"run-{index}"
        ls.append_run_event({"type": ls.T_RUN_SUBMITTED, "id": run_id, "status": "in_flight"})
        ls.append_run_event({"type": "run_finished", "id": run_id, "status": status})
        fired, detail = wk._check_run(run_id)
        assert fired is True, status
        assert detail["status"] == status


# ---------------------------------------------------------------------------
# the watcher
# ---------------------------------------------------------------------------
def test_the_watcher_fires_and_records_what_it_saw(workspace, no_spawn, monkeypatch):
    out = _arm(after=0, timeout=30.0)
    monkeypatch.setattr(wk, "deliver", lambda *a, **k: False)

    result = wk.watch(out["wake"])
    assert result["state"] == wk.FIRED

    wake = wk.get(out["wake"])
    assert wake["state"] == wk.FIRED
    assert wake["detail"] == {"elapsed_s": 0}


def test_a_condition_that_never_happens_expires_as_its_own_state(workspace, no_spawn, monkeypatch):
    """'The job finished' and 'the job has not finished in four hours' are
    different facts and lead to different next actions, so `expired` is not a
    kind of `fired`."""
    out = _arm(file="never.txt", timeout=0.4)
    monkeypatch.setattr(wk, "deliver", lambda *a, **k: False)

    result = wk.watch(out["wake"])
    assert result["state"] == wk.EXPIRED
    assert wk.get(out["wake"])["state"] == wk.EXPIRED


def test_an_expired_wake_says_so_in_the_turn_it_sends(workspace, no_spawn):
    out = _arm(file="never.txt", timeout=0.3)
    wake = wk.get(out["wake"])
    turn = wk.prompt_for(wake, {}, expired=True)
    assert "did not happen within the timeout" in turn


def test_the_waking_turn_carries_the_note_back(workspace, no_spawn):
    """The agent armed this and knows why; the note is how it tells itself."""
    out = _arm(after=0, note="the 4090 sweep — collect and judge against exp-7")
    wake = wk.get(out["wake"])
    turn = wk.prompt_for(wake, {"elapsed_s": 0}, expired=False)
    assert "exp-7" in turn
    assert out["wake"] in turn


def test_cancelling_stops_the_watcher_without_a_signal(workspace, no_spawn):
    out = _arm(file="never.txt", timeout=30.0)
    wakeup_tool.cmd_cancel(type("A", (), {"wake_id": out["wake"], "reason": "changed my mind"})())

    result = wk.watch(out["wake"])
    assert result["state"] == wk.CANCELLED
    assert wk.get(out["wake"])["state"] == wk.CANCELLED


def test_a_wake_with_no_resume_never_tries_to_deliver(workspace, no_spawn, monkeypatch):
    delivered: list[str] = []
    monkeypatch.setattr(wk, "deliver", lambda wid, prompt: delivered.append(wid) or True)

    out = _arm(after=0, no_resume=True)
    wk.watch(out["wake"])
    assert delivered == []


def test_a_fired_wake_nobody_took_is_kept_not_lost(workspace, no_spawn, monkeypatch):
    """A four-hour job finishing into a closed app is ordinary. Losing the wake
    there would be the same failure this module exists to prevent, reached by a
    different road."""
    monkeypatch.setattr(wk, "deliver", lambda *a, **k: False)
    out = _arm(after=0, timeout=30.0)
    wk.watch(out["wake"])

    pending = wk.pending_delivery()
    assert [w["id"] for w in pending] == [out["wake"]]

    listing = wakeup_tool.cmd_list(type("A", (), {"all": False})())
    assert listing["undelivered"] == 1
    assert "fired while nothing was listening" in listing["note"]


def test_status_prints_the_turn_an_undelivered_wake_would_have_sent(workspace, no_spawn, monkeypatch):
    monkeypatch.setattr(wk, "deliver", lambda *a, **k: False)
    out = _arm(after=0, note="check the sweep")
    wk.watch(out["wake"])

    status = wakeup_tool.cmd_status(type("A", (), {"wake_id": out["wake"]})())
    assert "check the sweep" in status["turn"]


def test_a_delivered_wake_is_recorded_as_delivered(workspace, no_spawn, monkeypatch):
    monkeypatch.setattr(wk, "deliver", lambda *a, **k: True)
    out = _arm(after=0)
    wk.watch(out["wake"])

    assert wk.get(out["wake"])["delivered"] == "session"
    assert wk.pending_delivery() == []


# ---------------------------------------------------------------------------
# the token
# ---------------------------------------------------------------------------
def test_the_token_is_stable_across_calls(workspace):
    """The watcher outlives the app: a wake armed before a restart has to still
    be deliverable after it."""
    assert wk.token() == wk.token()
    assert len(wk.token()) > 20


def test_the_token_is_not_in_the_workspace(workspace):
    """It is machine state and the workspace is a repository. A secret that
    lands beside the ledger is a secret in someone's next commit."""
    assert workspace not in wk.token_path().parents


def test_delivery_without_a_running_instance_is_false_not_an_error(workspace, monkeypatch):
    from core import instance

    monkeypatch.setattr(instance, "read_state", lambda: {})
    assert wk.deliver("wake-1", "hello") is False


# ---------------------------------------------------------------------------
# where a wake lands
# ---------------------------------------------------------------------------
class _Session:
    busy = False
    settled: list = []


class _Workspace:
    """A `Workspace` reduced to what delivery touches."""

    from ui.state import Workspace as _real

    MAX_PENDING_WAKES = _real.MAX_PENDING_WAKES
    accept_wake = _real.accept_wake
    _deliver_wakes = _real._deliver_wakes

    def __init__(self) -> None:
        self.session = _Session()
        self.pending_wakes: list[str] = []
        self.sent: list[str] = []
        self.said: list[str] = []
        self.opened: list[str] = []
        self.state = "idle"
        self.chat_send = lambda prompt: self.sent.append(prompt)

    def say(self, message: str) -> None:
        self.said.append(message)

    def open(self, window_id: str) -> None:
        self.opened.append(window_id)

    def set_agent_state(self, state: str) -> None:
        self.state = state


@pytest.mark.asyncio
async def test_a_wake_becomes_a_turn_on_the_next_tick(workspace):
    space = _Workspace()
    assert space.accept_wake("[wakeup] the run finished")
    assert await space._deliver_wakes() is True
    assert space.sent == ["[wakeup] the run finished"]
    assert space.state == "running"


@pytest.mark.asyncio
async def test_a_wake_never_lands_on_a_running_turn(workspace):
    """`Session.ask` refuses a prompt during a turn, so delivering into one
    would consume the wake and answer nothing -- the silent loss this whole
    mechanism exists to prevent."""
    space = _Workspace()
    space.session.busy = True
    space.accept_wake("[wakeup] the run finished")

    assert await space._deliver_wakes() is False
    assert space.sent == []
    assert space.pending_wakes == ["[wakeup] the run finished"]

    space.session.busy = False
    assert await space._deliver_wakes() is True
    assert space.sent == ["[wakeup] the run finished"]


@pytest.mark.asyncio
async def test_a_wake_with_no_chat_window_opens_one(workspace):
    space = _Workspace()
    space.chat_send = None
    space.accept_wake("[wakeup] the run finished")

    assert await space._deliver_wakes() is False
    assert space.opened == ["chat"]
    assert space.pending_wakes, "the wake was dropped rather than held"
    assert any("wakeup" in line for line in space.said)


@pytest.mark.asyncio
async def test_one_wake_per_tick(workspace):
    space = _Workspace()
    space.accept_wake("first")
    space.accept_wake("second")

    await space._deliver_wakes()
    assert space.sent == ["first"]
    await space._deliver_wakes()
    assert space.sent == ["first", "second"]


def test_a_runaway_watcher_cannot_flood_the_conversation(workspace):
    space = _Workspace()
    for i in range(_Workspace.MAX_PENDING_WAKES):
        assert space.accept_wake(f"wake {i}")
    assert space.accept_wake("one too many") is False


def test_an_empty_wake_is_refused(workspace):
    assert _Workspace().accept_wake("   ") is False
