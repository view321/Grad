"""The project dimension and its ceilings (HANDOFF-2 §15).

§24 is explicit about how these should be tested: "The budget gates in 2 deserve
the same treatment §6's gates got: tested against a real ledger, not mocks,
because they are what stands between a loop and a bill." So every test here
writes real records into a real temp workspace and reads them back through the
same code paths the CLIs use.
"""

from __future__ import annotations

import pytest

from core import budget, gates, ledger_store as ls, quota_log
from core.errors import EXIT_PROJECT_BUDGET, EXIT_SPEND, GateRefusal
from tests.test_gates import make_expectation, make_submission, pass_preflight


def make_project(pid="proj-1", *, payer=None, **ceilings):
    return budget.create(
        pid,
        title="a piece of research",
        budget=ceilings or {"gpu_usd": 10.0},
        payer=payer,
    )


def record_run(project: str, *, usd: float, collected: bool = True):
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "status": "in_flight",
            "submitted_at": ls.now_iso(),
            "project": project,
            "estimate_usd": usd,
        }
    )
    if collected:
        ls.append_run_event(
            {
                "type": ls.T_RUN_COLLECTED,
                "id": run_id,
                "status": "completed",
                "collected_at": ls.now_iso(),
                "cost_usd_actual": usd,
                "results": {},
                "deviations": [],
            }
        )
    return run_id


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def test_a_project_is_folded_from_its_events(workspace):
    make_project("proj-1", gpu_usd=50.0)
    proj = budget.project("proj-1")
    assert proj["status"] == "open"
    assert proj["budget"]["gpu_usd"] == 50.0


def test_raise_appends_rather_than_mutating(workspace):
    """"a ceiling that can be edited invisibly is not a ceiling." The original
    value must stay readable, so "we kept raising it" is visible."""
    make_project("proj-1", gpu_usd=50.0)
    budget.raise_ceiling("proj-1", budget={"gpu_usd": 75.0}, reason="second sweep")

    events = budget.events()
    assert [e["type"] for e in events] == [budget.T_PROJECT, budget.T_PROJECT_RAISED]
    # The creation event is untouched; only the fold moves.
    assert events[0]["budget"]["gpu_usd"] == 50.0
    assert events[1]["previous"]["gpu_usd"] == 50.0
    assert budget.project("proj-1")["budget"]["gpu_usd"] == 75.0
    assert budget.project("proj-1")["raises"][0]["reason"] == "second sweep"


def test_current_project_is_a_file_not_an_environment_variable(workspace, monkeypatch):
    """§15: "a selection mechanism that the agent's own startup deletes is a bug
    waiting to happen." `scrub_environment` must not be able to unselect."""
    from core import credentials

    make_project("proj-1")
    budget.set_current("proj-1")
    assert budget.current_project_path().exists()

    monkeypatch.setenv("GRAD_PROJECT", "proj-elsewhere")
    credentials.scrub_environment()
    assert budget.current_project() == "proj-1"


def test_closing_clears_the_selection(workspace):
    make_project("proj-1")
    budget.set_current("proj-1")
    budget.close("proj-1")
    assert budget.current_project() is None
    assert budget.project("proj-1")["status"] == "closed"


# ---------------------------------------------------------------------------
# spend attribution
# ---------------------------------------------------------------------------
def test_spend_is_attributed_per_project(workspace):
    make_project("proj-1", gpu_usd=100.0)
    make_project("proj-2", gpu_usd=100.0)
    record_run("proj-1", usd=7.0)
    record_run("proj-2", usd=3.0)

    assert budget.spend("proj-1")["gpu_usd"] == 7.0
    assert budget.spend("proj-2")["gpu_usd"] == 3.0


def test_in_flight_runs_count_at_their_estimates(workspace):
    """Same rule as §6's global ceiling: a job that has not been collected yet
    is not free."""
    make_project("proj-1", gpu_usd=100.0)
    record_run("proj-1", usd=12.0, collected=False)
    state = budget.spend("proj-1")
    assert state["gpu_usd"] == 12.0
    assert state["gpu_in_flight_usd"] == 12.0


def test_records_without_a_project_fold_as_unassigned(workspace):
    """"an additive schema change; existing ledgers keep loading." """
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": "run-old",
            "status": "in_flight",
            "submitted_at": ls.now_iso(),
            "estimate_usd": 5.0,
        }
    )
    assert ls.run("run-old").project == "unassigned"
    make_project("proj-1", gpu_usd=100.0)
    assert budget.spend("proj-1")["gpu_usd"] == 0.0


def test_quota_entries_carry_the_current_project(workspace):
    make_project("proj-1", quota_tokens=1000.0)
    budget.set_current("proj-1")
    quota_log.record("main", input_tokens=100, output_tokens=50, role="research")

    assert budget.spend("proj-1")["quota_tokens"] == 150
    summary = quota_log.summarise()
    assert summary["by_project"]["proj-1"]["calls"] == 1
    assert summary["by_role"]["research"]["input_tokens"] == 100


def test_credits_are_attributed_too(workspace):
    make_project("proj-1", credits_usd=5.0)
    budget.set_current("proj-1")
    quota_log.record("funnel.rerank", unit="credits", credits_usd=0.25)
    assert budget.spend("proj-1")["credits_usd"] == 0.25


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_over_allocation_refuses_with_exit_12(workspace):
    """Exit 12, not 6. "'this research ran out of its allocation' is never
    confused with 'the machine is out of money'." """
    make_project("proj-1", gpu_usd=10.0)
    with pytest.raises(GateRefusal) as exc:
        budget.check("proj-1", gpu_usd=11.0)
    assert exc.value.exit_code == EXIT_PROJECT_BUDGET
    assert exc.value.exit_code != EXIT_SPEND
    assert "raise" in (exc.value.fix or "")


def test_spend_already_recorded_counts_against_the_ceiling(workspace):
    make_project("proj-1", gpu_usd=10.0)
    record_run("proj-1", usd=8.0)
    budget.check("proj-1", gpu_usd=1.5)  # fits
    with pytest.raises(GateRefusal):
        budget.check("proj-1", gpu_usd=3.0)  # 8 + 3 > 10


def test_a_resource_with_no_ceiling_is_tracked_not_bounded(workspace):
    make_project("proj-1", gpu_usd=10.0)  # no token ceiling
    budget.check("proj-1", quota_tokens=10**9)
    assert budget.status("proj-1")["resources"]["quota_tokens"]["remaining"] is None


def test_no_project_selected_is_not_an_overrun(workspace):
    assert budget.check(None, gpu_usd=10**6) is None
    assert budget.over_budget(None) == []
    assert budget.over_budget("nonexistent") == []


def test_raising_the_ceiling_clears_the_refusal(workspace):
    make_project("proj-1", gpu_usd=10.0)
    record_run("proj-1", usd=10.0)
    assert budget.over_budget("proj-1") == []
    record_run("proj-1", usd=1.0)
    assert budget.over_budget("proj-1") == ["gpu_usd"]

    budget.raise_ceiling("proj-1", budget={"gpu_usd": 50.0}, reason="approved")
    assert budget.over_budget("proj-1") == []


# ---------------------------------------------------------------------------
# integration with the §6 submit gates
# ---------------------------------------------------------------------------
def test_check_submit_enforces_the_project_ceiling(workspace):
    sub = make_submission(workspace, hours=1.0, rate=5.0)
    pass_preflight(sub)
    make_project("proj-1", gpu_usd=1.0)

    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, make_expectation(), _cfg(), project="proj-1")
    assert exc.value.exit_code == EXIT_PROJECT_BUDGET


def test_the_global_ceiling_still_fires_first(workspace):
    """A caller who has blown both should hear about the machine's ceiling
    first: that one stops every other project too."""
    sub = make_submission(workspace, hours=1000.0, rate=5.0)
    pass_preflight(sub)
    make_project("proj-1", gpu_usd=0.5)

    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, make_expectation(), _cfg(), project="proj-1")
    assert exc.value.exit_code == EXIT_SPEND


def test_a_project_ceiling_does_not_replace_the_global_one(workspace):
    """A generous project allocation must not raise the machine's ceiling."""
    sub = make_submission(workspace, hours=100.0, rate=10.0)  # $1000
    pass_preflight(sub)
    make_project("proj-1", gpu_usd=10_000.0)

    with pytest.raises(GateRefusal) as exc:
        gates.check_submit(sub, make_expectation(), _cfg(), project="proj-1")
    assert exc.value.exit_code == EXIT_SPEND


def test_submit_gates_pass_with_headroom_in_both(workspace):
    sub = make_submission(workspace, hours=1.0, rate=2.0)
    pass_preflight(sub)
    make_project("proj-1", gpu_usd=50.0)
    summary = gates.check_submit(sub, make_expectation(), _cfg(), project="proj-1")
    assert summary["project"] == "proj-1"
    # Nothing is recorded yet at gate time, so `remaining` is the state before
    # the job and `projected_remaining` is the state after it.
    assert summary["project_budget"]["remaining"] == 50.0
    assert summary["project_budget"]["projected_remaining"] == 48.0


def _cfg():
    from core import config

    return config.load(reload=True)


# ---------------------------------------------------------------------------
# the two token mechanisms
# ---------------------------------------------------------------------------
def test_the_hook_denies_cost_bearing_commands_when_over_budget(workspace):
    """"`hooks.py:pre_tool_use` denies cost-bearing Bash commands once the
    project is over budget." This is the token loop's only enforcement point."""
    import hooks

    make_project("proj-1", gpu_usd=1.0)
    budget.set_current("proj-1")

    assert hooks.evaluate_bash("python -m tools.jobs submit --spec s.toml --expect e --json") is None

    record_run("proj-1", usd=5.0)
    denial = hooks.evaluate_bash("python -m tools.jobs submit --spec s.toml --expect e --json")
    assert denial is not None
    assert "over budget" in denial.reason
    assert "tools.budget raise" in denial.suggestion


@pytest.mark.parametrize(
    "command",
    [
        "python -m tools.jobs submit --spec s.toml --expect e",
        "python -m tools.gpu submit --spec s.toml --expect e",
        "python -m tools.evolve run --task-dir d --expect e",
        "python -m tools.report write --project p",
    ],
)
def test_every_cost_bearing_command_is_covered(workspace, command):
    import hooks

    make_project("proj-1", gpu_usd=1.0)
    budget.set_current("proj-1")
    record_run("proj-1", usd=5.0)
    assert hooks.evaluate_bash(command) is not None


def test_reading_commands_are_never_denied_by_budget(workspace):
    """A ceiling must not stop you finding out what the spend bought."""
    import hooks

    make_project("proj-1", gpu_usd=1.0)
    budget.set_current("proj-1")
    record_run("proj-1", usd=5.0)

    for command in (
        "python -m tools.report draft --project proj-1 --json",
        "python -m tools.budget status --json",
        "python -m tools.jobs collect run-123 --json",
        "python -m tools.ledger query --pending --json",
        "pytest -q",
    ):
        assert hooks.evaluate_bash(command) is None, command


def test_the_hook_fails_open_when_the_ledger_is_unreadable(workspace, monkeypatch):
    """Accounting must never be the reason research stops."""
    import hooks

    monkeypatch.setattr(budget, "current_project", lambda: (_ for _ in ()).throw(OSError("disk")))
    assert hooks.evaluate_bash("python -m tools.jobs submit --spec s.toml") is None


def test_the_agent_refuses_the_next_turn_over_a_token_ceiling(workspace):
    """"token budgets are enforced to a granularity of one turn's overrun." The
    turn that crossed it finishes; the next one does not start."""
    import agent

    make_project("proj-1", quota_tokens=100.0)
    budget.set_current("proj-1")
    assert agent.check_turn_budget() is None

    quota_log.record("main", input_tokens=200, output_tokens=0, role="research")
    refusal = agent.check_turn_budget()
    assert refusal is not None
    assert refusal["overrun"] == 100
    assert "no way to refuse mid-turn" in refusal["message"]
    assert "tools.budget raise" in refusal["fix"]


def test_the_stop_hook_warns_before_it_blocks(workspace):
    import hooks

    make_project("proj-1", quota_tokens=1000.0)
    budget.set_current("proj-1")
    quota_log.record("main", input_tokens=800, output_tokens=0)

    warning = hooks.budget_warning()
    assert warning is not None
    assert warning["threshold"] == 0.75
    assert warning["resource"] == "quota_tokens"
    assert "Cost-bearing commands are now denied" not in warning["message"]


def test_the_stop_hook_reports_the_resource_nearest_its_ceiling(workspace):
    import hooks

    make_project("proj-1", quota_tokens=1000.0, gpu_usd=100.0)
    budget.set_current("proj-1")
    quota_log.record("main", input_tokens=990, output_tokens=0)
    record_run("proj-1", usd=80.0)

    warning = hooks.budget_warning()
    assert warning["resource"] == "quota_tokens"


def test_the_stop_hook_is_silent_with_no_project(workspace):
    import hooks

    assert hooks.budget_warning() is None


# ---------------------------------------------------------------------------
# payer -> HF namespace (§17's dependency on §15)
# ---------------------------------------------------------------------------
def test_the_payer_becomes_the_hf_namespace(workspace):
    """"the org attribution in §17 is a consequence of choosing a project rather
    than a separate flag to forget." """
    budget.create("proj-1", title="t", budget={}, payer="hf:myorg")
    assert budget.hf_namespace("proj-1") == "myorg"


def test_a_non_hf_payer_yields_no_namespace(workspace):
    budget.create("proj-1", title="t", budget={}, payer="lab-account")
    assert budget.hf_namespace("proj-1") is None
    assert budget.hf_namespace(None) is None
