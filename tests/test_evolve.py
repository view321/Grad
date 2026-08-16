"""Evolutionary search (HANDOFF-2 §21).

§24: "7 needs a faked Shinka runner." The mutation engine is the one part that
is genuinely someone else's; everything that makes a campaign *safe* -- the
budget gate, the sub-run bookkeeping, the evolve-block escape check, the
top-K-not-argmax discipline -- is ours, and all of it is tested here against a
real ledger with a fake mutator.

The test that matters most is `test_the_gate_stops_a_runaway_campaign_at_a_
generation_boundary`, because the failure it prevents is the one §21 calls
"the dangerous one": without it, `check_spend` stops the campaign at generation
40 by abandoning an in-flight run, which then goes stale and blocks every future
submission through the §6 gate. Succeeding at the search would brick the system.
"""

from __future__ import annotations

import argparse

import pytest

from core import (
    budget,
    campaign as camp,
    config as config_mod,
    evolution,
    ledger_store as ls,
    paths,
)
from core.errors import EXIT_PROJECT_BUDGET, GateRefusal, GradError, NotFound, UsageError
from tools import evolve


# ---------------------------------------------------------------------------
# the evolve block
# ---------------------------------------------------------------------------
BASELINE = """import json

# EVOLVE-BLOCK-START
def solve(x):
    return x * 2
# EVOLVE-BLOCK-END

def main():
    print(json.dumps({}))
"""


def test_a_mutation_inside_the_block_has_not_escaped():
    mutated = BASELINE.replace("return x * 2", "return x * 3 + 1")
    assert camp.escaped_evolve_block(BASELINE, mutated)["escaped"] is False


def test_a_mutation_outside_the_block_has_escaped():
    """"The EVOLVE-BLOCK markers make 'did it escape' mechanically checkable,
    which is convenient." It is also what keeps a campaign affordable."""
    mutated = BASELINE.replace("import json", "import json\nimport subprocess")
    result = camp.escaped_evolve_block(BASELINE, mutated)
    assert result["escaped"] is True
    assert result["requires"] == "smoke"


def test_whitespace_outside_the_block_is_not_an_escape():
    """"a check that fires spuriously is a check that gets argued around."""
    mutated = BASELINE.replace("def main():", "\ndef main():   ")
    assert camp.escaped_evolve_block(BASELINE, mutated)["escaped"] is False


def test_a_file_with_no_markers_is_entirely_outside_the_block():
    """The conservative reading: an unmarked file that changed needs a fresh
    smoke run rather than being assumed safe."""
    assert camp.has_markers("def f(): pass") is False
    assert camp.escaped_evolve_block("def f(): return 1", "def f(): return 2")["escaped"] is True


# ---------------------------------------------------------------------------
# the campaign budget gate
# ---------------------------------------------------------------------------
def scaffold(workspace, *, evaluate_body: str | None = None):
    task_dir = workspace / "pipeline" / "evolve-lr"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "initial.py").write_text(BASELINE, encoding="utf-8")
    (task_dir / "evaluate.py").write_text(
        evaluate_body
        or "import json\nprint(json.dumps({'combined_score': 1.5, 'abs_error': 0.2}))\n",
        encoding="utf-8",
    )
    return task_dir


def make_expectation(quantity="combined_score"):
    record = ls.append_expectation(
        {
            "id": ls.new_id("exp"),
            "task": "evolve-lr",
            "created_at": ls.now_iso(),
            "quantity": quantity,
            "claim": "the evolved variant beats the baseline",
            "predicted": {"low": None, "high": None, "direction": "increase"},
            "basis": [],
            "comparability": "",
            "confidence": "medium",
        }
    )
    return record["id"]


def run_args(task_dir, expectation_id, **overrides):
    base = dict(
        task_dir=str(task_dir), expect=expectation_id, project=None, generations=2,
        population=2, estimate_per_candidate_usd=0.0, local=True, remote=False,
        overrides=[], timeout_s=30, json=True,
        # The search knobs. `islands=1` and no migration by default so the tests
        # that are about the *gate* are not also about the selection policy;
        # `test_evolution.py` covers that half on its own.
        mutator=evolve.MUTATOR_CLAUDE, jobs=2, eval_jobs=1, islands=1,
        migrate_every=0, pressure=1.0, seed=1234,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class FakeMutator:
    """An operator stand-in: real mutations of the parent's block, no model.

    Returns the shape `core/mutate.py:propose` returns, because the driver's
    contract with the operator is that shape and a fake that returned bare
    strings would test a driver nobody has.
    """

    def __init__(self, *, escape_at=None, fail_at=None, duplicate_at=None):
        self.escape_at = escape_at
        self.fail_at = fail_at
        self.duplicate_at = duplicate_at
        self.calls: list[int] = []

    def propose(self, *, plans, baseline, history):
        generation = plans[0]["generation"] if plans else 0
        self.calls.append(generation)
        out = []
        for plan in plans:
            i = plan["index"]
            meta = {
                "patch_type": plan["patch_type"],
                "island": plan["island"],
                "index": i,
                "generation": generation,
                "parent_id": (plan.get("parent") or {}).get("candidate_id"),
                "mate_id": (plan.get("mate") or {}).get("candidate_id"),
                "rationale": "a fake mutation",
                "error": None,
            }
            if self.fail_at == generation and i == 0:
                out.append({**meta, "source": "", "error": "the operator produced no source"})
                continue
            if self.escape_at == generation and i == 0:
                out.append(
                    {**meta, "source": BASELINE.replace("import json", "import json\nimport os")}
                )
                continue
            # Unique per (generation, index), because the driver deduplicates
            # proposals against everything ever proposed: a naive
            # `generation + i` repeats across generations and the fake would
            # spend the campaign colliding with itself.
            factor = 2 if self.duplicate_at == generation else generation * 100 + i + 3
            out.append({**meta, "source": BASELINE.replace("return x * 2", f"return x * {factor}")})
        return out


def test_the_gate_refuses_before_generation_zero(workspace):
    """"Before generation 0, refuse unless estimate_per_candidate x
    max_candidates fits under the project's remaining allocation." """
    budget.create("proj-1", title="t", budget={"gpu_usd": 1.0})
    task_dir = scaffold(workspace)
    args = run_args(
        task_dir, make_expectation(), project="proj-1",
        generations=10, population=10, estimate_per_candidate_usd=1.0,
    )
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(args)
    assert exc.value.exit_code == EXIT_PROJECT_BUDGET
    # And nothing was started.
    assert camp.campaigns() == {}


def test_candidate_cost_counts_against_the_project(workspace, monkeypatch):
    """Candidates live outside runs.jsonl by design, so the ceiling has to reach
    into candidates.jsonl or a campaign is invisible to the budget bounding it."""
    budget.create("proj-1", title="t", budget={"gpu_usd": 100.0})
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    evolve.cmd_run(
        run_args(task_dir, make_expectation(), project="proj-1",
                 generations=2, population=2, estimate_per_candidate_usd=0.5)
    )
    spend = budget.spend("proj-1")
    assert spend["candidates"] == 4
    assert spend["gpu_usd"] == 2.0


def test_the_gate_stops_a_runaway_campaign_at_a_generation_boundary(workspace, monkeypatch):
    """The dangerous case, and the reason this gate exists at all.

    Here the allocation is consumed *during* the campaign by a concurrent
    submission, which is the realistic way headroom disappears mid-run. The
    campaign notices at the next generation boundary and stops cleanly, with
    every candidate collected and the campaign marked `exhausted`.

    The failure this replaces: being killed mid-flight at generation 40, leaving
    a run in flight that goes stale and then blocks *every* future submission
    through the §6 gate (exit 7). Succeeding at the search would brick the
    system.
    """
    budget.create("proj-1", title="t", budget={"gpu_usd": 10.0})
    task_dir = scaffold(workspace)

    class ConcurrentSpender(FakeMutator):
        """Something else eats the allocation after the first generation."""

        def propose(self, *, plans, baseline, history):
            if plans and plans[0]["generation"] == 1:
                ls.append_run_event(
                    {
                        "type": ls.T_RUN_SUBMITTED, "id": ls.new_id("run"),
                        "status": "in_flight", "submitted_at": ls.now_iso(),
                        "project": "proj-1", "estimate_usd": 9.0,
                    }
                )
            return super().propose(plans=plans, baseline=baseline, history=history)

    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: ConcurrentSpender())
    result = evolve.cmd_run(
        run_args(task_dir, make_expectation(), project="proj-1",
                 generations=5, population=2, estimate_per_candidate_usd=0.5)
    )

    assert result["status"] == "exhausted"
    assert "allocation" in result["reason"]
    # It ran, then stopped -- rather than refusing outright or running to the end.
    assert 0 < result["candidates_evaluated"] < 10
    # Nothing the campaign started is left in flight: it stopped at a boundary,
    # not mid-candidate.
    assert camp.campaign(result["campaign"])["status"] == "exhausted"
    assert all(r.get("metrics") or r.get("error") for r in camp.candidates(result["campaign"]))


def test_an_unpriced_campaign_says_so_rather_than_implying_a_check(workspace, monkeypatch):
    budget.create("proj-1", title="t", budget={"gpu_usd": 1.0})
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(
        run_args(task_dir, make_expectation(), project="proj-1", estimate_per_candidate_usd=0.0)
    )
    assert result["status"] == "closed"


# ---------------------------------------------------------------------------
# the campaign is the unit of prediction
# ---------------------------------------------------------------------------
def test_a_campaign_binds_exactly_one_expectation(workspace, monkeypatch):
    """§7's rule, unchanged: an expectation that can be reused is an expectation
    that can be authored after the fact."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    expectation_id = make_expectation()

    evolve.cmd_run(run_args(task_dir, expectation_id))
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(run_args(task_dir, expectation_id))
    assert exc.value.code == "expectation_bound"


def test_a_missing_expectation_refuses(workspace):
    task_dir = scaffold(workspace)
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(run_args(task_dir, "exp-does-not-exist"))
    assert exc.value.code == "expectation_missing"


def test_candidates_do_not_enter_runs_jsonl(workspace, monkeypatch):
    """§23 item 4: "a 100-generation campaign is thousands of rows". They live
    in candidates.jsonl and only a promoted one becomes a run."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation()))

    assert result["candidates_evaluated"] == 4
    assert len(camp.candidates(result["campaign"])) == 4
    assert ls.runs() == []
    assert camp.candidates_path().exists()


def test_candidates_are_exempt_from_the_per_run_expectation_gate(workspace, monkeypatch):
    """Four candidates, one expectation. That is the 1:N resolution."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation()))
    record = camp.campaign(result["campaign"])
    assert record["expectation_id"]
    assert len(camp.candidates(result["campaign"])) > 1


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def test_an_escaping_candidate_is_recorded_but_not_evaluated(workspace, monkeypatch):
    """"candidates run --only tests,dry_run -- both local, both fast. Smoke is
    required [...] whenever a mutation escapes the evolve-block." Evaluating it
    anyway would mean a paid remote smoke run per candidate."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator(escape_at=0))
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=1, population=2))

    rows = camp.candidates(result["campaign"])
    escaped = [r for r in rows if r["escaped_block"]["escaped"]]
    assert len(escaped) == 1
    assert escaped[0]["skipped"] is True
    assert "smoke" in escaped[0]["error"]
    assert escaped[0].get("metrics") is None


def test_metrics_without_combined_score_are_refused(workspace, monkeypatch):
    """"a candidate that silently reports no score is indistinguishable from one
    that scored zero, and the search would optimise toward the fallback."""
    task_dir = scaffold(
        workspace, evaluate_body="import json\nprint(json.dumps({'accuracy': 0.9}))\n"
    )
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=1, population=1))
    row = camp.candidates(result["campaign"])[0]
    assert row["metrics"] is None
    assert "combined_score" in row["error"]


def test_a_crashing_evaluator_is_recorded_not_fatal(workspace, monkeypatch):
    task_dir = scaffold(workspace, evaluate_body="raise SystemExit(3)\n")
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=1, population=1))
    assert result["status"] == "closed"
    assert camp.candidates(result["campaign"])[0]["error"]


def test_validate_metrics_rejects_a_boolean_score():
    assert camp.validate_metrics({"combined_score": True}) is not None
    assert camp.validate_metrics({"combined_score": 1.0}) is None


# ---------------------------------------------------------------------------
# Goodhart
# ---------------------------------------------------------------------------
def test_status_surfaces_top_k_not_the_argmax(workspace, monkeypatch):
    """"A search optimising a scalar will find the bug in the metric." """
    task_dir = scaffold(
        workspace,
        evaluate_body=(
            "import json, pathlib\n"
            "src = pathlib.Path('initial.py').read_text()\n"
            "score = float(src.split('return x * ')[1].split('\\n')[0])\n"
            "print(json.dumps({'combined_score': score}))\n"
        ),
    )
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=2, population=2))

    assert len(result["top"]) > 1
    scores = [t["metrics"]["combined_score"] for t in result["top"]]
    assert scores == sorted(scores, reverse=True)
    assert "not a result until" in result["goodhart_note"]


def test_promote_writes_the_source_but_no_run_record(workspace, monkeypatch):
    """"The campaign winner goes through the normal verdict path before it
    counts as a result." Promotion must not shortcut preflight or the ledger."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation()))
    best = result["top"][0]["candidate_id"]

    promoted = evolve.cmd_promote(
        argparse.Namespace(campaign=result["campaign"], candidate=best, into=None, json=True)
    )
    assert (task_dir / "promoted.py").exists()
    assert ls.runs() == [], "promotion must not write a run record"
    assert any("preflight" in step for step in promoted["next"])
    assert any("expect" in step for step in promoted["next"])


def test_promoting_an_unevaluated_candidate_refuses(workspace, monkeypatch):
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator(escape_at=0))
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=1, population=1))
    escaped = camp.candidates(result["campaign"])[0]

    with pytest.raises(GradError) as exc:
        evolve.cmd_promote(
            argparse.Namespace(campaign=result["campaign"], candidate=escaped["candidate_id"],
                               into=None, json=True)
        )
    assert exc.value.code == "candidate_unevaluated"


# ---------------------------------------------------------------------------
# phasing and scaffolding
# ---------------------------------------------------------------------------
def test_remote_is_refused_in_phase_one(workspace):
    """"Do not run a single remote generation before this exists." """
    task_dir = scaffold(workspace)
    with pytest.raises(UsageError) as exc:
        evolve.cmd_run(run_args(task_dir, make_expectation(), remote=True))
    assert "phase 2" in str(exc.value)


def test_a_task_without_markers_is_refused(workspace):
    task_dir = scaffold(workspace)
    (task_dir / "initial.py").write_text("def solve(x): return x\n", encoding="utf-8")
    with pytest.raises(UsageError) as exc:
        evolve.cmd_run(run_args(task_dir, make_expectation()))
    assert "EVOLVE-BLOCK" in str(exc.value)


def test_init_scaffolds_a_valid_task(workspace):
    result = evolve.cmd_init(
        argparse.Namespace(task_dir="pipeline/e", force=False, json=True)
    )
    assert len(result["written"]) == 3
    source = (paths.root() / "pipeline" / "e" / "initial.py").read_text(encoding="utf-8")
    assert camp.has_markers(source)
    assert "combined_score" in (paths.root() / "pipeline" / "e" / "evaluate.py").read_text(
        encoding="utf-8"
    )
    # The brief the operator is shown on every proposal. Scaffolded because a
    # campaign whose operator was told nothing about the task spends its first
    # generations inferring one.
    assert (paths.root() / "pipeline" / "e" / "TASK.md").exists()


def test_the_default_models_are_an_ensemble(workspace):
    """"collapsing to a single model discards diversity the algorithm is built
    around." Sonnet 5 primary plus Haiku 4.5 explorer."""
    cfg = config_mod.load(reload=True)
    models = evolve._models(cfg, [])
    assert "claude-sonnet-5" in models
    assert "claude-haiku-4-5" in models


def test_shinkas_own_override_mechanism_wins(workspace):
    cfg = config_mod.load(reload=True)
    models = evolve._models(cfg, ["evo.llm_models=a,b,c"])
    assert models == ("a", "b", "c")


def test_capabilities_reports_both_engines(workspace):
    """§23 item 1, answered against the installed packages rather than a
    document -- and no longer the deciding question, because the built-in
    operator does not need a hook point: the loop is ours."""
    report = evolve.mutator_capabilities()
    assert report["default"] == evolve.MUTATOR_CLAUDE
    assert set(report["mutators"]) == {evolve.MUTATOR_CLAUDE, evolve.MUTATOR_SHINKA}
    built_in = report["mutators"][evolve.MUTATOR_CLAUDE]
    assert built_in["granularity"] == "candidate"
    assert set(built_in["patch_types"]) == set(evolution.PATCH_TYPES)
    shinka = report["mutators"][evolve.MUTATOR_SHINKA]
    if not shinka["available"]:
        assert shinka.get("reason") or shinka.get("note")


# ---------------------------------------------------------------------------
# review fixes
# ---------------------------------------------------------------------------
def test_an_escaped_candidate_is_not_charged(workspace, monkeypatch):
    """It never ran, so it cost nothing.

    Charging the per-candidate estimate for declined work would let a campaign
    that mostly escapes the evolve block exhaust its allocation having evaluated
    almost nothing.
    """
    budget.create("proj-1", title="t", budget={"gpu_usd": 100.0})
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator(escape_at=0))
    result = evolve.cmd_run(
        run_args(task_dir, make_expectation(), project="proj-1",
                 generations=1, population=2, estimate_per_candidate_usd=0.5)
    )

    rows = camp.candidates(result["campaign"])
    escaped = [r for r in rows if r["escaped_block"]["escaped"]]
    evaluated = [r for r in rows if not r["escaped_block"]["escaped"]]
    assert len(escaped) == 1 and len(evaluated) == 1
    assert escaped[0]["cost_usd"] == 0.0
    assert evaluated[0]["cost_usd"] == 0.5
    # Only the candidate that actually ran reaches the project's allocation.
    assert budget.spend("proj-1")["gpu_usd"] == 0.5


def test_the_shinka_driver_refuses_a_whole_loop_only_runner(workspace):
    """§23 item 1, answered: `ShinkaEvolveRunner` exposes `run` and `run_async`,
    both of which own the loop this driver needs to interrupt between
    generations. Calling a `propose()` that does not exist would be an
    AttributeError mid-campaign; refusing up front is the honest form, and it is
    the evidence §21 said a fork should wait for.
    """
    class WholeLoopOnly:
        def run(self): ...
        def run_async(self): ...

    assert evolve.ShinkaMutator._propose_method(WholeLoopOnly()) is None


def test_the_shinka_driver_accepts_a_per_generation_entry_point(workspace):
    class Steppable:
        def run(self): ...
        def propose(self, **kw): ...

    assert evolve.ShinkaMutator._propose_method(Steppable()) == "propose"


def test_capabilities_names_the_granularity_it_found(workspace):
    shinka = evolve.mutator_capabilities()["mutators"][evolve.MUTATOR_SHINKA]
    if shinka["available"]:
        assert shinka["granularity"] in ("candidate", "generation")
    else:
        # Either not installed, or installed and whole-loop-only. Both refuse,
        # and both name the built-in operator as the way forward.
        assert "shinka" in (shinka.get("reason") or "").lower() or (
            shinka.get("granularity") == "campaign"
        )


# ---------------------------------------------------------------------------
# halting (the UI's ■ HALT, and the CLI behind it)
# ---------------------------------------------------------------------------
class HaltingMutator(FakeMutator):
    """Requests a halt from inside generation 0, the way a human would from the
    workspace while the loop is mid-generation."""

    def __init__(self, campaign_getter, **kwargs):
        super().__init__(**kwargs)
        self._campaign = campaign_getter

    def propose(self, *, plans, baseline, history):
        if plans and plans[0]["generation"] == 0:
            camp.request_halt(self._campaign(), reason="halted from the workspace")
        return super().propose(plans=plans, baseline=baseline, history=history)


def test_a_halt_stops_the_loop_at_the_next_generation_boundary(workspace, monkeypatch):
    """Not a kill. Stopping mid-generation would abandon an in-flight candidate,
    which goes stale and blocks every future submission -- so the check sits at
    the same boundary the budget gate stops at."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(
        evolve, "_make_mutator",
        lambda *a, **k: HaltingMutator(lambda: next(iter(camp.campaigns()))),
    )
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=5, population=2))

    assert result["status"] == "halted"
    # Generation 0 completed; generation 1 never started -- and the folded
    # campaign has to agree with the returned count, because the evolve window
    # reads the folded one for its title bar.
    assert result["generations_run"] == 1
    assert camp.campaign(result["campaign"])["generations_run"] == 1
    assert len(camp.candidates(result["campaign"])) == 2
    # And nothing is left half-evaluated.
    assert all(r.get("metrics") or r.get("error") for r in camp.candidates(result["campaign"]))
    assert camp.campaign(result["campaign"])["status"] == "halted"


def test_halt_is_a_request_the_ledger_carries(workspace, monkeypatch):
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation()))
    campaign_id = result["campaign"]

    # A closed campaign is told so rather than handed a request nothing reads.
    payload = evolve.cmd_halt(argparse.Namespace(campaign=campaign_id, reason="", json=True))
    assert payload["halted"] is False
    assert "already" in payload["message"]
    assert camp.halt_requested(campaign_id) is False


def test_halt_on_an_open_campaign_records_the_request(workspace):
    camp.append_campaign(
        {"type": camp.T_CAMPAIGN, "id": "camp-1", "status": "open", "at": camp.now_iso()}
    )
    payload = evolve.cmd_halt(argparse.Namespace(campaign="camp-1", reason="too slow", json=True))
    assert payload["halted"] is True
    assert camp.halt_requested("camp-1") is True
    assert camp.campaign("camp-1")["halt_reason"] == "too slow"


def test_halting_an_unknown_campaign_is_a_not_found(workspace):
    with pytest.raises(NotFound):
        evolve.cmd_halt(argparse.Namespace(campaign="nope", reason="", json=True))


def test_a_halt_request_is_validated_inside_the_append_lock(workspace):
    """The loop closing a campaign and a human halting it are two processes
    racing over one file, so a check made before the lock is a check the other
    process can win. Same backstop as `append_run_event`'s binding check."""
    with pytest.raises(NotFound):
        camp.request_halt("camp-nope")
    assert camp.campaign_events() == []

    camp.append_campaign(
        {"type": camp.T_CAMPAIGN, "id": "camp-1", "status": "open", "at": camp.now_iso()}
    )
    camp.close_campaign("camp-1", status="closed")
    before = len(camp.campaign_events())

    with pytest.raises(GradError) as exc:
        camp.request_halt("camp-1")
    assert exc.value.code == "campaign_not_open"
    # Rejected under the lock means nothing was written.
    assert len(camp.campaign_events()) == before
    assert camp.halt_requested("camp-1") is False


def test_losing_the_halt_race_reports_closed_rather_than_raising(workspace, monkeypatch):
    """The campaign closed between the CLI's check and the append. That is the
    halt getting what it wanted a moment early, not a failure."""
    camp.append_campaign(
        {"type": camp.T_CAMPAIGN, "id": "camp-1", "status": "open", "at": camp.now_iso()}
    )
    camp.close_campaign("camp-1", status="closed")

    real = camp.campaign
    calls = {"n": 0}

    def stale_first(campaign_id):
        # The pre-check sees a campaign that is still open; the precondition,
        # reading under the lock, sees the truth.
        calls["n"] += 1
        record = dict(real(campaign_id))
        if calls["n"] == 1:
            record["status"] = "open"
        return record

    monkeypatch.setattr(evolve.camp, "campaign", stale_first)
    payload = evolve.cmd_halt(argparse.Namespace(campaign="camp-1", reason="", json=True))
    assert payload["halted"] is False
    assert payload["status"] == "closed"
    assert camp.halt_requested("camp-1") is False


def test_a_boundary_record_does_not_count_as_a_generation_that_ran(workspace):
    """Both the budget gate and a halt write a generation record at the boundary
    they stop *before*. Counting it reported a campaign halted after generation
    0 as having run two -- which is the number the evolve window puts in its
    title bar."""
    camp.append_campaign(
        {"type": camp.T_CAMPAIGN, "id": "camp-1", "status": "open", "at": camp.now_iso()}
    )
    camp.record_generation("camp-1", 0)
    assert camp.campaign("camp-1")["generations_run"] == 1

    camp.record_generation("camp-1", 1, halted=True, reason="halt requested")
    assert camp.campaign("camp-1")["generations_run"] == 1
    # The boundary is still on the record; it is just not counted as work done.
    assert camp.campaign("camp-1")["generation_log"][-1]["halted"] is True


# ---------------------------------------------------------------------------
# the built-in operator: what the driver does with what it returns
# ---------------------------------------------------------------------------
def test_a_duplicate_proposal_is_recorded_and_not_evaluated(workspace, monkeypatch):
    """An operator shown the elites every generation proposes one of them back,
    and evaluation is the expensive half. A duplicate is one wasted candidate;
    silently evaluating it twice is one wasted GPU hour."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator(duplicate_at=1))
    result = evolve.cmd_run(
        run_args(task_dir, make_expectation(), generations=2, population=2)
    )
    rows = camp.candidates(result["campaign"])
    dupes = [r for r in rows if r.get("duplicate_of")]
    # Generation 1 proposes `x * 2` twice; the first is new, the second is not.
    assert len(dupes) == 1
    assert result["duplicates_rejected"] == 1
    assert dupes[0]["metrics"] is None
    assert dupes[0]["cost_usd"] == 0.0


def test_an_operator_failure_is_one_candidate_not_one_campaign(workspace, monkeypatch):
    """A turn that ended without calling the tool, or an edit that matched
    nothing. The generation is one proposal short and the campaign carries on."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator(fail_at=0))
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=2, population=2))

    assert result["status"] == "closed"
    rows = camp.candidates(result["campaign"])
    failed = [r for r in rows if r.get("skipped") and "no source" in (r.get("error") or "")]
    assert len(failed) == 1
    assert failed[0]["cost_usd"] == 0.0
    assert len([r for r in rows if r.get("metrics")]) == 3


def test_the_seed_is_recorded_so_a_campaign_can_be_replayed(workspace, monkeypatch):
    """The selection policy is deterministic in the seed by construction, which
    is worth nothing if the seed is not in the ledger."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), seed=None))
    assert isinstance(result["seed"], int)
    assert camp.campaign(result["campaign"])["seed"] == result["seed"]


def test_the_campaign_records_which_operator_ran(workspace, monkeypatch):
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation()))
    record = camp.campaign(result["campaign"])
    assert record["mutator"] == evolve.MUTATOR_CLAUDE
    assert record["model"] == config_mod.load(reload=True).model_for("evolve")


def test_the_bandit_is_rebuilt_from_the_records(workspace, monkeypatch):
    """Held in no process's memory, so a crash cannot lose it and a reader can
    check it. `status` reports the same numbers the loop used."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(run_args(task_dir, make_expectation(), generations=2, population=2))

    reported = {row["patch_type"]: row["pulls"] for row in result["bandit"]}
    assert sum(reported.values()) == 4
    status = evolve.cmd_status(
        argparse.Namespace(campaign=result["campaign"], top=5, json=True)
    )
    assert {r["patch_type"]: r["pulls"] for r in status["bandit"]} == reported


def test_islands_and_migration_are_recorded(workspace, monkeypatch):
    """Migration is an event rather than an edit: `candidates.jsonl` is
    append-only and a candidate's record is written once, when it is evaluated."""
    task_dir = scaffold(workspace)
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(
        run_args(task_dir, make_expectation(), generations=3, population=2,
                 islands=2, migrate_every=1)
    )
    rows = camp.candidates(result["campaign"])
    assert {evolution.island_of(r) for r in rows} == {0, 1}
    # A champion made eligible on its neighbour, without leaving its own island.
    migrants = [r for r in rows if r.get("migrated_to")]
    assert migrants, "migration should have copied at least one champion"
    for row in migrants:
        assert evolution.island_of(row) in evolution.islands_of(row)


def test_the_budget_boundary_record_is_not_counted_either(workspace, monkeypatch):
    """The same off-by-one existed on the pre-existing exhausted path."""
    budget.create("proj-1", title="t", budget={"gpu_usd": 10.0})
    task_dir = scaffold(workspace)

    class ConcurrentSpender(FakeMutator):
        def propose(self, *, plans, baseline, history):
            if plans and plans[0]["generation"] == 1:
                ls.append_run_event(
                    {"type": ls.T_RUN_SUBMITTED, "id": ls.new_id("run"), "status": "in_flight",
                     "submitted_at": ls.now_iso(), "project": "proj-1", "estimate_usd": 9.0}
                )
            return super().propose(plans=plans, baseline=baseline, history=history)

    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: ConcurrentSpender())
    result = evolve.cmd_run(
        run_args(task_dir, make_expectation(), project="proj-1",
                 generations=5, population=2, estimate_per_candidate_usd=0.5)
    )
    assert result["status"] == "exhausted"
    folded = camp.campaign(result["campaign"])
    assert folded["generations_run"] == result["generations_run"]
