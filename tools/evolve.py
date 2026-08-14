"""grad-evolve -- evolutionary search over ShinkaEvolve (HANDOFF-2 §21).

    "An evolutionary loop is a machine for spending money with no human in it."

That sentence is why this file's first job is a gate and its second is a search.
`check_spend` alone *would* stop a runaway campaign -- at generation 40,
abandoning an in-flight run that then goes stale and blocks every future
submission through the §6 gate. Succeeding at the search would brick the system.
So there is a **campaign budget gate**: before generation 0, refuse unless
`estimate_per_candidate x max_candidates` fits under the project's remaining
allocation, and re-check before every generation. Shinka's own `max_api_costs`
covers the LLM side; the compute side is the expensive half and Grad owns it.

**Phase 1 is local only, and that is not a placeholder.** Shinka needs no GPU
for many tasks and its headless example is API-free, so a campaign evaluated
entirely through local subprocesses proves the campaign records, the sub-run
bookkeeping, and the budget integration while the blast radius is zero. Doing
the ledger work and the spend work simultaneously against live GPU jobs is how
you learn about exit 7 the hard way. `--remote` is refused here until phase 2.

**Driver, not fork.** Shinka exposes `EvolutionConfig`, `LocalJobConfig`,
`DatabaseConfig`, `ShinkaEvolveRunner(...).run()`. What we need is gating and
ledger integration *around* the loop, which is a driver. §23 item 1 asks whether
Shinka exposes a per-candidate callback, because that decides driver-vs-fork:
`mutator_capabilities()` below answers it against the installed package rather
than against a document, and the answer is reported in `run`'s output. Until a
hook point proves insufficient, a fork is a maintenance cost to defer.

**Models.** Per §16, the default is an *ensemble* -- Sonnet 5 primary plus Haiku
4.5 as a cheap explorer -- because Shinka's design is explicitly an ensemble of
LLMs acting as mutation operators, and collapsing to a single model discards
diversity the algorithm is built around. Shinka's bandit allocates between them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core import (
    budget,
    campaign as camp,
    config as config_mod,
    ledger_store as ls,
    paths,
    quota_log,
)
from core.cli import Cli, main
from core.errors import (
    EXIT_CHECK_FAILED,
    EXIT_PROJECT_BUDGET,
    ConfigError,
    GateRefusal,
    GradError,
    NotFound,
    UsageError,
)

cli = Cli(
    "grad-evolve",
    "Run an evolutionary search as a budgeted campaign, with candidates recorded "
    "as sub-runs and only promoted winners entering the ledger.",
    epilog=(
        "The campaign, not the candidate, is the unit of prediction: --expect binds one\n"
        "relational claim ('the evolved variant beats baseline X on Y by >= Z') and the\n"
        "candidates are exempt from the per-run expectation gate.\n\n"
        "Before generation 0 and before every generation after it, the projected cost of\n"
        "the remaining candidates must fit under the project's allocation. Exit 12 when\n"
        "it does not -- that is a project budget refusal, not the machine's ceiling.\n\n"
        "Phase 1 is local only. --remote is refused: doing the ledger work and the spend\n"
        "work simultaneously against live GPU jobs is how you learn about exit 7 the\n"
        "hard way."
    ),
)

DEFAULT_MODELS = ("claude-sonnet-5", "claude-haiku-4-5")
STAGE_EVOLVE = "evolve.mutate"


# ---------------------------------------------------------------------------
# the Shinka boundary
# ---------------------------------------------------------------------------
def _shinka() -> Any:
    try:
        import shinka  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "shinka-evolve is not installed, so there is no mutation engine",
            fix="pip install -e '.[evolve]'",
        ) from exc
    return shinka


def mutator_capabilities() -> dict[str, Any]:
    """Answer §23 item 1 against the installed package, not against a document.

    "Not verified in session: whether Shinka exposes a per-candidate callback.
     Check before starting; it decides driver-vs-fork."

    A per-candidate hook would let budget be charged inside the generation loop.
    Without one, generation boundaries are the finest granularity available --
    which is what this driver is built around, so its absence is a documented
    limit rather than a blocker.
    """
    try:
        shinka = _shinka()
    except ConfigError as exc:
        return {"installed": False, "reason": exc.message}

    import inspect  # noqa: PLC0415

    runner = getattr(shinka, "ShinkaEvolveRunner", None)
    hooks: list[str] = []
    methods: list[str] = []
    if runner is not None:
        try:
            params = inspect.signature(runner.__init__).parameters
        except (TypeError, ValueError):
            params = {}
        hooks = [
            name
            for name in params
            if any(word in name for word in ("callback", "hook", "on_", "listener"))
        ]
        methods = sorted(
            n for n in dir(runner) if not n.startswith("_") and callable(getattr(runner, n, None))
        )

    per_generation = next(
        (n for n in ShinkaMutator.PROPOSE_METHODS if n in methods), None
    )
    if hooks:
        granularity, note = "candidate", (
            "a per-candidate hook exists; finer budget charging is possible without a fork"
        )
    elif per_generation:
        granularity, note = "generation", (
            f"no per-candidate hook, but `{per_generation}()` yields one generation at a "
            "time, so the budget is re-checked at generation boundaries. Driver, not fork."
        )
    else:
        granularity, note = "campaign", (
            "no per-candidate hook and no per-generation entry point -- the runner exposes "
            "only whole-loop methods, which own the loop this driver needs to interrupt. "
            "This is the evidence §21 said a fork should wait for: `evolve run` refuses "
            "rather than handing control away with the budget unchecked."
        )

    return {
        "installed": True,
        "version": getattr(shinka, "__version__", None),
        "runner": runner is not None,
        "per_candidate_hooks": hooks,
        "per_generation_method": per_generation,
        "runner_methods": methods[:20],
        "granularity": granularity,
        "driver_viable": bool(hooks or per_generation),
        "note": note,
    }


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
_INITIAL_TEMPLATE = '''"""The program being evolved.

Everything between the EVOLVE-BLOCK markers is mutable; everything outside is
not. That boundary is not decoration -- `tools/evolve.py` checks it
mechanically, and a mutation that stays inside it needs only the two local
preflight checks, while one that escapes requires a fresh remote smoke run.
Keeping imports, I/O, and the entry point outside the block is what makes a
campaign affordable.
"""

import json


# EVOLVE-BLOCK-START
def solve(x: float) -> float:
    """The thing being searched over. Mutate freely."""
    return x * 2.0
# EVOLVE-BLOCK-END


def main() -> None:
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
'''

_EVALUATE_TEMPLATE = '''"""Score one candidate.

Contract, matching Shinka's: print ONE JSON object of metrics to stdout, and it
must contain `combined_score`. Everything else in the object is recorded
alongside it and is what makes the Goodhart failure visible -- a search
optimising a scalar will find the bug in the metric, so record the components
that scalar was built from.
"""

import json

import initial


def evaluate() -> dict:
    # Replace with the real objective.
    error = sum(abs(initial.solve(x) - (x * 2.0)) for x in range(10))
    return {
        "combined_score": -error,
        "abs_error": error,
        "n": 10,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate()))
'''


def _init_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task-dir", required=True, help="directory to scaffold, e.g. pipeline/evolve-lr")
    p.add_argument("--force", action="store_true", help="overwrite existing files")


@cli.command("init", "scaffold a task directory with the evolve-block contract", setup=_init_args)
def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    """`initial.py` with EVOLVE-BLOCK markers, `evaluate.py` returning
    `combined_score`. Shinka's contract and Grad's submission spec describe the
    same object, which is why this is an extension of §6/§7 rather than a
    bolt-on."""
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = paths.root() / task_dir
    task_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, body in (("initial.py", _INITIAL_TEMPLATE), ("evaluate.py", _EVALUATE_TEMPLATE)):
        target = task_dir / name
        if target.exists() and not args.force:
            continue
        target.write_text(body, encoding="utf-8")
        written.append(str(target))

    return {
        "task_dir": str(task_dir),
        "written": written,
        "skipped": [] if args.force else [
            str(task_dir / n) for n in ("initial.py", "evaluate.py")
            if str(task_dir / n) not in written
        ],
        "contract": {
            "initial.py": f"mutable region between {camp.BLOCK_START} and {camp.BLOCK_END}",
            "evaluate.py": "prints one JSON object of metrics including combined_score",
        },
        "next": (
            "python -m tools.ledger expect --task <task> --quantity combined_score "
            "--direction increase --claim 'the evolved variant beats baseline' --json"
        ),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task-dir", required=True)
    p.add_argument("--expect", required=True, help="campaign-level expectation id (§7, §21)")
    p.add_argument("--project", help="project whose allocation bounds this campaign")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population", type=int, default=4, help="candidates per generation")
    p.add_argument(
        "--estimate-per-candidate-usd",
        type=float,
        default=0.0,
        help="what one evaluation is expected to cost. The campaign gate multiplies it.",
    )
    p.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="evaluate locally (phase 1; the only supported mode)",
    )
    p.add_argument("--remote", action="store_true", help="refused: phase 2, behind the campaign gate")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="passed through to Shinka, e.g. --set evo.llm_models=...",
    )
    p.add_argument("--timeout-s", type=int, default=600, help="wall clock per candidate evaluation")


@cli.command("run", "run a budgeted campaign", setup=_run_args)
def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.remote:
        raise UsageError(
            "--remote is phase 2 and is not enabled: the campaign budget gate must be "
            "proven against zero-blast-radius local evaluation first. "
            "Do not run a single remote generation before that.",
            fix="drop --remote; a local campaign exercises the same records and the same gate",
        )

    cfg = config_mod.load()
    paths.ensure_workspace()
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = paths.root() / task_dir
    initial = task_dir / "initial.py"
    evaluate = task_dir / "evaluate.py"
    for path in (initial, evaluate):
        if not path.is_file():
            raise NotFound(
                f"{path} does not exist",
                fix=f"python -m tools.evolve init --task-dir {args.task_dir} --json",
            )

    baseline_source = initial.read_text(encoding="utf-8")
    if not camp.has_markers(baseline_source):
        raise UsageError(
            f"{initial} has no {camp.BLOCK_START}/{camp.BLOCK_END} markers, so no mutation "
            "can be checked for escaping the mutable region -- which means every candidate "
            "would need a paid remote smoke run",
            fix=f"wrap the mutable region in {camp.BLOCK_START} / {camp.BLOCK_END} comments",
        )

    # The campaign is the unit of prediction (§21 collision 2). The expectation
    # is bound here, once, and the candidates below are exempt from the per-run
    # gate precisely because this binding exists.
    expectation = _bind_expectation(args.expect)

    project_id = budget.resolve(args.project)
    max_candidates = max(1, args.generations) * max(1, args.population)
    projected = float(args.estimate_per_candidate_usd) * max_candidates

    # THE gate. Before generation 0, not after generation 40.
    _campaign_gate(project_id, projected, max_candidates, args.estimate_per_candidate_usd)

    campaign_id = camp.new_id("camp")
    record = {
        "type": camp.T_CAMPAIGN,
        "id": campaign_id,
        "created_at": camp.now_iso(),
        "task_dir": str(task_dir),
        "project": project_id or budget.UNASSIGNED,
        "expectation_id": args.expect,
        "quantity": expectation.get("quantity"),
        "generations": args.generations,
        "population": args.population,
        "max_candidates": max_candidates,
        "estimate_per_candidate_usd": args.estimate_per_candidate_usd,
        "projected_cost_usd": round(projected, 4),
        "mode": "local",
        "models": list(_models(cfg, args.overrides)),
        "mutator": mutator_capabilities(),
        "status": "open",
    }
    camp.append_campaign(record)

    result = _drive(
        campaign_id=campaign_id,
        task_dir=task_dir,
        baseline_source=baseline_source,
        generations=args.generations,
        population=args.population,
        project_id=project_id,
        per_candidate=args.estimate_per_candidate_usd,
        timeout_s=args.timeout_s,
        overrides=args.overrides,
        cfg=cfg,
    )

    camp.close_campaign(campaign_id, status=result["status"], reason=result.get("reason", ""))
    return {
        "campaign": campaign_id,
        "expectation_id": args.expect,
        "project": project_id,
        **result,
        "next": f"python -m tools.evolve status --campaign {campaign_id} --json",
    }


def _bind_expectation(expectation_id: str) -> dict[str, Any]:
    """The campaign's one prediction, and the same uniqueness rule as a run.

    §7's argument applies unchanged: an expectation that can be reused is an
    expectation that can be authored after the fact.
    """
    try:
        expectation = ls.expectation(expectation_id)
    except GradError:
        raise GateRefusal(
            "expectation_missing",
            f"expectation {expectation_id!r} does not exist",
            5,
            fix=(
                "python -m tools.ledger expect --task <task> --quantity combined_score "
                "--direction increase --claim 'the evolved variant beats baseline X' --json"
            ),
        ) from None
    bound = {
        c.get("expectation_id")
        for c in camp.campaigns().values()
        if c.get("expectation_id")
    } | ls.bound_expectation_ids()
    if expectation_id in bound:
        raise GateRefusal(
            "expectation_bound",
            f"expectation {expectation_id!r} is already bound to a run or campaign",
            5,
            fix="mint a new expectation for this campaign",
        )
    return expectation


def _campaign_gate(
    project_id: str | None, projected: float, max_candidates: int, per_candidate: float
) -> None:
    """Before generation 0. This is the one that matters.

    Without it, `check_spend` stops the campaign mid-flight -- at generation 40,
    abandoning an in-flight run that goes stale and blocks every future
    submission (exit 7). Succeeding at the search would brick the system.
    """
    if per_candidate <= 0:
        # Not an error: a genuinely free local campaign is the phase-1 case. But
        # an unpriced campaign cannot be gated, and saying so beats implying it
        # was checked.
        return
    budget.check(
        project_id,
        gpu_usd=projected,
        what=f"a campaign of {max_candidates} candidates at ${per_candidate:.2f} each",
    )


def _models(cfg: config_mod.Config, overrides: list[str]) -> tuple[str, ...]:
    """Sonnet 5 primary plus Haiku 4.5 explorer, unless overridden.

    Shinka's design is an ensemble of LLMs acting as mutation operators;
    collapsing to a single model discards diversity the algorithm is built
    around. `--set evo.llm_models=...` is Shinka's own mechanism and wins.
    """
    for override in overrides:
        if override.startswith("evo.llm_models="):
            value = override.split("=", 1)[1]
            return tuple(m.strip() for m in value.split(",") if m.strip())
    primary = cfg.model_for("evolve")
    return (primary, "claude-haiku-4-5") if primary not in DEFAULT_MODELS[1:] else (primary,)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def _drive(
    *,
    campaign_id: str,
    task_dir: Path,
    baseline_source: str,
    generations: int,
    population: int,
    project_id: str | None,
    per_candidate: float,
    timeout_s: int,
    overrides: list[str],
    cfg: config_mod.Config,
    mutator: Any = None,
) -> dict[str, Any]:
    """Generation by generation, with the gate between each.

    `mutator` is injectable so the driver can be tested against a faked Shinka
    runner -- §24's testing note asks for exactly that, and a campaign loop
    tested only against the real thing is a campaign loop tested never.
    """
    mutator = mutator or _make_mutator(task_dir, overrides, cfg)
    evaluated = 0
    best: dict[str, Any] | None = None
    status = "closed"
    reason = ""

    for generation in range(generations):
        # Checked at the generation boundary, alongside the gate and for the
        # same reason: this is the point where the campaign can end with every
        # candidate collected rather than one abandoned in flight.
        if camp.halt_requested(campaign_id):
            status = "halted"
            reason = "halt requested"
            camp.record_generation(campaign_id, generation, halted=True, reason=reason)
            break

        remaining = (generations - generation) * population
        try:
            _campaign_gate(project_id, per_candidate * remaining, remaining, per_candidate)
        except GateRefusal as exc:
            # Stopping here is the success case for the gate: the campaign ends
            # cleanly at a generation boundary with every candidate collected,
            # rather than being killed mid-flight with a run left in flight.
            status = "exhausted"
            reason = exc.message
            camp.record_generation(
                campaign_id, generation, halted=True, reason=exc.message, code=EXIT_PROJECT_BUDGET
            )
            break

        started = time.time()
        try:
            proposals = mutator.propose(generation=generation, population=population, best=best)
        except GradError:
            raise
        except Exception as exc:  # noqa: BLE001 - a mutation engine failure ends the campaign, not the process
            status = "failed"
            reason = f"the mutation engine failed at generation {generation}: {exc}"
            camp.record_generation(campaign_id, generation, error=str(exc))
            break

        scores = []
        for index, source in enumerate(proposals):
            candidate = _evaluate_candidate(
                campaign_id=campaign_id,
                generation=generation,
                index=index,
                source=source,
                baseline_source=baseline_source,
                task_dir=task_dir,
                timeout_s=timeout_s,
                per_candidate=per_candidate,
            )
            evaluated += 1
            score = (candidate.get("metrics") or {}).get("combined_score")
            if isinstance(score, (int, float)):
                scores.append(score)
                if best is None or score > best["score"]:
                    best = {"score": score, "source": source, "candidate_id": candidate["candidate_id"]}

        camp.record_generation(
            campaign_id,
            generation,
            candidates=len(proposals),
            evaluated=len(scores),
            best_score=max(scores) if scores else None,
            duration_s=round(time.time() - started, 2),
        )

    spend = camp.campaign_spend(campaign_id)
    return {
        "status": status,
        "reason": reason,
        "generations_run": min(generations, evaluated // max(1, population)),
        "candidates_evaluated": evaluated,
        "spend": spend,
        # Top-K, not the argmax. A search optimising a scalar finds the bug in
        # the metric, so the shape of the leaderboard is part of the output.
        "top": [
            {
                "candidate_id": c["candidate_id"],
                "generation": c["generation"],
                "metrics": c["metrics"],
                "escaped_block": c.get("escaped_block", {}).get("escaped", False),
            }
            for c in camp.top_k(campaign_id, 5)
        ],
        "goodhart_note": (
            "top-K, not the argmax. A winner is not a result until it goes through "
            "`evolve promote` and the normal verdict path."
        ),
    }


def _evaluate_candidate(
    *,
    campaign_id: str,
    generation: int,
    index: int,
    source: str,
    baseline_source: str,
    task_dir: Path,
    timeout_s: int,
    per_candidate: float,
) -> dict[str, Any]:
    """Evaluate one candidate locally and record it as a sub-run.

    Candidates go to `ledger/candidates.jsonl`, never to `runs.jsonl`: a
    100-generation campaign is thousands of rows and would dominate a ledger
    meant to be read by hand (§23 item 4). Only a promoted candidate becomes a
    run.
    """
    candidate_id = f"{campaign_id}-g{generation}-c{index}"
    escaped = camp.escaped_evolve_block(baseline_source, source)

    workdir = paths.run_artifacts(candidate_id)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "initial.py").write_text(source, encoding="utf-8")
    evaluate_src = (task_dir / "evaluate.py").read_text(encoding="utf-8")
    (workdir / "evaluate.py").write_text(evaluate_src, encoding="utf-8")

    started = time.time()
    record: dict[str, Any] = {
        "campaign": campaign_id,
        "candidate_id": candidate_id,
        "generation": generation,
        "index": index,
        "at": camp.now_iso(),
        "escaped_block": escaped,
        "workdir": str(workdir),
        "cost_usd": per_candidate,
    }

    if escaped["escaped"]:
        # §21 collision 3: a mutation outside the block changed code the
        # baseline's smoke result no longer covers. Recorded, not evaluated --
        # the alternative is a paid remote smoke run per candidate, which is the
        # cost this whole mechanism exists to avoid.
        record.update(
            {
                "skipped": True,
                "error": "mutation escaped the evolve block; needs a fresh smoke run",
                "duration_s": round(time.time() - started, 3),
                # It never ran, so it cost nothing. Charging the per-candidate
                # estimate anyway would consume the project's allocation for
                # work that was declined, and a campaign that mostly escapes
                # would exhaust its budget having evaluated almost nothing.
                "cost_usd": 0.0,
            }
        )
        camp.append_candidate(record)
        return record

    try:
        proc = subprocess.run(
            [sys.executable, "evaluate.py"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        output = (proc.stdout or "").strip()
        stderr = (proc.stderr or "")[-4000:]
        metrics: Any = None
        problem: str | None = None
        if proc.returncode != 0:
            problem = f"evaluate.py exited {proc.returncode}"
        else:
            try:
                metrics = json.loads(output.splitlines()[-1]) if output else None
            except (json.JSONDecodeError, IndexError):
                problem = "evaluate.py did not print a JSON object of metrics"
            if problem is None:
                problem = camp.validate_metrics(metrics)
        (workdir / "evaluate.log").write_text(output + "\n" + stderr, encoding="utf-8")
        record.update(
            {
                "metrics": metrics if problem is None else None,
                "error": problem,
                "duration_s": round(time.time() - started, 3),
            }
        )
    except subprocess.TimeoutExpired:
        record.update(
            {
                "metrics": None,
                "error": f"evaluate.py exceeded {timeout_s}s",
                "duration_s": round(time.time() - started, 3),
            }
        )

    camp.append_candidate(record)
    return record


# ---------------------------------------------------------------------------
# mutation engines
# ---------------------------------------------------------------------------
class ShinkaMutator:
    """Thin driver over Shinka's Python API.

    Deliberately thin: everything worth owning -- the gate, the ledger records,
    the escape check -- is outside it, so replacing this class is a small change
    and forking Shinka remains a decision that can be deferred.
    """

    def __init__(self, task_dir: Path, models: tuple[str, ...], overrides: list[str]) -> None:
        self.task_dir = task_dir
        self.models = models
        self.overrides = overrides
        self._runner: Any = None
        self._method: str = ""

    # A per-generation entry point, under any of the names an upstream release
    # might plausibly use. `run` and `run_async` are deliberately NOT here: they
    # drive the whole campaign themselves, which is precisely the control this
    # driver needs to keep in order to charge the budget between generations.
    PROPOSE_METHODS = ("propose", "propose_generation", "step", "ask")

    def _build(self) -> Any:
        shinka = _shinka()
        try:
            evolution = shinka.EvolutionConfig(
                llm_models=list(self.models),
                init_program_path=str(self.task_dir / "initial.py"),
            )
            job = shinka.LocalJobConfig(eval_program_path=str(self.task_dir / "evaluate.py"))
            database = shinka.DatabaseConfig()
            runner = shinka.ShinkaEvolveRunner(
                evo_config=evolution, job_config=job, db_config=database
            )
        except (AttributeError, TypeError) as exc:
            raise ConfigError(
                f"the installed shinka-evolve does not match the expected API: {exc}",
                fix=(
                    "check the constructor against the installed version "
                    "(`python -m tools.docs signature shinka ShinkaEvolveRunner --json`), "
                    "then adjust ShinkaMutator"
                ),
            ) from exc

        method = self._propose_method(runner)
        if method is None:
            # This is §23 item 1 answered at runtime, and it is the evidence §21
            # said a fork should wait for: `ShinkaEvolveRunner` exposes `run` and
            # `run_async`, both of which own the whole loop. A driver cannot
            # charge the budget between generations through an API that only
            # offers "run everything", so it refuses rather than either handing
            # control away or calling a method that does not exist.
            available = sorted(
                n for n in dir(runner) if not n.startswith("_") and callable(getattr(runner, n, None))
            )
            raise ConfigError(
                "the installed ShinkaEvolveRunner exposes no per-generation entry point, "
                "only whole-loop methods, so the campaign budget could not be re-checked "
                "between generations. HANDOFF-2 §21 defers a fork until exactly this "
                "evidence appears; this is it. "
                f"Available: {', '.join(available[:12]) or '(none)'}",
                fix=(
                    "run the campaign against a driver you control -- or fork Shinka to "
                    "expose one generation at a time, which is the point at which §21's "
                    "'driver, not fork' decision flips. "
                    "`python -m tools.evolve capabilities --json` reports what was found."
                ),
            )
        self._method = method
        return runner

    @classmethod
    def _propose_method(cls, runner: Any) -> str | None:
        for name in cls.PROPOSE_METHODS:
            if callable(getattr(runner, name, None)):
                return name
        return None

    def propose(self, *, generation: int, population: int, best: dict[str, Any] | None) -> list[str]:
        if self._runner is None:
            self._runner = self._build()
        proposals = getattr(self._runner, self._method)(
            generation=generation, population=population, parent=(best or {}).get("source")
        )
        quota_log.record(
            STAGE_EVOLVE,
            role="evolve",
            model=",".join(self.models),
            detail={"generation": generation, "population": population, "method": self._method},
        )
        return [str(p) for p in proposals]


def _make_mutator(task_dir: Path, overrides: list[str], cfg: config_mod.Config) -> Any:
    return ShinkaMutator(task_dir, _models(cfg, overrides), overrides)


# ---------------------------------------------------------------------------
# status / promote
# ---------------------------------------------------------------------------
def _status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--campaign", help="one campaign; omit to list them all")
    p.add_argument("--top", type=int, default=5, help="how many leaders to show")


@cli.command("status", "campaigns, their spend, and their top-K", setup=_status_args)
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    if not args.campaign:
        return {
            "campaigns": [
                {
                    "id": cid,
                    "status": c.get("status"),
                    "task_dir": c.get("task_dir"),
                    "project": c.get("project"),
                    "generations_run": c.get("generations_run"),
                    "spend": camp.campaign_spend(cid),
                }
                for cid, c in camp.campaigns().items()
            ]
        }
    record = camp.campaign(args.campaign)
    return {
        "campaign": record,
        "spend": camp.campaign_spend(args.campaign),
        "top": camp.top_k(args.campaign, args.top),
        "note": (
            "top-K rather than the argmax, on purpose: a search optimising a scalar "
            "will find the bug in the metric"
        ),
    }


def _halt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--campaign", required=True)
    p.add_argument("--reason", default="", help="recorded on the halt event")


@cli.command("halt", "stop a campaign at the next generation boundary", setup=_halt_args)
def cmd_halt(args: argparse.Namespace) -> dict[str, Any]:
    """Ask a running campaign to stop cleanly.

    Not a kill. `evolve run` holds the loop in whatever process started it, so
    this writes a request the loop reads between generations -- which is also
    the only boundary at which stopping is safe, because every candidate is
    collected there. A campaign that is already closed is reported as such
    rather than being handed a request nothing will ever read.
    """
    record = camp.campaign(args.campaign)
    if record.get("status") != "open":
        return _already_closed(args.campaign, record.get("status"))
    try:
        camp.request_halt(args.campaign, reason=args.reason)
    except GradError as exc:
        # The loop closed the campaign between the check above and the append.
        # That is the halt getting what it wanted a moment early, not a failure,
        # so it reports the same way as finding it closed in the first place.
        if exc.code != "campaign_not_open":
            raise
        return _already_closed(args.campaign, camp.campaign(args.campaign).get("status"))
    return {
        "campaign": args.campaign,
        "halted": True,
        "status": "open",
        "message": (
            "halt requested; the campaign stops before the next generation, with every "
            "candidate collected"
        ),
        "next": f"python -m tools.evolve status --campaign {args.campaign} --json",
    }


def _already_closed(campaign_id: str, status: Any) -> dict[str, Any]:
    return {
        "campaign": campaign_id,
        "halted": False,
        "status": status,
        "message": f"campaign {campaign_id} is already {status}",
    }


def _promote_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--campaign", required=True)
    p.add_argument("--candidate", required=True, help="candidate id, or its index within the campaign")
    p.add_argument("--into", help="path to write the promoted source to (defaults to the task dir)")


@cli.command("promote", "turn a candidate into a normal, judged run", setup=_promote_args)
def cmd_promote(args: argparse.Namespace) -> dict[str, Any]:
    """The Goodhart resolution.

    "The campaign winner goes through the normal verdict path before it counts
     as a result." Promotion writes the source out and hands you the ordinary
    preflight -> expect -> submit -> collect -> verdict cycle. It does not
    shortcut any of it, and it deliberately does not write a run record itself.
    """
    record = camp.campaign(args.campaign)
    rows = camp.candidates(args.campaign)
    match = next(
        (c for c in rows if c["candidate_id"] == args.candidate),
        None,
    ) or next(
        (c for c in rows if str(c.get("index")) == str(args.candidate)),
        None,
    )
    if match is None:
        raise NotFound(
            f"candidate {args.candidate!r} is not in campaign {args.campaign}",
            fix=f"python -m tools.evolve status --campaign {args.campaign} --json",
        )
    if not match.get("metrics"):
        raise GradError(
            "candidate_unevaluated",
            f"candidate {args.candidate} has no metrics: {match.get('error') or 'never evaluated'}",
            exit_code=EXIT_CHECK_FAILED,
            fix="promote a candidate that produced a combined_score",
        )

    source = Path(match["workdir"]) / "initial.py"
    destination = Path(args.into) if args.into else Path(record["task_dir"]) / "promoted.py"
    if not destination.is_absolute():
        destination = paths.root() / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "campaign": args.campaign,
        "candidate": match["candidate_id"],
        "metrics": match["metrics"],
        "written": str(destination),
        "escaped_block": match.get("escaped_block", {}).get("escaped", False),
        # No run record is written here. A promoted candidate re-enters the
        # system through the front door, with its own preflight and its own
        # prediction, because a number produced by a scalar-maximising search is
        # a hypothesis rather than a result.
        "next": [
            f"python -m tools.preflight run --spec <spec pointing at {destination.name}> --json",
            "python -m tools.ledger expect --task <task> --quantity <q> ... --json",
            "python -m tools.jobs submit --spec <spec> --expect <id> --json",
        ],
        "why": (
            "a campaign winner is not a result until it has been judged: the search "
            "optimised a scalar, and finding the bug in the metric is what that does best"
        ),
    }


@cli.command("capabilities", "what the installed ShinkaEvolve supports (§23 item 1)")
def cmd_capabilities(_: argparse.Namespace) -> dict[str, Any]:
    """Driver or fork? Answered against the installed package."""
    return mutator_capabilities()


if __name__ == "__main__":
    main(cli)
