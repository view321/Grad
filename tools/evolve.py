"""grad-evolve -- evolutionary search as a budgeted campaign (HANDOFF-2 §21).

    "An evolutionary loop is a machine for spending money with no human in it."

That sentence is why this file's first job is a gate and its second is a search.
`check_spend` alone *would* stop a runaway campaign -- at generation 40,
abandoning an in-flight run that then goes stale and blocks every future
submission through the §6 gate. Succeeding at the search would brick the system.
So there is a **campaign budget gate**: before generation 0, refuse unless
`estimate_per_candidate x max_candidates` fits under the project's remaining
allocation, and re-check before every generation.

**The mutation operator is ours now, and that is the change worth reading about.**
§21 said "driver, not fork", and deferred the decision until evidence appeared
that a driver was insufficient. Three pieces appeared:

1. `ShinkaEvolveRunner.run()` owns the generation loop, which is the boundary the
   budget gate needs. That was §23 item 1 and `capabilities` still reports it.
2. There is no generation boundary in it to take. The 0.0.7 runner is async --
   `max_proposal_jobs`, `max_evaluation_jobs` -- so proposals are in flight
   together and complete out of order. A fork exposing one generation at a time
   would be a fork that removes the concurrency.
3. Its subscription rail is `headless/claude`, which drives the Claude Code CLI
   through `npx`. Every mutation would be a model call `agent.drive_turn` never
   issued and `ledger/quota.jsonl` never saw. The token ceiling would have been
   blind to the one loop in this system that runs without a human -- which is the
   same failure as the ceiling that could see one per cent of the tokens, in the
   one place it matters most.

So `core/mutate.py` proposes through the Agent SDK, `core/evolution.py` decides
what to propose, and both are ours. **`--mutator shinka` still exists** and still
works the moment upstream grows a per-generation entry point: a path that already
works should not stop working because a better one arrived.

**Remote evaluation is phase 2, and the gate is what made it safe to enable.**
Phase 1 was local-only on purpose: a campaign evaluated through local
subprocesses proves the campaign records, the sub-run bookkeeping and the budget
integration while the blast radius is zero. Those are proven, so `--remote` now
puts candidates on real hardware -- behind a refusal that is stricter than the
one for an ordinary job. `--remote-spec` names a pipeline whose preflight must be
complete and passing *including the smoke run*, which is the only check that sees
the real driver stack, the real data path and the real per-device batch size. The
config's `[preflight] checks` list is deliberately not consulted: a machine
configured without `smoke` would otherwise let a loop with no human in it put
forty candidates on hardware nothing had ever run one step on.

**All three backends, because a candidate is a training run.** A mutation here
changes an architecture or an optimiser, so evaluating one takes minutes to
hours -- which is what makes a fresh container or kernel per candidate a
reasonable unit rather than an absurd one. Each backend owns its own adapter,
because they differ in the one thing that matters: how the mutated program
reaches the machine. `gpu.py` copies the pipeline directory to a host that stays
up; `kaggle.py` swaps one file inside the base64 payload already embedded in the
generated notebook; `jobs.py` has no upload step at all -- the pipeline is in the
image -- so the candidate rides in as a gzipped tar in one environment variable,
unpacked by a prelude in front of the command.

Every adapter bounds the work *where it runs*, not only where it is watched. A
poll that gives up ends the function; it does not end a detached training run,
and an abandoned candidate keeps holding the GPU that the next one is about to be
measured on -- which would make the next score a measurement of this one's
overrun.

**Kaggle gets a second gate, because the dollar gate cannot see it.** That
backend rations *hours*, so a campaign priced at zero passes `_campaign_gate`
unconditionally. `_kaggle_hours_gate` projects the campaign against the weekly
allowance before generation 0, and `core/kaggle_quota.py` folds candidate rows
beside runs so the hours are visible afterwards as well -- without that fold a
campaign would burn real GPU hours nothing could account for, and the first
symptom would be an ordinary submission refused.

A candidate still never becomes a run: no adapter writes a ledger row, the
campaign remains the ledgered unit, and its expectation remains the bound
prediction.

**Models.** Sonnet 5 by default, from `[models] evolve`. The ensemble Shinka
built its bandit around is replaced by a bandit over *patch types* -- `diff`,
`full`, `cross` -- which is the axis that turned out to matter: one model
proposing three kinds of change is more diversity per token than three models
proposing the same kind.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core import (
    budget,
    campaign as camp,
    config as config_mod,
    evolution,
    ledger_store as ls,
    mutate,
    paths,
    quota_log,
)
from core.cli import Cli, main
from core.submission import Submission
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
        "Mutations are proposed by Sonnet 5 through the Agent SDK, so every proposal is\n"
        "metered in ledger/quota.jsonl under the `evolve.mutate` stage and bounded by the\n"
        "project's token allocation. --mutator shinka switches to ShinkaEvolve, which\n"
        "refuses unless the installed release exposes a per-generation entry point.\n\n"
        "--remote {ssh|hf_jobs|kaggle|modal} --remote-spec <spec> evaluates every candidate\n"
        "real hardware. It refuses unless that spec's preflight is complete and passing --\n"
        "tests, dry run, and a real smoke run on that hardware -- because a search is a\n"
        "loop with no human in it and the environment it lands in has to be proven once,\n"
        "before generation 0, rather than rediscovered forty times.\n\n"
        "A Kaggle campaign is projected against the weekly accelerator allowance as well,\n"
        "since that backend rations hours and the dollar gate cannot see them."
    ),
)

MUTATOR_CLAUDE = "claude"
MUTATOR_SHINKA = "shinka"
STAGE_EVOLVE = quota_log.STAGE_EVOLVE

#: Where a candidate can be evaluated. The names are the `platform` strings the
#: run records already use, so a campaign's `backend` and a run's `platform` are
#: the same vocabulary rather than two spellings of one idea.
BACKEND_SSH = "ssh"
BACKEND_HF = "hf_jobs"
BACKEND_KAGGLE = "kaggle"
BACKEND_MODAL = "modal"
REMOTE_BACKENDS = (BACKEND_SSH, BACKEND_HF, BACKEND_KAGGLE, BACKEND_MODAL)

#: The checks a remote campaign's spec must have passed. Not `[preflight] checks`
#: from the config, and that is the point: a machine configured to skip `smoke`
#: would otherwise let a campaign put forty candidates on hardware nothing had
#: ever run one step on. The gate names what it needs.
REMOTE_REQUIRED_CHECKS = ("tests", "dry_run", "smoke")


# ---------------------------------------------------------------------------
# the Shinka boundary -- kept, no longer the default
# ---------------------------------------------------------------------------
def _shinka() -> Any:
    try:
        import shinka  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "shinka-evolve is not installed, so --mutator shinka has no engine",
            fix="pip install -e '.[evolve]'   # or drop --mutator shinka to use the built-in operator",
        ) from exc
    return shinka


def mutator_capabilities() -> dict[str, Any]:
    """What each mutator can do, against the installed packages.

    §23 item 1 asked whether Shinka exposes a per-candidate callback, because it
    decided driver-vs-fork. It is answered here and it is no longer the deciding
    question -- the built-in operator does not need one, since the loop is ours.
    What this is now for is telling you whether `--mutator shinka` will work
    before you ask for it in the middle of a campaign.
    """
    report: dict[str, Any] = {"default": MUTATOR_CLAUDE, "mutators": {}}

    try:
        import claude_agent_sdk  # noqa: PLC0415, F401

        sdk_present = True
    except ImportError:
        sdk_present = False
    report["mutators"][MUTATOR_CLAUDE] = {
        "available": sdk_present,
        "granularity": "candidate",
        "patch_types": list(evolution.PATCH_TYPES),
        "note": (
            "the built-in operator: one Agent SDK call per candidate, metered under the "
            "`evolve.mutate` stage and bounded by the project's token allocation. The loop, "
            "the selection policy and the budget gate are all local, so the budget is "
            "re-checked at every generation boundary by construction."
        )
        if sdk_present
        else "claude-agent-sdk is not installed: pip install -e '.[agent]'",
    }

    try:
        shinka = _shinka()
    except ConfigError as exc:
        report["mutators"][MUTATOR_SHINKA] = {"available": False, "reason": exc.message}
        return report

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

    per_generation = next((n for n in ShinkaMutator.PROPOSE_METHODS if n in methods), None)
    if hooks:
        granularity, note = "candidate", (
            "a per-candidate hook exists; this release could charge the budget inside the loop"
        )
    elif per_generation:
        granularity, note = "generation", (
            f"`{per_generation}()` yields one generation at a time, so the budget can be "
            "re-checked at generation boundaries. --mutator shinka will work."
        )
    else:
        granularity, note = "campaign", (
            "no per-candidate hook and no per-generation entry point -- the runner exposes "
            "only whole-loop methods, which own the loop this driver needs to interrupt. "
            "--mutator shinka refuses rather than handing control away with the budget "
            "unchecked; the built-in operator is unaffected."
        )

    report["mutators"][MUTATOR_SHINKA] = {
        "available": bool(hooks or per_generation),
        "version": getattr(shinka, "__version__", None),
        "runner": runner is not None,
        "per_candidate_hooks": hooks,
        "per_generation_method": per_generation,
        "runner_methods": methods[:20],
        "granularity": granularity,
        "note": note,
    }
    return report


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
_INITIAL_TEMPLATE = '''"""The program being evolved.

Everything between the EVOLVE-BLOCK markers is mutable; everything outside is
not. That boundary is not decoration -- the mutation operator is only ever given
the region's contents and `core/campaign.py:replace_blocks` splices them back
between the baseline's own markers, so code out here cannot change. Keeping
imports, I/O, and the entry point outside the block is what keeps a campaign
affordable, because a mutation that escapes needs a fresh remote smoke run.
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

Contract: print ONE JSON object of metrics to stdout, and it must contain
`combined_score`, where higher is better. Everything else in the object is
recorded alongside it and is what makes the Goodhart failure visible -- a search
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

_TASK_TEMPLATE = """# What this campaign is searching for

Replace this with a short brief: what the program does, what "better" means
beyond the scalar, and any constraint the operator cannot infer from the code
(a memory budget, an interface something else depends on, an approach already
tried and rejected).

This file is put in front of the mutation operator on every proposal. It is the
cheapest place to stop a campaign spending four generations rediscovering
something you already know.
"""


def _init_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task-dir", required=True, help="directory to scaffold, e.g. pipeline/evolve-lr")
    p.add_argument("--force", action="store_true", help="overwrite existing files")


@cli.command("init", "scaffold a task directory with the evolve-block contract", setup=_init_args)
def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    """`initial.py` with EVOLVE-BLOCK markers, `evaluate.py` returning
    `combined_score`, and `TASK.md` for the brief the operator is shown."""
    task_dir = Path(args.task_dir)
    if not task_dir.is_absolute():
        task_dir = paths.root() / task_dir
    task_dir.mkdir(parents=True, exist_ok=True)

    templates = (
        ("initial.py", _INITIAL_TEMPLATE),
        ("evaluate.py", _EVALUATE_TEMPLATE),
        ("TASK.md", _TASK_TEMPLATE),
    )
    written = []
    for name, body in templates:
        target = task_dir / name
        if target.exists() and not args.force:
            continue
        target.write_text(body, encoding="utf-8")
        written.append(str(target))

    return {
        "task_dir": str(task_dir),
        "written": written,
        "skipped": [
            str(task_dir / n) for n, _ in templates if str(task_dir / n) not in written
        ],
        "contract": {
            "initial.py": f"mutable region between {camp.BLOCK_START} and {camp.BLOCK_END}",
            "evaluate.py": "prints one JSON object of metrics including combined_score",
            "TASK.md": "the brief shown to the mutation operator on every proposal",
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
        "--mutator",
        choices=(MUTATOR_CLAUDE, MUTATOR_SHINKA),
        default=MUTATOR_CLAUDE,
        help="which mutation engine proposes (default: the built-in Agent SDK operator)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help=(
            "proposals in flight at once. Network-bound and independent by construction, "
            "so this is close to a linear speedup on a generation."
        ),
    )
    p.add_argument(
        "--eval-jobs",
        type=int,
        default=1,
        help=(
            "evaluations in flight at once. Defaults to 1 because a local evaluation is "
            "compute-bound and may hold the GPU -- raise it only if yours does not."
        ),
    )
    p.add_argument("--islands", type=int, default=2, help="sub-populations; 1 disables them")
    p.add_argument(
        "--migrate-every",
        type=int,
        default=3,
        help="generations between island migrations; 0 disables migration",
    )
    p.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="parent selection pressure; 0 is uniform, higher is greedier",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="RNG seed, recorded on the campaign. Omit for a fresh one, also recorded.",
    )
    p.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="evaluate locally, in a subprocess on this machine (the default)",
    )
    p.add_argument(
        "--remote",
        choices=REMOTE_BACKENDS,
        help=(
            "evaluate every candidate on real hardware instead of locally. Requires "
            "--remote-spec, and refuses unless that spec has a complete, passing "
            "preflight -- tests, dry run, and a real smoke run on the hardware."
        ),
    )
    p.add_argument(
        "--remote-spec",
        help="the pipeline spec whose preflighted environment candidates run in",
    )
    p.add_argument(
        "--remote-timeout-s",
        type=int,
        default=0,
        help="wall clock per remote candidate; defaults to --timeout-s",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="passed through to the Shinka mutator, e.g. --set evo.llm_models=...",
    )
    p.add_argument("--timeout-s", type=int, default=600, help="wall clock per candidate evaluation")


@cli.command("run", "run a budgeted campaign", setup=_run_args)
def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
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
            f"{initial} has no {camp.BLOCK_START}/{camp.BLOCK_END} markers, so there is no "
            "mutable region to propose into -- the operator is only ever given the contents "
            "of a region, and a file with none has nothing to give it",
            fix=f"wrap the mutable region in {camp.BLOCK_START} / {camp.BLOCK_END} comments",
        )

    # Before the expectation is bound and before a campaign id exists, because
    # every refusal in here is a configuration problem and none of them should
    # cost an expectation that then has to be re-minted.
    remote = _remote_target(args, cfg)

    # The campaign is the unit of prediction (§21 collision 2). The expectation
    # is bound here, once, and the candidates below are exempt from the per-run
    # gate precisely because this binding exists.
    expectation = _bind_expectation(args.expect)

    project_id = budget.resolve(args.project)
    max_candidates = max(1, args.generations) * max(1, args.population)
    projected = float(args.estimate_per_candidate_usd) * max_candidates

    # THE gate. Before generation 0, not after generation 40.
    _campaign_gate(project_id, projected, max_candidates, args.estimate_per_candidate_usd)

    # Recorded whether or not it was given, because a campaign whose seed is not
    # in the ledger is a campaign that cannot be replayed -- and the selection
    # policy is deterministic in it by design.
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)

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
        **_remote_note(remote),
        "mutator": args.mutator,
        "model": cfg.model_for("evolve"),
        "seed": seed,
        "islands": max(1, args.islands),
        "migrate_every": max(0, args.migrate_every),
        "pressure": args.pressure,
        "jobs": max(1, args.jobs),
        "eval_jobs": max(1, args.eval_jobs),
        "capabilities": mutator_capabilities(),
        "status": "open",
    }
    camp.append_campaign(record)

    # `_drive` raises for the expected case as well as the unexpected one --
    # `--mutator shinka` against a release with no per-generation entry point
    # raises ConfigError by design. Left uncaught, the campaign record stayed
    # `open` forever: consuming its expectation, accepting halt requests nothing
    # would ever read, and counting against the project's allocation. A campaign
    # that stopped is a campaign that closes, whichever way it stopped.
    try:
        result = _drive(
            campaign_id=campaign_id,
            task_dir=task_dir,
            baseline_source=baseline_source,
            args=args,
            project_id=project_id,
            seed=seed,
            cfg=cfg,
            remote=remote,
        )
    except BaseException as exc:  # noqa: BLE001 - including KeyboardInterrupt
        camp.close_campaign(
            campaign_id,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise

    camp.close_campaign(campaign_id, status=result["status"], reason=result.get("reason", ""))
    return {
        "campaign": campaign_id,
        "expectation_id": args.expect,
        "project": project_id,
        "seed": seed,
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
    # One predicate, shared with `gates.check_expectation`: runs, campaigns, and
    # retractions.
    if expectation_id in ls.consumed_expectation_ids():
        retracted = expectation_id in ls.falsified_ids()
        raise GateRefusal(
            "expectation_falsified" if retracted else "expectation_bound",
            f"expectation {expectation_id!r} was retracted"
            if retracted
            else f"expectation {expectation_id!r} is already bound to a run or campaign",
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


# ---------------------------------------------------------------------------
# where candidates run
# ---------------------------------------------------------------------------
def _remote_target(args: argparse.Namespace, cfg: config_mod.Config) -> dict[str, Any] | None:
    """Resolve and gate `--remote`. `None` means the campaign evaluates locally.

    **The gate is the whole function.** Everything below the first two refusals
    exists to answer one question: has the environment these candidates will land
    in already been through the ordinary §6 path? A remote campaign is the one
    place in this system where a loop with no human in it spends money in a
    tight cycle, so the answer has to be yes *before* generation 0, for the same
    reason `_campaign_gate` runs there rather than at generation 40.
    """
    if not args.remote:
        if args.remote_spec:
            raise UsageError(
                "--remote-spec names an environment but no --remote backend to run it on",
                fix=f"--remote {BACKEND_SSH} --remote-spec {args.remote_spec}",
            )
        return None

    if not args.remote_spec:
        raise UsageError(
            "--remote needs the spec whose preflighted environment candidates run in: "
            "there is no such thing as 'the remote' in general, only a pipeline that has "
            "been proven on one",
            fix="--remote-spec pipeline/spec.toml",
        )

    sub = Submission.load(args.remote_spec)
    _remote_gate(sub, cfg)

    target: dict[str, Any] = {
        "backend": args.remote,
        "spec": str(sub.spec_path),
        "submission_hash": sub.hash(),
        "sub": sub,
    }

    if args.remote == BACKEND_SSH:
        host_name = str(sub.target.get("host") or "")
        if not host_name:
            raise ConfigError(
                f"{sub.spec_path} names no [target] host, so there is nowhere to send candidates",
                fix="add `host = \"<name>\"` under [target], matching a [hosts.*] entry",
            )
        # Resolved here rather than per candidate: an unknown host is a
        # configuration error and it should be one before generation 0, not
        # forty evaluations in. Every backend below resolves the equivalent for
        # the same reason.
        host = cfg.host(host_name)
        target.update({"host": host.name, "rate_usd_per_hour": host.rate_usd_per_hour})
        return target

    if args.remote == BACKEND_KAGGLE:
        from tools import kaggle as kaggle_tool  # noqa: PLC0415 - optional deps

        accelerator = kaggle_tool.resolve_accelerator(None, sub, cfg)
        kind = cfg.accelerator_kind(accelerator)
        target.update({"accelerator": accelerator, "accelerator_kind": kind})
        _kaggle_hours_gate(cfg, args, sub, accelerator=accelerator, kind=kind)
        return target

    if args.remote == BACKEND_MODAL:
        from tools import modal as modal_tool  # noqa: PLC0415 - optional deps

        gpu = modal_tool.resolve_gpu(None, sub, cfg)
        # Refused before generation 0, for the reason the HF branch below gives:
        # an unpriced accelerator makes the campaign's projected cost a fiction,
        # and the campaign budget gate is the only thing between a search and an
        # allocation.
        rate = modal_tool.gpu_rate(gpu, cfg)
        if rate is None:
            raise ConfigError(
                f"Modal GPU {gpu!r} has no rate in [modal.gpu_rates], so a campaign on it "
                "cannot be priced",
                fix=f'add `"{gpu}" = <usd_per_hour>` under [modal.gpu_rates] in config/grad.toml',
            )
        target.update({"gpu": gpu, "rate_usd_per_hour": rate})
        return target

    from tools import jobs as jobs_tool  # noqa: PLC0415 - optional deps

    flavor = sub.target.get("flavor") or cfg.get("hf", "default_flavor", "a10g-small")
    # Refused before generation 0 rather than priced at zero: an unpriced flavor
    # makes the campaign's projected cost a fiction, and the campaign budget gate
    # is the only thing standing between a search and an allocation.
    if jobs_tool.flavor_rate(flavor, cfg) is None:
        raise ConfigError(
            f"flavor {flavor!r} has no rate in [hf.flavor_rates], so a campaign on it "
            "cannot be priced",
            fix=f'add `"{flavor}" = <usd_per_hour>` under [hf.flavor_rates] in config/grad.toml',
        )
    target.update({"flavor": flavor, "rate_usd_per_hour": jobs_tool.flavor_rate(flavor, cfg)})
    return target


def _kaggle_hours_gate(
    cfg: config_mod.Config,
    args: argparse.Namespace,
    sub: Submission,
    *,
    accelerator: str,
    kind: str,
) -> None:
    """Refuse a campaign that cannot fit in the week's accelerator hours.

    **The dollar gate cannot see this one.** Kaggle rations *hours*, not money,
    so a campaign priced at zero passes `_campaign_gate` unconditionally and
    would then spend the whole weekly GPU allowance -- with the first symptom
    being an ordinary submission refused for hours nothing could account for.
    `core/kaggle_quota.py` folds candidate rows beside runs so the hours are
    visible after the fact; this is what stops them being spent in the first
    place.

    Projected the same way the dollar gate projects: the per-candidate estimate
    times the whole campaign, checked before generation 0 rather than at
    generation 40. The per-candidate number is the spec's own estimate, which is
    the same one a submission of this pipeline would be gated on.
    """
    from core import kaggle_quota  # noqa: PLC0415
    from tools import kaggle as kaggle_tool  # noqa: PLC0415

    per_candidate = kaggle_tool.estimated_hours(sub)
    candidates = max(1, args.generations) * max(1, args.population)
    projected = per_candidate * candidates

    # Two ceilings, and they take *different* numbers -- which is the whole
    # reason this is not one call to `kaggle_quota.check`. The session cap is
    # what Kaggle stops a single kernel at, so it is asked about one candidate;
    # handing it the campaign total would refuse a perfectly ordinary search of
    # twenty one-hour candidates for exceeding a twelve-hour session. The weekly
    # allowance is the opposite: it is about the pool, so it gets the projection
    # for the whole campaign.
    #
    # Both *raise* rather than return a refusal, and their messages and fixes are
    # already the right ones -- `quota_weekly` names what is holding the hours
    # and points at `kaggle quota --json`. The session cap passes through
    # untouched for that reason. Only the weekly one is re-framed, because "this
    # run estimates 160h" is a confusing way to describe forty four-hour
    # candidates, and the number a reader needs is the shape of the campaign.
    kaggle_quota.check_session(cfg, kind, per_candidate, accelerator=accelerator)

    try:
        kaggle_quota.check_quota(cfg, kind, projected, accelerator=accelerator)
    except GateRefusal as exc:
        raise GateRefusal(
            exc.code,
            f"a campaign of {candidates} candidates at {per_candidate:.2f}h each projects "
            f"{projected:.1f} {kind} hours, which does not fit the week's allowance. "
            f"{exc.message}",
            exc.exit_code,
            fix=(
                "lower --generations/--population, shorten the evaluation, or wait for the "
                "rolling week to move: python -m tools.kaggle quota --json"
            ),
            detail=exc.detail,
        ) from None


def _remote_gate(sub: Submission, cfg: config_mod.Config) -> None:
    """Refuse unless this spec has a complete, passing preflight including smoke.

    `gates.check_preflight` is reused rather than reimplemented, and the required
    list is named here rather than read from `[preflight] checks`. That is
    deliberate: the config's list is a machine's policy for ordinary
    submissions, and a machine configured without `smoke` would otherwise let a
    campaign put every candidate it has on hardware that nothing has ever run a
    single step on. The smoke run is the only check that sees the real driver
    stack, the real data path and the real per-device batch size, which is
    exactly the set of things a search will otherwise discover forty times.
    """
    from core import gates  # noqa: PLC0415

    gates.check_preflight(sub, cfg, required=list(REMOTE_REQUIRED_CHECKS))


def _remote_note(target: dict[str, Any] | None) -> dict[str, Any]:
    """The target as it goes into the campaign record -- without the Submission.

    `Submission` is a live object with paths and a resolved config in it; the
    campaign record is JSON that outlives this process and gets read by hand.
    """
    if target is None:
        return {"mode": "local"}
    note = {
        "mode": "remote",
        "backend": target["backend"],
        "remote_spec": target["spec"],
        "submission_hash": target["submission_hash"],
    }
    # Only what this backend actually resolved. A record carrying `host: null`
    # on a Kaggle campaign reads as a host that could not be found rather than
    # as a dimension that does not apply.
    for key in ("host", "rate_usd_per_hour", "accelerator", "accelerator_kind", "flavor", "gpu"):
        if target.get(key) is not None:
            note[key] = target[key]
    return note


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def _drive(
    *,
    campaign_id: str,
    task_dir: Path,
    baseline_source: str,
    args: argparse.Namespace,
    project_id: str | None,
    seed: int,
    cfg: config_mod.Config,
    mutator: Any = None,
    remote: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generation by generation, with the gate between each.

    `mutator` is injectable so the driver can be tested against a fake operator
    -- §24's testing note asks for exactly that, and a campaign loop tested only
    against a live model is a campaign loop tested never.
    """
    mutator = mutator or _make_mutator(args, task_dir, cfg, project_id, campaign_id)
    generations = max(1, args.generations)
    population = max(1, args.population)
    islands = max(1, args.islands)
    per_candidate = float(args.estimate_per_candidate_usd)
    rng = random.Random(seed)

    evaluated = 0
    duplicates = 0
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
        # Everything the policy needs is folded from the ledger, so the loop
        # holds no state a crash could lose and the arm statistics are a function
        # of records anyone can read.
        history = camp.candidates(campaign_id)
        stats = evolution.bandit_from(history)
        seen = evolution.seen_sources(history)

        migrations = (
            evolution.migrate(history, islands=islands)
            if evolution.should_migrate(generation, interval=args.migrate_every)
            else []
        )
        if migrations:
            camp.record_migration(campaign_id, generation, migrations)
            history = camp.candidates(campaign_id)

        plans = evolution.plan_generation(
            history,
            generation=generation,
            population=population,
            islands=islands,
            stats=stats,
            rng=rng,
            pressure=args.pressure,
        )

        try:
            proposals = mutator.propose(plans=plans, baseline=baseline_source, history=history)
        except GradError:
            raise
        except Exception as exc:  # noqa: BLE001 - a mutation engine failure ends the campaign, not the process
            status = "failed"
            reason = f"the mutation engine failed at generation {generation}: {exc}"
            camp.record_generation(campaign_id, generation, error=str(exc))
            break

        # Deduplicated against everything ever proposed, not just this
        # generation: the operator is shown the elites every time and proposing
        # one of them back unchanged is its most common repeated mistake. A
        # duplicate is recorded rather than dropped silently -- a generation that
        # produced four copies of one idea is a fact about the search.
        prepared: list[dict[str, Any]] = []
        for proposal in proposals:
            if proposal.get("error") or not proposal.get("source"):
                prepared.append(proposal)
                continue
            key = evolution.source_key(proposal["source"])
            if key in seen:
                duplicates += 1
                prepared.append({**proposal, "duplicate_of": key, "error": "duplicate proposal"})
                continue
            seen.add(key)
            prepared.append({**proposal, "source_key": key})

        records = _evaluate_generation(
            campaign_id=campaign_id,
            generation=generation,
            proposals=prepared,
            baseline_source=baseline_source,
            task_dir=task_dir,
            timeout_s=args.timeout_s,
            per_candidate=per_candidate,
            jobs=max(1, args.eval_jobs),
            remote=remote,
            cfg=cfg,
            remote_timeout_s=int(args.remote_timeout_s or args.timeout_s),
        )
        evaluated += len(records)
        scores = [
            s for s in (evolution.score_of(r) for r in records) if s is not None
        ]

        camp.record_generation(
            campaign_id,
            generation,
            candidates=len(records),
            evaluated=len(scores),
            best_score=max(scores) if scores else None,
            duplicates=sum(1 for r in records if r.get("duplicate_of")),
            migrations=len(migrations),
            patch_types=sorted({str(r.get("patch_type")) for r in records if r.get("patch_type")}),
            duration_s=round(time.time() - started, 2),
        )

    final = camp.candidates(campaign_id)
    spend = camp.campaign_spend(campaign_id)
    return {
        "status": status,
        "reason": reason,
        "generations_run": min(generations, evaluated // max(1, population)),
        "candidates_evaluated": evaluated,
        "duplicates_rejected": duplicates,
        "spend": spend,
        "bandit": evolution.bandit_report(evolution.bandit_from(final)),
        # Top-K, not the argmax. A search optimising a scalar finds the bug in
        # the metric, so the shape of the leaderboard is part of the output.
        "top": [
            {
                "candidate_id": c["candidate_id"],
                "generation": c["generation"],
                "island": evolution.island_of(c),
                "patch_type": c.get("patch_type"),
                "rationale": c.get("rationale"),
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


def _evaluate_generation(
    *,
    campaign_id: str,
    generation: int,
    proposals: list[dict[str, Any]],
    baseline_source: str,
    task_dir: Path,
    timeout_s: int,
    per_candidate: float,
    jobs: int,
    remote: dict[str, Any] | None = None,
    cfg: Any = None,
    remote_timeout_s: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate one generation, at most `jobs` at once.

    Threads rather than processes: each evaluation is a `subprocess.run` and the
    thread spends its life blocked in `wait`, so the GIL is irrelevant here. The
    default of one is not timidity -- a local evaluation on this machine may be
    the thing holding the GPU, and four of those at once is four out-of-memory
    failures recorded as four bad mutations.

    That default is *local* reasoning, and it is why `--eval-jobs` is worth
    raising on a remote campaign: an ssh evaluation blocks the thread on the
    network rather than on this machine's GPU, and the host's own capacity is
    the thing to size it against instead.

    Results come back in plan order rather than completion order, so a campaign
    with the same seed produces the same `candidate_id` for the same slot. The
    ledger appends in whatever order they finish, which is fine -- `candidates()`
    folds by id.
    """
    def one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, proposal = item
        return _evaluate_candidate(
            campaign_id=campaign_id,
            generation=generation,
            index=index,
            proposal=proposal,
            baseline_source=baseline_source,
            task_dir=task_dir,
            timeout_s=timeout_s,
            per_candidate=per_candidate,
            remote=remote,
            cfg=cfg,
            remote_timeout_s=remote_timeout_s or timeout_s,
        )

    items = list(enumerate(proposals))
    if jobs <= 1 or len(items) <= 1:
        return [one(item) for item in items]
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="grad-eval") as pool:
        return list(pool.map(one, items))


def _evaluate_candidate(
    *,
    campaign_id: str,
    generation: int,
    index: int,
    proposal: dict[str, Any],
    baseline_source: str,
    task_dir: Path,
    timeout_s: int,
    per_candidate: float,
    remote: dict[str, Any] | None = None,
    cfg: Any = None,
    remote_timeout_s: int = 0,
) -> dict[str, Any]:
    """Evaluate one candidate and record it as a sub-run.

    Candidates go to `ledger/candidates.jsonl`, never to `runs.jsonl`: a
    100-generation campaign is thousands of rows and would dominate a ledger
    meant to be read by hand (§23 item 4). Only a promoted candidate becomes a
    run. **That holds on a remote campaign too** -- see
    `tools/gpu.py:evaluate_candidate` for why a per-candidate ledger row would
    undo the rule the moment the search left this machine, and where the gate
    sits instead.

    Everything up to the point of execution is identical either way: the same
    escape check, the same two files written into the same local working
    directory. A remote candidate then runs those files on the host instead of
    in a subprocess, and its `cost_usd` becomes a measurement rather than the
    campaign's per-candidate estimate.
    """
    candidate_id = candidate_id_for(campaign_id, generation, index)
    source = proposal.get("source") or ""
    workdir = paths.run_artifacts(candidate_id)
    workdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record: dict[str, Any] = {
        "campaign": campaign_id,
        "candidate_id": candidate_id,
        "generation": generation,
        "index": index,
        "at": camp.now_iso(),
        "island": proposal.get("island", 0),
        "patch_type": proposal.get("patch_type"),
        "parent_id": proposal.get("parent_id"),
        "mate_id": proposal.get("mate_id"),
        "rationale": proposal.get("rationale", ""),
        "source_key": proposal.get("source_key"),
        "duplicate_of": proposal.get("duplicate_of"),
        "workdir": str(workdir),
        "cost_usd": per_candidate,
    }

    if proposal.get("error") or not source:
        # The operator produced nothing usable: no tool call, an edit that did
        # not match, or a duplicate. Recorded so the failure is visible to the
        # next generation's prompt, and charged nothing -- it never ran, and
        # charging the estimate anyway would let a campaign whose operator is
        # misconfigured exhaust its allocation having evaluated nothing.
        record.update(
            {
                "skipped": True,
                "metrics": None,
                "error": proposal.get("error") or "the operator produced no source",
                "duration_s": round(time.time() - started, 3),
                "cost_usd": 0.0,
            }
        )
        camp.append_candidate(record)
        return record

    escaped = camp.escaped_evolve_block(baseline_source, source)
    record["escaped_block"] = escaped
    (workdir / "initial.py").write_text(source, encoding="utf-8")
    evaluate_src = (task_dir / "evaluate.py").read_text(encoding="utf-8")
    (workdir / "evaluate.py").write_text(evaluate_src, encoding="utf-8")

    if escaped["escaped"]:
        # §21 collision 3: a mutation outside the block changed code the
        # baseline's smoke result no longer covers. With the built-in operator
        # this is close to unreachable -- it is handed the region's contents and
        # never a file -- so reaching it means a replacement carried marker text
        # past the operator's own validation, which is exactly the case
        # `replace_blocks` cannot check for itself.
        record.update(
            {
                "skipped": True,
                "metrics": None,
                "error": "mutation escaped the evolve block; needs a fresh smoke run",
                "duration_s": round(time.time() - started, 3),
                "cost_usd": 0.0,
            }
        )
        camp.append_candidate(record)
        return record

    if remote is not None:
        record.update(
            _evaluate_remotely(
                remote=remote,
                cfg=cfg,
                candidate_id=candidate_id,
                source=source,
                evaluator=evaluate_src,
                timeout_s=remote_timeout_s or timeout_s,
                workdir=workdir,
                started=started,
            )
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
        if problem is not None and stderr.strip():
            # The traceback's last line, appended to the reason. This is what the
            # next generation's prompt shows under "what has failed", and
            # "evaluate.py exited 1" on its own teaches the operator nothing.
            tail = [line for line in stderr.strip().splitlines() if line.strip()]
            problem = f"{problem}: {tail[-1].strip()[:300]}"
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


def _evaluate_remotely(
    *,
    remote: dict[str, Any],
    cfg: Any,
    candidate_id: str,
    source: str,
    evaluator: str,
    timeout_s: int,
    workdir: Path,
    started: float,
) -> dict[str, Any]:
    """Run one candidate on the campaign's host and read its metrics back.

    The returned dict is the same set of fields the local path fills in, plus
    where it ran. Two of them differ in meaning and both differences are the
    point of going remote at all:

    * `cost_usd` is measured -- wall clock against the host's rate -- rather
      than the campaign's flat per-candidate estimate. The estimate is what the
      budget gate projects with; this is what was actually spent.
    * `error` can now be a transport failure rather than a bad mutation. Those
      are recorded distinctly, because a search that reads "the host refused the
      connection" as "this idea scored nothing" will quietly select against
      whatever was being proposed when the network wobbled.
    """
    result = _run_on_backend(
        remote,
        cfg or config_mod.load(),
        candidate_id=candidate_id,
        files={"initial.py": source, "evaluate.py": evaluator},
        timeout_s=int(timeout_s),
        artifacts=workdir,
    )

    output = str(result.get("output") or "")
    (workdir / "evaluate.log").write_text(output, encoding="utf-8")

    fields: dict[str, Any] = {
        "duration_s": round(time.time() - started, 3),
        "cost_usd": float(result.get("cost_usd") or 0.0),
        "ran_on": result.get("where"),
        "backend": remote["backend"],
    }
    # Kaggle rations hours rather than dollars, so the number that bounds a
    # campaign there is not `cost_usd`. Recorded under the field names
    # `core/kaggle_quota.py` folds, which is what lets a campaign's candidates
    # count against the weekly allowance at all -- they never reach `runs.jsonl`,
    # so the fold has nowhere else to read them from.
    if result.get("hours") is not None:
        from core import kaggle_quota  # noqa: PLC0415

        fields[kaggle_quota.F_ACTUAL] = float(result["hours"])
        fields[kaggle_quota.F_ACCELERATOR] = result.get("accelerator")
        fields[kaggle_quota.F_KIND] = result.get("accelerator_kind")

    if result.get("exit_code") is None:
        # The candidate never ran. Recorded as `skipped` for the same reason an
        # operator that produced nothing is: it has no score, and folding it into
        # the population as a zero would teach the next generation that whatever
        # was proposed here is bad.
        fields.update(
            {
                "skipped": True,
                "metrics": None,
                "error": f"the host could not run it: {result.get('error')}",
            }
        )
        return fields

    metrics, problem = _metrics_from(output)
    if not result.get("ok") and problem is None:
        problem = str(result.get("error") or "the candidate exited non-zero")
    fields.update({"metrics": metrics if problem is None else None, "error": problem})
    return fields


def _run_on_backend(
    remote: dict[str, Any],
    cfg: Any,
    *,
    candidate_id: str,
    files: dict[str, str],
    timeout_s: int,
    artifacts: Path,
) -> dict[str, Any]:
    """Hand one candidate to whichever backend the campaign is running on.

    The three adapters answer the same question and return the same shape -- ok,
    exit code, output, error, cost, where -- but they get there differently
    enough that a shared implementation would be a lie: `gpu.py` copies a
    directory to a machine that stays up, `kaggle.py` packs the pipeline into a
    notebook, and `jobs.py` has no upload step at all because the pipeline is
    already in the image. Each one's own module owns that difference, which is
    the same division `core/submit.py` already draws for real submissions.

    Imported at the point of use, because each backend brings optional
    dependencies and a campaign on one of them must not need the others
    installed.
    """
    backend = remote["backend"]
    common = {
        "candidate_id": candidate_id,
        "files": files,
        "command": ["python", "evaluate.py"],
        "timeout_s": int(timeout_s),
    }

    if backend == BACKEND_SSH:
        from tools import gpu as gpu_tool  # noqa: PLC0415

        return gpu_tool.evaluate_candidate(remote["sub"], cfg, **common)

    if backend == BACKEND_KAGGLE:
        from tools import kaggle as kaggle_tool  # noqa: PLC0415

        return kaggle_tool.evaluate_candidate(
            remote["sub"], cfg, artifacts=artifacts,
            accelerator=remote.get("accelerator"), **common,
        )

    if backend == BACKEND_MODAL:
        from tools import modal as modal_tool  # noqa: PLC0415

        return modal_tool.evaluate_candidate(
            remote["sub"], cfg, artifacts=artifacts, gpu=remote.get("gpu"), **common
        )

    if backend != BACKEND_HF:
        # Explicit rather than a fall-through, and this is a bug that already
        # happened once in the making: `modal` was added to `REMOTE_BACKENDS`
        # before it was added here, and the effect of the old `return
        # jobs_tool...` at the end of this function was that `--remote modal`
        # ran the whole campaign on Hugging Face Jobs. Silently, with the right
        # hardware name in the log.
        raise ConfigError(
            f"no candidate evaluator is wired up for backend {backend!r}",
            fix=f"--remote {'|'.join(REMOTE_BACKENDS)}",
        )

    from tools import jobs as jobs_tool  # noqa: PLC0415

    return jobs_tool.evaluate_candidate(
        remote["sub"], cfg, artifacts=artifacts, flavor=remote.get("flavor"), **common
    )


def _metrics_from(output: str) -> tuple[Any, str | None]:
    """The evaluator's one JSON object, out of a combined stdout/stderr stream.

    The local path can read stdout on its own; over ssh the two are merged so a
    traceback is not lost, and `_ssh` appends the `EXIT:` line the exit code is
    read from. So the marker is dropped and the *last* line is taken -- the same
    rule the local path uses, deliberately, rather than a more forgiving scan.
    A search whose metric can be found anywhere in the output is a search that
    can be fed a number by a log line.
    """
    lines = [
        line for line in output.strip().splitlines() if not line.startswith("EXIT:")
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None, "the candidate printed nothing"
    try:
        metrics = json.loads(lines[-1])
    except json.JSONDecodeError:
        tail = lines[-1].strip()[:300]
        return None, f"evaluate.py did not print a JSON object of metrics: {tail}"
    return metrics, camp.validate_metrics(metrics)


# ---------------------------------------------------------------------------
# mutation engines
# ---------------------------------------------------------------------------
def candidate_id_for(campaign_id: str, generation: int, index: int) -> str:
    """The one place a candidate's id is spelled.

    Two callers need it and they run at different times: `_evaluate_candidate`
    mints it, and `ClaudeMutator` needs the same string *before* the candidate
    exists, so the prompt and the response land in the artifacts directory the
    source and the evaluation log will land in. A second spelling would put the
    mutation log next to nothing.
    """
    return f"{campaign_id}-g{generation}-c{index}"


def candidate_source(candidate: dict[str, Any]) -> str | None:
    """A candidate's full source, read back from its artifacts directory.

    Read rather than carried in the record: the ledger is meant to be read by
    hand, and a hundred candidates each carrying a copy of the program would make
    `candidates.jsonl` unreadable in the literal sense.
    """
    workdir = candidate.get("workdir")
    if not workdir:
        return None
    try:
        return (Path(workdir) / "initial.py").read_text(encoding="utf-8")
    except OSError:
        return None


class ClaudeMutator:
    """The built-in operator: one Agent SDK call per candidate, `--jobs` at once.

    Thin on purpose. Everything worth owning -- the gate, the ledger records, the
    escape check, the selection policy -- is outside it, which is what made
    replacing ShinkaEvolve a change to one class rather than a rewrite.
    """

    def __init__(
        self,
        task_dir: Path,
        *,
        campaign_id: str,
        model: str,
        jobs: int,
        project: str | None,
        timeout_s: float = mutate.PROPOSE_TIMEOUT_S,
    ) -> None:
        self.task_dir = task_dir
        self.campaign_id = campaign_id
        self.model = model
        self.jobs = max(1, jobs)
        self.project = project
        self.timeout_s = timeout_s

    def _brief(self) -> str:
        for name in ("TASK.md", "README.md"):
            path = self.task_dir / name
            if path.is_file():
                return path.read_text(encoding="utf-8")
        return ""

    def _evaluator(self) -> str:
        path = self.task_dir / "evaluate.py"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def _log_dir(self, plan: dict[str, Any]) -> Path:
        return paths.run_artifacts(
            candidate_id_for(self.campaign_id, int(plan["generation"]), int(plan["index"]))
        )

    def propose(
        self, *, plans: list[dict[str, Any]], baseline: str, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return mutate.propose_all(
            plans,
            jobs=self.jobs,
            log_dir_for=self._log_dir,
            baseline=baseline,
            task_brief=self._brief(),
            evaluator=self._evaluator(),
            source_of=candidate_source,
            model=self.model,
            project=self.project,
            timeout_s=self.timeout_s,
        )


class ShinkaMutator:
    """The ShinkaEvolve path, kept and no longer the default.

    It refuses unless the installed release exposes a per-generation entry point,
    for the reason §21 gives: a driver cannot charge the budget between
    generations through an API that only offers "run everything". That refusal is
    unchanged. What changed is what happens next -- there is now a built-in
    operator to fall back to, so the refusal names it.
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
            evolution_cfg = shinka.EvolutionConfig(
                llm_models=list(self.models),
                init_program_path=str(self.task_dir / "initial.py"),
            )
            job = shinka.LocalJobConfig(eval_program_path=str(self.task_dir / "evaluate.py"))
            database = shinka.DatabaseConfig()
            runner = shinka.ShinkaEvolveRunner(
                evo_config=evolution_cfg, job_config=job, db_config=database
            )
        except (AttributeError, TypeError) as exc:
            raise ConfigError(
                f"the installed shinka-evolve does not match the expected API: {exc}",
                fix=(
                    "check the constructor against the installed version "
                    "(`python -m tools.docs signature shinka ShinkaEvolveRunner --json`), "
                    "then adjust ShinkaMutator -- or drop --mutator shinka"
                ),
            ) from exc

        method = self._propose_method(runner)
        if method is None:
            available = sorted(
                n for n in dir(runner) if not n.startswith("_") and callable(getattr(runner, n, None))
            )
            raise ConfigError(
                "the installed ShinkaEvolveRunner exposes no per-generation entry point, "
                "only whole-loop methods, so the campaign budget could not be re-checked "
                "between generations. "
                f"Available: {', '.join(available[:12]) or '(none)'}",
                fix=(
                    "drop --mutator shinka: the built-in operator holds the loop here, so the "
                    "budget is re-checked at every generation boundary and every proposal is "
                    "metered under the `evolve.mutate` stage. "
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

    def propose(
        self, *, plans: list[dict[str, Any]], baseline: str, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self._runner is None:
            self._runner = self._build()
        generation = plans[0]["generation"] if plans else 0
        best = next(iter(evolution.scored(history)), None)
        sources = getattr(self._runner, self._method)(
            generation=generation,
            population=len(plans),
            parent=candidate_source(best) if best else None,
        )
        # Shinka returns whole programs, so its output goes through the escape
        # check the way it always did -- the structural guarantee `replace_blocks`
        # gives the built-in operator does not extend to an engine that writes
        # files. That difference is the reason the check survived the rewrite.
        quota_log.record(
            STAGE_EVOLVE,
            role="evolve",
            model=",".join(self.models),
            detail={
                "generation": generation,
                "population": len(plans),
                "method": self._method,
                "engine": MUTATOR_SHINKA,
                # Stated, because the record cannot carry what it cannot see:
                # Shinka spends through its own rail and reports no usage here.
                "tokens": "not reported by shinka-evolve; this row counts the call only",
            },
        )
        # One record per *plan*, never per source. `zip` truncated to the shorter
        # of the two, so an upstream runner that returned three sources for a
        # population of four produced a generation of three -- with nothing
        # anywhere saying a slot had gone missing. The plan is what the budget
        # gate counted and what `plan_generation` decided; a source that never
        # arrived is a failed proposal, which is a record this already knows how
        # to write.
        out: list[dict[str, Any]] = []
        for index, plan in enumerate(plans):
            source = sources[index] if index < len(sources) else None
            out.append(
                {
                    "patch_type": plan.get("patch_type"),
                    "island": plan.get("island"),
                    "index": plan.get("index"),
                    "generation": generation,
                    "parent_id": (plan.get("parent") or {}).get("candidate_id"),
                    "source": "" if source is None else str(source),
                    "rationale": "",
                    "error": (
                        None
                        if source is not None
                        else f"{MUTATOR_SHINKA} returned {len(sources)} source(s) "
                        f"for {len(plans)} plan(s); this slot got none"
                    ),
                }
            )
        return out


def _models(cfg: config_mod.Config, overrides: list[str]) -> tuple[str, ...]:
    """The ensemble handed to ShinkaEvolve. Only `--mutator shinka` reads this.

    Shinka's design is an ensemble of LLMs acting as mutation operators and its
    bandit allocates between them, so collapsing to one model discards diversity
    that engine is built around. The built-in operator gets its diversity from
    the patch-type bandit instead, which is why it takes one model.
    """
    for override in overrides:
        if override.startswith("evo.llm_models="):
            value = override.split("=", 1)[1]
            return tuple(m.strip() for m in value.split(",") if m.strip())
    primary = cfg.model_for("evolve")
    return (primary, "claude-haiku-4-5") if primary != "claude-haiku-4-5" else (primary,)


def _make_mutator(
    args: argparse.Namespace,
    task_dir: Path,
    cfg: config_mod.Config,
    project_id: str | None,
    campaign_id: str,
) -> Any:
    if args.mutator == MUTATOR_SHINKA:
        return ShinkaMutator(task_dir, _models(cfg, args.overrides), args.overrides)
    return ClaudeMutator(
        task_dir,
        campaign_id=campaign_id,
        model=cfg.model_for("evolve"),
        jobs=args.jobs,
        project=project_id,
        timeout_s=float(cfg.get("evolve", "propose_timeout_s", mutate.PROPOSE_TIMEOUT_S)),
    )


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
                    "mutator": c.get("mutator"),
                    "generations_run": c.get("generations_run"),
                    "spend": camp.campaign_spend(cid),
                }
                for cid, c in camp.campaigns().items()
            ]
        }
    record = camp.campaign(args.campaign)
    rows = camp.candidates(args.campaign)
    return {
        "campaign": record,
        "spend": camp.campaign_spend(args.campaign),
        "bandit": evolution.bandit_report(evolution.bandit_from(rows)),
        "islands": {
            str(island): len(evolution.members(rows, island))
            for island in range(int(record.get("islands", 1) or 1))
        },
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
        "rationale": match.get("rationale"),
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


@cli.command("capabilities", "which mutation engines are installed and usable")
def cmd_capabilities(_: argparse.Namespace) -> dict[str, Any]:
    """Which operator will run, and whether the other one could."""
    return mutator_capabilities()


if __name__ == "__main__":
    main(cli)
