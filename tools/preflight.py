"""grad-preflight -- the QA gate (HANDOFF §6).

    "No job that costs money runs until a machine-checkable artifact says it
     will work."

This CLI produces that artifact: `ledger/preflight/<submission_hash>.json`. It
does not decide whether a job may run -- `jobs.py` and `gpu.py` do, by reading
the artifact. The separation matters: a checker that also submits is a checker
with a bypass flag.

There is no TTL. Nothing about a preflight record decays by sitting still; what
invalidates it is state change, and the submission hash is what notices state
change (see `core/submission.py`).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core import config as config_mod, gates, jsonl, paths
from core.cli import Cli, main
from core.errors import EXIT_CHECK_FAILED, GradError, UsageError
from core.ledger_store import now_iso
from core.submission import Submission, parse_override

cli = Cli(
    "grad-preflight",
    "Run the pre-flight QA gate and write the preflight record for a submission.",
    epilog=(
        "The record is keyed by the hash of the *resolved* submission: entrypoint plus\n"
        "its first-party import graph, resolved config, lock file, dataset revision,\n"
        "image digest, and argv. Change any of those and the record no longer applies.\n\n"
        "If only one check ever runs, it is `smoke`. If two, `dry_run` then `smoke`."
    ),
)

# Checks that are always available. Anything else must be declared in the spec's
# [checks] table as a command, because preflight cannot guess how a given
# pipeline asserts its own shapes or gradients.
BUILTIN = ("tests", "dry_run", "smoke", "cost")
DECLARED = ("shapes", "grads", "symbolic", "invariants")


def _spec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", required=True, help="path to the submission spec (TOML or JSON)")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, applied before hashing (repeatable)",
    )
    p.add_argument(
        "--no-digest",
        action="store_true",
        help="skip container digest resolution (for local-only pipelines with no image registry)",
    )


def _load(args: argparse.Namespace) -> Submission:
    overrides = dict(parse_override(o) for o in args.overrides)
    return Submission.load(args.spec, overrides=overrides, resolve_digest=not args.no_digest)


# ---------------------------------------------------------------------------
@cli.command("hash", "print the submission hash and the resolved document", setup=_spec_args)
def cmd_hash(args: argparse.Namespace) -> dict[str, Any]:
    """The hash is what `jobs.py` looks up. Print it before asking why a gate fired."""
    sub = _load(args)
    return {
        "submission_hash": sub.hash(),
        "full_hash": sub.full_hash(),
        "resolved": sub.resolved(),
        "warnings": sub.warnings,
        "record": str(paths.preflight_record(sub.hash())),
        "record_exists": paths.preflight_record(sub.hash()).exists(),
    }


def _run_args(p: argparse.ArgumentParser) -> None:
    _spec_args(p)
    p.add_argument(
        "--only",
        help="comma-separated subset of checks to run (the rest keep their previous result)",
    )
    p.add_argument("--skip", help="comma-separated checks to skip")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-run checks that already passed for this hash",
    )
    p.add_argument(
        "--jobs",
        type=int,
        help=(
            "local checks to run at once (default from [execution] default_jobs). "
            "`smoke` never runs concurrently with anything -- see --smoke-anyway."
        ),
    )
    p.add_argument(
        "--smoke-anyway",
        action="store_true",
        help="run the smoke check even if a local check failed (it costs a real remote job)",
    )


@cli.command("run", "run the checks and write the preflight record", setup=_run_args)
def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    sub = _load(args)
    h = sub.hash()
    paths.preflight_dir().mkdir(parents=True, exist_ok=True)

    configured = list(cfg.get("preflight", "checks", ["tests", "dry_run", "smoke"]))
    spec_checks = _declared_checks(sub)
    wanted = configured + [c for c in spec_checks if c not in configured]
    if args.only:
        requested = [c.strip() for c in args.only.split(",") if c.strip()]
        unknown = [c for c in requested if c not in BUILTIN + DECLARED and c not in spec_checks]
        if unknown:
            raise UsageError(
                f"unknown check(s): {', '.join(unknown)}",
                fix=f"available: {', '.join(sorted(set(BUILTIN + DECLARED) | set(spec_checks)))}",
            )
        wanted = requested
    if args.skip:
        skipped = {c.strip() for c in args.skip.split(",")}
        wanted = [c for c in wanted if c not in skipped]

    existing = jsonl.read_json(paths.preflight_record(h)) or {}
    results: dict[str, Any] = dict(existing.get("checks", {}))

    #: Only what *this* invocation computed. `results` starts as a snapshot of
    #: the record taken before the checks ran, and those checks take minutes --
    #: so writing all of it back would overwrite a concurrent update with a copy
    #: that was already stale when it was read. A check this run did not touch
    #: is a check this run has nothing to say about.
    computed: dict[str, Any] = {}

    todo = []
    for name in wanted:
        if not args.force and results.get(name, {}).get("ok"):
            results[name]["skipped_because"] = "already passing for this hash"
            continue
        todo.append(name)

    # `smoke` is separated from the rest, and it is the only check that is. Two
    # reasons, and neither is about speed:
    #
    # * it is a *paid remote job*, and the others are local processes. Running it
    #   alongside them would put a GPU on a queue while the tests that decide
    #   whether the code is worth running are still going.
    # * it depends on them. Before this, a spec whose tests failed still went on
    #   to submit a smoke run -- money spent to learn something the first check
    #   had already said. It now runs only if the local checks passed, and
    #   `--smoke-anyway` is there for the case where the remote environment is
    #   precisely what you are debugging.
    local = [n for n in todo if n != "smoke"]
    smoke_wanted = "smoke" in todo
    jobs = args.jobs if args.jobs is not None else int(cfg.get("execution", "default_jobs", 4))

    for name, result in _run_checks(local, sub, cfg, spec_checks, jobs=jobs).items():
        results[name] = computed[name] = result

    # Over the record, not only over what this invocation ran. `local` is empty
    # under `--only smoke`, which made the one check that costs money the one
    # check that could run after `tests` had already failed for this very hash --
    # the exact spend the separation above exists to prevent.
    #
    # A check with no recorded result is still not a failure: `results` is keyed
    # by submission hash, so "never run" and "ran and failed" are different
    # facts, and the missing one is `check_preflight`'s refusal to make.
    prerequisites = [
        n for n in dict.fromkeys([*configured, *spec_checks, *local]) if n != "smoke"
    ]
    local_failed = [
        n
        for n in prerequisites
        if isinstance(results.get(n), dict) and results[n].get("ok") is not True
    ]
    smoke_skipped = None
    if smoke_wanted:
        if local_failed and not args.smoke_anyway:
            # Deliberately *not* written into the record as a failing check. A
            # check that did not run is not a check that failed -- the same
            # distinction `submit.compute_deviations` makes about `in_range` --
            # and the gate refusing with "no result for check: smoke" is both
            # true and the more useful message.
            smoke_skipped = (
                f"not run: {', '.join(local_failed)} failed first, and a smoke check is a "
                "real remote job. Fix those and re-run, or --smoke-anyway."
            )
        else:
            results["smoke"] = computed["smoke"] = _run_check("smoke", sub, cfg, spec_checks)

    # Merged under the lock rather than written over the top. The smoke path
    # writes its own result through `record_check_result` while the checks
    # above are still running, and a plain write would drop it -- which for
    # this file means a gate reading a record that is missing a check that
    # actually passed.
    def _merge(current: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict((current or {}).get("checks") or {})
        merged.update(computed)
        return {
            "submission_hash": h,
            "full_hash": sub.full_hash(),
            "spec": str(sub.spec_path),
            "verified_at": now_iso(),
            "resolved": sub.resolved(),
            "checks": merged,
            "warnings": sub.warnings,
            "estimate_usd": sub.estimated_cost_usd(),
            "estimated_duration_s": sub.estimated_duration_s(),
        }

    record = jsonl.update_json(paths.preflight_record(h), _merge)
    results = record["checks"]

    # `is not True`, matching `gates.check_preflight`: a check that recorded no
    # verdict at all is not a check that passed, and the CLI must not report a
    # clean run where the gate will refuse.
    failing = [n for n, r in results.items() if not (isinstance(r, dict) and r.get("ok") is True)]
    payload = {
        "submission_hash": h,
        "record": str(paths.preflight_record(h)),
        "checks": {n: {k: v for k, v in r.items() if k != "output"} for n, r in results.items()},
        "failing": failing,
        "not_run": {"smoke": smoke_skipped} if smoke_skipped else {},
        "warnings": sub.warnings,
    }
    if smoke_skipped and not failing:
        # Only reachable with `--only smoke` plus a stale failing local result in
        # the record, but the alternative is reporting a clean run for a spec the
        # gate will refuse -- which is the exact thing the `is not True` check
        # below exists to prevent.
        raise GradError(
            "preflight_incomplete",
            smoke_skipped,
            exit_code=EXIT_CHECK_FAILED,
            fix="python -m tools.preflight run --spec <spec> --json   # runs the local checks first",
            detail=payload,
        )
    if failing:
        first = results[failing[0]]
        raise GradError(
            "preflight_failed",
            f"{len(failing)} check(s) failed: {', '.join(failing)}",
            exit_code=EXIT_CHECK_FAILED,
            fix=first.get("fix") or f"read the log: {first.get('log')}",
            detail=payload,
        )
    return payload


def _run_checks(
    names: list[str],
    sub: Submission,
    cfg: config_mod.Config,
    declared: dict[str, Any],
    *,
    jobs: int,
) -> dict[str, Any]:
    """Run the local checks, at most `jobs` at once, and return them by name.

    Threads rather than processes: every check is a `subprocess.run` and the
    thread spends its life blocked in `wait`, so the GIL never enters into it.

    These are genuinely independent -- `tests` runs the suite, `dry_run` runs the
    entrypoint on a tiny config, and a declared check is whatever the spec says.
    None of them reads another's result, and each writes its own log path. The
    one check that is *not* independent is `smoke`, which is why it is not here.

    Results come back keyed by name rather than in completion order, so the record
    is identical whatever order they finish in.
    """
    if not names:
        return {}
    if jobs <= 1 or len(names) == 1:
        return {name: _run_check(name, sub, cfg, declared) for name in names}
    with ThreadPoolExecutor(max_workers=min(jobs, len(names)), thread_name_prefix="grad-pre") as pool:
        futures = {name: pool.submit(_run_check, name, sub, cfg, declared) for name in names}
        return {name: future.result() for name, future in futures.items()}


def _declared_checks(sub: Submission) -> dict[str, Any]:
    raw = sub.config.get("checks")
    if isinstance(raw, dict):
        return raw
    return {}


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _run_check(name: str, sub: Submission, cfg: config_mod.Config, declared: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    try:
        if name == "tests":
            result = _check_tests(sub, cfg)
        elif name == "dry_run":
            result = _check_dry_run(sub, cfg)
        elif name == "smoke":
            result = _check_smoke(sub, cfg)
        elif name == "cost":
            result = _check_cost(sub, cfg)
        elif name in declared:
            result = _check_command(name, declared[name], sub, cfg)
        else:
            result = {
                "ok": False,
                "reason": f"check {name!r} is not built in and is not declared in the spec",
                "fix": f'declare it: [config.checks]\n{name} = "pytest -q tests/test_{name}.py"',
            }
    except GradError as exc:
        result = {"ok": False, "reason": exc.message, "fix": exc.fix}
    result["at"] = now_iso()
    result["duration_s"] = round(time.time() - started, 2)
    return result


def _log_path(sub: Submission, name: str) -> Path:
    d = paths.preflight_dir() / sub.hash()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"


def _exec(argv: list[str], cwd: Path, timeout: float, log: Path, env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    import os

    env = {**os.environ, **(env_extra or {})}
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
        timed_out = False
    except FileNotFoundError as exc:
        return {"ok": False, "reason": f"command not found: {argv[0]} ({exc})",
                "fix": f"install {argv[0]} or fix the command in config/grad.toml"}
    except subprocess.TimeoutExpired as exc:
        # Each stream independently: a process that wrote only to stderr before
        # the timeout leaves `exc.stdout` as None, and gating the whole
        # expression on it would write an empty log while the error still tells
        # the operator to go read that log.
        def _text(stream: Any) -> str:
            if stream is None:
                return ""
            return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")

        output = _text(exc.stdout) + _text(exc.stderr) + f"\n[preflight] timed out after {timeout}s\n"
        code = -1
        timed_out = True

    log.write_text(output, encoding="utf-8")
    tail = "\n".join(output.splitlines()[-25:])
    return {
        "ok": code == 0 and not timed_out,
        "exit_code": code,
        "timed_out": timed_out,
        "command": " ".join(shlex.quote(a) for a in argv),
        "log": str(log),
        "output": tail,
        "fix": None if code == 0 and not timed_out else f"read the full log: {log}",
    }


def _check_tests(sub: Submission, cfg: config_mod.Config) -> dict[str, Any]:
    """Regressions in pipeline code."""
    argv = list(cfg.get("preflight", "test_command", ["pytest", "-q"]))
    return _exec(
        argv,
        sub.spec_path.parent,
        float(cfg.get("preflight", "test_timeout_s", 900)),
        _log_path(sub, "tests"),
    )


def _check_dry_run(sub: Submission, cfg: config_mod.Config) -> dict[str, Any]:
    """The fast filter: same entrypoint, 1 step, batch 2, tiny model, locally.

    It proves the code is internally coherent and nothing more. The earlier
    draft of the handoff claimed it catches missing dependencies and OOM; those
    are exactly what a local tiny-model run cannot see, which is why `smoke`
    exists.
    """
    import sys

    dry = sub.config.get("dry_run", {})
    extra = [str(a) for a in dry.get("argv", ["--steps", "1", "--batch-size", "2", "--max-samples", "10"])]
    argv = [sys.executable, str(sub.entrypoint), *sub.argv, *extra]
    result = _exec(
        argv,
        sub.spec_path.parent,
        float(cfg.get("preflight", "dry_run_timeout_s", 900)),
        _log_path(sub, "dry_run"),
        env_extra={"GRAD_DRY_RUN": "1"},
    )
    result["scope"] = "local; proves internal coherence only"
    return result


def _check_smoke(sub: Submission, cfg: config_mod.Config) -> dict[str, Any]:
    """The single highest-value check: one step on the real target.

    Smoke is itself a paid remote job and must go through the submitters, which
    is the bootstrap problem §6 calls out. It is resolved by the submitters'
    `--smoke` path: gate-exempt, hard-capped in code, result written back into
    this record.
    """
    platform = (sub.target.get("platform") or "").lower()
    if platform in ("hf", "hf_jobs", "huggingface"):
        from tools import jobs as submitter
    elif platform in ("ssh", "gpu"):
        from tools import gpu as submitter  # type: ignore[no-redef]
    elif platform == "kaggle":
        from tools import kaggle as submitter  # type: ignore[no-redef]
    else:
        return {
            "ok": False,
            "reason": f"target.platform is {platform!r}; cannot smoke without a real target",
            "fix": 'set [target] platform = "hf", "ssh", or "kaggle" in the submission spec',
        }
    # Positionally, with no backend-specific arguments. That is the contract this
    # dispatch rests on and the reason every `run_smoke` resolves its own target
    # from the spec: a backend whose smoke needed a parameter from here could be
    # submitted but never preflighted, and gate 1 would then refuse every one of
    # its jobs for a reason that had nothing to do with the job.
    return submitter.run_smoke(sub, cfg)


def _check_cost(sub: Submission, cfg: config_mod.Config) -> dict[str, Any]:
    """Surprise bills, this job and cumulatively."""
    estimate = sub.estimated_cost_usd()
    try:
        detail = gates.check_spend(estimate, cfg)
    except GradError as exc:
        return {"ok": False, "reason": exc.message, "fix": exc.fix, "detail": exc.detail}
    return {"ok": True, "estimate_usd": estimate, **detail}


def _check_command(name: str, command: Any, sub: Submission, cfg: config_mod.Config) -> dict[str, Any]:
    """A check the pipeline declares itself: shapes, grads, symbolic, invariants.

    Preflight cannot know how a given pipeline asserts equivariance or checks a
    hand-written gradient, so it runs what the spec declares and reports the
    exit code. `torch.autograd.gradcheck`, `hypothesis`, `einops`/`jaxtyping`
    assertions, and SymPy comparisons all fit this shape.
    """
    argv = command if isinstance(command, list) else shlex.split(str(command))
    return _exec(argv, sub.spec_path.parent, 900.0, _log_path(sub, name))


# ---------------------------------------------------------------------------
def _show_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", help="show the record for this spec's current hash")
    p.add_argument("--hash", dest="hash_", help="show the record for an explicit hash")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--no-digest", action="store_true")


@cli.command("show", "show a preflight record", setup=_show_args)
def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    if args.hash_:
        h = args.hash_
    elif args.spec:
        h = _load(args).hash()
    else:
        raise UsageError("give --spec or --hash", fix="grad-preflight show --spec pipeline/spec.toml")
    record = jsonl.read_json(paths.preflight_record(h))
    if record is None:
        return {"submission_hash": h, "exists": False,
                "fix": "python -m tools.preflight run --spec <spec> --json"}
    return {"submission_hash": h, "exists": True, "record": record}


@cli.command("list", "list preflight records on disk")
def cmd_list(_: argparse.Namespace) -> dict[str, Any]:
    out = []
    for path in sorted(paths.preflight_dir().glob("*.json")):
        rec = jsonl.read_json(path) or {}
        checks = rec.get("checks", {})
        out.append(
            {
                "submission_hash": rec.get("submission_hash", path.stem),
                "verified_at": rec.get("verified_at"),
                "spec": rec.get("spec"),
                "passing": [n for n, r in checks.items() if r.get("ok")],
                "failing": [n for n, r in checks.items() if r.get("ok") is False],
            }
        )
    return {"records": out}


# ---------------------------------------------------------------------------
def record_check_result(submission_hash: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Write one check's result into a (possibly not yet existing) record.

    Used by the submitters to fold a smoke result back into the pending
    preflight record for the submission it validates.

    Locked read-modify-write: a submitter folding a smoke result here while
    `preflight run` writes its own checks would otherwise drop one of the two
    sets, and a record missing a check is a record the gate refuses on -- or,
    worse, one whose missing check nobody notices.
    """

    def _fold(current: dict[str, Any] | None) -> dict[str, Any]:
        record = current or {"submission_hash": submission_hash, "checks": {}}
        if not isinstance(record.get("checks"), dict):
            record["checks"] = {}
        record["checks"][name] = {**result, "at": now_iso()}
        record["verified_at"] = now_iso()
        return record

    return jsonl.update_json(paths.preflight_record(submission_hash), _fold)


if __name__ == "__main__":
    main(cli)
