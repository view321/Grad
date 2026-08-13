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

    for name in wanted:
        if not args.force and results.get(name, {}).get("ok"):
            results[name]["skipped_because"] = "already passing for this hash"
            continue
        results[name] = _run_check(name, sub, cfg, spec_checks)

    record = {
        "submission_hash": h,
        "full_hash": sub.full_hash(),
        "spec": str(sub.spec_path),
        "verified_at": now_iso(),
        "resolved": sub.resolved(),
        "checks": results,
        "warnings": sub.warnings,
        "estimate_usd": sub.estimated_cost_usd(),
        "estimated_duration_s": sub.estimated_duration_s(),
    }
    jsonl.write_json(paths.preflight_record(h), record)

    failing = [n for n, r in results.items() if r.get("ok") is False]
    payload = {
        "submission_hash": h,
        "record": str(paths.preflight_record(h)),
        "checks": {n: {k: v for k, v in r.items() if k != "output"} for n, r in results.items()},
        "failing": failing,
        "warnings": sub.warnings,
    }
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
    else:
        return {
            "ok": False,
            "reason": f"target.platform is {platform!r}; cannot smoke without a real target",
            "fix": 'set [target] platform = "hf" or "ssh" in the submission spec',
        }
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
    """
    path = paths.preflight_record(submission_hash)
    record = jsonl.read_json(path) or {"submission_hash": submission_hash, "checks": {}}
    record.setdefault("checks", {})[name] = {**result, "at": now_iso()}
    record["verified_at"] = now_iso()
    jsonl.write_json(path, record)
    return record


if __name__ == "__main__":
    main(cli)
