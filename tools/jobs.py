"""grad-jobs -- submit, watch, and collect Hugging Face Jobs (HANDOFF §6, §7).

This is one of the only two paths in the system that can authenticate to a
remote machine (`gpu.py` is the other). That is the actual security control:
the `PreToolUse` hook denying bare `hf` is a speed bump, but the HF token living
in Windows Credential Manager and never in the agent's environment is a wall.

Four gates run before anything costs money, and there is no flag that disables
them. `--smoke` is not that flag: it is a separate, hard-capped path.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any

from core import config as config_mod, credentials, gates, ledger_store as ls, submit as submit_lib
from core.cli import Cli, main
from core.config import Config
from core.errors import ConfigError, EXIT_RUNNING, GradError, UpstreamError, UsageError
from core.submission import Submission, parse_override

cli = Cli(
    "grad-jobs",
    "Submit and collect Hugging Face Jobs. Refuses to submit without a passing "
    "preflight, an open expectation, and headroom under both spend ceilings.",
    epilog=(
        "gate refusals have their own exit codes (4 preflight, 5 expectation, 6 spend,\n"
        "7 stale run) so a refusal is never confused with an upstream failure.\n\n"
        "`collect` is non-blocking by default: a two-hour poll inside the agent's only\n"
        "shell is a tool timeout waiting to happen. Use --wait --timeout to opt in."
    ),
)

PLATFORM = "hf_jobs"


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
def _hub() -> Any:
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "huggingface_hub is not installed, so HF Jobs cannot be reached",
            fix="pip install 'huggingface_hub>=0.24'",
        ) from exc
    for fn in ("run_job", "inspect_job", "fetch_job_logs"):
        if not hasattr(huggingface_hub, fn):
            raise ConfigError(
                f"the installed huggingface_hub has no {fn}(); the Jobs API is missing",
                fix="pip install -U 'huggingface_hub>=0.24'",
            )
    return huggingface_hub


def _token() -> str:
    """Fetched at the moment of use and never exported (HANDOFF §9)."""
    token = credentials.get(credentials.HF_TOKEN)
    assert token  # credentials.get raises when required and missing
    return token


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
def _submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", required=True, help="path to the submission spec")
    p.add_argument(
        "--expect",
        help="expectation id to bind to this run. REQUIRED unless --smoke: "
        "no pre-registration, no submission",
    )
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--flavor", help="HF Jobs hardware flavor (overrides the spec)")
    p.add_argument("--task", help="task id for the ledger (defaults to the spec directory name)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="the gate-exempt, hard-capped one-step check from §6. Cannot train anything.",
    )
    p.add_argument("--no-digest", action="store_true", help=argparse.SUPPRESS)


@cli.command("submit", "submit a job (gated) or a smoke check (capped)", setup=_submit_args)
def cmd_submit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    sub = Submission.load(
        args.spec,
        overrides=dict(parse_override(o) for o in args.overrides),
        resolve_digest=not args.no_digest,
    )
    if args.flavor:
        sub.target["flavor"] = args.flavor

    if args.smoke:
        if args.expect:
            raise UsageError(
                "--smoke does not take --expect: a smoke check is not a result and binds no prediction",
                fix="drop --expect, or drop --smoke",
            )
        result = run_smoke(sub, cfg)
        from tools import preflight

        preflight.record_check_result(sub.hash(), "smoke", result)
        if not result.get("ok"):
            raise GradError(
                "smoke_failed",
                result.get("reason", "the smoke check failed on the real target"),
                exit_code=9,
                fix=result.get("fix") or "read the smoke log under ledger/runs/",
                detail=result,
            )
        return {"smoke": result, "submission_hash": sub.hash()}

    flavor = sub.target.get("flavor") or cfg.get("hf", "default_flavor", "a10g-small")
    command = _command_for(sub)

    # Gates first: a refusal is the most actionable thing a submitter can say.
    summary = submit_lib.check(sub, args.expect, cfg)
    # Then the backend and the credential, before any record exists: a missing
    # package or an absent token is a configuration problem, not an in-flight
    # job, and it must not leave a phantom estimate sitting on the ceiling.
    hub = _hub()
    _token()
    run_id, _ = submit_lib.record_submission(
        sub,
        expectation_id=args.expect,
        platform=PLATFORM,
        target={"flavor": flavor, "platform": "hf"},
        command=command,
        task=args.task,
    )

    try:
        job = hub.run_job(
            image=sub.image,
            command=command,
            flavor=flavor,
            env=_job_env(sub),
            secrets=None,
            token=_token(),
            timeout=sub.target.get("timeout"),
        )
    except Exception as exc:  # noqa: BLE001 - hub raises a wide family of errors
        submit_lib.finish(
            run_id,
            status="submit_failed",
            results={},
            cost_usd_actual=0.0,
            artifacts_dir=submit_lib.artifacts_dir(run_id),
            expectation=None,
            extra={"error": str(exc)},
        )
        raise UpstreamError(
            f"HF Jobs refused the submission: {exc}",
            fix="check the token scope and the image digest, then resubmit",
        ) from exc

    job_id = getattr(job, "id", None) or getattr(job, "job_id", None) or str(job)
    submit_lib.attach_handle(run_id, {"job_id": job_id, "flavor": flavor})
    return {
        "run_id": run_id,
        "job_id": job_id,
        "flavor": flavor,
        "gates": summary,
        "next": f"python -m tools.jobs collect {run_id} --json",
    }


def _command_for(sub: Submission) -> list[str]:
    if sub.target.get("command"):
        return [str(c) for c in sub.target["command"]]
    entry = sub.entrypoint.name
    return ["python", entry, *sub.argv]


def _job_env(sub: Submission) -> dict[str, str]:
    env = {str(k): str(v) for k, v in (sub.target.get("env") or {}).items()}
    env["GRAD_METRICS_FILE"] = sub.metrics_file
    return env


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def run_smoke(sub: Submission, cfg: Config) -> dict[str, Any]:
    """The §6 carve-out: gate-exempt, hard-capped, still ledgered.

    The caps are applied here in code -- one step, a wall-clock ceiling of
    minutes, a cost ceiling of cents, no artifact upload -- rather than trusted
    to the caller. Nothing useful can be trained inside them, which is what
    keeps the exemption from becoming the way real jobs escape the gate.

    Unlike a real submission this blocks, because it is bounded to minutes by
    construction and preflight needs the answer.
    """
    caps = gates.check_smoke_caps(sub, cfg)
    flavor = sub.target.get("smoke_flavor") or sub.target.get("flavor") or cfg.get("hf", "default_flavor", "a10g-small")
    command = _smoke_command(sub, caps)
    run_id = submit_lib.record_smoke_run(
        sub, cfg=cfg, platform=PLATFORM, target={"flavor": flavor, "platform": "hf"},
        caps=caps, command=command,
    )
    artifacts = submit_lib.artifacts_dir(run_id)

    try:
        hub = _hub()
        job = hub.run_job(
            image=sub.image,
            command=command,
            flavor=flavor,
            env={**_job_env(sub), "GRAD_SMOKE": "1"},
            token=_token(),
            timeout=caps["timeout_s"],
        )
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        submit_lib.finish(
            run_id, status="submit_failed", results={}, cost_usd_actual=0.0,
            artifacts_dir=artifacts, expectation=None, extra={"error": str(exc)},
        )
        return {"ok": False, "reason": f"smoke submission failed: {exc}",
                "fix": "check the HF token scope and the image digest", "run_id": run_id}

    job_id = getattr(job, "id", None) or getattr(job, "job_id", None) or str(job)
    submit_lib.attach_handle(run_id, {"job_id": job_id, "flavor": flavor})

    state, info = _poll(job_id, deadline=time.time() + caps["timeout_s"])
    logs = _logs(job_id)
    (artifacts / "smoke.log").write_text(logs, encoding="utf-8")
    cost = _actual_cost(info, flavor, cfg)
    ok = state == "COMPLETED"
    submit_lib.finish(
        run_id,
        status="completed" if ok else "failed",
        results={},
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=None,
        extra={"job_state": state, "smoke": True},
    )
    return {
        "ok": ok,
        "run_id": run_id,
        "job_id": job_id,
        "state": state,
        "flavor": flavor,
        "cost_usd": cost,
        "caps": caps,
        "log": str(artifacts / "smoke.log"),
        "output": "\n".join(logs.splitlines()[-25:]),
        "reason": None if ok else f"the smoke job ended in state {state}",
        "fix": None if ok else f"read {artifacts / 'smoke.log'} -- this is the environment the real job would have used",
        "scope": "remote; the only check that exercises the real image, data path, and hardware",
    }


def _smoke_command(sub: Submission, caps: dict[str, Any]) -> list[str]:
    """One step, real per-device batch size, truncated sequence count.

    §6 is specific about this: running smoke at batch 2 does not test the thing
    that most often kills the real run.
    """
    base = _command_for(sub)
    return [*base, "--steps", str(caps["steps"]), "--smoke"]


# ---------------------------------------------------------------------------
# status / collect
# ---------------------------------------------------------------------------
def _poll(job_id: str, *, deadline: float) -> tuple[str, Any]:
    hub = _hub()
    info: Any = None
    state = "UNKNOWN"
    while True:
        info = hub.inspect_job(job_id=job_id, token=_token())
        state = _state_of(info)
        if state in ("COMPLETED", "ERROR", "CANCELED", "FAILED") or time.time() > deadline:
            return state, info
        time.sleep(5)


def _state_of(info: Any) -> str:
    stage = getattr(getattr(info, "status", None), "stage", None)
    if stage:
        return str(stage).upper()
    if isinstance(info, dict):
        status = info.get("status") or {}
        return str(status.get("stage") or info.get("stage") or "UNKNOWN").upper()
    return "UNKNOWN"


def _logs(job_id: str) -> str:
    try:
        return "\n".join(str(line) for line in _hub().fetch_job_logs(job_id=job_id, token=_token()))
    except Exception as exc:  # noqa: BLE001 - logs are best-effort; never fail a collect over them
        return f"(could not fetch logs: {exc})"


def _actual_cost(info: Any, flavor: str, cfg: Config) -> float:
    """Cost from the platform's own accounting of the run.

    HF reports the job's start and end timestamps; the price of a flavor comes
    from the rate table in config/grad.toml. The estimate is never reused here
    -- that is the whole point of collecting.
    """
    started = _ts(info, "started_at") or _ts(info, "created_at")
    ended = _ts(info, "ended_at") or _dt.datetime.now(_dt.timezone.utc)
    if not started:
        return 0.0
    hours = max(0.0, (ended - started).total_seconds() / 3600.0)
    rates = cfg.get("hf", "flavor_rates", {}) or {}
    rate = float(rates.get(flavor, 0.0))
    return round(hours * rate, 4)


def _ts(info: Any, field: str) -> _dt.datetime | None:
    value = getattr(info, field, None)
    if value is None and isinstance(info, dict):
        value = info.get(field)
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, str):
        return ls.parse_iso(value.replace("Z", "+00:00"))
    return None


@cli.command(
    "status",
    "report a run's state without collecting it",
    setup=lambda p: p.add_argument("run_id"),
)
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    r = ls.run(args.run_id)
    handle = r.get("handle") or {}
    payload: dict[str, Any] = {
        "run_id": r.id,
        "ledger_status": r.status,
        "collected": r.collected,
        "stale": ls.is_stale(r),
        "submitted_at": r.get("submitted_at"),
        "estimate_usd": r.get("estimate_usd"),
    }
    if handle.get("job_id") and not r.collected:
        try:
            info = _hub().inspect_job(job_id=handle["job_id"], token=_token())
            payload["remote_state"] = _state_of(info)
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            payload["remote_state"] = f"unavailable: {exc}"
    return payload


def _collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id")
    p.add_argument("--wait", action="store_true", help="block until the job finishes")
    p.add_argument("--timeout", type=int, default=900, help="seconds, with --wait")


@cli.command("collect", "fetch artifacts, compute deviations, write the run record", setup=_collect_args)
def cmd_collect(args: argparse.Namespace) -> dict[str, Any]:
    """Closes the loop that the model would otherwise close from memory.

    Writes results, actual cost, and the deviations array. Leaves `verdict`
    unset: that is `ledger.py verdict`'s job, and judgement must not be able to
    overwrite the record.
    """
    r = submit_lib.require_uncollected(args.run_id)
    handle = r.get("handle") or {}
    job_id = handle.get("job_id")
    if not job_id:
        raise GradError(
            "no_handle",
            f"run {r.id} has no HF job id; it never reached the platform",
            exit_code=3,
            fix=f"python -m tools.ledger show {r.id} --json",
        )

    deadline = time.time() + (args.timeout if args.wait else 0)
    state, info = _poll(job_id, deadline=deadline)
    if state not in ("COMPLETED", "ERROR", "CANCELED", "FAILED"):
        raise GradError(
            "still_running",
            f"job {job_id} is {state}",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.jobs collect {r.id} --wait --timeout 3600 --json",
            detail={"run_id": r.id, "state": state},
        )

    artifacts = submit_lib.artifacts_dir(r.id)
    logs = _logs(job_id)
    (artifacts / "job.log").write_text(logs, encoding="utf-8")

    results: dict[str, Any] = {}
    metrics_error = None
    metrics_path = artifacts / Path(r.get("metrics_file") or "metrics.json").name
    try:
        _download_artifacts(r, artifacts)
        results = submit_lib.parse_metrics(metrics_path)
    except GradError as exc:
        metrics_error = exc.message

    expectation = None
    if r.get("expectation_id"):
        try:
            expectation = ls.expectation(r["expectation_id"])
        except GradError:
            expectation = None

    cost = _actual_cost(info, (r.get("target") or {}).get("flavor", ""), config_mod.load())
    record = submit_lib.finish(
        r.id,
        status="completed" if state == "COMPLETED" else "failed",
        results=results,
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=expectation,
        extra={"job_state": state, "metrics_error": metrics_error},
    )
    unjudged = [d for d in record["deviations"] if d.get("in_range") is not True]
    return {
        "run": record,
        "artifacts": str(artifacts),
        "needs_verdict": unjudged,
        "next": (
            f"python -m tools.ledger verdict {r.id} --quantity {unjudged[0]['quantity']} "
            "--verdict bug|real|inconclusive --note '...' --json"
        ) if unjudged else None,
    }


def _download_artifacts(r: ls.Run, dest: Path) -> None:
    """Pull the metrics file and any declared artifacts out of the job's repo.

    HF Jobs have no artifact channel of their own, so the contract is that the
    pipeline uploads to a dataset/model repo named in the spec. If none is
    declared, the metrics file is expected to have been written into the log
    directory by the job's own uploader.
    """
    repo = (r.get("config") or {}).get("artifact_repo")
    if not repo:
        return
    hub = _hub()
    try:
        hub.snapshot_download(
            repo_id=repo,
            repo_type=(r.get("config") or {}).get("artifact_repo_type", "dataset"),
            local_dir=str(dest),
            token=_token(),
        )
    except Exception as exc:  # noqa: BLE001 - a missing artifact repo is reported, not fatal
        (dest / "artifact_download_error.txt").write_text(str(exc), encoding="utf-8")


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------
# Names only. `credentials.status()` probes the credential store, and argparse
# setup runs at import time -- listing it here would read all four entries out
# of Windows Credential Manager on every `jobs.py` invocation, including
# `collect` and `ceilings`.
CREDENTIAL_NAMES = (
    credentials.HF_TOKEN,
    credentials.OPENROUTER_KEY,
    credentials.VOYAGE_KEY,
    credentials.S2_KEY,
)


def _credential_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=["status", "set", "delete"])
    p.add_argument("name", nargs="?", help=f"one of: {', '.join(CREDENTIAL_NAMES)}")


@cli.command("credential", "inspect or set stored credentials (values are never printed)", setup=_credential_args)
def cmd_credential(args: argparse.Namespace) -> dict[str, Any]:
    """Credentials live in Windows Credential Manager, never in the environment
    and never in a file under the workspace."""
    if args.action == "status":
        return {"credentials": credentials.status(), "service": credentials.SERVICE}
    if not args.name:
        raise UsageError("give a credential name", fix=f"one of: {', '.join(CREDENTIAL_NAMES)}")
    if args.action == "delete":
        credentials.delete(args.name)
        return {"deleted": args.name}
    import getpass

    value = getpass.getpass(f"value for {args.name} (not echoed): ")
    if not value:
        raise UsageError("empty value", fix="run it again and paste the token")
    credentials.set_(args.name, value)
    return {"stored": args.name}


@cli.command("ceilings", "show the spend ceilings and current rolling total")
def cmd_ceilings(_: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    window = int(cfg.get("spend", "window_days", 30))
    rolling = ls.rolling_spend(window)
    stale = [r.id for r in ls.stale_runs(cfg=cfg)]
    return {
        "per_job_usd": cfg.get("spend", "per_job_usd"),
        "monthly_usd": cfg.get("spend", "monthly_usd"),
        "rolling": {k: v for k, v in rolling.items() if k != "runs"},
        "in_flight_runs": [r.id for r in ls.in_flight()],
        "stale_runs": stale,
        "blocked": bool(stale),
    }


if __name__ == "__main__":
    main(cli)
