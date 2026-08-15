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
import sys
import time
from pathlib import Path
from typing import Any

from core import (
    budget,
    config as config_mod,
    credentials,
    gates,
    ledger_store as ls,
    submit as submit_lib,
)
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
    if not token:
        # Not an assert: `python -O` strips those, and the failure mode there is
        # passing None as the token and getting an opaque upstream error instead
        # of "you have no credential stored".
        raise ConfigError(
            "no Hugging Face token is stored, so HF Jobs cannot be reached",
            fix=f"python -m tools.jobs credential set {credentials.HF_TOKEN}",
        )
    return token


# ---------------------------------------------------------------------------
# organization namespace (HANDOFF-2 §17)
# ---------------------------------------------------------------------------
# The trap this section exists to avoid: `namespace` is a property of the job
# *handle*, not a submit-time parameter. Passing it only to `run_job` produces a
# job that cannot be found again -- `inspect_job` and `fetch_job_logs` look
# under the personal namespace and 404, the run never collects, goes stale, and
# then blocks every future submission through the §6 stale-run gate. The failure
# appears very far from its cause.
#
# So it is persisted onto the handle at submit and threaded through every call
# that takes one. `_ns_kwargs` is the single place that decides how to pass it.
def _ns_kwargs(namespace: str | None) -> dict[str, Any]:
    """Namespace kwargs for a hub call, or nothing at all when personal.

    Omitted rather than passed as None so an older `huggingface_hub` without the
    parameter still works for personal jobs; a *requested* namespace on such a
    version is a hard error rather than a silently personal job.
    """
    if not namespace:
        return {}
    import inspect  # noqa: PLC0415

    hub = _hub()
    for fn in (hub.run_job, hub.inspect_job, hub.fetch_job_logs):
        try:
            if "namespace" not in inspect.signature(fn).parameters:
                raise ConfigError(
                    f"the installed huggingface_hub's {fn.__name__}() takes no `namespace`, "
                    "so a job submitted to an organization could not be collected from it",
                    fix="pip install -U 'huggingface_hub>=1.16'   # or drop --namespace",
                )
        except (TypeError, ValueError):
            # Some builds wrap these; an unreadable signature is not evidence
            # the parameter is missing, so it is not treated as such.
            continue
    return {"namespace": namespace}


def resolve_namespace(
    flag: str | None, sub: Submission | None, cfg: Config, project_id: str | None
) -> str | None:
    """`--namespace` -> spec `[target] namespace` -> project payer -> `[hf] namespace` -> personal.

    Mirrors how `flavor` already resolves, and the project step is why §15's
    `payer` lives on the project: org attribution becomes a consequence of
    choosing a project rather than a flag to forget.
    """
    if flag:
        return flag
    if sub is not None and sub.target.get("namespace"):
        return str(sub.target["namespace"])
    from_project = budget.hf_namespace(project_id)
    if from_project:
        return from_project
    return cfg.get("hf", "namespace") or None


def validate_namespace(namespace: str | None, token: str) -> dict[str, Any]:
    """Check membership *before* `record_submission`.

    Deliberately in the same place `_hub()` and `_token()` are already called,
    and for the same reason: a configuration problem must not leave a phantom
    estimate sitting on the ceiling. One network call per submit is acceptable
    on a path about to spend dollars, and it is not cached aggressively --
    org membership changing is precisely the case worth catching.
    """
    hub = _hub()
    try:
        me = hub.whoami(token=token) or {}
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            f"could not read the token's identity from Hugging Face: {exc}",
            fix="check that the stored hf_token is valid and not expired",
        ) from exc
    user = me.get("name") or me.get("user")
    orgs = [
        str(e.get("name") if isinstance(e, dict) else e)
        for e in (me.get("orgs") or [])
        if (e.get("name") if isinstance(e, dict) else e)
    ]
    if namespace and namespace != user and namespace not in orgs:
        raise ConfigError(
            f"this token cannot act for namespace {namespace!r}; it is {user!r} and a member of: "
            + (", ".join(orgs) or "(no organizations)"),
            fix=(
                "pick a namespace the token belongs to, or store a token with that org's scope: "
                f"python -m tools.jobs credential set {credentials.HF_TOKEN}"
            ),
        )
    return {"user": user, "orgs": orgs, "namespace": namespace}


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
        "--project",
        help="project to charge this run to (defaults to the current one; §15)",
    )
    p.add_argument(
        "--namespace",
        help="HF organization to run under. Resolution: this flag, the spec's "
        "[target] namespace, the project's payer, [hf] namespace, then personal.",
    )
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

    project_id = budget.resolve(args.project)
    namespace = resolve_namespace(args.namespace, sub, cfg, project_id)

    if args.smoke:
        if args.expect:
            raise UsageError(
                "--smoke does not take --expect: a smoke check is not a result and binds no prediction",
                fix="drop --expect, or drop --smoke",
            )
        result = run_smoke(sub, cfg, namespace=namespace, project=project_id)
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
    summary = submit_lib.check(sub, args.expect, cfg, project=project_id)
    # Then the backend and the credential, before any record exists: a missing
    # package or an absent token is a configuration problem, not an in-flight
    # job, and it must not leave a phantom estimate sitting on the ceiling. The
    # namespace check joins them for exactly that reason -- an org the token
    # cannot act for is a configuration problem too.
    hub = _hub()
    token = _token()
    identity = validate_namespace(namespace, token)
    warnings = _namespace_warnings(sub, namespace)

    run_id, _ = submit_lib.record_submission(
        sub,
        expectation_id=args.expect,
        platform=PLATFORM,
        target={"flavor": flavor, "platform": "hf", "namespace": namespace},
        command=command,
        task=args.task,
        project=project_id,
    )

    try:
        job = hub.run_job(
            image=sub.image,
            command=command,
            flavor=flavor,
            env=_job_env(sub),
            secrets=None,
            token=token,
            timeout=sub.target.get("timeout"),
            **_ns_kwargs(namespace),
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
    # The namespace goes onto the *handle*, not just into the submit call. This
    # is the whole point of §17: every later `inspect_job` / `fetch_job_logs`
    # reads it from here, so a job submitted to an org is collectable from it.
    submit_lib.attach_handle(
        run_id, {"job_id": job_id, "flavor": flavor, "namespace": namespace}
    )
    return {
        "run_id": run_id,
        "job_id": job_id,
        "flavor": flavor,
        "namespace": namespace,
        "identity": identity,
        "project": project_id,
        "gates": summary,
        "warnings": warnings,
        "next": f"python -m tools.jobs collect {run_id} --json",
    }


def _namespace_warnings(sub: Submission, namespace: str | None) -> list[str]:
    """Warn -- not refuse -- when smoke ran under a different namespace.

    The submission hash deliberately excludes `target` (which is why `flavor` is
    not hashed), and `namespace` follows the same rule for consistency. The
    consequence is real and handled rather than ignored: a preflight whose
    `smoke` check ran under personal credentials validates a job that will run
    in an organization. Warn, not refuse, consistent with how `target` and
    `flavor` already behave.
    """
    record = gates.preflight_record(sub.hash()) or {}
    smoke = (record.get("checks") or {}).get("smoke") or {}
    if "namespace" not in smoke:
        return []
    used = smoke.get("namespace")
    if used == namespace:
        return []
    return [
        f"the smoke check ran under namespace {used or 'personal'} but this job runs under "
        f"{namespace or 'personal'}; the environment it validated may differ "
        "(namespace is not part of the submission hash, by the same rule as flavor)"
    ]


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
def run_smoke(
    sub: Submission,
    cfg: Config,
    *,
    namespace: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
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

    # Resolved once, and used for both the namespace and the accounting.
    # `preflight.py` calls this with no project at all, and deriving the
    # namespace from the current project while booking the cost as `unassigned`
    # meant a smoke run charged an org's job to nobody -- the two halves of the
    # same decision disagreeing.
    project = project or budget.current_project()
    if namespace is None:
        namespace = resolve_namespace(None, sub, cfg, project)

    # Everything that can fail for a *configuration* reason resolves before the
    # ledger record exists. `_hub()` raises when huggingface_hub is missing,
    # `_ns_kwargs()` raises when the installed one cannot take a namespace, and
    # `_token()` raises when no credential is stored; any of them landing after
    # `record_smoke_run` leaves a phantom in-flight estimate sitting on the
    # monthly ceiling for a job that never reached the platform -- which then
    # goes stale and blocks every later submission.
    hub = _hub()
    ns_kwargs = _ns_kwargs(namespace)
    token = _token()

    run_id = submit_lib.record_smoke_run(
        sub, cfg=cfg, platform=PLATFORM,
        target={"flavor": flavor, "platform": "hf", "namespace": namespace},
        caps=caps, command=command, project=project,
    )
    artifacts = submit_lib.artifacts_dir(run_id)

    try:
        job = hub.run_job(
            image=sub.image,
            command=command,
            flavor=flavor,
            env={**_job_env(sub), "GRAD_SMOKE": "1"},
            token=token,
            timeout=caps["timeout_s"],
            **ns_kwargs,
        )
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        submit_lib.finish(
            run_id, status="submit_failed", results={}, cost_usd_actual=0.0,
            artifacts_dir=artifacts, expectation=None, extra={"error": str(exc)},
        )
        return {"ok": False, "reason": f"smoke submission failed: {exc}",
                "fix": "check the HF token scope and the image digest", "run_id": run_id,
                "namespace": namespace}

    job_id = getattr(job, "id", None) or getattr(job, "job_id", None) or str(job)
    submit_lib.attach_handle(run_id, {"job_id": job_id, "flavor": flavor, "namespace": namespace})

    state, info = _poll(job_id, deadline=time.time() + caps["timeout_s"], namespace=namespace)
    logs = _logs(job_id, namespace=namespace)
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
        # Recorded into the check result, which flows into the preflight record.
        # `submit` compares it against the namespace the real job will use and
        # warns on a difference -- see `_namespace_warnings`.
        "namespace": namespace,
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
def _poll(job_id: str, *, deadline: float, namespace: str | None = None) -> tuple[str, Any]:
    hub = _hub()
    ns = _ns_kwargs(namespace)
    info: Any = None
    state = "UNKNOWN"
    while True:
        info = hub.inspect_job(job_id=job_id, token=_token(), **ns)
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


def _logs(job_id: str, *, namespace: str | None = None) -> str:
    try:
        return "\n".join(
            str(line)
            for line in _hub().fetch_job_logs(
                job_id=job_id, token=_token(), **_ns_kwargs(namespace)
            )
        )
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
        "project": r.project,
        "namespace": handle.get("namespace"),
    }
    if handle.get("job_id") and not r.collected:
        try:
            # From the handle, not re-resolved: the config or the current project
            # may have changed since submit, and looking under a namespace the
            # job was not submitted to is exactly the 404 that makes a run go
            # stale and block every later submission.
            info = _hub().inspect_job(
                job_id=handle["job_id"], token=_token(), **_ns_kwargs(handle.get("namespace"))
            )
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

    namespace = handle.get("namespace")
    deadline = time.time() + (args.timeout if args.wait else 0)
    state, info = _poll(job_id, deadline=deadline, namespace=namespace)
    if state not in ("COMPLETED", "ERROR", "CANCELED", "FAILED"):
        raise GradError(
            "still_running",
            f"job {job_id} is {state}",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.jobs collect {r.id} --wait --timeout 3600 --json",
            detail={"run_id": r.id, "state": state},
        )

    artifacts = submit_lib.artifacts_dir(r.id)
    logs = _logs(job_id, namespace=namespace)
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
    credentials.CONTEXT7_KEY,
    credentials.CLAUDE_TOKEN,
    credentials.ASTA_KEY,
)


def _credential_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=["status", "set", "delete"])
    p.add_argument("name", nargs="?", help=f"one of: {', '.join(CREDENTIAL_NAMES)}")
    p.add_argument(
        "--stdin",
        action="store_true",
        help="read the value from stdin rather than prompting (for the workspace UI)",
    )


@cli.command("credential", "inspect or set stored credentials (values are never printed)", setup=_credential_args)
def cmd_credential(args: argparse.Namespace) -> dict[str, Any]:
    """Credentials live in Windows Credential Manager, never in the environment
    and never in a file under the workspace."""
    if args.action == "status":
        payload: dict[str, Any] = {
            "credentials": credentials.status(),
            "service": credentials.SERVICE,
        }
        # Which namespaces the HF token can actually act for (§17). Surfaced
        # here so "the org submit failed" is diagnosable without a submission.
        if credentials.present(credentials.HF_TOKEN):
            try:
                payload["hf_identity"] = validate_namespace(None, _token())
            except GradError as exc:
                payload["hf_identity"] = {"error": exc.message, "fix": exc.fix}
        return payload
    if not args.name:
        raise UsageError("give a credential name", fix=f"one of: {', '.join(CREDENTIAL_NAMES)}")
    if args.action == "delete":
        credentials.delete(args.name)
        return {"deleted": args.name}
    if args.name not in CREDENTIAL_NAMES:
        raise UsageError(
            f"unknown credential {args.name!r}",
            fix=f"one of: {', '.join(CREDENTIAL_NAMES)}",
        )

    if args.stdin:
        # A pipe, not an argument. The value is a token, and an argv is visible
        # to anything that can list processes -- which on a shared machine is
        # everything. `getpass` is the same guarantee for a human at a terminal;
        # this is the guarantee for a caller that has no terminal to prompt at,
        # which is what the workspace's credential panel is.
        value = sys.stdin.read().strip()
    else:
        import getpass

        value = getpass.getpass(f"value for {args.name} (not echoed): ")
    if not value:
        raise UsageError("empty value", fix="run it again and paste the token")
    credentials.set_(args.name, value)
    return {"stored": args.name}


@cli.command(
    "ceilings",
    "show the spend ceilings and current rolling total",
    setup=lambda p: p.add_argument("--project", help="also show this project's allocation"),
)
def cmd_ceilings(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    window = int(cfg.get("spend", "window_days", 30))
    rolling = ls.rolling_spend(window)
    stale = [r.id for r in ls.stale_runs(cfg=cfg)]
    payload: dict[str, Any] = {
        "per_job_usd": cfg.get("spend", "per_job_usd"),
        "monthly_usd": cfg.get("spend", "monthly_usd"),
        "rolling": {k: v for k, v in rolling.items() if k != "runs"},
        "in_flight_runs": [r.id for r in ls.in_flight()],
        "stale_runs": stale,
        "blocked": bool(stale),
    }
    # Two ceilings, never conflated: the machine's, and this research's (§15).
    project_id = budget.resolve(args.project)
    if project_id and budget.exists(project_id):
        payload["project"] = budget.status(project_id)
    else:
        payload["project"] = {"project": project_id, "bounded": False}
    return payload


if __name__ == "__main__":
    main(cli)
