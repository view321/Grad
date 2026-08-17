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
import shlex
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
        # Re-checks the spend ceilings inside the append lock, so two submitters
        # racing cannot both pass and both commit.
        cfg=cfg,
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
    # The flavor resolves before the caps, because the caps are computed against
    # its hourly rate: the wall clock is clamped to what the cost cap affords,
    # and an unpriced flavor is refused rather than assumed free.
    flavor = sub.target.get("smoke_flavor") or sub.target.get("flavor") or cfg.get("hf", "default_flavor", "a10g-small")
    caps = gates.check_smoke_caps(
        sub, cfg, rate_usd_per_hour=flavor_rate(flavor, cfg), target_name=f"flavor {flavor!r}"
    )
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
    cost, cost_warning = _actual_cost(
        info, flavor, cfg, estimate_usd=float(caps.get("projected_cost_usd") or 0.0)
    )
    ok = state == "COMPLETED"
    submit_lib.finish(
        run_id,
        status="completed" if ok else "failed",
        results={},
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=None,
        extra={
            "job_state": state,
            "smoke": True,
            "cost_warning": cost_warning,
            "cost_basis": "estimate" if cost_warning else "measured",
        },
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


# ---------------------------------------------------------------------------
# evolve candidates
# ---------------------------------------------------------------------------
#: How many bytes of a candidate's output are kept, matching `tools/gpu.py`.
CANDIDATE_OUTPUT_BYTES = 8000
#: The environment variable the candidate's files ride in. Named rather than
#: positional so a person reading a job's configuration on the Hub can see what
#: it is.
CANDIDATE_ENV = "GRAD_CANDIDATE_B64"
#: How large that blob may get. This is a *container environment variable*, not
#: a file: the practical ceiling is the platform's, it is not documented, and
#: discovering it by exceeding it means a job that fails for a reason with
#: nothing to do with the research. A couple of Python modules gzip to a few
#: kilobytes, so anything approaching this is a pipeline being smuggled through
#: the wrong door.
MAX_CANDIDATE_B64 = 200_000


def evaluate_candidate(
    sub: Submission,
    cfg: Config,
    *,
    candidate_id: str,
    files: dict[str, str],
    command: list[str],
    timeout_s: int,
    artifacts: Path,
    flavor: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Run one evolve candidate as a Hugging Face Job, and read its metrics back.

    **The delivery path is the interesting part, and it is different from the
    other two backends.** `gpu.py` copies the pipeline directory to a host and
    `kaggle.py` packs it into the notebook; on HF Jobs the pipeline is *in the
    image*, and there is no upload step at all -- `_command_for` just runs the
    entrypoint the image already contains. So a candidate, which is by definition
    a program the image does not contain, needs a way in.

    It rides as a gzipped tar in one environment variable, unpacked by a prelude
    in front of the command. That is the same shape as Kaggle's embedded payload
    and it is deliberately the *small* version of it: only the candidate's own
    files travel, because everything else is already in the image that the
    preflight proved. If that starts to look like a way to ship a pipeline, the
    size refusal above says so rather than letting it half-work.

    The files land in the container's working directory, which is the image's
    `WORKDIR` -- the same place `_command_for` runs the entrypoint from. A spec
    whose image puts the pipeline somewhere else needs `[target] command` to say
    so, exactly as it already would for a submission.

    Like the other adapters: no ledger row, cost measured rather than estimated,
    and a transport failure reported as one rather than as a candidate that
    scored nothing.
    """
    flavor = flavor or sub.target.get("flavor") or cfg.get("hf", "default_flavor", "a10g-small")
    project = budget.current_project()
    if namespace is None:
        namespace = resolve_namespace(None, sub, cfg, project)
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def _failed(message: str, **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "exit_code": None,
            "output": "",
            "error": message,
            "cost_usd": round(_elapsed_cost(started, flavor, cfg), 4),
            "flavor": flavor,
            "namespace": namespace,
            "where": f"hf:{namespace or ''}",
            **extra,
        }

    try:
        blob = _candidate_blob(files)
    except GradError as exc:
        return _failed(exc.message)

    # Everything that can fail for a configuration reason resolves before the
    # job exists, for the reason `run_smoke` gives about phantom estimates.
    try:
        hub = _hub()
        ns_kwargs = _ns_kwargs(namespace)
        token = _token()
    except GradError as exc:
        return _failed(exc.message)

    try:
        job = hub.run_job(
            image=sub.image,
            command=_candidate_command(command),
            flavor=flavor,
            env={**_job_env(sub), CANDIDATE_ENV: blob, "GRAD_CANDIDATE": candidate_id},
            token=token,
            timeout=int(timeout_s),
            **ns_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a refused submission is not a bad mutation
        return _failed(f"the candidate could not be submitted: {exc}")

    job_id = getattr(job, "id", None) or getattr(job, "job_id", None) or str(job)
    state, info = _poll(job_id, deadline=time.time() + int(timeout_s), namespace=namespace)
    logs = _logs(job_id, namespace=namespace)
    (artifacts / "candidate.log").write_text(logs, encoding="utf-8")
    cost, _warning = _actual_cost(info, flavor, cfg, estimate_usd=0.0)

    ok = state == "COMPLETED"
    return {
        "ok": ok,
        # HF reports a *state*, not an exit code. `0` on COMPLETED and `1`
        # otherwise would be inventing a number nobody measured, so the state is
        # what is reported and `exit_code` stays None -- which the driver already
        # distinguishes from a candidate that never ran.
        "exit_code": 0 if ok else None,
        "output": logs[-CANDIDATE_OUTPUT_BYTES:],
        "error": None if ok else f"the candidate's job ended in state {state}",
        "cost_usd": cost,
        "flavor": flavor,
        "namespace": namespace,
        "job_state": state,
        "where": f"hf:{namespace}/{job_id}" if namespace else f"hf:{job_id}",
    }


def _elapsed_cost(started: float, flavor: str, cfg: Config) -> float:
    rate = flavor_rate(flavor, cfg) or 0.0
    return (time.time() - started) / 3600.0 * float(rate)


def _candidate_blob(files: dict[str, str]) -> str:
    """The candidate's files as one base64 gzipped tar.

    Deterministic in the same way `kaggle.py:_payload_b64` is -- sorted names,
    `mtime=0` -- so the same candidate produces the same blob, which is what
    makes two job configurations comparable when one of them misbehaves.
    """
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as tar:
        for name in sorted(files):
            if name.startswith("/") or ".." in Path(name).parts:
                raise UsageError(
                    f"refusing to send {name!r}: a candidate file is a path inside the workdir",
                    fix="pass a name relative to the image's working directory",
                )
            data = str(files[name]).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    blob = base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(blob) > MAX_CANDIDATE_B64:
        raise UsageError(
            f"the candidate packs to {len(blob):,} base64 bytes, past the "
            f"{MAX_CANDIDATE_B64:,} this backend will put in an environment variable",
            fix=(
                "keep the evolve block to code -- the pipeline belongs in the image the "
                "preflight proved, not in the candidate"
            ),
        )
    return blob


#: Unpacks `CANDIDATE_ENV` into the working directory. A `python -c` rather than
#: a shell one-liner because the payload is base64 and `base64 -d` is not on
#: every image; Python is, by construction, since the entrypoint is Python.
_UNPACK = (
    "import base64,io,os,tarfile;"
    f"b=os.environ['{CANDIDATE_ENV}'];"
    "t=tarfile.open(fileobj=io.BytesIO(base64.b64decode(b)));"
    # `filter='data'` where the interpreter has it. The image's Python is not
    # this machine's, so the version that matters cannot be checked from here --
    # the same reason `kaggle.py` names it conditionally.
    "t.extractall('.', filter='data') if hasattr(tarfile,'data_filter') else t.extractall('.');"
    "print('grad: unpacked', len(t.getnames()), 'candidate files')"
)


def _candidate_command(command: list[str]) -> list[str]:
    """The command with the unpack in front of it.

    `sh -c` with the two joined by `&&`, so a failed unpack is a failed job
    rather than a job that runs the image's *own* entrypoint against the
    candidate's name and reports a score for the wrong program. That is the
    failure worth engineering against here: it would not look like an error, it
    would look like every candidate scoring the same.
    """
    inner = " ".join(shlex.quote(c) for c in command)
    return ["sh", "-c", f"python -c {shlex.quote(_UNPACK)} && {inner}"]


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


def flavor_rate(flavor: str, cfg: Config) -> float | None:
    """The hourly rate for a flavor, or None when the table does not price it.

    None rather than 0.0, and the difference is the whole point: HF serves
    flavors this table has never heard of (`l4x4`, `h100`, whatever ships next),
    and pricing an unknown one at zero books a real job as free -- permanently
    understating rolling spend and the project ceiling, which is exactly the
    "stale in the optimistic direction makes the ceiling decoration" failure the
    config comment warns about. Callers must decide what to do with the None;
    none of them may treat it as free.
    """
    rates = cfg.get("hf", "flavor_rates", {}) or {}
    if flavor not in rates:
        return None
    try:
        rate = float(rates[flavor])
    except (TypeError, ValueError):
        return None
    return rate if rate >= 0 else None


def _actual_cost(info: Any, flavor: str, cfg: Config, *, estimate_usd: float = 0.0) -> tuple[float, str | None]:
    """Cost from the platform's own accounting of the run.

    HF reports the job's start and end timestamps; the price of a flavor comes
    from the rate table in config/grad.toml. The estimate is never reused here
    -- that is the whole point of collecting -- *except* when the flavor is
    unpriced or the platform reported no start time, where the alternative is
    booking the run at $0. Falling back to the estimate keeps the ceiling
    honest, and the returned warning is what says the number is not measured.
    """
    started = _ts(info, "started_at") or _ts(info, "created_at")
    ended = _ts(info, "ended_at") or _dt.datetime.now(_dt.timezone.utc)
    rate = flavor_rate(flavor, cfg)
    if rate is None:
        return round(float(estimate_usd), 4), (
            f"flavor {flavor!r} is not priced in [hf.flavor_rates]; this run is booked at its "
            f"estimate of ${float(estimate_usd):.2f} rather than at $0. Add the rate to "
            "config/grad.toml and re-collect for a measured figure."
        )
    if not started:
        return round(float(estimate_usd), 4), (
            "the platform reported no start time for this run, so its duration is unknown; "
            f"booked at its estimate of ${float(estimate_usd):.2f} rather than at $0."
        )
    hours = max(0.0, (ended - started).total_seconds() / 3600.0)
    return round(hours * rate, 4), None


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
    # Every value the run reported per quantity, not just the last. A run that
    # reports one quantity several times is a replicated run -- see core/stats.py.
    samples: dict[str, list[Any]] = {}
    metrics_error = None
    metrics_path = artifacts / Path(r.get("metrics_file") or "metrics.json").name
    try:
        _download_artifacts(r, artifacts)
        results, samples = submit_lib.read_metrics(metrics_path)
    except GradError as exc:
        metrics_error = exc.message

    expectation = None
    if r.get("expectation_id"):
        try:
            expectation = ls.expectation(r["expectation_id"])
        except GradError:
            expectation = None

    cost, cost_warning = _actual_cost(
        info,
        (r.get("target") or {}).get("flavor", ""),
        config_mod.load(),
        estimate_usd=float(r.get("estimate_usd") or 0.0),
    )
    record = submit_lib.finish(
        r.id,
        status="completed" if state == "COMPLETED" else "failed",
        results=results,
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=expectation,
        samples=samples,
        extra={
            "job_state": state,
            "metrics_error": metrics_error,
            # On the record, not just in this reply: whether a cost was measured
            # or fallen back to is a property of the run, and `report` and the
            # ceiling both read the record rather than this envelope.
            "cost_warning": cost_warning,
            "cost_basis": "estimate" if cost_warning else "measured",
        },
    )
    unjudged = [d for d in record["deviations"] if d.get("in_range") is not True]
    return {
        "run": record,
        "artifacts": str(artifacts),
        "cost_warning": cost_warning,
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
CREDENTIAL_NAMES = credentials.ALL


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
