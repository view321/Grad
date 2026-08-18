"""grad-modal -- submit, watch, and collect Modal Sandboxes (HANDOFF §6, §7).

The fourth backend, and the first one whose billing model the `[spend]` ceilings
were already the right instrument for: Modal charges per second against a
published rate table, so none of Kaggle's hours machinery is needed here. A
dollar ceiling measures something real again.

Four decisions are load-bearing.

**A Sandbox, not a Function.** Modal's headline API is a decorated function in a
deployed app, which would mean the research pipeline had to be written for
Modal. A Sandbox takes a container image, a command and a GPU -- which is
exactly the shape `core/submission.py` already resolves -- so a spec that runs
on HF Jobs runs here with a different `[target] platform` and nothing else.
`Sandbox.from_id` is the other half: `submit` and `collect` are separate CLI
invocations in separate processes, and the sandbox id in the ledger handle is
what reconnects them.

**The credential never leaves this process.** Every other backend here hands a
secret to a child -- an environment variable, a config file written for the
duration of a call. Modal's client takes the token pair as *arguments* and sends
them as gRPC headers, so `modal.Client.from_credentials` is the strongest form
of §9 available anywhere in this project: there is nothing to scrub because
nothing is exported. `MODAL_TOKEN_ID` is deliberately never set.

**Results come back through a Volume, because a sandbox's filesystem does not
outlive it.** A finished sandbox cannot be `exec`'d into, so `cat metrics.json`
works only while the job is still running -- which is precisely when there is
nothing to read. The run writes into a mounted Volume instead, under a directory
named for the run, and `collect` reads the Volume afterwards from a process the
sandbox knows nothing about.

**The image must still be digest-pinned.** `Image.from_registry` takes a
registry reference and `core/submission.py` already requires a digest, so the
preflight hash keeps meaning what it says: the same hash is the same image.
Modal's own image-building DSL (`pip_install`, `run_commands`) is deliberately
not exposed -- an image assembled from a Python expression has no digest to hash
until it is built, and a preflight record keyed by a hash that does not cover
the environment is a preflight record that certifies nothing.
"""

from __future__ import annotations

import argparse
import shlex
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
    "grad-modal",
    "Submit and collect Modal Sandboxes. Refuses to submit without a passing "
    "preflight, an open expectation, and headroom under both spend ceilings.",
    epilog=(
        "gate refusals have their own exit codes (4 preflight, 5 expectation, 6 spend,\n"
        "7 stale run) so a refusal is never confused with an upstream failure.\n\n"
        "Modal bills per second, so the [spend] ceilings are the gate here and there is\n"
        "no hours allowance to exhaust -- exit 13 never comes from this backend.\n\n"
        "A Sandbox's maximum lifetime is 24 hours and Modal enforces it: a spec whose\n"
        "[estimate] hours exceeds that is refused rather than started."
    ),
)

PLATFORM = "modal"

#: Modal's own hard ceiling on a Sandbox's lifetime. Not a policy of ours: the
#: container is killed at this point whatever it was doing, so a run planned past
#: it is a run that cannot finish.
MODAL_MAX_HOURS = 24.0

#: Where the run's outputs go inside the sandbox, as an environment variable the
#: pipeline can read. Named rather than positional so a pipeline that wants to
#: write a checkpoint next to its metrics has somewhere documented to put it.
OUT_ENV = "GRAD_OUT_DIR"


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
def _modal() -> Any:
    """The Modal SDK, or a configuration error naming the extra to install."""
    try:
        import modal  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "the modal SDK is not installed, so Modal cannot be reached",
            fix="pip install -e '.[modal]'   # or: python -m pip install modal",
        ) from exc
    # The exact surface this module uses, not just the top-level names. Checked
    # the way `tools/jobs.py:_hub` checks for `run_job`/`inspect_job`, and for
    # the same reason: a version floor in `pyproject.toml` is a hint to the
    # resolver, and this is the thing that actually decides whether a submission
    # can work. Without it an older SDK fails at `Client.from_credentials` with
    # an AttributeError several frames into a submit that has already written a
    # ledger record.
    needed = {
        "Client": ("from_credentials",),
        "Sandbox": ("create", "from_id"),
        "Image": ("from_registry",),
        "Volume": ("from_name",),
        "App": ("lookup",),
    }
    for attr, methods in needed.items():
        target = getattr(modal, attr, None)
        if target is None:
            raise ConfigError(
                f"the installed modal has no {attr}; this is not a version this understands",
                fix="python -m pip install -U modal",
            )
        missing = [m for m in methods if not hasattr(target, m)]
        if missing:
            raise ConfigError(
                f"the installed modal's {attr} has no {', '.join(missing)}, "
                "so this backend cannot reach it",
                fix="python -m pip install -U modal",
            )
    return modal


def _client() -> Any:
    """An authenticated client, built from credentials fetched at the moment of use.

    `from_credentials` rather than the environment, and that is the whole
    argument of §9 in one call: the token pair is passed as arguments and sent as
    gRPC headers, so it is never in `os.environ` for the agent to read and never
    in a child's environment for a subprocess to inherit. Nothing here needs
    scrubbing because nothing is exported.
    """
    modal = _modal()
    token_id = credentials.get(credentials.MODAL_TOKEN_ID, required=False)
    token_secret = credentials.get(credentials.MODAL_TOKEN_SECRET, required=False)
    missing = [
        name
        for name, value in (
            (credentials.MODAL_TOKEN_ID, token_id),
            (credentials.MODAL_TOKEN_SECRET, token_secret),
        )
        if not value
    ]
    if missing:
        raise ConfigError(
            f"no Modal credential stored: {', '.join(missing)}",
            fix=" && ".join(f"python -m tools.jobs credential set {n}" for n in missing),
        )
    try:
        return modal.Client.from_credentials(token_id, token_secret)
    except Exception as exc:  # noqa: BLE001 - the SDK raises a wide family here
        raise ConfigError(
            f"Modal refused the stored credentials: {exc}",
            fix=(
                "mint a fresh token pair at modal.com/settings/tokens, then "
                f"python -m tools.jobs credential set {credentials.MODAL_TOKEN_ID}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# hardware and money
# ---------------------------------------------------------------------------
def resolve_gpu(flag: str | None, sub: Submission, cfg: Config) -> str:
    """Which accelerator this run asks for: the flag, the spec, then the config.

    The same resolution order every backend here uses, and the same reason: a
    flag is a decision about one submission, a spec is a decision about the
    pipeline, and the config is a decision about the machine.
    """
    chosen = flag or sub.target.get("gpu") or cfg.get("modal", "default_gpu", "H100")
    return str(chosen).strip()


def gpu_rate(gpu: str, cfg: Config) -> float | None:
    """Dollars per hour for one accelerator, or None if it is not priced.

    None is not zero and the callers treat it as a refusal rather than a
    bargain: an unpriced GPU is one whose spend the ceiling cannot bound, and
    `[spend]` is the only gate this backend has.

    A count suffix is stripped before the lookup -- Modal spells eight H100s
    `H100:8` -- and multiplies the rate, because eight cards cost eight times as
    much and a table with an entry per count would be a table that goes stale one
    row at a time.
    """
    rates = cfg.get("modal", "gpu_rates", {}) or {}
    name, _, count = str(gpu).partition(":")
    rate = (rates or {}).get(name.strip())
    if rate is None:
        return None
    if not count.strip():
        return float(rate)
    try:
        multiplier = max(1, int(count))
    except ValueError:
        # Not one card. `H100:eight` is a spec nobody can price, and pricing it
        # as a single card would book an eight-GPU run at an eighth of its cost
        # -- silently, and in the direction the ceiling cannot catch.
        return None
    return float(rate) * multiplier


def _rate_or_refuse(gpu: str, cfg: Config) -> float:
    rate = gpu_rate(gpu, cfg)
    if rate is None:
        known = ", ".join(sorted(cfg.get("modal", "gpu_rates", {}) or {})) or "(none)"
        raise ConfigError(
            f"no price is configured for Modal GPU {gpu!r}, so this run's cost cannot be bounded",
            fix=f"add it under [modal.gpu_rates] in config/grad.toml. Priced now: {known}",
        )
    return rate


def _timeout_seconds(sub: Submission, cfg: Config) -> int:
    """How long the sandbox may live, from the spec's own estimate.

    Modal kills a Sandbox at its timeout, so this is the number that decides
    whether a run finishes. It is the estimate plus a margin rather than the
    estimate, because an estimate that was exactly right is the one case a job
    would be killed for being on time.

    A spec asking for more than Modal's 24-hour ceiling is refused here rather
    than started: the alternative is a run that trains for a day and is killed
    with nothing collected.
    """
    hours = float(sub.estimate.get("hours") or 0.0)
    if hours <= 0:
        raise ConfigError(
            f"{sub.spec_path} has no `[estimate] hours`, so the sandbox has no timeout to set",
            fix="add `hours = <your estimate>` under [estimate] in the spec",
        )
    margin = float(cfg.get("modal", "timeout_margin", 1.25) or 1.25)
    ceiling = min(float(cfg.get("modal", "max_hours", MODAL_MAX_HOURS)), MODAL_MAX_HOURS)
    wanted = hours * margin
    if wanted > ceiling:
        raise ConfigError(
            f"this spec asks for {hours:.2f}h which needs a {wanted:.2f}h sandbox, and Modal's "
            f"ceiling is {ceiling:.2f}h -- the container would be killed mid-run",
            fix="reduce `[estimate] hours`, checkpoint and resume across runs, or use a bigger GPU",
        )
    return int(wanted * 3600)


# ---------------------------------------------------------------------------
# the sandbox
# ---------------------------------------------------------------------------
def _command_for(sub: Submission) -> list[str]:
    if sub.target.get("command"):
        return [str(c) for c in sub.target["command"]]
    return ["python", sub.entrypoint.name, *sub.argv]


def _job_env(sub: Submission, out_dir: str) -> dict[str, str]:
    env = {str(k): str(v) for k, v in (sub.target.get("env") or {}).items()}
    # Into the Volume, not the container filesystem: a sandbox's disk is gone the
    # moment it exits, and `collect` runs afterwards by construction.
    env["GRAD_METRICS_FILE"] = f"{out_dir}/{Path(sub.metrics_file).name}"
    env[OUT_ENV] = out_dir
    return env


def _wrapped_command(command: list[str], sub: Submission, out_dir: str) -> list[str]:
    """The command, plus the copy that makes a well-behaved pipeline unnecessary.

    `GRAD_METRICS_FILE` tells the pipeline where to write, and every pipeline in
    this project respects it. A pipeline that does not -- one written for another
    harness, one that hardcodes `metrics.json` beside itself -- would produce a
    run that succeeded and collected nothing, which is the most expensive way to
    fail: the money is spent and the result is gone.

    So the metrics file is copied into the Volume afterwards if it is sitting in
    the working directory, and **the exit code is preserved across the copy**.
    `cp` failing must not turn a failed run into a successful one or the reverse,
    which is what `rc=$?` and the final `exit $rc` are for.
    """
    name = Path(sub.metrics_file).name
    inner = shlex.join(command)
    out = shlex.quote(out_dir)
    metrics = shlex.quote(name)
    script = (
        f"mkdir -p {out}; {inner}; rc=$?; "
        f"if [ -f {metrics} ]; then cp -f {metrics} {out}/ 2>/dev/null || true; fi; "
        "exit $rc"
    )
    return ["sh", "-c", script]


def _image(sub: Submission, modal: Any, cfg: Config, env: dict[str, str]) -> Any:
    """The digest-pinned registry image, with the pipeline copied in.

    `copy=True` rather than the default mount: a mounted directory is attached at
    startup and is a property of the *client* that started the sandbox, and this
    client exits as soon as `submit` returns. Baking the code into a layer is
    what makes the run survive its submitter.
    """
    workdir = str(cfg.get("modal", "workdir", "/grad/pipeline"))
    image = modal.Image.from_registry(sub.image)
    image = image.add_local_dir(str(sub.spec_path.parent), workdir, copy=True)
    if env:
        image = image.env(env)
    return image


def _volume(modal: Any, cfg: Config, client: Any) -> tuple[Any, str]:
    name = str(cfg.get("modal", "volume_name", "grad-runs"))
    volume = modal.Volume.from_name(name, create_if_missing=True, client=client)
    return volume, name


def _run_dir(cfg: Config, run_id: str) -> str:
    mount = str(cfg.get("modal", "mount_path", "/grad/out")).rstrip("/")
    return f"{mount}/{run_id}"


def _create_sandbox(
    sub: Submission,
    cfg: Config,
    *,
    client: Any,
    gpu: str,
    command: list[str],
    timeout_s: int,
    run_id: str,
) -> tuple[Any, str]:
    """Start the sandbox and return it with the id the ledger will hold."""
    modal = _modal()
    volume, volume_name = _volume(modal, cfg, client)
    mount = str(cfg.get("modal", "mount_path", "/grad/out")).rstrip("/")
    out_dir = _run_dir(cfg, run_id)
    env = _job_env(sub, out_dir)
    app = modal.App.lookup(
        str(cfg.get("modal", "app_name", "grad")), create_if_missing=True, client=client
    )
    sandbox = modal.Sandbox.create(
        *_wrapped_command(command, sub, out_dir),
        app=app,
        image=_image(sub, modal, cfg, env),
        gpu=gpu,
        timeout=timeout_s,
        workdir=str(cfg.get("modal", "workdir", "/grad/pipeline")),
        volumes={mount: volume},
        client=client,
    )
    sandbox_id = getattr(sandbox, "object_id", None)
    if not sandbox_id or not isinstance(sandbox_id, str):
        # Never `str(sandbox)`. The fallback looks harmless and writes
        # `<modal.Sandbox object at 0x...>` into the ledger handle, which
        # `collect` then hands to `Sandbox.from_id` -- so the run cannot be
        # collected, goes stale, and blocks every later submission through the
        # §6 gate. Failing here instead lets the caller mark the submission
        # failed while the sandbox is still the only thing that exists.
        raise UpstreamError(
            "Modal returned a sandbox with no id, so this run could never be collected",
            fix="retry; if it persists, check the Modal dashboard and `python -m pip install -U modal`",
        )
    # Detached, so the sandbox outlives this CLI. `submit` returns in seconds and
    # the run takes hours; without this the client-side connection is the thing
    # holding it, and `collect` in a later process could not reach it.
    _detach(sandbox)
    return sandbox_id, volume_name


def _detach(sandbox: Any) -> None:
    """Release the client-side connection without stopping the sandbox.

    Best effort: an SDK without `detach` leaves a connection that dies with this
    process anyway, which is the outcome `detach` asks for politely.
    """
    try:
        sandbox.detach()
    except Exception:  # noqa: BLE001 - see the docstring
        pass


def _reattach(sandbox_id: str, client: Any) -> Any:
    modal = _modal()
    try:
        return modal.Sandbox.from_id(sandbox_id, client=client)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            f"Modal could not find sandbox {sandbox_id}: {exc}",
            fix="check the run in the Modal dashboard; if it is gone, `ledger abandon` the run",
        ) from exc


def _state_of(sandbox: Any) -> tuple[str, int | None]:
    """`(state, exit_code)`, where state is one of running/completed/failed.

    `poll()` returns None while the sandbox is alive and the exit code once it is
    not, which is the whole state machine -- there is no queued state to wait
    through, because a Sandbox either starts or fails to.
    """
    try:
        code = sandbox.poll()
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            f"Modal would not report the sandbox's state: {exc}",
            fix="retry, or check the run in the Modal dashboard",
        ) from exc
    if code is None:
        return "running", None
    return ("completed" if int(code) == 0 else "failed"), int(code)


def _poll(sandbox: Any, *, deadline: float, interval: float) -> tuple[str, int | None]:
    while True:
        state, code = _state_of(sandbox)
        if state != "running" or time.time() >= deadline:
            return state, code
        time.sleep(max(1.0, interval))


def _logs(sandbox: Any) -> str:
    """Whatever the sandbox said, best effort and never fatal.

    A collected run whose logs could not be fetched is still a collected run --
    the metrics are in the Volume, which is the part the ledger needs.
    """
    parts = []
    for stream in ("stdout", "stderr"):
        try:
            handle = getattr(sandbox, stream, None)
            text = handle.read() if handle is not None else ""
            if text:
                parts.append(f"----- {stream} -----\n{text}")
        except Exception as exc:  # noqa: BLE001 - see the docstring
            parts.append(f"----- {stream} unavailable: {type(exc).__name__}: {exc} -----")
    return "\n".join(parts)


def _download_outputs(cfg: Config, run_id: str, dest: Path, client: Any) -> list[str]:
    """Copy this run's directory out of the Volume. Returns what arrived.

    Reads the Volume rather than the sandbox, for the reason in the module
    docstring: by the time `collect` runs, the sandbox that wrote these files no
    longer exists to be read from.
    """
    modal = _modal()
    volume, _ = _volume(modal, cfg, client)
    prefix = _run_dir(cfg, run_id).lstrip("/")
    mount = str(cfg.get("modal", "mount_path", "/grad/out")).strip("/")
    # Paths inside a Volume are relative to its mount point, so the mount prefix
    # comes back off before anything is asked for.
    inside = prefix[len(mount):].strip("/") if prefix.startswith(mount) else prefix
    dest.mkdir(parents=True, exist_ok=True)
    arrived: list[str] = []
    try:
        entries = list(volume.iterdir(f"/{inside}", recursive=True))
    except Exception:  # noqa: BLE001 - an empty or missing directory is a real outcome
        return arrived
    for entry in entries:
        remote = getattr(entry, "path", None)
        if not remote:
            continue
        relative = str(remote).split(inside, 1)[-1].strip("/")
        if not relative:
            continue
        # The same guard `evaluate_candidate` puts on candidate filenames, for
        # the same reason: these names come back from the Volume, which is
        # written by the job, and `dest / "../../x"` resolves outside the run's
        # artifacts directory. A traversal here writes to the researcher's disk.
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(target, "wb") as fh:
                for chunk in volume.read_file(str(remote)):
                    fh.write(chunk)
        except Exception:  # noqa: BLE001 - a directory entry, or a file that vanished
            continue
        arrived.append(relative)
    return arrived


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
    p.add_argument("--gpu", help="Modal GPU, e.g. H100 or H100:8 (overrides the spec)")
    p.add_argument("--task", help="task id for the ledger (defaults to the spec directory name)")
    p.add_argument("--project", help="project to charge this run to (defaults to the current one)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="the gate-exempt, hard-capped one-step check from §6. Cannot train anything.",
    )
    p.add_argument("--no-digest", action="store_true", help=argparse.SUPPRESS)


@cli.command("submit", "submit a sandbox (gated) or a smoke check (capped)", setup=_submit_args)
def cmd_submit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    sub = Submission.load(
        args.spec,
        overrides=dict(parse_override(o) for o in args.overrides),
        resolve_digest=not args.no_digest,
    )
    if args.gpu:
        sub.target["gpu"] = args.gpu

    project_id = budget.resolve(args.project)
    gpu = resolve_gpu(args.gpu, sub, cfg)

    if args.smoke:
        if args.expect:
            raise UsageError(
                "--smoke does not take --expect: a smoke check is not a result and binds no prediction",
                fix="drop --expect, or drop --smoke",
            )
        result = run_smoke(sub, cfg, gpu=gpu, project=project_id)
        from tools import preflight  # noqa: PLC0415

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

    command = _command_for(sub)
    # Resolved before the gates so a misconfigured GPU or an impossible timeout
    # is a configuration error rather than a gate refusal -- they have different
    # exit codes because they need different actions.
    rate = _rate_or_refuse(gpu, cfg)
    timeout_s = _timeout_seconds(sub, cfg)

    summary = submit_lib.check(sub, args.expect, cfg, project=project_id)
    # Then the backend and the credential, before any record exists: a missing
    # package or an absent token is a configuration problem, not an in-flight
    # run, and it must not leave a phantom estimate sitting on the ceiling.
    client = _client()

    run_id, _ = submit_lib.record_submission(
        sub,
        expectation_id=args.expect,
        platform=PLATFORM,
        target={"gpu": gpu, "platform": PLATFORM, "rate_usd_per_hour": rate,
                "timeout_s": timeout_s},
        command=command,
        task=args.task,
        project=project_id,
        cfg=cfg,
    )

    try:
        sandbox_id, volume_name = _create_sandbox(
            sub, cfg, client=client, gpu=gpu, command=command,
            timeout_s=timeout_s, run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 - the SDK raises a wide family here
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
            f"Modal refused the submission: {exc}",
            fix="check the token, the GPU name, and that the image digest is pullable",
        ) from exc

    submit_lib.attach_handle(
        run_id,
        {"sandbox_id": sandbox_id, "gpu": gpu, "volume": volume_name,
         "run_dir": _run_dir(cfg, run_id), "timeout_s": timeout_s},
    )
    return {
        "run_id": run_id,
        "sandbox_id": sandbox_id,
        "gpu": gpu,
        "rate_usd_per_hour": rate,
        "timeout_hours": round(timeout_s / 3600, 2),
        "project": project_id,
        "gates": summary,
        "next": f"python -m tools.modal collect {run_id} --json",
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def run_smoke(
    sub: Submission, cfg: Config, *, gpu: str | None = None, project: str | None = None
) -> dict[str, Any]:
    """The §6 carve-out: gate-exempt, hard-capped, still ledgered.

    Same shape as every other backend's, and the caps come from the same place
    (`gates.check_smoke_caps`) so "one step, minutes of wall clock, cents of
    money" means the same thing here as on HF Jobs. Blocks, because it is bounded
    to minutes by construction and preflight needs the answer.
    """
    gpu = gpu or sub.target.get("smoke_gpu") or resolve_gpu(None, sub, cfg)
    rate = _rate_or_refuse(gpu, cfg)
    caps = gates.check_smoke_caps(sub, cfg, rate_usd_per_hour=rate, target_name=f"gpu {gpu!r}")

    project = project or budget.current_project()
    # Everything that can fail for a configuration reason resolves before the
    # ledger record exists, or a phantom in-flight estimate sits on the monthly
    # ceiling for a run that never reached the platform -- and then goes stale
    # and blocks every later submission.
    client = _client()

    run_id = submit_lib.record_smoke_run(
        sub, cfg=cfg, platform=PLATFORM,
        target={"gpu": gpu, "platform": PLATFORM, "rate_usd_per_hour": rate},
        caps=caps, command=_command_for(sub), project=project,
    )
    artifacts = submit_lib.artifacts_dir(run_id)

    command = _smoke_command(sub, caps)
    try:
        sandbox_id, _ = _create_sandbox(
            sub, cfg, client=client, gpu=gpu, command=command,
            timeout_s=int(caps["timeout_s"]), run_id=run_id,
        )
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        submit_lib.finish(
            run_id, status="submit_failed", results={}, cost_usd_actual=0.0,
            artifacts_dir=artifacts, expectation=None, extra={"error": str(exc)},
        )
        return {"ok": False, "reason": f"smoke submission failed: {exc}",
                "fix": "check the Modal token and that the image digest is pullable",
                "run_id": run_id}

    submit_lib.attach_handle(run_id, {"sandbox_id": sandbox_id, "gpu": gpu})
    sandbox = _reattach(sandbox_id, client)
    state, code = _poll(
        sandbox,
        deadline=time.time() + float(caps["timeout_s"]),
        interval=float(cfg.get("modal", "poll_interval_s", 20) or 20),
    )
    logs = _logs(sandbox)
    (artifacts / "smoke.log").write_text(logs, encoding="utf-8")

    ok = state == "completed"
    elapsed = submit_lib.elapsed_hours(ls.run(run_id))
    cost = round(min(elapsed, float(caps["timeout_s"]) / 3600) * rate, 4)
    submit_lib.finish(
        run_id,
        status="completed" if ok else "failed",
        results={},
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=None,
        extra={"sandbox_state": state, "exit_code": code, "smoke": True,
               "cost_basis": "wall_clock"},
    )
    return {
        "ok": ok,
        "run_id": run_id,
        "sandbox_id": sandbox_id,
        "state": state,
        "exit_code": code,
        "gpu": gpu,
        "cost_usd": cost,
        "caps": caps,
        "log": str(artifacts / "smoke.log"),
        "output": "\n".join(logs.splitlines()[-25:]),
        "reason": None if ok else f"the smoke sandbox ended {state} (exit {code})",
        "fix": None if ok else f"read {artifacts / 'smoke.log'} -- this is the environment the real run would have used",
        "scope": "remote; the only check that exercises the real image, data path, and hardware",
    }


def _smoke_command(sub: Submission, caps: dict[str, Any]) -> list[str]:
    """One step, real per-device batch size -- the same shape `tools/jobs.py`
    builds, because §6's argument is about the check and not about the backend:
    smoking at batch 2 does not test the thing that most often kills the run."""
    return [*_command_for(sub), "--steps", str(caps["steps"]), "--smoke"]


# ---------------------------------------------------------------------------
# evolve candidates
# ---------------------------------------------------------------------------
#: How many bytes of a candidate's output are kept, matching the other backends.
CANDIDATE_OUTPUT_BYTES = 8000


def evaluate_candidate(
    sub: Submission,
    cfg: Config,
    *,
    candidate_id: str,
    files: dict[str, str],
    command: list[str],
    timeout_s: int,
    artifacts: Path,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Run one evolve candidate in a Sandbox, and read its metrics back.

    **The delivery path is the simplest of the four backends, and that is
    Modal's doing rather than ours.** `gpu.py` copies the pipeline to a host,
    `kaggle.py` packs it into the notebook, and `jobs.py` smuggles the candidate
    through a base64 environment variable because on HF Jobs the pipeline is in
    the image and a candidate by definition is not. Modal builds the image
    client-side, so a candidate is a second `add_local_dir` layered over the
    first -- no encoding, no size ceiling, no prelude to unpack it.

    Like the other adapters: no ledger row, cost measured rather than estimated,
    and a transport failure reported as one rather than as a candidate that
    scored nothing. A campaign is a search, and a candidate that could not be
    *delivered* says nothing about the mutation that produced it.
    """
    import tempfile  # noqa: PLC0415 - only the candidate path stages files

    gpu = gpu or resolve_gpu(None, sub, cfg)
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rate = gpu_rate(gpu, cfg) or 0.0

    def _failed(message: str, **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "exit_code": None,
            "output": "",
            "error": message,
            "cost_usd": round((time.time() - started) / 3600.0 * rate, 4),
            "gpu": gpu,
            "where": f"modal:{candidate_id}",
            **extra,
        }

    try:
        client = _client()
        modal = _modal()
    except GradError as exc:
        return _failed(exc.message)

    with tempfile.TemporaryDirectory(prefix="grad-candidate-") as staging:
        root = Path(staging)
        for name, body in files.items():
            # Written under the staging root and never outside it: a candidate's
            # filenames come from a model, and a `..` in one would otherwise be a
            # path this process writes to on the researcher's machine.
            target = (root / name).resolve()
            if root.resolve() not in target.parents:
                return _failed(f"candidate file {name!r} escapes the staging directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")

        workdir = str(cfg.get("modal", "workdir", "/grad/pipeline"))
        out_dir = f"{str(cfg.get('modal', 'mount_path', '/grad/out')).rstrip('/')}/{candidate_id}"
        try:
            volume, _ = _volume(modal, cfg, client)
            app = modal.App.lookup(
                str(cfg.get("modal", "app_name", "grad")), create_if_missing=True, client=client
            )
            image = _image(sub, modal, cfg, {**_job_env(sub, out_dir), "GRAD_CANDIDATE": candidate_id})
            # Layered *after* the pipeline, so a candidate replaces the file it
            # is a mutation of rather than sitting beside it.
            image = image.add_local_dir(str(root), workdir, copy=True)
            sandbox = modal.Sandbox.create(
                *_wrapped_command(command, sub, out_dir),
                app=app,
                image=image,
                gpu=gpu,
                timeout=int(timeout_s),
                workdir=workdir,
                volumes={str(cfg.get("modal", "mount_path", "/grad/out")).rstrip("/"): volume},
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 - a refused sandbox is not a bad mutation
            return _failed(f"the candidate could not be submitted: {exc}")

    state, code = _poll(
        sandbox,
        deadline=time.time() + int(timeout_s),
        interval=float(cfg.get("modal", "poll_interval_s", 20) or 20),
    )
    logs = _logs(sandbox)
    (artifacts / "candidate.log").write_text(logs, encoding="utf-8")
    _detach(sandbox)

    ok = state == "completed"
    return {
        "ok": ok,
        # A real exit code, unlike HF Jobs, which reports only a state: Modal's
        # `poll()` returns the process's own status, so this is measured.
        "exit_code": code,
        "output": logs[-CANDIDATE_OUTPUT_BYTES:],
        "error": None if ok else f"the candidate's sandbox ended {state} (exit {code})",
        "cost_usd": round((time.time() - started) / 3600.0 * rate, 4),
        "gpu": gpu,
        "sandbox_state": state,
        "where": f"modal:{getattr(sandbox, 'object_id', candidate_id)}",
    }


# ---------------------------------------------------------------------------
# status and collect
# ---------------------------------------------------------------------------
def _status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id")


@cli.command("status", "what a submitted run is doing right now", setup=_status_args)
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    """Poll one run without collecting it."""
    r = ls.run(args.run_id)
    handle = r.get("handle") or {}
    sandbox_id = handle.get("sandbox_id")
    if not sandbox_id:
        return {"run_id": r.id, "state": "no_handle", "collected": bool(r.collected)}
    sandbox = _reattach(sandbox_id, _client())
    state, code = _state_of(sandbox)
    return {
        "run_id": r.id,
        "sandbox_id": sandbox_id,
        "state": state,
        "exit_code": code,
        "gpu": handle.get("gpu"),
        "elapsed_hours": round(submit_lib.elapsed_hours(r), 3),
        "collected": bool(r.collected),
    }


def _collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id")
    p.add_argument("--wait", action="store_true", help="block until the sandbox finishes")
    p.add_argument("--timeout", type=int, default=900, help="seconds, with --wait")


@cli.command("collect", "fetch outputs, compute deviations, write the run record", setup=_collect_args)
def cmd_collect(args: argparse.Namespace) -> dict[str, Any]:
    """Closes the loop the model would otherwise close from memory.

    Writes results, actual cost and the deviations array, and leaves `verdict`
    unset -- that is `ledger.py verdict`'s job, and judgement must not be able to
    overwrite the record it judges.
    """
    r = submit_lib.require_uncollected(args.run_id)
    handle = r.get("handle") or {}
    sandbox_id = handle.get("sandbox_id")
    if not sandbox_id:
        raise GradError(
            "no_handle",
            f"run {r.id} has no Modal sandbox id; it never reached the platform",
            exit_code=3,
            fix=f"python -m tools.ledger abandon {r.id} --reason '...' --json",
        )

    cfg = config_mod.load()
    client = _client()
    sandbox = _reattach(sandbox_id, client)
    deadline = time.time() + (args.timeout if args.wait else 0)
    state, code = _poll(
        sandbox, deadline=deadline,
        interval=float(cfg.get("modal", "poll_interval_s", 20) or 20),
    )
    if state == "running":
        raise GradError(
            "still_running",
            f"sandbox {sandbox_id} is still running",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.modal collect {r.id} --wait --timeout 3600 --json",
            detail={"run_id": r.id, "state": state},
        )

    artifacts = submit_lib.artifacts_dir(r.id)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "sandbox.log").write_text(_logs(sandbox), encoding="utf-8")

    results: dict[str, Any] = {}
    samples: dict[str, list[Any]] = {}
    metrics_error = None
    downloaded = _download_outputs(cfg, r.id, artifacts, client)
    metrics_path = artifacts / Path(r.get("metrics_file") or "metrics.json").name
    try:
        results, samples = submit_lib.read_metrics(metrics_path)
    except GradError as exc:
        metrics_error = exc.message

    expectation = None
    if r.get("expectation_id"):
        try:
            expectation = ls.expectation(r["expectation_id"])
        except GradError:
            expectation = None

    cost, cost_warning = _actual_cost(r, handle, cfg)
    record = submit_lib.finish(
        r.id,
        status="completed" if state == "completed" else "failed",
        results=results,
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=expectation,
        samples=samples,
        extra={
            "sandbox_state": state,
            "exit_code": code,
            "metrics_error": metrics_error,
            "downloaded": downloaded,
            "cost_warning": cost_warning,
            "cost_basis": "wall_clock",
        },
    )
    unjudged = [d for d in record["deviations"] if d.get("in_range") is not True]
    return {
        "run": record,
        "artifacts": str(artifacts),
        "downloaded": downloaded,
        "cost_warning": cost_warning,
        "needs_verdict": unjudged,
        "next": (
            f"python -m tools.ledger verdict {r.id} --quantity {unjudged[0]['quantity']} "
            "--verdict bug|real|inconclusive --note '...' --json"
        ) if unjudged else None,
    }


def _actual_cost(r: ls.Run, handle: dict[str, Any], cfg: Config) -> tuple[float, str | None]:
    """What this run cost, and an honest note about how that was arrived at.

    **Wall clock from our own record, not Modal's accounting**, and the warning
    says so on every run rather than only when something went wrong. Modal bills
    per second from container start to exit; what is measurable here is the
    interval between the ledger's `submitted_at` and now, which includes the
    image pull and however long the run sat between finishing and being
    collected. It is an upper bound, and the direction is the safe one for a
    ceiling -- but it is not a measurement and the record must not imply it is.

    Bounded by the sandbox's own timeout, because the container cannot have run
    longer than Modal would let it: collecting a week later must not book a
    week of H100 time against the project.
    """
    gpu = str(handle.get("gpu") or (r.get("target") or {}).get("gpu") or "")
    rate = gpu_rate(gpu, cfg)
    if rate is None:
        return 0.0, (
            f"no rate configured for {gpu!r}, so this run is booked at $0 and its cost is "
            "not bounded by the ceiling -- price it under [modal.gpu_rates] before the next one"
        )
    elapsed = submit_lib.elapsed_hours(r)
    timeout_h = float(handle.get("timeout_s") or 0) / 3600 or None
    if timeout_h:
        elapsed = min(elapsed, timeout_h)
    return round(elapsed * rate, 4), (
        "cost is wall clock from submission priced against [modal.gpu_rates], not Modal's own "
        "billing. It is long in one direction (it includes the image pull and any delay before "
        "collection) and short in another (GPU time only; the CPU and memory the sandbox held "
        "are not counted). Collect promptly if the number matters."
    )


# ---------------------------------------------------------------------------
# account and ceilings
# ---------------------------------------------------------------------------
def _account_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--check", action="store_true", help="verify the stored token pair authenticates")


@cli.command("account", "which Modal workspace the stored credentials reach", setup=_account_args)
def cmd_account(args: argparse.Namespace) -> dict[str, Any]:
    """Answer "is this wired up" without submitting anything.

    `--check` is the useful half: `credential status` says a secret is *stored*,
    which is a different claim from a secret that *works*, and the gap between
    them is only ever discovered at the worst moment otherwise.
    """
    stored = {
        name: bool(credentials.get(name, required=False))
        for name in (credentials.MODAL_TOKEN_ID, credentials.MODAL_TOKEN_SECRET)
    }
    out: dict[str, Any] = {"stored": stored, "authenticated": None}
    if not args.check:
        out["next"] = "python -m tools.modal account --check --json"
        return out
    try:
        client = _client()
        modal = _modal()
        modal.App.lookup(
            str(config_mod.load().get("modal", "app_name", "grad")),
            create_if_missing=True,
            client=client,
        )
        out["authenticated"] = True
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        out["authenticated"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


@cli.command("gpus", "which accelerators are priced, and at what")
def cmd_gpus(_: argparse.Namespace) -> dict[str, Any]:
    """The rate table, which is also the list of what may be asked for.

    An unpriced GPU is refused at submit rather than assumed free, so this is not
    decoration: it is the set of hardware this installation can bound the cost of.
    """
    cfg = config_mod.load()
    rates = cfg.get("modal", "gpu_rates", {}) or {}
    return {
        "gpus": dict(sorted(rates.items())),
        "default": cfg.get("modal", "default_gpu", "H100"),
        "max_hours": min(float(cfg.get("modal", "max_hours", MODAL_MAX_HOURS)), MODAL_MAX_HOURS),
        "note": "dollars per hour; a count suffix like H100:8 multiplies the rate",
    }


def _ceilings_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="which project's allocation to report (defaults to current)")


@cli.command("ceilings", "spend headroom for this backend", setup=_ceilings_args)
def cmd_ceilings(args: argparse.Namespace) -> dict[str, Any]:
    """The same two ceilings `tools/jobs.py ceilings` reports, never conflated:
    the machine's rolling spend, and this project's own allocation (§15)."""
    cfg = config_mod.load()
    window = int(cfg.get("spend", "window_days", 30))
    stale = [r.id for r in ls.stale_runs(cfg=cfg)]
    payload: dict[str, Any] = {
        "platform": PLATFORM,
        "per_job_usd": cfg.get("spend", "per_job_usd"),
        "monthly_usd": cfg.get("spend", "monthly_usd"),
        "rolling": {k: v for k, v in ls.rolling_spend(window).items() if k != "runs"},
        "in_flight_runs": [r.id for r in ls.in_flight()],
        "stale_runs": stale,
        "blocked": bool(stale),
        "note": (
            "Modal bills per second against [modal.gpu_rates], so the [spend] ceilings are "
            "the only gate here -- there is no hours allowance and exit 13 never comes from "
            "this backend."
        ),
    }
    project_id = budget.resolve(args.project)
    if project_id and budget.exists(project_id):
        payload["project"] = budget.status(project_id)
    else:
        payload["project"] = {"project": project_id, "bounded": False}
    return payload


if __name__ == "__main__":
    main(cli)
