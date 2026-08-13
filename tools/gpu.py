"""grad-gpu -- submit, watch, and collect jobs on known SSH hosts (HANDOFF §6, §7, §9).

The host inventory is hardcoded in `config/grad.toml` and an unknown host is a
configuration error, never an ad-hoc connection. Together with the key material
living in Windows Credential Manager rather than the environment, that is what
makes §9's "no general remote-execution capability" claim true: this CLI is not
a wrapper around ssh, it is a small allowlist over our own operations.

Same four gates as `jobs.py`, same `--smoke` carve-out, same collect contract.
SSH hosts have no billing API, so `collect` prices wall clock against the
per-host rate in the inventory (rate 0 for hosts that are free to use).
"""

from __future__ import annotations

import argparse
import os
import shlex
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from core import config as config_mod, credentials, gates, ledger_store as ls, submit as submit_lib
from core.cli import Cli, main
from core.config import Config, Host
from core.errors import EXIT_RUNNING, GradError, UpstreamError, UsageError
from core.submission import Submission, parse_override

cli = Cli(
    "grad-gpu",
    "Submit and collect jobs on known SSH GPU hosts.",
    epilog=(
        "Hosts come from the [hosts.*] inventory in config/grad.toml. There is no\n"
        "--host-address flag on purpose: an allowlist over our own operations is\n"
        "enforceable in a way that an allowlist over a shell is not.\n\n"
        "Cost is wall clock x the host's rate; rate 0 means the host is free."
    ),
)

PLATFORM = "ssh"
REMOTE_MARKER = "grad_status.json"


# ---------------------------------------------------------------------------
# ssh plumbing
# ---------------------------------------------------------------------------
class _Key:
    """A private key materialised for the lifetime of one call.

    ssh needs a key file, so the key is written to a mode-600 file in the OS
    temp directory and deleted immediately afterwards. This is weaker than never
    materialising it at all, and it is recorded here rather than hidden: the
    file never lands under the workspace, never enters the agent's environment,
    and exists only while a `gpu.py` subprocess is running. Prefer an SSH agent
    or a host entry in ~/.ssh/config where you can; leave `key_credential`
    unset and this class is never used.
    """

    def __init__(self, host: Host) -> None:
        self.host = host
        self.path: Path | None = None

    def __enter__(self) -> Path | None:
        if not self.host.key_credential:
            return None
        material = credentials.get(self.host.key_credential)
        fd, name = tempfile.mkstemp(prefix="grad-key-")
        os.close(fd)
        path = Path(name)
        path.write_text((material or "").rstrip("\n") + "\n", encoding="utf-8")
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        self.path = path
        return path

    def __exit__(self, *exc: Any) -> None:
        if self.path and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass


def _ssh_argv(host: Host, key: Path | None, remote_command: str) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if key:
        argv += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
    argv += [f"{host.user}@{host.hostname}" if host.user else host.hostname, remote_command]
    return argv


def _run(argv: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise UpstreamError(
            f"{argv[0]} is not on PATH",
            fix="install OpenSSH client tools (Windows: Settings > Optional features > OpenSSH Client)",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpstreamError(f"{argv[0]} timed out after {timeout}s", fix="check the host is reachable") from exc


def _ssh(host: Host, command: str, *, timeout: float = 300.0) -> str:
    with _Key(host) as key:
        proc = _run(_ssh_argv(host, key, command), timeout=timeout)
    if proc.returncode != 0:
        raise UpstreamError(
            f"ssh to {host.name} failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:400]}",
            fix=f"check connectivity and credentials for host {host.name!r}",
        )
    return proc.stdout


def _scp(host: Host, source: str, dest: str, *, recursive: bool = True, timeout: float = 1800.0) -> None:
    with _Key(host) as key:
        argv = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if recursive:
            argv.append("-r")
        if key:
            argv += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
        argv += [source, dest]
        proc = _run(argv, timeout=timeout)
    if proc.returncode != 0:
        raise UpstreamError(
            f"scp failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:400]}",
            fix="check the remote path exists and the key has access",
        )


def _remote(host: Host, path: str) -> str:
    prefix = f"{host.user}@{host.hostname}" if host.user else host.hostname
    return f"{prefix}:{path}"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
def _submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", required=True)
    p.add_argument("--host", help="host name from the inventory (defaults to the spec's target.host)")
    p.add_argument("--expect", help="expectation id to bind. REQUIRED unless --smoke")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--task")
    p.add_argument("--smoke", action="store_true", help="gate-exempt, hard-capped one-step check (§6)")
    p.add_argument("--no-digest", action="store_true", help=argparse.SUPPRESS)


@cli.command("submit", "submit a job (gated) or a smoke check (capped)", setup=_submit_args)
def cmd_submit(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    sub = Submission.load(
        args.spec,
        overrides=dict(parse_override(o) for o in args.overrides),
        resolve_digest=not args.no_digest,
    )
    host = cfg.host(args.host or sub.target.get("host") or "")

    if args.smoke:
        if args.expect:
            raise UsageError("--smoke binds no prediction", fix="drop --expect, or drop --smoke")
        result = run_smoke(sub, cfg, host=host)
        from tools import preflight

        preflight.record_check_result(sub.hash(), "smoke", result)
        if not result.get("ok"):
            raise GradError(
                "smoke_failed",
                result.get("reason", "the smoke check failed on the real host"),
                exit_code=9,
                fix=result.get("fix") or "read the smoke log under ledger/runs/",
                detail=result,
            )
        return {"smoke": result, "submission_hash": sub.hash()}

    # Gates first, then the record. Nothing is staged to a host until both the
    # four gates have passed and the run is on the ledger at its estimate.
    summary = submit_lib.check(sub, args.expect, cfg)
    run_id, _ = submit_lib.record_submission(
        sub,
        expectation_id=args.expect,
        platform=PLATFORM,
        target={"host": host.name, "platform": "ssh", "rate_usd_per_hour": host.rate_usd_per_hour},
        command=_command_for(sub),
        task=args.task,
    )
    remote_dir = f"{host.workdir}/{run_id}"
    _stage(host, sub, remote_dir)
    pid = _launch(host, sub, remote_dir, _command_for(sub))
    submit_lib.attach_handle(run_id, {"pid": pid, "remote_dir": remote_dir, "host": host.name})
    return {
        "run_id": run_id,
        "host": host.name,
        "remote_dir": remote_dir,
        "pid": pid,
        "gates": summary,
        "next": f"python -m tools.gpu collect {run_id} --json",
    }


def _command_for(sub: Submission) -> list[str]:
    if sub.target.get("command"):
        return [str(c) for c in sub.target["command"]]
    return ["python", sub.entrypoint.name, *sub.argv]


def _stage(host: Host, sub: Submission, remote_dir: str) -> None:
    """Copy the pipeline directory to the host. Everything in the submission
    hash comes from here, so the remote sees exactly what was preflighted."""
    _ssh(host, f"mkdir -p {shlex.quote(remote_dir)}")
    _scp(host, str(sub.spec_path.parent) + "/.", _remote(host, remote_dir))


def _launch(host: Host, sub: Submission, remote_dir: str, command: list[str]) -> str:
    """Start the job detached and record its own status marker remotely.

    The marker is what `status` and `collect` read, so neither has to scrape
    logs to know whether the job finished.
    """
    inner = " ".join(shlex.quote(c) for c in command)
    script = (
        f"cd {shlex.quote(remote_dir)} && "
        f"echo '{{\"state\":\"running\"}}' > {REMOTE_MARKER} && "
        f"nohup sh -c '{inner} > stdout.log 2> stderr.log; "
        f"printf \"{{\\\"state\\\":\\\"finished\\\",\\\"exit_code\\\":%d,\\\"ended_at\\\":\\\"%s\\\"}}\" "
        f"$? \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > {REMOTE_MARKER}' > /dev/null 2>&1 & echo $!"
    )
    return _ssh(host, script).strip() or "unknown"


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def run_smoke(sub: Submission, cfg: Config, *, host: Host | None = None) -> dict[str, Any]:
    """One step on the real host, capped in code (§6).

    This is the only check that sees the remote image, the remote driver stack,
    the real data path, and the real per-device batch size.
    """
    host = host or cfg.host(sub.target.get("host") or "")
    caps = gates.check_smoke_caps(sub, cfg)
    command = [*_command_for(sub), "--steps", str(caps["steps"]), "--smoke"]
    run_id = submit_lib.record_smoke_run(
        sub, cfg=cfg, platform=PLATFORM,
        target={"host": host.name, "platform": "ssh", "rate_usd_per_hour": host.rate_usd_per_hour},
        caps=caps, command=command,
    )
    artifacts = submit_lib.artifacts_dir(run_id)
    remote_dir = f"{host.workdir}/{run_id}"

    started = time.time()
    try:
        _stage(host, sub, remote_dir)
        inner = " ".join(shlex.quote(c) for c in command)
        proc_out = _ssh(
            host,
            f"cd {shlex.quote(remote_dir)} && timeout {caps['timeout_s']} sh -c {shlex.quote(inner)} 2>&1; echo EXIT:$?",
            timeout=caps["timeout_s"] + 60,
        )
    except GradError as exc:
        submit_lib.finish(
            run_id, status="failed", results={}, cost_usd_actual=0.0,
            artifacts_dir=artifacts, expectation=None, extra={"error": exc.message},
        )
        return {"ok": False, "reason": exc.message, "fix": exc.fix, "run_id": run_id}

    (artifacts / "smoke.log").write_text(proc_out, encoding="utf-8")
    exit_code = _exit_code_from(proc_out)
    hours = (time.time() - started) / 3600.0
    cost = round(hours * host.rate_usd_per_hour, 4)
    ok = exit_code == 0

    if not caps["artifact_upload"]:
        _ssh(host, f"rm -rf {shlex.quote(remote_dir)}", timeout=120)

    submit_lib.finish(
        run_id,
        status="completed" if ok else "failed",
        results={},
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=None,
        extra={"exit_code": exit_code, "smoke": True, "host": host.name},
    )
    return {
        "ok": ok,
        "run_id": run_id,
        "host": host.name,
        "exit_code": exit_code,
        "cost_usd": cost,
        "caps": caps,
        "log": str(artifacts / "smoke.log"),
        "output": "\n".join(proc_out.splitlines()[-25:]),
        "reason": None if ok else f"the smoke run exited {exit_code} on {host.name}",
        "fix": None if ok else f"read {artifacts / 'smoke.log'} -- this is the environment the real job would have used",
        "scope": "remote; exercises the real driver stack, data path, and per-device batch size",
    }


def _exit_code_from(output: str) -> int:
    for line in reversed(output.splitlines()):
        if line.startswith("EXIT:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return -1
    return -1


# ---------------------------------------------------------------------------
# status / collect
# ---------------------------------------------------------------------------
def _marker(host: Host, remote_dir: str) -> dict[str, Any]:
    import json

    try:
        text = _ssh(host, f"cat {shlex.quote(remote_dir + '/' + REMOTE_MARKER)} 2>/dev/null || echo '{{}}'")
    except GradError:
        return {}
    try:
        return json.loads(text.strip() or "{}")
    except json.JSONDecodeError:
        return {}


@cli.command("status", "report a run's state without collecting it", setup=lambda p: p.add_argument("run_id"))
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    r = ls.run(args.run_id)
    handle = r.get("handle") or {}
    payload = {
        "run_id": r.id,
        "ledger_status": r.status,
        "collected": r.collected,
        "stale": ls.is_stale(r, cfg=cfg),
        "host": handle.get("host"),
        "remote_dir": handle.get("remote_dir"),
    }
    if handle.get("host") and not r.collected:
        payload["remote"] = _marker(cfg.host(handle["host"]), handle["remote_dir"])
    return payload


def _collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id")
    p.add_argument("--wait", action="store_true", help="poll until the job finishes")
    p.add_argument("--timeout", type=int, default=900, help="seconds, with --wait")
    p.add_argument("--keep-remote", action="store_true", help="do not delete the remote working directory")


@cli.command("collect", "fetch artifacts, compute deviations, write the run record", setup=_collect_args)
def cmd_collect(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    r = submit_lib.require_uncollected(args.run_id)
    handle = r.get("handle") or {}
    if not handle.get("remote_dir"):
        raise GradError(
            "no_handle",
            f"run {r.id} has no remote directory; it never reached a host",
            exit_code=3,
            fix=f"python -m tools.ledger show {r.id} --json",
        )
    host = cfg.host(handle["host"])
    remote_dir = handle["remote_dir"]

    deadline = time.time() + (args.timeout if args.wait else 0)
    marker = _marker(host, remote_dir)
    while marker.get("state") != "finished" and time.time() < deadline:
        time.sleep(10)
        marker = _marker(host, remote_dir)
    if marker.get("state") != "finished":
        raise GradError(
            "still_running",
            f"run {r.id} is still running on {host.name}",
            exit_code=EXIT_RUNNING,
            fix=f"python -m tools.gpu collect {r.id} --wait --timeout 3600 --json",
            detail={"run_id": r.id, "remote": marker},
        )

    artifacts = submit_lib.artifacts_dir(r.id)
    for name in ("stdout.log", "stderr.log", Path(r.get("metrics_file") or "metrics.json").name):
        try:
            _scp(host, _remote(host, f"{remote_dir}/{name}"), str(artifacts / name), recursive=False, timeout=600)
        except GradError:
            continue
    for extra in (r.get("config") or {}).get("artifact_paths", []):
        try:
            _scp(host, _remote(host, f"{remote_dir}/{extra}"), str(artifacts), timeout=1800)
        except GradError:
            continue

    results: dict[str, Any] = {}
    metrics_error = None
    try:
        results = submit_lib.parse_metrics(artifacts / Path(r.get("metrics_file") or "metrics.json").name)
    except GradError as exc:
        metrics_error = exc.message

    expectation = None
    if r.get("expectation_id"):
        try:
            expectation = ls.expectation(r["expectation_id"])
        except GradError:
            expectation = None

    # No billing API on an SSH host: price wall clock against the inventory rate.
    ended = ls.parse_iso(marker.get("ended_at"))
    hours = submit_lib.elapsed_hours(r, until=ended)
    cost = round(hours * host.rate_usd_per_hour, 4)

    exit_code = marker.get("exit_code")
    record = submit_lib.finish(
        r.id,
        status="completed" if exit_code == 0 else "failed",
        results=results,
        cost_usd_actual=cost,
        artifacts_dir=artifacts,
        expectation=expectation,
        extra={
            "exit_code": exit_code,
            "host": host.name,
            "wall_clock_hours": round(hours, 4),
            "rate_usd_per_hour": host.rate_usd_per_hour,
            "cost_basis": "wall clock x host rate (no billing API on SSH hosts)",
            "metrics_error": metrics_error,
        },
    )
    if not args.keep_remote:
        try:
            _ssh(host, f"rm -rf {shlex.quote(remote_dir)}", timeout=120)
        except GradError:
            pass

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


@cli.command("hosts", "list the host inventory")
def cmd_hosts(_: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    return {
        "hosts": [
            {
                "name": h.name,
                "hostname": h.hostname,
                "user": h.user,
                "gpus": h.gpus,
                "rate_usd_per_hour": h.rate_usd_per_hour,
                "workdir": h.workdir,
                "key_credential": h.key_credential,
                "notes": h.notes,
            }
            for h in cfg.hosts.values()
        ]
    }


if __name__ == "__main__":
    main(cli)
