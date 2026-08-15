"""grad-nb -- the persistent Jupyter kernel (HANDOFF §6).

    "The kernel is the agent's only compute channel; a training cell blocks it."

Three commands and one rule. `exec` runs code in a kernel that survives between
CLI invocations, bounded by a wall clock; exceeding the bound is an error whose
message says to move the work to `jobs.py`. `verify` restarts the kernel and
runs a notebook top to bottom, exiting non-zero on the first failure -- because
a persistent kernel plus an agent editing cells in place produces notebooks that
work live and fail on a clean run, and a system-prompt line is the weak form of
that fix.

Figures are written to `figures/NNN.png` and the path is printed; the agent
Reads the path (which handles images) and the UI renders it inline.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core import appdata, config as config_mod, paths, spawn
from core.cli import Cli, main
from core.errors import EXIT_CHECK_FAILED, ConfigError, GradError, NotFound, UsageError

cli = Cli(
    "grad-nb",
    "Run code in a persistent Jupyter kernel, and verify notebooks on a fresh one.",
    epilog=(
        "The kernel is for exploration. Anything long is a job:\n"
        "  python -m tools.jobs submit --spec <spec> --expect <id> --json"
    ),
)

CONNECTION_DIR = "kernel"


def _jupyter() -> Any:
    try:
        import jupyter_client  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "jupyter_client is not installed, so there is no kernel to talk to",
            fix="pip install 'jupyter-client>=8.6' 'ipykernel>=6.29' nbformat",
        ) from exc
    return jupyter_client


def _conn_path(name: str) -> Path:
    d = appdata.state_dir() / CONNECTION_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


# ---------------------------------------------------------------------------
# kernel lifecycle
# ---------------------------------------------------------------------------
def _start_kernel(name: str, kernel_name: str) -> dict[str, Any]:
    """Start a kernel that outlives this process.

    This is the whole reason the kernel is "persistent": `exec` is invoked fresh
    for every cell, so the kernel must survive the CLI exiting. A
    `KernelManager`-owned kernel does not -- it is torn down with its manager,
    which is correct for a notebook server and useless here. So the connection
    file is written first and `ipykernel_launcher` is spawned detached.

    Through `core/spawn.py` rather than by hand: the flags for "outlives me and
    shows no window, for its children too" are subtle enough that a second copy
    of them is a second thing to get wrong, and this one *was* the second copy.
    """
    jc = _jupyter()
    conn = _conn_path(name)
    conn.unlink(missing_ok=True)
    jc.write_connection_file(fname=str(conn), kernel_name=kernel_name)

    log = conn.with_suffix(".log")
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "ipykernel_launcher", "-f", str(conn)],
            cwd=str(paths.root()),
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **spawn.detached(),
        )
    conn.with_suffix(".pid").write_text(str(proc.pid), encoding="utf-8")
    return {"connection_file": str(conn), "kernel_name": kernel_name, "pid": proc.pid, "started": True}


def _connect(name: str, ready_timeout: float) -> Any:
    """Attach to the kernel described by a connection file and wait for it.

    `wait_for_ready` raises "Kernel died before replying to kernel_info" while a
    freshly spawned kernel is still binding its sockets, because a standalone
    client infers liveness from a heartbeat channel that is not beating yet. So
    it is retried against an overall deadline rather than trusted once.
    """
    jc = _jupyter()
    conn = _conn_path(name)
    client = jc.BlockingKernelClient()
    client.load_connection_info(json.loads(conn.read_text(encoding="utf-8")))
    client.start_channels()
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        try:
            client.wait_for_ready(timeout=3)
            return client
        except (RuntimeError, TimeoutError):
            time.sleep(0.4)
    client.stop_channels()
    return None


def _client(name: str, kernel_name: str, *, autostart: bool = True) -> Any:
    conn = _conn_path(name)
    if conn.exists():
        client = _connect(name, 15)
        if client is not None:
            return client
        # A stale connection file: the kernel died between invocations.
        conn.unlink(missing_ok=True)
    if not autostart:
        raise NotFound(
            f"no live kernel named {name!r}",
            fix="python -m tools.nb restart --json   # starts one",
        )
    _start_kernel(name, kernel_name)
    client = _connect(name, 60)
    if client is None:
        raise GradError(
            "kernel_start_failed",
            f"kernel {name!r} did not become ready",
            exit_code=EXIT_CHECK_FAILED,
            fix=f"read the kernel log: {conn.with_suffix('.log')}  (is ipykernel installed?)",
        )
    return client


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _next_figure_path() -> Path:
    paths.figures_dir().mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for p in paths.figures_dir().glob("*.png") if (m := re.fullmatch(r"(\d+)", p.stem))]
    return paths.figures_dir() / f"{(max(existing) + 1) if existing else 1:03d}.png"


def _stop_running_cell(client: Any, kernel: str | None) -> None:
    """Stop a cell that blew its wall clock, by whatever means the platform has.

    `interrupt_request` over the control channel works for a normally launched
    ipykernel on POSIX. On Windows ipykernel reports message-based interrupts as
    unsupported and the cell keeps running, and this CLI launches the kernel
    directly rather than through a KernelManager that could deliver a real
    signal -- so there the kernel is terminated and the next `exec` gets a fresh
    one. Losing the kernel's state is the lesser evil against a wedged kernel
    that silently swallows every later cell.
    """
    try:
        client.control_channel.send(client.session.msg("interrupt_request", {}))
    except Exception:  # noqa: BLE001, S110 - best effort; the fallback below is the real one
        pass
    if os.name != "nt":
        return
    time.sleep(0.5)
    try:
        client.stop_channels()
    except Exception:  # noqa: BLE001, S110 - channels that are already down are the desired state
        pass
    if kernel:
        _shutdown(kernel)


def execute(client: Any, code: str, timeout: float, *, kernel: str | None = None) -> dict[str, Any]:
    """Run one cell, collect its outputs, and save images to figures/.

    Returns a structured result rather than a transcript: `ok`, `stdout`,
    `result`, `error`, `figures`. A traceback comes back as a list of lines, not
    as a blob the model has to re-parse.
    """
    import base64

    msg_id = client.execute(code, allow_stdin=False)
    deadline = time.time() + timeout
    stdout: list[str] = []
    stderr: list[str] = []
    result: Any = None
    error: dict[str, Any] | None = None
    figures: list[str] = []

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            # Do not just walk away: the cell keeps running, so the kernel stays
            # busy and the *next* exec would queue behind it with no sign why.
            _stop_running_cell(client, kernel)
            raise GradError(
                "kernel_timeout",
                f"the cell exceeded the {timeout:.0f}s wall clock and was abandoned",
                exit_code=EXIT_CHECK_FAILED,
                fix=(
                    "the kernel is for exploration; move long work to a job:\n"
                    "  python -m tools.jobs submit --spec <spec> --expect <id> --json\n"
                    "then `python -m tools.nb restart` to get a clean kernel back"
                ),
            )
        try:
            msg = client.get_iopub_msg(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        kind = msg["header"]["msg_type"]
        content = msg["content"]

        if kind == "stream":
            (stdout if content.get("name") == "stdout" else stderr).append(content.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            data = content.get("data", {})
            if "image/png" in data:
                path = _next_figure_path()
                path.write_bytes(base64.b64decode(data["image/png"]))
                figures.append(str(path))
            if "text/plain" in data and kind == "execute_result":
                result = data["text/plain"]
        elif kind == "error":
            error = {
                "ename": content.get("ename"),
                "evalue": content.get("evalue"),
                "traceback": [_strip_ansi(t) for t in content.get("traceback", [])],
            }
        elif kind == "status" and content.get("execution_state") == "idle":
            break

    return {
        "ok": error is None,
        "stdout": "".join(stdout),
        "stderr": "".join(stderr),
        "result": result,
        "error": error,
        "figures": figures,
    }


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _exec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--code", help="code to run")
    p.add_argument("--file", help="run the contents of this file")
    p.add_argument("--timeout", type=float, help="wall clock seconds (default from config)")
    p.add_argument("--kernel", default="default", help="named kernel session")


@cli.command("exec", "run code in the persistent kernel (timeout-bounded)", setup=_exec_args)
def cmd_exec(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    if not args.code and not args.file:
        raise UsageError("give --code or --file", fix="python -m tools.nb exec --code 'import sympy' --json")
    code = args.code or Path(args.file).read_text(encoding="utf-8")
    timeout = args.timeout or float(cfg.get("notebook", "exec_timeout_s", 300))
    client = _client(args.kernel, str(cfg.get("notebook", "kernel_name", "python3")))
    try:
        out = execute(client, code, timeout, kernel=args.kernel)
    finally:
        client.stop_channels()
    if not out["ok"]:
        raise GradError(
            "cell_error",
            f"{out['error']['ename']}: {out['error']['evalue']}",
            exit_code=EXIT_CHECK_FAILED,
            fix="fix the cell and re-run; the kernel still holds its previous state",
            detail=out,
        )
    return out


def _verify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("notebook")
    p.add_argument("--timeout", type=float, help="per-cell wall clock (default from config)")
    p.add_argument("--write", action="store_true", help="write executed outputs back into the notebook")


@cli.command("verify", "run a notebook top to bottom on a fresh kernel", setup=_verify_args)
def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    """Non-zero on the first failing cell.

    Run this before a notebook is cited in `notes/` or referenced from a ledger
    entry: a notebook that only works in the kernel that grew it is not evidence.
    """
    try:
        import nbformat  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError("nbformat is not installed", fix="pip install nbformat") from exc

    path = Path(args.notebook)
    if not path.is_file():
        raise NotFound(f"notebook {path} not found", fix="check the path")
    cfg = config_mod.load()
    timeout = args.timeout or float(cfg.get("notebook", "verify_timeout_s", 1800))

    nb = nbformat.read(path, as_version=4)
    session = f"verify-{path.stem}"
    _conn_path(session).unlink(missing_ok=True)
    _start_kernel(session, str(cfg.get("notebook", "kernel_name", "python3")))

    executed = 0
    client = None
    try:
        # Inside the protected region: if the client never becomes ready,
        # `_client` raises, and the kernel spawned a line above would otherwise
        # be left running -- holding the GPU memory the verify was meant to free.
        client = _client(session, str(cfg.get("notebook", "kernel_name", "python3")), autostart=False)
        for index, cell in enumerate(nb.cells):
            if cell.get("cell_type") != "code" or not (cell.get("source") or "").strip():
                continue
            out = execute(client, cell["source"], timeout, kernel=session)
            executed += 1
            if args.write:
                cell["outputs"] = _as_nb_outputs(out)
                cell["execution_count"] = executed
            if not out["ok"]:
                if args.write:
                    nbformat.write(nb, path)
                raise GradError(
                    "notebook_cell_failed",
                    f"cell {index} failed: {out['error']['ename']}: {out['error']['evalue']}",
                    exit_code=EXIT_CHECK_FAILED,
                    fix="fix the cell, then re-run `python -m tools.nb verify` -- a clean top-to-bottom run is the contract",
                    detail={"cell_index": index, "cells_executed": executed, **out},
                )
    finally:
        if client is not None:
            client.stop_channels()
        _shutdown(session)

    if args.write:
        nbformat.write(nb, path)
    return {"notebook": str(path), "cells_executed": executed, "clean": True}


def _as_nb_outputs(out: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    if out["stdout"]:
        outputs.append({"output_type": "stream", "name": "stdout", "text": out["stdout"]})
    if out["stderr"]:
        outputs.append({"output_type": "stream", "name": "stderr", "text": out["stderr"]})
    if out["result"] is not None:
        outputs.append(
            {"output_type": "execute_result", "data": {"text/plain": out["result"]}, "metadata": {}, "execution_count": None}
        )
    return outputs


def _shutdown(name: str) -> None:
    """Ask the kernel to exit, then make sure it did.

    Because we spawn the kernel detached, nothing else will reap it: a kernel
    that ignores the shutdown request would otherwise sit holding VRAM the
    experiments need.
    """
    conn = _conn_path(name)
    pid_file = conn.with_suffix(".pid")
    if conn.exists():
        try:
            jc = _jupyter()
            client = jc.BlockingKernelClient()
            client.load_connection_info(json.loads(conn.read_text(encoding="utf-8")))
            client.start_channels()
            # Send the request, do not wait for the reply: `client.shutdown()`
            # blocks on a control-channel response that a wedged kernel will
            # never send, and this function is called from a `finally`.
            client.control_channel.send(client.session.msg("shutdown_request", {"restart": False}))
            time.sleep(0.4)
            client.stop_channels()
        except Exception:  # noqa: BLE001 - a kernel that is already gone is the desired state
            pass
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            time.sleep(0.3)
            os.kill(pid, 9 if os.name != "nt" else 15)
        except (ValueError, OSError, PermissionError):
            pass
        pid_file.unlink(missing_ok=True)
    conn.unlink(missing_ok=True)


@cli.command(
    "restart",
    "restart the persistent kernel (state is discarded)",
    setup=lambda p: p.add_argument("--kernel", default="default"),
)
def cmd_restart(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    _shutdown(args.kernel)
    info = _start_kernel(args.kernel, str(cfg.get("notebook", "kernel_name", "python3")))
    return {"kernel": args.kernel, **info}


@cli.command(
    "stop",
    "shut the kernel down",
    setup=lambda p: p.add_argument("--kernel", default="default"),
)
def cmd_stop(args: argparse.Namespace) -> dict[str, Any]:
    _shutdown(args.kernel)
    return {"kernel": args.kernel, "stopped": True}


@cli.command("status", "which kernels have connection files")
def cmd_status(_: argparse.Namespace) -> dict[str, Any]:
    d = appdata.state_dir() / CONNECTION_DIR
    return {
        "kernels": [p.stem for p in d.glob("*.json")] if d.exists() else [],
        "figures_dir": str(paths.figures_dir()),
    }


if __name__ == "__main__":
    main(cli)
