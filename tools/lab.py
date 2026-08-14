"""grad-lab -- the embedded JupyterLab server (HANDOFF-2 §19).

Amends §10's notebook handling, and §14 explains why this honours the original
reasoning rather than reversing it: the rejection was of *building a notebook
editor*, and that still stands -- we build none. What changed is that "editing
links out to Lab" was never wired up; `ui/app.py` pointed at a `localhost:8888`
nobody started. Embedding the real Lab is what makes arbitrary Lab extensions
possible at all.

**The kernel-ownership rule, which must not be lost.** `tools/nb.py` spawns
detached kernels through its own connection files. Lab has its own kernel
manager. Two owners over one notebook reproduces exactly the "works in the
kernel that grew it" failure that `nb verify` exists to catch. So the discipline
is unchanged: **anything edited in Lab passes `nb verify` before it is cited in
`notes/` or referenced from a ledger entry.** The Verify button in the
Notebooks tab is worth more than the embed, and it is why that was built first.

**Three things to know before installing an extension:**

1. *Server extensions run as you.* A frontend extension is confined to the
   browser; a server extension runs in this process with your filesystem
   rights -- it can read `ledger/` and `notes/`, and it can `import keyring` and
   reach the credential store. That is the same honest residual
   `core/credentials.py` already names. Read a server extension before
   installing it.
2. *Origin.* An extension is code running in Lab's origin, and the Lab iframe is
   deliberately unsandboxed. Lab stays on its own port and never shares the UI's
   storage secret.
3. *Pin everything.* The JupyterLab 3->4 break is what killed the Tabnine
   extension. `pyproject.toml`'s `lab` extra pins JupyterLab itself and every
   extension, so an unrelated `pip install -U` cannot take the app down.

"Connect an arbitrary extension" therefore means: add a pin, reinstall, restart.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from core import jsonl, paths
from core.cli import Cli, main
from core.errors import ConfigError, GradError

cli = Cli(
    "grad-lab",
    "Manage the JupyterLab server the UI embeds. Human editing surface; the "
    "agent still edits notebooks through Write/Edit plus tools/nb.py.",
    epilog=(
        "Lab and tools/nb.py are two kernel owners over one notebook, which is the\n"
        "'works in the kernel that grew it' failure nb verify exists to catch. The rule\n"
        "is unchanged:\n\n"
        "  python -m tools.nb verify notebooks/<name>.ipynb --json\n\n"
        "before anything edited here is cited in notes/ or referenced from the ledger.\n\n"
        "Server extensions run with your filesystem rights and can reach the credential\n"
        "store. Read one before installing it; frontend-only extensions are low risk."
    ),
)

DEFAULT_PORT = 8889


def _state_path() -> Path:
    return paths.data_dir() / "lab" / "lab.json"


def _log_path() -> Path:
    return paths.data_dir() / "lab" / "lab.log"


def _jupyter_config_dir() -> Path:
    return paths.root() / "config" / "jupyter"


def _read_state() -> dict[str, Any]:
    return jsonl.read_json(_state_path()) or {}


def _executable() -> str:
    """The `jupyter` entry point, or a clear error naming the extra to install."""
    found = shutil.which("jupyter")
    if found:
        return found
    raise ConfigError(
        "jupyter is not installed, so there is no Lab server to start",
        fix="pip install -e '.[lab]'   # pins jupyterlab and every extension",
    )


def _free_port(preferred: int) -> int:
    """The preferred port if it is free, otherwise an OS-assigned one.

    Reported back rather than assumed: the UI reads the port out of the state
    file, so a fallback does not strand the iframe on a dead address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, check=False,
        )
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# ---------------------------------------------------------------------------
def _start_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"preferred port (default {DEFAULT_PORT})")
    p.add_argument(
        "--ui-origin",
        default="http://127.0.0.1:8080",
        help="the Grad UI's origin, which is the only origin permitted to frame Lab",
    )
    p.add_argument("--force", action="store_true", help="start even if a server looks alive")


@cli.command("start", "start the Lab server and return its port and token", setup=_start_args)
def cmd_start(args: argparse.Namespace) -> dict[str, Any]:
    """Detached, on 127.0.0.1, behind a freshly minted token.

    The token is new on every start rather than persisted: it is written to a
    state file under `data/`, and a long-lived secret in a workspace file is a
    worse trade than re-reading the file after a restart.
    """
    state = _read_state()
    if not args.force and state.get("port") and _listening(int(state["port"])) and _alive(state.get("pid")):
        return {**state, "already_running": True,
                "next": "python -m tools.lab status --json"}

    executable = _executable()
    port = _free_port(args.port)
    token = secrets.token_urlsafe(32)
    log = _log_path()
    log.parent.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        # Read by config/jupyter/jupyter_server_config.py, so the framing
        # headers and the actual port cannot drift apart.
        "GRAD_UI_ORIGIN": args.ui_origin,
        "GRAD_LAB_PORT": str(port),
        "JUPYTER_CONFIG_DIR": str(_jupyter_config_dir()),
    }
    argv = [
        executable, "lab",
        "--no-browser",
        f"--port={port}",
        "--ip=127.0.0.1",
        f"--IdentityProvider.token={token}",
        f"--ServerApp.root_dir={paths.root()}",
        f"--ServerApp.config_file={_jupyter_config_dir() / 'jupyter_server_config.py'}",
    ]

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        start_new_session = True

    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            argv, cwd=str(paths.root()), stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env,
            creationflags=creationflags, start_new_session=start_new_session,
        )

    deadline = time.time() + 30
    while time.time() < deadline and not _listening(port):
        if proc.poll() is not None:
            raise GradError(
                "lab_died",
                f"the Lab server exited immediately (code {proc.returncode})",
                exit_code=8,
                fix=f"read {log}",
                detail={"log": str(log)},
            )
        time.sleep(0.3)

    record = {
        "port": port,
        "token": token,
        "pid": proc.pid,
        "url": f"http://127.0.0.1:{port}/lab",
        "root_dir": str(paths.root()),
        "ui_origin": args.ui_origin,
        "log": str(log),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "listening": _listening(port),
    }
    jsonl.write_json(_state_path(), record)
    if not record["listening"]:
        raise GradError(
            "lab_not_listening",
            f"the Lab server did not start listening on port {port} within 30s",
            exit_code=8,
            fix=f"read {log}",
            detail=record,
        )
    return {
        **record,
        # The rule that survives the embed.
        "discipline": (
            "anything edited in Lab passes `python -m tools.nb verify <path> --json` "
            "before it is cited in notes/ or referenced from a ledger entry"
        ),
    }


@cli.command("status", "is the Lab server up, and where")
def cmd_status(_: argparse.Namespace) -> dict[str, Any]:
    state = _read_state()
    if not state:
        return {"running": False, "fix": "python -m tools.lab start --json"}
    port = int(state.get("port") or 0)
    return {
        **{k: v for k, v in state.items() if k != "token"},
        "running": bool(port and _listening(port) and _alive(state.get("pid"))),
        "process_alive": _alive(state.get("pid")),
        # The token is what the iframe needs and what a screenshot should not
        # carry. `lab url` is the deliberate way to get one that includes it.
        "token_available": bool(state.get("token")),
    }


@cli.command(
    "url",
    "the full URL for one notebook, token included (for the UI)",
    setup=lambda p: p.add_argument("path", nargs="?", help="notebook path relative to the workspace"),
)
def cmd_url(args: argparse.Namespace) -> dict[str, Any]:
    state = _read_state()
    if not state.get("port"):
        raise GradError(
            "lab_not_started", "the Lab server is not running", exit_code=3,
            fix="python -m tools.lab start --json",
        )
    base = f"http://127.0.0.1:{state['port']}/lab"
    target = f"{base}/tree/{args.path}" if args.path else base
    return {"url": f"{target}?token={state['token']}", "port": state["port"]}


@cli.command("extensions", "what is installed, so the state is inspectable")
def cmd_extensions(_: argparse.Namespace) -> dict[str, Any]:
    """Lab already has a plugin system; we do not design one.

    What this builds is the reproducibility layer: the extension set is declared
    in `pyproject.toml`'s `lab` extra rather than accumulated, and this command
    is how you see what actually ended up installed.
    """
    executable = _executable()
    env = {**os.environ, "JUPYTER_CONFIG_DIR": str(_jupyter_config_dir())}

    def _run(argv: list[str]) -> dict[str, Any]:
        try:
            out = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, env=env, check=False
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "timed out"}
        return {
            "ok": out.returncode == 0,
            "output": ((out.stdout or "") + (out.stderr or "")).strip().splitlines(),
        }

    return {
        "frontend": _run([executable, "labextension", "list"]),
        # The ones that matter for the §19 caveat: a server extension runs in
        # the Lab process with your filesystem rights.
        "server": _run([executable, "server", "extension", "list"]),
        "declared_in": str(paths.root() / "pyproject.toml") + " [project.optional-dependencies] lab",
        "caveat": (
            "server extensions run as you: they can read ledger/ and notes/, and can "
            "import keyring and reach the credential store. Read one before installing it."
        ),
    }


@cli.command("stop", "stop the Lab server")
def cmd_stop(_: argparse.Namespace) -> dict[str, Any]:
    state = _read_state()
    pid = state.get("pid")
    if not pid or not _alive(pid):
        jsonl.write_json(_state_path(), {})
        return {"stopped": False, "note": "no Lab server was running"}
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        import signal  # noqa: PLC0415

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 10
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    jsonl.write_json(_state_path(), {})
    return {"stopped": True, "pid": pid, "still_alive": _alive(pid)}


# ---------------------------------------------------------------------------
def lab_state() -> dict[str, Any]:
    """Read by the UI so the Lab tab knows where to point its iframe."""
    state = _read_state()
    port = int(state.get("port") or 0)
    state["running"] = bool(port and _listening(port))
    return state


if __name__ == "__main__":
    main(cli)
