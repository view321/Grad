"""Spawning a child process without flashing a console window.

One Windows detail, in one place, because it is invisible on the platform most
of this was written on and unmissable on the one it runs on.

A console-subsystem executable -- `python.exe`, `jupyter.exe`, `tasklist` --
inherits its parent's console when there is one. When there is *not*, Windows
allocates a fresh one, and a fresh console is a black window that appears over
whatever you were looking at. The desktop app is precisely the case with no
console to inherit: `ui.run(native=True)` is a GUI process, and every button in
the workspace runs a CLI. Starting JupyterLab flashed one window for the CLI and
another for the `tasklist` that checks whether a previous server is still alive.

`CREATE_NO_WINDOW` is the flag for "console app, no window". It is **mutually
exclusive with `DETACHED_PROCESS`** -- passing both fails with
`ERROR_INVALID_PARAMETER` rather than being redundant -- so `detached()` is a
separate function rather than an argument, and it does not need the flag: a
detached process has no console at all, which is a stronger promise than having
one that is not shown.

Everything here is a no-op off Windows, where none of it is a problem.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

WINDOWS = os.name == "nt"

#: "This is a console application; do not give it a window."
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if WINDOWS else 0
#: "This process does not belong to my console, and gets none of its own." The
#: two flags above cannot be combined; see the module docstring.
DETACHED = (
    getattr(subprocess, "DETACHED_PROCESS", 0)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
) if WINDOWS else 0


def quiet() -> dict[str, Any]:
    """Keyword arguments for a child that should not open a window.

    For `subprocess.run`, `subprocess.Popen` and `asyncio.create_subprocess_exec`
    alike -- they all take `creationflags`, and all of them ignore it off
    Windows because the flag resolves to zero there.
    """
    return {"creationflags": NO_WINDOW} if NO_WINDOW else {}


def detached() -> dict[str, Any]:
    """Keyword arguments for a child that must outlive its parent.

    `start_new_session` off Windows is the same idea by another mechanism: the
    child leads its own process group, so a signal to ours does not reach it.
    """
    if WINDOWS:
        return {"creationflags": DETACHED}
    return {"start_new_session": True}


def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """`subprocess.run`, without a window. Callers pass everything else."""
    return subprocess.run(argv, **{**quiet(), **kwargs})
