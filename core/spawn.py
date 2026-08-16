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
separate function rather than an argument.

**Which of the two a long-lived child gets is not a matter of taste.**
`DETACHED_PROCESS` reads like the stronger promise -- no console at all, rather
than one that is merely hidden -- and it is the weaker one in the only way that
matters here, because a console is *inherited*. A detached child has none to
lend, so the first console program *it* starts is given a fresh one, and a fresh
console is the black window this module exists to prevent. That is not
hypothetical: the Lab server was started detached, and `jupyter-lsp` runs
`npm prefix -g` to locate language servers (`npm` is `npm.cmd` on Windows, so
that is `cmd.exe`). The window appeared a second after the Lab tab opened, from
a grandchild nobody here wrote.

So `detached()` asks for `CREATE_NO_WINDOW` too: the child gets a console of its
own -- not the parent's, so closing a terminal cannot signal it -- and it is
never shown, and every console descendant inherits that invisibility. The
"outlives its parent" half of the promise is carried by
`CREATE_NEW_PROCESS_GROUP`, which is what actually keeps a Ctrl+C to our group
away from it; process lifetime was never tied to the console.

Everything here is a no-op off Windows, where none of it is a problem.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

WINDOWS = os.name == "nt"

#: "This is a console application; do not give it a window."
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if WINDOWS else 0
#: "A console of its own, never shown, and its own process group." Not
#: `DETACHED_PROCESS`, which the two flags above cannot be combined with anyway
#: -- and which is what gave *grandchildren* windows. See the module docstring.
DETACHED = (
    NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
) if WINDOWS else 0


def quiet() -> dict[str, Any]:
    """Keyword arguments for a child that should not open a window.

    For `subprocess.run`, `subprocess.Popen` and `asyncio.create_subprocess_exec`
    alike -- they all take `creationflags`, and all of them ignore it off
    Windows because the flag resolves to zero there.
    """
    return {"creationflags": NO_WINDOW} if NO_WINDOW else {}


def detached() -> dict[str, Any]:
    """Keyword arguments for a child that must outlive its parent, quietly.

    Quietly for its whole subtree, which is the part that is easy to get wrong:
    see the module docstring for why this is `CREATE_NO_WINDOW` rather than
    `DETACHED_PROCESS`.

    `start_new_session` off Windows is the same idea by another mechanism: the
    child leads its own process group, so a signal to ours does not reach it.
    """
    if WINDOWS:
        return {"creationflags": DETACHED}
    return {"start_new_session": True}


def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """`subprocess.run`, without a window. Callers pass everything else."""
    return subprocess.run(argv, **{**quiet(), **kwargs})


_sdk_masked = False


def mask_sdk_console() -> None:
    """Make the Agent SDK's own child -- the `claude` CLI -- spawn quietly.

    Everything this module quiets is a process *we* spawn, but the one console
    program this app cannot avoid starting is spawned by someone else:
    `claude-agent-sdk` launches `claude.exe` through `anyio.open_process` and
    passes no `creationflags`. From a terminal that is invisible -- the child
    inherits the console -- which is exactly why it survived development. From
    the installed app (`pythonw.exe`, no console) every new session put a black
    window titled "claude" on top of the workspace.

    `anyio.open_process` accepts `creationflags`; the SDK just never sends one.
    So the seam is anyio's: wrap it once, add `CREATE_NO_WINDOW` when the caller
    asked for nothing, and the child gets the same hidden console every other
    console child here gets -- along with every console descendant *it* starts,
    which is the property the module docstring is about.

    Idempotent, Windows-only, and deliberately narrow: an explicit
    `creationflags` from any caller is passed through untouched.
    """
    global _sdk_masked

    if not WINDOWS or _sdk_masked:
        return
    import anyio  # noqa: PLC0415 - the SDK's own dependency, present iff it is

    real = anyio.open_process

    async def _quiet_open_process(*args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = NO_WINDOW
        return await real(*args, **kwargs)

    anyio.open_process = _quiet_open_process
    _sdk_masked = True
