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


def utf8_env() -> dict[str, str]:
    """Environment for a Python child that will meet text from the internet.

    Python's `open()` defaults to `locale.getpreferredencoding()`, which on
    Windows is the ANSI code page rather than UTF-8. On the machine this was
    found on that is **cp1251** -- a Cyrillic code page, on a machine doing
    English-language ML research -- because the ANSI code page follows the
    system locale and has nothing to do with what the files contain.

    What the files contain is arXiv LaTeX, and there are two failures, one of
    which survives doing the obvious thing right:

        open(tex).read()
        UnicodeDecodeError: 'charmap' codec can't decode byte 0x98

        print(open(tex, encoding='utf-8').read())
        UnicodeEncodeError: 'charmap' codec can't encode character '\\xf6'

    The first is a curly quote -- `\\u2018` is `e2 80 98` in UTF-8, and cp1251
    has no character at 0x98. The second is the one worth the paragraph: the
    read was *correct*, and the crash moved to `print`, because the standard
    streams take their encoding from the same code page. Remembering
    `encoding="utf-8"` on every open is not sufficient and never was.

    `PYTHONUTF8=1` is PEP 540's UTF-8 Mode and answers both: it makes
    `getpreferredencoding` return utf-8, so `open()` defaults to it, and it
    reconfigures stdin/stdout/stderr to utf-8 with `surrogateescape`.

    **`PYTHONIOENCODING` is set too, and the error handler is why.** UTF-8 Mode
    alone looked sufficient and is not: `PYTHONIOENCODING` takes precedence over
    it for the standard streams, so an ambient one -- a stray export, a shell
    profile, a CI image -- silently defeats half the fix and leaves the `print`
    failure above exactly as it was. Setting it here overrides that.

    It is spelled `utf-8:surrogateescape` rather than `utf-8` for the reason
    that made leaving it out tempting: the bare form defaults the handler to
    `strict`, which turns a byte that survived a lossy read into a crash on the
    way out. Naming the handler keeps UTF-8 Mode's behaviour instead of
    replacing it with a stricter one.

    Nothing here is Windows-only. A Linux box with `LC_ALL=C` has the same
    problem in ASCII, and the fix is the same two variables.
    """
    return {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8:surrogateescape"}


def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """`subprocess.run`, without a window. Callers pass everything else."""
    return subprocess.run(argv, **{**quiet(), **kwargs})


def console_script(name: str) -> str | None:
    """Find a console script installed *beside this interpreter*, then on PATH.

    `shutil.which` alone was wrong here, and wrong in the way that is hardest to
    argue with: it searches `PATH`, and a virtualenv's `Scripts` directory is on
    `PATH` only while the environment is *activated*. Grad is launched from a
    desktop shortcut pointing at `.venv\\Scripts\\pythonw.exe`, and Explorer
    starts it with the ambient environment -- so the interpreter is the venv's
    and `PATH` is the machine's.

    That produced both halves of one bug report. `repowiki` was installed in the
    venv, `which` did not find it, and `tools/wiki.py` reported "repowiki is not
    installed" with a `pip install -e '.[wiki]'` that had already been run. And
    `kaggle` was found -- in the *user-site* Python, not the venv -- so the wiki
    said a package was missing while Kaggle silently shelled out to a different
    installation's CLI, against a version this project pins.

    Beside the interpreter first, therefore, because `sys.executable` is the one
    thing that is always right about which environment this is. PATH stays as
    the fallback for a tool that genuinely lives elsewhere.
    """
    import shutil  # noqa: PLC0415 - only this function needs it
    import sys  # noqa: PLC0415

    scripts = os.path.dirname(sys.executable)
    if scripts:
        # `shutil.which` with an explicit `path`, rather than joining a name and
        # testing it: on Windows the extension is PATHEXT's business, and
        # `repowiki` on disk is `repowiki.exe`.
        found = shutil.which(name, path=scripts)
        if found:
            return found
    return shutil.which(name)


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
