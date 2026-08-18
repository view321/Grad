"""The options a session runs under, for the two that describe its *world*.

`cwd` told the agent's shell where it was and nothing told it what `python`
meant, so the bare word resolved through whatever `PATH` the launcher happened
to carry. On the machine this was found on that was a global interpreter holding
a second, editable Grad -- so the app read one workspace through one
installation while the agent wrote it through another, and every tool call in
`prompts/system.md` is spelled `python -m tools.<name>`.

The file-checkpointing half is the same kind of claim: `core/rewind.py` puts the
conversation back and has never been able to put the *files* back, and the
option that changes that is one flag the SDK defaults to off.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import agent


# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------
def test_this_interpreters_scripts_directory_is_first(monkeypatch):
    """Not merely present -- first. `pip` had three entries ahead of the venv's
    on the machine this was written on, so anything but the front is a coin
    toss."""
    monkeypatch.setenv("PATH", r"C:\Python314\Scripts" + os.pathsep + r"C:\Windows")
    import sysconfig

    first = agent.interpreter_env()["PATH"].split(os.pathsep)[0]
    assert Path(first) == Path(sysconfig.get_path("scripts"))


def test_the_machines_path_is_kept_behind_it(monkeypatch):
    """Prepended, never replaced: the agent legitimately needs `git`, `docker`
    and `latexmk`, and this is not the place to decide what a research machine
    has installed."""
    monkeypatch.setenv("PATH", os.pathsep.join([r"C:\Windows", r"C:\tools"]))
    parts = agent.interpreter_env()["PATH"].split(os.pathsep)
    assert parts[1:] == [r"C:\Windows", r"C:\tools"]


def test_it_does_not_grow_a_copy_per_compaction(monkeypatch):
    """A compaction builds a fresh client through `build_options`, so this runs
    again in a process whose PATH it has already fixed. Applying it twice has to
    be the same as applying it once."""
    monkeypatch.setenv("PATH", r"C:\Windows")
    once = agent.interpreter_env()["PATH"]
    monkeypatch.setenv("PATH", once)
    assert agent.interpreter_env()["PATH"] == once


def test_an_empty_ambient_path_is_not_a_leading_separator(monkeypatch):
    """`"".split(os.pathsep)` is `[""]`, and joining that produces a PATH whose
    first entry is the empty string -- which POSIX reads as the current
    directory."""
    monkeypatch.setenv("PATH", "")
    value = agent.interpreter_env()["PATH"]
    assert value
    assert not value.endswith(os.pathsep)
    assert "" not in value.split(os.pathsep)


# ---------------------------------------------------------------------------
# VIRTUAL_ENV and PYTHONPATH
# ---------------------------------------------------------------------------
@pytest.mark.skipif(sys.prefix == sys.base_prefix, reason="not running in a venv")
def test_virtual_env_names_the_environment_actually_in_use(monkeypatch):
    """`pip` and `uv` both read it, and an ambient one pointing somewhere else is
    the exact confusion this function exists to end."""
    monkeypatch.setenv("VIRTUAL_ENV", r"C:\somewhere\else")
    assert agent.interpreter_env()["VIRTUAL_ENV"] == sys.prefix


def test_an_installed_grad_gets_no_pythonpath(monkeypatch):
    """Exporting the install directory unconditionally would put `config`,
    `data`, `notes` and `figures` on the import path as namespace packages, and
    `import data` is a thing research code genuinely does."""
    monkeypatch.setattr(agent, "_installed_as_distribution", lambda: True)
    assert "PYTHONPATH" not in agent.interpreter_env()


def test_a_bare_checkout_gets_one(monkeypatch):
    """`python agent.py` from a checkout nobody pip-installed: the agent's cwd is
    the workspace, and the workspace has no `tools/` in it."""
    monkeypatch.setattr(agent, "_installed_as_distribution", lambda: False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    from core import paths

    assert agent.interpreter_env()["PYTHONPATH"] == str(paths.install_dir())


def test_an_existing_pythonpath_is_kept_behind_ours(monkeypatch):
    monkeypatch.setattr(agent, "_installed_as_distribution", lambda: False)
    monkeypatch.setenv("PYTHONPATH", r"C:\theirs")
    assert agent.interpreter_env()["PYTHONPATH"].split(os.pathsep)[-1] == r"C:\theirs"


def test_the_distribution_check_never_raises(monkeypatch):
    """False is the safe answer: it adds a redundant path entry, where True on a
    machine that cannot answer is every tool call raising ModuleNotFoundError."""
    import importlib.metadata as md

    def boom(_name):
        raise RuntimeError("no metadata here")

    monkeypatch.setattr(md, "distribution", boom)
    assert agent._installed_as_distribution() is False


# ---------------------------------------------------------------------------
# text from the internet
# ---------------------------------------------------------------------------
#: A line of the sort every arXiv paper has in it: curly quotes and an umlaut.
#: Written as bytes so the test does not depend on this file's own encoding.
TEX_SAMPLE = b"the \xe2\x80\x98scaling law\xe2\x80\x99 of Sch\xc3\xb6lkopf\n"


def test_the_shell_asks_for_utf8_mode():
    from core import spawn

    assert agent.interpreter_env()["PYTHONUTF8"] == "1"
    assert spawn.utf8_env()["PYTHONUTF8"] == "1"


def test_the_stream_encoding_names_its_error_handler():
    """Two ways to get this wrong, and the test exists because the first draft
    got the first one wrong.

    Omitting `PYTHONIOENCODING` leaves an *ambient* one in force -- it takes
    precedence over UTF-8 Mode for the standard streams, so a stray export
    silently restores the `print` crash. Setting it to a bare `utf-8` defaults
    the handler to `strict`, which is a different regression: a byte that
    survived a lossy read then crashes on the way out.
    """
    assert agent.interpreter_env()["PYTHONIOENCODING"] == "utf-8:surrogateescape"


@pytest.mark.parametrize(
    "snippet",
    [
        # Reading it: the curly quote is `e2 80 98`, and a Windows ANSI code page
        # has no character at 0x98.
        "print(open(PATH).read())",
        # Reading it *correctly* and printing it, which is the failure that
        # survives remembering `encoding=`: the standard streams take their
        # encoding from the same code page.
        "print(open(PATH, encoding='utf-8').read())",
    ],
)
def test_reading_arxiv_latex_works_in_the_environment_the_agent_gets(tmp_path, snippet):
    """Run for real, in a child, because this bug lives entirely in the defaults
    of a fresh interpreter -- there is nothing to assert about it in-process,
    where the encoding was decided before the test started."""
    sample = tmp_path / "paper.tex"
    sample.write_bytes(TEX_SAMPLE)
    code = f"PATH = r'{sample}'\n{snippet}"

    hostile = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1251"}
    broken = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=hostile
    )
    if broken.returncode == 0:
        pytest.skip("this machine's default encoding decodes the sample anyway")
    assert "Unicode" in broken.stderr

    fixed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**hostile, **agent.interpreter_env()},
    )
    assert fixed.returncode == 0, fixed.stderr
    assert "Schölkopf" in fixed.stdout


def test_the_kernel_is_given_the_same_environment_as_the_shell(monkeypatch):
    """The kernel has two parents: the agent's Bash, which carries UTF-8 Mode,
    and the desktop app, which does not. "Reading a paper crashes in the
    notebook but not in the shell" is not a difference anyone should meet."""
    import tools.nb as nb

    captured = {}

    class _Proc:
        pid = 4321

    def fake_popen(_argv, **kwargs):
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(nb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(nb, "_jupyter", lambda: _FakeJupyterClient())

    nb._start_kernel("default", "python3")
    assert captured["env"]["PYTHONUTF8"] == "1"
    # Merged over the ambient environment, never replacing it: a kernel without
    # SYSTEMROOT or TEMP does not start.
    assert "PATH" in captured["env"]


class _FakeJupyterClient:
    @staticmethod
    def write_connection_file(fname: str, kernel_name: str) -> None:  # noqa: ARG004
        from pathlib import Path as _P

        _P(fname).write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# what actually reaches the SDK
# ---------------------------------------------------------------------------
def test_the_options_carry_the_environment(monkeypatch):
    """The function above can be perfect and change nothing if the dict never
    reaches `ClaudeAgentOptions` -- which is the state this file was written in.
    """
    sdk = pytest.importorskip("claude_agent_sdk", reason="the SDK is not installed")
    from core import config as config_mod

    monkeypatch.setattr(agent, "system_prompt", lambda: "prompt")
    options = agent.build_options(config_mod.load())
    assert isinstance(options, sdk.ClaudeAgentOptions)
    assert "PATH" in options.env
    import sysconfig

    first = options.env["PATH"].split(os.pathsep)[0]
    assert Path(first) == Path(sysconfig.get_path("scripts"))


def test_the_options_ask_the_cli_to_checkpoint_files(monkeypatch):
    """Defaults to off in the SDK, and every test of the *rewind* half passes
    with it dropped -- they stub the client. This is the one that notices."""
    sdk = pytest.importorskip("claude_agent_sdk", reason="the SDK is not installed")
    from core import config as config_mod

    monkeypatch.setattr(agent, "system_prompt", lambda: "prompt")
    options = agent.build_options(config_mod.load())
    assert options.enable_file_checkpointing is True
    # The SDK refuses the combination outright, so this is not a style note.
    assert options.session_store is None


def test_the_sdk_merges_our_environment_over_the_ambient_one():
    """The whole fix rests on `options.env` being a *patch* rather than a
    replacement -- if the SDK swapped the environment wholesale, the child would
    lose SYSTEMROOT, TEMP and the credential the session authenticates with.

    Pinned against the installed SDK's source rather than assumed, because this
    is the kind of thing a release changes quietly.
    """
    pytest.importorskip("claude_agent_sdk", reason="the SDK is not installed")
    from claude_agent_sdk._internal.transport import subprocess_cli

    source = Path(subprocess_cli.__file__).read_text(encoding="utf-8")
    assert "**inherited_env," in source
    assert "**self._options.env," in source
