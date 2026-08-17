"""The Lab surface (§19) and RepoWiki (§20).

Both are thin wrappers around external processes, so what is tested here is the
part that is *ours*: the rules the wrappers exist to enforce. For Lab that is
the framing configuration and the kernel-ownership discipline; for RepoWiki it
is the scope allowlist and the staleness check.

Neither test starts a real server or a real scan. §24's discipline holds: no
network, no external process.
"""

from __future__ import annotations

import os

import argparse
import json

import pytest

from core import appdata, jsonl, paths
from core.errors import ConfigError, GradError, UsageError
from tools import lab, wiki


# ---------------------------------------------------------------------------
# §19: JupyterLab
# ---------------------------------------------------------------------------
def test_the_jupyter_config_permits_framing_from_the_app_origin(workspace):
    """"JupyterLab ships X-Frame-Options / CSP headers that block embedding."
    Getting this wrong costs an afternoon and produces a blank iframe."""
    source = (paths.root().parent / "config" / "jupyter" / "jupyter_server_config.py")
    if not source.exists():
        source = _repo_root() / "config" / "jupyter" / "jupyter_server_config.py"
    text = source.read_text(encoding="utf-8")

    assert "tornado_settings" in text
    assert "frame-ancestors" in text
    assert "GRAD_UI_ORIGIN" in text
    # Scoped to the app, never to `*`: Lab can execute code as this user.
    assert "frame-ancestors *" not in text


def test_lab_binds_to_localhost_only(workspace):
    text = (_repo_root() / "config" / "jupyter" / "jupyter_server_config.py").read_text(
        encoding="utf-8"
    )
    assert '"127.0.0.1"' in text
    assert "allow_remote_access = False" in text


def test_the_kernel_ownership_rule_is_recorded_where_it_is_needed(workspace):
    """"Two owners over one notebook reproduces exactly the 'works in the kernel
    that grew it' failure that `nb.py verify` exists to catch." The rule must be
    visible in the module that creates the second owner."""
    assert "nb verify" in lab.__doc__
    assert "nb.py" in (
        _repo_root() / "config" / "jupyter" / "jupyter_server_config.py"
    ).read_text(encoding="utf-8")


def test_status_reports_not_running_before_a_start(workspace):
    result = lab.cmd_status(argparse.Namespace(json=True))
    assert result["running"] is False
    assert "tools.lab start" in result["fix"]


def test_the_token_is_not_in_the_status_payload(workspace):
    """A status output is the sort of thing that ends up in a screenshot."""
    jsonl.write_json(
        appdata.state_dir() / "lab" / "lab.json",
        {"port": 8889, "token": "super-secret", "pid": 1, "url": "http://127.0.0.1:8889/lab"},
    )
    payload = lab.cmd_status(argparse.Namespace(json=True))
    assert "token" not in payload
    assert payload["token_available"] is True
    assert "super-secret" not in json.dumps(payload)


def test_url_includes_the_token_because_the_iframe_needs_it(workspace):
    jsonl.write_json(
        appdata.state_dir() / "lab" / "lab.json",
        {"port": 8889, "token": "tok123", "pid": 1},
    )
    result = lab.cmd_url(argparse.Namespace(path="notebooks/a.ipynb", json=True))
    assert result["url"].endswith("?token=tok123")
    assert "notebooks/a.ipynb" in result["url"]


def test_url_refuses_before_a_start(workspace):
    with pytest.raises(GradError) as exc:
        lab.cmd_url(argparse.Namespace(path=None, json=True))
    assert exc.value.code == "lab_not_started"


def test_notebook_edit_stays_denied(workspace):
    """"This item is about the *human* editing by hand; the agent continues to
    edit notebooks through Write/Edit plus nb.py." """
    import agent

    assert "NotebookEdit" in agent.DENIED_TOOLS
    assert "Task" in agent.DENIED_TOOLS


def test_the_lab_extra_pins_exactly(workspace):
    """"Pin `jupyterlab` itself and every extension, or an unrelated
    `pip install -U` takes the app down." The JupyterLab 3->4 break is what
    killed the Tabnine extension."""
    import tomllib

    doc = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    pins = doc["project"]["optional-dependencies"]["lab"]
    assert pins
    assert any(p.startswith("jupyterlab==") for p in pins), "JupyterLab itself must be pinned"
    for pin in pins:
        assert "==" in pin, f"{pin} is not pinned exactly"


# ---------------------------------------------------------------------------
# §20: RepoWiki
# ---------------------------------------------------------------------------
def test_the_scope_is_an_allowlist(workspace):
    """"**Never** `ledger/`, `notes/`, or any papers directory -- it ships
    content to a third party, and those hold research data." """
    assert wiki.SCOPE == ("core", "tools")
    for forbidden in ("ledger", "notes", "data", "figures", "evals"):
        assert forbidden not in wiki.SCOPE


def test_the_source_hash_covers_only_the_scope(workspace):
    (workspace / "core").mkdir(parents=True, exist_ok=True)
    (workspace / "core" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "notes").mkdir(parents=True, exist_ok=True)
    (workspace / "notes" / "secret.py").write_text("y = 2\n", encoding="utf-8")

    digest = wiki.source_hash(workspace)
    assert "core/a.py" in digest["files"]
    assert not any("notes" in k for k in digest["files"])


def test_the_source_hash_moves_when_the_code_does(workspace):
    (workspace / "core").mkdir(parents=True, exist_ok=True)
    target = workspace / "core" / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    before = wiki.source_hash(workspace)["hash"]
    target.write_text("x = 2\n", encoding="utf-8")
    assert wiki.source_hash(workspace)["hash"] != before


def test_research_data_changing_does_not_invalidate_the_wiki(workspace):
    (workspace / "core").mkdir(parents=True, exist_ok=True)
    (workspace / "core" / "a.py").write_text("x = 1\n", encoding="utf-8")
    before = wiki.source_hash(workspace)["hash"]
    (workspace / "notes").mkdir(parents=True, exist_ok=True)
    (workspace / "notes" / "log.md").write_text("today I learned\n", encoding="utf-8")
    assert wiki.source_hash(workspace)["hash"] == before


def test_check_refuses_when_no_wiki_exists(workspace):
    with pytest.raises(GradError) as exc:
        wiki.cmd_check(argparse.Namespace(json=True))
    assert exc.value.code == "no_wiki"


def test_the_scope_is_resolved_against_the_installation_not_the_workspace(workspace, monkeypatch):
    """`core/` and `tools/` ship with the code. Resolved against `paths.root()`
    they exist only in a checkout you are developing in -- in every installed
    configuration the workspace is a research folder, so `map` raised "none of
    core, tools exist under …" and the map could never be generated at all."""
    from core import paths as paths_mod

    installed = workspace.parent / "installation"
    (installed / "core").mkdir(parents=True, exist_ok=True)
    (installed / "core" / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(paths_mod, "install_dir", lambda: installed)

    assert wiki.source_root() == installed
    assert "core/a.py" in wiki.source_hash()["files"]
    assert wiki._scope_paths(wiki.source_root()) == [str(installed / "core")]


def test_check_detects_staleness_and_names_the_files(workspace, monkeypatch):
    """"A wiki behind the code is worse than none, because it is trusted." """
    (workspace / "core").mkdir(parents=True, exist_ok=True)
    target = workspace / "core" / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    # The source tree is the installation's; this test's is the temp workspace.
    monkeypatch.setattr(wiki, "source_root", lambda: workspace)

    jsonl.write_json(
        wiki.output_dir() / "manifest.json",
        {"generated_at": "2026-08-14T00:00:00Z", "source": wiki.source_hash(workspace),
         "output_dir": str(wiki.output_dir())},
    )
    assert wiki.cmd_check(argparse.Namespace(json=True))["current"] is True

    target.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(GradError) as exc:
        wiki.cmd_check(argparse.Namespace(json=True))
    assert exc.value.code == "wiki_stale"
    assert "core/a.py" in exc.value.detail["changed"]


def test_scan_is_refused_with_the_reasoning(workspace):
    """"RepoWiki reads ANTHROPIC_API_KEY by default, which is exactly what
    `credentials.scrub_environment()` deletes." """
    with pytest.raises(UsageError) as exc:
        wiki.cmd_scan(argparse.Namespace(json=True))
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "map" in (exc.value.fix or "")


def test_wiki_is_not_in_the_agents_tool_list(workspace):
    """"Scope: human-facing only. Not in the agent's tool list, not in
    prompts/system.md, no context cost." """
    system = (_repo_root() / "prompts" / "system.md").read_text(encoding="utf-8")
    assert "tools.wiki" not in system
    assert "repowiki" not in system.lower()


def test_a_missing_repowiki_names_the_extra(workspace, monkeypatch):
    from core import spawn

    monkeypatch.setattr(spawn, "console_script", lambda name: None)
    with pytest.raises(ConfigError) as exc:
        wiki._repowiki()
    assert "[wiki]" in (exc.value.fix or "")


def test_a_repowiki_in_the_venv_is_found_without_activating_it(workspace, monkeypatch, tmp_path):
    r"""The bug this replaced `shutil.which` for.

    A virtualenv's `Scripts` directory is on PATH only while the environment is
    *activated*, and the desktop shortcut points straight at
    `.venv\Scripts\pythonw.exe` -- so the interpreter was the venv's and PATH
    was the machine's. `repowiki` was installed, `which` did not find it, and
    the error said "not installed" with a `pip install` that had already been
    run.
    """
    import shutil
    import sys

    from core import spawn

    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    installed = scripts / ("repowiki.exe" if os.name == "nt" else "repowiki")
    installed.write_text("", encoding="utf-8")
    installed.chmod(0o755)

    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    # Nothing on PATH at all: the venv copy is the only one, which is exactly the
    # situation that used to report "not installed".
    monkeypatch.setattr(shutil, "which", _venv_only_which())

    found = spawn.console_script("repowiki")
    # Case-folded: on Windows `which` returns the name with PATHEXT's casing, so
    # a file written as `repowiki.exe` comes back as `repowiki.EXE`.
    assert found is not None
    assert found.casefold() == str(installed).casefold()


def _venv_only_which():
    """`shutil.which` that answers only when given an explicit `path`.

    Which is the whole shape of the bug: the tool exists on disk beside the
    interpreter and is invisible to a PATH search.
    """
    import shutil as _shutil

    real = _shutil.which

    def which(name, mode=os.F_OK | os.X_OK, path=None):
        if path is None:
            return None            # PATH does not have it
        return real(name, mode=mode, path=path)

    return which


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# review fixes
# ---------------------------------------------------------------------------
def _server_app(monkeypatch, origin: str):
    """Execute the real Jupyter config and return the ServerApp it configured.

    Asserting on the resulting settings rather than grepping the source: the
    docstring deliberately *mentions* `xheaders` to record why it is the wrong
    lever, and a text search cannot tell an explanation from a setting.
    """
    source = (_repo_root() / "config" / "jupyter" / "jupyter_server_config.py").read_text(
        encoding="utf-8"
    )

    class _Section:
        """`get_config()` returns a traitlets `Config`, which creates a section
        the first time one is touched -- `c.ServerApp.ip = ...` needs no
        declaration, and neither does `c.LanguageServerManager.node_roots`. A
        stub that predeclares the sections it happens to know about turns a new
        setting in the real config into an AttributeError here, which says
        nothing about the setting and everything about the stub.
        """

        def __getattr__(self, name: str):
            section = _Section()
            setattr(self, name, section)  # traitlets caches it; so does this
            return section

    config = _Section()
    namespace: dict = {"get_config": lambda: config}
    monkeypatch.setenv("GRAD_UI_ORIGIN", origin)
    exec(compile(source, "jupyter_server_config.py", "exec"), namespace)
    return config.ServerApp


def test_x_frame_options_is_cleared_explicitly(workspace, monkeypatch):
    """`xheaders` controls trust of X-Forwarded-* proxy headers and has nothing
    to do with framing. Relying on it left Jupyter emitting
    `X-Frame-Options: SAMEORIGIN`, and since the UI and Lab are different
    origins the iframe stayed blank -- the exact failure the file prevents.
    """
    settings = _server_app(monkeypatch, "http://127.0.0.1:8080").tornado_settings
    assert settings["headers"]["X-Frame-Options"] == ""
    assert "xheaders" not in settings, "xheaders is not the lever for this"


def test_the_csp_is_well_formed_with_and_without_a_port(workspace, monkeypatch):
    """`rsplit(':', 1)[-1]` on a portless origin yields the hostname, and
    `http://localhost:example.com` is an invalid source that browsers drop --
    silently narrowing the allowed ancestors instead of widening them."""
    def csp_for(origin: str) -> str:
        return _server_app(monkeypatch, origin).tornado_settings["headers"][
            "Content-Security-Policy"
        ]

    with_port = csp_for("http://127.0.0.1:8080")
    assert "http://127.0.0.1:8080" in with_port
    assert "http://localhost:8080" in with_port

    without_port = csp_for("http://example.com")
    assert "http://example.com" in without_port
    assert "localhost:example.com" not in without_port
    # And it must still be a valid, non-empty directive.
    assert without_port.startswith("frame-ancestors 'self' ")


def test_repowiki_is_invoked_with_one_path_and_a_supported_format(workspace, monkeypatch):
    """repowiki 0.3.1's `map` takes exactly one `path`, `--format text|json`,
    and has no `--output` / `--open`. HANDOFF-2 §20's `--format html --open`
    would fail immediately, so the HTML is rendered here instead.
    """
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = '{"files": [{"path": "core/budget.py", "rank": 0.12, "language": "python", "lines": 300}]}'
        stderr = ""

    def fake_run(argv, **kw):
        calls.append(argv)
        return Result()

    (workspace / "core").mkdir(parents=True, exist_ok=True)
    (workspace / "core" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "tools").mkdir(parents=True, exist_ok=True)
    (workspace / "tools" / "b.py").write_text("y = 2\n", encoding="utf-8")

    from core import spawn

    monkeypatch.setattr(spawn, "console_script", lambda name: "repowiki")
    monkeypatch.setattr(wiki.subprocess, "run", fake_run)

    result = wiki.cmd_map(argparse.Namespace(top=200, open=False, json=True))

    assert len(calls) == 2, "one invocation per scope directory"
    for argv in calls:
        paths_given = [a for a in argv[2:] if not a.startswith("-") and a not in ("json", "200")]
        assert len(paths_given) == 1, f"map takes exactly one path: {argv}"
        assert "--format" in argv and argv[argv.index("--format") + 1] == "json"
        assert "--output" not in argv and "--open" not in argv

    # HTML is produced by us, because repowiki cannot emit it.
    assert result["html"].endswith("index.html")
    html = (wiki.output_dir() / "index.html").read_text(encoding="utf-8")
    assert "core/budget.py" in html


# ---------------------------------------------------------------------------
# the app ships the server it embeds
# ---------------------------------------------------------------------------
def test_the_desktop_app_brings_the_lab_server_with_it(workspace):
    """The notebook window's interior *is* Lab, so an app installed without a
    Lab server ships a window whose only content is a button that fails. That
    was the state: `lab` was an extra nobody's install line mentioned."""
    import tomllib

    doc = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    extras = doc["project"]["optional-dependencies"]
    assert "grad[lab]" in extras["ui"]
    assert any(p.startswith("jupyterlab==") for p in extras["lab"])


def test_the_extension_set_stays_opt_in(workspace):
    """Heavier than everything else in the file put together, and a preference
    rather than a requirement -- so it is a second extra rather than a reason
    to make the first one enormous."""
    import tomllib

    doc = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    extras = doc["project"]["optional-dependencies"]
    assert "grad[lab-extensions]" not in extras["ui"]
    for pin in extras["lab-extensions"]:
        assert "==" in pin, f"{pin} is not pinned exactly"


def test_the_missing_jupyter_message_names_an_install_that_provides_it(workspace, monkeypatch):
    from core.errors import ConfigError

    monkeypatch.setattr(lab.shutil, "which", lambda name: None)
    with pytest.raises(ConfigError) as exc:
        lab._executable()
    assert "[ui]" in (exc.value.fix or "")


# ---------------------------------------------------------------------------
# the origin the iframe is framed from
# ---------------------------------------------------------------------------
def test_the_websocket_accepts_both_spellings_of_the_loopback_host(workspace, monkeypatch):
    """`allow_origin` takes exactly one origin, and `127.0.0.1:8080` and
    `localhost:8080` are different origins to a browser. Naming only one is the
    confusing half of the failure: the page renders, the frame loads, and only
    the kernel connection dies."""
    import re

    app = _server_app(monkeypatch, "http://127.0.0.1:8080")
    pattern = app.allow_origin_pat
    assert re.fullmatch(pattern, "http://127.0.0.1:8080")
    assert re.fullmatch(pattern, "http://localhost:8080")
    assert not re.fullmatch(pattern, "http://evil.example")
    assert getattr(app, "allow_origin", "") != "*"


def test_a_server_running_on_the_wrong_origin_is_restarted(workspace, monkeypatch):
    """Framing headers are fixed at launch, and a blocked frame is reported by
    the browser as "127.0.0.1 refused to connect" -- which reads as a dead port
    and sends you looking for a server that is running perfectly well."""
    from core import jsonl

    state = {"port": 8889, "pid": 4242, "ui_origin": "http://127.0.0.1:8080", "token": "t"}
    jsonl.write_json(lab._state_path(), state)
    monkeypatch.setattr(lab, "_listening", lambda port: True)
    monkeypatch.setattr(lab, "_alive", lambda pid: True)

    class Restarted(Exception):
        """A sentinel, so "it got as far as launching" is an assertion rather
        than a mock of the launch itself."""

    stopped: list[bool] = []
    monkeypatch.setattr(lab, "cmd_stop", lambda _: stopped.append(True))
    monkeypatch.setattr(lab, "_executable", lambda: (_ for _ in ()).throw(Restarted()))

    # Same origin: left alone, which is the one case worth not restarting.
    same = lab.cmd_start(_start_namespace("http://127.0.0.1:8080"))
    assert same["already_running"] is True
    assert stopped == []

    # Different origin: the server has to come down for the header to change.
    with pytest.raises(Restarted):
        lab.cmd_start(_start_namespace("http://127.0.0.1:9000"))
    assert stopped == [True]


def _start_namespace(origin: str):
    import argparse

    return argparse.Namespace(port=lab.DEFAULT_PORT, ui_origin=origin, force=False)


def test_the_window_asks_the_page_which_origin_it_is_on(workspace):
    """`--port` moves the app, and pywebview may open `localhost` where the
    config assumed `127.0.0.1`. The page knows the answer to both; guessing
    covers neither."""
    import asyncio

    pytest.importorskip("nicegui", reason="the ui extra is not installed")
    from ui.windows import notebook as notebook_window

    class Page:
        @staticmethod
        async def run_javascript(code, timeout=None):
            assert "location.origin" in code
            return "http://localhost:8099"

    assert asyncio.run(notebook_window._origin(Page)) == "http://localhost:8099"


def test_a_page_that_cannot_answer_falls_back_to_the_port_the_app_bound(workspace, monkeypatch):
    import asyncio

    pytest.importorskip("nicegui", reason="the ui extra is not installed")
    from ui import app as app_mod
    from ui.windows import notebook as notebook_window

    class Gone:
        @staticmethod
        async def run_javascript(code, timeout=None):
            raise RuntimeError("the client disconnected")

    monkeypatch.setattr(app_mod, "PORT", 9123)
    assert asyncio.run(notebook_window._origin(Gone)) == "http://127.0.0.1:9123"


# ---------------------------------------------------------------------------
# no console windows
# ---------------------------------------------------------------------------
def test_a_long_lived_child_keeps_a_console_so_its_own_children_stay_quiet(workspace):
    """`DETACHED_PROCESS` gives a child no console, so the first console program
    *it* starts is given a fresh -- visible -- one. That is where the `npm
    prefix` window came from: jupyter-lsp probing for language servers under a
    Lab server we had started detached.

    `CREATE_NO_WINDOW` is the fix and is also mutually exclusive with
    `DETACHED_PROCESS` (both together is ERROR_INVALID_PARAMETER, not
    redundancy), so this asserts the swap rather than the coexistence.
    """
    import subprocess

    from core import spawn

    if not spawn.WINDOWS:
        assert spawn.detached() == {"start_new_session": True}
        return

    flags = spawn.detached()["creationflags"]
    assert flags & subprocess.CREATE_NO_WINDOW
    assert not flags & subprocess.DETACHED_PROCESS
    # The half of the promise the console was never carrying: a Ctrl+C to our
    # group must not reach it.
    assert flags & subprocess.CREATE_NEW_PROCESS_GROUP


def test_the_verify_kernel_is_spawned_through_the_same_one_definition(workspace):
    """`tools/nb.py` had its own hand-rolled copy of the flags, which meant its
    detached kernels had the same invisible-grandchild problem."""
    source = __import__("inspect").getsource(__import__("tools.nb", fromlist=["nb"]))
    assert "**spawn.detached()" in source
    assert "DETACHED_PROCESS" not in source


def test_the_lab_server_is_started_without_a_console(workspace, monkeypatch):
    """Every button in the workspace runs a CLI, and `ui.run(native=True)` is a
    GUI process with no console to lend -- so Windows gave each child a fresh
    one, which is a black window over the workspace."""
    from core import spawn

    seen: dict = {}

    class Fake:
        returncode = None
        pid = 999

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        seen.update(kwargs)
        return Fake()

    monkeypatch.setattr(lab, "_executable", lambda: "jupyter")
    monkeypatch.setattr(lab, "_free_port", lambda preferred: preferred)
    monkeypatch.setattr(lab, "_listening", lambda port: True)
    monkeypatch.setattr(lab.subprocess, "Popen", fake_popen)
    lab.cmd_start(_start_namespace("http://127.0.0.1:8080"))

    for key, value in spawn.detached().items():
        assert seen.get(key) == value


def test_the_workspaces_own_commands_are_started_without_a_console(workspace):
    """`tasklist` for the liveness check and the CLI itself were two windows
    per Lab start."""
    from core import spawn
    from ui import tasks as tasks_mod

    source = __import__("inspect").getsource(tasks_mod)
    assert source.count("**spawn.quiet()") >= 2
    assert spawn.quiet() == ({} if not spawn.WINDOWS else {"creationflags": spawn.NO_WINDOW})
