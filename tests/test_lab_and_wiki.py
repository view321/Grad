"""The Lab surface (§19) and RepoWiki (§20).

Both are thin wrappers around external processes, so what is tested here is the
part that is *ours*: the rules the wrappers exist to enforce. For Lab that is
the framing configuration and the kernel-ownership discipline; for RepoWiki it
is the scope allowlist and the staleness check.

Neither test starts a real server or a real scan. §24's discipline holds: no
network, no external process.
"""

from __future__ import annotations

import argparse
import json

import pytest

from core import jsonl, paths
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
        paths.data_dir() / "lab" / "lab.json",
        {"port": 8889, "token": "super-secret", "pid": 1, "url": "http://127.0.0.1:8889/lab"},
    )
    payload = lab.cmd_status(argparse.Namespace(json=True))
    assert "token" not in payload
    assert payload["token_available"] is True
    assert "super-secret" not in json.dumps(payload)


def test_url_includes_the_token_because_the_iframe_needs_it(workspace):
    jsonl.write_json(
        paths.data_dir() / "lab" / "lab.json",
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


def test_check_detects_staleness_and_names_the_files(workspace):
    """"A wiki behind the code is worse than none, because it is trusted." """
    (workspace / "core").mkdir(parents=True, exist_ok=True)
    target = workspace / "core" / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

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
    monkeypatch.setattr(wiki.shutil, "which", lambda name: None)
    with pytest.raises(ConfigError) as exc:
        wiki._repowiki()
    assert "[wiki]" in (exc.value.fix or "")


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
    config = type("Config", (), {})()
    config.ServerApp = type("ServerApp", (), {})()
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

    monkeypatch.setattr(wiki.shutil, "which", lambda name: "repowiki")
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
