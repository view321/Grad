"""The CLI contract (HANDOFF §8).

    "A failed CLI returns a stack trace on stderr and an exit code of 1, and the
     characteristic model response to that is to retry with guessed flags."

So: stable envelope, distinct exit codes, fixes not just faults, and unknown
flags that fail fast naming the closest valid one.
"""

from __future__ import annotations

import json

import pytest

from core.cli import Cli
from core.errors import EXIT_CONFIG, EXIT_NOT_FOUND, EXIT_OK, EXIT_USAGE, NotFound
from tools import ledger as ledger_cli, preflight as preflight_cli, quota as quota_cli


def envelope(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_success_envelope_shape(workspace, capsys):
    assert quota_cli.cli.run(["summary", "--json"]) == EXIT_OK
    payload = envelope(capsys)
    assert payload["ok"] is True and payload["error"] is None
    assert "by_stage" in payload["data"]


def test_error_envelope_carries_a_fix(workspace, capsys):
    assert ledger_cli.cli.run(["show", "run-nope", "--json"]) == EXIT_NOT_FOUND
    payload = envelope(capsys)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "not_found"
    assert payload["error"]["fix"]
    assert payload["error"]["exit_code"] == EXIT_NOT_FOUND


def test_unknown_flag_names_the_closest_valid_one(workspace, capsys):
    assert ledger_cli.cli.run(["query", "--pendingg", "--json"]) == EXIT_USAGE
    payload = envelope(capsys)
    assert "--pending" in payload["error"]["message"]


def test_unknown_flag_is_never_silently_ignored(workspace, capsys):
    assert quota_cli.cli.run(["summary", "--dayz", "3", "--json"]) == EXIT_USAGE
    assert envelope(capsys)["ok"] is False


def test_json_flag_is_accepted_before_or_after_the_subcommand(workspace, capsys):
    assert quota_cli.cli.run(["--json", "summary"]) == EXIT_OK
    assert envelope(capsys)["ok"] is True


def test_no_command_is_a_usage_error(workspace, capsys):
    assert quota_cli.cli.run(["--json"]) == EXIT_USAGE
    assert envelope(capsys)["error"]["code"] == "usage"


def test_gate_refusals_have_their_own_exit_codes(workspace, capsys):
    """A usage error, a gate refusal, and an upstream failure are three
    different things, and the model should not have to read prose to tell them
    apart."""
    d = workspace / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("print(1)\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/i@sha256:a'\n[target]\nplatform = 'hf'\n", encoding="utf-8"
    )
    from tools import jobs as jobs_cli

    code = jobs_cli.cli.run(["submit", "--spec", str(d / "spec.toml"), "--no-digest", "--json"])
    payload = envelope(capsys)
    assert code == 4  # EXIT_PREFLIGHT, not a generic 1
    assert payload["error"]["code"] == "preflight_missing"
    assert "preflight" in payload["error"]["fix"]


def test_internal_errors_do_not_print_a_traceback_on_stdout(workspace, capsys):
    cli = Cli("t", "test")

    @cli.command("boom", "raise")
    def _boom(_args):
        raise RuntimeError("kaboom")

    assert cli.run(["boom", "--json"]) == 1
    payload = envelope(capsys)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "internal"


def test_human_output_goes_to_stderr_on_failure(workspace, capsys):
    cli = Cli("t", "test")

    @cli.command("nope", "raise")
    def _nope(_args):
        raise NotFound("nothing here", fix="try something else")

    assert cli.run(["nope"]) == EXIT_NOT_FOUND
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fix: try something else" in captured.err


def test_help_lists_the_exit_codes(workspace, capsys):
    with pytest.raises(SystemExit):
        preflight_cli.cli.parser.parse_args(["--help"])
    assert "gate refusal" in capsys.readouterr().out


def test_a_command_with_only_a_summary_describes_itself_with_it(workspace, capsys):
    """`--help` for a command that has no docstring and no explicit description.

    The fallback chain is `description or __doc__ or summary or name`, and while
    the third of those was called `help` the line read that parameter. Renaming
    it to `summary` left a bare `help` there resolving to the *builtin*, which is
    truthy -- so every command in this position described itself as
    "<built-in function help>". Only reachable through `--help`, which is exactly
    the surface no other test in this file exercises per command.
    """
    cli = Cli("t", "test")

    @cli.command("bare", "the summary and nothing else")
    def _bare(_args):  # no docstring, deliberately
        return {}

    # The subparser directly, like `test_help_lists_the_exit_codes`: `run`
    # converts a SystemExit into an exit code, so `--help` never escapes it.
    with pytest.raises(SystemExit):
        cli.sub.choices["bare"].parse_args(["--help"])
    out = capsys.readouterr().out
    assert "the summary and nothing else" in out
    assert "built-in function" not in out


# ---------------------------------------------------------------------------
# the install-shape guard, at the point every tool passes through
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module",
    ["jobs", "kaggle", "modal", "gpu", "nb", "preflight", "report", "quota", "ledger"],
)
def test_every_tool_refuses_a_workspace_inside_site_packages(
    module, tmp_path, monkeypatch, capsys
):
    """The guard used to hang off `paths.ensure_workspace()`, which fifteen
    tools never call.

    The uncovered set was not a random fifteen: `jobs`, `kaggle`, `modal` and
    `gpu` are the submitters, and a run record written into `site-packages` is
    an uncollected run and real money. So it moved to `Cli.run`, which is the one
    place every tool goes through -- one call instead of fifteen, and a tool
    added tomorrow is covered on the day it is written rather than whenever
    somebody remembers.

    Parametrised over the modules rather than asserting on `Cli` directly,
    because the property is *coverage* and a unit test of the check itself
    cannot fail when a tool is missing from it."""
    from importlib import import_module

    installed = _as_installed_copy(tmp_path, monkeypatch)
    cli = import_module(f"tools.{module}").cli
    assert cli.checks_install is True
    assert cli.run(["--json"]) == EXIT_CONFIG
    out = json.loads(capsys.readouterr().out)
    assert "site-packages" in out["error"]["message"]
    assert out["error"]["fix"]
    # Refused before anything was created, which is the point of the guard.
    assert not (installed / "ledger").exists()


def test_the_tool_that_fixes_it_is_exempt(tmp_path, monkeypatch):
    """A guard that blocked its own remedy would leave `GRAD_ROOT` as the only
    way out. `grad-workspace select` rewrites the pointer, which is the fix."""
    from tools import workspace as workspace_cli

    _as_installed_copy(tmp_path, monkeypatch)
    assert workspace_cli.cli.checks_install is False
    assert workspace_cli.cli.run(["show", "--json"]) != EXIT_CONFIG


def _as_installed_copy(tmp_path, monkeypatch):
    """Make `paths.root()` resolve somewhere that looks like an installed copy."""
    from core import paths

    installed = tmp_path / "venv" / "Lib" / "site-packages" / "grad"
    installed.mkdir(parents=True)
    monkeypatch.delenv("GRAD_ROOT", raising=False)
    monkeypatch.setattr(paths, "root", lambda: installed)
    return installed
