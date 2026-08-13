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
from core.errors import EXIT_NOT_FOUND, EXIT_OK, EXIT_USAGE, NotFound
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
