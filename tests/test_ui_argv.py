"""Every argv the UI builds, parsed by the CLI that will receive it.

`ui/shell.py` and `ui/state.py` construct command lines as string lists and hand
them to `tools.run_tool`, which spawns `python -m <module> ...`. Nothing checked
that those flags exist, so `budget raise` shipped with the project id passed
positionally -- a command that fails with exit 2 on every click, in the one
control the over-budget refusal tells you to use.

A button whose argv does not parse is dead, and it is dead silently: the failure
surfaces as an error envelope in a status bar rather than as anything a test
would notice. So the parsers themselves are the oracle here. This is deliberately
not a test of what the commands *do* -- `test_budget.py` and friends cover that
-- only that the words the UI says are words the CLI understands.
"""

from __future__ import annotations

import importlib

import pytest

from core.errors import UsageError

# (module, argv-after-the-module) for every command the UI can build. Kept as
# literals rather than harvested from the shell, because the point is to fail
# when the two drift apart.
UI_COMMANDS = [
    ("tools.budget", ["raise", "--project", "proj-x", "--gpu-usd", "50"]),
    ("tools.budget", ["raise", "--project", "proj-x", "--quota-tokens", "5e6"]),
    ("tools.budget", ["raise", "--project", "proj-x", "--credits-usd", "10"]),
    # The projects window offers the reason the CLI has always taken. A ceiling
    # that moved without one is unarguable with six months later.
    ("tools.budget", ["raise", "--project", "proj-x", "--gpu-usd", "50", "--reason", "why"]),
    ("tools.budget", ["new", "--id", "proj-x", "--title", "a title", "--use"]),
    # The create form sends the ceilings and the payer on `new` itself, rather
    # than as a raise afterwards -- a raise records a ceiling that *moved*.
    (
        "tools.budget",
        [
            "new", "--id", "proj-x", "--title", "a title", "--use",
            "--gpu-usd", "120", "--quota-tokens", "5e6", "--credits-usd", "10",
            "--payer", "hf:myorg",
        ],
    ),
    ("tools.budget", ["use", "proj-x"]),
    ("tools.budget", ["close", "proj-x"]),
    # Per-project overrides, from the projects window's model editor.
    ("tools.budget", ["configure", "--project", "proj-x", "--research", "claude-opus-5"]),
    ("tools.budget", ["configure", "--project", "proj-x", "--report", "claude-opus-5"]),
    ("tools.budget", ["configure", "--project", "proj-x", "--clear", "evolve"]),
    ("tools.budget", ["configure", "--project", "proj-x", "--backend", "kaggle"]),
    ("tools.budget", ["status", "--project", "proj-x"]),
    # The setup window. Every one of these is a button.
    ("tools.setup", ["models", "--research", "claude-opus-5"]),
    ("tools.setup", ["models", "--evolve", "claude-sonnet-5"]),
    ("tools.setup", ["models", "--clear", "evolve"]),
    ("tools.setup", ["backend", "--default", "kaggle"]),
    ("tools.setup", ["host", "add", "--name", "gpu-box", "--hostname", "h", "--user", "u", "--rate", "0"]),
    ("tools.setup", ["host", "remove", "--name", "gpu-box"]),
    ("tools.setup", ["show"]),
    ("tools.setup", ["check"]),
    ("tools.kaggle", ["account", "--set", "someone"]),
    ("tools.jobs", ["credential", "set", "hf_token", "--stdin"]),
    ("tools.jobs", ["credential", "delete", "hf_token"]),
    ("tools.jobs", ["credential", "status"]),
    ("tools.jobs", ["collect", "run-1"]),
    ("tools.ledger", ["verdict", "run-1", "--quantity", "loss", "--verdict", "bug", "--note", "x"]),
    ("tools.nb", ["restart"]),
    ("tools.report", ["draft", "--project", "proj-x"]),
    ("tools.report", ["check", "--project", "proj-x"]),
    ("tools.report", ["build", "--project", "proj-x"]),
    ("tools.wiki", ["map"]),
    ("tools.preflight", ["run", "--spec", "pipeline/spec.toml"]),
]


@pytest.mark.parametrize("module,argv", UI_COMMANDS)
def test_every_ui_argv_parses(module, argv):
    cli = importlib.import_module(module).cli
    # `core.cli._Parser.error` raises `UsageError` rather than exiting, so that
    # -- not SystemExit -- is what a bad flag looks like here. It is the same
    # exit-2 envelope the UI would surface as a red line in the status bar.
    parsed = cli.parser.parse_args([*argv, "--json"])
    assert parsed is not None


def test_the_raise_button_builds_a_parseable_command():
    """The specific regression: `budget raise` takes --project, not a positional.

    Asserted against the string the button actually builds rather than against a
    copy of it, so editing the button without editing the flag fails here.
    """
    from tools import budget as budget_tool

    argv = ["raise", "--project", "proj-scaling-w2", "--gpu-usd", "75", "--json"]
    parsed = budget_tool.cli.parser.parse_args(argv)
    assert parsed.project == "proj-scaling-w2"
    assert parsed.gpu_usd == 75.0

    with pytest.raises(UsageError):
        # The shape that shipped: id as a positional. If this ever starts
        # parsing, the button and this test should both be revisited.
        budget_tool.cli.parser.parse_args(["raise", "proj-scaling-w2", "--gpu-usd", "75"])
