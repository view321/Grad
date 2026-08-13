"""The kernel discipline rules (HANDOFF §6).

These are integration tests against a real kernel, skipped when one is not
installed. The two behaviours worth pinning are the two the handoff singles out:
`exec` is bounded by a wall clock, and `verify` exits non-zero on the first
failing cell.
"""

from __future__ import annotations

import json

import pytest

jupyter_client = pytest.importorskip("jupyter_client")
pytest.importorskip("nbformat")
pytest.importorskip("ipykernel")

from core.errors import EXIT_CHECK_FAILED, EXIT_OK  # noqa: E402
from tools import nb as nb_cli  # noqa: E402


@pytest.fixture(autouse=True)
def kernel_cleanup():
    yield
    for name in ("default", "verify-clean", "verify-dirty"):
        nb_cli._shutdown(name)


def _notebook(path, third_cell: str) -> None:
    doc = {
        "cells": [
            {"cell_type": "code", "source": "a = 1", "metadata": {}, "outputs": [], "execution_count": None},
            {"cell_type": "markdown", "source": "prose", "metadata": {}},
            {"cell_type": "code", "source": third_cell, "metadata": {}, "outputs": [], "execution_count": None},
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.mark.slow
def test_exec_runs_and_returns_stdout(workspace, capsys):
    assert nb_cli.cli.run(["exec", "--code", "print(6*7)", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["data"]["stdout"].strip() == "42"


@pytest.mark.slow
def test_kernel_state_persists_between_invocations(workspace, capsys):
    """The kernel is persistent across CLI invocations -- that is the whole
    reason it is spawned detached rather than owned by a KernelManager."""
    assert nb_cli.cli.run(["exec", "--code", "carried = 17", "--json"]) == EXIT_OK
    capsys.readouterr()
    assert nb_cli.cli.run(["exec", "--code", "print(carried)", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["data"]["stdout"].strip() == "17"


@pytest.mark.slow
def test_exec_timeout_says_to_move_the_work_to_a_job(workspace, capsys):
    """'a training loop in a cell blocks it indefinitely with no way to observe
    progress.'"""
    code = nb_cli.cli.run(["exec", "--code", "import time; time.sleep(30)", "--timeout", "2", "--json"])
    assert code == EXIT_CHECK_FAILED
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["error"]["code"] == "kernel_timeout"
    assert "tools.jobs submit" in payload["error"]["fix"]


@pytest.mark.slow
def test_verify_passes_a_clean_notebook(workspace, capsys):
    path = workspace / "notebooks" / "clean.ipynb"
    _notebook(path, "print('a is', a)")
    assert nb_cli.cli.run(["verify", str(path), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["data"]["cells_executed"] == 2


@pytest.mark.slow
def test_verify_fails_on_the_first_bad_cell(workspace, capsys):
    """A notebook that only works in the kernel that grew it is not evidence."""
    path = workspace / "notebooks" / "dirty.ipynb"
    _notebook(path, "print(undefined_name)")
    assert nb_cli.cli.run(["verify", str(path), "--json"]) == EXIT_CHECK_FAILED
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["error"]["detail"]["cell_index"] == 2
