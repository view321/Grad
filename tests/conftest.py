from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Point the whole system at a temp directory.

    Every path is derived from GRAD_ROOT precisely so the gates can be tested
    against a real ledger rather than against mocks -- these are the checks that
    stand between the agent and a $40 mistake, and a mock of a gate proves
    nothing about the gate.
    """
    monkeypatch.setenv("GRAD_ROOT", str(tmp_path))
    monkeypatch.setenv("GRAD_CONFIG", str(tmp_path / "config" / "grad.toml"))
    from core import config, paths

    config._cache.clear()
    paths.ensure_workspace()
    yield tmp_path
    config._cache.clear()


@pytest.fixture(autouse=True)
def clean_process_state():
    """Module-level registries outlive a fixture, so they are emptied around
    every test. Both exist for the same reason -- a task and a session are
    process-wide facts rather than per-client ones -- and both would otherwise
    let one test decide another's outcome."""
    from ui import sessions, tasks

    tasks.reset()
    sessions.reset_claims()
    yield
    tasks.reset()
    sessions.reset_claims()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every HTTP client in this project goes through `core.http._httpx`, so
    replacing it is enough to make the suite's "no network" claim mechanical.

    It was not, and the failure mode is why this exists rather than being left
    to discipline. Changing the funnel's default tier-1 client from Semantic
    Scholar to Asta left two tests monkeypatching the class that was no longer
    constructed -- so instead of failing they began POSTing to a real endpoint,
    with a 60-second timeout and a rate limiter between calls. A suite that
    reaches the network does not fail; it *hangs*, which is the one outcome that
    does not point at its own cause.

    A test that wants a fake sets `_httpx` itself; monkeypatch applies in order,
    so its patch replaces this one for the duration.
    """
    from core import http

    def refuse() -> Any:
        raise AssertionError(
            "a test reached for the network. Fake the client it uses "
            "(monkeypatch core.http._httpx, or the class on core.http) -- the suite "
            "runs with no network by design, and a real call hangs rather than fails."
        )

    monkeypatch.setattr(http, "_httpx", refuse)


@pytest.fixture
def cfg():
    from core import config

    return config.load(reload=True)
