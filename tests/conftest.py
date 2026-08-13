from __future__ import annotations

import os
import sys
from pathlib import Path

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


@pytest.fixture
def cfg():
    from core import config

    return config.load(reload=True)
