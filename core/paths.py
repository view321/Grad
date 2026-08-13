"""Workspace layout (HANDOFF §4).

Every path in the system is derived from one root so that tests can point the
whole thing at a temp directory via GRAD_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    """Workspace root. GRAD_ROOT overrides, otherwise the repo directory."""
    env = os.environ.get("GRAD_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def _p(*parts: str) -> Path:
    return root().joinpath(*parts)


# --- ledger (source of truth, append-only) ---------------------------------
def ledger_dir() -> Path:
    return _p("ledger")


def expectations_path() -> Path:
    return _p("ledger", "expectations.jsonl")


def runs_path() -> Path:
    return _p("ledger", "runs.jsonl")


def quota_path() -> Path:
    return _p("ledger", "quota.jsonl")


def preflight_dir() -> Path:
    return _p("ledger", "preflight")


def preflight_record(submission_hash: str) -> Path:
    return preflight_dir() / f"{submission_hash}.json"


def run_artifacts(run_id: str) -> Path:
    return _p("ledger", "runs", run_id)


def ledger_sqlite() -> Path:
    return _p("ledger", "ledger.sqlite")


# --- derived / working directories -----------------------------------------
def data_dir() -> Path:
    return _p("data")


def corpus_sqlite() -> Path:
    return _p("data", "corpus.sqlite")


def papers_dir() -> Path:
    return _p("data", "papers")


def notes_dir() -> Path:
    return _p("notes")


def notebooks_dir() -> Path:
    return _p("notebooks")


def figures_dir() -> Path:
    return _p("figures")


def evals_dir() -> Path:
    return _p("evals")


def config_path() -> Path:
    env = os.environ.get("GRAD_CONFIG")
    if env:
        return Path(env).resolve()
    return _p("config", "grad.toml")


def cache_dir() -> Path:
    return _p("data", "cache")


def ensure_workspace() -> None:
    """Create the directories the CLIs write into. Cheap and idempotent."""
    for d in (
        ledger_dir(),
        preflight_dir(),
        data_dir(),
        papers_dir(),
        cache_dir(),
        notes_dir(),
        notebooks_dir(),
        figures_dir(),
        evals_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
