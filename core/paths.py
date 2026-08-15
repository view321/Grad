"""Workspace layout (HANDOFF §4).

Every path in the system is derived from one root so that tests can point the
whole thing at a temp directory via GRAD_ROOT.

**Three entries are not research and do not move with the root.** `prompts/`,
`skills/` and `config/grad.toml` ship *with the code*: they are tracked in the
repository, an update replaces them, and they describe how Grad behaves rather
than what it found. Resolving them against the workspace alone was the one thing
standing between this layout and a workspace that lives outside the
installation -- point the root at an empty folder and `agent.py` reads its
system prompt from a file that is not there.

So those three resolve **workspace first, installation second**. A file in the
workspace wins, which is what makes a per-workspace system prompt or a
per-workspace ceiling possible at all; absent one, the shipped copy is used. The
research directories below have no such fallback, and must not: silently reading
the installation's ledger because the workspace has none is how a report ends up
citing runs from someone else's project.
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    """Workspace root: GRAD_ROOT, then the remembered choice, then the repo.

    The middle rule is what makes the app's folder chooser survive a restart;
    `core/workspace.py` holds the pointer and explains why it is stored beside
    the code rather than inside the workspace it names. GRAD_ROOT still wins, so
    an explicit override -- the test suite's, or one typed on a command line --
    is never quietly beaten by a remembered one.
    """
    env = os.environ.get("GRAD_ROOT")
    if env:
        return Path(env).resolve()
    # Imported here rather than at module scope: `workspace` raises the CLI's
    # error type, and this module is imported by almost everything.
    from core import workspace  # noqa: PLC0415

    chosen = workspace.remembered()
    if chosen is not None:
        return chosen
    return Path(__file__).resolve().parent.parent


def install_dir() -> Path:
    """Where Grad itself is installed, which is not always where the research is.

    Imported lazily for the same reason `root()` does it: `core/workspace.py`
    raises the CLI's error type, and this module is imported by almost
    everything.
    """
    from core import workspace  # noqa: PLC0415

    return workspace.code_dir()


def _p(*parts: str) -> Path:
    return root().joinpath(*parts)


def _shipped(*parts: str) -> Path:
    """Workspace copy if there is one, the installed copy otherwise.

    Falls back to the *workspace* path when neither exists, because that is the
    answer to the question the caller is really asking in that case -- an error
    message saying "create it here" should name the folder the user is working
    in, not the one Grad happens to be installed in.
    """
    local = root().joinpath(*parts)
    if local.exists():
        return local
    shipped = install_dir().joinpath(*parts)
    return shipped if shipped.exists() else local


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


# --- shipped with the code, overridable per workspace ----------------------
def prompt_path() -> Path:
    """The system prompt. See the note at the top about the three shipped paths."""
    return _shipped("prompts", "system.md")


def skills_dir() -> Path:
    """The skills the system prompt tells the agent to reach for."""
    return _shipped("skills")


def config_path() -> Path:
    env = os.environ.get("GRAD_CONFIG")
    if env:
        return Path(env).resolve()
    return _shipped("config", "grad.toml")


def cache_dir() -> Path:
    """Regenerable downloads, which are an installation's business rather than a
    workspace's -- so this is the one path here that resolves outside the root.
    See `core/appdata.py` for the split."""
    from core import appdata  # noqa: PLC0415

    return appdata.cache_dir()


def ensure_workspace() -> None:
    """Create the directories the CLIs write into. Cheap and idempotent.

    `cache_dir` is absent on purpose: it lives under the app directory now, and
    `appdata.ensure` is what creates that side.
    """
    for d in (
        ledger_dir(),
        preflight_dir(),
        data_dir(),
        papers_dir(),
        notes_dir(),
        notebooks_dir(),
        figures_dir(),
        evals_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
