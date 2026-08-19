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


def credential_log_path() -> Path:
    """Which credentials were read, when, and by what. Never the values.

    Detection rather than prevention -- see `core/credentials.py`, which explains
    why prevention is not on the table for a process that has to be able to
    authenticate.
    """
    return _p("ledger", "credential_reads.jsonl")


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


# --- per-project memory (see core/projects.py) -----------------------------
def projects_dir() -> Path:
    return _p("projects")


def project_dir(project_id: str) -> Path:
    """One project's artifact directory.

    Named by the project id, which `core/budget.py` already constrains to
    something id-shaped -- so this is not a place a caller can escape from with
    a `../`. Checked anyway in `core/projects.py:resolve_dir`, because the id
    reaches that function from a `--project` flag and a path built from
    unvalidated input is the one bug in this file that would matter.
    """
    return projects_dir() / project_id


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


#: Directory names that mean "this is an installed copy of the code, not a
#: checkout". Both spellings, because Debian renames the first.
_INSTALL_DIRS = frozenset({"site-packages", "dist-packages"})


def check_not_installed_copy() -> None:
    """Refuse to treat an installed copy of the code as the workspace.

    `root()`'s last resort is the directory the code sits in, which is right for
    a checkout and wrong for everything else. `README.md` documents
    `pip install -e`, and an install that records itself as non-editable puts the
    code in `site-packages` -- so the fallback resolves there, and every ledger
    write, every note and every figure lands inside the installed package. It
    fails at nothing. It just writes the research somewhere no one will look,
    somewhere the next `pip install` overwrites, and the only symptom is a
    ledger that keeps coming up empty.

    That happened, and it took a while to see, because "which copy of the code is
    live" is invisible from inside the code -- the same class as a stale
    `build/lib`. `direct_url.json` records the answer and nobody reads it, so
    this asks the question directly at the one moment it is cheap to answer.

    **`GRAD_ROOT` is the only exemption, and a remembered choice is not one.**
    An environment variable is typed for this process and this run, so it is an
    answer somebody is giving right now; the pointer file is a decision from some
    earlier session that may well have been made *by* the bug -- a first run
    under a non-editable install would remember the install directory, and
    honouring that would make the guard agree with the state it exists to catch.
    The cost is that `grad-workspace` has to be exempt as a tool, since it is
    what rewrites the pointer.

    Called from `core/cli.py` for every tool rather than from
    `ensure_workspace()`, which fifteen tools never call -- among them every
    submitter, whose ledger writes are the most expensive in the system to lose.
    """
    if os.environ.get("GRAD_ROOT"):
        return
    resolved = root()
    if not _INSTALL_DIRS.intersection(resolved.parts):
        return
    from core.errors import ConfigError  # noqa: PLC0415 - only on the refusal path

    raise ConfigError(
        f"the workspace resolved to {resolved}, which is inside an installed "
        "package rather than a research directory. Every ledger write would go "
        "there and be lost on the next install",
        fix=(
            "reinstall editable from the checkout (`python -m pip install -e .`), "
            "or set GRAD_ROOT to the directory the research should live in"
        ),
    )


def ensure_workspace() -> None:
    """Create the directories the CLIs write into. Cheap and idempotent.

    `cache_dir` is absent on purpose: it lives under the app directory now, and
    `appdata.ensure` is what creates that side.

    The check comes first, because the failure it catches is one where every
    directory below is created successfully in a place that is wrong.
    """
    check_not_installed_copy()
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
