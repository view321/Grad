"""grad-project -- the per-project memory directory.

Six files in `projects/<id>/`. Three are yours (`MEMORY.md`, `PLAN.md`,
`TODO.md`) and three are rendered from the ledger (`EXPECTATIONS.md`,
`RESULTS.md`, `DONE.md`). See `core/projects.py` for why the split is there.

`MEMORY.md` is loaded into the agent's system prompt at the start of every
session, and `memory` below prints exactly what that will be -- including the
truncation, so "why does it not know that" has an answer you can read rather
than infer.
"""

from __future__ import annotations

import argparse
from typing import Any

from core import budget, paths, projects
from core.cli import Cli, main
from core.errors import NotFound

cli = Cli(
    "grad-project",
    "The project's memory: conventions and plans you write, results rendered from the ledger.",
    epilog=(
        "MEMORY.md is the one file loaded into the agent's context every session.\n"
        "Keep it factual and keep it short; `memory` shows what the agent actually gets.\n\n"
        "EXPECTATIONS.md, RESULTS.md and DONE.md are generated. Editing them is\n"
        "refused rather than silently overwritten, but the right home for a sentence\n"
        "about a result is MEMORY.md, not the file the ledger renders."
    ),
)


def _project_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="project id (defaults to the current selection)")


def _resolve(args: argparse.Namespace) -> str:
    """The project this command acts on.

    `budget.resolve` rather than `current_project` alone, because it is the one
    that refuses an id that does not exist -- and a memory directory scaffolded
    under a typo is a directory that silently accumulates notes nothing will ever
    load into a session.
    """
    project_id = budget.resolve(getattr(args, "project", None))
    if not project_id:
        raise NotFound(
            "no project is selected, and project memory is per project",
            fix="python -m tools.budget use <id> --json   # or pass --project <id>",
        )
    return project_id


# ---------------------------------------------------------------------------
def _init_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite a generated file that has been edited by hand",
    )


@cli.command("init", "create the memory directory and render the ledger views", setup=_init_args)
def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    """Scaffold the three authored files and generate the three derived ones.

    Idempotent, and safe on a project that already has notes: `scaffold` never
    overwrites an authored file, so this is also the command that adds a file
    introduced by a later release to a project that predates it.

    `--force` forwards to `sync`, and it is what makes the sentence above true
    for a project whose generated files have been edited: without it the sync
    half refuses, *after* the scaffold half has already written -- so a command
    documented as idempotent failed on its second run, having done part of its
    work, and named a `sync --force` the caller then had to run by hand.
    """
    project_id = _resolve(args)
    paths.ensure_workspace()
    created = projects.scaffold(project_id)
    synced = projects.sync(project_id, force=args.force)
    return {
        **created,
        "generated": synced["written"],
        "next": f"python -m tools.project memory --project {project_id} --json",
    }


def _sync_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite a generated file that has been edited by hand",
    )


@cli.command("sync", "re-render EXPECTATIONS.md, RESULTS.md and DONE.md", setup=_sync_args)
def cmd_sync(args: argparse.Namespace) -> dict[str, Any]:
    """Regenerate the derived files from the ledger. Free, and no model is used."""
    project_id = _resolve(args)
    return projects.sync(project_id, force=args.force)


@cli.command("show", "what the ledger says about this project", setup=_project_arg)
def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    """The fold the generated files are rendered from, as data.

    The same numbers as `RESULTS.md` and `DONE.md`, for a caller that would
    rather parse than read -- the UI's project window is the first.
    """
    project_id = _resolve(args)
    state = projects.state(project_id)
    directory = projects.resolve_dir(project_id)
    return {
        "project": project_id,
        "dir": str(directory),
        "exists": directory.is_dir(),
        "documents": {
            name: {"path": str(directory / name), "present": (directory / name).is_file()}
            for name in projects.DOCS
        },
        "expectations": len(state["expectations"]),
        # `.get` on both sides. The filter already tolerated a record with no
        # id and the projection did not, so the one shape that reached here --
        # an expectation the ledger wrote without one -- passed the test and
        # then raised `KeyError` building the answer. Same accessor as
        # `core/projects.py:_render_expectations`.
        "open_expectations": [
            e.get("id") for e in state["expectations"]
            if e.get("id") not in state["bound_to"] and e.get("id") not in state["falsified"]
        ],
        "runs": len(state["runs"]),
        "collected": len(state["collected"]),
        "in_flight": [r.id for r in state["in_flight"]],
        "done": [r.id for r in state["done"]],
        "awaiting_verdict": [r.id for r in state["awaiting_verdict"]],
    }


def _memory_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    p.add_argument(
        "--raw",
        action="store_true",
        help="the file's own text, rather than the block the system prompt receives",
    )


@cli.command("memory", "print what the agent is given about this project", setup=_memory_args)
def cmd_memory(args: argparse.Namespace) -> dict[str, Any]:
    """Exactly what `agent.system_prompt` appends, truncation included.

    Worth having as its own command: memory that is silently truncated, or
    silently absent because the project directory was never created, fails in a
    way that looks like a model that did not read carefully.
    """
    project_id = _resolve(args)
    path = projects.doc_path(project_id, "MEMORY.md")
    raw = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = projects.prompt_block(project_id)
    return {
        "project": project_id,
        "path": str(path),
        "present": path.is_file(),
        "chars": len(raw),
        # Against the same text `memory_text` measures -- stripped -- because
        # this flag is a claim about the block in `text` below. Measured on the
        # raw file, a memory ending in a run of blank lines reported itself
        # truncated while the block it returned was whole.
        "truncated": len(raw.strip()) > projects.MEMORY_MAX_CHARS,
        "limit_chars": projects.MEMORY_MAX_CHARS,
        "text": raw if args.raw else block,
    }


@cli.command("path", "print the project's memory directory", setup=_project_arg)
def cmd_path(args: argparse.Namespace) -> str:
    return str(projects.resolve_dir(_resolve(args)))


if __name__ == "__main__":
    main(cli)
