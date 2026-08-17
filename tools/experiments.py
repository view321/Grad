"""grad-experiments -- the cross-workspace archive of everything that was run.

Every other ledger here is scoped to one workspace. This one is global to the
user, and it exists to answer the question that scoping makes unanswerable:
*have I run this before, and what did it give me?*

Runs enter it automatically -- `collect` and `abandon` both go through
`core/submit.py:finish`, and `ledger verdict` re-files a run once it has been
judged. `archive` below is for backfilling a ledger that predates this.

The archive is a copy. Where it and a workspace's `runs.jsonl` disagree, the
workspace wins; `verify` is how you find out that they do.
"""

from __future__ import annotations

import argparse
from typing import Any

from core import experiments, ledger_store as ls, paths
from core.cli import Cli, main
from core.errors import EXIT_CHECK_FAILED, GradError, UsageError

cli = Cli(
    "grad-experiments",
    "Every experiment ever run, across every workspace and project.",
    epilog=(
        "Runs are archived automatically when they become terminal. This CLI is for\n"
        "reading that archive, for backfilling one that predates it, and for checking\n"
        "that the artifacts it points at are still the ones it hashed.\n\n"
        "It is a copy, not a source of truth. A workspace's ledger is authoritative for\n"
        "its own runs, and `verify` reports where the two have drifted apart."
    ),
)


def _list_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="only this project")
    p.add_argument("--workspace", help="only this workspace (path fragment or key)")
    p.add_argument("--task", help="only this task id")
    p.add_argument("--quantity", help="only experiments that reported this metric")
    p.add_argument("--no-smoke", action="store_true", help="exclude smoke runs")
    p.add_argument("--judged", action="store_true", help="only fully judged experiments")
    p.add_argument("--limit", type=int, default=50, help="how many to return (default 50)")


@cli.command("list", "what has been run, newest first", setup=_list_args)
def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    """Filtered rows, one per experiment.

    Deliberately thin: ids, task, status, cost and the metric names, but not the
    metric values or the spec. A listing that inlined every result would be
    unreadable at fifty rows and would be the wrong thing to hand a model, which
    should pick a row and ask for it.
    """
    rows = experiments.search(
        project=args.project,
        workspace=args.workspace,
        task=args.task,
        quantity=args.quantity,
        include_smoke=not args.no_smoke,
        judged_only=args.judged,
    )
    limit = max(1, int(args.limit))
    return {
        "matched": len(rows),
        "shown": min(len(rows), limit),
        "experiments": [
            {
                "experiment_id": r.get("experiment_id"),
                "run_id": r.get("run_id"),
                "project": r.get("project"),
                "workspace": r.get("workspace"),
                "task": r.get("task"),
                "platform": r.get("platform"),
                "status": r.get("status"),
                "smoke": r.get("smoke"),
                "submitted_at": r.get("submitted_at"),
                "cost_usd_actual": r.get("cost_usd_actual"),
                "accelerator_hours": r.get("accelerator_hours_actual"),
                "all_judged": r.get("all_judged"),
                "quantities": sorted((r.get("results") or {})),
            }
            for r in rows[:limit]
        ],
    }


def _show_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("identifier", help="an experiment id, or a run id if it is unambiguous")


@cli.command("show", "one experiment in full", setup=_show_args)
def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    """Everything archived for one experiment, including the resolved spec.

    This is the record that makes a result reproducible without the workspace:
    the spec that was submitted, the prediction it was bound to, what it
    returned, how that deviated, and what was concluded.
    """
    return experiments.get(args.identifier)


def _archive_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id", nargs="*", help="run ids; omit for every terminal run in this workspace")
    p.add_argument(
        "--all",
        action="store_true",
        help="archive every collected or abandoned run in this workspace",
    )


@cli.command("archive", "copy runs from this workspace into the archive", setup=_archive_args)
def cmd_archive(args: argparse.Namespace) -> dict[str, Any]:
    """Backfill. Runs archive themselves when they become terminal.

    Only terminal runs, and that restriction is the point: an in-flight run has
    no result, no cost and no artifacts, and archiving one would put a row in the
    global store that says an experiment happened when what happened is that one
    was started.
    """
    paths.ensure_workspace()
    if not args.run_id and not args.all:
        raise UsageError(
            "name at least one run id, or pass --all to archive every terminal run here",
            fix="python -m tools.experiments archive --all --json",
        )
    if args.run_id:
        wanted = [ls.run(rid) for rid in args.run_id]
    else:
        wanted = [r for r in ls.runs() if r.collected]

    archived, skipped = [], []
    for run in wanted:
        if not run.collected:
            skipped.append({"run_id": run.id, "why": "still in flight; nothing to archive yet"})
            continue
        record = experiments.archive(run.id)
        archived.append(record["experiment_id"])
    return {
        "archived": archived,
        "skipped": skipped,
        "archive": str(experiments.archive_path()),
    }


@cli.command("summary", "what the archive holds")
def cmd_summary(_: argparse.Namespace) -> dict[str, Any]:
    return experiments.summary()


def _verify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("identifier", nargs="?", help="one experiment; omit to check every one")


@cli.command("verify", "re-hash artifacts and re-derive submission hashes", setup=_verify_args)
def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    """Exit 9 when something has drifted, 0 when nothing has.

    Two things are checked and they mean different things. An artifact whose
    digest has changed means a figure or a metrics file is no longer what was
    measured. A resolved spec that does not hash to its recorded submission hash
    means the archive's own copy of what was submitted is wrong, which is the one
    failure this store can detect without anything else being present.
    """
    result = experiments.verify(args.identifier)
    if not result["ok"]:
        raise GradError(
            "archive_drift",
            f"{len(result['findings'])} finding(s): artifacts or specs no longer match what "
            "was archived",
            exit_code=EXIT_CHECK_FAILED,
            fix=(
                "read the findings; an artifact_changed is a file edited since the run, and a "
                "spec_hash_mismatch means the archived spec is not the one that was submitted"
            ),
            detail=result,
        )
    return result


@cli.command("reindex", "rebuild the derived SQLite index from the JSONL")
def cmd_reindex(_: argparse.Namespace) -> dict[str, Any]:
    """The JSONL is the source of truth; this is a view over it, always safe to
    rebuild."""
    counts = experiments.rebuild_index()
    return {**counts, "index": str(experiments.index_path())}


if __name__ == "__main__":
    main(cli)
