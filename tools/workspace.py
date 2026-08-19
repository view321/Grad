"""grad-workspace -- which folder holds the research, and how to move it out.

The workspace defaults to the installation folder, which is the simplest thing
that works and the reason `grad update` needs a story at all: research committed
into the same checkout as the code puts a user's notebooks and ledger on the
same branch as upstream's releases, and every update becomes a merge.

`move` is the way out of that, and it is deliberately not a `mv`. It **copies**,
verifies the copy, and leaves the originals where they are unless asked twice.
The files being moved are the only record of someone's experiments; a tool that
deletes them because a copy *appeared* to succeed is trading an afternoon of
inconvenience against a category of loss that cannot be undone. The follow-up
command is printed rather than run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from core import paths, version, workspace as workspace_mod
from core.cli import Cli, main
from core.errors import UsageError

cli = Cli(
    "grad-workspace",
    "Show, choose, or relocate the folder that holds your research.",
    epilog=(
        "The workspace holds the ledger, notebooks, notes, figures and reports.\n"
        "The installation holds the code, the prompts and the skills. They may be the\n"
        "same folder -- that is the default -- but keeping them apart is what makes\n"
        "`grad update` a fast-forward rather than a merge with your own research."
    ),
    # The one tool exempt from the install-shape guard, because it is the cure.
    # A workspace that has resolved into `site-packages` is fixed by pointing it
    # somewhere else, and `select` is what does that -- so refusing here would
    # leave `GRAD_ROOT` as the only way out of a problem this command exists to
    # solve.
    checks_install=False,
)

#: What `move` relocates: the research directories, in the order they are
#: reported. `.grad-workspace.json` is excluded on purpose -- it is the pointer
#: *to* a workspace and lives beside the code, so copying it into the
#: destination would leave a stale pointer inside the folder it names.
MOVABLE = tuple(p for p in version.WORKSPACE_PATHS if not p.startswith("."))

#: Regenerable or machine-local, and skipped. `data/cache`, `data/layouts` and
#: `data/kernel` have already moved to the app directory on any installation
#: that has started once (`core/appdata.py`); `data/lab` is a live port and
#: token; `data/wiki` is generated output. Copying them would move state that is
#: keyed to one machine into a folder meant to be portable.
SKIP = ("data/cache", "data/layouts", "data/kernel", "data/lab", "data/wiki", "data/papers")


@cli.command("path", "which folder is in use, and which rule chose it")
def cmd_path(_: argparse.Namespace) -> dict[str, Any]:
    root = paths.root()
    install = paths.install_dir()
    return {
        "workspace": str(root),
        "installation": str(install),
        "chosen_by": workspace_mod.source(),
        "separate": root != install,
        "recent": [str(p) for p in workspace_mod.recent()],
        "message": (
            f"workspace {root}"
            + ("" if root != install else "  (inside the installation — see: grad-workspace move --help)")
        ),
    }


@cli.command(
    "use",
    "point Grad at another folder",
    setup=lambda p: (
        p.add_argument("folder", help="the folder to use as the workspace"),
        p.add_argument("--create", action="store_true", help="create it if it does not exist"),
    ),
)
def cmd_use(args: argparse.Namespace) -> dict[str, Any]:
    """Remember a folder as the workspace.

    Only the pointer changes: nothing is copied and nothing is deleted, so an
    empty folder gives an empty workspace rather than an error. The scaffolding
    happens on first use -- `paths.ensure_workspace` creates the directories the
    CLIs write into.
    """
    chosen = workspace_mod.select(args.folder, create=args.create)
    paths.ensure_workspace()
    return {
        "workspace": str(chosen),
        "installation": str(paths.install_dir()),
        "message": f"workspace is now {chosen}",
    }


def _move_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", help="where the research should live from now on")
    p.add_argument(
        "--dry-run", action="store_true", help="report what would be copied and stop"
    )
    p.add_argument(
        "--remove-originals",
        action="store_true",
        help="delete the originals once every file has been verified in the destination",
    )
    p.add_argument(
        "--keep-pointer",
        action="store_true",
        help="copy, but do not switch this Grad over to the new folder",
    )


@cli.command("move", "copy the research out of the installation folder", setup=_move_args)
def cmd_move(args: argparse.Namespace) -> dict[str, Any]:
    """Copy ledger, notebooks, notes, figures, reports and evals to a new folder.

    Refuses a destination that already holds a ledger. Merging two ledgers is
    not a file operation -- both are append-only logs of events that really
    happened, and interleaving them by copy order would produce a history that
    is not either one's.
    """
    source = paths.root()
    target = workspace_mod.validate(args.folder, create=not args.dry_run)
    if target == source:
        raise UsageError(
            "the destination is the folder the research is already in",
            fix="choose a folder outside the installation, e.g. ~/Grad",
        )
    # A destination *inside* the source is the one that does real damage rather
    # than merely refusing: copying `notebooks/` into `notebooks/new/notebooks/`
    # descends into the copy it is making and fills the disk. Checked on the
    # resolved paths, because `validate` has already resolved both and a
    # relative `./sub` reaches here as an absolute path.
    if source in target.parents:
        raise UsageError(
            f"{target} is inside {source}",
            fix="choose a folder outside the workspace; copying a folder into itself does not end",
        )

    planned = _plan_copy(source, target)
    # Refuse to merge, rather than refusing only when a ledger is in the way.
    # `_copy_entry` passes `dirs_exist_ok=True` -- it has to, since it copies
    # into a folder it may have just created -- so an existing `notebooks/` at
    # the destination would be silently interleaved with this one, and the
    # result would be a folder whose history is neither workspace's.
    occupied = sorted(entry["name"] for entry in planned if (target / entry["name"]).exists())
    if occupied:
        raise UsageError(
            f"{target} already has: {', '.join(occupied)}",
            fix="choose an empty folder; two workspaces cannot be merged by copying",
        )
    total = sum(entry["files"] for entry in planned)
    if args.dry_run:
        return {
            "from": str(source),
            "to": str(target),
            "entries": planned,
            "files": total,
            "message": f"would copy {total} file(s) in {len(planned)} entr(ies) to {target}",
        }

    for entry in planned:
        entry["copied"] = _copy_entry(source, target, entry["name"])

    # Verified against everything that was *planned*, not against what `copy`
    # claimed to do. A directory whose copy raised is exactly the case that
    # matters, and checking only the successful ones would drop its files out of
    # `missing` entirely -- reporting a complete move whose largest entry never
    # arrived.
    verified, missing = _verify(source, target, [entry["name"] for entry in planned])

    removed: list[str] = []
    if missing and args.remove_originals:
        raise UsageError(
            f"{len(missing)} file(s) did not arrive; nothing was deleted",
            fix="re-run without --remove-originals and compare the two folders by hand",
        )
    if args.remove_originals:
        removed = _remove(source, [entry["name"] for entry in planned])

    # And the pointer moves only for a copy that is whole. Switching to a
    # workspace that is missing files would leave the app reading a partial
    # ledger while the complete one sits in the folder it just left -- with
    # nothing on screen to say which is which.
    moved_pointer = None
    if not args.keep_pointer and not missing:
        moved_pointer = str(workspace_mod.select(target))
        paths.ensure_workspace()

    return {
        "from": str(source),
        "to": str(target),
        "entries": planned,
        "files_copied": verified,
        "missing": missing[:20],
        "removed": removed,
        "workspace": moved_pointer,
        "message": _move_message(target, verified, removed, missing, args),
    }


def _move_message(
    target: Path,
    verified: int,
    removed: list[str],
    missing: list[str],
    args: argparse.Namespace,
) -> str:
    head = f"copied {verified} file(s) to {target}"
    if missing:
        return (
            f"{head}, but {len(missing)} did not arrive — the workspace was NOT switched and "
            f"nothing was removed. First missing: {missing[0]}"
        )
    if removed:
        return f"{head}; originals removed"
    if args.remove_originals:
        return head
    return (
        f"{head}. The originals are untouched — check the new folder, then remove them with: "
        f"python -m tools.workspace move {target} --remove-originals"
    )


def _plan_copy(source: Path, target: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in MOVABLE:
        path = source / name
        if not path.exists():
            continue
        files = [p for p in _walk(path) if not _skipped(source, p)]
        if not files and not path.is_dir():
            continue
        out.append(
            {
                "name": name,
                "files": len(files),
                "bytes": sum(_size(p) for p in files),
                "to": str(target / name),
            }
        )
    return out


def _walk(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.is_file()]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _skipped(source: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(source).as_posix()
    except ValueError:
        return False
    return any(rel == skip or rel.startswith(f"{skip}/") for skip in SKIP)


def _copy_entry(source: Path, target: Path, name: str) -> bool:
    src = source / name
    dst = target / name
    try:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=lambda directory, names: [
                n for n in names if _skipped(source, Path(directory) / n)
            ],
        )
        return True
    except OSError:
        return False


def _verify(source: Path, target: Path, names: list[str]) -> tuple[int, list[str]]:
    """Count what arrived, and name what did not.

    By size and existence rather than by hash: this runs over a corpus that can
    be gigabytes, and the failure it has to catch -- a file that did not copy at
    all, or copied short because the disk filled -- is visible in both.
    """
    arrived, missing = 0, []
    for name in names:
        for path in _walk(source / name):
            if _skipped(source, path):
                continue
            rel = path.relative_to(source)
            other = target / rel
            if other.exists() and _size(other) == _size(path):
                arrived += 1
            else:
                missing.append(rel.as_posix())
    return arrived, missing


def _remove(source: Path, names: list[str]) -> list[str]:
    removed: list[str] = []
    for name in names:
        path = source / name
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            removed.append(name)
        except OSError:
            # Held open, or read-only. The copy is verified by the time this
            # runs, so leaving it is a tidying problem rather than a loss.
            continue
    return removed


# ---------------------------------------------------------------------------
# version control
# ---------------------------------------------------------------------------
def _vcs_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "action",
        choices=("init", "status", "log", "commit"),
        help="init: start versioning this workspace; commit: checkpoint now",
    )
    p.add_argument("--message", default="manual checkpoint", help="the commit subject")
    p.add_argument("--limit", type=int, default=20, help="how many checkpoints `log` shows")


@cli.command("vcs", "keep a local history of the research", setup=_vcs_args)
def cmd_vcs(args: argparse.Namespace) -> dict[str, Any]:
    """Version the workspace, locally, with no remote.

    `init` is deliberate and one-time: creating a repository inside somebody's
    folder is a side effect they did not ask for. Everything after it is
    automatic -- a run collected, a verdict recorded and a project's documents
    regenerated each leave a commit, because those are the moments the system
    already treats as meaningful.

    There is no `push` and no remote, and that is a decision rather than an
    omission: a research workspace holds the pipeline, the data pointers and
    whatever a notebook has printed, and publishing it is not a thing to make
    one flag away. See `core/vcs.py`.
    """
    from core import vcs  # noqa: PLC0415

    if args.action == "init":
        result = vcs.initialise()
        if result.get("error"):
            raise UsageError(result["error"], fix=result.get("fix"))
        return {
            **result,
            "next": "nothing — collect, verdict and project sync now checkpoint on their own",
        }
    if args.action == "commit":
        if not vcs.enabled():
            raise UsageError(
                "this workspace is not versioned",
                fix="python -m tools.workspace vcs init --json",
            )
        return vcs.checkpoint(args.message)
    if args.action == "log":
        return {"root": str(vcs.root()), "checkpoints": vcs.history(args.limit)}
    return vcs.status()


if __name__ == "__main__":
    main(cli)
