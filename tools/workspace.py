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
    if (target / "ledger" / "runs.jsonl").exists():
        raise UsageError(
            f"{target} already holds a ledger",
            fix="choose an empty folder; two ledgers cannot be merged by copying",
        )

    planned = _plan_copy(source, target)
    total = sum(entry["files"] for entry in planned)
    if args.dry_run:
        return {
            "from": str(source),
            "to": str(target),
            "entries": planned,
            "files": total,
            "message": f"would copy {total} file(s) in {len(planned)} entr(ies) to {target}",
        }

    copied = [entry for entry in planned if _copy_entry(source, target, entry["name"])]
    verified, missing = _verify(source, target, [entry["name"] for entry in copied])

    removed: list[str] = []
    if args.remove_originals:
        if missing:
            raise UsageError(
                f"{len(missing)} file(s) did not arrive; nothing was deleted",
                fix="re-run without --remove-originals and compare the two folders by hand",
            )
        removed = _remove(source, [entry["name"] for entry in copied])

    moved_pointer = None
    if not args.keep_pointer:
        moved_pointer = str(workspace_mod.select(target))
        paths.ensure_workspace()

    return {
        "from": str(source),
        "to": str(target),
        "entries": copied,
        "files_copied": verified,
        "missing": missing[:20],
        "removed": removed,
        "workspace": moved_pointer,
        "message": _move_message(target, verified, removed, args),
    }


def _move_message(target: Path, verified: int, removed: list[str], args: argparse.Namespace) -> str:
    head = f"copied {verified} file(s) to {target}"
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


if __name__ == "__main__":
    main(cli)
