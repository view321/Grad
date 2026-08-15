"""grad-update -- move this installation to a release, and say what that costs.

The mechanics are in `core/update.py`; this is the contract around them. Three
commands, and the split between them is the point:

  * `check` never changes anything and never raises for a "no". It is what the
    app's menu renders and what the background poll caches.
  * `apply` is the half with side effects, and it refuses rather than returning
    a refusal -- a caller that ignored a returned "no" would be half-updated.
  * `versions` lists what there is to move to, because `--to` needs a name and
    guessing at tag spellings is exactly the retry-with-guessed-flags loop the
    CLI contract exists to prevent.

`grad --update` is the same thing with no arguments, for people who will never
type `python -m`.
"""

from __future__ import annotations

import argparse
from typing import Any

from core import update as update_mod, version
from core.cli import Cli, main
from core.errors import UsageError

cli = Cli(
    "grad-update",
    "Update Grad to the newest release, or pin it to an older one.",
    epilog=(
        "The installation is a git checkout and an editable install, so a release that\n"
        "changed only Python code is live as soon as the checkout moves -- the dependency\n"
        "install is skipped unless pyproject.toml itself changed.\n\n"
        "Your research is never touched. If it lives inside the installation folder,\n"
        "`grad workspace move <folder>` separates the two."
    ),
)


@cli.command(
    "check",
    "is there a newer release, and would anything stop it",
    setup=lambda p: p.add_argument(
        "--offline",
        action="store_true",
        help="do not contact the remote; answer from the refs already fetched",
    ),
)
def cmd_check(args: argparse.Namespace) -> dict[str, Any]:
    """What an update would do. Changes nothing, and caches the answer.

    Cached because the app renders this and must never run git on a draw: the
    menu reads `update.json` and the background poll is what writes it. Running
    the command by hand refreshes the same file, so a check from a terminal and
    a badge in the window cannot disagree.
    """
    plan = update_mod.plan(do_fetch=not args.offline)
    update_mod.write_cache(plan)
    return {
        **plan,
        "message": _check_message(plan),
    }


def _check_message(plan: dict[str, Any]) -> str:
    if plan["blockers"]:
        return f"{plan['blockers'][0]['message']} — {plan['blockers'][0].get('fix', '')}".strip(" —")
    if not plan["available"]:
        return f"up to date ({plan['label']})"
    target = plan["target"]
    behind = f", {plan['behind']} commit(s) behind" if plan.get("behind") else ""
    cost = "reinstalls dependencies" if plan["needs_reinstall"] else "no dependency change"
    return f"{target['tag']} is available{behind} — {cost}. Apply it with: grad update"


def _apply_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--to", metavar="TAG", help="pin a specific release instead of the newest")
    p.add_argument(
        "--rollback",
        action="store_true",
        help="go back to the release before this one (reproducing an older result)",
    )
    p.add_argument(
        "--extras",
        help="dependency extras for the reinstall; defaults to the ones this was installed with",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="apply despite the blockers check reported (it will explain what it is overriding)",
    )
    p.add_argument("--offline", action="store_true", help="do not contact the remote first")


@cli.command("apply", "move this installation to a release", setup=_apply_args)
def cmd_apply(args: argparse.Namespace) -> dict[str, Any]:
    """Fast-forward, reinstall if the dependencies moved, migrate state.

    `--rollback` resolves to a tag here rather than inside `core/update.py` so
    that the refusal for "there is nothing to roll back to" is a usage error
    with a list of releases in it, which is more useful than a plan with an
    empty target.
    """
    if args.rollback and args.to:
        raise UsageError(
            "--rollback and --to name two different targets",
            fix="use one: --rollback goes back exactly one release",
        )
    target = args.to
    if args.rollback:
        target = update_mod.rollback_target()
        if not target:
            raise UsageError(
                "there is no earlier release to roll back to",
                fix=f"releases: {', '.join(update_mod.release_tags()[:10]) or '(none)'}",
            )
    return update_mod.apply(
        to=target,
        with_extras=update_mod.parse_extras(args.extras),
        force=args.force,
        do_fetch=not args.offline,
    )


@cli.command(
    "versions",
    "what this is, and which releases exist",
    setup=lambda p: p.add_argument("--offline", action="store_true", help="skip the fetch"),
)
def cmd_versions(args: argparse.Namespace) -> dict[str, Any]:
    if not args.offline:
        update_mod.fetch()
    identity = version.identity(reload=True)
    tags = update_mod.release_tags()
    return {
        "installed": identity,
        "label": version.label(identity),
        "extras": update_mod.extras(),
        "releases": tags,
        "newest": tags[0] if tags else None,
        "rollback_to": update_mod.rollback_target(),
        "message": f"{version.label(identity)} — {len(tags)} release(s) known",
    }


if __name__ == "__main__":
    main(cli)
