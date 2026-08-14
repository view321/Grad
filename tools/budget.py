"""grad-budget -- projects and their ceilings (HANDOFF-2 §15).

    "So README.md's claim that 'cumulative spend stays bounded' holds for one
     resource in three. The other two are instrumented and unbounded."

This closes that. Three resources are consumed -- GPU dollars, API credits, and
subscription tokens -- and all three now carry a ceiling scoped to a project.

**The honest statement about enforcement, because it differs by resource:**

  * **GPU dollars** are enforced at submit, which is a discrete gateable event.
    Refused before anything is spent.
  * **API credits** are enforced at the same boundary and measured continuously.
  * **Subscription tokens are enforced to a granularity of one turn's overrun.**
    Tokens are consumed continuously inside a turn and there is no way to refuse
    mid-turn, so `agent.py` checks the remaining allocation *before* issuing the
    next turn and `hooks.py` denies cost-bearing Bash once the project is over.
    A turn already in flight finishes.

And a second honesty note, because a meter that overclaims is worse than none:
subscription quota is not linear in tokens, and the real limits are rolling
windows (5-hour and weekly on Max) that the SDK does not expose as a remaining
balance. **A token ceiling here is a proxy you control, not a mirror of
Anthropic's limit.** Hitting the real rate limit is an event this system can
only observe after the fact.

`raise` appends an event rather than editing the record: a ceiling that can be
changed invisibly is not a ceiling.
"""

from __future__ import annotations

import argparse
from typing import Any

from core import budget, paths
from core.cli import Cli, main
from core.errors import EXIT_PROJECT_BUDGET, UsageError

cli = Cli(
    "grad-budget",
    "Create research projects and bound what they may spend: GPU dollars, API "
    "credits, and subscription tokens.",
    epilog=(
        "Exit 12 is a project budget refusal, distinct from 6 (the machine's global\n"
        "spend ceiling), so 'this research ran out of its allocation' is never confused\n"
        "with 'the machine is out of money'.\n\n"
        "Enforcement differs by resource, and the difference is structural:\n"
        "  gpu_usd       refused at submit, before anything is spent\n"
        "  credits_usd   refused at the same boundary\n"
        "  quota_tokens  enforced to a granularity of ONE TURN'S OVERRUN -- tokens are\n"
        "                consumed continuously inside a turn and there is no way to\n"
        "                refuse mid-turn, so the check runs before the *next* one\n\n"
        "A token ceiling is a proxy you control, not a mirror of Anthropic's limit:\n"
        "the real caps are rolling windows the SDK does not expose as a balance."
    ),
)


# ---------------------------------------------------------------------------
def _budget_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-usd", type=float, help="ceiling on GPU dollars for this project")
    p.add_argument(
        "--quota-tokens",
        type=float,
        help="ceiling on subscription tokens (accepts 5e6). Enforced to one turn's overrun.",
    )
    p.add_argument("--credits-usd", type=float, help="ceiling on API credits (Voyage, OpenRouter)")


def _collect_budget(args: argparse.Namespace) -> dict[str, float]:
    out: dict[str, float] = {}
    if args.gpu_usd is not None:
        out["gpu_usd"] = float(args.gpu_usd)
    if args.quota_tokens is not None:
        # `--quota-tokens 5e6` is the documented spelling, so it parses as a
        # float and is stored as a whole number of tokens.
        out["quota_tokens"] = float(int(args.quota_tokens))
    if args.credits_usd is not None:
        out["credits_usd"] = float(args.credits_usd)
    for name, value in out.items():
        if value < 0:
            raise UsageError(
                f"--{name.replace('_', '-')} must not be negative",
                fix="a ceiling of 0 blocks all spend; omit the flag to leave it unbounded",
            )
    return out


def _new_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--id", required=True, help="short slug, e.g. proj-scaling-w2")
    p.add_argument("--title", required=True, help="what this research is")
    p.add_argument(
        "--payer",
        help="who pays. `hf:<org>` attributes HF jobs to that organization namespace (§17)",
    )
    p.add_argument("--use", action="store_true", help="also select it as the current project")
    _budget_flags(p)


@cli.command("new", "create a project and its ceilings", setup=_new_args)
def cmd_new(args: argparse.Namespace) -> dict[str, Any]:
    """A project is the unit three separate requirements turned out to share:
    HF payer attribution, the bound on an evolutionary campaign, and a budget
    for a piece of research."""
    paths.ensure_workspace()
    record = budget.create(
        args.id, title=args.title, budget=_collect_budget(args), payer=args.payer
    )
    if args.use:
        budget.set_current(args.id)
    return {
        "project": record,
        "current": budget.current_project(),
        "next": f"python -m tools.budget use {args.id} --json" if not args.use else None,
    }


@cli.command(
    "use",
    "select the current project (written to ledger/.current_project)",
    setup=lambda p: p.add_argument("project_id"),
)
def cmd_use(args: argparse.Namespace) -> dict[str, Any]:
    """A file, not an environment variable.

    `credentials.scrub_environment()` strips the agent's environment at startup,
    and a selection mechanism that the agent's own startup deletes is a bug
    waiting to happen.
    """
    proj = budget.project(args.project_id)
    if proj["status"] == "closed":
        raise UsageError(
            f"project {args.project_id!r} is closed",
            fix="python -m tools.budget new --id <new-id> --title '...' --json",
        )
    budget.set_current(args.project_id)
    return {"current": args.project_id, "title": proj["title"], "payer": proj["payer"]}


def _project_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="override the current project for this invocation")


@cli.command("status", "spend and remaining, per resource", setup=_project_arg)
def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    project_id = budget.resolve_or_fail(args.project, what="status")
    state = budget.status(project_id)
    if state["over_budget"]:
        # Reported as data, not raised: `status` exists to be readable *while*
        # over budget. The refusal belongs at the point of spend.
        state["blocked"] = [
            "python -m tools.jobs submit",
            "python -m tools.evolve run",
            "python -m tools.report write",
        ]
        state["fix"] = (
            f"python -m tools.budget raise --project {project_id} "
            f"--{state['over_budget'][0].replace('_', '-')} <new ceiling> --json"
        )
    return state


def _raise_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    p.add_argument("--reason", default="", help="why the ceiling moved. it ages badly without one")
    _budget_flags(p)


@cli.command("raise", "move a ceiling, as a logged event", setup=_raise_args)
def cmd_raise(args: argparse.Namespace) -> dict[str, Any]:
    """Appends rather than mutates.

    Same argument as §7's append-only ledger: the previous value stays readable,
    so "we kept raising it" is visible instead of inferred.
    """
    project_id = budget.resolve_or_fail(args.project, what="raise")
    record = budget.raise_ceiling(project_id, budget=_collect_budget(args), reason=args.reason)
    return {"raised": record, "status": budget.status(project_id)}


@cli.command(
    "close",
    "close a project (its records stay; nothing is deleted)",
    setup=lambda p: p.add_argument("project_id"),
)
def cmd_close(args: argparse.Namespace) -> dict[str, Any]:
    record = budget.close(args.project_id)
    return {"closed": record, "current": budget.current_project()}


@cli.command("list", "every project, with spend against its ceilings")
def cmd_list(_: argparse.Namespace) -> dict[str, Any]:
    current = budget.current_project()
    rows = []
    for pid, proj in budget.projects().items():
        state = budget.status(pid)
        rows.append(
            {
                "id": pid,
                "title": proj["title"],
                "status": proj["status"],
                "payer": proj["payer"],
                "current": pid == current,
                "resources": state["resources"],
                "over_budget": state["over_budget"],
            }
        )
    return {"current": current, "projects": rows}


def _check_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    p.add_argument("--gpu-usd", type=float, default=0.0, help="dollars this would add")
    p.add_argument("--quota-tokens", type=float, default=0.0, help="tokens this would add")
    p.add_argument("--credits-usd", type=float, default=0.0, help="credits this would add")


@cli.command("check", "would this spend fit? exits 12 if not", setup=_check_args)
def cmd_check(args: argparse.Namespace) -> dict[str, Any]:
    """The gate, callable directly.

    `tools.evolve` uses this shape before each generation, and it is here so a
    pipeline can ask the question without importing anything.
    """
    project_id = budget.resolve(args.project)
    state = budget.check(
        project_id,
        gpu_usd=args.gpu_usd,
        quota_tokens=int(args.quota_tokens),
        credits_usd=args.credits_usd,
        what="the proposed spend",
    )
    if state is None:
        return {
            "project": project_id,
            "bounded": False,
            "note": "no project selected, or the project has no ceilings; spend is tracked, not bounded",
        }
    return {"project": project_id, "bounded": True, "fits": True, "resources": state["resources"]}


# Re-exported so a caller can `from tools.budget import EXIT_PROJECT_BUDGET`
# instead of remembering the number.
__all__ = ["cli", "EXIT_PROJECT_BUDGET"]


if __name__ == "__main__":
    main(cli)
