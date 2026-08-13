"""grad-ledger -- the expectations ledger (HANDOFF §7).

    "Predict before you run. Record the prediction. Compare. Keep both."

This CLI writes predictions and verdicts. It deliberately cannot write results:
those come from `jobs.py collect` / `gpu.py collect`, because a result the model
types from memory at the end of a long session is the failure mode the whole
document is built to avoid.
"""

from __future__ import annotations

import argparse
from typing import Any

from core import jsonl, ledger_store as ls, paths
from core.cli import Cli, main
from core.errors import EXIT_USAGE, GradError, NotFound, UsageError

cli = Cli(
    "grad-ledger",
    "Append and query the expectations ledger (predictions, verdicts, runs).",
    epilog=(
        "The JSONL files under ledger/ are the source of truth; ledger.sqlite is a\n"
        "derived index and can be deleted and rebuilt at any time.\n\n"
        "Results are written by `jobs.py collect`, never here."
    ),
)


# ---------------------------------------------------------------------------
def _expect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, help="task id this prediction belongs to")
    p.add_argument("--quantity", required=True, help="e.g. val_loss@1e9_tokens")
    p.add_argument("--claim", default="", help="one sentence, in words")
    p.add_argument("--low", type=float, help="low end of the predicted range")
    p.add_argument("--high", type=float, help="high end of the predicted range")
    p.add_argument(
        "--direction",
        choices=["lower_is_better", "higher_is_better", "increase", "decrease"],
        help="for relational predictions that have no absolute range",
    )
    p.add_argument(
        "--basis",
        action="append",
        default=[],
        metavar="PAPER|LOCATOR|VALUE|CONDITIONS",
        help="provenance, repeatable. e.g. 'arXiv:2001.08361|Table 3, row 2|3.05|1.3B params'",
    )
    p.add_argument(
        "--comparability",
        default="",
        help="how our setup differs from the basis. REQUIRED for absolute predictions",
    )
    p.add_argument("--confidence", choices=list(ls.CONFIDENCES), default="medium")


@cli.command("expect", "pre-register a prediction (must exist before submit)", setup=_expect_args)
def cmd_expect(args: argparse.Namespace) -> dict[str, Any]:
    """Write an expectation. `jobs.py submit --expect <id>` binds it to a run."""
    paths.ensure_workspace()

    if args.low is None and args.high is None and not args.direction:
        raise UsageError(
            "a prediction needs either a range (--low/--high) or a --direction",
            fix="--low 2.9 --high 3.2   (or)   --direction decrease",
        )
    if args.low is not None and args.high is not None and args.low > args.high:
        raise UsageError("--low is greater than --high", fix="swap the two values")

    absolute = args.low is not None or args.high is not None
    if absolute and not args.comparability.strip():
        # HANDOFF §7: "Absolute numbers require a populated comparability field
        # to be recorded at all." A number from a paper means nothing without
        # matching tokenizer, dataset, eval protocol, sequence length, params.
        raise UsageError(
            "an absolute prediction requires --comparability describing how this setup "
            "differs from the basis (tokenizer, dataset, eval protocol, sequence length, "
            "parameter count)",
            fix=(
                '--comparability "our tokenizer differs; eval is a 5k held-out subset"  '
                "(or make the prediction relational with --direction)"
            ),
        )

    if args.task in ls.tasks_with_results():
        raise GradError(
            "task_has_results",
            f"task {args.task!r} already has a collected run; an expectation written now "
            "would be a prediction authored after the fact",
            exit_code=EXIT_USAGE,
            fix="use a new --task id for the next experiment",
        )

    basis = [_parse_basis(b) for b in args.basis]
    if absolute and not basis:
        raise UsageError(
            "an absolute prediction with no --basis is a guess wearing a citation's clothes",
            fix="--basis 'arXiv:2001.08361|Table 3, row 2|3.05|1.3B params, 100B tokens'",
        )

    record = {
        "id": ls.new_id("exp"),
        "task": args.task,
        "created_at": ls.now_iso(),
        "quantity": args.quantity,
        "claim": args.claim or _synthesise_claim(args),
        "predicted": {"low": args.low, "high": args.high, "direction": args.direction},
        "basis": basis,
        "comparability": args.comparability,
        "confidence": args.confidence,
    }
    ls.append_expectation(record)
    return {
        "expectation": record,
        "next": f"python -m tools.jobs submit --spec <spec> --expect {record['id']} --json",
    }


def _parse_basis(text: str) -> dict[str, Any]:
    parts = [p.strip() for p in text.split("|")]
    if not parts or not parts[0]:
        raise UsageError(
            f"malformed --basis {text!r}",
            fix="--basis 'PAPER|LOCATOR|VALUE|CONDITIONS' (locator and value optional)",
        )
    value: Any = None
    if len(parts) > 2 and parts[2]:
        try:
            value = float(parts[2])
        except ValueError:
            value = parts[2]
    return {
        "paper": parts[0],
        "locator": parts[1] if len(parts) > 1 else "",
        "value": value,
        "conditions": parts[3] if len(parts) > 3 else "",
    }


def _synthesise_claim(args: argparse.Namespace) -> str:
    if args.low is not None and args.high is not None:
        return f"{args.quantity} should land between {args.low} and {args.high}"
    if args.direction:
        return f"{args.quantity} should {args.direction.replace('_', ' ')}"
    return args.quantity


# ---------------------------------------------------------------------------
def _query_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quantity")
    p.add_argument("--task")
    p.add_argument("--expectations", action="store_true", help="list expectations")
    p.add_argument("--runs", action="store_true", help="list runs")
    p.add_argument(
        "--pending",
        action="store_true",
        help="uncollected runs and unjudged deviations (the things that quietly accumulate)",
    )
    p.add_argument("--open", action="store_true", help="expectations not yet bound to a run")
    p.add_argument("--limit", type=int, default=50)


@cli.command("query", "query predictions, runs, and what is still pending", setup=_query_args)
def cmd_query(args: argparse.Namespace) -> dict[str, Any]:
    if args.pending:
        return ls.pending()

    out: dict[str, Any] = {}
    want_both = not (args.expectations or args.runs)

    if args.expectations or want_both:
        bound = ls.bound_expectation_ids()
        falsified = ls.falsified_ids()
        rows = [
            {**e, "bound": e["id"] in bound, "falsified": e["id"] in falsified}
            for e in ls.expectations()
            if (not args.quantity or e.get("quantity") == args.quantity)
            and (not args.task or e.get("task") == args.task)
            and (not args.open or e["id"] not in bound)
        ]
        out["expectations"] = rows[-args.limit :]

    if args.runs or want_both:
        rows = [
            r.data
            for r in ls.runs()
            if (not args.task or r.get("task") == args.task)
            and (
                not args.quantity
                or args.quantity in (r.get("results") or {})
                or any(d.get("quantity") == args.quantity for d in r.get("deviations", []))
            )
        ]
        out["runs"] = rows[-args.limit :]

    return out


# ---------------------------------------------------------------------------
def _verdict_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_id")
    p.add_argument("--quantity", required=True)
    p.add_argument("--verdict", required=True, choices=list(ls.VERDICTS))
    p.add_argument("--note", default="", help="why. a verdict with no reasoning ages badly")


@cli.command("verdict", "judge a deviation (bug | real | inconclusive)", setup=_verdict_args)
def cmd_verdict(args: argparse.Namespace) -> dict[str, Any]:
    """The one field a program does not fill in.

    `collect` computes the deviation mechanically and leaves the verdict unset;
    this is where judgement enters, and it cannot overwrite the record it judges.
    """
    r = ls.run(args.run_id)
    quantities = [d.get("quantity") for d in r.get("deviations", [])]
    if args.quantity not in quantities:
        raise NotFound(
            f"run {args.run_id} has no deviation for quantity {args.quantity!r}; "
            f"it has: {', '.join(q for q in quantities if q) or '(none)'}",
            fix=f"python -m tools.ledger query --runs --task {r.get('task')} --json",
        )
    record = {
        "type": ls.T_VERDICT,
        "id": args.run_id,
        "quantity": args.quantity,
        "verdict": args.verdict,
        "note": args.note,
        "judged_at": ls.now_iso(),
    }
    ls.append_run_event(record)
    return {"verdict": record, "remaining_unjudged": len(ls.run(args.run_id).unjudged_deviations())}


# ---------------------------------------------------------------------------
def _falsify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("expectation_id")
    p.add_argument("--note", required=True, help="what showed it wrong")


@cli.command("falsify", "mark an expectation wrong (never delete it)", setup=_falsify_args)
def cmd_falsify(args: argparse.Namespace) -> dict[str, Any]:
    """A wrong prediction with a recorded correction is more useful to a future
    session than a gap, so entries are marked, never removed."""
    exp = ls.expectation(args.expectation_id)
    record = {
        "type": ls.T_EXPECTATION_FALSIFIED,
        "id": exp["id"],
        "at": ls.now_iso(),
        "note": args.note,
    }
    ls.append_expectation_event(record)
    return {"falsified": record}


# ---------------------------------------------------------------------------
@cli.command("show", "show one expectation or run in full", setup=lambda p: p.add_argument("id"))
def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    if args.id.startswith("exp-"):
        return {"expectation": ls.expectation(args.id)}
    r = ls.run(args.id)
    exp = None
    if r.get("expectation_id"):
        try:
            exp = ls.expectation(r["expectation_id"])
        except NotFound:
            exp = None
    return {"run": r.data, "expectation": exp, "artifacts": str(paths.run_artifacts(r.id))}


@cli.command("reindex", "rebuild ledger.sqlite from the JSONL")
def cmd_reindex(_: argparse.Namespace) -> dict[str, Any]:
    counts = ls.rebuild_index()
    return {"rebuilt": str(paths.ledger_sqlite()), **counts}


@cli.command("verify", "check the ledgers for damage and dangling references")
def cmd_verify(_: argparse.Namespace) -> dict[str, Any]:
    """Readers tolerate a torn final line; this reports what was tolerated."""
    exp_bad = jsonl.damaged_lines(paths.expectations_path())
    run_bad = jsonl.damaged_lines(paths.runs_path())
    known = {e["id"] for e in ls.expectations()}
    dangling = [
        {"run_id": r.id, "expectation_id": r.get("expectation_id")}
        for r in ls.runs()
        if r.get("expectation_id") and r["expectation_id"] not in known
    ]
    report = {
        "damaged_lines": {"expectations.jsonl": exp_bad, "runs.jsonl": run_bad},
        "dangling_expectation_refs": dangling,
        "ok": not (exp_bad or run_bad or dangling),
    }
    if not report["ok"]:
        raise GradError(
            "ledger_damaged",
            "the ledger has damaged lines or dangling references",
            exit_code=9,
            fix="inspect the reported line numbers by hand; the JSONL is the source of truth",
            detail=report,
        )
    return report


if __name__ == "__main__":
    main(cli)
