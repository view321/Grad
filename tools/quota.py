"""grad-quota -- read the token and credit log, summarised by stage (HANDOFF §12 step 4).

    "'does this stage earn its quota' is unanswerable without measuring quota."

This is the measurement instrument for every later cost decision, and it is
deliberately honest about what it can measure: Anthropic exposes no
remaining-quota API and the Max 5x window (5-hour rolling plus weekly caps) is
opaque, so these are *self-measured* usage numbers against an assumed budget --
relative attribution by stage, not a fuel gauge.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from core import config as config_mod, ledger_store as ls, quota_log
from core.cli import Cli, main
from core.errors import UsageError

cli = Cli(
    "grad-quota",
    "Summarise measured token usage by stage, credits spent, and rolling GPU spend.",
    epilog=(
        "Two currencies, never conflated: `quota` is the Max 5x window (anything via the\n"
        "Agent SDK), `credits` are dollars (OpenRouter rerank, Voyage embeddings).\n"
        "Stage 2 stays quota-free on purpose."
    ),
)


def _summary_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--days", type=int, help="restrict to the last N days")
    p.add_argument("--stage", help="only this stage")


@cli.command("summary", "totals by stage", setup=_summary_args)
def cmd_summary(args: argparse.Namespace) -> dict[str, Any]:
    summary = quota_log.summarise(args.days)
    if args.stage:
        by_stage = summary["by_stage"]
        if args.stage not in by_stage:
            raise UsageError(
                f"no usage recorded for stage {args.stage!r}",
                fix=f"known stages: {', '.join(by_stage) or '(none yet)'}",
            )
        summary["by_stage"] = {args.stage: by_stage[args.stage]}
    return summary


@cli.command(
    "funnel",
    "what the retrieval funnel costs, stage by stage",
    setup=lambda p: p.add_argument("--days", type=int),
)
def cmd_funnel(args: argparse.Namespace) -> dict[str, Any]:
    """The numbers behind the §5 stage-0/stage-3 decision.

    Stages 0 and 3 are the deliberate subagent exception; this is what tells you
    whether they earned it. Pair it with `evals/retrieval.jsonl` -- cost alone
    cannot answer the question, only half of it.
    """
    summary = quota_log.summarise(args.days)
    stages = [
        quota_log.STAGE_EXPAND,
        quota_log.STAGE_RETRIEVE,
        quota_log.STAGE_RERANK,
        quota_log.STAGE_TRIAGE,
    ]
    rows = {s: summary["by_stage"].get(s, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "credits_usd": 0.0}) for s in stages}
    return {
        "window_days": args.days,
        "stages": rows,
        "quota_tokens_stage0_and_3": sum(
            rows[s]["input_tokens"] + rows[s]["output_tokens"]
            for s in (quota_log.STAGE_EXPAND, quota_log.STAGE_TRIAGE)
        ),
        "credits_usd_stage2": rows[quota_log.STAGE_RERANK]["credits_usd"],
        "note": "cost is half the question; the other half is evals/retrieval.jsonl",
    }


@cli.command("spend", "rolling GPU spend against the monthly ceiling")
def cmd_spend(_: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    window = int(cfg.get("spend", "window_days", 30))
    rolling = ls.rolling_spend(window)
    monthly = float(cfg.get("spend", "monthly_usd", 200.0))
    stale = [r.id for r in ls.stale_runs(cfg=cfg)]
    return {
        "window_days": window,
        "actual_usd": rolling["actual_usd"],
        "in_flight_usd": rolling["in_flight_usd"],
        "total_usd": rolling["total_usd"],
        "monthly_ceiling_usd": monthly,
        "headroom_usd": round(monthly - rolling["total_usd"], 4),
        "uncollected_runs": [r.id for r in ls.in_flight()],
        "stale_runs": stale,
        "submissions_blocked": bool(stale),
    }


def _record_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", required=True)
    p.add_argument("--model")
    p.add_argument("--input-tokens", type=int, default=0)
    p.add_argument("--output-tokens", type=int, default=0)
    p.add_argument("--credits-usd", type=float, default=0.0)
    p.add_argument("--unit", choices=["quota", "credits"], default="quota")
    p.add_argument("--session")


@cli.command("record", "append a usage record (used by the Stop hook)", setup=_record_args)
def cmd_record(args: argparse.Namespace) -> dict[str, Any]:
    # This is the measurement instrument for every later cost decision, so a
    # negative count (which would reduce reported usage) or a NaN (which is not
    # valid JSON and would poison every later sum) is refused rather than stored.
    if args.input_tokens < 0 or args.output_tokens < 0 or args.credits_usd < 0:
        raise UsageError("usage values must be non-negative", fix="check the arguments")
    if not math.isfinite(args.credits_usd):
        raise UsageError("--credits-usd must be a finite number", fix="pass a real dollar amount")
    return {
        "recorded": quota_log.record(
            args.stage,
            model=args.model,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            credits_usd=args.credits_usd,
            unit=args.unit,
            session=args.session,
        )
    }


@cli.command(
    "tail",
    "the most recent usage records",
    setup=lambda p: p.add_argument("-n", type=int, default=20),
)
def cmd_tail(args: argparse.Namespace) -> dict[str, Any]:
    if args.n < 0:
        raise UsageError("-n must be non-negative", fix="python -m tools.quota tail -n 20 --json")
    entries = quota_log.entries()
    # `entries[-0:]` is the whole list, which is the opposite of what -n 0 asks.
    return {"entries": entries[-args.n :] if args.n else []}


if __name__ == "__main__":
    main(cli)
