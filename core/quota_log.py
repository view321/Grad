"""Token and credit accounting (HANDOFF §12 step 4, §10 widget 3).

    "'does this stage earn its quota' is unanswerable without measuring quota."

Every entry is tagged by stage, so the stage-0/stage-3 decision in §5 can be
made from numbers rather than from taste. Stages 0 and 3 additionally log their
full prompt and raw response -- they are the one place subagents are used, and
"debugging a funnel whose middle is invisible is guesswork".
"""

from __future__ import annotations

from typing import Any

from core import jsonl, paths
from core.ledger_store import now_iso

# Funnel stages plus the main loop. Free-form strings are allowed; these are the
# ones the UI knows how to group.
STAGE_MAIN = "main"
STAGE_EXPAND = "funnel.expand"        # stage 0
STAGE_RETRIEVE = "funnel.retrieve"    # stage 1 (free, logged for latency)
STAGE_RERANK = "funnel.rerank"        # stage 2 (credits, not quota)
STAGE_TRIAGE = "funnel.triage"        # stage 3
STAGE_EMBED = "embed"
STAGE_INGEST = "ingest"


def record(
    stage: str,
    *,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    credits_usd: float = 0.0,
    unit: str = "quota",
    session: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one usage record.

    `unit` distinguishes the two currencies this system spends: "quota" (the Max
    5x window, drawn on by anything going through the Agent SDK) and "credits"
    (OpenRouter/Voyage dollars). Conflating them is how stage 2 ends up looking
    expensive when it is the cheap one.
    """
    return jsonl.append(
        paths.quota_path(),
        {
            "at": now_iso(),
            "stage": stage,
            "model": model,
            "unit": unit,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cache_read_tokens": int(cache_read_tokens),
            "cache_write_tokens": int(cache_write_tokens),
            "credits_usd": round(float(credits_usd), 6),
            "session": session,
            "detail": detail or {},
        },
    )


def from_sdk_usage(
    stage: str, usage: Any, *, model: str | None = None, session: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Best-effort translation of an SDK usage payload.

    The SDK's result shape has changed between releases, so this reads
    defensively and records zeros rather than failing a research session over
    accounting.
    """
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
    return record(
        stage,
        model=model,
        input_tokens=get("input_tokens", 0) or 0,
        output_tokens=get("output_tokens", 0) or 0,
        cache_read_tokens=get("cache_read_input_tokens", 0) or 0,
        cache_write_tokens=get("cache_creation_input_tokens", 0) or 0,
        session=session,
        detail=detail,
    )


def entries() -> list[dict[str, Any]]:
    return jsonl.read(paths.quota_path())


def summarise(days: int | None = None) -> dict[str, Any]:
    """Totals by stage. This is what the header meter in §10 renders."""
    import datetime as dt

    rows = entries()
    if days:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        kept = []
        for r in rows:
            try:
                at = dt.datetime.fromisoformat(r.get("at", ""))
            except ValueError:
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=dt.timezone.utc)
            if at >= cutoff:
                kept.append(r)
        rows = kept

    by_stage: dict[str, dict[str, Any]] = {}
    for r in rows:
        node = by_stage.setdefault(
            r.get("stage", "unknown"),
            {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0, "credits_usd": 0.0},
        )
        node["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            node[k] += int(r.get(k, 0) or 0)
        node["credits_usd"] += float(r.get("credits_usd", 0.0) or 0.0)

    total_tokens = sum(n["input_tokens"] + n["output_tokens"] for n in by_stage.values())
    return {
        "window_days": days,
        "by_stage": {k: {**v, "credits_usd": round(v["credits_usd"], 4)} for k, v in sorted(by_stage.items())},
        "total_tokens": total_tokens,
        "total_credits_usd": round(sum(n["credits_usd"] for n in by_stage.values()), 4),
        # Anthropic exposes no remaining-quota API and the Max 5x window is
        # opaque, so this is self-measured usage against an assumed budget --
        # relative attribution by stage, not a fuel gauge. Labelled as such
        # here and in the UI.
        "authoritative": False,
    }
