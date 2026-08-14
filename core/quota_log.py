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
    role: str | None = None,
    project: str | None = None,
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

    `role` is the §16 model role, so `tools.quota summary --by-role` can answer
    "what did Opus cost me this week" without inferring it from model ids.
    `project` is the §15 dimension; it defaults to the current selection so a
    caller cannot forget to attribute a cost, and an unselected project lands as
    `unassigned` rather than as an error.
    """
    return jsonl.append(
        paths.quota_path(),
        {
            "at": now_iso(),
            "stage": stage,
            "model": model,
            "role": role,
            "project": project if project is not None else _current_project(),
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


def _current_project() -> str:
    """Imported at point of use: `core.budget` imports `core.jsonl`, and
    accounting must never be the reason a research session dies."""
    try:
        from core import budget  # noqa: PLC0415

        return budget.current_project() or budget.UNASSIGNED
    except Exception:  # noqa: BLE001
        return "unassigned"


def from_sdk_usage(
    stage: str, usage: Any, *, model: str | None = None, role: str | None = None,
    project: str | None = None, session: str | None = None,
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
        role=role,
        project=project,
        input_tokens=get("input_tokens", 0) or 0,
        output_tokens=get("output_tokens", 0) or 0,
        cache_read_tokens=get("cache_read_input_tokens", 0) or 0,
        cache_write_tokens=get("cache_creation_input_tokens", 0) or 0,
        session=session,
        detail=detail,
    )


def entries() -> list[dict[str, Any]]:
    return jsonl.read(paths.quota_path())


def summarise(days: int | None = None, *, project: str | None = None) -> dict[str, Any]:
    """Totals by stage, by role, and by project.

    This is what the header meter in §10 renders. `by_role` is what makes
    "what did Opus cost me this week" answerable without inference (§16), and
    `by_project` is the per-project breakdown the Quota tab shows (§15).
    """
    import datetime as dt

    rows = entries()
    if project:
        rows = [r for r in rows if (r.get("project") or "unassigned") == project]
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

    def _fold(key: str, fallback: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            node = out.setdefault(
                str(r.get(key) or fallback),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                 "cache_read_tokens": 0, "cache_write_tokens": 0, "credits_usd": 0.0},
            )
            node["calls"] += 1
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                node[k] += int(r.get(k, 0) or 0)
            node["credits_usd"] += float(r.get("credits_usd", 0.0) or 0.0)
        return {k: {**v, "credits_usd": round(v["credits_usd"], 4)} for k, v in sorted(out.items())}

    by_stage = _fold("stage", "unknown")
    total_tokens = sum(n["input_tokens"] + n["output_tokens"] for n in by_stage.values())
    return {
        "window_days": days,
        "project": project,
        "by_stage": by_stage,
        # Records written before §16 carry no role; they fold as `untagged`
        # rather than being attributed to a role they never had.
        "by_role": _fold("role", "untagged"),
        "by_project": _fold("project", "unassigned"),
        "total_tokens": total_tokens,
        "total_credits_usd": round(sum(n["credits_usd"] for n in by_stage.values()), 4),
        # Anthropic exposes no remaining-quota API and the Max 5x window is
        # opaque, so this is self-measured usage against an assumed budget --
        # relative attribution by stage, not a fuel gauge. Labelled as such
        # here and in the UI.
        "authoritative": False,
    }
