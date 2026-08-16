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
#: The summary a compaction writes. Its own stage rather than folded into
#: `main`, because "what did compacting cost me" is the question that decides
#: whether the threshold is set right, and it is unanswerable if the compaction
#: turn is filed under the conversation it was compacting.
STAGE_COMPACT = "compaction"
STAGE_EXPAND = "funnel.expand"        # stage 0
STAGE_RETRIEVE = "funnel.retrieve"    # stage 1 (free, logged for latency)
STAGE_RERANK = "funnel.rerank"        # stage 2 (credits, not quota)
STAGE_TRIAGE = "funnel.triage"        # stage 3
STAGE_EMBED = "embed"
STAGE_INGEST = "ingest"
#: One mutation proposal (§21). Here rather than in `tools/evolve.py` so it sits
#: with the other stages a summary groups by -- and because the campaign loop is
#: the one place in this system that issues model calls with no human waiting on
#: them, which makes it the stage most worth being able to total.
STAGE_EVOLVE = "evolve.mutate"


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


# ---------------------------------------------------------------------------
# what a token counts as
# ---------------------------------------------------------------------------
#: The four kinds, and the config key that weights each. Ordered as they are
#: displayed, which is also cheapest-to-dearest for everything except the first.
KINDS: tuple[tuple[str, str], ...] = (
    ("input_tokens", "weight_input"),
    ("output_tokens", "weight_output"),
    ("cache_read_tokens", "weight_cache_read"),
    ("cache_write_tokens", "weight_cache_write"),
)

#: Used when the config cannot be read at all. Same numbers as `config.DEFAULTS`
#: -- duplicated rather than imported, because `core.budget` calls into here on
#: the gate path and accounting must not be what takes a session down.
FALLBACK_WEIGHTS: dict[str, float] = {
    "weight_input": 1.0,
    "weight_output": 1.0,
    "weight_cache_read": 0.1,
    "weight_cache_write": 1.25,
}


def weights(cfg: Any = None) -> dict[str, float]:
    """The `[quota]` weights, as floats, with every key present.

    A weight that is missing, non-numeric or negative falls back rather than
    raising: this is read on the path that decides whether a turn may be issued,
    and a typo in `grad.toml` should not be able to strand a session. A negative
    weight is refused specifically because it would make spending *lower* the
    measured total, which is the one error here that a ceiling cannot survive.
    """
    if cfg is None:
        try:
            from core import config as config_mod  # noqa: PLC0415

            cfg = config_mod.load()
        except Exception:  # noqa: BLE001 - see the docstring
            return dict(FALLBACK_WEIGHTS)
    out: dict[str, float] = {}
    for _, key in KINDS:
        try:
            value = float(cfg.get("quota", key, FALLBACK_WEIGHTS[key]))
        except (TypeError, ValueError):
            value = FALLBACK_WEIGHTS[key]
        if value != value or value in (float("inf"), float("-inf")) or value < 0:
            value = FALLBACK_WEIGHTS[key]
        out[key] = value
    return out


def counts(row: Any) -> dict[str, int]:
    """The four raw token counts of one record, defaulting to zero.

    **Clamped at zero**, for the same reason `weights` refuses a negative weight:
    a negative count would *reduce* the measured total, which is the one error a
    ceiling cannot survive -- one malformed row and a project has spending power
    it was never allocated. `tools/quota.py record` already refuses a negative
    at the CLI, but that is not the only door: `from_sdk_usage` records whatever
    the SDK reports, and the ledger is a file on disk that can be edited. The
    guard belongs at the read, where every path passes.
    """
    get = row.get if isinstance(row, dict) else (lambda k, d=0: getattr(row, k, d))
    out: dict[str, int] = {}
    for field, _ in KINDS:
        try:
            out[field] = max(0, int(get(field, 0) or 0))
        except (TypeError, ValueError):
            out[field] = 0
    return out


def billable(row: Any, weight: dict[str, float] | None = None) -> float:
    """One record's tokens as a single weighted number.

    **This is the only place the four kinds become one.** `core/budget.py`
    charges a ceiling with it, `summarise` totals it, and the UI meters it, so a
    change to what a cache read is worth lands everywhere at once. The four raw
    counts stay in the record and stay in every summary -- the weighting is how
    they are *compared*, never a substitute for having them.
    """
    weight = weights() if weight is None else weight
    n = counts(row)
    return sum(n[field] * weight.get(key, FALLBACK_WEIGHTS[key]) for field, key in KINDS)


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

    weight = weights()

    def _fold(key: str, fallback: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            node = out.setdefault(
                str(r.get(key) or fallback),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                 "cache_read_tokens": 0, "cache_write_tokens": 0,
                 "billable_tokens": 0.0, "credits_usd": 0.0},
            )
            node["calls"] += 1
            # Through `counts`, not straight off the record, so the raw figures
            # and the weighted total are clamped by one rule. Read separately,
            # a negative row would be excluded from `billable` and included in
            # the counts printed beside it -- two numbers describing the same
            # row and disagreeing, which is worse than either being wrong.
            for k, value in counts(r).items():
                node[k] += value
            node["billable_tokens"] += billable(r, weight)
            node["credits_usd"] += float(r.get("credits_usd", 0.0) or 0.0)
        # Rounded here and *not* re-rounded into the grand total below: rounding
        # each group and then summing accumulates up to half a token of error per
        # group, which is small but is also entirely avoidable.
        return {
            k: {**v,
                "billable_tokens": round(v["billable_tokens"]),
                "credits_usd": round(v["credits_usd"], 4)}
            for k, v in sorted(out.items())
        }

    by_stage = _fold("stage", "unknown")
    # `total_tokens` stays input + output, because that is what it has always
    # meant and something reads every field in here. The number that a ceiling
    # is charged against is `billable_tokens`, and the four raw counts are
    # reported beside both so the difference between them is visible rather than
    # buried in a weight -- on the first fortnight of use the two differ by a
    # factor of twelve, and a reader who cannot see why would be right not to
    # trust either.
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
        "totals": {
            field: sum(n[field] for n in by_stage.values()) for field, _ in KINDS
        },
        # From the rows, not from the rounded per-stage figures above. Summing
        # values that have each already been rounded carries every group's
        # rounding error into the one number a ceiling is compared against.
        "billable_tokens": round(sum(billable(r, weight) for r in rows)),
        "weights": weight,
        "total_credits_usd": round(sum(n["credits_usd"] for n in by_stage.values()), 4),
        # Anthropic exposes no remaining-quota API and the Max 5x window is
        # opaque, so this is self-measured usage against an assumed budget --
        # relative attribution by stage, not a fuel gauge. Labelled as such
        # here and in the UI.
        "authoritative": False,
    }
