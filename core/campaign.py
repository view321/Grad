"""Campaigns and candidates (HANDOFF-2 §21).

An evolutionary campaign collides with four things this system already
believes, and this module is where three of those collisions are resolved. The
fourth (Goodhart) is resolved in `tools/evolve.py`, at promotion time.

**1. The expectation gate is 1:1 with a run; evolution is 1:N.** You cannot
pre-register a prediction per candidate. So the **campaign** is the unit of
prediction -- "the evolved variant beats baseline X on metric Y by >= Z" -- and
candidate evaluations are recorded as sub-runs exempt from the per-run
expectation gate. That is arguably more faithful to §7 than the current design,
and it is exactly the relational shape `prompts/system.md` already prefers.

**2. The ledger would drown.** §23 item 4 asks how granular sub-runs should be:
one record per candidate is honest but a 100-generation campaign is thousands of
rows in `runs.jsonl`. Resolved as specified there -- a separate
`ledger/candidates.jsonl`, folded into the campaign record, with **only promoted
candidates entering `runs.jsonl`** through the normal submit path.

**3. Every mutation invalidates the preflight hash -- correctly, and
expensively.** `Submission.hash()` covers the entrypoint's import graph, so each
candidate needs a fresh preflight, and `smoke` is a *paid remote job*. Naively
that doubles per-candidate cost. Resolved by `escaped_evolve_block()` below: the
`EVOLVE-BLOCK` markers make "did the mutation stay inside the mutable region"
mechanically checkable, so candidates run `--only tests,dry_run` (both local,
both fast) and smoke is required once per campaign at the baseline, and again
only when a mutation escapes.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from core import jsonl, paths
from core.errors import EXIT_USAGE, GradError, NotFound

T_CAMPAIGN = "campaign"
T_GENERATION = "campaign_generation"
T_CAMPAIGN_CLOSED = "campaign_closed"
T_HALT_REQUESTED = "campaign_halt_requested"
T_CANDIDATE = "candidate"
T_CANDIDATE_PROMOTED = "candidate_promoted"

STATUSES = ("open", "closed", "exhausted", "failed", "halted")

# Shinka's own markers, kept verbatim so a task directory works with the
# upstream tool unmodified.
BLOCK_START = "EVOLVE-BLOCK-START"
BLOCK_END = "EVOLVE-BLOCK-END"


def campaigns_path() -> Path:
    return paths.ledger_dir() / "campaigns.jsonl"


def candidates_path() -> Path:
    return paths.ledger_dir() / "candidates.jsonl"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    import secrets  # noqa: PLC0415

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


# ---------------------------------------------------------------------------
# the evolve block
# ---------------------------------------------------------------------------
def split_blocks(source: str) -> tuple[list[str], list[str]]:
    """(mutable regions, everything outside them).

    A file with no markers is entirely *outside* the block, which is the
    conservative reading: an unmarked file that changed is treated as an escape
    and therefore requires a fresh smoke check.
    """
    inside: list[str] = []
    outside: list[str] = []
    current: list[str] = []
    in_block = False
    for line in source.splitlines():
        if BLOCK_START in line:
            outside.extend(current)
            current = []
            in_block = True
            continue
        if BLOCK_END in line:
            inside.extend(current)
            current = []
            in_block = False
            continue
        current.append(line)
    (inside if in_block else outside).extend(current)
    return inside, outside


def escaped_evolve_block(baseline: str, candidate: str) -> dict[str, Any]:
    """Did this mutation change anything outside the mutable region?

    This is the check that keeps a campaign affordable. Inside the block, a
    mutation cannot change the environment the job runs in, so the baseline's
    smoke result still applies and candidates need only the two local checks.
    Outside it, all bets are off and a fresh remote smoke is required.

    Whitespace-only differences outside the block do not count as an escape:
    a reformatter is not an environment change, and a check that fires
    spuriously is a check that gets argued around (§6).

    **The markers are not evidence about themselves.** Each side's "outside" was
    computed from its own markers, so a mutation that wrapped injected code --
    new imports, a file write, an environment change -- in a *fresh*
    `EVOLVE-BLOCK-START`/`END` pair moved that code into `inside` and left the
    two outsides identical, and the escape check reported no escape. An LLM
    mutation operator imitating the marker syntax it can see in its input is a
    realistic accident, not just an attack. So the marker structure itself has
    to match the baseline's before the outside comparison means anything.
    """
    base_starts, base_ends = baseline.count(BLOCK_START), baseline.count(BLOCK_END)
    cand_starts, cand_ends = candidate.count(BLOCK_START), candidate.count(BLOCK_END)

    if cand_starts != cand_ends:
        return {
            "escaped": True,
            "reason": (
                f"the candidate has {cand_starts} EVOLVE-BLOCK-START marker(s) and "
                f"{cand_ends} END marker(s); an unbalanced file has no well-defined "
                "mutable region"
            ),
            "requires": "smoke",
        }
    if (cand_starts, cand_ends) != (base_starts, base_ends):
        return {
            "escaped": True,
            "reason": (
                f"the mutation changed the number of EVOLVE-BLOCK regions "
                f"({base_starts} -> {cand_starts}); new markers can hide changed code "
                "from this check, so the region structure is fixed by the baseline"
            ),
            "requires": "smoke",
        }

    _, base_outside = split_blocks(baseline)
    _, cand_outside = split_blocks(candidate)

    def normalise(lines: list[str]) -> list[str]:
        return [line.rstrip() for line in lines if line.strip()]

    before, after = normalise(base_outside), normalise(cand_outside)
    if before == after:
        return {"escaped": False}
    changed = [line for line in after if line not in before][:20]
    return {
        "escaped": True,
        "reason": "the mutation changed code outside EVOLVE-BLOCK markers",
        "changed_lines": changed,
        "requires": "smoke",
    }


def has_markers(source: str) -> bool:
    return BLOCK_START in source and BLOCK_END in source


# ---------------------------------------------------------------------------
# campaigns
# ---------------------------------------------------------------------------
def campaign_events() -> list[dict[str, Any]]:
    return jsonl.read(campaigns_path())


def campaigns() -> dict[str, dict[str, Any]]:
    """Every campaign, folded from its events."""
    folded: dict[str, dict[str, Any]] = {}
    for rec in campaign_events():
        cid = rec.get("id")
        if not cid:
            continue
        kind = rec.get("type")
        if kind == T_CAMPAIGN:
            folded[cid] = {**{k: v for k, v in rec.items() if k != "type"}, "generations_run": 0}
        elif cid in folded and kind == T_GENERATION:
            node = folded[cid]
            # A record carrying `halted` documents the generation the loop
            # stopped *before*, not one it ran -- both the budget gate and a
            # halt request write one at the boundary they break on. Counting it
            # reported a campaign halted after generation 0 as having run two,
            # which is the number the evolve window puts in its title bar.
            if not rec.get("halted"):
                node["generations_run"] = max(
                    node["generations_run"], int(rec.get("generation", 0)) + 1
                )
            node["last_generation_at"] = rec.get("at")
            node.setdefault("generation_log", []).append(
                {k: v for k, v in rec.items() if k not in ("type", "id")}
            )
        elif cid in folded and kind == T_HALT_REQUESTED:
            folded[cid]["halt_requested"] = True
            folded[cid]["halt_requested_at"] = rec.get("at")
            folded[cid]["halt_reason"] = rec.get("reason")
        elif cid in folded and kind == T_CAMPAIGN_CLOSED:
            folded[cid]["status"] = rec.get("status", "closed")
            folded[cid]["closed_at"] = rec.get("at")
            folded[cid]["closed_reason"] = rec.get("reason")
    return folded


def campaign(campaign_id: str) -> dict[str, Any]:
    found = campaigns().get(campaign_id)
    if not found:
        raise NotFound(
            f"campaign {campaign_id!r} does not exist",
            fix="python -m tools.evolve status --json   # lists known campaigns",
        )
    return found


def append_campaign(record: dict[str, Any]) -> dict[str, Any]:
    return jsonl.append(campaigns_path(), record)


def record_generation(campaign_id: str, generation: int, **fields: Any) -> dict[str, Any]:
    return append_campaign(
        {"type": T_GENERATION, "id": campaign_id, "generation": generation, "at": now_iso(), **fields}
    )


def close_campaign(campaign_id: str, *, status: str = "closed", reason: str = "") -> dict[str, Any]:
    return append_campaign(
        {"type": T_CAMPAIGN_CLOSED, "id": campaign_id, "at": now_iso(),
         "status": status, "reason": reason}
    )


def request_halt(campaign_id: str, *, reason: str = "") -> dict[str, Any]:
    """Ask a running campaign to stop at the next generation boundary.

    A *request*, written to the ledger, rather than a signal: `evolve run` holds
    the loop in whatever process started it -- usually the agent's -- and the UI
    or a second terminal has no handle on it. An event both processes can see is
    the only mechanism that works across all three callers, and it has the
    side benefit of being diffable afterwards.

    The boundary matters as much as the request. Killing the loop mid-generation
    abandons an in-flight candidate, which goes stale and blocks every future
    submission (exit 7) -- so `_drive` checks this exactly where the budget gate
    already stops cleanly, with every candidate collected.

    The campaign must exist and still be open, and that is checked *inside the
    append lock* for the same reason `append_run_event` re-checks its binding
    there: the loop closing a campaign and a human asking it to halt are two
    processes racing over one file, and a check made before the lock is a check
    the other process can win. `tools.evolve halt` still checks first, because
    it produces the better message; this is the backstop, not the explanation.
    """

    def _still_open() -> None:
        record = campaigns().get(campaign_id)
        if record is None:
            raise NotFound(
                f"campaign {campaign_id!r} does not exist",
                fix="python -m tools.evolve status --json   # lists known campaigns",
            )
        status = record.get("status")
        if status != "open":
            raise GradError(
                "campaign_not_open",
                f"campaign {campaign_id!r} is {status}, so there is nothing to halt",
                exit_code=EXIT_USAGE,
                fix=f"python -m tools.evolve status --campaign {campaign_id} --json",
            )

    return jsonl.append(
        campaigns_path(),
        {"type": T_HALT_REQUESTED, "id": campaign_id, "at": now_iso(), "reason": reason},
        precondition=_still_open,
    )


def halt_requested(campaign_id: str) -> bool:
    record = campaigns().get(campaign_id) or {}
    return bool(record.get("halt_requested"))


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def candidate_events() -> list[dict[str, Any]]:
    return jsonl.read(candidates_path())


def append_candidate(record: dict[str, Any]) -> dict[str, Any]:
    return jsonl.append(candidates_path(), {"type": T_CANDIDATE, **record})


def candidates(campaign_id: str | None = None) -> list[dict[str, Any]]:
    """Candidate evaluations, oldest first.

    These live outside `runs.jsonl` on purpose (§23 item 4): they are sub-runs
    of a campaign, exempt from the per-run expectation gate, and a thousand of
    them would dominate a ledger meant to be read by hand.
    """
    promoted = {
        rec.get("candidate_id")
        for rec in candidate_events()
        if rec.get("type") == T_CANDIDATE_PROMOTED
    }
    out = []
    for rec in candidate_events():
        if rec.get("type") != T_CANDIDATE:
            continue
        if campaign_id and rec.get("campaign") != campaign_id:
            continue
        out.append({**rec, "promoted": rec.get("candidate_id") in promoted})
    return out


def promote_candidate(campaign_id: str, candidate_id: str, run_id: str) -> dict[str, Any]:
    return jsonl.append(
        candidates_path(),
        {
            "type": T_CANDIDATE_PROMOTED,
            "campaign": campaign_id,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "at": now_iso(),
        },
    )


def top_k(campaign_id: str, k: int = 5, *, metric: str = "combined_score") -> list[dict[str, Any]]:
    """The best K candidates, not the argmax.

    "`combined_score` is a Goodhart machine. A search optimising a scalar will
    find the bug in the metric." Surfacing a ranked list rather than a single
    winner is half the mitigation; the other half is that a winner still goes
    through the normal verdict path before it counts as a result.
    """
    scored = [
        c for c in candidates(campaign_id)
        if isinstance((c.get("metrics") or {}).get(metric), (int, float))
    ]
    scored.sort(key=lambda c: c["metrics"][metric], reverse=True)
    return scored[: max(1, k)]


def campaign_spend(campaign_id: str) -> dict[str, Any]:
    """What this campaign has actually consumed so far."""
    rows = candidates(campaign_id)
    return {
        "candidates": len(rows),
        "evaluated": sum(1 for r in rows if r.get("metrics")),
        "failed": sum(1 for r in rows if r.get("error")),
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 4),
        "wall_clock_s": round(sum(float(r.get("duration_s") or 0.0) for r in rows), 1),
    }


# ---------------------------------------------------------------------------
def validate_metrics(metrics: Any) -> str | None:
    """Shinka's contract: a metrics dict containing `combined_score`.

    Checked rather than assumed, because a candidate that silently reports no
    score is indistinguishable from one that scored zero, and the search would
    happily optimise toward whichever the fallback happened to be.
    """
    if not isinstance(metrics, dict):
        return "evaluate.py must print a JSON object of metrics"
    if "combined_score" not in metrics:
        return "the metrics object must contain `combined_score`"
    if not isinstance(metrics["combined_score"], (int, float)) or isinstance(
        metrics["combined_score"], bool
    ):
        return "`combined_score` must be a number"
    return None
