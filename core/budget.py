"""The project dimension and its ceilings (HANDOFF-2 §15).

    "Three separate requirements turned out to want the same thing -- a
     dimension carried on every cost-bearing record."

§17 needs to know which account paid for an HF job, §21 needs to bound a
campaign made of many runs, and the user wants a budget for a piece of research.
Those are three faces of one abstraction, so it is built once: every run record,
every `quota.jsonl` entry, and every credit spend carries a `project` id.

Three resources, and the enforcement quality differs by resource in a way that
is structural rather than a shortcoming:

  * **GPU dollars** -- clean. Submission is a discrete, gateable event.
  * **Credits** -- clean at the same boundary, and measured continuously.
  * **Subscription tokens** -- granular to *one turn's overrun*. Tokens are
    consumed continuously inside a turn and there is no way to refuse mid-turn,
    so `agent.py` checks remaining allocation before issuing the next turn and
    `hooks.py` denies cost-bearing Bash once a project is over. That is the
    honest statement, and it belongs in `--help`, not only in the handoff.

A second honesty note, kept next to the code rather than in a document: the real
Anthropic limits are rolling windows the SDK does not expose as a remaining
balance. A token ceiling here is **a proxy the user controls**, not a mirror of
Anthropic's limit. The meter must not imply otherwise.

`raise` appends an event rather than mutating the record: a ceiling that can be
edited invisibly is not a ceiling. Same argument as §7's append-only ledger.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any

from core import jsonl, paths
from core.errors import EXIT_PROJECT_BUDGET, GateRefusal, NotFound, UsageError

# --- record types (folded like runs.jsonl) ----------------------------------
T_PROJECT = "project"
T_PROJECT_RAISED = "project_budget_raised"
T_PROJECT_CLOSED = "project_closed"

# Records that predate the dimension fold as this, per §23 item 6: specified as
# left alone rather than retrofitted, and cheap to change while the ledger is
# small.
UNASSIGNED = "unassigned"

# The three resources. Names are the keys in a project's `budget` table and in
# every spend report, so a caller never has to guess which spelling is current.
RESOURCES = ("gpu_usd", "quota_tokens", "credits_usd")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def projects_path() -> Path:
    return paths.ledger_dir() / "projects.jsonl"


def current_project_path() -> Path:
    return paths.ledger_dir() / ".current_project"


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def current_project() -> str | None:
    """The project selected by `tools.budget use`.

    Deliberately a file rather than an environment variable:
    `credentials.scrub_environment()` strips the agent's environment at startup,
    and a selection mechanism that the agent's own startup deletes is a bug
    waiting to happen.
    """
    path = current_project_path()
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def set_current(project_id: str | None) -> None:
    path = current_project_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if project_id is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(project_id, encoding="utf-8")


def resolve(explicit: str | None = None) -> str | None:
    """`--project` beats the selection file. Neither is an error: work outside
    a project is allowed and lands under `unassigned`."""
    return explicit or current_project()


def resolve_or_fail(explicit: str | None, *, what: str) -> str:
    """For the commands that genuinely cannot proceed without one."""
    project_id = resolve(explicit)
    if not project_id:
        raise UsageError(
            f"{what} needs a project: it is what the budget is charged against",
            fix="python -m tools.budget use <id> --json   # or pass --project <id>",
        )
    return project_id


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def events() -> list[dict[str, Any]]:
    return jsonl.read(projects_path())


def projects() -> dict[str, dict[str, Any]]:
    """Every project, folded from its events, insertion-ordered by creation."""
    folded: dict[str, dict[str, Any]] = {}
    for rec in events():
        pid = rec.get("id")
        if not pid:
            continue
        kind = rec.get("type")
        if kind == T_PROJECT:
            folded[pid] = {
                "id": pid,
                "created_at": rec.get("created_at"),
                "title": rec.get("title", ""),
                "payer": rec.get("payer"),
                "budget": dict(rec.get("budget") or {}),
                "status": rec.get("status", "open"),
                "raises": [],
            }
        elif pid in folded and kind == T_PROJECT_RAISED:
            node = folded[pid]
            for resource, value in (rec.get("budget") or {}).items():
                node["budget"][resource] = value
            node["raises"].append(
                {"at": rec.get("at"), "budget": rec.get("budget"), "reason": rec.get("reason")}
            )
        elif pid in folded and kind == T_PROJECT_CLOSED:
            folded[pid]["status"] = "closed"
            folded[pid]["closed_at"] = rec.get("at")
    return folded


def project(project_id: str) -> dict[str, Any]:
    found = projects().get(project_id)
    if not found:
        raise NotFound(
            f"project {project_id!r} does not exist",
            fix="python -m tools.budget list --json   # or `new` to create one",
        )
    return found


def exists(project_id: str | None) -> bool:
    return bool(project_id) and project_id in projects()


def create(
    project_id: str,
    *,
    title: str,
    budget: dict[str, float],
    payer: str | None = None,
) -> dict[str, Any]:
    if not _ID_RE.match(project_id):
        raise UsageError(
            f"project id {project_id!r} must be a short slug of letters, digits, dot, dash, underscore",
            fix="python -m tools.budget new --id proj-scaling-w2 --title '...' --json",
        )
    if project_id in projects():
        raise UsageError(
            f"project {project_id!r} already exists",
            fix=f"python -m tools.budget status --project {project_id} --json",
        )
    record = {
        "type": T_PROJECT,
        "id": project_id,
        "created_at": now_iso(),
        "title": title,
        "payer": payer,
        "budget": {k: float(v) for k, v in budget.items() if v is not None},
        "status": "open",
    }
    jsonl.append(projects_path(), record)
    return record


def raise_ceiling(project_id: str, *, budget: dict[str, float], reason: str = "") -> dict[str, Any]:
    """Append a raise event. Never mutates the original record.

    Lowering is permitted and recorded the same way -- the point is not that a
    ceiling only goes up, it is that it never moves invisibly.
    """
    current = project(project_id)
    changed = {k: float(v) for k, v in budget.items() if v is not None}
    if not changed:
        raise UsageError(
            "nothing to change",
            fix="python -m tools.budget raise --gpu-usd 75 --json",
        )
    unknown = [k for k in changed if k not in RESOURCES]
    if unknown:
        raise UsageError(
            f"unknown resource(s): {', '.join(unknown)}",
            fix=f"resources are: {', '.join(RESOURCES)}",
        )
    record = {
        "type": T_PROJECT_RAISED,
        "id": project_id,
        "at": now_iso(),
        "budget": changed,
        "previous": {k: current["budget"].get(k) for k in changed},
        "reason": reason,
    }
    jsonl.append(projects_path(), record)
    return record


def close(project_id: str) -> dict[str, Any]:
    project(project_id)
    record = {"type": T_PROJECT_CLOSED, "id": project_id, "at": now_iso()}
    jsonl.append(projects_path(), record)
    if current_project() == project_id:
        set_current(None)
    return record


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# spend
# ---------------------------------------------------------------------------
def project_of(record: Any) -> str:
    """The project a cost-bearing record belongs to.

    Unknown-project records fold as `unassigned` so existing ledgers keep
    loading -- this is an additive schema change, not a migration.
    """
    get = record.get if hasattr(record, "get") else (lambda k, d=None: d)
    return str(get("project") or UNASSIGNED)


def spend(project_id: str) -> dict[str, Any]:
    """What this project has consumed, per resource.

    GPU dollars count in-flight runs at their estimates, for the same reason
    §6's global ceiling does: a job that has not been collected yet is not free.
    """
    from core import ledger_store as ls, quota_log

    gpu_usd = 0.0
    in_flight_usd = 0.0
    runs: list[str] = []
    for r in ls.runs():
        if project_of(r.data) != project_id:
            continue
        runs.append(r.id)
        amount = r.cost_for_ceiling()
        gpu_usd += amount
        if not (r.collected and r.get("cost_usd_actual") is not None):
            in_flight_usd += amount

    quota_tokens = 0
    credits_usd = 0.0
    for entry in quota_log.entries():
        if project_of(entry) != project_id:
            continue
        quota_tokens += int(entry.get("input_tokens", 0) or 0) + int(entry.get("output_tokens", 0) or 0)
        credits_usd += float(entry.get("credits_usd", 0.0) or 0.0)

    # Campaign candidates consume real resources and live outside runs.jsonl by
    # design (§23 item 4). Leaving them out here would make a campaign invisible
    # to the ceiling that is supposed to bound it -- the re-check before each
    # generation would compare against a spend figure that never moved, and the
    # gate would be decoration.
    candidate_usd = 0.0
    candidates = 0
    try:
        from core import campaign as _campaign  # noqa: PLC0415 - avoids an import cycle

        owned = {
            cid for cid, c in _campaign.campaigns().items()
            if (c.get("project") or UNASSIGNED) == project_id
        }
        for row in _campaign.candidates():
            if row.get("campaign") in owned:
                candidate_usd += float(row.get("cost_usd") or 0.0)
                candidates += 1
    except Exception:  # noqa: BLE001 - a missing campaign ledger is not an error
        pass

    return {
        "project": project_id,
        "gpu_usd": round(gpu_usd + candidate_usd, 4),
        "gpu_in_flight_usd": round(in_flight_usd, 4),
        "gpu_candidate_usd": round(candidate_usd, 4),
        "quota_tokens": quota_tokens,
        "credits_usd": round(credits_usd, 6),
        "runs": runs,
        "candidates": candidates,
    }


def status(project_id: str) -> dict[str, Any]:
    """Ceilings, spend, and remaining, per resource.

    A resource with no ceiling reports `remaining: null` rather than infinity:
    "unbounded" and "a very large number" should not look the same in a meter.
    """
    proj = project(project_id)
    used = spend(project_id)
    resources: dict[str, Any] = {}
    for resource in RESOURCES:
        ceiling = proj["budget"].get(resource)
        consumed = used[resource]
        resources[resource] = {
            "ceiling": ceiling,
            "spent": consumed,
            "remaining": None if ceiling is None else round(float(ceiling) - consumed, 6),
            "fraction": (
                None if not ceiling else min(1.0, consumed / float(ceiling))
            ),
            "over": bool(ceiling is not None and consumed > float(ceiling)),
        }
    return {
        "project": proj["id"],
        "title": proj["title"],
        "payer": proj["payer"],
        "status": proj["status"],
        "resources": resources,
        "gpu_in_flight_usd": used["gpu_in_flight_usd"],
        "run_count": len(used["runs"]),
        "over_budget": [r for r, d in resources.items() if d["over"]],
        # The meter must not imply a fuel gauge. Anthropic exposes no remaining
        # balance and the real limits are rolling windows, so a token ceiling is
        # a proxy the user controls.
        "quota_tokens_authoritative": False,
    }


def over_budget(project_id: str | None) -> list[str]:
    """Which resources are over. Empty for no project and for unknown ids --
    a missing project is not an overrun, and the caller decides whether it is
    an error."""
    if not project_id or not exists(project_id):
        return []
    return status(project_id)["over_budget"]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def check(
    project_id: str | None,
    *,
    gpu_usd: float = 0.0,
    quota_tokens: int = 0,
    credits_usd: float = 0.0,
    what: str = "this",
) -> dict[str, Any] | None:
    """Refuse if the projected spend leaves the project over its allocation.

    Exit code **12**, distinct from 6 (the global spend ceiling), so "this
    research ran out of its allocation" is never confused with "the machine is
    out of money". Returns None when there is no project or no ceiling: an
    unbudgeted project is still tracked, just not bounded.
    """
    if not project_id or not exists(project_id):
        return None
    state = status(project_id)
    proposed = {"gpu_usd": float(gpu_usd), "quota_tokens": int(quota_tokens), "credits_usd": float(credits_usd)}
    for resource, add in proposed.items():
        node = state["resources"][resource]
        ceiling = node["ceiling"]
        if ceiling is None:
            continue
        projected = node["spent"] + add
        if projected > float(ceiling):
            raise GateRefusal(
                "project_budget",
                (
                    f"project {project_id!r} would spend {_fmt(resource, projected)} of "
                    f"{_fmt(resource, float(ceiling))} on {resource} "
                    f"({_fmt(resource, node['spent'])} already spent"
                    + (f" + {_fmt(resource, add)} for {what}" if add else "")
                    + "); this research has run out of its allocation, "
                    "which is not the same as the machine being out of money"
                ),
                EXIT_PROJECT_BUDGET,
                fix=(
                    f"python -m tools.budget raise --project {project_id} "
                    f"--{resource.replace('_', '-')} <new ceiling> --json   "
                    "# deliberate, logged, never silent"
                ),
                detail={"project": project_id, "resource": resource,
                        "spent": node["spent"], "ceiling": ceiling, "proposed": add},
            )
    return state


def _fmt(resource: str, value: float) -> str:
    if resource.endswith("_usd"):
        return f"${value:.2f}"
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# payer (§17)
# ---------------------------------------------------------------------------
def hf_namespace(project_id: str | None) -> str | None:
    """The HF namespace a project's costs are attributed to.

    `payer` lives on the project rather than being invented per submission, so
    the org attribution in §17 is a consequence of choosing a project rather
    than a separate flag to forget.
    """
    if not project_id or not exists(project_id):
        return None
    payer = project(project_id).get("payer") or ""
    if payer.startswith("hf:"):
        return payer[3:] or None
    return None
