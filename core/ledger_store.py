"""Reading and writing the expectations and runs ledgers (HANDOFF §7).

    "Append-only JSONL is the source of truth [...] `ledger.sqlite` is a derived
     index [...] If they ever disagree, the JSONL wins."

Both files are event logs. A run's current state is the fold of its events:
`run_submitted` (written by the submitter, at submit time, so §6's spend ceiling
can count in-flight jobs), then `run_collected` (written by `collect`, never by
hand), then zero or more `verdict` events supplied by the model.

Nothing here interprets a result. `collect` computes deviations mechanically;
the verdict field is left unset on purpose, because that is the one part that
requires judgement, and judgement must not be able to overwrite the record.
"""

from __future__ import annotations

import datetime as _dt
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from core import jsonl, paths
from core.errors import NotFound

# --- record types -----------------------------------------------------------
T_EXPECTATION = "expectation"
T_EXPECTATION_FALSIFIED = "expectation_falsified"
T_RUN_SUBMITTED = "run_submitted"
T_RUN_COLLECTED = "run_collected"
T_VERDICT = "verdict"

VERDICTS = ("bug", "real", "inconclusive")
CONFIDENCES = ("low", "medium", "high")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def parse_iso(text: str | None) -> _dt.datetime | None:
    if not text:
        return None
    try:
        dt = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=_dt.timezone.utc)


def new_id(prefix: str) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


# ---------------------------------------------------------------------------
# expectations
# ---------------------------------------------------------------------------
def expectations() -> list[dict[str, Any]]:
    return [r for r in jsonl.read(paths.expectations_path()) if r.get("type") == T_EXPECTATION]


def expectation(expectation_id: str) -> dict[str, Any]:
    for rec in expectations():
        if rec.get("id") == expectation_id:
            return rec
    raise NotFound(
        f"expectation {expectation_id!r} does not exist",
        fix="grad-ledger expect --task ... --quantity ... --low ... --high ... --json",
    )


def falsified_ids() -> set[str]:
    return {
        r["id"]
        for r in jsonl.read(paths.expectations_path())
        if r.get("type") == T_EXPECTATION_FALSIFIED and r.get("id")
    }


def append_expectation(record: dict[str, Any]) -> dict[str, Any]:
    record = {"type": T_EXPECTATION, **record}
    return jsonl.append(paths.expectations_path(), record)


def append_expectation_event(record: dict[str, Any]) -> dict[str, Any]:
    return jsonl.append(paths.expectations_path(), record)


def tasks_with_results() -> set[str]:
    """Tasks that already have a collected run.

    `ledger.py expect` refuses to write against these. HANDOFF §7 is explicit
    that this is the weaker of the two lines of defence -- the real gate is
    binding-at-submit, below -- but it stays as cheap insurance.
    """
    bound: dict[str, str] = {}
    for rec in runs_events():
        if rec.get("type") == T_RUN_SUBMITTED and rec.get("task"):
            bound[rec["id"]] = rec["task"]
    out: set[str] = set()
    for rec in runs_events():
        if rec.get("type") == T_RUN_COLLECTED and rec.get("results"):
            task = bound.get(rec.get("id", ""))
            if task:
                out.add(task)
    return out


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def runs_events() -> list[dict[str, Any]]:
    return jsonl.read(paths.runs_path())


def append_run_event(record: dict[str, Any]) -> dict[str, Any]:
    """Append one run event.

    A `run_submitted` event binds an expectation, and the gate that checks the
    binding runs before the write -- so two submitters racing could both pass
    the check and both bind the same prediction. The uniqueness check is
    therefore repeated here, inside the append lock, where it is atomic with the
    write. `check_expectation` still runs first because it produces the better
    error message; this is the backstop, not the explanation.
    """
    expectation_id = record.get("expectation_id") if record.get("type") == T_RUN_SUBMITTED else None
    if not expectation_id:
        return jsonl.append(paths.runs_path(), record)

    def _still_unbound() -> None:
        if expectation_id in bound_expectation_ids():
            from core.errors import EXIT_EXPECTATION, GateRefusal

            raise GateRefusal(
                "expectation_bound",
                f"expectation {expectation_id!r} was bound to another run while this one was "
                "being submitted; each prediction covers exactly one run",
                EXIT_EXPECTATION,
                fix="mint a new expectation and resubmit: python -m tools.ledger expect ... --json",
            )

    return jsonl.append(paths.runs_path(), record, precondition=_still_unbound)


@dataclass
class Run:
    """The fold of one run's events."""

    id: str
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def status(self) -> str:
        return self.data.get("status", "unknown")

    @property
    def collected(self) -> bool:
        return bool(self.data.get("collected_at"))

    @property
    def is_smoke(self) -> bool:
        return bool(self.data.get("smoke"))

    @property
    def project(self) -> str:
        """HANDOFF-2 §15. Records written before the dimension existed fold as
        `unassigned`, which is what keeps this an additive schema change rather
        than a migration."""
        return str(self.data.get("project") or "unassigned")

    def cost_for_ceiling(self) -> float:
        """Actual once collected, estimate while in flight.

        "a job that has not been collected yet is not free. Without this, N jobs
         submitted before any is collected all pass the ceiling check."
        """
        if self.collected and self.data.get("cost_usd_actual") is not None:
            return float(self.data["cost_usd_actual"])
        return float(self.data.get("estimate_usd") or 0.0)

    def unjudged_deviations(self) -> list[dict[str, Any]]:
        """Anything not confirmed in range and not yet judged.

        `in_range` is False for a numeric miss and **None** for the cases no
        program can settle: a relational prediction, a non-numeric result, or a
        run that reported nothing for the predicted quantity. All of them need a
        verdict, and relational predictions are the kind §7 says to prefer -- so
        the predicate here is `is not True`, matching what `collect` surfaces.
        Filtering on `is False` would drop the preferred prediction type out of
        the pending list the moment the agent moved on.
        """
        return [
            d
            for d in self.data.get("deviations", [])
            if d.get("in_range") is not True and not d.get("verdict")
        ]


def runs() -> list[Run]:
    """All runs, oldest submission first, each folded from its events."""
    order: list[str] = []
    folded: dict[str, dict[str, Any]] = {}
    for rec in runs_events():
        run_id = rec.get("id")
        if not run_id:
            continue
        if run_id not in folded:
            folded[run_id] = {"id": run_id}
            order.append(run_id)
        node = folded[run_id]
        kind = rec.get("type")
        if kind == T_VERDICT:
            _apply_verdict(node, rec)
            continue
        for key, value in rec.items():
            if key in ("type",):
                continue
            node[key] = value
    return [Run(rid, folded[rid]) for rid in order]


def run(run_id: str) -> Run:
    for r in runs():
        if r.id == run_id:
            return r
    raise NotFound(
        f"run {run_id!r} does not exist",
        fix="grad-ledger query --runs --json  # to list known run ids",
    )


def _apply_verdict(node: dict[str, Any], rec: dict[str, Any]) -> None:
    deviations = node.setdefault("deviations", [])
    quantity = rec.get("quantity")
    for dev in deviations:
        if dev.get("quantity") == quantity:
            dev["verdict"] = rec.get("verdict")
            dev["note"] = rec.get("note")
            dev["judged_at"] = rec.get("judged_at")
            return
    deviations.append(
        {
            "quantity": quantity,
            "verdict": rec.get("verdict"),
            "note": rec.get("note"),
            "judged_at": rec.get("judged_at"),
            "orphan": True,  # a verdict for a quantity the run never reported
        }
    )


def bound_expectation_ids() -> set[str]:
    return {
        rec["expectation_id"]
        for rec in runs_events()
        if rec.get("type") == T_RUN_SUBMITTED and rec.get("expectation_id")
    }


def in_flight() -> list[Run]:
    return [r for r in runs() if not r.collected and r.status == "in_flight"]


def pending() -> dict[str, list[dict[str, Any]]]:
    """What `--pending` and the UI surface: uncollected runs and unjudged
    deviations, the two things that quietly accumulate otherwise."""
    uncollected = [
        {
            "run_id": r.id,
            "submitted_at": r.get("submitted_at"),
            "estimate_usd": r.get("estimate_usd"),
            "platform": r.get("platform"),
            "stale": is_stale(r),
        }
        for r in in_flight()
    ]
    unjudged = [
        {"run_id": r.id, **dev}
        for r in runs()
        for dev in r.unjudged_deviations()
    ]
    return {"uncollected_runs": uncollected, "unjudged_deviations": unjudged}


# ---------------------------------------------------------------------------
# staleness and spend (the numbers the §6 gates compare against)
# ---------------------------------------------------------------------------
def is_stale(r: Run, *, cfg: Any = None, now: _dt.datetime | None = None) -> bool:
    from core import config as _config

    cfg = cfg or _config.load()
    if r.collected or r.status != "in_flight":
        return False
    submitted = parse_iso(r.get("submitted_at"))
    if not submitted:
        return False
    grace_factor = float(cfg.get("spend", "stale_grace_factor", 3.0))
    floor = float(cfg.get("spend", "stale_grace_floor_s", 1800))
    estimated = float(r.get("estimated_duration_s") or 0.0)
    window = max(estimated * grace_factor, floor)
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return (now - submitted).total_seconds() > window


def stale_runs(*, cfg: Any = None, now: _dt.datetime | None = None) -> list[Run]:
    return [r for r in in_flight() if is_stale(r, cfg=cfg, now=now)]


def rolling_spend(
    window_days: int = 30, *, now: _dt.datetime | None = None, project: str | None = None
) -> dict[str, Any]:
    """Rolling total: actuals for collected runs, estimates for in-flight ones.

    `project` narrows it to one allocation (§15). The unfiltered total is still
    what the global ceiling compares against -- a project ceiling is an extra
    bound, never a replacement for the machine's.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(days=window_days)
    actual = 0.0
    estimated = 0.0
    counted: list[dict[str, Any]] = []
    for r in runs():
        if project and r.project != project:
            continue
        submitted = parse_iso(r.get("submitted_at"))
        if submitted and submitted < cutoff:
            continue
        amount = r.cost_for_ceiling()
        if r.collected and r.get("cost_usd_actual") is not None:
            actual += amount
            basis = "actual"
        else:
            estimated += amount
            basis = "estimate"
        counted.append(
            {"run_id": r.id, "usd": amount, "basis": basis, "smoke": r.is_smoke, "project": r.project}
        )
    return {
        "window_days": window_days,
        "project": project,
        "total_usd": round(actual + estimated, 4),
        "actual_usd": round(actual, 4),
        "in_flight_usd": round(estimated, 4),
        "runs": counted,
    }


# ---------------------------------------------------------------------------
# derived sqlite index (rebuildable; the JSONL wins on disagreement)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS expectations (
    id TEXT PRIMARY KEY, task TEXT, created_at TEXT, quantity TEXT, claim TEXT,
    low REAL, high REAL, direction TEXT, comparability TEXT, confidence TEXT,
    falsified INTEGER DEFAULT 0, basis_json TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, task TEXT, status TEXT, smoke INTEGER,
    submitted_at TEXT, collected_at TEXT, platform TEXT, target TEXT,
    submission_hash TEXT, expectation_id TEXT,
    estimate_usd REAL, cost_usd_actual REAL, results_json TEXT,
    project TEXT
);
CREATE TABLE IF NOT EXISTS deviations (
    run_id TEXT, expectation_id TEXT, quantity TEXT,
    expected_low REAL, expected_high REAL, actual REAL, ratio REAL,
    in_range INTEGER, verdict TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE INDEX IF NOT EXISTS idx_dev_quantity ON deviations(quantity);
CREATE INDEX IF NOT EXISTS idx_exp_quantity ON expectations(quantity);
"""


def rebuild_index(db_path: Any = None) -> dict[str, int]:
    db_path = db_path or paths.ledger_sqlite()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        falsified = falsified_ids()
        exps = expectations()
        for e in exps:
            pred = e.get("predicted") or {}
            con.execute(
                "INSERT OR REPLACE INTO expectations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    e.get("id"), e.get("task"), e.get("created_at"), e.get("quantity"),
                    e.get("claim"), pred.get("low"), pred.get("high"), pred.get("direction"),
                    e.get("comparability"), e.get("confidence"),
                    1 if e.get("id") in falsified else 0,
                    _dumps(e.get("basis")),
                ),
            )
        rs = runs()
        for r in rs:
            con.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r.id, r.get("task"), r.status, 1 if r.is_smoke else 0,
                    r.get("submitted_at"), r.get("collected_at"), r.get("platform"),
                    _dumps(r.get("target")), r.get("submission_hash"), r.get("expectation_id"),
                    r.get("estimate_usd"), r.get("cost_usd_actual"), _dumps(r.get("results")),
                    r.project,
                ),
            )
            for dev in r.get("deviations", []) or []:
                expected = dev.get("expected") or {}
                con.execute(
                    "INSERT INTO deviations VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        r.id, dev.get("expectation_id"), dev.get("quantity"),
                        expected.get("low"), expected.get("high"), dev.get("actual"),
                        dev.get("ratio"),
                        # Tri-state on purpose: NULL means "no program can settle
                        # this" (relational prediction, non-numeric result,
                        # missing quantity). Collapsing it to 0 would make the
                        # index unable to separate "out of range" from "needs a
                        # verdict", which the JSONL keeps distinct.
                        None if dev.get("in_range") is None else (1 if dev["in_range"] else 0),
                        dev.get("verdict"), dev.get("note"),
                    ),
                )
        con.commit()
        return {"expectations": len(exps), "runs": len(rs)}
    finally:
        con.close()


def _dumps(obj: Any) -> str | None:
    import json

    return None if obj is None else json.dumps(obj, ensure_ascii=False, default=str)


def query_index(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    db = paths.ledger_sqlite()
    if not db.exists():
        rebuild_index(db)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(sql, tuple(params))]
    finally:
        con.close()
