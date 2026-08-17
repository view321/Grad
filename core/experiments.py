"""The cross-workspace experiment archive.

Every other ledger in this system is scoped to one workspace, which is right:
`report check` must never resolve a claim against someone else's runs, and
`paths.py` says so in its docstring. But that scoping leaves one question with
no answer anywhere -- *have I run this before?* -- and it is a question that gets
asked across the two or three workspaces a person accumulates and across the
projects inside them.

So this is the one deliberately global store, and it lives under
`core/appdata.py`'s directory rather than under a root.

**It is a copy, not a second source of truth.** A workspace's `runs.jsonl`
remains authoritative for its own runs. This is a snapshot taken at the moment a
run becomes terminal, so that deleting a workspace loses the working files and
not the record that the experiment happened. Where the two disagree, the
workspace wins, exactly as the JSONL wins over `ledger.sqlite`.

Three decisions worth stating, because each is a place this could have been
dishonest:

* **Snapshots, folded by id.** `archive()` appends a whole record rather than
  patching one. A verdict supplied an hour after collection appends a second
  snapshot and the fold takes the newest. There is no partial update to get
  wrong, and the history of what was believed when is preserved for free.
* **Artifacts by reference, hashed.** The files stay in their workspace; this
  records path, size and SHA-256. `verify()` re-hashes them, so an artifact that
  was moved, truncated or deleted is *detectable* rather than silently missing.
  Copying them here instead would double the disk and create a second copy free
  to diverge from the one the report cites.
* **Tokens are not attributed per run, and this does not pretend otherwise.**
  `ledger/quota.jsonl` is tagged by stage and project, never by run -- a
  conversation that submits three jobs spends its tokens on the conversation. So
  the record carries the project's *running* weighted total at the moment of
  archiving, named for what it is. The difference between two consecutive
  experiments in one project is meaningful; the number on any one of them is
  not that experiment's cost.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from core import appdata, jsonl, ledger_store as ls, paths
from core.errors import NotFound

T_EXPERIMENT = "experiment"

#: Files larger than this are recorded with their size but no digest. Hashing a
#: multi-gigabyte checkpoint on the `collect` path would turn a bookkeeping step
#: into a visible stall, and the size alone still catches the common corruptions.
#: Overridable through `[experiments] hash_max_bytes`.
HASH_MAX_BYTES = 64 * 1024 * 1024


def archive_dir() -> Path:
    return appdata.experiments_dir()


def archive_path() -> Path:
    return archive_dir() / "experiments.jsonl"


def index_path() -> Path:
    return archive_dir() / "experiments.sqlite"


def experiment_id(workspace_key: str, run_id: str) -> str:
    """Stable across re-archiving, unique across workspaces.

    Keyed on the workspace rather than on the run id alone because run ids carry
    a timestamp and six random hex characters -- collision-resistant within a
    ledger, and not something to bet a global store on.
    """
    return f"{workspace_key}:{run_id}"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def archive(run_id: str, *, cfg: Any = None, root: Path | None = None) -> dict[str, Any]:
    """Snapshot one run into the global archive.

    Called by `core/submit.py:finish` when a run becomes terminal and again by
    `ledger verdict` once judgement lands. Safe to call any number of times: it
    appends, and the fold takes the newest.
    """
    root = (root or paths.root()).resolve()
    run = ls.run(run_id)
    record = snapshot(run, cfg=cfg, root=root)
    jsonl.append(archive_path(), record)
    return record


def snapshot(run: ls.Run, *, cfg: Any = None, root: Path | None = None) -> dict[str, Any]:
    """The record `archive` writes. Pure apart from hashing the artifacts."""
    root = (root or paths.root()).resolve()
    key = appdata.workspace_key(root)
    deviations = run.get("deviations") or []
    expectation = _expectation_of(run)
    return {
        "type": T_EXPERIMENT,
        "experiment_id": experiment_id(key, run.id),
        "archived_at": ls.now_iso(),
        "workspace": str(root),
        "workspace_key": key,
        "project": run.project,
        "run_id": run.id,
        "task": run.get("task"),
        "platform": run.get("platform"),
        "target": run.get("target"),
        "status": run.status,
        "smoke": run.is_smoke,
        "submitted_at": run.get("submitted_at"),
        "collected_at": run.get("collected_at"),
        "submission_hash": run.get("submission_hash"),
        # The pre-image of the hash above, when the submitter recorded one.
        # `verify()` re-derives the hash from it, which is what makes the archive
        # able to answer "is this the spec that produced that number" without the
        # workspace being present.
        "spec_resolved": run.get("spec_resolved"),
        "spec_path": run.get("spec"),
        "command": run.get("command"),
        "image": run.get("image"),
        "expectation_id": run.get("expectation_id"),
        "expectation": expectation,
        "results": run.get("results") or {},
        "deviations": deviations,
        "all_judged": not run.unjudged_deviations(),
        "code_version": run.get("code_version"),
        "estimate_usd": run.get("estimate_usd"),
        "cost_usd_actual": run.get("cost_usd_actual"),
        "accelerator_hours_actual": run.get("accelerator_hours_actual"),
        "accelerator": run.get("accelerator"),
        "error": run.get("error"),
        "artifacts": artifact_manifest(run, cfg=cfg),
        # See the module docstring: a project-level running total, not this
        # run's cost. Named so that nothing can read it as the latter.
        "project_quota_tokens_running": _project_quota(run.project),
    }


def _expectation_of(run: ls.Run) -> dict[str, Any] | None:
    """The bound prediction, copied in full.

    Copied rather than referenced because the archive has to stand alone: an
    expectation id means nothing once the workspace holding
    `expectations.jsonl` is gone, and the prediction is half of what makes a
    result interpretable.
    """
    bound = run.get("expectation_id")
    if not bound:
        return None
    try:
        return ls.expectation(str(bound))
    except NotFound:
        return None


def _project_quota(project: str) -> float | None:
    """The project's weighted token total right now, or None if unreadable.

    Never raises: this is on the `collect` path and accounting must not be the
    reason a run fails to be recorded.
    """
    try:
        from core import quota_log  # noqa: PLC0415

        weight = quota_log.weights()
        return round(
            sum(
                quota_log.billable(row, weight)
                for row in quota_log.entries()
                if (row.get("project") or "unassigned") == project
            )
        )
    except Exception:  # noqa: BLE001 - see the docstring
        return None


def artifact_manifest(run: ls.Run, *, cfg: Any = None) -> list[dict[str, Any]]:
    """Every file under the run's artifact directory, with a digest.

    Sorted by path so two archivings of an unchanged directory produce identical
    manifests -- which is what lets `verify` compare them at all.
    """
    directory = run.get("artifacts")
    if not directory:
        return []
    base = Path(str(directory))
    if not base.is_dir():
        return []
    limit = _hash_limit(cfg)
    out: list[dict[str, Any]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entry: dict[str, Any] = {
            "path": str(path),
            "relative": path.relative_to(base).as_posix(),
            "bytes": size,
        }
        if size > limit:
            entry["sha256"] = None
            entry["skipped"] = f"larger than hash_max_bytes ({limit:,})"
        else:
            entry["sha256"] = sha256_of(path)
        out.append(entry)
    return out


def _hash_limit(cfg: Any = None) -> int:
    if cfg is None:
        try:
            from core import config as config_mod  # noqa: PLC0415

            cfg = config_mod.load()
        except Exception:  # noqa: BLE001
            return HASH_MAX_BYTES
    try:
        return max(0, int(cfg.get("experiments", "hash_max_bytes", HASH_MAX_BYTES)))
    except (TypeError, ValueError):
        return HASH_MAX_BYTES


def sha256_of(path: Path) -> str | None:
    """Streamed, so a large artifact does not have to fit in memory."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def events() -> list[dict[str, Any]]:
    return jsonl.read(archive_path())


def experiments() -> dict[str, dict[str, Any]]:
    """Every experiment, newest snapshot per id, insertion-ordered."""
    folded: dict[str, dict[str, Any]] = {}
    for record in events():
        eid = record.get("experiment_id")
        if not eid or record.get("type") != T_EXPERIMENT:
            continue
        # Replace rather than merge: each record is a whole snapshot, and
        # merging would let a field from an older one survive a newer snapshot
        # that deliberately dropped it.
        folded[str(eid)] = record
    return folded


def get(identifier: str) -> dict[str, Any]:
    """One experiment by experiment id, or by run id when it is unambiguous."""
    everything = experiments()
    if identifier in everything:
        return everything[identifier]
    matches = [e for e in everything.values() if e.get("run_id") == identifier]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise NotFound(
            f"no experiment {identifier!r} in the archive",
            fix="python -m tools.experiments list --json",
        )
    raise NotFound(
        f"run id {identifier!r} appears in {len(matches)} workspaces; name the experiment id",
        fix="python -m tools.experiments list --json   # the experiment_id column",
    )


def search(
    *,
    project: str | None = None,
    workspace: str | None = None,
    task: str | None = None,
    quantity: str | None = None,
    include_smoke: bool = True,
    judged_only: bool = False,
) -> list[dict[str, Any]]:
    """Filtered experiments, newest first.

    Plain Python over the folded records rather than SQL. The index exists for
    joins a person types by hand; the filters a tool needs are four equality
    tests over a few thousand rows, and doing them here keeps the JSONL
    authoritative for every path this module answers on.
    """
    rows = list(experiments().values())
    if project:
        rows = [r for r in rows if r.get("project") == project]
    if workspace:
        rows = [
            r
            for r in rows
            if workspace in (r.get("workspace") or "") or r.get("workspace_key") == workspace
        ]
    if task:
        rows = [r for r in rows if r.get("task") == task]
    if quantity:
        rows = [r for r in rows if quantity in (r.get("results") or {})]
    if not include_smoke:
        rows = [r for r in rows if not r.get("smoke")]
    if judged_only:
        rows = [r for r in rows if r.get("all_judged")]
    rows.sort(key=lambda r: str(r.get("submitted_at") or r.get("archived_at") or ""), reverse=True)
    return rows


def verify(identifier: str | None = None) -> dict[str, Any]:
    """Re-hash artifacts and re-derive submission hashes. Reports; changes nothing.

    Two independent checks, and they fail for different reasons:

    * an artifact whose digest no longer matches has been edited or replaced
      since it was archived, and a figure in a report that cites it is no longer
      showing what was measured;
    * a `spec_resolved` that does not hash to its recorded `submission_hash`
      means the archive's copy of the spec is not the one that was submitted,
      which is the only integrity property this store can check entirely on its
      own.
    """
    rows = [get(identifier)] if identifier else list(experiments().values())
    findings: list[dict[str, Any]] = []
    checked_files = 0
    for row in rows:
        eid = row.get("experiment_id")
        for entry in row.get("artifacts") or []:
            recorded = entry.get("sha256")
            if not recorded:
                continue
            checked_files += 1
            path = Path(str(entry.get("path")))
            if not path.is_file():
                findings.append(
                    {"experiment_id": eid, "kind": "artifact_missing", "path": str(path)}
                )
                continue
            if sha256_of(path) != recorded:
                findings.append(
                    {"experiment_id": eid, "kind": "artifact_changed", "path": str(path)}
                )
        finding = _verify_hash(row)
        if finding:
            findings.append(finding)
    return {
        "experiments": len(rows),
        "files_checked": checked_files,
        "findings": findings,
        "ok": not findings,
    }


def _verify_hash(row: dict[str, Any]) -> dict[str, Any] | None:
    resolved, recorded = row.get("spec_resolved"), row.get("submission_hash")
    if not resolved or not recorded:
        return None
    try:
        from core.submission import hash_resolved  # noqa: PLC0415

        derived = hash_resolved(resolved)
    except Exception:  # noqa: BLE001 - an unhashable snapshot is a finding, not a crash
        return {"experiment_id": row.get("experiment_id"), "kind": "spec_unhashable"}
    if derived != recorded:
        return {
            "experiment_id": row.get("experiment_id"),
            "kind": "spec_hash_mismatch",
            "recorded": recorded,
            "derived": derived,
        }
    return None


def summary() -> dict[str, Any]:
    """What the archive holds, by workspace and by project."""
    rows = list(experiments().values())
    by_workspace: dict[str, int] = {}
    by_project: dict[str, int] = {}
    for row in rows:
        by_workspace[str(row.get("workspace") or "?")] = (
            by_workspace.get(str(row.get("workspace") or "?"), 0) + 1
        )
        by_project[str(row.get("project") or "unassigned")] = (
            by_project.get(str(row.get("project") or "unassigned"), 0) + 1
        )
    collected = [r for r in rows if r.get("collected_at")]
    return {
        "path": str(archive_path()),
        "experiments": len(rows),
        "collected": len(collected),
        "smoke": sum(1 for r in rows if r.get("smoke")),
        "judged": sum(1 for r in rows if r.get("all_judged")),
        "by_workspace": dict(sorted(by_workspace.items())),
        "by_project": dict(sorted(by_project.items())),
        "usd": round(sum(float(r.get("cost_usd_actual") or 0.0) for r in rows), 4),
        "accelerator_hours": round(
            sum(float(r.get("accelerator_hours_actual") or 0.0) for r in rows), 4
        ),
    }


# ---------------------------------------------------------------------------
# the derived index
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY, archived_at TEXT, workspace TEXT, workspace_key TEXT,
    project TEXT, run_id TEXT, task TEXT, platform TEXT, accelerator TEXT,
    status TEXT, smoke INTEGER, all_judged INTEGER,
    submitted_at TEXT, collected_at TEXT, submission_hash TEXT, expectation_id TEXT,
    estimate_usd REAL, cost_usd_actual REAL, accelerator_hours REAL,
    code_version TEXT, results_json TEXT, spec_json TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id TEXT, quantity TEXT, value REAL, text_value TEXT
);
CREATE TABLE IF NOT EXISTS deviations (
    experiment_id TEXT, quantity TEXT, expected_low REAL, expected_high REAL,
    actual REAL, in_range INTEGER, verdict TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    experiment_id TEXT, path TEXT, relative TEXT, bytes INTEGER, sha256 TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_project ON experiments(project);
CREATE INDEX IF NOT EXISTS idx_exp_task ON experiments(task);
CREATE INDEX IF NOT EXISTS idx_metrics_quantity ON metrics(quantity);
CREATE INDEX IF NOT EXISTS idx_dev_quantity ON deviations(quantity);
"""


def rebuild_index(db_path: Path | None = None) -> dict[str, int]:
    """Rebuild `experiments.sqlite` from the JSONL. Always safe to run.

    Dropped and rebuilt rather than updated, for the reason
    `ledger_store.rebuild_index` gives: it is derived, it is small, and an index
    that can only be repaired by incremental patching is one that stays wrong.
    """
    db_path = db_path or index_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    rows = list(experiments().values())
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        metrics = deviations = artifacts = 0
        for row in rows:
            eid = row.get("experiment_id")
            con.execute(
                "INSERT OR REPLACE INTO experiments VALUES (" + ",".join("?" * 22) + ")",
                (
                    eid, row.get("archived_at"), row.get("workspace"), row.get("workspace_key"),
                    row.get("project"), row.get("run_id"), row.get("task"), row.get("platform"),
                    row.get("accelerator"), row.get("status"), 1 if row.get("smoke") else 0,
                    1 if row.get("all_judged") else 0, row.get("submitted_at"),
                    row.get("collected_at"), row.get("submission_hash"), row.get("expectation_id"),
                    row.get("estimate_usd"), row.get("cost_usd_actual"),
                    row.get("accelerator_hours_actual"),
                    _dumps(row.get("code_version")), _dumps(row.get("results")),
                    _dumps(row.get("spec_resolved")),
                ),
            )
            for quantity, value in (row.get("results") or {}).items():
                numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                con.execute(
                    "INSERT INTO metrics VALUES (?,?,?,?)",
                    (eid, quantity, numeric, None if numeric is not None else str(value)),
                )
                metrics += 1
            for dev in row.get("deviations") or []:
                expected = dev.get("expected") or {}
                con.execute(
                    "INSERT INTO deviations VALUES (?,?,?,?,?,?,?,?)",
                    (
                        eid, dev.get("quantity"), expected.get("low"), expected.get("high"),
                        dev.get("actual"),
                        # Tri-state preserved, as in the workspace index: NULL is
                        # "no program can settle this", which is a different fact
                        # from "out of range".
                        None if dev.get("in_range") is None else (1 if dev["in_range"] else 0),
                        dev.get("verdict"), dev.get("note"),
                    ),
                )
                deviations += 1
            for entry in row.get("artifacts") or []:
                con.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?,?)",
                    (eid, entry.get("path"), entry.get("relative"),
                     entry.get("bytes"), entry.get("sha256")),
                )
                artifacts += 1
        con.commit()
        return {
            "experiments": len(rows), "metrics": metrics,
            "deviations": deviations, "artifacts": artifacts,
        }
    finally:
        con.close()


def _dumps(obj: Any) -> str | None:
    import json  # noqa: PLC0415

    return None if obj is None else json.dumps(obj, ensure_ascii=False, default=str)


def _older_than(derived: Path, source: Path) -> bool:
    """Is `derived` behind `source`? False when either cannot be stated.

    An unreadable mtime is not a reason to rebuild: the rebuild would fail on
    the same file, and answering from the index that exists beats refusing.
    """
    try:
        return source.exists() and derived.stat().st_mtime < source.stat().st_mtime
    except OSError:
        return False


def query_index(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    db = index_path()
    # Rebuilt when the index is missing *or* older than the JSONL it indexes.
    # `experiments.jsonl` is the authoritative store and the SQLite file is a
    # derived view of it, so an archive appended by another workspace since this
    # one last built the index is a query answered from a stale copy -- silently,
    # and with the newest experiments being exactly the ones missing from it.
    if not db.exists() or _older_than(db, archive_path()):
        rebuild_index(db)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(sql, tuple(params))]
    finally:
        con.close()
