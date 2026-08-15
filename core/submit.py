"""Shared submitter machinery for `jobs.py` (HF Jobs) and `gpu.py` (SSH).

The two submitters differ only in how they reach a machine. Everything that
makes them *gates* -- refusing without a preflight, binding an expectation,
writing the in-flight run record at submit time, computing deviations
mechanically at collect time -- is here, so neither backend can quietly grow a
bypass the other does not have.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from core import budget, gates, ledger_store as ls, paths, version
from core.config import Config
from core.errors import EXIT_RUNNING, EXIT_USAGE, GradError
from core.submission import Submission


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
def check(
    sub: Submission, expectation_id: str | None, cfg: Config, *, project: str | None = None
) -> dict[str, Any]:
    """Run the gates. Raises `GateRefusal` on the first that refuses.

    Called before the backend is even resolved, so a refusal is always the first
    thing a submitter says -- a gate message is more actionable than "install
    huggingface_hub", and the model should hear the gate first.
    """
    return gates.check_submit(sub, expectation_id, cfg, project=project)


def spend_precondition(
    estimate_usd: float, cfg: Config, *, project: str | None
) -> Any:
    """Re-run the spend gates inside the append lock.

    `check_spend` and `budget.check` read the ledger, and the run record that
    makes this job's estimate visible is written afterwards -- so two submitters
    racing (the agent and a terminal, or the agent and a UI-spawned task) could
    both pass a $200 ceiling with $100 estimates against $50 spent, and both
    commit. The binding check already closes this shape of race for
    expectations; the ceilings get the same treatment rather than a comment
    explaining why they are the exception.
    """

    def _still_affordable() -> None:
        gates.check_spend(estimate_usd, cfg)
        gates.check_project_spend(project, estimate_usd)

    return _still_affordable


def record_submission(
    sub: Submission,
    *,
    expectation_id: str | None,
    platform: str,
    target: dict[str, Any],
    command: list[str],
    task: str | None = None,
    project: str | None = None,
    extra: dict[str, Any] | None = None,
    cfg: Config | None = None,
) -> tuple[str, dict[str, Any]]:
    """Mint the run id and write the in-flight record.

    Written *at submit time*, not at collect time. That is what lets §6's
    ceiling count in-flight jobs at their estimates, and what makes an
    expectation impossible to author retroactively: the run already names the id
    it was submitted with.

    Call this only once the gates have passed and the backend is known to be
    reachable, so a configuration problem never leaves a phantom estimate
    sitting on the ceiling.

    Pass `cfg` to re-check the spend ceilings inside the append lock; without it
    the ceilings are only as tight as the window between the gate and the write.
    """
    run_id = ls.new_id("run")
    record = {
        "type": ls.T_RUN_SUBMITTED,
        "id": run_id,
        "task": task or (sub.config.get("task") or sub.spec_path.parent.name),
        "status": "in_flight",
        "smoke": False,
        "submitted_at": ls.now_iso(),
        # HANDOFF-2 §15: every cost-bearing record carries the dimension. An
        # unselected project lands as `unassigned` rather than as null, so the
        # fold has one spelling for "not attributed" and existing ledgers keep
        # loading unchanged.
        "project": project or budget.UNASSIGNED,
        "platform": platform,
        "target": target,
        "submission_hash": sub.hash(),
        "spec": str(sub.spec_path),
        "expectation_id": expectation_id,
        "estimate_usd": sub.estimated_cost_usd(),
        "estimated_duration_s": sub.estimated_duration_s(),
        "command": command,
        "image": sub.image,
        "dataset": sub.dataset,
        "metrics_file": sub.metrics_file,
        "config": sub.config,
        **(extra or {}),
        # After the spread, and that ordering is the point. Which Grad submitted
        # this is what `core/report.py:check_code_versions` rests on, so it is
        # not a default a backend may override: a submitter that happened to put
        # `code_version` in `extra` would silently decide what the ledger says
        # about the code that produced a number.
        #
        # The README's claim is that every number in a report traces to a run
        # record; that is only complete if the record says which code produced
        # it, because a run from before an update and one from after are two
        # different experiments and nothing else in here can tell them apart.
        "code_version": version.stamp(),
    }
    ls.append_run_event(
        record,
        precondition=(
            None
            if cfg is None
            # `project`, not the record's `project or UNASSIGNED`: the in-lock
            # check must ask exactly what the gate asked.
            else spend_precondition(sub.estimated_cost_usd(), cfg, project=project)
        ),
    )
    return run_id, record


def record_smoke_run(
    sub: Submission,
    *,
    cfg: Config,
    platform: str,
    target: dict[str, Any],
    caps: dict[str, Any],
    command: list[str],
    project: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Smoke skips the gates but not the ledger.

    "Smoke spend still lands in runs.jsonl and counts toward the monthly
     ceiling." Otherwise the exemption would be a hole in the ceiling as well as
    in the gate.
    """
    run_id = ls.new_id("smoke")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "task": sub.config.get("task") or sub.spec_path.parent.name,
            "status": "in_flight",
            "smoke": True,
            "submitted_at": ls.now_iso(),
            "project": project or budget.UNASSIGNED,
            "platform": platform,
            "target": target,
            "submission_hash": sub.hash(),
            "spec": str(sub.spec_path),
            "expectation_id": None,
            "estimate_usd": float(caps["cost_ceiling_usd"]),
            "estimated_duration_s": float(caps["timeout_s"]),
            "command": command,
            "caps": caps,
            "image": sub.image,
            **(extra or {}),
            # After the spread, for the reason `record_submission` gives. Smoke
            # skips the gates but not the ledger, and not this either: a record
            # without it would be the one hole in "which code ran this".
            "code_version": version.stamp(),
        }
    )
    return run_id


def attach_handle(run_id: str, handle: dict[str, Any]) -> None:
    """Record the backend's own identifier for the job (HF job id, remote PID)."""
    ls.append_run_event({"type": "run_handle", "id": run_id, "handle": handle})


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def require_uncollected(run_id: str) -> ls.Run:
    r = ls.run(run_id)
    if r.collected:
        raise GradError(
            "already_collected",
            f"run {run_id} was already collected at {r.get('collected_at')}",
            exit_code=EXIT_USAGE,
            fix=f"python -m tools.ledger show {run_id} --json",
        )
    return r


def still_running(run_id: str, state: str, *, fix: str) -> GradError:
    return GradError(
        "still_running",
        f"run {run_id} is {state}",
        exit_code=EXIT_RUNNING,
        fix=fix,
    )


def parse_metrics(path: Path) -> dict[str, Any]:
    """Read the machine-readable metrics artifact.

    HANDOFF §7 makes this a contract rather than a convention: "the pipeline is
    required to emit a machine-readable metrics artifact (one JSON per eval, or
    a JSONL of scalar records), which is a cheap contract that removes all
    log-scraping."
    """
    if not path.exists():
        raise GradError(
            "metrics_missing",
            f"the run produced no metrics artifact at {path.name}",
            exit_code=9,
            fix=(
                "make the pipeline write a metrics file (JSON object of quantity -> value, "
                "or JSONL of {\"quantity\": ..., \"value\": ...} records) and set "
                "`metrics_file` in the submission spec"
            ),
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if path.suffix == ".jsonl" or "\n" in text and not text.startswith("{"):
        out: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                if "quantity" in rec and "value" in rec:
                    out[str(rec["quantity"])] = rec["value"]
                else:
                    out.update({k: v for k, v in rec.items() if _scalar(v)})
        return out
    doc = json.loads(text)
    if isinstance(doc, dict):
        return {k: v for k, v in doc.items() if _scalar(v)}
    return {}


def _scalar(v: Any) -> bool:
    """Metrics are scalars. Nested structures are artifacts, not quantities."""
    return isinstance(v, (int, float, str)) and not isinstance(v, bool)


def compute_deviations(expectation: dict[str, Any] | None, results: dict[str, Any]) -> list[dict[str, Any]]:
    """Mechanical comparison of results against the bound prediction.

    `verdict` is deliberately absent. "the machine records what happened, the
    model interprets it, and the interpretation cannot overwrite the record."
    """
    if not expectation:
        return []
    quantity = expectation.get("quantity")
    if quantity not in results:
        return [
            {
                "expectation_id": expectation.get("id"),
                "quantity": quantity,
                "actual": None,
                # None, not False. `Run.unjudged_deviations` documents None as
                # "the cases no program can settle" and names this one; the
                # SQLite index stores NULL to separate "needs a verdict" from
                # "numerically out of range". Writing False put missing-quantity
                # runs in with the numeric misses, so a query for `in_range = 0`
                # returned rows that never reported a number at all. Both still
                # demand a verdict -- the predicate is `is not True`.
                "in_range": None,
                "reason": "the run reported no value for the predicted quantity",
            }
        ]
    actual = results[quantity]
    predicted = expectation.get("predicted") or {}
    low, high = predicted.get("low"), predicted.get("high")
    dev: dict[str, Any] = {
        "expectation_id": expectation.get("id"),
        "quantity": quantity,
        "expected": {"low": low, "high": high, "direction": predicted.get("direction")},
        "actual": actual,
    }
    if not isinstance(actual, (int, float)):
        dev["in_range"] = None
        dev["reason"] = "non-numeric result; compare by hand"
        return [dev]
    if low is None and high is None:
        # A relational prediction has no range to test mechanically; it is
        # surfaced for judgement rather than silently marked in-range.
        dev["in_range"] = None
        dev["reason"] = "relational prediction; needs a verdict"
        return [dev]
    lo = low if low is not None else float("-inf")
    hi = high if high is not None else float("inf")
    dev["in_range"] = bool(lo <= actual <= hi)
    midpoint = None
    if low is not None and high is not None:
        midpoint = (low + high) / 2
    elif low is not None:
        midpoint = low
    elif high is not None:
        midpoint = high
    if midpoint:
        dev["ratio"] = round(actual / midpoint, 4)
    return [dev]


def finish(
    run_id: str,
    *,
    status: str,
    results: dict[str, Any],
    cost_usd_actual: float | None,
    artifacts_dir: Path,
    expectation: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the completed run record. Only `collect` calls this."""
    deviations = compute_deviations(expectation, results)
    record = {
        "type": ls.T_RUN_COLLECTED,
        "id": run_id,
        "status": status,
        "collected_at": ls.now_iso(),
        "results": results,
        "cost_usd_actual": cost_usd_actual,
        "artifacts": str(artifacts_dir),
        "deviations": deviations,
        **(extra or {}),
    }
    ls.append_run_event(record)
    return record


def artifacts_dir(run_id: str) -> Path:
    d = paths.run_artifacts(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def elapsed_hours(run: ls.Run, *, until: _dt.datetime | None = None) -> float:
    started = ls.parse_iso(run.get("submitted_at"))
    if not started:
        return 0.0
    until = until or _dt.datetime.now(_dt.timezone.utc)
    return max(0.0, (until - started).total_seconds() / 3600.0)
