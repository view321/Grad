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
import logging
from pathlib import Path
from typing import Any

from core import budget, gates, ledger_store as ls, paths, stats, version
from core.config import Config
from core.errors import EXIT_RUNNING, EXIT_USAGE, GradError
from core import submission as submission_mod
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
    estimate_usd: float, cfg: Config, *, project: str | None, sub: Submission | None = None
) -> Any:
    """Re-run the spend and concurrency gates inside the append lock.

    `check_spend` and `budget.check` read the ledger, and the run record that
    makes this job's estimate visible is written afterwards -- so two submitters
    racing (the agent and a terminal, or the agent and a UI-spawned task) could
    both pass a $200 ceiling with $100 estimates against $50 spent, and both
    commit. The binding check already closes this shape of race for
    expectations; the ceilings get the same treatment rather than a comment
    explaining why they are the exception.

    **The concurrency ceiling needs it more than the others do**, because racing
    submitters are no longer a corner case there -- they are the intended use.
    `tools/task.py` exists so two `submit`s can be started a second apart, and
    "count the in-flight runs, then write one" is exactly the read-decide-write
    the lock is for. A ceiling of 2 that two concurrent starts can both walk
    through is a ceiling of 4.
    """

    def _still_affordable() -> None:
        gates.check_spend(estimate_usd, cfg)
        gates.check_project_spend(project, estimate_usd)
        gates.check_concurrency(cfg, sub=sub)

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
    precondition: Any = None,
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

    `precondition` is a backend's *own* in-lock re-check, run after the spend one
    and subject to the same argument. A ceiling that only one backend has is
    still a ceiling two racing submitters can walk through together: Kaggle's
    weekly allowance has exactly the shape the dollar ceilings do -- read the
    ledger, decide, write afterwards -- so it needs the same treatment rather
    than a comment explaining why it is the exception.
    """
    run_id = ls.new_id("run")
    # One snapshot, and the hash derived from it rather than taken separately.
    # `Submission.hash()` resolves the document itself, so asking for both meant
    # digesting every source file twice -- and, in the window between the two,
    # an edit would make `spec_resolved` stop being the pre-image of
    # `submission_hash`. That is a false `spec_hash_mismatch` in
    # `experiments verify` for a run that was fine, which is the worst kind of
    # bug a verifier can have. Resolved by construction instead.
    resolved = sub.resolved()
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
        "submission_hash": submission_mod.hash_resolved(resolved),
        "spec": str(sub.spec_path),
        # The document the hash above is taken over, not just its digest. A path
        # names a file that will be edited; this is what the run actually ran, and
        # it is what lets `core/experiments.py` re-derive the hash long after the
        # workspace has moved on -- the archive's one self-contained integrity
        # check. It is already computed for the hash, so it costs a dict.
        "spec_resolved": resolved,
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
    # `project`, not the record's `project or UNASSIGNED`: the in-lock check must
    # ask exactly what the gate asked. `sub` for the same reason -- the ceiling
    # can come from the spec, and re-deriving it from config would re-check a
    # different number than the gate did.
    spend = (
        None
        if cfg is None
        else spend_precondition(sub.estimated_cost_usd(), cfg, project=project, sub=sub)
    )

    def _still_allowed() -> None:
        # Spend before the backend's own check, matching the order the caller ran
        # them in outside the lock: when both have gone stale, the caller should
        # hear the same refusal first inside it as it would have outside.
        if spend is not None:
            spend()
        if precondition is not None:
            precondition()

    ls.append_run_event(
        record,
        precondition=None if (spend is None and precondition is None) else _still_allowed,
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
    resolved = sub.resolved()  # once, for the reason `record_submission` gives
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
            "submission_hash": submission_mod.hash_resolved(resolved),
            "spec": str(sub.spec_path),
            "spec_resolved": resolved,
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


#: Which CLI owns `collect` for each `platform` a run record can carry. Only
#: used to name the right command in a refusal, so an unknown platform degrades
#: to the ledger rather than to a wrong instruction.
COLLECTORS = {
    "hf_jobs": "python -m tools.jobs collect",
    "kaggle": "python -m tools.kaggle collect",
    "ssh": "python -m tools.gpu collect",
}


def collect_command(run: ls.Run) -> str:
    tool = COLLECTORS.get(str(run.get("platform") or ""))
    return f"{tool} {run.id} --json" if tool else f"python -m tools.ledger show {run.id} --json"


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
    """Read the machine-readable metrics artifact, keeping only one value each.

    HANDOFF §7 makes this a contract rather than a convention: "the pipeline is
    required to emit a machine-readable metrics artifact (one JSON per eval, or
    a JSONL of scalar records), which is a cheap contract that removes all
    log-scraping."

    Kept for every caller that wants "the number for this quantity". A
    replicated run has several, and `parse_samples` is what does not throw them
    away -- see its docstring for what this used to lose.
    """
    return {q: values[-1] for q, values in parse_samples(path).items() if values}


def parse_samples(path: Path) -> dict[str, list[Any]]:
    """Every value the artifact reports for each quantity, in order.

    **A quantity reported more than once is a replicated run**, and this is the
    function that stopped discarding the replication. The JSONL branch used to
    write `out[quantity] = value` per record, so a pipeline emitting one
    `val_loss` per seed had all but the last of them silently overwritten -- the
    run looked like a single measurement, the ledger recorded it as one, and
    nothing anywhere said that two thirds of the evidence had been dropped on
    the floor.

    No new contract was needed for this, only for the existing one to stop
    losing data. A pipeline that already writes one record per seed is already
    replicating correctly.
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
    out: dict[str, list[Any]] = {}

    def _add(quantity: str, value: Any) -> None:
        out.setdefault(str(quantity), []).append(value)

    if path.suffix == ".jsonl" or "\n" in text and not text.startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if "quantity" in rec and "value" in rec:
                # Through the same filter as the flat form, for the reason
                # `_measurement` gives: the question is about the key, not about
                # the shape the pipeline wrote it in. Without this a run that
                # names its seed explicitly -- `{"quantity": "seed", ...}` --
                # gets it averaged as a measurement, and a non-scalar value
                # reaches `stats.summarise`, which has no answer for a list.
                if _measurement(str(rec["quantity"]), rec["value"]):
                    _add(rec["quantity"], rec["value"])
            else:
                for key, value in rec.items():
                    if _measurement(key, value):
                        _add(key, value)
        return out

    doc = json.loads(text)
    if isinstance(doc, dict):
        for key, value in doc.items():
            if _measurement(key, value):
                _add(key, value)
        return out
    return {}


#: Keys that label a record rather than measure anything. A pipeline that tags
#: its output by seed is doing exactly the right thing and must not find the tag
#: averaged into its results -- which is what would happen the moment
#: `core/stats.py` started treating repeated keys as samples.
PROVENANCE_KEYS = frozenset({"seed", "run", "run_id", "replicate", "trial", "fold"})


def _measurement(key: str, value: Any) -> bool:
    """Whether a `key: value` pair in a metrics record is a quantity.

    Applied to both artifact shapes, because the question is about the key and
    not about the file format -- and a rule that held in the JSONL branch and
    not in the JSON one would mean the same pipeline measured different things
    depending on which it wrote.
    """
    return key not in PROVENANCE_KEYS and _scalar(value)


def read_metrics(path: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """`(results, samples)` -- what to publish, and everything behind it.

    The one place the mean-versus-last rule lives, because all three collectors
    need it and three copies would drift. **The mean only when there is
    replication to average**: a single reading is recorded exactly as the
    pipeline wrote it, since turning an int into a float would change the shape
    of every existing record for no gain at all.

    `results` is what `report.quantity_value` resolves a `\\gradnum` against, so
    for a replicated run it has to be the mean rather than whichever seed
    happened to be written last -- publishing "the third seed" as though it were
    the result is the failure this whole module exists to prevent, in miniature.
    """
    samples = parse_samples(path)
    results: dict[str, Any] = {}
    for quantity, values in samples.items():
        if not values:
            continue
        numbers = stats.numeric(values)
        if len(values) > 1 and len(numbers) == len(values):
            results[quantity] = stats.summarise(values)["mean"]
        else:
            results[quantity] = values[-1]
    return results, samples


def _scalar(v: Any) -> bool:
    """Metrics are scalars. Nested structures are artifacts, not quantities."""
    return isinstance(v, (int, float, str)) and not isinstance(v, bool)


def compute_deviations(
    expectation: dict[str, Any] | None,
    results: dict[str, Any],
    samples: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mechanical comparison of results against the bound prediction.

    `verdict` is deliberately absent. "the machine records what happened, the
    model interprets it, and the interpretation cannot overwrite the record."

    `samples` is every value the run reported for each quantity. When a quantity
    has more than one, the comparison is **interval against interval** rather
    than point against interval -- see `core/stats.py` for why that has three
    outcomes rather than two. Optional, and defaulting to the single value in
    `results`, so every existing caller keeps its previous behaviour and gains
    only an `n: 1` beside the verdict.
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
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        dev["in_range"] = None
        dev["reason"] = "non-numeric result; compare by hand"
        return [dev]
    if low is None and high is None:
        # A relational prediction has no range to test mechanically; it is
        # surfaced for judgement rather than silently marked in-range.
        dev["in_range"] = None
        dev["reason"] = "relational prediction; needs a verdict"
        return [dev]

    # The replicated comparison, when the run reported this quantity more than
    # once. `samples` defaults to the single value, so an unreplicated run takes
    # exactly the path it always did and records `n: 1` beside its verdict --
    # which is the number `report check` needs in order to say that a published
    # figure rests on one seed.
    values = (samples or {}).get(quantity) or [actual]
    summary = stats.summarise(values)
    dev["stats"] = stats.round_summary(summary)
    verdict = stats.compare(summary, low, high)
    dev["in_range"] = verdict["in_range"]
    dev["relation"] = verdict["relation"]
    if verdict["in_range"] is not True:
        dev["reason"] = verdict["reason"]
    # Against the mean rather than the single reading, so the ratio and the
    # verdict above describe the same observation.
    centre = summary.get("mean")
    midpoint = None
    if low is not None and high is not None:
        midpoint = (low + high) / 2
    elif low is not None:
        midpoint = low
    elif high is not None:
        midpoint = high
    # `not in (None, 0)` rather than a bare truth test. The behaviour is the same
    # -- a zero midpoint has no ratio either way -- but the two reasons for
    # refusing are different: there is no interval to compare against, and there
    # is one but it is centred on zero. A prediction of `low=-x, high=x` is the
    # second, and reading `if midpoint` there suggests the interval was missing.
    if midpoint not in (None, 0) and centre is not None:
        dev["ratio"] = round(centre / midpoint, 4)
    return [dev]


def finish(
    run_id: str,
    *,
    status: str,
    results: dict[str, Any],
    cost_usd_actual: float | None,
    artifacts_dir: Path,
    expectation: dict[str, Any] | None,
    samples: dict[str, list[Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the completed run record. Only `collect` calls this.

    `samples` is what a replicated run reported per quantity. Recorded whole
    rather than only as the summary in `deviations`, for the reason the raw
    token counts survive beside `billable_tokens`: a summary is how numbers are
    *compared*, never a substitute for having them. A later change to what a
    95% interval means must be able to run over the samples that were actually
    measured, and a record holding only `mean` and `sd` could not answer it.
    """
    deviations = compute_deviations(expectation, results, samples)
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
    # Only when there is replication to record. An unreplicated run would
    # otherwise carry a `samples` block restating `results` one list at a time,
    # which is noise in every record in the ledger to no one's benefit.
    replicated = {q: v for q, v in (samples or {}).items() if len(v) > 1}
    if replicated:
        record["samples"] = replicated
    ls.append_run_event(record)
    archive_quietly(run_id)
    return record


def archive_quietly(run_id: str) -> dict[str, Any] | None:
    """Snapshot a terminal run into the cross-workspace archive.

    Here because `finish` is the one place a run becomes terminal, which is the
    same argument that puts `code_version` in `record_submission` -- one writer,
    so no backend can forget.

    **Failures are swallowed and reported, never raised.** The workspace ledger
    is the source of truth and it has already been written by the time this runs;
    an unwritable app directory, a locked archive, a full disk must not turn a
    successfully collected run into a failed `collect`, because the expensive
    thing has already happened and the record of it is already safe. The return
    value says whether it worked, for a caller that wants to mention it.
    """
    try:
        from core import experiments  # noqa: PLC0415 - keeps sqlite off every import of this module

        return experiments.archive(run_id)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logging.getLogger("grad.experiments").warning(
            "could not archive run %s (%s: %s); the workspace ledger is unaffected",
            run_id, type(exc).__name__, exc,
        )
        return None


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


# ---------------------------------------------------------------------------
# abandon: the terminal path for a run that never reached a backend
# ---------------------------------------------------------------------------
def abandon(run_id: str, *, reason: str) -> dict[str, Any]:
    """Close out an in-flight run that never reached hf/kaggle/gpu.

    Every submitter writes the in-flight record *before* the network call and
    the handle *after* it, so the ceiling can count the job while it is still
    being submitted. All three finalise the failure they can see -- an exception
    out of `run_job` / `_launch` / `_push` becomes a `submit_failed` record with
    a zero actual. What none of them can finalise is the failure they are not
    present for: Ctrl-C, a killed agent process, a lost machine, an exception
    that misses the handler. That leaves a run `in_flight` with no handle, and
    such a run

      * holds its estimate against the monthly ceiling, and its accelerator
        hours against the weekly pool, indefinitely;
      * cannot be collected -- every `collect` refuses it with `no_handle`,
        because there is no job id to poll; and
      * goes stale after the grace window, at which point `gates.check_stale`
        refuses *every* later submission on *every* backend.

    Those three together are a dead end: the fix that gate printed was
    `collect <run_id>`, the one command guaranteed to fail on exactly these
    runs, and the only way out was editing `runs.jsonl` by hand. So the escape
    hatch is here, and `check_stale` now points at it.

    It is deliberately narrow. **A run with a handle cannot be abandoned**: that
    run reached a platform, may be burning money right now, and dropping it off
    the ceiling to unblock a submission is precisely the bypass §6 exists to
    prevent. Those go to `collect`, or to the platform's own cancel.

    What is left is the window between the backend accepting and `attach_handle`
    landing. A run stranded *there* did reach the platform and is still booked at
    $0 here, because no one is left who can say otherwise -- so the record
    carries the basis it was written on rather than leaving a reader to infer
    it, and `reason` is required and recorded. A run that leaves the ledger
    without a result should say who decided that, and why.
    """
    r = require_uncollected(run_id)
    reason = (reason or "").strip()
    if not reason:
        raise GradError(
            "reason_required",
            "abandoning a run needs a reason; it is the only record of why this one left "
            "the ledger without a result",
            exit_code=EXIT_USAGE,
            fix='--reason "the submitter was killed before the job id came back"',
        )
    handle = r.get("handle") or {}
    if handle:
        raise GradError(
            "run_reached_backend",
            f"run {r.id} has a backend handle ({', '.join(sorted(handle))}), so it did reach "
            f"{r.get('platform') or 'a platform'} and may still be running there. Abandoning it "
            "would drop a live job off the spend ceiling, which is the bypass the gates exist "
            "to stop",
            exit_code=EXIT_USAGE,
            fix=collect_command(r),
            detail={"run_id": r.id, "handle": handle, "platform": r.get("platform")},
        )

    extra: dict[str, Any] = {
        "reason": reason,
        "cost_basis": (
            "abandoned: no handle was ever recorded, so the run is booked at $0 rather than "
            "at its estimate"
        ),
    }
    # A metered backend gets its units back too. Kaggle's weekly pool reads the
    # estimate until an actual is written, so a run finalised without one holds
    # its hours for a week -- the same leak the dollar ceiling would have had, in
    # a different unit. Keyed off the record rather than off the platform string:
    # a run that is holding hours gives them back, whichever backend wrote it.
    from core import kaggle_quota  # noqa: PLC0415 - one field name, at point of use

    if r.get(kaggle_quota.F_ESTIMATE) is not None:
        extra[kaggle_quota.F_ACTUAL] = 0.0

    return finish(
        r.id,
        status=ls.ABANDONED,
        results={},
        cost_usd_actual=0.0,
        # `paths.run_artifacts`, not `artifacts_dir`: `finish` only writes the
        # path into the record, and the helper would *create* the directory --
        # so every abandoned run left an empty folder behind, for a run defined
        # by having produced nothing. The path is still recorded, because where
        # the artifacts would have been is a fact about the run either way.
        artifacts_dir=paths.run_artifacts(r.id),
        # No expectation, so `compute_deviations` writes none. The prediction was
        # never tested -- inventing a deviation for a run that produced nothing
        # would put a row in the unjudged list that no verdict can honestly
        # settle. The binding stands; §7 has one spelling of "consumed" and this
        # is not the place to add a second.
        expectation=None,
        extra=extra,
    )
