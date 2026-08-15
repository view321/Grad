"""What each window shows, as plain data.

Every window in `ui/windows/` is a renderer over one function in here. That
split is the reason this redesign is testable at all: the interesting part of a
window is never the borders, it is "which of these expectations counts as
broken", "what does the observed value do to the band strip when the band is a
single point", "is this run in flight or merely uncollected". Those are
decisions, they have edge cases, and none of them need a browser.

Two rules hold throughout:

* **No NiceGUI import.** Not at module scope, not inside a function. `tests/
  test_ui_models.py` imports this module with the UI extra uninstalled.
* **A broken source degrades to an empty window, never to a broken app.** Every
  model catches its own read failures and returns an `error` string in the
  result. Eleven windows over eight ledgers means eight chances per refresh for
  one malformed JSONL line to take the whole workspace down, and the workspace
  is what the user is looking at when they need to find out what went wrong.

The honesty note from §10 survives the redesign intact: the token meter is
self-measured usage against an assumed ceiling, not a fuel gauge, and it says so
in the window rather than in a docstring.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from core import paths

# What "recent" means in the windows that cap their own lists. Deliberately
# generous: these are ledgers, not feeds.
RECENT = 40


def _safe(fn: Callable[[], Any], default: Any = None) -> tuple[Any, str | None]:
    """Run a reader, returning `(value, error)` rather than raising.

    Bare `except Exception` on purpose, and the reason is specific: the readers
    below touch sqlite, JSONL, tomllib, an optional `sqlite-vec` extension and
    an optional Hub client. The set of exception types that reach here is not
    knowable from this module, and the correct response to every one of them is
    identical -- show the window, say what failed.
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - see above
        return default, f"{type(exc).__name__}: {exc}"


def _iso(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)


def _short(value: Any, limit: int = 90) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _usd(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$—"


def _tokens(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "—"
    for unit, size in (("M", 1_000_000), ("k", 1_000)):
        if abs(number) >= size:
            return f"{number / size:.1f}{unit}"
    return str(number)


# ---------------------------------------------------------------------------
# global header state
# ---------------------------------------------------------------------------
AGENT_STATES = ("idle", "running", "awaiting_gate", "paused")

# Which accent each agent state gets. Total, so no state borrows another's --
# that is the "one accent per state" rule made mechanical.
AGENT_ACCENT = {
    "idle": "neutral",
    "running": "ok",
    "awaiting_gate": "attention",
    "paused": "attention",
}


def workspaces_model() -> dict[str, Any]:
    """What the project menu shows: this folder, the recent ones, the projects.

    Every reader is wrapped, because this is the one panel that has to render
    when the workspace is *wrong* -- an empty folder, a ledger that will not
    parse, a drive that is not mounted. A menu that cannot open because the
    workspace it is meant to let you leave is broken is the one failure mode it
    must not have.
    """
    from core import budget as budget_mod, workspace as workspace_mod

    root, root_error = _safe(lambda: str(paths.root()), "")
    recent, _ = _safe(lambda: [str(p) for p in workspace_mod.recent()], [])
    current, _ = _safe(budget_mod.current_project)
    records, projects_error = _safe(budget_mod.projects, {})

    projects = []
    for project_id, record in sorted((records or {}).items()):
        state, _ = _safe(lambda pid=project_id: budget_mod.status(pid), {})
        projects.append(
            {
                "id": project_id,
                "title": _short(record.get("title") or "", 60),
                "status": record.get("status") or "open",
                "current": project_id == current,
                "spend": _spend_line(state or {}),
            }
        )
    return {
        "root": root,
        # The rule that picked it -- "why is it still pointing there?" is
        # otherwise unanswerable from inside the app.
        "source": _safe(workspace_mod.source, "default")[0],
        # The one already open is not offered as somewhere to go.
        "recent": [p for p in (recent or []) if p != root],
        "projects": projects,
        "current_project": current,
        "error": root_error or projects_error,
    }


def _spend_line(state: dict[str, Any]) -> str:
    """One line per project: what it has spent against what it may.

    A project with no ceilings is the common case -- they are optional -- and it
    says so rather than rendering an empty bar, which would read as "nothing
    spent" when it means "nothing to exceed".
    """
    resources = state.get("resources") or {}
    parts: list[str] = []
    for name, render in (("gpu_usd", _usd), ("quota_tokens", _tokens), ("credits_usd", _usd)):
        entry = resources.get(name) or {}
        ceiling = entry.get("ceiling")
        if not ceiling:
            continue
        spent = render(entry.get("spent", 0))
        parts.append(f"{name.split('_')[0]} {spent}/{render(ceiling)}")
    return " · ".join(parts) or "no ceilings"


def header_model(*, agent_state: str = "idle", step: int | None = None) -> dict[str, Any]:
    """The workspace title bar: project, agent state, session quota strip.

    The session strip is the 5-hour rolling window, which is the one the user
    actually bumps into. It is split into the two segments the mock shows --
    what chat spent and what tools spent -- because "you are near the cap" and
    "your retrieval is what put you there" are different pieces of information.
    """
    from core import budget as budget_mod, config as config_mod

    project, project_error = _safe(budget_mod.current_project)
    cfg, _ = _safe(config_mod.load)

    ceiling = 8.0
    if cfg is not None:
        raw, _ = _safe(lambda: float(cfg.get("spend", "session_usd", 8.0)), 8.0)
        ceiling = raw or 8.0

    window, error = _safe(lambda: _session_window(hours=5), {})
    window = window or {}
    used = float(window.get("credits_usd", 0.0))
    return {
        "project": project or "unassigned",
        "agent_state": agent_state if agent_state in AGENT_STATES else "idle",
        "accent": AGENT_ACCENT.get(agent_state, "neutral"),
        "step": step,
        "session": {
            "used_usd": round(used, 4),
            "ceiling_usd": ceiling,
            "chat_fraction": window.get("chat_fraction", 0.0),
            "tool_fraction": window.get("tool_fraction", 0.0),
            "used_label": f"{_usd(used)} / {_usd(ceiling)}",
            "resets_in": window.get("resets_in", "—"),
            "tokens": window.get("tokens", 0),
        },
        "error": project_error or error,
        # Repeated wherever a token number appears, per §10. The provider
        # exposes no remaining-quota API; this is our own tally.
        "honesty": "self-measured tally, not the provider's — an estimate within ±5%",
    }


def _session_window(*, hours: int = 5) -> dict[str, Any]:
    """Spend inside the rolling window, split by what asked for it.

    `stage` is what `quota_log` records, so the split is exact rather than
    guessed: `main` is the conversation, every `funnel.*` and `embed`/`ingest`
    stage is a tool spending on the conversation's behalf.
    """
    from core import quota_log

    now = _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(hours=hours)
    chat = tool = 0.0
    tokens = 0
    oldest: _dt.datetime | None = None
    for entry in quota_log.entries():
        at = _iso(entry.get("at"))
        if not at or at < cutoff:
            continue
        oldest = at if oldest is None or at < oldest else oldest
        credits = float(entry.get("credits_usd", 0.0) or 0.0)
        tokens += int(entry.get("input_tokens", 0) or 0) + int(entry.get("output_tokens", 0) or 0)
        if str(entry.get("stage") or "") == quota_log.STAGE_MAIN:
            chat += credits
        else:
            tool += credits
    total = chat + tool
    resets_in = "—"
    if oldest is not None:
        remaining = (oldest + _dt.timedelta(hours=hours)) - now
        if remaining.total_seconds() > 0:
            hrs, rem = divmod(int(remaining.total_seconds()), 3600)
            resets_in = f"{hrs}h {rem // 60:02d}m"
    return {
        "credits_usd": total,
        "chat_usd": chat,
        "tool_usd": tool,
        "chat_fraction": (chat / total) if total else 0.0,
        "tool_fraction": (tool / total) if total else 0.0,
        "tokens": tokens,
        "resets_in": resets_in,
    }


def status_model() -> dict[str, Any]:
    """The workspace status bar: cwd, kernel, queue and gpu counts.

    Counted as *uncollected*, not as `ls.in_flight()`. That helper is narrower
    -- it requires `status == "in_flight"` exactly, because the staleness gate it
    feeds should only chase jobs the platform still calls live. For a status bar
    the honest number is everything submitted and not yet collected, which is
    also exactly what the queue window lists as not-DONE. Two counters that
    disagree about the same runs are worse than either one alone.
    """
    from core import ledger_store as ls

    from ui import tasks as tasks_mod

    kernel, _ = _safe(_kernel_state, "no kernel")
    runs, runs_error = _safe(ls.runs, [])
    uncollected = [r for r in (runs or []) if not r.collected]
    return {
        "cwd": str(paths.root()),
        "kernel": kernel,
        "queued": len(uncollected),
        "gpu": len([r for r in uncollected if not r.is_smoke]),
        # Local subprocesses, counted apart from the remote runs beside them.
        # A wiki rebuild and a GPU job are both "running" and are not remotely
        # the same fact -- one is this machine's CPU, the other is money.
        "tasks": len(tasks_mod.running()),
        "error": runs_error,
    }


def _kernel_state() -> str:
    from tools import lab as lab_tool

    state = lab_tool.lab_state()
    if state.get("running"):
        return f"lab :{state['port']}"
    return "lab stopped"


# ---------------------------------------------------------------------------
# 0. chat sessions
# ---------------------------------------------------------------------------
def sessions_model(current: str | None = None) -> dict[str, Any]:
    """The stored conversations, and which one is open.

    Read here rather than in the window for the reason every other window reads
    a model: a file read belongs on this side of the line, and it makes the
    picker testable without a browser. `chat` has no entry in `MODEL_BUILDERS`
    -- its state is the live session, not a file, and the poll must not redraw
    it -- so this is called directly, the way `workspaces_model` is.
    """
    from ui import sessions as sessions_mod

    listed, error = _safe(sessions_mod.listing, [])
    rows = list(listed or [])
    for row in rows:
        # Held by *another* window. Opening it there too would put two writers
        # on one file, so the picker says so rather than letting the refusal
        # arrive as a surprise on click.
        held = sessions_mod.holder(row["id"])
        row["held_elsewhere"] = bool(held) and row["id"] != current
    return {
        "rows": rows,
        "current": current,
        "count": len(rows),
        # Reopening a session whose SDK id was never recorded shows the
        # transcript without continuing the conversation. Counted so the window
        # can say so rather than let it be discovered.
        "transcript_only": len([r for r in rows if not r["resumable"]]),
        "error": error,
    }


# ---------------------------------------------------------------------------
# 0a. credentials
# ---------------------------------------------------------------------------
#: What each credential unlocks, and whether the system works without it. The
#: text matters as much as the flag: "missing" is not the same fact for a token
#: that gates GPU submission as for one that raises a rate limit.
CREDENTIAL_NOTES: dict[str, tuple[str, bool]] = {
    "hf_token": ("Hugging Face Jobs — submitting and collecting runs", True),
    "openrouter_key": ("the reranker, funnel stage 2 (costs credits)", False),
    "voyage_key": ("embeddings for the local index (costs credits)", False),
    "asta_api_key": ("raises Asta's rate limits; discovery works without it", False),
    "s2_api_key": ("Semantic Scholar direct — only issued to institutional addresses", False),
    "context7_key": ("raises Context7's rate limits; lookups work without it", False),
    "claude_oauth_token": ("the funnel's Haiku stages, when the agent runs them", True),
}


def credentials_model() -> dict[str, Any]:
    """Which credentials are stored. Values are never read, let alone returned.

    The point of the panel this feeds is that storing a credential was the one
    thing the workspace could not do: `jobs.py credential set` prompts with
    `getpass`, which needs a terminal, so the four commands in the README's
    install section were the reason to keep a shell open beside the app.
    """
    from core import credentials as credentials_mod

    present, error = _safe(credentials_mod.status, {})
    rows = []
    for name, stored in (present or {}).items():
        purpose, required = CREDENTIAL_NOTES.get(name, ("", False))
        rows.append(
            {
                "name": name,
                "stored": bool(stored),
                "purpose": purpose,
                "required": required,
                "tone": "ok" if stored else ("broken" if required else "neutral"),
                "state": "STORED" if stored else ("MISSING" if required else "not set"),
            }
        )
    return {
        "rows": rows,
        "missing_required": [r["name"] for r in rows if r["required"] and not r["stored"]],
        "error": error,
        # The store itself, so "nothing is stored" and "nothing can be stored"
        # are distinguishable on screen.
        "service": getattr(credentials_mod, "SERVICE", "grad"),
    }


# ---------------------------------------------------------------------------
# 0b. background tasks
# ---------------------------------------------------------------------------
def tasks_model() -> dict[str, Any]:
    """Local commands the workspace started, newest first.

    Deliberately not merged into `queue_model`. Both lists hold things that are
    "running", and that is where the resemblance stops: a wiki rebuild is this
    machine's CPU for two minutes, a GPU job is money against a ceiling that a
    gate refuses at. The queue window's own docstring makes the same argument in
    the other direction about campaign candidates, and one table showing both
    would make each one harder to read for no gain.

    The output tail is included whole. It is bounded at the source
    (`tasks.TAIL_LINES`), and the poll's fingerprint is what turns "a line
    arrived" into a redraw -- so a task that is quiet costs one comparison.
    """
    from ui import tasks as tasks_mod

    rows = []
    for task in tasks_mod.all_tasks():
        rows.append(
            {
                "id": task.id,
                "label": task.label,
                "command": "python -m " + " ".join(task.argv),
                "state": task.state,
                "tone": tasks_mod.STATE_TONE.get(task.state, "neutral"),
                "running": task.running,
                "elapsed": _duration(task.elapsed),
                "exit_code": task.exit_code,
                "stoppable": task.running,
                # Named so the button can say what stopping will actually do:
                # asking the tool, or signalling it. See `tasks.cancel`.
                "halt": ("python -m " + " ".join(task.halt)) if task.halt else None,
                "message": tasks_mod.task_message(task),
                "tail": _tail_runs(task.tail),
                "dropped": task.dropped,
            }
        )
    return {
        "rows": rows,
        "running": len([r for r in rows if r["running"]]),
        "finished": len([r for r in rows if not r["running"]]),
        "empty_fix": (
            "nothing has been started from the workspace yet — VERIFY, RE-CHECK, "
            "REBUILD and BUILD PDF all run here"
        ),
    }


def _tail_runs(tail: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Consecutive lines of the same kind, joined into one block.

    A tail is up to `tasks.TAIL_LINES` long, and a `<pre>` per line would be 400
    elements per task rebuilt on every poll that moved one of them. Runs collapse
    that to two or three in practice, while keeping stderr distinguishable and --
    the part a naive "all stdout, then all stderr" split would lose -- keeping
    every line in the order the command emitted it.
    """
    runs: list[tuple[str, list[str]]] = []
    for tag, line in tail:
        if runs and runs[-1][0] == tag:
            runs[-1][1].append(line)
        else:
            runs.append((tag, [line]))
    return [(tag, "\n".join(lines)) for tag, lines in runs]


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


# ---------------------------------------------------------------------------
# 1. notebook
# ---------------------------------------------------------------------------
VERIFY_STATES = ("clean", "failed", "stale", "unverified")


def verify_store_path() -> Path:
    return paths.data_dir() / "nb_verify.json"


def read_verify_store() -> dict[str, Any]:
    from core import jsonl

    data = jsonl.read_json(verify_store_path())
    return data if isinstance(data, dict) else {}


def write_verify_record(name: str, record: dict[str, Any]) -> dict[str, Any]:
    from core import jsonl

    store = read_verify_store()
    store[name] = record
    jsonl.write_json(verify_store_path(), store)
    return record


def record_verify(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist what `nb verify` said, so the banner survives a reload.

    The banner is the sole source of the citable state, so it has to outlive the
    page that produced it -- otherwise every restart silently downgrades a
    verified notebook to "never verified", and a discipline that quietly resets
    is a discipline nobody keeps.
    """
    from core import ledger_store as ls

    if payload.get("ok"):
        data = payload.get("data") or {}
        return write_verify_record(
            name,
            {
                "ok": True,
                "at": ls.now_iso(),
                "cells_executed": data.get("cells_executed"),
                "duration_s": data.get("duration_s"),
            },
        )

    error = payload.get("error") or {}
    detail = error.get("detail") or {}
    # `nb verify` nests the kernel's own error under `error`, with the traceback
    # already stripped of ANSI escapes.
    traceback = (detail.get("error") or {}).get("traceback") or detail.get("stdout")
    if isinstance(traceback, list):
        traceback = "\n".join(str(t) for t in traceback)
    return write_verify_record(
        name,
        {
            "ok": False,
            "at": ls.now_iso(),
            "message": error.get("message"),
            "cell_index": detail.get("cell_index"),
            "traceback": traceback,
            "fix": error.get("fix"),
        },
    )


def verify_state(name: str, *, store: dict[str, Any] | None = None) -> dict[str, Any]:
    """The banner's state, and the one place that decides it.

    "the banner is the sole source of the citable/not-citable state, and it goes
    stale (yellow) on any edit". Staleness is decided by comparing the
    notebook's mtime against the verification's, so an edit made in Lab -- which
    the host never sees -- still invalidates the banner. That is the whole point:
    Lab and `tools/nb.py` are two kernel owners over one file.
    """
    store = read_verify_store() if store is None else store
    record = store.get(name)
    path = paths.notebooks_dir() / name
    if not isinstance(record, dict) or not record.get("at"):
        return {
            "state": "unverified",
            "accent": "attention",
            "citable": False,
            "sentence": "never verified on a fresh kernel",
            "chip": "NOT CITABLE",
        }
    verified_at = _iso(record.get("at"))
    edited_after = False
    if verified_at and path.exists():
        mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.timezone.utc)
        # One second of slack: the verify run itself rewrites execution counts.
        edited_after = mtime > verified_at + _dt.timedelta(seconds=1)

    if edited_after:
        return {
            "state": "stale",
            "accent": "attention",
            "citable": False,
            "sentence": f"edited since it was verified at {record.get('at')}",
            "chip": "RE-VERIFY",
        }
    if record.get("ok"):
        cells = record.get("cells_executed", "?")
        seconds = record.get("duration_s")
        timing = f" · {seconds:.1f}s" if isinstance(seconds, (int, float)) else ""
        return {
            "state": "clean",
            "accent": "ok",
            "citable": True,
            "sentence": f"clean — {cells} cells ran top to bottom on a fresh kernel{timing} · {record.get('at')}",
            "chip": "CITABLE",
        }
    return {
        "state": "failed",
        "accent": "broken",
        "citable": False,
        "sentence": _short(record.get("message") or "verification failed", 140),
        "chip": "NOT CITABLE",
        "cell_index": record.get("cell_index"),
        "traceback": record.get("traceback"),
        "fix": record.get("fix"),
    }


def notebook_model() -> dict[str, Any]:
    from tools import lab as lab_tool

    directory = paths.notebooks_dir()
    names = sorted(p.name for p in directory.glob("*.ipynb")) if directory.exists() else []
    store = read_verify_store()
    lab, lab_error = _safe(lab_tool.lab_state, {})
    lab = lab or {}
    return {
        "notebooks": [{"name": n, "verify": verify_state(n, store=store)} for n in names],
        "lab_running": bool(lab.get("running")),
        "lab_port": lab.get("port"),
        "lab_token": lab.get("token"),
        "ruler": 88,
        "error": lab_error,
    }


def lab_url(state: dict[str, Any], notebook: str | None = None) -> str:
    """The iframe src, or `about:blank` when there is nothing to embed."""
    if not state.get("lab_running"):
        return "about:blank"
    base = f"http://127.0.0.1:{state['lab_port']}"
    token = state.get("lab_token") or ""
    if notebook:
        return f"{base}/lab/tree/notebooks/{notebook}?token={token}"
    return f"{base}/lab?token={token}"


# ---------------------------------------------------------------------------
# 2. ledger / expectations
# ---------------------------------------------------------------------------
def band_geometry(
    *, low: Any, high: Any, actual: Any, falsifier_low: Any = None, falsifier_high: Any = None
) -> dict[str, Any] | None:
    """Where the band, the observed tick and the falsifier bounds sit, 0..1.

    Returns `None` when there is nothing to draw, which is a real case rather
    than an error: a relational expectation ("the evolved variant beats
    baseline") has no numeric band, and §7 says to *prefer* those. Drawing a
    degenerate strip for them would be worse than drawing nothing.

    Three shapes that all occur in `runs.jsonl` and all have to render:

    * **Closed** -- `low` and `high` both set. The ordinary case.
    * **Half-open** -- one bound only, which `compute_deviations` treats as
      ±infinity. The band runs to that edge of the axis, so "below 3.2" draws as
      a block reaching the left edge rather than as nothing at all.
    * **Point** -- `low == high`. The axis is padded outward so the block is
      visible instead of collapsing to a hairline.

    `falsifier_*` has no producer today: the ledger records falsification as an
    event (`T_EXPECTATION_FALSIFIED`), not as a pair of bounds on the
    expectation. The parameters exist because the design asks for the ticks and
    the strip should not need reworking on the day the field is added; until
    then nothing is invented and no ticks are drawn.
    """
    if not isinstance(actual, (int, float)):
        return None
    low_set = isinstance(low, (int, float))
    high_set = isinstance(high, (int, float))
    if not (low_set or high_set):
        return None
    if low_set and high_set and low > high:
        low, high = high, low

    values = [v for v in (low, high, actual, falsifier_low, falsifier_high) if isinstance(v, (int, float))]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        # Every number identical. Invent a symmetric axis so the ticks separate.
        pad = abs(lo) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    else:
        pad = span * 0.12
        lo, hi = lo - pad, hi + pad
    width = hi - lo

    def position(value: Any) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        return round(max(0.0, min(1.0, (value - lo) / width)), 6)

    # An unset bound is ±infinity to `compute_deviations`, so it is the axis
    # edge here -- the band reaches the wall rather than vanishing.
    return {
        "axis_min": lo,
        "axis_max": hi,
        "band_start": position(low) if low_set else 0.0,
        "band_end": position(high) if high_set else 1.0,
        "actual": position(actual),
        "actual_value": actual,
        "falsifier_low": position(falsifier_low),
        "falsifier_high": position(falsifier_high),
        "in_band": (not low_set or low <= actual) and (not high_set or actual <= high),
        "open_low": not low_set,
        "open_high": not high_set,
    }


def _expectation_state(expectation: dict[str, Any], falsified: set[str], deviations: list[dict[str, Any]]) -> str:
    if expectation.get("id") in falsified:
        return "broken"
    if any(d.get("in_range") is False for d in deviations):
        return "broken"
    if deviations and all(d.get("in_range") is True for d in deviations):
        return "met"
    return "open"


LEDGER_ACCENT = {"open": "attention", "met": "ok", "broken": "broken"}


def ledger_model() -> dict[str, Any]:
    """Expectations, their outcomes, and the band strip for each.

    An expectation with no run yet is `open` and shows no strip; one whose runs
    all landed in band is `met`; one that was explicitly falsified, or whose run
    landed outside, is `broken`. The falsified event is checked first because it
    is a human's judgement and outranks the arithmetic.
    """
    from core import ledger_store as ls

    expectations, error = _safe(ls.expectations, [])
    runs, runs_error = _safe(ls.runs, [])
    falsified, _ = _safe(ls.falsified_ids, set())
    expectations = expectations or []
    runs = runs or []
    falsified = falsified or set()

    by_expectation: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for deviation in run.get("deviations", []) or []:
            key = deviation.get("expectation_id")
            if key:
                by_expectation.setdefault(str(key), []).append({**deviation, "run_id": run.id})

    entries: list[dict[str, Any]] = []
    for expectation in expectations:
        eid = str(expectation.get("id") or "")
        deviations = by_expectation.get(eid, [])
        state = _expectation_state(expectation, falsified, deviations)
        latest = deviations[-1] if deviations else {}
        predicted = expectation.get("predicted") or latest.get("expected") or {}
        falsifier = expectation.get("falsifier") or {}
        band = band_geometry(
            low=predicted.get("low"),
            high=predicted.get("high"),
            actual=latest.get("actual"),
            falsifier_low=falsifier.get("low") if isinstance(falsifier, dict) else None,
            falsifier_high=falsifier.get("high") if isinstance(falsifier, dict) else None,
        )
        entries.append(
            {
                "id": eid,
                "state": state,
                "accent": LEDGER_ACCENT[state],
                "claim": expectation.get("claim") or expectation.get("quantity") or "—",
                "quantity": expectation.get("quantity"),
                "at": expectation.get("at") or expectation.get("created_at"),
                "band": band,
                "predicted": predicted,
                "falsifier": falsifier if isinstance(falsifier, dict) else {},
                "comparability": expectation.get("comparability"),
                "basis": expectation.get("basis") or [],
                "confidence": expectation.get("confidence"),
                "runs": [d.get("run_id") for d in deviations],
                "unjudged": bool(
                    [d for d in deviations if d.get("in_range") is not True and not d.get("verdict")]
                ),
            }
        )
    entries.reverse()
    counts = {state: len([e for e in entries if e["state"] == state]) for state in LEDGER_ACCENT}
    return {
        "entries": entries[:RECENT],
        "counts": counts,
        "total": len(entries),
        "error": error or runs_error,
        "empty_fix": (
            "python -m tools.ledger expect --task <task> --quantity <q> --low <lo> --high <hi> "
            "--basis '<paper>|<locator>|<value>|<conditions>' --json"
        ),
    }


# ---------------------------------------------------------------------------
# 3. quota & budget
# ---------------------------------------------------------------------------
def quota_model(*, days: int = 1) -> dict[str, Any]:
    """The 5-hour meter, today's spend by model, and the honesty note.

    Three ceilings, not one: the rolling session window (what stops the
    conversation), the project allocation (what stops the research), and the
    machine's monthly GPU ceiling (what stops the money). They fail in different
    ways and a single bar would hide two of them.
    """
    from core import budget as budget_mod, config as config_mod, ledger_store as ls, quota_log

    window, window_error = _safe(lambda: _session_window(hours=5), {})
    summary, summary_error = _safe(lambda: quota_log.summarise(days), {})
    cfg, _ = _safe(config_mod.load)
    project, _ = _safe(budget_mod.current_project)

    monthly = 200.0
    session_ceiling = 8.0
    spend_window = 30
    if cfg is not None:
        monthly = float(cfg.get("spend", "monthly_usd", 200.0) or 200.0)
        session_ceiling = float(cfg.get("spend", "session_usd", 8.0) or 8.0)
        spend_window = int(cfg.get("spend", "window_days", 30) or 30)

    rolling, rolling_error = _safe(lambda: ls.rolling_spend(spend_window), {})
    status, status_error = _safe(
        lambda: budget_mod.status(project) if project else None
    )

    summary = summary or {}
    by_role = summary.get("by_role") or {}
    roles = []
    for role, node in sorted(by_role.items(), key=lambda kv: -kv[1].get("credits_usd", 0.0)):
        roles.append(
            {
                "role": role,
                "credits_usd": node.get("credits_usd", 0.0),
                "tokens": node.get("input_tokens", 0) + node.get("output_tokens", 0),
                "calls": node.get("calls", 0),
                # sonnet reads ink, opus reads link, anything gpu-ish reads
                # teal: three bars that are never the same colour as a state.
                "tone": "opus" if "opus" in role.lower() else ("tool" if "gpu" in role.lower() else "ink"),
            }
        )

    rolling = rolling or {}
    return {
        "session": {
            **(window or {}),
            "ceiling_usd": session_ceiling,
            "fraction": min(1.0, (window or {}).get("credits_usd", 0.0) / session_ceiling)
            if session_ceiling
            else 0.0,
        },
        "roles": roles,
        "stages": summary.get("by_stage") or {},
        "total_tokens": summary.get("total_tokens", 0),
        "total_credits_usd": summary.get("total_credits_usd", 0.0),
        "gpu": {
            "total_usd": rolling.get("total_usd", 0.0),
            "actual_usd": rolling.get("actual_usd", 0.0),
            "in_flight_usd": rolling.get("in_flight_usd", 0.0),
            "monthly_usd": monthly,
            "window_days": spend_window,
            "fraction": min(1.0, rolling.get("total_usd", 0.0) / monthly) if monthly else 0.0,
        },
        "project": status,
        "days": days,
        "error": window_error or summary_error or rolling_error or status_error,
        "honesty": (
            "Token counts are Grad's own tally, not the provider's. Anthropic exposes no "
            "remaining-quota API and the 5-hour window is opaque, so this is self-measured "
            "usage against a ceiling you set — an estimate within about ±5%, not a fuel gauge. "
            "GPU dollars are real, and count in-flight runs at their estimates."
        ),
    }


# ---------------------------------------------------------------------------
# 4. preflight + gates
# ---------------------------------------------------------------------------
def _preflight_files() -> list[Path]:
    """Preflight records, newest first.

    `stat()` is guarded per file rather than in the sort key's caller: these
    records are written by a concurrent `preflight run` (temp file plus
    `os.replace`), so a path returned by `glob` can be gone by the time it is
    stat'd, and one vanished file must not take the whole listing down.
    """
    directory = paths.preflight_dir()
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), key=_mtime, reverse=True)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def preflight_model() -> dict[str, Any]:
    """The most recent submission's checklist, with its one-click remedy.

    Reads are guarded **per record**, not around the loop. `jsonl.read_json`
    returns `None` for a missing or malformed file but propagates
    `UnicodeDecodeError` and `OSError` -- and `UnicodeDecodeError` is a sibling
    of `JSONDecodeError` under `ValueError`, not a subclass, so its own
    `except` does not catch it. An unreadable record left to escape here is
    caught upstream by `Workspace.rebuild` and replaces the *entire* model with
    an error, which this window renders as "No preflight records yet." -- the
    one wrong answer, since it says nothing is there precisely when something is
    there and cannot be read.
    """
    from core import jsonl

    files, listing_error = _safe(_preflight_files, [])
    problems = [listing_error] if listing_error else []

    records = []
    for path in (files or [])[:20]:
        record, error = _safe(lambda p=path: jsonl.read_json(p))
        if error:
            problems.append(f"{path.name}: {error}")
        elif isinstance(record, dict):
            records.append(record)
    error = "; ".join(problems) or None

    if not records:
        return {
            "records": [],
            "current": None,
            "blocking": 0,
            "can_proceed": False,
            "error": error,
            "empty_fix": "python -m tools.preflight run --spec <spec> --json",
        }

    current = records[0]
    checks = current.get("checks") or {}
    rows = []
    for name, result in checks.items():
        result = result if isinstance(result, dict) else {}
        ok = result.get("ok")
        rows.append(
            {
                "name": name,
                "state": "ok" if ok else ("broken" if ok is False else "attention"),
                "glyph": "✓" if ok else ("✕" if ok is False else "!"),
                "sentence": result.get("reason") or _check_sentence(name, ok),
                "detail": f"{result.get('duration_s', '—')}s",
                "output": result.get("output"),
                "fix": result.get("fix"),
            }
        )
    blocking = len([r for r in rows if r["state"] == "broken"])
    return {
        "records": records,
        "current": {
            "hash": current.get("submission_hash"),
            "spec": current.get("spec"),
            "verified_at": current.get("verified_at"),
            "warnings": current.get("warnings") or [],
            "rows": rows,
        },
        "blocking": blocking,
        "can_proceed": blocking == 0 and bool(rows),
        "remedy": next((r["fix"] for r in rows if r["state"] == "broken" and r.get("fix")), None),
        "error": error,
        "empty_fix": "python -m tools.preflight run --spec <spec> --json",
    }


def _check_sentence(name: str, ok: Any) -> str:
    verb = {"tests": "the test suite", "dry_run": "a dry run", "smoke": "the smoke job", "cost": "the cost estimate"}
    subject = verb.get(name, name)
    if ok:
        return f"{subject} passed"
    if ok is False:
        return f"{subject} failed"
    return f"{subject} has not run"


# ---------------------------------------------------------------------------
# 5. funnel (retrieval)
# ---------------------------------------------------------------------------
def funnel_traces() -> list[str]:
    directory = paths.notes_dir() / "funnel"
    if not directory.exists():
        return []
    return sorted((p.stem for p in directory.glob("*.json")), reverse=True)


def funnel_model(name: str | None = None) -> dict[str, Any]:
    """The four stage bars, the survivors in rank order, and what was dropped.

    The dropped chunks are the point. A funnel that only shows survivors cannot
    answer the question you actually have when retrieval goes wrong, which is
    "why is the obviously relevant paper not in here".
    """
    names = funnel_traces()
    if not names:
        return {
            "traces": [],
            "trace": None,
            "empty_fix": 'python -m tools.paper_search search "..." --json',
        }
    name = name if name in names else names[0]
    path = paths.notes_dir() / "funnel" / f"{name}.json"
    trace, error = _safe(lambda: json.loads(path.read_text(encoding="utf-8")), {})
    trace = trace if isinstance(trace, dict) else {}

    stages = trace.get("stages") or {}
    corpus = stages.get("1_retrieve", {}).get("corpus_chunks")
    retrieved = stages.get("1_retrieve", {}).get("candidates", 0)
    reranked = stages.get("2_rerank", {}).get("out", 0)
    survivors = trace.get("survivors") or []
    kept = stages.get("3_triage", {}).get("returned", len(survivors))

    bars = [
        {"label": f"CORPUS · {corpus if corpus is not None else '?'} CHUNKS", "tone": "corpus", "width": 1.0},
        {"label": f"BM25 + EMBED → {retrieved}", "tone": "corpus", "width": 0.82},
        {"label": f"RERANK → {reranked}", "tone": "rerank", "width": 0.64},
        {"label": f"IN CONTEXT {kept}", "tone": "context", "width": 0.46},
    ]
    dropped = [
        {
            "title": d.get("title") or d.get("id"),
            "score": d.get("rerank_score"),
            "reason": d.get("reason") or d.get("dropped_reason") or "did not survive triage",
        }
        for d in (trace.get("dropped") or [])
    ]
    return {
        "traces": names,
        "trace": {
            "name": name,
            "question": trace.get("question", ""),
            "bars": bars,
            "survivors": [
                {
                    "rank": i + 1,
                    "title": s.get("title") or s.get("id"),
                    "year": s.get("year"),
                    "source": s.get("source"),
                    "score": round(s["rerank_score"], 4)
                    if isinstance(s.get("rerank_score"), (int, float))
                    else None,
                    "reason": s.get("reason", ""),
                }
                for i, s in enumerate(survivors)
            ],
            "dropped": dropped,
            "warnings": trace.get("warnings") or [],
            "expansion": stages.get("0_expand") or {},
        },
        "error": error,
    }


# ---------------------------------------------------------------------------
# 6. run queue / gpu jobs
# ---------------------------------------------------------------------------
QUEUE_STATE_ACCENT = {
    "RUNNING": "ok",
    "WAITING GATE": "attention",
    "DONE": "neutral",
    "FAILED": "broken",
    "QUEUED": "neutral",
}


#: The statuses `core/submit.py` actually writes. Spelled out rather than
#: guessed at, because the failure is silent in the worst way: an unrecognised
#: status falls through to "WAITING GATE", so a fleet of running jobs would
#: render as a queue waiting on a human who has nothing to approve.
RUNNING_STATUSES = ("in_flight", "running", "in_progress")
FAILED_STATUSES = ("failed", "submit_failed", "error")
QUEUED_STATUSES = ("queued", "submitted", "pending")


def _queue_state(run: Any) -> tuple[str, str]:
    """One run's state chip, and the progress bar variant that goes with it."""
    status = str(run.get("status") or "").lower()
    if run.collected:
        if status in FAILED_STATUSES:
            error = run.get("error") or {}
            name = error.get("type") or error.get("code") or "error"
            return f"FAILED · {name}", "failed"
        return "DONE", "done"
    if status in FAILED_STATUSES:
        return "FAILED", "failed"
    if status in RUNNING_STATUSES:
        return "RUNNING", "running"
    if status in QUEUED_STATUSES:
        return "QUEUED", "queued"
    # Submitted but with no status the ledger recognises: the honest reading is
    # that nothing has moved it, which is what a gate looks like.
    return "WAITING GATE", "queued"


def queue_model() -> dict[str, Any]:
    """Runs and campaign candidates in one table.

    Candidates are included because they spend the same GPU dollars against the
    same ceiling; a queue that showed only `runs.jsonl` would show a campaign as
    idle while it burned the budget.
    """
    from core import campaign as campaign_mod, ledger_store as ls

    runs, error = _safe(ls.runs, [])
    rows = []
    for run in reversed(runs or []):
        state, tone = _queue_state(run)
        progress = 1.0 if run.collected else (0.5 if tone == "running" else 0.0)
        rows.append(
            {
                "job": run.id,
                "what": _short(run.get("task") or run.get("spec") or "—", 46),
                "device": run.get("flavor") or run.get("host") or ("smoke" if run.is_smoke else "—"),
                "progress": progress,
                "tone": tone,
                "eta": run.get("eta") or ("—" if run.collected else "unknown"),
                "cost": _usd(run.cost_for_ceiling()),
                "state": state,
                "accent": QUEUE_STATE_ACCENT.get(state.split(" · ")[0], "neutral"),
                "kind": "run",
                "project": run.project,
            }
        )

    campaigns, campaign_error = _safe(campaign_mod.campaigns, {})
    for cid, record in (campaigns or {}).items():
        if str(record.get("status")) != "open":
            continue
        spend, _ = _safe(lambda c=cid: campaign_mod.campaign_spend(c), {})
        rows.insert(
            0,
            {
                "job": cid,
                "what": _short(f"evolve · {record.get('task_dir') or ''}", 46),
                "device": "local",
                "progress": 0.5,
                "tone": "running",
                "eta": f"gen {record.get('generations_run', 0)}",
                "cost": _usd((spend or {}).get("cost_usd", 0.0)),
                "state": "RUNNING",
                "accent": "ok",
                "kind": "campaign",
                "project": record.get("project") or "unassigned",
            },
        )

    return {
        "rows": rows[:RECENT],
        "running": len([r for r in rows if r["tone"] == "running"]),
        "failed": len([r for r in rows if r["tone"] == "failed"]),
        "error": error or campaign_error,
        "empty_fix": "python -m tools.jobs submit --spec <spec> --json",
    }


# ---------------------------------------------------------------------------
# 7. evolve (ShinkaEvolve)
# ---------------------------------------------------------------------------
def evolve_model(campaign_id: str | None = None) -> dict[str, Any]:
    """Population stats, the lineage bars, and the champion diff.

    The lineage is drawn from candidate records rather than from a Shinka
    export, so the window works whether or not the optional dependency is
    installed -- the campaign bookkeeping is ours either way.
    """
    from core import campaign as campaign_mod

    campaigns, error = _safe(campaign_mod.campaigns, {})
    campaigns = campaigns or {}
    if not campaigns:
        return {
            "campaigns": [],
            "campaign": None,
            "empty_fix": "python -m tools.evolve init --task <dir> --json",
            "error": error,
        }

    ids = list(campaigns)
    campaign_id = campaign_id if campaign_id in campaigns else ids[0]
    record = campaigns[campaign_id]
    candidates, cand_error = _safe(lambda: campaign_mod.candidates(campaign_id), [])
    spend, _ = _safe(lambda: campaign_mod.campaign_spend(campaign_id), {})
    top, _ = _safe(lambda: campaign_mod.top_k(campaign_id, 5), [])
    candidates = candidates or []

    # `metrics.combined_score`, not a top-level field: that is Shinka's own
    # contract and what `campaign.top_k` sorts on. Reading the wrong key does
    # not raise -- it silently produces an empty lineage and no champion, which
    # looks exactly like a campaign that has not evaluated anything yet.
    scored = [c for c in candidates if isinstance(_score_of(c), (int, float))]
    best_so_far = float("-inf")
    bars = []
    for candidate in scored:
        score = float(_score_of(candidate))
        is_new_best = score > best_so_far
        best_so_far = max(best_so_far, score)
        bars.append(
            {
                "generation": candidate.get("generation"),
                "score": score,
                "id": candidate.get("id"),
                "tone": "best" if is_new_best else "ordinary",
            }
        )
    if bars:
        champion = max(range(len(bars)), key=lambda i: bars[i]["score"])
        bars[champion]["tone"] = "champion"
        lo = min(b["score"] for b in bars)
        hi = max(b["score"] for b in bars)
        span = (hi - lo) or 1.0
        for bar in bars:
            # A floor of 8%: a bar with the worst score in the population is
            # still a bar, and a zero-height rectangle reads as a missing
            # generation rather than a bad one.
            bar["height"] = round(0.08 + 0.92 * (bar["score"] - lo) / span, 4)

    champion_record = (top or [None])[0]
    baseline = next((c for c in candidates if c.get("generation") in (0, None)), None)
    delta = None
    if champion_record and baseline:
        a, b = _score_of(champion_record), _score_of(baseline)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = round(a - b, 6)

    return {
        "campaigns": ids,
        "campaign": {
            "id": campaign_id,
            "status": record.get("status"),
            "task_dir": record.get("task_dir"),
            "project": record.get("project") or "unassigned",
            "objective": record.get("objective") or record.get("metric") or "combined_score",
            "generations_run": record.get("generations_run", 0),
            "islands": record.get("islands"),
            "migrations": record.get("migrations"),
            "novelty": record.get("novelty"),
            "spend": spend or {},
            "candidates": len(candidates),
            "bars": bars,
            # Flattened here so the window never has to reach into `metrics`:
            # the shape of a candidate record is this module's problem.
            "top": [
                {"id": c.get("id"), "generation": c.get("generation"), "score": _score_of(c)}
                for c in (top or [])
            ],
            "champion": champion_record,
            "champion_score": _score_of(champion_record) if champion_record else None,
            "delta": delta,
            "running": str(record.get("status")) == "open",
            "halt_requested": bool(record.get("halt_requested")),
        },
        "error": error or cand_error,
    }


def _score_of(candidate: Any, metric: str = "combined_score") -> Any:
    if not isinstance(candidate, dict):
        return None
    return (candidate.get("metrics") or {}).get(metric)


def diff_lines(before: str, after: str, *, context: int = 3) -> list[dict[str, str]]:
    """A unified diff as rows, ready to paint. `difflib`, not a dependency."""
    import difflib

    rows: list[dict[str, str]] = []
    for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=context, fromfile="baseline", tofile="champion"
    ):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            kind = "meta"
        elif line.startswith("+"):
            kind = "add"
        elif line.startswith("-"):
            kind = "del"
        else:
            kind = "same"
        rows.append({"kind": kind, "text": line})
    return rows


# ---------------------------------------------------------------------------
# 8. cited papers
# ---------------------------------------------------------------------------
def papers_model(*, filter_name: str = "cited") -> dict[str, Any]:
    """Papers on disk, annotated with what in the ledger depends on them.

    The status chips are the reason this window is not a file listing: "3 claims
    depend on this" is computed by walking every expectation's `basis`, and
    "contradicts exp-…" by intersecting that with the falsified set. A paper the
    agent queued but never read is shown as such rather than omitted.
    """
    from core import ledger_store as ls

    directory = paths.papers_dir()
    on_disk = sorted(p for p in directory.iterdir() if p.is_dir()) if directory.exists() else []

    expectations, error = _safe(ls.expectations, [])
    falsified, _ = _safe(ls.falsified_ids, set())
    expectations = expectations or []
    falsified = falsified or set()

    depends: dict[str, list[str]] = {}
    contradicts: dict[str, list[str]] = {}
    for expectation in expectations:
        eid = str(expectation.get("id") or "")
        for basis in expectation.get("basis") or []:
            key = _paper_key(basis.get("paper"))
            if not key:
                continue
            depends.setdefault(key, []).append(eid)
            if eid in falsified:
                contradicts.setdefault(key, []).append(eid)

    # A workspace with no corpus yet is the normal starting state, not a fault.
    # Surfacing "no local index" in the error strip would put a red card on the
    # papers window of every fresh install, which teaches people to ignore the
    # strip -- and then it is no longer useful for the failures that matter.
    ingested: set[str] = set()
    stats, _ = _safe(_corpus_stats, {})
    for doc in (stats or {}).get("documents", []) or []:
        key = _paper_key(doc)
        if key:
            ingested.add(key)

    rows = []
    for path in on_disk:
        key = _paper_key(path.name)
        claims = depends.get(key, [])
        broken = contradicts.get(key, [])
        read = key in ingested or any(path.glob("*.tex")) or any(path.glob("*.txt"))
        chips = []
        if claims:
            chips.append({"text": f"{len(claims)} CLAIMS DEPEND ON THIS", "tone": "attention"})
        for eid in broken:
            chips.append({"text": f"CONTRADICTS {eid}", "tone": "ok"})
        if not read:
            chips.append({"text": "QUEUED BY GRAD · NOT READ", "tone": "dashed"})
        rows.append(
            {
                "id": path.name,
                "title": _paper_title(path),
                "authors": _paper_authors(path),
                "path": str(path),
                "read": read,
                "claims": claims,
                "chips": chips,
                "cited": bool(claims),
            }
        )

    if filter_name == "cited":
        visible = [r for r in rows if r["cited"]]
    elif filter_name == "read":
        visible = [r for r in rows if r["read"]]
    elif filter_name == "queued":
        visible = [r for r in rows if not r["read"]]
    else:
        visible = rows

    return {
        "rows": visible,
        "all": rows,
        "filter": filter_name,
        "counts": {
            "cited": len([r for r in rows if r["cited"]]),
            "read": len([r for r in rows if r["read"]]),
            "queued": len([r for r in rows if not r["read"]]),
        },
        "error": error,
        "empty_fix": "python -m tools.paper_ingest arxiv <id> --json",
    }


def _corpus_stats() -> dict[str, Any]:
    from core import corpus

    con = corpus.connect(create=False)
    try:
        stats = corpus.stats(con)
        rows = con.execute("SELECT id FROM documents").fetchall()
        return {**stats, "documents": [r[0] for r in rows]}
    finally:
        con.close()


def _paper_key(value: Any) -> str:
    """Normalise the many ways one paper gets named into one key.

    A basis cites "arXiv:2001.08361", a directory is called `2001.08361`, and
    the corpus stores `arxiv_2001.08361`. Comparing them raw means the claim
    count is always zero, which looks like working software.
    """
    if not value:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^(arxiv[:_/]?)", "", text)
    return re.sub(r"[^a-z0-9.]", "", text)


def _paper_title(path: Path) -> str:
    for candidate in ("title.txt", "meta.json"):
        target = path / candidate
        if not target.exists():
            continue
        try:
            if candidate.endswith(".json"):
                data = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("title"):
                    return str(data["title"])
            else:
                return target.read_text(encoding="utf-8").strip().splitlines()[0]
        except (OSError, json.JSONDecodeError, IndexError):
            continue
    return path.name


def _paper_authors(path: Path) -> str:
    target = path / "meta.json"
    if not target.exists():
        return f"arXiv:{path.name}"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"arXiv:{path.name}"
    if not isinstance(data, dict):
        return f"arXiv:{path.name}"
    authors = data.get("authors")
    if isinstance(authors, list):
        authors = ", ".join(str(a) for a in authors[:3]) + ("et al." if len(authors) > 3 else "")
    year = data.get("year")
    return f"{authors or '—'} · arXiv:{path.name}" + (f" · {year}" if year else "")


# ---------------------------------------------------------------------------
# 9. wiki + references
# ---------------------------------------------------------------------------
def wiki_model() -> dict[str, Any]:
    """The generated wiki, whether it still matches the source, and its scope.

    The design shows a chat pane with a references rail. The references here are
    the wiki's own scope entries and the files that changed since it was built,
    because that is what actually exists: `tools/wiki.py` deliberately does not
    enable `repowiki scan` (the LLM half reads `ANTHROPIC_API_KEY`, the exact
    variable the credential scrub deletes), so there is no answer engine to
    quote. Questions go to the agent with an `@wiki` mention instead.
    """
    from core import jsonl
    from tools import wiki as wiki_tool

    manifest_path = wiki_tool.output_dir() / "manifest.json"
    # Guarded for the same reason as the preflight records: `read_json` returns
    # `None` for missing and malformed, but lets `UnicodeDecodeError` and
    # `OSError` through, and "the manifest is unreadable" must not render as
    # "no wiki has been generated yet".
    manifest, manifest_error = _safe(lambda: jsonl.read_json(manifest_path))
    if not isinstance(manifest, dict):
        return {
            "built": False,
            "stale": False,
            "scopes": [],
            "changed": [],
            "error": manifest_error,
            "empty_fix": "python -m tools.wiki map --json",
        }

    current, error = _safe(wiki_tool.source_hash, {})
    current = current or {}
    recorded = manifest.get("source") or {}
    before, after = recorded.get("files") or {}, current.get("files") or {}
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))

    scopes = manifest.get("scopes")
    if isinstance(scopes, dict):
        scope_rows = [{"name": k, "entries": v} for k, v in sorted(scopes.items())]
    else:
        scope_rows = [{"name": s, "entries": None} for s in (current.get("scope") or [])]

    html = wiki_tool.output_dir() / "index.html"
    return {
        "built": True,
        "stale": bool(changed),
        "generated_at": manifest.get("generated_at"),
        "output_dir": str(wiki_tool.output_dir()),
        "html": str(html) if html.exists() else None,
        "source_hash": current.get("hash"),
        "recorded_hash": recorded.get("hash"),
        "scopes": scope_rows,
        "changed": changed[:50],
        "changed_total": len(changed),
        "error": error,
        "empty_fix": "python -m tools.wiki map --json",
    }


# ---------------------------------------------------------------------------
# 10. paper editor (LaTeX)
# ---------------------------------------------------------------------------
SECTION_RE = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}")


def editor_model(project_id: str | None = None) -> dict[str, Any]:
    """The source, its outline, and every claim the build gate would refuse.

    The design's mock invents `\\gradcite{run-…}` and `\\gradexp{exp-…}`. The
    real macro already in `core/report.py` is `\\gradnum{key}` resolving through
    `claims.json` to a `(run_id, quantity)` *and its value* -- strictly stronger,
    because it catches a citation that points at the right run and prints the
    wrong number. So the window renders the real one and the outline warning
    counts findings from `report.check_*` rather than a regex of its own.
    """
    from core import budget as budget_mod, report as report_mod

    project_id = project_id or (_safe(budget_mod.current_project)[0] or "")
    if not project_id:
        return {
            "project": None,
            "exists": False,
            "empty_fix": "python -m tools.budget new --project <id> --json",
        }

    targets = report_mod.paths_for(project_id)
    tex_path = targets["tex"]
    if not tex_path.exists():
        return {
            "project": project_id,
            "exists": False,
            "empty_fix": f"python -m tools.report draft --project {project_id} --json",
        }

    tex, read_error = _safe(lambda: tex_path.read_text(encoding="utf-8"), "")
    tex = tex or ""
    claims, _ = _safe(lambda: report_mod.load_claims(project_id), {})
    claims = claims or {}
    bib_path = targets["bib"]
    bib, _ = _safe(
        lambda: report_mod.parse_bib(bib_path.read_text(encoding="utf-8")) if bib_path.exists() else {},
        {},
    )

    findings: list[dict[str, Any]] = []
    for check in (
        lambda: report_mod.check_claims(tex, claims),
        lambda: report_mod.check_citations(tex, bib or {}),
        lambda: report_mod.check_latex(tex),
    ):
        result, error = _safe(check, [])
        findings.extend(result or [])
        if error:
            findings.append({"rule": "internal", "problem": error, "fix": "read the app log"})

    lines = tex.splitlines()
    outline = []
    for number, line in enumerate(lines, start=1):
        match = SECTION_RE.search(line)
        if match:
            outline.append({"title": match.group(1), "line": number})

    flagged = {f.get("line") for f in findings if isinstance(f.get("line"), int)}
    return {
        "project": project_id,
        "exists": True,
        "tex_path": str(tex_path),
        "pdf_path": str(targets["pdf"]),
        "pdf_exists": targets["pdf"].exists(),
        "source": tex,
        "lines": lines,
        "flagged_lines": flagged,
        "outline": outline,
        "findings": findings,
        "blocking": len(findings),
        "claims": claims,
        "cited_runs": sorted(_safe(lambda: report_mod.cited_run_ids(tex, claims), set())[0] or set()),
        "error": read_error,
        "warning": (
            f"{len(findings)} finding(s). Grad blocks the build until each asserted number is "
            "bound to a run and a quantity."
        )
        if findings
        else None,
    }


def highlight_tex(line: str) -> list[dict[str, str]]:
    """Split one source line into spans so the macros can be painted.

    Deliberately not a LaTeX parser: three token classes (a `\\gradnum` macro, an
    ordinary command, a comment) is all the design asks for, and anything more
    would be a syntax highlighter pretending the preview is an editor.
    """
    spans: list[dict[str, str]] = []
    index = 0
    pattern = re.compile(r"(\\gradnum\{[^}]*\})|(%[^\n]*$)|(\\[A-Za-z@]+)")
    for match in pattern.finditer(line):
        if match.start() > index:
            spans.append({"kind": "text", "text": line[index : match.start()]})
        if match.group(1):
            spans.append({"kind": "gradnum", "text": match.group(1)})
        elif match.group(2):
            spans.append({"kind": "comment", "text": match.group(2)})
        else:
            spans.append({"kind": "command", "text": match.group(3)})
        index = match.end()
    if index < len(line):
        spans.append({"kind": "text", "text": line[index:]})
    return spans or [{"kind": "text", "text": line}]


# ---------------------------------------------------------------------------
# chat: the message anatomy the transcript is parsed into
# ---------------------------------------------------------------------------
def parse_message(text: str) -> list[dict[str, Any]]:
    """Split an assistant turn into the blocks the design draws differently.

    Prose, tool calls, expectation cards and gates each have their own anatomy
    in the handoff, and the transcript is markdown. So: fenced shell blocks
    become tool cards, an `EXPECTATION REGISTERED` line introduces an expectation
    card, a `GATE` line introduces a gate card, everything else stays prose. The
    parser is small on purpose -- a richer one would start inventing structure
    the agent did not intend.
    """
    blocks: list[dict[str, Any]] = []
    for index, part in enumerate(text.split("```")):
        if index % 2:
            language, _, body = part.partition("\n")
            language = language.strip()
            body = body.strip()
            if language in ("bash", "sh", "console", "shell"):
                first = body.splitlines()[0] if body else "command"
                blocks.append({"kind": "tool", "title": _short(first, 80), "text": body})
            else:
                blocks.append({"kind": "code", "language": language or "text", "text": body})
            continue
        for chunk in _split_cards(part):
            blocks.append(chunk)
    return [b for b in blocks if b.get("text") or b.get("rows")]


_CARD_RE = re.compile(
    r"^(?P<kind>EXPECTATION REGISTERED|GATE — YOUR CALL|GATE)\b[:\s]*(?P<id>\S*)\s*$",
    re.MULTILINE,
)


def _split_cards(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    index = 0
    for match in _CARD_RE.finditer(text):
        if match.start() > index:
            out.append({"kind": "text", "text": text[index : match.start()].strip()})
        rest = text[match.end() :]
        # The header match stops *before* its own newline (MULTILINE `$`), so
        # exactly one is consumed here. Stripping all of them would swallow the
        # blank line that means "this card has no rows".
        lead = 1 if rest.startswith("\n") else 0
        body, consumed = _card_body(rest[lead:])
        kind = "expectation" if match.group("kind").startswith("EXPECTATION") else "gate"
        out.append({"kind": kind, "id": match.group("id"), "rows": body, "text": " "})
        index = match.end() + lead + consumed
    if index < len(text):
        out.append({"kind": "text", "text": text[index:].strip()})
    return out


def _card_body(text: str) -> tuple[list[tuple[str, str]], int]:
    """`key: value` lines directly after a card header, until a blank line."""
    rows: list[tuple[str, str]] = []
    consumed = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            consumed += len(line)
            break
        key, sep, value = stripped.partition(":")
        if not sep:
            break
        rows.append((key.strip(), value.strip()))
        consumed += len(line)
    return rows, consumed


def figures_in(text: str) -> list[str]:
    """Figure paths the agent named, resolved against the workspace."""
    found: list[str] = []
    for token in text.replace("(", " ").replace(")", " ").split():
        if not token.endswith(".png"):
            continue
        normalised = token.replace("\\", "/")
        if "figures" not in normalised:
            continue
        path = Path(token)
        if not path.is_absolute():
            path = paths.root() / path
        if path.exists() and str(path) not in found:
            found.append(str(path))
    return found
