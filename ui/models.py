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
import os
import re
import time as _time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

from core import ledger_store as ls_mod, paths

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


#: The three ceilings a project carries: the resource name `core/budget.py` uses,
#: the `tools.budget raise` flag it maps to, and how to say it on screen.
#:
#: One list, because there were two. `ui/shell.py` kept its own copy for the
#: menu's raise controls, keyed by flag, while every meter in the app was keyed
#: by resource -- so the two spellings of one idea were maintained in different
#: files and neither knew the other existed.
CEILINGS: tuple[tuple[str, str, str, str], ...] = (
    ("gpu_usd", "gpu-usd", "GPU $", "dollars of remote compute"),
    ("quota_tokens", "quota-tokens", "tokens", "subscription tokens, all roles"),
    ("credits_usd", "credits-usd", "credits $", "the reranker and embeddings"),
)

#: How each resource is written. Tokens are counted, not priced, and rendering
#: 4.2M of them as `$4,200,000.00` was the specific thing this separates.
_CEILING_FORMAT = {"gpu_usd": _usd, "quota_tokens": _tokens, "credits_usd": _usd}


def _project_memory(project_id: str) -> dict[str, Any]:
    """Which of the six per-project documents exist. Six `exists()`, no reads.

    "Scaffolded but empty" and "never scaffolded" are different states and the
    window says which: the first is a project nobody has written in yet, the
    second is one whose `new` failed to scaffold -- `tools/budget.py` guards that
    step precisely because it must not fail the creation, and this is where the
    consequence becomes visible instead of being discovered by `project sync`.
    """
    from core import projects as projects_mod

    directory, error = _safe(lambda: projects_mod.resolve_dir(project_id))
    if directory is None:
        return {"dir": None, "present": [], "missing": list(projects_mod.DOCS), "error": error}
    present, listing_error = _safe(
        lambda: [name for name in projects_mod.DOCS if (directory / name).exists()], []
    )
    present = present or []
    return {
        "dir": str(directory),
        "present": present,
        "missing": [name for name in projects_mod.DOCS if name not in present],
        "scaffolded": bool(present),
        "error": error or listing_error,
    }


def projects_model() -> dict[str, Any]:
    """Every project in this folder: what bounds it, what it has spent, and
    whether anything has been written down about it.

    This replaced a section of the `project ▾` dialog and is deliberately not the
    same data. A menu row had space for an id, a title and one summary line, and
    the ceiling controls under the list addressed only the *selected* project --
    so reading what bounds a project you were not on meant switching to it first,
    which charges nothing but reloads every window in the app.

    Wrapped reader by reader, for `workspaces_model`'s reason: this window has to
    render when the workspace is wrong. `status` folds the whole run ledger for
    one project, so it is caught per row -- one project whose spend will not
    compute says so in its own row instead of taking the list down with it.
    """
    from core import budget as budget_mod, config as config_mod, settings as settings_mod

    root, root_error = _safe(lambda: str(paths.root()), "")
    current, _ = _safe(budget_mod.current_project)
    records, projects_error = _safe(budget_mod.projects, {})
    cfg, _ = _safe(config_mod.load)

    # What a role resolves to with the project layer removed. Every project but
    # the selected one has *someone else's* project layer in effect, so this is
    # the only honest thing to show beside an override that is not set.
    workspace_models: dict[str, str] = {}
    for role in config_mod.MODEL_ROLES:
        value, _ = _safe(lambda r=role: cfg.model_for(r, project=False) if cfg else "", "")
        workspace_models[role] = value or config_mod.DEFAULTS["models"][role]

    rows: list[dict[str, Any]] = []
    for project_id, record in sorted((records or {}).items()):
        state, state_error = _safe(lambda pid=project_id: budget_mod.status(pid), {})
        resources = (state or {}).get("resources") or {}
        ceilings = []
        for resource, flag, caption_text, hint in CEILINGS:
            node = resources.get(resource) or {}
            ceiling = node.get("ceiling")
            render = _CEILING_FORMAT[resource]
            ceilings.append(
                {
                    "resource": resource,
                    "flag": flag,
                    "caption": caption_text,
                    "hint": hint,
                    "ceiling": ceiling,
                    "spent": node.get("spent", 0.0),
                    "fraction": node.get("fraction"),
                    "over": bool(node.get("over")),
                    "set": ceiling is not None,
                    "label": (
                        f"{render(node.get('spent', 0.0))} spent · no ceiling"
                        if ceiling is None
                        else f"{render(node.get('spent', 0.0))} / {render(ceiling)}"
                    ),
                }
            )
        rows.append(
            {
                "id": project_id,
                "title": _short(record.get("title") or "", 90),
                "status": record.get("status") or "open",
                "current": project_id == current,
                "payer": record.get("payer"),
                # The date alone. A project list is read for "which of these am I
                # still working on", and a timestamp to the second answers a
                # question nobody asked while costing the row half its width.
                "created": (record.get("created_at") or "")[:10],
                "spend": _spend_line(state or {}),
                "run_count": (state or {}).get("run_count", 0),
                "over_budget": (state or {}).get("over_budget") or [],
                "raise_count": len(record.get("raises") or []),
                "ceilings": ceilings,
                # What this project overrides about how it is run, and what each
                # role would be without it. The model per role is the main lever
                # on cost and quality, which is exactly why it should be able to
                # differ between a cheap exploratory project and one being
                # written up.
                "models": [
                    {
                        "role": role,
                        "override": (record.get("models") or {}).get(role),
                        "workspace": workspace_models[role],
                        "effective": (record.get("models") or {}).get(role)
                        or workspace_models[role],
                    }
                    for role in config_mod.MODEL_ROLES
                ],
                "override_count": len(record.get("models") or {}),
                "backend": record.get("backend"),
                "configured_count": len(record.get("configured") or []),
                # A project with no ceilings bounds nothing and every gate that
                # reads one passes silently. Surfaced as a flag so the window can
                # say it where it is true, rather than in a caption under a form.
                "unbounded": not any(c["set"] for c in ceilings),
                "memory": _project_memory(project_id),
                "error": state_error,
            }
        )

    # Whether the machine half of setup still has something in it. Cheap -- one
    # credential-store read -- and it is what lets the create form point at the
    # wizard instead of containing it.
    needs_setup, _ = _safe(setup_needed, False)

    return {
        "root": root,
        "rows": rows,
        "current_project": current,
        "needs_setup": bool(needs_setup),
        "known_models": list(settings_mod.KNOWN_MODELS),
        "known_backends": list(settings_mod.BACKENDS),
        "count": len(rows),
        "open_count": len([r for r in rows if r["status"] != "closed"]),
        "unbounded": [r["id"] for r in rows if r["unbounded"] and r["status"] != "closed"],
        "error": root_error or projects_error,
    }


def update_model() -> dict[str, Any]:
    """What the project menu says about updating, read from the cache only.

    Never runs git. The background thread in `ui/app.py:_start_update_check`
    owns the network call and writes `update.json`; this reads it. A model that
    fetched would freeze the window for as long as the remote took, every time
    someone opened the menu to switch project.

    So the shape here is "what was true when we last looked", and it says when
    that was. An installation that has never managed a check renders as unknown
    rather than as up to date -- claiming the latter on no evidence is the one
    answer that would stop someone looking.
    """
    from core import update as update_mod, version as version_mod  # noqa: PLC0415

    identity, error = _safe(version_mod.identity, {})
    cached, _ = _safe(update_mod.read_cache, {})
    cached = cached or {}
    age, _ = _safe(update_mod.cache_age_s, float("inf"))
    target = cached.get("target") or {}
    blockers = cached.get("blockers") or []
    warnings = cached.get("warnings") or []

    return {
        "installed": version_mod.label(identity or {}),
        "commit": (identity or {}).get("commit"),
        "dirty": bool((identity or {}).get("dirty")),
        "is_checkout": (identity or {}).get("source") == "git",
        "available": bool(cached.get("available")) and not blockers,
        "target": target.get("tag"),
        "behind": cached.get("behind") or 0,
        "needs_reinstall": bool(cached.get("needs_reinstall")),
        "blockers": [
            {"message": _short(b.get("message"), 160), "fix": _short(b.get("fix"), 160)}
            for b in blockers
            if isinstance(b, dict)
        ],
        "warnings": [
            {"message": _short(w.get("message"), 160), "fix": _short(w.get("fix"), 160)}
            for w in warnings
            if isinstance(w, dict)
        ],
        "checked": "never" if age == float("inf") else _ago(age),
        "stale": age >= update_mod.CHECK_INTERVAL_S,
        "error": error or cached.get("fetch_error"),
    }


def _ago(seconds: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def _spend_line(state: dict[str, Any]) -> str:
    """One line per project: what it has spent against what it may.

    A project with no ceilings is the common case -- they are optional -- and it
    says so rather than rendering an empty bar, which would read as "nothing
    spent" when it means "nothing to exceed".
    """
    resources = state.get("resources") or {}
    parts: list[str] = []
    # Derived from `CEILINGS` rather than listed again. This held its own copy of
    # the three resources and their formatters, which is the third place that
    # list has existed -- and a fourth ceiling added to `core/budget.py` would
    # have appeared in every meter in the app except this line.
    for name, _flag, _caption, _hint in CEILINGS:
        render = _CEILING_FORMAT[name]
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
        # `is None`, not `or`: a configured ceiling of 0 is a deliberate "no
        # credit spend in this session", and coercing it to the 8.0 default
        # silently granted eight dollars nobody asked for.
        ceiling = 8.0 if raw is None else raw

    window, error = _safe(lambda: _session_window(hours=5), {})
    window = window or {}
    used = float(window.get("credits_usd", 0.0))
    # The folder, for the appbar's `workspace ▾`. Read here rather than from
    # `workspaces_model`, which folds every project and its spend to answer a
    # question the title bar is not asking -- and which the title bar redraws on
    # every tick.
    root, root_error = _safe(lambda: paths.root(), None)
    return {
        "project": project or "unassigned",
        "root": str(root) if root else "",
        # The basename. An absolute path does not fit an appbar cell, and the
        # full one is the button's tooltip.
        "root_name": root.name if root else "—",
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
        "error": project_error or error or root_error,
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
    # Weighted, like the ceiling. This meter is meant to answer "how much of the
    # rolling window have I used", and cache reads are most of what draws on it
    # -- counting only input + output drew a bar that barely moved through a
    # session that was in fact consuming the window steadily.
    weight = quota_log.weights()
    chat = tool = 0.0
    chat_tokens = tool_tokens = 0
    tokens = 0
    oldest: _dt.datetime | None = None
    for entry in quota_log.entries():
        at = _iso(entry.get("at"))
        if not at or at < cutoff:
            continue
        oldest = at if oldest is None or at < oldest else oldest
        credits = float(entry.get("credits_usd", 0.0) or 0.0)
        # Kept fractional and rounded once, at the bottom. Rounding each entry
        # first costs up to half a token per row, and this window folds every
        # record in five hours -- an error that grows with how busy the window
        # was is exactly the wrong shape for a meter about how busy it was.
        entry_tokens = quota_log.billable(entry, weight)
        tokens += entry_tokens
        if str(entry.get("stage") or "") == quota_log.STAGE_MAIN:
            chat += credits
            chat_tokens += entry_tokens
        else:
            tool += credits
            tool_tokens += entry_tokens
    total = chat + tool
    resets_in = "—"
    if oldest is not None:
        remaining = (oldest + _dt.timedelta(hours=hours)) - now
        if remaining.total_seconds() > 0:
            hrs, rem = divmod(int(remaining.total_seconds()), 3600)
            resets_in = f"{hrs}h {rem // 60:02d}m"
    # The split falls back to tokens when no credits were spent in the window.
    # Chat costs subscription quota and never credits, so splitting the strip by
    # `credits_usd` alone made the chat segment structurally zero: the meter
    # claimed to show "what chat spent and what tools spent" while only ever
    # drawing the second. Dollars stay the unit when there are dollars, because
    # mixing two currencies in one bar is worse than either.
    token_total = chat_tokens + tool_tokens
    if total:
        chat_fraction = chat / total
        tool_fraction = tool / total
    elif token_total:
        chat_fraction = chat_tokens / token_total
        tool_fraction = tool_tokens / token_total
    else:
        chat_fraction = tool_fraction = 0.0
    return {
        "credits_usd": total,
        "chat_usd": chat,
        "tool_usd": tool,
        # Rounded once, on the way out. The fractions above are computed from the
        # unrounded figures, so the split does not shift with the rounding.
        "chat_tokens": round(chat_tokens),
        "tool_tokens": round(tool_tokens),
        "chat_fraction": chat_fraction,
        "tool_fraction": tool_fraction,
        "split_basis": "credits" if total else ("tokens" if token_total else "empty"),
        "tokens": round(tokens),
        "resets_in": resets_in,
    }


# ---------------------------------------------------------------------------
# the context meter
# ---------------------------------------------------------------------------
#: Fractions of the way to compaction at which the chip changes tone. Below the
#: first it is ordinary; past the second, compaction is the next thing that will
#: happen and the strip should say so before it does rather than after.
CONTEXT_WARN = 0.75
CONTEXT_NEAR = 0.92


def _reading(source: Any, key: str) -> int | None:
    """A non-negative integer out of a `get_context_usage` payload, or None.

    None and 0 are kept distinct all the way through this file, which is the
    whole discipline of the context meter: "I could not read it" and "there is
    nothing in it" are opposite facts and only one of them means there is room.
    Every caller here decides for itself which way to fail.
    """
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def context_model(usage: Any, *, compact_at: int = 0) -> dict[str, Any]:
    """What the statusline's context chip shows, from one `get_context_usage`.

    Pure, and separate from the call that produces `usage`, because the
    interesting part is not the request -- it is which limit the fraction is
    measured against. There are two, and they mean different things:

    * `maxTokens` is where the *CLI* will compact, which a live session reports
      as 967,000 of a 1,000,000 window. By the time that matters, every tool
      round-trip has been re-reading most of a million cached tokens for a long
      while, and the meter has been reading "ok" throughout.
    * `compact_at` is where *Grad* will compact, from `[agent]
      compact_at_tokens`. When it is set it is always the lower of the two, and
      it is the one worth drawing, because it is the one that is going to fire.

    So the fraction is against whichever limit will actually be reached first,
    and `limit_source` says which one that was -- a meter reading 40% means two
    quite different things at a 300k threshold and at a 967k one.

    `usage` may be None (no client yet, or the call failed). That is a real and
    ordinary state -- the meter is drawn as "—" rather than as zero, because a
    context of zero and an unknown context look identical at a glance and only
    one of them is worth acting on.
    """
    tokens = _reading(usage, "totalTokens") if isinstance(usage, dict) else None
    if tokens is None:
        # Not only "no dict yet". A reading whose `totalTokens` is missing or is
        # not a number is *also* unknown, and it used to land here as a confident
        # zero -- which is the one reading this function's docstring says must
        # never happen, drawn by this function. A meter reporting "ctx 0 · 0%"
        # for a session it cannot measure is worse than one reporting nothing,
        # because it invites exactly the conclusion that there is plenty of room.
        return {
            "known": False, "tokens": 0, "limit": 0, "fraction": 0.0,
            "label": "ctx —", "tone": "", "limit_source": "unknown",
            "detail": "no context reading yet — it arrives once a session is connected",
            "categories": [],
        }

    ceiling = _reading(usage, "maxTokens") or _reading(usage, "rawMaxTokens") or 0
    source = "cli"
    if compact_at and (not ceiling or compact_at < ceiling):
        ceiling, source = compact_at, "grad"
    fraction = min(1.0, tokens / ceiling) if ceiling else 0.0

    tone = ""
    if fraction >= CONTEXT_NEAR:
        tone = "attention"
    elif fraction >= CONTEXT_WARN:
        tone = "warn"

    # Built with a loop rather than a comprehension because a bare
    # `int(c.get("tokens"))` raises on a non-numeric value, and this is drawn
    # from a timer -- one odd category in one reading would not produce one bad
    # tooltip, it would raise several times a second for as long as the session
    # lasted. A category that cannot be read is skipped.
    categories: list[dict[str, Any]] = []
    for entry in usage.get("categories") or []:
        if not isinstance(entry, dict):
            continue
        size = _reading(entry, "tokens")
        if not size:
            continue
        name = str(entry.get("name") or "?").strip()
        # `Free space` is a category in the CLI's own breakdown and is the
        # complement of everything else, so listing it in a tooltip about what
        # is *using* the context is worse than noise -- it is always the largest
        # entry and it is not a consumer.
        if name.lower() == "free space":
            continue
        categories.append({"name": name or "?", "tokens": size})
    categories.sort(key=lambda c: -c["tokens"])

    where = "Grad compacts" if source == "grad" else "the CLI compacts"
    detail = f"{tokens:,} of {ceiling:,} tokens ({where} here)" if ceiling else f"{tokens:,} tokens"
    if categories:
        detail += " — " + ", ".join(f"{c['name']} {_tokens(c['tokens'])}" for c in categories[:4])
    return {
        "known": True,
        "tokens": tokens,
        "limit": ceiling,
        "fraction": fraction,
        "label": f"ctx {_tokens(tokens)}" + (f" · {fraction * 100:.0f}%" if ceiling else ""),
        "tone": tone,
        "limit_source": source,
        "detail": detail,
        "categories": categories,
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
#: What each credential unlocks, and which part of the system it belongs to. The
#: text matters as much as the group: "missing" is not the same fact for a token
#: that gates GPU submission as for one that raises a rate limit.
#:
#: The group replaced a bare `required` flag, and the flag was making a claim it
#: could not support. `hf_token` was marked required, so a user who had chosen
#: Kaggle -- the free backend, and the one a new user is most likely to start on
#: -- was shown a red MISSING for a token they will never need. What is actually
#: true is that HF Jobs needs it, which is a fact about a *backend*, and
#: `tools/setup.py:readiness` is where that belongs.
CREDENTIAL_GROUPS: dict[str, str] = {
    # Nothing at all works without this one.
    "agent": "the agent itself",
    # Needed by one backend each, and only if you use that backend.
    "backend": "where runs execute",
    # Buys or widens retrieval. Everything here degrades rather than fails.
    "retrieval": "papers and embeddings",
    "extras": "nice to have",
}

CREDENTIAL_NOTES: dict[str, tuple[str, str]] = {
    "claude_oauth_token": (
        "the agent's own loop, the funnel's Haiku stages and the mutation operator",
        "agent",
    ),
    "hf_token": ("Hugging Face Jobs — submitting and collecting runs", "backend"),
    # The eighth credential. It was in `credentials.ALL` and not here, so the
    # panel drew it with an empty purpose column -- the one credential whose row
    # said nothing about what it was for, and the one belonging to the backend
    # that costs nothing to try.
    "kaggle_key": (
        "Kaggle kernels — the free GPU/TPU backend; useless without the username",
        "backend",
    ),
    "voyage_key": ("the reranker and the local index's embeddings (costs credits)", "retrieval"),
    "openrouter_key": (
        "optional second rail for the reranker; Voyage is used by default",
        "retrieval",
    ),
    "asta_api_key": ("raises Asta's rate limits; discovery works without it", "retrieval"),
    "s2_api_key": (
        "Semantic Scholar direct — only issued to institutional addresses",
        "retrieval",
    ),
    "context7_key": ("raises Context7's rate limits; lookups work without it", "extras"),
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
        purpose, group = CREDENTIAL_NOTES.get(name, ("", "extras"))
        # Only the agent's own token is unconditionally required: without it
        # nothing runs at all. A backend credential is required *for that
        # backend*, which `tools/setup.py:readiness` reports against the backend
        # rather than against the key.
        required = group == "agent"
        rows.append(
            {
                "name": name,
                "stored": bool(stored),
                "purpose": purpose,
                "group": group,
                "group_label": CREDENTIAL_GROUPS.get(group, group),
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
# 0a2. setup
# ---------------------------------------------------------------------------
#: The steps of the machine half, in order. `id`, the caption, and the one line
#: saying what answering it buys.
#:
#: Machine-scoped, all four of them, which is why they are not asked again when
#: a project is created. A project's own step -- ceilings, payer, the backend it
#: reaches for -- is Stage 4 and lives with the project, because it is the only
#: part of this that differs per project.
SETUP_STEPS: tuple[tuple[str, str, str], ...] = (
    ("token", "subscription", "the OAuth token every model call authenticates with"),
    ("models", "models", "which model runs which role"),
    ("context", "context", "how much conversation the agent carries before it compacts"),
    ("backends", "backends", "where a training run actually executes"),
    ("extras", "extras", "optional keys that widen retrieval or raise a rate limit"),
)


def setup_needed() -> bool:
    """Whether this install has nothing to run with.

    Deliberately narrow: only the subscription token. An unconfigured backend
    means no remote training, which is a real limitation and not a reason to put
    a wizard in front of someone who wanted to read a ledger -- but with no token
    the main loop cannot authenticate, so the four windows a fresh workspace
    opens are four windows that can do nothing.

    Both places the token can be are checked, because both work: `credentials`
    is the durable one and the environment is what a terminal export leaves.
    Never raises -- `present` already swallows an unreachable store, and a
    machine with no keyring is a machine with nothing configured, which is the
    same answer.
    """
    from core import credentials as credentials_mod

    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return False
    stored, _ = _safe(lambda: credentials_mod.present(credentials_mod.CLAUDE_TOKEN), False)
    return not stored


def setup_model() -> dict[str, Any]:
    """What is configured, what is not, and what each answer would buy.

    Wrapped reader by reader like every other model here, and this one has the
    strongest claim to it: the whole point of the window is to be usable on a
    machine where nothing works yet, which is exactly the machine where a reader
    is most likely to fail.
    """
    from core import config as config_mod, credentials as credentials_mod, settings as settings_mod
    from tools import setup as setup_tool

    cfg, cfg_error = _safe(config_mod.load)
    stored, cred_error = _safe(credentials_mod.status, {})
    stored = stored or {}

    # -- the token ---------------------------------------------------------
    ambient = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
    in_store = bool(stored.get("claude_oauth_token"))
    token = {
        "stored": in_store,
        "ambient": ambient,
        "state": "stored" if in_store else ("environment" if ambient else "missing"),
        "ready": in_store or ambient,
        # The distinction that matters on Windows: a token exported in a shell
        # authenticates that shell, and the desktop shortcut launches from
        # Explorer with whatever the user made persistent -- usually nothing.
        "durable": in_store,
        "name": credentials_mod.CLAUDE_TOKEN,
        "mint": "claude setup-token",
    }

    # -- the six roles -----------------------------------------------------
    overlay_models, _ = _safe(settings_mod.models, {})
    overlay_models = overlay_models or {}
    roles = []
    for role in config_mod.MODEL_ROLES:
        configured = ((getattr(cfg, "user", None) or {}).get("models") or {}).get(role)
        chosen, _ = _safe(lambda r=role: cfg.model_for(r) if cfg else "", "")
        roles.append(
            {
                "role": role,
                "model": chosen,
                "source": (
                    "setup" if role in overlay_models else ("config" if configured else "default")
                ),
                "default": config_mod.DEFAULTS["models"][role],
                "overridden": role in overlay_models,
            }
        )

    # -- how much conversation the agent carries ----------------------------
    context = _context_setting(cfg)

    # -- where a run executes ----------------------------------------------
    backends, backend_error = _safe(lambda: setup_tool.readiness(cfg) if cfg else [], [])
    backends = backends or []
    hosts, _ = _safe(lambda: sorted(cfg.hosts) if cfg else [], [])
    kaggle_account, _ = _safe(_kaggle_account, {})

    credentials_panel = credentials_model()
    shadowing, _ = _safe(lambda: settings_mod.shadowing(cfg) if cfg else [], [])

    steps = []
    for step_id, caption, hint in SETUP_STEPS:
        if step_id == "token":
            ready, detail = token["ready"], token["state"]
        elif step_id == "models":
            # Always resolvable -- there are defaults for all six. The step is
            # here to be adjusted, not to be satisfied, so it never blocks.
            ready, detail = True, f"{len(overlay_models)} of {len(roles)} chosen here"
        elif step_id == "context":
            # Always satisfied: there is a default, and "off" is a real answer
            # rather than an unanswered question. The step is here to be adjusted.
            ready = True
            detail = (
                f"compacts at {context['tokens']:,}" if context["enabled"] else "compaction off"
            )
        elif step_id == "backends":
            usable = [b["backend"] for b in backends if b["ready"]]
            ready = bool(usable)
            detail = ", ".join(usable) if usable else "none configured"
        else:
            optional = [r for r in credentials_panel["rows"] if r["group"] in ("retrieval", "extras")]
            ready = True
            detail = f"{len([r for r in optional if r['stored']])} of {len(optional)} stored"
        steps.append(
            {
                "id": step_id,
                "caption": caption,
                "hint": hint,
                "ready": ready,
                "detail": detail,
                "tone": "ok" if ready else "attention",
            }
        )

    return {
        "steps": steps,
        "token": token,
        "roles": roles,
        "known_models": list(settings_mod.KNOWN_MODELS),
        "context": context,
        "backends": backends,
        "default_backend": _safe(settings_mod.default_backend)[0],
        "known_backends": list(settings_mod.BACKENDS),
        "hosts": hosts,
        "kaggle": kaggle_account,
        "credentials": credentials_panel,
        "settings_path": str(_safe(settings_mod.path, "")[0] or ""),
        "config_path": str(_safe(paths.config_path, "")[0] or ""),
        "shadowing": shadowing or [],
        # Nothing here blocks the app; this is what the appbar and the first-run
        # arrangement ask about.
        "complete": token["ready"] and any(b["ready"] for b in backends),
        "error": cfg_error or cred_error or backend_error,
    }


#: The choices the context step offers as buttons, in tokens. Not a restriction
#: -- the field beside them takes any number `settings.AGENT_SETTINGS` accepts --
#: and not evenly spaced, because the interesting region is the low end: the
#: difference between 100k and 200k is a different working style, the difference
#: between 800k and 900k is which side of the CLI's own wall you hit first.
CONTEXT_PRESETS: tuple[int, ...] = (100_000, 200_000, 300_000, 500_000, 800_000)


def _context_setting(cfg: Any) -> dict[str, Any]:
    """Where Grad compacts, and where that number came from.

    Not the context *window*: that is the model's, the CLI reports it as
    1,000,000 with its own threshold at 967,000, and `core/compaction.py`
    explains why neither number is reachable from here. This is the one Grad
    owns, and it is the one that decides how much conversation the agent is
    actually carrying.
    """
    from core import compaction, config as config_mod, settings as settings_mod

    overlay, _ = _safe(settings_mod.agent, {})
    overlay = overlay or {}
    tokens, error = _safe(lambda: compaction.threshold(cfg) if cfg else 0, 0)
    tokens = int(tokens or 0)
    configured = ((getattr(cfg, "user", None) or {}).get("agent") or {}).get("compact_at_tokens")
    return {
        "tokens": tokens,
        "enabled": bool(tokens),
        "source": (
            "setup" if "compact_at_tokens" in overlay
            else ("config" if configured is not None else "default")
        ),
        "overridden": "compact_at_tokens" in overlay,
        "default": config_mod.DEFAULTS["agent"]["compact_at_tokens"],
        "presets": list(CONTEXT_PRESETS),
        "bounds": list(settings_mod.AGENT_SETTINGS["compact_at_tokens"]),
        "error": error,
    }


def _kaggle_account() -> dict[str, Any]:
    """The half of the Kaggle credential that is not a secret.

    Its own entry rather than a credential row, for the reason
    `tools/kaggle.py` gives: only the key is secret, and an account name you
    cannot read back is a worse answer to "whose kernels are these?" than a file
    you can.
    """
    from core import config as config_mod, credentials as credentials_mod
    from tools import kaggle as kaggle_tool

    cfg = config_mod.load()
    username, source = kaggle_tool.resolve_username(cfg)
    return {
        "username": username or "",
        "source": source,
        "key_stored": credentials_mod.present(credentials_mod.KAGGLE_KEY),
    }


# ---------------------------------------------------------------------------
# 0b. background tasks
# ---------------------------------------------------------------------------
def tasks_model(agent: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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

    Which is why `elapsed` is bucketed rather than exact: rendering "5s" then
    "7s" changed the fingerprint on every two-second poll, so a *quiet* running
    task redrew the whole window forever -- the opposite of what the paragraph
    above claims. Seconds resolution below a minute is finer than anyone reads
    a background task at, and it makes the docstring true.

    `agent` is the other half of the same question -- the calls the agent itself
    has in flight, from `agent_calls_model`. Passed in rather than read here
    because it comes from the live session rather than from a file, which is the
    one thing this module never reaches for.
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
                # Bucketed while running so a quiet task does not change its own
                # fingerprint every poll; exact once it has finished, where the
                # number stops moving and the precision is worth something.
                "elapsed": _duration(task.elapsed, coarse=task.running),
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
    calls = list(agent or [])
    background = _background_rows()
    return {
        "rows": rows,
        "running": len([r for r in rows if r["running"]]),
        "finished": len([r for r in rows if not r["running"]]),
        "agent": calls,
        "agent_running": len([c for c in calls if c["running"]]),
        "background": background,
        "background_running": len([r for r in background if r["running"]]),
        "empty_fix": (
            "nothing has been started from the workspace yet — VERIFY, RE-CHECK, "
            "REBUILD and BUILD PDF all run here, the agent's own calls appear here "
            "while they run, and anything it backgrounded with `tools.task` appears "
            "here until it is cleared"
        ),
    }


#: How a background task's state is drawn. `lost` is `attention` rather than
#: `broken`: nothing failed, we simply have no exit code -- usually a reboot.
BACKGROUND_TONE = {
    "running": "dashed",
    "ok": "ok",
    "failed": "broken",
    "stopped": "attention",
    "lost": "attention",
}


def _background_rows() -> list[dict[str, Any]]:
    """The agent's own background tasks, from `core/tasks.py`.

    The third way work ends up running with nobody watching it, and the one that
    had nowhere to be seen: `ui/tasks.py`'s registry is in-process, so it holds
    only what *this app* started. A `tools.task start` issued by the agent -- or
    by a second terminal -- is a process on this machine that the workspace could
    not see at all, which is precisely the "what is this machine doing" gap the
    task runner was supposed to close rather than widen.

    Read from the registry file, which is why it can be seen across processes at
    all. `liveness_ttl_s` is set because this runs on the window's poll: see
    `core/tasks.py:alive_pids`.

    Newest first, matching the other two lists.
    """
    from core import tasks as core_tasks

    try:
        found = core_tasks.tasks(liveness_ttl_s=2.0)
    except OSError:
        # An unreadable registry must not take the window down; the other two
        # lists are still worth drawing.
        return []
    rows = []
    # Bounded before `summarise`, not after. A terminal task whose envelope was
    # not folded into the registry sends `summarise` to `last_envelope`, which
    # opens that task's log -- so an unbounded history is a file read per
    # finished task, on the poll, forever, to build rows the window then throws
    # away. The registry is never cleared unless someone runs `task clear`.
    for task in list(reversed(list(found.values())))[:RECENT]:
        running = task["state"] == core_tasks.RUNNING
        summary = core_tasks.summarise(task)
        rows.append(
            {
                "id": task["id"],
                "label": task["label"],
                "command": summary["command"],
                "state": task["state"],
                "tone": BACKGROUND_TONE.get(task["state"], "neutral"),
                "running": running,
                "exit_code": task.get("exit_code"),
                "stoppable": running,
                "halt": " ".join(task["halt"]) if task.get("halt") else None,
                "error": summary["error"],
                "notes": summary["notes"],
                "log": task.get("log"),
            }
        )
    return rows


#: How many of the agent's own calls the window keeps. Enough to cover the turn
#: in flight and the ones just before it; not so many that this becomes a second
#: transcript, which is what the chat window already is.
AGENT_CALLS = 20
#: Lines of a finished call's output kept on the row. The whole thing is already
#: bounded by `agent.clip`; this is the glance version.
AGENT_TAIL_LINES = 8

#: A call's recorded status -> how the tasks window reports it. `running` is
#: absent because it depends on *where* the call is: still running in the turn in
#: flight, or left running by a turn that ended, which is a different fact.
CALL_STATE_TONE = {"ok": ("ok", "ok"), "error": ("failed", "broken")}


def agent_calls_model(session: Any) -> list[dict[str, Any]]:
    """The agent's own tool calls: the turn in flight first, then the ones before.

    Every capability in this project is reached by a Bash into `tools/`, so the
    agent's calls are the other half of "what is running on this machine right
    now" -- and until this they were visible only in the transcript, which is the
    wrong place to look for it once the conversation has scrolled on.

    They are listed *apart* from the workspace's own tasks rather than merged,
    for the reason `tasks_model` gives about the queue: a task is a process this
    app started and can stop, a call is one the agent made and only the agent can
    stop. One table showing both would imply a STOP button that does not exist.

    A call that is still `running` in a turn that has already settled is reported
    as `unfinished`, not as running. The turn died or was interrupted mid-call,
    and the process behind it is not this app's to know about -- saying "running"
    of something nothing is waiting for would be the same lie as a task that
    silently forgets its tail.
    """
    live = list(getattr(session, "blocks", None) or [])
    rows = [_call_row(b, live=True) for b in reversed(live) if b.get("kind") == "tool"]
    for record in reversed(list(getattr(session, "settled", None) or [])):
        if len(rows) >= AGENT_CALLS:
            break
        for block in reversed(record.get("blocks") or []):
            if block.get("kind") == "tool":
                rows.append(_call_row(block, live=False))
    return rows[:AGENT_CALLS]


def _call_row(block: dict[str, Any], *, live: bool) -> dict[str, Any]:
    status = str(block.get("status") or "ok")
    running = live and status == "running"
    if running:
        state, tone = "running", "dashed"
    elif status == "running":
        state, tone = "unfinished", "attention"
    else:
        state, tone = CALL_STATE_TONE.get(status, ("done", "neutral"))
    result = str(block.get("result") or "")
    return {
        "id": str(block.get("id") or ""),
        "name": str(block.get("name") or "tool"),
        "subject": str(block.get("title") or ""),
        "state": state,
        "tone": tone,
        "running": running,
        # Only for a call actually in flight. `started` is wall clock so it
        # survives the session file, but a *finished* call's duration is not
        # something the block records, and inventing one would be worse than the
        # dash. Bucketed for the same reason a task's is: the poll fingerprints
        # the whole model, and a second-by-second clock would redraw the window
        # forever on a call that is doing nothing visible.
        "elapsed": _call_elapsed(block) if running else "",
        "tail": "\n".join(result.splitlines()[-AGENT_TAIL_LINES:]),
        "lines": len(result.splitlines()),
    }


def _call_elapsed(block: dict[str, Any]) -> str:
    started = block.get("started")
    if not isinstance(started, (int, float)) or isinstance(started, bool):
        # A transcript written before calls were stamped. The row is still worth
        # showing; the clock is the part that is not known.
        return ""
    return _duration(max(0.0, _time.time() - float(started)), coarse=True)


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


#: How coarsely a *running* duration is reported. The poll is every 2 s and the
#: fingerprint is the whole model, so any finer and a task that has produced no
#: output still forces a redraw on every tick.
COARSE_ELAPSED_S = 15


def _duration(seconds: float, *, coarse: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    if coarse:
        seconds = (int(seconds) // COARSE_ELAPSED_S) * COARSE_ELAPSED_S
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
        # `mtime` is in the model rather than read at draw time because it is
        # what the read-only render's URL is keyed on: the iframe rebuilds only
        # when its `src` changes, so without it an edit made in Lab would leave
        # a stale document on screen. It is also what moves the fingerprint, so
        # the pane redraws at all.
        "notebooks": [
            {"name": n, "verify": verify_state(n, store=store), "mtime": _mtime(directory / n)}
            for n in names
        ],
        "lab_running": bool(lab.get("running")),
        "lab_port": lab.get("port"),
        "lab_token": lab.get("token"),
        "lab_origin": lab.get("ui_origin"),
        "origin_mismatch": origin_mismatch(lab),
        "ruler": 88,
        "error": lab_error,
    }


def _mtime(path: Path) -> int:
    """Whole seconds, so a fingerprint does not churn on filesystems that report
    sub-second precision differently between reads. A file that vanished between
    the glob and here is zero rather than an exception."""
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def app_port() -> int:
    """The port `ui/app.py` bound. Imported lazily to avoid a cycle: `app`
    imports the shell, which imports this module."""
    from ui import app as app_mod  # noqa: PLC0415

    return int(getattr(app_mod, "PORT", 8080))


def origin_mismatch(lab: dict[str, Any]) -> bool:
    """Whether the running Lab was scoped to a different origin than this app.

    Lab fixes `frame-ancestors` at launch from the origin it was given, so an
    app that has since moved ports cannot embed it -- and the browser reports
    that as *"127.0.0.1 refused to connect"*, which reads as a dead server and
    sends you hunting for a process that is running perfectly well. Detecting it
    is what lets the window say the true thing instead.

    Only the port is compared. `config/jupyter/jupyter_server_config.py`
    deliberately allows both `127.0.0.1:<port>` and `localhost:<port>`, because
    a browser treats them as different origins and which one the window opened
    on is not Lab's business -- so a host difference is not a mismatch, and
    flagging one would put a banner on a perfectly good server.
    """
    if not lab.get("running"):
        return False
    recorded = str(lab.get("ui_origin") or "").rstrip("/")
    if not recorded:
        return False
    # A portless origin -- `http://127.0.0.1`, or anything this cannot parse --
    # makes `rsplit` hand back the *host*, which never equals a port number, so
    # every healthy server would be flagged. That is precisely the false banner
    # this function exists to avoid, so an origin whose shape cannot be read is
    # treated as agreeing rather than as disagreeing.
    authority = recorded.split("//", 1)[-1]
    if ":" not in authority:
        return False
    return authority.rsplit(":", 1)[-1] != str(app_port())


#: The Lab workspace the app's own window uses. JupyterLab keeps one layout per
#: named workspace *on the server*, and two clients on the same one do not
#: cooperate -- Lab detects the collision, and what the second client sees is a
#: reload that drops its kernel connections mid-cell. Giving the app's window a
#: name of its own means opening Lab in a browser at the same time is two
#: independent sessions rather than a fight, which is the whole failure.
APP_LAB_WORKSPACE = "grad-app"


def lab_url(
    state: dict[str, Any], notebook: str | None = None, *, lab_workspace: str | None = None
) -> str:
    """A URL into the running Lab, or `about:blank` when there is nothing to open.

    `lab_workspace` names a JupyterLab workspace; omit it for the default one a
    browser would use. See `APP_LAB_WORKSPACE`.
    """
    if not state.get("lab_running"):
        return "about:blank"
    base = f"http://127.0.0.1:{state['lab_port']}"
    token = state.get("lab_token") or ""
    root = f"/lab/workspaces/{lab_workspace}" if lab_workspace else "/lab"
    if notebook:
        return f"{base}{root}/tree/notebooks/{notebook}?token={token}"
    return f"{base}{root}?token={token}"


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
        # The four kinds beside the one number, because they are the answer to
        # the first question the one number provokes. `billable_tokens` is what
        # a ceiling is charged; `totals` is what it is charged *for*.
        "token_counts": summary.get("totals") or {},
        "billable_tokens": summary.get("billable_tokens", 0),
        "token_weights": summary.get("weights") or {},
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
    # Neither done nor failed: nothing ran. Their own chips because "DONE"
    # beside a run that produced no result is the one reading that sends someone
    # looking for metrics that do not exist, and dashed because that is already
    # what the rest of the design means by written off.
    "ABANDONED": "dashed",
    "FORGOTTEN": "dashed",
}


#: The statuses `core/submit.py` actually writes. Spelled out rather than
#: guessed at, because the failure is silent in the worst way: an unrecognised
#: status falls through to "WAITING GATE", so a fleet of running jobs would
#: render as a queue waiting on a human who has nothing to approve.
RUNNING_STATUSES = ("in_flight", "running", "in_progress")
FAILED_STATUSES = ("failed", "submit_failed", "error")
QUEUED_STATUSES = ("queued", "submitted", "pending")
#: Written off rather than finished: terminal, but not a result and not a
#: failure. Both are collected in the fold's sense -- that is how they stop
#: holding the ceiling -- so they have to be caught before the `DONE` branch
#: below, which would otherwise label a run that produced nothing as one that
#: finished.
#:
#: Two words because they are two different events, and the difference is the
#: one a reader of the queue actually wants: `abandoned` (`ledger abandon`) means
#: the run never reached a backend, `forgotten` (`kaggle forget`) means it did
#: and the platform no longer has any record of it.
#:
#: Read from `core/ledger_store.py` rather than spelled again here: the same
#: vocabulary decides what `DONE.md` counts as established, and two copies of it
#: is how a run ends up written off in one surface and finished in another.
ABANDONED_STATUS = ls_mod.ABANDONED
FORGOTTEN_STATUS = ls_mod.FORGOTTEN
WRITTEN_OFF_STATUSES = ls_mod.WRITTEN_OFF_STATUSES


def _queue_state(run: Any) -> tuple[str, str]:
    """One run's state chip, and the progress bar variant that goes with it."""
    status = str(run.get("status") or "").lower()
    if status in WRITTEN_OFF_STATUSES:
        return status.upper(), "queued"
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
        # Off the variant, not off `run.collected`. An abandoned run *is*
        # collected -- that is how it stops holding the ceiling -- so reading the
        # flag directly drew a full bar for a run that never produced anything,
        # which is precisely the "DONE beside a run with no result" reading that
        # `ABANDONED`'s own chip exists to prevent. `_queue_state` has already
        # made the judgement; this follows it.
        if tone == "queued":
            progress = 0.0
        else:
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
def papers_model(*, filter_name: str | None = None) -> dict[str, Any]:
    """Papers on disk, annotated with what in the ledger depends on them.

    The status chips are the reason this window is not a file listing: "3 claims
    depend on this" is computed by walking every expectation's `basis`, and
    "contradicts exp-…" by intersecting that with the falsified set. A paper the
    agent queued but never read is shown as such rather than omitted.

    **Titles come from the corpus**, not from the paper directory. `paper_ingest`
    records the title it parsed out of the LaTeX in `documents.title` and writes
    nothing beside the source file, so a window reading only the directory shows
    seven rows called `2405.21060` -- the id it already knew. This opens that
    database anyway, for the read/unread split; taking one more column off the
    same query is free.

    **The filter falls back to one that has rows.** It defaulted to `cited`, and
    a workspace whose ledger cites nothing yet -- which is every workspace until
    the first expectation is registered -- opened this window on "Nothing matches
    cited" with seven papers sitting behind a chip nobody had reason to press.
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
    ingested: dict[str, dict[str, Any]] = {}
    stats, _ = _safe(_corpus_stats, {})
    for doc in (stats or {}).get("documents", []) or []:
        key = _paper_key(doc.get("id"))
        if key:
            ingested[key] = doc

    rows = []
    for path in on_disk:
        key = _paper_key(path.name)
        claims = depends.get(key, [])
        broken = contradicts.get(key, [])
        indexed = ingested.get(key)
        read = indexed is not None or any(path.glob("*.tex")) or any(path.glob("*.txt"))
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
                "title": _paper_title(path, indexed),
                "authors": _paper_authors(path, indexed),
                "path": str(path),
                "read": read,
                "claims": claims,
                "chips": chips,
                "cited": bool(claims),
            }
        )

    counts = {
        "cited": len([r for r in rows if r["cited"]]),
        "read": len([r for r in rows if r["read"]]),
        "queued": len([r for r in rows if not r["read"]]),
    }
    # A chip the user never pressed is not a choice they made, so `None` lands
    # on one that has something behind it. An explicit selection is left alone
    # **even when it is empty** -- pressing CITED and being moved to READ would
    # be the window arguing with the click, and the caller passes `None` rather
    # than a default precisely so the two can be told apart.
    if filter_name not in FILTER_KEYS:
        filter_name = next((k for k in FILTER_KEYS if counts.get(k)), "cited")

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
        "counts": counts,
        "error": error,
        "empty_fix": "python -m tools.paper_ingest arxiv <id> --json",
    }


#: The filter chips, in the order they are offered and searched for a default.
FILTER_KEYS: tuple[str, ...] = ("cited", "read", "queued")


def _corpus_stats() -> dict[str, Any]:
    """The index's counts, plus one row per document.

    `SELECT *`, not a column list. `corpus.connect` creates the schema with
    `IF NOT EXISTS`, so a database built before a column was added never gains
    it -- and naming `authors` on an index that predates it raises
    `OperationalError: no such column`, which `_safe` would turn into "no titles
    at all" for a reason that has nothing to do with titles. Every reader below
    uses `.get`, so a row with fewer columns is a row with fewer answers.
    """
    from core import corpus

    con = corpus.connect(create=False)
    try:
        stats = corpus.stats(con)
        rows = con.execute("SELECT * FROM documents").fetchall()
        return {**stats, "documents": [dict(r) for r in rows]}
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


def _paper_title(path: Path, indexed: dict[str, Any] | None = None) -> str:
    """The paper's title: the directory's own files, then the corpus, then the id.

    The directory comes first because a `meta.json` beside the source is
    something a person can correct; the corpus row is what `paper_ingest` parsed
    out of `\\title{...}` and is right most of the time and mangled the rest.
    """
    for candidate in ("title.txt", "meta.json"):
        target = path / candidate
        if not target.exists():
            continue
        try:
            if candidate.endswith(".json"):
                data = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("title"):
                    return _clean_latex(str(data["title"])) or path.name
            else:
                first = target.read_text(encoding="utf-8").strip().splitlines()[0]
                return _clean_latex(first) or path.name
        except (OSError, json.JSONDecodeError, IndexError):
            continue
    if indexed and indexed.get("title"):
        return _clean_latex(str(indexed["title"])) or path.name
    return path.name


#: Formatting macros that survive `\title{...}` extraction and mean nothing in a
#: window: `\LARGE \bf Bandwidth-Efficient…` is a title with two font commands in
#: front of it, not a title about fonts.
_LATEX_MACRO = re.compile(r"\\[a-zA-Z@]+\s*")
_LATEX_SPACING = re.compile(r"\\{2,}|\$|~")


def _clean_latex(text: str) -> str:
    """Enough LaTeX stripped for a title to read as one. Not a parser.

    A real one belongs in `tools/paper_ingest.py`, at the point the title is
    recorded; this is the display side making the best of what is already
    stored, and it has to be safe on input it does not understand -- so it drops
    markup and never drops words.
    """
    out = _LATEX_SPACING.sub(" ", text)
    out = _LATEX_MACRO.sub(" ", out)
    out = out.replace("{", " ").replace("}", " ")
    out = " ".join(out.split())
    return re.sub(r"\s+([:;,.!?])", r"\1", out).strip(" .,:;-")


def _paper_authors(path: Path, indexed: dict[str, Any] | None = None) -> str:
    """`authors · arXiv:id · year`, with whichever halves are known."""
    authors: Any = None
    year: Any = None
    target = path / "meta.json"
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            authors, year = data.get("authors"), data.get("year")
    if authors is None and indexed:
        authors, year = indexed.get("authors"), year or indexed.get("year")
    if isinstance(authors, list):
        authors = ", ".join(str(a) for a in authors[:3]) + (" et al." if len(authors) > 3 else "")
    authors = _clean_latex(str(authors)) if authors else ""
    # The id alone when nothing else is known, rather than "— · arXiv:…": a dash
    # standing in for an author list is a row that looks like it failed to load.
    parts = [p for p in (authors, f"arXiv:{path.name}", str(year) if year else "") if p]
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# 9. wiki + references
# ---------------------------------------------------------------------------
def wiki_model(page_id: str | None = None) -> dict[str, Any]:
    """The selected project's wiki: its pages, the one being read, and whether
    the code underneath has moved since it was written.

    The subject is `pipelines/<name>/` -- the research code the agent generated
    -- rather than Grad's own source, which is `tools/wiki.py`'s job and a
    different reader's question. The design's chat-plus-references-rail becomes
    a page-list rail beside the page, because that is what a wiki is: the rail
    holds the pages and, when the wiki has gone stale, exactly which files
    differ.

    The prose half is written by `core/wikigen.py`; nothing is generated *here*.
    This reads what a build wrote, which is what keeps a window that redraws
    every two seconds from being a window that can spend money.
    """
    from core import budget as budget_mod, jsonl, projwiki
    from tools import projwiki as projwiki_tool

    project, project_error = _safe(budget_mod.current_project)
    if not project:
        return {
            "built": False,
            "project": None,
            "pages": [],
            "changed": [],
            "error": project_error,
            "empty_fix": "python -m tools.budget use <id> --json",
            "empty_message": "No project is selected, and a wiki is written about one.",
        }

    # Guarded for the same reason as the preflight records: `read_json` returns
    # `None` for missing and malformed, but lets `UnicodeDecodeError` and
    # `OSError` through, and "the manifest is unreadable" must not render as
    # "no wiki has been built yet".
    manifest, manifest_error = _safe(lambda: projwiki_tool.manifest(project))
    pages_raw, pages_error = _safe(lambda: jsonl.read_json(projwiki_tool.output_dir(project) / "pages.json"))
    if not isinstance(manifest, dict):
        return {
            "built": False,
            "project": project,
            "pages": [],
            "changed": [],
            "error": manifest_error or pages_error,
            "empty_fix": f"python -m tools.projwiki build --project {project} --json",
            "empty_message": f"No wiki has been built for {project} yet.",
        }

    current, hash_error = _safe(lambda: projwiki.source_hash(project), {})
    current = current or {}
    recorded = manifest.get("source") or {}
    before, after = recorded.get("files") or {}, current.get("files") or {}
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))

    written = pages_raw if isinstance(pages_raw, list) else []
    by_id = {str(p.get("id")): p for p in written if isinstance(p, dict)}
    rows = []
    for entry in manifest.get("pages") or []:
        page = by_id.get(str(entry.get("id"))) or {}
        rows.append(
            {
                "id": entry.get("id"),
                "kind": entry.get("kind"),
                "title": page.get("title") or entry.get("title") or entry.get("id"),
                "summary": page.get("summary") or "",
                "written": bool(page.get("sections")),
                "unverified_refs": page.get("unverified_refs") or [],
                "error": page.get("error"),
            }
        )

    # The first *written* page, not the first planned one: opening on an empty
    # page because its call failed would make a build with one bad page look
    # like a build that produced nothing.
    selected = page_id if page_id in by_id else None
    if selected is None:
        selected = next((r["id"] for r in rows if r["written"]), rows[0]["id"] if rows else None)

    return {
        "built": True,
        "project": project,
        "stale": bool(changed),
        "generated_at": manifest.get("generated_at"),
        "model": manifest.get("model"),
        "prose": manifest.get("prose", True),
        "output_dir": manifest.get("output_dir"),
        "source_hash": current.get("hash"),
        "recorded_hash": recorded.get("hash"),
        "pages": rows,
        "selected": selected,
        "page": by_id.get(str(selected)) if selected else None,
        "changed": changed[:50],
        "changed_total": len(changed),
        "unverified_total": len({r for p in rows for r in p["unverified_refs"]}),
        "error": hash_error or pages_error,
        "empty_fix": f"python -m tools.projwiki build --project {project} --json",
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
        # The same fourth rule `tools/report.py:check` runs. The editor and the
        # gate have to agree about whether a report is clean: a badge saying
        # "no findings" over a report that `report check` will refuse is worse
        # than no badge, because it is the surface someone actually watches
        # while writing.
        lambda: report_mod.check_code_versions(report_mod.cited_run_ids(tex, claims)),
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

    Figures are the one thing extracted rather than left to the markdown
    renderer, and the reason is that the renderer cannot draw them: an
    `![loss](figures/001.png)` becomes an `<img>` whose src resolves against the
    page origin, and the workspace is not served there. They come out as
    `figure` blocks *in place*, which is also what fixes the older behaviour --
    every figure in a turn used to be appended after the last block of it, so a
    message that put a plot in the middle of its argument showed a broken icon
    there and the plot itself several paragraphs later.
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
            if chunk.get("kind") == "text":
                blocks.extend(_split_figures(chunk["text"]))
            else:
                blocks.append(chunk)
    return [b for b in blocks if b.get("text") or b.get("rows") or b.get("src")]


#: `![alt](path)`, the only markdown a figure arrives as. The path stops at the
#: first whitespace or `)` so a title -- `![alt](p.png "caption")` -- is not
#: swallowed into the filename.
_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\(\s*(?P<src>[^\s)]+)[^)]*\)")

#: A figure path named in prose rather than linked: "wrote figures/001.png".
#: Anchored on the directory name so an arbitrary `.png` elsewhere is not
#: hunted for on disk on every redraw of every message.
_FIGURE_MENTION_RE = re.compile(r"(?<![\w/\\.])((?:[\w.\-]+[/\\])*figures[/\\][\w.\-]+\.(?:png|svg|jpe?g|webp|gif))")


def figure_url(token: str) -> str | None:
    """The URL that serves `token`, or None if nothing here can.

    A path is servable when it resolves inside the workspace's `figures/`
    directory and exists. Both halves matter: the route serves that directory by
    *name*, so a figure written anywhere else has no URL, and a link to a file
    that was never created should keep its markdown -- a visible wrong path is
    more use than a silently dropped one.
    """
    text = (token or "").strip().strip("<>\"'")
    if not text:
        return None
    try:
        path = Path(text)
        if not path.is_absolute():
            path = paths.root() / path
        path = path.resolve()
        directory = paths.figures_dir().resolve()
        if directory not in path.parents or not path.is_file():
            return None
    except (OSError, ValueError):  # a path the platform will not even parse
        return None
    return f"/__grad/figure/{quote(path.name)}"


def _split_figures(text: str) -> list[dict[str, Any]]:
    """One prose chunk, split at the figures it draws.

    Markdown images split the text -- the image belongs exactly where it was
    written. A path merely *mentioned* in a sentence does not: the sentence
    still reads as a sentence and cutting it in half to insert a plot would be
    worse than putting the plot under it, so those are emitted after the
    paragraph that named them.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def flush(chunk: str) -> None:
        """One run of prose, then any figure it named but did not link."""
        pending: list[dict[str, Any]] = []
        for paragraph in re.split(r"\n\s*\n", chunk):
            if paragraph.strip():
                out.append({"kind": "text", "text": paragraph.strip()})
            for match in _FIGURE_MENTION_RE.finditer(paragraph):
                url = figure_url(match.group(1))
                if url and url not in seen:
                    seen.add(url)
                    pending.append({"kind": "figure", "src": url, "alt": match.group(1)})
            out.extend(pending)
            pending = []

    index = 0
    for match in _IMAGE_RE.finditer(text):
        url = figure_url(match.group("src"))
        if url is None:
            # Left as written, so a link to a figure that does not exist reads
            # as the broken link it is rather than vanishing.
            continue
        flush(text[index : match.start()])
        if url not in seen:
            seen.add(url)
            out.append({"kind": "figure", "src": url, "alt": match.group("alt")})
        index = match.end()
    flush(text[index:])
    return out


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
