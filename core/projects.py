"""Per-project memory: six files in `projects/<id>/`, three of them derived.

A project already had a *dimension* (§15) -- an id, three ceilings, and a set of
runs charged against it. What it had no place for was everything a person knows
about the work that is not a number: the conventions this project settled on,
why the second approach was abandoned, what the next thing to try is. That lived
in the conversation, and a conversation is compacted.

So each project gets a directory, and the files in it split cleanly in two:

**Authored.** `MEMORY.md`, `PLAN.md`, `TODO.md`. Written by the agent and by the
human, with no machine opinion about their contents. `MEMORY.md` is the one that
is appended to the system prompt at the start of every session -- which is what
makes it memory rather than a file nobody opens.

**Generated.** `EXPECTATIONS.md`, `RESULTS.md`, `DONE.md`. Rendered from
`ledger/expectations.jsonl` and `ledger/runs.jsonl` by `sync()` below. No model
is involved and nothing here interprets a result.

The split is not tidiness, it is HANDOFF §7 applied to a new surface:

    "the machine records what happened, the model interprets it, and the
     interpretation cannot overwrite the record."

A `RESULTS.md` the agent wrote from memory at the end of a long session is
exactly the artefact §7 exists to prevent -- it would read well, it would be
cited, and it would be free to disagree with the ledger. Rendering it instead
costs one function and makes the disagreement impossible.

Each generated file carries a marker line with a digest of its own body, so a
hand edit is *detectable*. `sync` refuses to overwrite one that has been edited
rather than silently discarding the edit: the file is derived and cheap to
rebuild, but the sentence someone typed into it is not.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from core import ledger_store as ls, paths
from core.errors import EXIT_CHECK_FAILED, GradError, UsageError

#: The authored files, and what each is for. The text is written into the
#: scaffold as a comment, because a blank `PLAN.md` teaches nobody what belongs
#: in it and the answer should not have to be looked up.
AUTHORED: dict[str, str] = {
    "MEMORY.md": (
        "Durable facts about this project: conventions it settled on, decisions "
        "and why, dead ends not worth revisiting, gotchas about the data or the "
        "hardware. This file is loaded into the agent's context at the start of "
        "every session -- keep it short enough to be worth that, and factual "
        "enough to be worth trusting."
    ),
    "PLAN.md": (
        "What this project is trying to establish, and the sequence intended to "
        "establish it. Longer-lived than TODO.md: this is the argument, not the "
        "next action."
    ),
    "TODO.md": (
        "The next actions, in order. Things that are done move out of here -- "
        "DONE.md is generated from the ledger and records what was actually "
        "established, which is a different claim from what was finished."
    ),
}

#: The generated files, and the ledger each is a view of.
GENERATED: dict[str, str] = {
    "EXPECTATIONS.md": "ledger/expectations.jsonl",
    "RESULTS.md": "ledger/runs.jsonl",
    "DONE.md": "ledger/runs.jsonl + ledger/expectations.jsonl",
}

DOCS: tuple[str, ...] = tuple(AUTHORED) + tuple(GENERATED)

#: How much of `MEMORY.md` is allowed into the system prompt, in characters.
#: Characters rather than tokens because counting tokens here would mean loading
#: a tokeniser on the startup path for a bound that does not need to be exact.
#: Roughly four characters to the token, so this is about 4k tokens -- large
#: enough for real conventions, small enough that it cannot quietly become the
#: dominant cost of every turn.
MEMORY_MAX_CHARS = 16_000

#: The marker line, whose digest is what makes a hand edit detectable.
#:
#: The source is **quoted**, and that is not decoration: `DONE.md` is rendered
#: from two ledgers and its source string contains spaces, so an unquoted
#: `(?P<source>\S+)` matched the first path, failed to find `sha256=` after it,
#: and reported the file as hand-edited every single time. It was the one
#: generated file with a space in its provenance and therefore the only one that
#: failed -- which is exactly the shape of bug that reaches a user, because the
#: other two proved the mechanism worked.
_MARKER = re.compile(
    r'^<!--\s*grad:generated\s+source="(?P<source>[^"]*)"\s+sha256=(?P<sha>[0-9a-f]{16})\s*-->$',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# the directory
# ---------------------------------------------------------------------------
def resolve_dir(project_id: str) -> Path:
    """`projects/<id>/`, with the id checked before it becomes a path.

    The id arrives from `--project` and from the selection file, and a path built
    by joining unvalidated input is the one way this module could write outside
    the workspace. `core/budget.py` constrains ids at creation; this does not
    trust that, because the two are edited independently and a check that is
    cheap and local beats a guarantee that lives in another file.
    """
    if not project_id or not project_id.strip():
        raise UsageError(
            "a project id is required",
            fix="python -m tools.budget use <id> --json   # or pass --project <id>",
        )
    if project_id != Path(project_id).name or project_id in (".", ".."):
        raise UsageError(
            f"{project_id!r} is not a usable project id: it has to be one path segment",
            fix="use the id from `python -m tools.budget status --json`",
        )
    return paths.project_dir(project_id)


def doc_path(project_id: str, name: str) -> Path:
    if name not in DOCS:
        raise UsageError(
            f"unknown project document {name!r}",
            fix=f"one of: {', '.join(DOCS)}",
        )
    return resolve_dir(project_id) / name


def exists(project_id: str) -> bool:
    return resolve_dir(project_id).is_dir()


# ---------------------------------------------------------------------------
# scaffolding the authored half
# ---------------------------------------------------------------------------
def scaffold(project_id: str) -> dict[str, Any]:
    """Create the authored files if they are absent. Never overwrites one.

    Returns what it created and what it left alone, because "nothing happened"
    and "three files were rewritten" must not look the same in a JSON envelope.
    """
    directory = resolve_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    kept: list[str] = []
    for name, purpose in AUTHORED.items():
        path = directory / name
        if path.exists():
            kept.append(name)
            continue
        _write(path, _scaffold_body(project_id, name, purpose))
        created.append(name)
    return {"project": project_id, "dir": str(directory), "created": created, "kept": kept}


def _scaffold_body(project_id: str, name: str, purpose: str) -> str:
    title = name.removesuffix(".md").title()
    wrapped = "\n".join(f"> {line}" for line in _wrap(purpose, 76))
    return (
        f"# {title} -- {project_id}\n\n"
        f"{wrapped}\n\n"
        f"*(This file is yours. Nothing generates it and nothing overwrites it.)*\n\n"
    )


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# what the agent is given
# ---------------------------------------------------------------------------
def memory_text(project_id: str | None, *, max_chars: int = MEMORY_MAX_CHARS) -> str:
    """`MEMORY.md`, bounded, for the system prompt. Empty when there is none.

    Truncation cuts at a line boundary and says what it dropped, naming the path
    -- a memory file that silently loses its second half is worse than one the
    agent knows to go and read.

    Never raises. This runs on the path that builds a session, and a project id
    that has no directory, or a file that cannot be read, is a session with no
    project memory rather than a session that does not start.
    """
    if not project_id:
        return ""
    try:
        path = doc_path(project_id, "MEMORY.md")
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UsageError):
        return ""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = head.rfind("\n")
    if cut > max_chars // 2:
        head = head[:cut]
    dropped = len(text) - len(head)
    return (
        f"{head.rstrip()}\n\n"
        f"[...{dropped:,} more characters. This file was truncated to keep the system "
        f"prompt bounded; read the whole of it at `{_relative(path)}` if you need it.]"
    )


def _relative(path: Path) -> str:
    """A workspace-relative path when possible, since that is what the agent types."""
    try:
        return path.relative_to(paths.root()).as_posix()
    except ValueError:
        return str(path)


def prompt_block(project_id: str | None, *, max_chars: int = MEMORY_MAX_CHARS) -> str:
    """The section `agent.system_prompt` appends, or "" when there is nothing.

    The other five files are *named* rather than included. A path in the prompt
    costs a dozen tokens and an agent that wants `RESULTS.md` can Read it; the
    file itself costs its whole length on every tool round-trip of the session,
    re-read from cache each time. That asymmetry is the entire argument for
    loading one file and listing five.
    """
    if not project_id:
        return ""
    try:
        directory = _relative(resolve_dir(project_id))
    except UsageError:
        # The same contract `memory_text` states and keeps: this builds a system
        # prompt, and an unusable project id is a session without project notes
        # rather than a session that will not start. The id reaches here from a
        # selection file, so it can be wrong without anyone having just typed it.
        return ""
    memory = memory_text(project_id, max_chars=max_chars)
    lines = [
        "## This project",
        "",
        f"You are working in project `{project_id}`. Its notes are in `{directory}/`:",
        "",
        "* `MEMORY.md` -- durable facts, included below. **Keep it current**: when you "
        "settle a convention, abandon an approach, or learn something about the data or "
        "the hardware that you would want to know next session, write it there.",
        "* `PLAN.md`, `TODO.md` -- the argument and the next actions. Yours to edit.",
        "* `EXPECTATIONS.md`, `RESULTS.md`, `DONE.md` -- generated from the ledger by "
        "`python -m tools.project sync --json`. Read them; never edit them.",
        "",
    ]
    if memory:
        lines += ["### MEMORY.md", "", memory, ""]
    else:
        lines += [
            "`MEMORY.md` is empty. Start it as soon as this project has a fact worth "
            "carrying to the next session.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the ledger fold these views are rendered from
# ---------------------------------------------------------------------------
def state(project_id: str) -> dict[str, Any]:
    """Everything the three generated files draw on, folded once.

    One fold rather than three, so `EXPECTATIONS.md` and `DONE.md` cannot
    disagree about whether an expectation is bound.
    """
    runs = [r for r in ls.runs() if r.project == project_id]
    by_expectation: dict[str, str] = {}
    for run in runs:
        bound = run.get("expectation_id")
        if bound:
            by_expectation[str(bound)] = run.id

    falsified = ls.falsified_ids()
    expectations = [
        e for e in ls.expectations() if _expectation_project(e, by_expectation) == project_id
    ]

    collected = [r for r in runs if r.collected]
    done, open_runs = [], []
    for run in collected:
        (open_runs if run.unjudged_deviations() else done).append(run)

    return {
        "project": project_id,
        "expectations": expectations,
        "bound_to": by_expectation,
        "falsified": sorted(falsified),
        "runs": runs,
        "collected": collected,
        "in_flight": [r for r in runs if not r.collected],
        "done": done,
        "awaiting_verdict": open_runs,
    }


def _expectation_project(expectation: dict[str, Any], bound_to: dict[str, str]) -> str | None:
    """Which project an expectation belongs to.

    Its own `project` field when it has one -- `ledger expect` stamps the current
    selection, the same way a run record is stamped. Records written before that
    field existed have none, so they fall back to the project of whatever run
    bound them, which is the only other evidence there is. An unbound expectation
    from before the field is genuinely unattributable and is left out rather than
    guessed into someone's project.
    """
    declared = expectation.get("project")
    if declared:
        return str(declared)
    run_id = bound_to.get(str(expectation.get("id")))
    if not run_id:
        return None
    try:
        return ls.run(run_id).project
    except Exception:  # noqa: BLE001 - a missing run is not this function's finding
        return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def sync(project_id: str, *, force: bool = False) -> dict[str, Any]:
    """Regenerate the three derived files.

    Refuses any that has been hand-edited, unless `force`. The digest in the
    marker line is what makes that detectable: it covers the body below the
    marker, so a file this function wrote and nobody touched hashes to what it
    says it does, and one that somebody typed into does not.
    """
    directory = resolve_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = state(project_id)
    bodies = {
        "EXPECTATIONS.md": _render_expectations(snapshot),
        "RESULTS.md": _render_results(snapshot),
        "DONE.md": _render_done(snapshot),
    }

    edited = [] if force else [n for n in bodies if _hand_edited(directory / n)]
    if edited:
        raise GradError(
            "generated_file_edited",
            f"{', '.join(edited)} {'has' if len(edited) == 1 else 'have'} been edited by hand, "
            "and these files are rendered from the ledger -- syncing would discard the edit",
            exit_code=EXIT_CHECK_FAILED,
            fix=(
                f"move the edit into {_relative(directory / 'MEMORY.md')}, then "
                f"python -m tools.project sync --project {project_id} --force --json"
            ),
            detail={"edited": edited, "dir": str(directory)},
        )

    written = []
    for name, body in bodies.items():
        _write(directory / name, _with_marker(GENERATED[name], body))
        written.append(name)
    return {
        "project": project_id,
        "dir": str(directory),
        "written": written,
        "expectations": len(snapshot["expectations"]),
        "runs": len(snapshot["runs"]),
        "collected": len(snapshot["collected"]),
        "done": len(snapshot["done"]),
        "awaiting_verdict": len(snapshot["awaiting_verdict"]),
    }


def _digest(body: str) -> str:
    """The digest of a body, taken over LF-normalised text.

    Normalised because a digest that changes with the line endings would report
    "hand-edited" for a file an editor merely re-saved -- and on Windows that is
    most editors. The content is what is being protected, not the bytes.
    """
    return hashlib.sha256(_lf(body).encode("utf-8")).hexdigest()[:16]


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _with_marker(source: str, body: str) -> str:
    return f'<!-- grad:generated source="{source}" sha256={_digest(body)} -->\n{body}'


def _write(path: Path, text: str) -> None:
    """Write with LF endings, whatever the platform would have chosen.

    `Path.write_text` translates to the platform's separator, so on Windows the
    bytes on disk are not the bytes that were hashed. `_digest` normalises, so
    this is belt and braces -- but a digest-checked file whose on-disk form
    matches what was hashed is one fewer thing to reason about for anyone who
    ever checks it with a tool that is not this one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _hand_edited(path: Path) -> bool:
    """Whether an existing generated file no longer matches its own digest.

    A file with no marker at all is treated as edited when it exists: it was
    written by something other than this function, and overwriting it is the
    thing being guarded against. A file that does not exist is not edited.
    """
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return True
    match = _MARKER.search(text)
    if not match:
        return True
    body = text[match.end() :].lstrip("\n")
    return _digest(body) != match.group("sha")


def _header(title: str, project_id: str, source: str) -> list[str]:
    return [
        f"# {title} -- {project_id}",
        "",
        f"*Generated from `{source}`. Do not edit: "
        "`python -m tools.project sync` overwrites this file. "
        "Prose belongs in `MEMORY.md`.*",
        "",
    ]


def _render_expectations(snapshot: dict[str, Any]) -> str:
    project_id = snapshot["project"]
    lines = _header("Expectations", project_id, "ledger/expectations.jsonl")
    expectations = snapshot["expectations"]
    if not expectations:
        lines += [
            "No expectations registered for this project yet.",
            "",
            "```bash",
            "python -m tools.ledger expect --task <task> --quantity <q> \\",
            "    --low <lo> --high <hi> --basis '<paper>|<locator>|<value>|<conditions>' \\",
            '    --comparability "..." --json',
            "```",
            "",
        ]
        return "\n".join(lines)

    bound_to, falsified = snapshot["bound_to"], set(snapshot["falsified"])
    lines += ["| id | task | quantity | predicted | confidence | status |", "|---|---|---|---|---|---|"]
    for e in expectations:
        eid = str(e.get("id") or "")
        lines.append(
            "| `{id}` | {task} | `{quantity}` | {predicted} | {confidence} | {status} |".format(
                id=eid,
                task=_cell(e.get("task")),
                quantity=_cell(e.get("quantity")),
                predicted=_predicted(e.get("predicted") or {}),
                confidence=_cell(e.get("confidence")),
                status=_expectation_status(eid, bound_to, falsified),
            )
        )
    lines.append("")

    for e in expectations:
        eid = str(e.get("id") or "")
        lines += [f"### `{eid}`", ""]
        if e.get("claim"):
            lines += [f"**Claim.** {e['claim']}", ""]
        if e.get("comparability"):
            lines += [f"**Comparability.** {e['comparability']}", ""]
        basis = e.get("basis") or []
        if basis:
            lines.append("**Basis.**")
            lines += [
                "* {paper} -- {locator}: {value}{conditions}".format(
                    paper=b.get("paper", "?"),
                    locator=b.get("locator", "?"),
                    value=b.get("value", "?"),
                    conditions=f" ({b['conditions']})" if b.get("conditions") else "",
                )
                for b in basis
                if isinstance(b, dict)
            ]
            lines.append("")
    return "\n".join(lines)


def _expectation_status(eid: str, bound_to: dict[str, str], falsified: set[str]) -> str:
    if eid in falsified:
        return "retracted"
    run_id = bound_to.get(eid)
    return f"bound to `{run_id}`" if run_id else "**open**"


def _predicted(predicted: dict[str, Any]) -> str:
    low, high, direction = predicted.get("low"), predicted.get("high"), predicted.get("direction")
    if low is not None and high is not None:
        return f"{low} - {high}"
    if low is not None:
        return f">= {low}"
    if high is not None:
        return f"<= {high}"
    # `direction` is free text off the expectation record, and it is the only
    # branch here that is not a number this function formatted itself.
    return f"*{_cell(direction)}*" if direction else "--"


def _render_results(snapshot: dict[str, Any]) -> str:
    project_id = snapshot["project"]
    lines = _header("Results", project_id, "ledger/runs.jsonl")
    runs, collected = snapshot["runs"], snapshot["collected"]
    if not runs:
        lines += ["No runs have been submitted for this project yet.", ""]
        return "\n".join(lines)

    in_flight = snapshot["in_flight"]
    lines += [
        f"{len(collected)} collected, {len(in_flight)} in flight, "
        f"{len(snapshot['awaiting_verdict'])} awaiting a verdict.",
        "",
        "| run | task | status | smoke | cost | collected |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            "| `{id}` | {task} | {status} | {smoke} | {cost} | {when} |".format(
                id=r.id,
                task=_cell(r.get("task")),
                status=r.status,
                smoke="yes" if r.is_smoke else "",
                cost=_usd(r.get("cost_usd_actual"), r.get("estimate_usd"), r.collected),
                when=_cell(r.get("collected_at")) or "--",
            )
        )
    lines.append("")

    for r in collected:
        lines += [f"### `{r.id}` -- {r.get('task') or 'untitled'}", ""]
        results = r.get("results") or {}
        if results:
            lines += ["| quantity | value |", "|---|---|"]
            # Both halves through `_cell`. These are ledger-derived: the key is a
            # quantity name the pipeline chose and the value is whatever it
            # reported, so either can carry a pipe or a newline and take the
            # table with it -- in the one file whose job is to be readable when
            # a run has gone wrong.
            lines += [
                f"| `{_cell(k)}` | {_cell(v)} |" for k, v in sorted(results.items())
            ]
            lines.append("")
        else:
            lines += ["*The run reported no metrics.*", ""]
        deviations = r.get("deviations") or []
        if deviations:
            lines += ["| quantity | expected | actual | in range | verdict | note |", "|---|---|---|---|---|---|"]
            for d in deviations:
                lines.append(
                    "| `{q}` | {exp} | {actual} | {ok} | {verdict} | {note} |".format(
                        q=_cell(d.get("quantity")),
                        exp=_predicted(d.get("expected") or {}),
                        actual=_observed(d),
                        ok=_in_range(d.get("in_range")),
                        verdict=_cell(d.get("verdict")) or "**unjudged**",
                        note=_cell(d.get("note") or d.get("reason")),
                    )
                )
            lines.append("")
        if r.get("error"):
            lines += [f"**Error.** {_cell(r.get('error'), limit=400)}", ""]
    return "\n".join(lines)


def _in_range(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "**no**"
    return "*undecidable*"


def _observed(deviation: dict[str, Any]) -> str:
    """What the run measured, with its replication shown rather than implied.

    `3.01` and `3.01 (n=3, 95% CI 2.98-3.04)` are different claims, and the
    difference is the whole of `core/stats.py`. A table that printed the first
    for both would hide exactly the thing this column exists to report -- so an
    unreplicated number is marked `n=1` rather than left to look like a
    measurement whose spread nobody has asked about.
    """
    summary = deviation.get("stats")
    if not isinstance(summary, dict) or not summary.get("n"):
        return _cell(deviation.get("actual"))
    n = int(summary["n"])
    mean = summary.get("mean")
    if n == 1:
        return f"{_cell(mean if mean is not None else deviation.get('actual'))} (n=1)"
    interval = summary.get("ci95") or []
    if len(interval) == 2:
        return f"{mean:.6g} (n={n}, 95% CI {interval[0]:.6g}-{interval[1]:.6g})"
    return f"{mean:.6g} (n={n})"


def _usd(actual: Any, estimate: Any, collected: bool) -> str:
    if collected and actual is not None:
        return f"${float(actual):.2f}"
    if estimate is not None:
        return f"~${float(estimate):.2f}"
    return "--"


def _render_done(snapshot: dict[str, Any]) -> str:
    project_id = snapshot["project"]
    lines = _header("Done", project_id, "ledger/runs.jsonl + ledger/expectations.jsonl")
    done, waiting, in_flight = snapshot["done"], snapshot["awaiting_verdict"], snapshot["in_flight"]

    lines += [
        "A run is *done* here when it has been collected **and** every deviation it "
        "produced has a verdict. Anything else is listed underneath, so this file "
        "cannot read as finished while work is open.",
        "",
        f"**{len(done)} done | {len(waiting)} awaiting a verdict | {len(in_flight)} in flight**",
        "",
    ]

    if done:
        lines += ["## Established", ""]
        for r in done:
            verdicts = [d for d in (r.get("deviations") or []) if d.get("verdict")]
            summary = "; ".join(
                f"`{d.get('quantity')}` -> **{d.get('verdict')}**"
                + (f" ({d['note']})" if d.get("note") else "")
                for d in verdicts
            )
            lines.append(
                f"* `{r.id}` -- {r.get('task') or 'untitled'}"
                + (f" -- {summary}" if summary else " -- no deviation to judge")
            )
        lines.append("")
    else:
        lines += ["## Established", "", "*Nothing yet.*", ""]

    if waiting:
        lines += ["## Collected, awaiting a verdict", ""]
        for r in waiting:
            for d in r.unjudged_deviations():
                lines.append(
                    f"* `{r.id}` -- `{d.get('quantity')}`: {d.get('reason') or 'out of range'}  \n"
                    f"  `python -m tools.ledger verdict {r.id} --quantity {d.get('quantity')} "
                    "--verdict bug|real|inconclusive --note '...' --json`"
                )
        lines.append("")

    if in_flight:
        lines += ["## In flight", ""]
        lines += [
            f"* `{r.id}` -- {r.get('task') or 'untitled'}, submitted {r.get('submitted_at')}"
            for r in in_flight
        ]
        lines.append("")
    return "\n".join(lines)


def _cell(value: Any, *, limit: int = 120) -> str:
    """One table cell: single-line, pipe-escaped, bounded.

    A newline or an unescaped pipe in a metric name or an error string breaks
    the table around it, which turns a rendering detail into a file that cannot
    be read at the moment it is most worth reading.
    """
    if value is None:
        return ""
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "..."
