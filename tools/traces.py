"""grad-traces -- what the sessions so far actually contain (HANDOFF §12 step 3).

    "The order in §12 is deliberate -- build the agent, use it for a week,
    *then* harvest `evals/retrieval.jsonl` from what retrieval was actually
    reached for. Authoring it cold would measure the imagination rather than
    the system."

The harvesting step needs the week of use to have left something sliceable
behind, and until now it had not: the transcripts are a record, but "every
session where a submitter refused" was a full-text search whose answer depended
on phrasing. `core/traces.py` tags each trajectory; this reads the sessions and
applies it.

`harvest` is the verb the handoff is actually asking for. It pulls the questions
that were really put to `paper_search` out of the transcripts and writes them as
candidate rows in the `evals/retrieval.jsonl` schema -- **unlabelled**, with
`relevant` empty and `seed` false, because which papers were the right answer is
the one part of an eval row that cannot be recovered from a trace. Grading is
yours. What this removes is the part that was never worth a human: remembering
what you asked.

The tags are metadata and never a filter. Nothing here decides that a session
does not count.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from core import paths, quota_log, traces
from core.cli import Cli, main
from core.errors import NotFound, UsageError

cli = Cli(
    "grad-traces",
    "Tag stored sessions, and harvest evaluation candidates from real use.",
    epilog=(
        "Tags are `namespace:value` and are metadata, not a filter.\n"
        "`gate:` is the one worth having: every claim in the README is that some gate\n"
        "refuses under some condition, and a corpus tagged by which one refused is the\n"
        "difference between believing that and knowing it."
    ),
)


def _sessions() -> Any:
    """`ui.sessions`, imported at the point of use.

    It reads and writes files and imports nothing from NiceGUI, so this is safe
    on a machine with no `ui` extra -- but the import still belongs here rather
    than at module scope, so that `--help` works on a checkout where `ui/` has
    been trimmed away.
    """
    from ui import sessions  # noqa: PLC0415

    return sessions


def _trajectory(path: Any) -> list[dict[str, Any]]:
    """The records of one session file, skipping the meta line and any junk.

    Deliberately the same tolerance `ui/app.py:restore` applies: these files
    outlive the version that wrote them, and one malformed line should cost that
    line rather than the command.
    """
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("role"):
            out.append(record)
    return out


def _usage_by_session() -> dict[str, list[dict[str, Any]]]:
    """`ledger/quota.jsonl`, grouped by the session that spent it.

    Read once for the whole command rather than per session: the ledger is one
    append-only file and re-reading it per session turns a listing into a
    quadratic one.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for row in quota_log.entries():
        key = row.get("session")
        if isinstance(key, str) and key:
            out.setdefault(key, []).append(row)
    return out


def _tagged(session: dict[str, Any], usage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sessions = _sessions()
    session_id = session["id"]
    records = _trajectory(sessions.path_for(session_id))
    rows = usage.get(session_id, [])
    return {
        "id": session_id,
        "title": session.get("title") or "",
        "created_at": session.get("created_at"),
        "turns": sum(1 for r in records if r.get("role") == "user"),
        "billable_tokens": round(sum(quota_log.billable(r) for r in rows)),
        "tags": traces.tag_session(records, usage=rows),
    }


def _list_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", action="append", help="only sessions carrying this tag (repeatable, AND)")


@cli.command("list", "every stored session, with its tags", setup=_list_args)
def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    usage = _usage_by_session()
    rows = [_tagged(s, usage) for s in _sessions().listing()]
    wanted = set(args.tag or ())
    if wanted:
        rows = [r for r in rows if wanted <= set(r["tags"])]
    return {
        "sessions": rows,
        "count": len(rows),
        "filtered_by": sorted(wanted),
        # Every tag in the corpus with how many sessions carry it. This is the
        # part worth reading first: it says what a week of use actually
        # consisted of, which is usually not what it felt like it consisted of.
        "tag_counts": _counts(rows),
    }


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for tag in row["tags"]:
            out[tag] = out.get(tag, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


@cli.command(
    "show",
    "one session's tags, and the commands behind them",
    setup=lambda p: p.add_argument("session_id"),
)
def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    sessions = _sessions()
    if not sessions.is_id(args.session_id):
        raise UsageError(
            f"not a session id: {args.session_id!r}",
            fix="python -m tools.traces list --json",
        )
    path = sessions.path_for(args.session_id)
    if not path.exists():
        raise NotFound(
            f"no session {args.session_id!r}",
            fix="python -m tools.traces list --json",
        )
    records = _trajectory(path)
    rows = _usage_by_session().get(args.session_id, [])
    return {
        "id": args.session_id,
        "meta": sessions.read_meta(path),
        "tags": traces.tag_session(records, usage=rows),
        "outcome": traces.outcome(records),
        "commands": traces.commands(records),
        "questions": _questions(records),
        "billable_tokens": round(sum(quota_log.billable(r) for r in rows)),
    }


#: How a search reaches `paper_search`, as the system prompt tells the agent to
#: write it. Both quote styles, because the agent writes whichever the shell
#: wants and a harvester that only knew one would silently find half of them.
SEARCH_RE = re.compile(
    r"python\s+-m\s+tools\.paper_search\s+(?:search|local)\s+(?P<q>\"[^\"]+\"|'[^']+')"
)


def _questions(records: list[dict[str, Any]]) -> list[str]:
    """The questions actually put to the funnel, in the order they were asked."""
    out: list[str] = []
    for command in traces.commands(records):
        for match in SEARCH_RE.finditer(command):
            question = match.group("q")[1:-1].strip()
            if question and question not in out:
                out.append(question)
    return out


def _harvest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--write",
        action="store_true",
        help="append the candidates to evals/retrieval.jsonl instead of printing them",
    )


@cli.command("harvest", "candidate eval rows from the searches actually run", setup=_harvest_args)
def cmd_harvest(args: argparse.Namespace) -> dict[str, Any]:
    """Turn real searches into unlabelled `evals/retrieval.jsonl` rows.

    Unlabelled on purpose. `relevant` is empty and has to be filled in by hand,
    because which papers were the right answer is the one thing a transcript
    cannot say -- the trace records what was asked and what came back, never
    whether what came back was any good. The eval README's own warning applies
    to an automated harvester as much as to a person: a row whose relevance
    labels were guessed measures the guess.

    Existing rows are never rewritten and duplicates are never appended, so this
    is safe to re-run as the corpus grows -- which is the intended use. It is
    the mechanism §12 step 3 describes, not a one-off migration.
    """
    sessions = _sessions()
    seen: list[str] = []
    rows: list[dict[str, Any]] = []
    for session in sessions.listing():
        records = _trajectory(sessions.path_for(session["id"]))
        for question in _questions(records):
            if question in seen:
                continue
            seen.append(question)
            rows.append(
                {
                    "id": "",  # assigned below, after the existing file is read
                    "question": question,
                    "asked_at": (session.get("created_at") or "")[:10],
                    "relevant": [],
                    "notes": f"harvested from session {session['id']}; relevance not yet graded",
                    "seed": False,
                }
            )

    path = paths.root() / "evals" / "retrieval.jsonl"
    existing, known = _existing_eval_rows(path)
    fresh = [r for r in rows if r["question"] not in known]
    for offset, row in enumerate(fresh, start=len(existing) + 1):
        row["id"] = f"q{offset:03d}"

    written = False
    if args.write and fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written = True

    return {
        "path": str(path),
        "searches_found": len(seen),
        "already_present": len(seen) - len(fresh),
        "candidates": fresh,
        "written": written,
        "note": (
            "`relevant` is empty in every row: which papers were the right answer is the "
            "one part of an eval row a trace cannot recover. Grade them by hand, then "
            "drop the seed rows."
        ),
        "fix": None if written or not fresh else "re-run with --write to append them",
    }


def _existing_eval_rows(path: Any) -> tuple[list[dict[str, Any]], set[str]]:
    """What is already in the eval file, and the questions it already covers.

    Seed rows count as present. They are examples of the schema rather than real
    queries, but a harvest that re-added a question a seed row already asks
    would put the same question in the file twice, and the README's instruction
    is to *replace* the seeds rather than pad around them.
    """
    if not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, {str(r.get("question") or "") for r in rows}


if __name__ == "__main__":
    main(cli)
