"""What happened in a session, as tags a later query can slice on.

HANDOFF §12 step 3 says the eval set is harvested from a week of real use rather
than authored cold, because "a benchmark of imagined queries measures the
imagination". That step has a prerequisite nobody wrote down: the week of real
use has to leave something behind that can be sliced. A directory of transcripts
is a record, but "show me every session where a submitter refused" is a
full-text search over prose, and the answer depends on how the refusal happened
to be phrased.

So each trajectory gets tags, in the shape ml-intern's `sft/tagger.py` uses --
`namespace:value` strings, deduplicated, no filtering and no mutation -- and a
downstream pass selects on them. The namespaces here are not theirs, because the
interesting facts about a session are not the same:

    tool:<name>       a CLI the agent actually ran, by module name
    gate:<exit>       a gate that refused, by the exit code it refused with
    ledger:<verb>     expectation and verdict traffic: expect, verdict, collect…
    outcome:<end>     how the session ended
    turns:<bucket>    short (<5) / medium (5-20) / long (>20)
    cost:<bucket>     by weighted tokens: low (<100k) / med (<1M) / high
    search:<n>        whether the retrieval funnel was reached for, and how often
    compaction:<n>    how many times the conversation was compacted

`gate:` is the namespace this project has and ml-intern does not, and it is the
one worth having. Every claim in the README's table is that some gate refuses
under some condition; a corpus of real sessions tagged by which gate refused is
the difference between believing that and knowing it. It is also the raw
material for the gates-on/gates-off comparison, which is the only measurement
that is about this harness rather than about the model underneath it.

**Nothing here reads a file.** The input is a trajectory -- the list of records
`ui/app.py:Session.restore` produces -- so tagging is a pure function over data,
testable without a workspace, and `tools/traces.py` owns the reading. Tags are
metadata and never a filter: this module's job is to describe a session, never
to decide that one does not count.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from core import quota_log

#: `python -m tools.X` / `python -m tools.X ...` in a Bash command. Anchored to
#: the module path rather than to a bare word so that a session *discussing*
#: `report check` is not tagged as having run it -- the tags exist to find
#: sessions where something happened, and prose is not an event.
TOOL_RE = re.compile(r"\bpython\s+-m\s+tools\.([a-z_]+)")

#: A command that only asked what the interface is. Tagged for the module, since
#: reaching for a tool is a fact about the session, but not for the verb.
HELP_RE = re.compile(r"(?:^|\s)(?:--help|-h)(?:\s|$)")

#: A ledger verb, as it appears after the module. `expect` and `verdict` are the
#: two halves of the pre-registration loop and are the point of the namespace;
#: the rest are here so that "an expectation was opened and never judged" is a
#: query rather than an inference.
LEDGER_VERBS = ("expect", "verdict", "falsify", "verify", "query", "collect", "submit")

#: Exit codes that mean a gate refused, from `core/errors.py`. Named here rather
#: than imported as a set so the tag carries the *meaning* and not the number: a
#: corpus tagged `gate:4` ages badly the first time a code is renumbered.
GATE_EXITS = {
    4: "preflight",
    5: "expectation",
    6: "spend",
    7: "stale",
    12: "project_budget",
}

#: Turn-count buckets, as (upper bound exclusive, label). The last is unbounded.
TURN_BUCKETS = ((5, "short"), (20, "medium"))
#: Weighted-token buckets, same shape. 100k and 1M because those are roughly
#: "one question" and "an afternoon" on the usage measured so far.
COST_BUCKETS = ((100_000, "low"), (1_000_000, "med"))


def tag_session(trajectory: Iterable[Any], *, usage: Iterable[Any] = ()) -> list[str]:
    """Tags for one trajectory. Pure, order-stable, deduplicated.

    `usage` is this session's rows from `ledger/quota.jsonl`, which is where the
    cost lives -- the transcript records what was said and never what it cost.
    Passing none is fine and simply leaves the session untagged for cost, which
    is honest: an unmeasured session is not a cheap one.
    """
    records = [r for r in trajectory if isinstance(r, dict)]
    tags: list[str] = []

    turns = sum(1 for r in records if r.get("role") == "user")
    tags.append(f"turns:{_bucket(turns, TURN_BUCKETS, 'long')}")

    compactions = sum(1 for r in records if r.get("kind") == "compaction")
    if compactions:
        tags.append(f"compaction:{compactions}")

    searches = 0
    for command in commands(records):
        # A verb asked about is not a verb run. `ledger expect --help` would
        # otherwise tag the session `ledger:expect`, and "sessions where an
        # expectation was registered" is precisely the query this namespace
        # exists to answer -- a corpus that cannot tell reading the interface
        # from using it is no use for the gates-on/gates-off comparison.
        # Measured on the real corpus, where four of the five `ledger:` verbs
        # on the busiest session came from `--help` calls.
        #
        # The *module* is still tagged either way: reaching for a tool at all is
        # a fact about the session, and one that reached for `ledger` and then
        # did nothing with it is a more interesting row than one that never
        # thought of it.
        asking = bool(HELP_RE.search(command))
        for module in TOOL_RE.findall(command):
            tags.append(f"tool:{module}")
            if asking:
                continue
            if module == "paper_search":
                searches += 1
            if module == "ledger":
                tags.extend(f"ledger:{v}" for v in LEDGER_VERBS if _has_word(command, v))
            if module in ("jobs", "gpu"):
                tags.extend(f"ledger:{v}" for v in ("submit", "collect") if _has_word(command, v))
    if searches:
        tags.append(f"search:{searches}")

    for code in _exit_codes(records):
        name = GATE_EXITS.get(code)
        if name:
            tags.append(f"gate:{name}")

    tags.append(f"outcome:{outcome(records)}")

    total = sum(quota_log.billable(row) for row in usage if isinstance(row, dict))
    if total:
        tags.append(f"cost:{_bucket(total, COST_BUCKETS, 'high')}")

    return _dedupe(tags)


def outcome(records: list[dict[str, Any]]) -> str:
    """How the session ended, from its last assistant turn.

    Five endings, and the ordering of the checks is the meaning. A session that
    was refused by the budget and *then* went on to do more is not a refused
    session, so only the last turn is consulted -- which is also why this is not
    simply "did the word 'refused' ever appear".
    """
    if not records:
        return "empty"
    last = records[-1]
    if last.get("role") == "user":
        # A prompt with no answer under it: the turn died before it settled, or
        # the app was closed mid-turn. Either way nothing came back.
        return "unanswered"
    if last.get("kind") == "compaction":
        return "compacted"
    text = str(last.get("text") or "")
    if "the session failed:" in text:
        return "errored"
    if "Refusing the next turn" in text or "token allocation" in text:
        return "budget_refused"
    return "completed"


def commands(records: list[dict[str, Any]]) -> list[str]:
    """Every Bash command the agent actually ran.

    Off the `tool` blocks rather than out of the prose, which is the same
    distinction the chat window draws: a command the agent ran and a command it
    said it would run are different events, and only the first is evidence.
    """
    out: list[str] = []
    for record in records:
        for block in record.get("blocks") or []:
            if not isinstance(block, dict) or block.get("kind") != "tool":
                continue
            if str(block.get("name") or "").lower() != "bash":
                continue
            out.append(str(block.get("text") or ""))
            # `rows` is the rest of the tool input, and on a Bash call the
            # command can land there rather than in `text` depending on which
            # field `describe_tool` chose as the subject.
            for row in block.get("rows") or []:
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    out.append(str(row[1]))
    return out


def _exit_codes(records: list[dict[str, Any]]) -> list[int]:
    """Exit codes reported by tool results, best effort.

    The CLIs answer with a JSON envelope carrying `exit`, and the result text is
    what the block holds. Parsed leniently -- a truncated result is normal, since
    `agent.clip` bounds what is kept -- because a missed code costs one tag and a
    raise here would cost the whole tagging pass.
    """
    codes: list[int] = []
    for record in records:
        for block in record.get("blocks") or []:
            if not isinstance(block, dict) or block.get("kind") != "tool":
                continue
            result = str(block.get("result") or "")
            for match in re.finditer(r'"exit"\s*:\s*(\d+)', result):
                codes.append(int(match.group(1)))
    return codes


def _has_word(command: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", command) is not None


def _bucket(value: float, buckets: tuple[tuple[float, str], ...], last: str) -> str:
    for limit, label in buckets:
        if value < limit:
            return label
    return last


def _dedupe(tags: Iterable[str]) -> list[str]:
    """Deduplicate, keeping first-seen order.

    Order-stable rather than sorted so a reader sees the session's shape --
    turns, then what it did, then how it ended -- in the order it happened.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out
