"""Tagging a session, and harvesting evals out of one (HANDOFF §12 step 3).

The eval set is meant to be harvested from a week of real use rather than
authored cold. That step has a prerequisite: the week has to leave something
sliceable behind. These tests are about the two ways that can quietly fail --
tagging a session for something it did not do, and harvesting a question twice.

Both are precision failures rather than crashes, which is why they are worth
tests. A tagger that is merely noisy produces a corpus that looks usable and
answers the gates-on/gates-off question wrongly.
"""

from __future__ import annotations

import json

from core import traces


def bash(command: str, *, result: str = "") -> dict:
    """One tool block, shaped the way `agent.tool_block` shapes it."""
    return {
        "kind": "tool", "id": "t1", "name": "Bash",
        "title": command[:60], "text": command,
        "rows": [], "status": "ok", "result": result,
    }


def turn(*blocks: dict, text: str = "") -> dict:
    return {"role": "assistant", "text": text, "blocks": list(blocks)}


def asked(text: str = "do the thing") -> dict:
    return {"role": "user", "text": text}


# ---------------------------------------------------------------------------
# what a session did
# ---------------------------------------------------------------------------
def test_a_command_the_agent_ran_is_tagged_and_prose_about_it_is_not():
    """The same distinction the chat window draws: a command the agent ran and a
    command it said it would run are different events, and only the first is
    evidence."""
    ran = [asked(), turn(bash("python -m tools.ledger expect --task t --quantity q"))]
    said = [asked(), turn(text="Next I will run `python -m tools.ledger expect` for this.")]
    assert "ledger:expect" in traces.tag_session(ran)
    assert "ledger:expect" not in traces.tag_session(said)
    assert "tool:ledger" not in traces.tag_session(said)


def test_asking_for_an_interface_is_not_using_it():
    """Measured on the real corpus, where four of the five `ledger:` verbs on the
    busiest session came from `--help` calls. "Sessions where an expectation was
    registered" is the query this namespace exists to answer."""
    records = [asked(), turn(bash("python -m tools.ledger expect --help 2>&1 | head -40"))]
    tags = traces.tag_session(records)
    assert "ledger:expect" not in tags
    # The module still counts: reaching for a tool at all is a fact about the
    # session, and one that reached for the ledger and did nothing with it is a
    # more interesting row than one that never thought of it.
    assert "tool:ledger" in tags


def test_a_gate_refusal_is_tagged_by_meaning_rather_than_by_number():
    """A corpus tagged `gate:4` ages badly the first time a code is renumbered."""
    records = [
        asked(),
        turn(bash("python -m tools.jobs submit --spec s.toml --json",
                  result='{"ok": false, "exit": 4, "error": {"message": "no preflight"}}')),
    ]
    tags = traces.tag_session(records)
    assert "gate:preflight" in tags
    assert "gate:4" not in tags


def test_the_gate_namespace_covers_every_refusing_exit_code():
    """These are the rows of the README's table. A corpus that could only see
    some of them would answer "does the discipline pay for itself" from a
    subset."""
    from core import errors

    refusing = {
        errors.EXIT_PREFLIGHT, errors.EXIT_EXPECTATION, errors.EXIT_SPEND,
        errors.EXIT_STALE_RUN, errors.EXIT_PROJECT_BUDGET,
    }
    assert set(traces.GATE_EXITS) == refusing


def test_a_compaction_shows_up_in_the_tags_and_in_the_outcome():
    records = [
        asked(),
        turn(text="worked on it"),
        {"role": "system", "kind": "compaction", "text": "**Compacted.**"},
    ]
    tags = traces.tag_session(records)
    assert "compaction:1" in tags
    assert "outcome:compacted" in tags


def test_only_the_last_turn_decides_the_outcome():
    """A session refused by the budget and then continued is not a refused
    session -- which is also why this is not "did the word ever appear"."""
    recovered = [
        asked(),
        turn(text="Refusing the next turn; the turn that crossed the ceiling finished."),
        asked(),
        turn(text="done"),
    ]
    stopped = [asked(), turn(text="Refusing the next turn — over the token allocation.")]
    assert traces.outcome(recovered) == "completed"
    assert traces.outcome(stopped) == "budget_refused"


def test_a_prompt_with_nothing_under_it_reads_as_unanswered():
    assert traces.outcome([asked()]) == "unanswered"
    assert traces.outcome([]) == "empty"


def test_cost_is_tagged_from_the_ledger_and_absent_when_it_was_never_measured():
    """An unmeasured session is not a cheap one, so it gets no cost tag at all
    rather than `cost:low`."""
    records = [asked(), turn(text="hi")]
    assert not [t for t in traces.tag_session(records) if t.startswith("cost:")]
    priced = traces.tag_session(
        records, usage=[{"output_tokens": 1_000, "cache_read_tokens": 20_000_000}]
    )
    assert "cost:high" in priced


def test_tags_are_deduplicated_and_keep_the_order_things_happened():
    records = [
        asked(),
        turn(bash("python -m tools.paper_search search \"a\" --json")),
        turn(bash("python -m tools.paper_search search \"b\" --json")),
    ]
    tags = traces.tag_session(records)
    assert tags.count("tool:paper_search") == 1
    assert "search:2" in tags
    assert tags[0].startswith("turns:")


# ---------------------------------------------------------------------------
# harvesting
# ---------------------------------------------------------------------------
def test_harvested_rows_are_never_graded_for_you(workspace, monkeypatch):
    """Which papers were the right answer is the one part of an eval row a trace
    cannot recover. The eval README's warning about authoring cold applies to an
    automated harvester as much as to a person."""
    _store_session(monkeypatch, "s-1", [
        asked(),
        turn(bash('python -m tools.paper_search search "how does loss scale with width" --json')),
    ])
    from tools import traces as cli

    out = cli.cmd_harvest(_args(write=False))
    assert out["searches_found"] == 1
    row = out["candidates"][0]
    assert row["question"] == "how does loss scale with width"
    assert row["relevant"] == []
    assert row["seed"] is False


def test_harvesting_twice_does_not_add_the_question_twice(workspace, monkeypatch):
    """This is meant to be re-run as the corpus grows, not once."""
    _store_session(monkeypatch, "s-1", [
        asked(),
        turn(bash("python -m tools.paper_search search 'equivariance error with depth' --json")),
    ])
    from tools import traces as cli

    first = cli.cmd_harvest(_args(write=True))
    assert first["written"] is True and len(first["candidates"]) == 1
    second = cli.cmd_harvest(_args(write=True))
    assert second["candidates"] == []
    assert second["already_present"] == 1
    lines = (workspace / "evals" / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([l for l in lines if l.strip()]) == 1


def test_a_gap_in_the_ids_does_not_produce_a_duplicate(workspace, monkeypatch):
    """The count-based numbering this started with was wrong the moment the ids
    were not exactly q001..qN. Delete one graded row from a file of five and the
    count says the next id is q005, which is already taken -- and a duplicate id
    in an eval set is two different questions that cannot be told apart in a
    result table."""
    evals = workspace / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    (evals / "retrieval.jsonl").write_text(
        "\n".join(
            json.dumps({"id": i, "question": f"already asked {i}"})
            for i in ("q001", "q003", "q004", "q005")
        )
        + "\n",
        encoding="utf-8",
    )
    _store_session(monkeypatch, "s-1", [
        asked(),
        turn(bash('python -m tools.paper_search search "a brand new question" --json')),
    ])
    from tools import traces as cli

    out = cli.cmd_harvest(_args(write=True))
    assert [r["id"] for r in out["candidates"]] == ["q006"]
    ids = [
        json.loads(line)["id"]
        for line in (evals / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ids) == len(set(ids))


def test_ids_that_are_not_qnnn_at_all_do_not_derail_the_numbering(workspace, monkeypatch):
    evals = workspace / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    (evals / "retrieval.jsonl").write_text(
        json.dumps({"id": "scaling-laws-1", "question": "hand-named"}) + "\n"
        + json.dumps({"id": "q007", "question": "numbered"}) + "\n",
        encoding="utf-8",
    )
    _store_session(monkeypatch, "s-1", [
        asked(),
        turn(bash('python -m tools.paper_search search "something else" --json')),
    ])
    from tools import traces as cli

    assert [r["id"] for r in cli.cmd_harvest(_args(write=True))["candidates"]] == ["q008"]


def test_an_id_taken_between_deriving_and_writing_is_derived_again(workspace, monkeypatch):
    """`jsonl.append` serialises the record before it takes the lock, so a
    precondition cannot renumber the row -- it can only refuse, which makes this
    a retry. The precondition is where the race is actually decided."""
    from core import jsonl

    evals = workspace / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    path = evals / "retrieval.jsonl"
    path.write_text(json.dumps({"id": "q001", "question": "there first"}) + "\n", encoding="utf-8")

    real_append = jsonl.append
    raced = {"done": False}

    def append_but_race_first(target, record, *, precondition=None):
        # Another writer lands the id this attempt is about to ask for, in the
        # window between deriving it and the lock closing over the write.
        if not raced["done"]:
            raced["done"] = True
            with open(target, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"id": "q002", "question": "raced in"}) + "\n")
        return real_append(target, record, precondition=precondition)

    monkeypatch.setattr(jsonl, "append", append_but_race_first)
    _store_session(monkeypatch, "s-1", [
        asked(),
        turn(bash('python -m tools.paper_search search "mine" --json')),
    ])
    from tools import traces as cli

    out = cli.cmd_harvest(_args(write=True))
    assert out["candidates"][0]["id"] == "q003"
    ids = [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert ids == ["q001", "q002", "q003"]


def test_harvested_ids_continue_after_the_rows_already_there(workspace, monkeypatch):
    evals = workspace / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    (evals / "retrieval.jsonl").write_text(
        json.dumps({"id": "q001", "question": "a seed row", "seed": True}) + "\n",
        encoding="utf-8",
    )
    _store_session(monkeypatch, "s-1", [
        asked(),
        turn(bash('python -m tools.paper_search search "something new" --json')),
    ])
    from tools import traces as cli

    out = cli.cmd_harvest(_args(write=False))
    assert [r["id"] for r in out["candidates"]] == ["q002"]


def _args(*, write: bool):
    import argparse

    return argparse.Namespace(write=write)


def _store_session(monkeypatch, session_id: str, records: list[dict]) -> None:
    """Write a session file the way `ui/sessions.py` would, and list it."""
    from ui import sessions

    path = sessions.path_for(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"meta": True, "title": "t", "created_at": "2026-08-15T00:00:00+00:00"})]
    lines += [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
