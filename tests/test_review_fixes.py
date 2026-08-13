"""Regressions for the second review pass.

Each test here corresponds to a defect that was found by review rather than by
use, which makes them the ones most likely to come back: nothing in normal
operation exercises a registry port, a reordered embedding batch, or a rerank
response with an out-of-range index.
"""

from __future__ import annotations

import json

import pytest

from core import config as config_mod, corpus, jsonl, ledger_store as ls, paths
from core.errors import EXIT_USAGE, ConfigError, GateRefusal, UpstreamError


# ---------------------------------------------------------------------------
# embedding alignment: position is identity
# ---------------------------------------------------------------------------
def _stub_httpx(monkeypatch, payload, status=200):
    class _Resp:
        status_code = status
        text = json.dumps(payload)

        @staticmethod
        def json():
            return payload

    class _Httpx:
        @staticmethod
        def post(*_args, **_kwargs):
            return _Resp()

    monkeypatch.setattr("core.http._httpx", lambda: _Httpx)
    monkeypatch.setattr("core.credentials.get", lambda name, required=True: "k")


def test_embeddings_are_reordered_by_index(workspace, monkeypatch, cfg):
    """The caller zips these against chunk ids, so order *is* meaning."""
    from core import http

    _stub_httpx(
        monkeypatch,
        {"data": [
            {"index": 2, "embedding": [3.0]},
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]},
    )
    assert http.embed(["a", "b", "c"], cfg=cfg) == [[1.0], [2.0], [3.0]]


def test_a_short_embedding_batch_is_refused(workspace, monkeypatch, cfg):
    """A partial batch cannot be aligned, and a silently truncated write would
    pair chunk k with some other chunk's vector."""
    from core import http

    _stub_httpx(monkeypatch, {"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(UpstreamError):
        http.embed(["a", "b", "c"], cfg=cfg)


def test_wrong_dimension_vectors_are_refused(workspace):
    """`vector_search` skips mismatched rows, so a bad write would show up only
    as a dense ranking that is quietly always empty."""
    con = corpus.connect()
    try:
        corpus.upsert_document(con, {"id": "d1", "title": "t", "source": "notes"})
        ids = corpus.replace_chunks(con, "d1", [{"text": "a chunk of text"}])
        corpus.bind_embedding_model(con, "voyage-4", 4)
        with pytest.raises(ConfigError):
            corpus.store_vectors(con, ids, [[1.0, 2.0]])
    finally:
        con.close()


# ---------------------------------------------------------------------------
# rerank indices come from upstream
# ---------------------------------------------------------------------------
# "1", 1.0 and True are *in range* for this pool on purpose: a type check that
# only caught out-of-range values would pass a parametrisation of 99 and -1 alone.
@pytest.mark.parametrize("index", [99, -1, None, "1", 1.0, True])
def test_unusable_rerank_indices_are_dropped(index):
    """An IndexError here would abandon a funnel run that already spent stage-0
    quota; a negative index would silently promote the wrong candidate."""
    from tools.paper_search import apply_rerank

    pool = [{"id": "a"}, {"id": "b"}]
    assert apply_rerank(pool, [{"index": index, "score": 0.5}]) == []


def test_a_wholly_unusable_rerank_falls_back_to_the_pool(workspace, monkeypatch, capsys):
    """End to end: every index unusable must not empty the funnel.

    The pool is what stage 1 already paid quota for. Returning nothing because
    an upstream reranker answered with indices we cannot use reads, from the
    agent's side, as "there is no literature on this".
    """
    from core import corpus, http
    from tools import paper_search

    con = corpus.connect()
    try:
        # Two, because stage 2 does not run on a pool of one.
        for doc_id, text in (("d1", "a note about gradient flow"), ("d2", "gradient flow again")):
            corpus.upsert_document(con, {"id": doc_id, "title": "gradient flow", "source": "notes"})
            corpus.replace_chunks(con, doc_id, [{"text": text}])
    finally:
        con.close()

    monkeypatch.setattr(http, "rerank", lambda *a, **k: [{"index": 99, "score": 0.5}])

    assert paper_search.cli.run(
        ["search", "gradient flow", "--local-only", "--no-expand", "--no-triage", "--json"]
    ) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["funnel"] == {"candidates": 2, "reranked": 2, "returned": 2}
    assert {r["id"] for r in data["results"]} == {"local:d1#1", "local:d2#2"}
    assert any("no usable indices" in w for w in data["trace"]["warnings"])


def test_valid_rerank_indices_reorder_the_pool():
    from tools.paper_search import apply_rerank

    pool = [{"id": "a"}, {"id": "b"}]
    ranked = apply_rerank(pool, [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}])
    assert [r["id"] for r in ranked] == ["b", "a"]
    assert [r["rerank_score"] for r in ranked] == [0.9, 0.2]


# ---------------------------------------------------------------------------
# trace names are slugs, not paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["../secret", "../../secret", "funnel/../../secret", "C:/Windows/secret", "..\\secret"],
)
def test_trace_refuses_a_traversing_name(workspace, capsys, name):
    """The agent can invoke this CLI, so a name that reaches outside notes/funnel
    would widen the deny-by-default file boundary."""
    from tools import paper_search

    funnel = paths.notes_dir() / "funnel"
    funnel.mkdir(parents=True, exist_ok=True)
    # In the root but outside notes/funnel: `../../secret` reaches it, and that
    # is the whole claim. Writing above the root would have this test scribble
    # into the directory that holds the temp workspace.
    (paths.root() / "secret.json").write_text('{"password": "hunter2"}', encoding="utf-8")

    assert paper_search.cli.run(["trace", name, "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "hunter2" not in json.dumps(payload)


def test_trace_reads_a_real_slug(workspace, capsys):
    from tools import paper_search

    funnel = paths.notes_dir() / "funnel"
    funnel.mkdir(parents=True, exist_ok=True)
    (funnel / "2026-08-13-a-question.json").write_text('{"question": "q"}', encoding="utf-8")

    assert paper_search.cli.run(["trace", "2026-08-13-a-question", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["question"] == "q"


# ---------------------------------------------------------------------------
# the derived index keeps the tri-state
# ---------------------------------------------------------------------------
def test_sqlite_index_preserves_needs_a_verdict(workspace):
    """NULL means "no program can settle this"; 0 means "out of range". An index
    that cannot tell them apart is not a faithful projection of the JSONL."""
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "r1", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 0.0, "estimated_duration_s": 1}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "r1", "status": "completed", "collected_at": ls.now_iso(),
         "cost_usd_actual": 0.0, "results": {},
         "deviations": [
             {"quantity": "relational", "in_range": None},
             {"quantity": "missed", "in_range": False},
             {"quantity": "hit", "in_range": True},
         ]}
    )
    ls.rebuild_index()
    rows = {r["quantity"]: r["in_range"] for r in ls.query_index("SELECT quantity, in_range FROM deviations")}
    assert rows == {"relational": None, "missed": 0, "hit": 1}


# ---------------------------------------------------------------------------
# expectation binding is atomic with the append
# ---------------------------------------------------------------------------
def test_binding_the_same_expectation_twice_is_refused_at_the_write(workspace):
    """The gate runs before the write, so two racing submitters could both pass
    it. The check is repeated inside the append lock, where it is atomic."""
    exp_id = "exp-race"
    ls.append_expectation(
        {"id": exp_id, "task": "t", "created_at": ls.now_iso(), "quantity": "q",
         "predicted": {"direction": "decrease"}, "basis": [], "comparability": "", "confidence": "low"}
    )
    base = {
        "type": ls.T_RUN_SUBMITTED, "status": "in_flight", "submitted_at": ls.now_iso(),
        "expectation_id": exp_id, "estimate_usd": 0.0, "estimated_duration_s": 1,
    }
    ls.append_run_event({**base, "id": "run-first"})
    with pytest.raises(GateRefusal) as exc:
        ls.append_run_event({**base, "id": "run-second"})
    assert exc.value.code == "expectation_bound"
    assert [r.id for r in ls.runs()] == ["run-first"]


def test_the_precondition_runs_under_the_lock(workspace):
    path = workspace / "ledger" / "pre.jsonl"

    def _refuse():
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        jsonl.append(path, {"a": 1}, precondition=_refuse)
    assert jsonl.read(path) == []
    jsonl.append(path, {"a": 2}, precondition=lambda: None)
    assert jsonl.read(path) == [{"a": 2}]


# ---------------------------------------------------------------------------
# configuration errors read as configuration errors
# ---------------------------------------------------------------------------
def _write_config(workspace, text: str):
    path = workspace / "config" / "grad.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    config_mod._cache.clear()
    return path


def test_a_non_numeric_ceiling_is_a_config_error(workspace):
    _write_config(workspace, '[spend]\nmonthly_usd = "lots"\n')
    with pytest.raises(ConfigError) as exc:
        config_mod.load(reload=True)
    assert "monthly_usd" in exc.value.message


def test_a_negative_ceiling_is_a_config_error(workspace):
    _write_config(workspace, "[spend]\nper_job_usd = -5\n")
    with pytest.raises(ConfigError):
        config_mod.load(reload=True)


def test_a_malformed_host_is_a_config_error(workspace):
    """`rate_usd_per_hour` is what `collect` prices wall clock against, so a bad
    value is a spend-accounting problem, not a stray ValueError."""
    _write_config(workspace, '[hosts.box]\nhostname = "h"\nrate_usd_per_hour = "free"\n')
    with pytest.raises(ConfigError) as exc:
        config_mod.load(reload=True)
    assert "box" in exc.value.message


def test_a_scalar_hosts_section_is_a_config_error(workspace):
    _write_config(workspace, 'hosts = "gpu-box"\n')
    with pytest.raises(ConfigError):
        config_mod.load(reload=True)


@pytest.mark.parametrize("text", ['spend = "lots"\n', "smoke = 1\n", 'hf = "a10g-small"\n'])
def test_a_scalar_where_a_section_belongs_is_a_config_error(workspace, text):
    """`cfg.get` subscripts the section: a scalar there would come back as an
    AttributeError from inside the validator, which reads like a loader bug."""
    _write_config(workspace, text)
    with pytest.raises(ConfigError) as exc:
        config_mod.load(reload=True)
    assert "must be a table" in exc.value.message


# ---------------------------------------------------------------------------
# usage accounting refuses values that would corrupt it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "argv",
    [
        ["record", "--stage", "main", "--input-tokens", "-5", "--json"],
        ["record", "--stage", "main", "--credits-usd", "-1", "--json"],
        ["record", "--stage", "main", "--credits-usd", "nan", "--json"],
        ["record", "--stage", "main", "--credits-usd", "inf", "--json"],
    ],
)
def test_invalid_usage_values_are_refused(workspace, argv):
    """This is the measurement instrument behind every later cost decision: a
    negative count would reduce reported usage, and a NaN is not valid JSON."""
    from tools import quota

    assert quota.cli.run(argv) == EXIT_USAGE
    assert jsonl.read(paths.quota_path()) == []


def test_tail_zero_returns_nothing(workspace, capsys):
    from core import quota_log
    from tools import quota

    quota_log.record("main", input_tokens=1)
    assert quota.cli.run(["tail", "-n", "0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["entries"] == []


def test_negative_tail_is_refused(workspace):
    from tools import quota

    assert quota.cli.run(["tail", "-n", "-3", "--json"]) == EXIT_USAGE


def test_query_limit_zero_returns_one_row_not_all(workspace):
    from tools import ledger as ledger_cli

    for i in range(3):
        ls.append_expectation(
            {"id": f"exp-{i}", "task": "t", "created_at": ls.now_iso(), "quantity": "q",
             "predicted": {"direction": "decrease"}, "basis": [], "comparability": "", "confidence": "low"}
        )
    import io
    import contextlib

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert ledger_cli.cli.run(["query", "--expectations", "--limit", "0", "--json"]) == 0
    assert len(json.loads(out.getvalue())["data"]["expectations"]) == 1


# ---------------------------------------------------------------------------
# credential scrubbing covers the env fallback
# ---------------------------------------------------------------------------
def test_scrub_removes_the_env_credential_fallbacks(workspace, monkeypatch):
    """GRAD_ALLOW_ENV_CREDENTIALS exists for CI, where no agent is running.
    Under the agent those variables are exactly the environment-resident
    credentials §9 argues must not exist."""
    from core import credentials

    monkeypatch.setenv("GRAD_HF_TOKEN", "secret")
    monkeypatch.setenv("GRAD_OPENROUTER_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    removed = credentials.scrub_environment()

    assert {"GRAD_HF_TOKEN", "GRAD_OPENROUTER_KEY", "ANTHROPIC_API_KEY"} <= set(removed)
    import os

    assert "GRAD_HF_TOKEN" not in os.environ


# ---------------------------------------------------------------------------
# the deny probe does not read tea leaves
# ---------------------------------------------------------------------------
def test_the_probe_verdict_comes_from_the_hook_not_the_transcript():
    """The deny message itself contains "gpu.py", and a model narrating a
    successful run can say "denied". A false `denied` is the one outcome this
    probe must never produce."""
    import asyncio

    import hooks

    hooks.DENIALS.clear()
    asyncio.run(
        hooks.pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "ssh probe-host echo hello"}}, None, None
        )
    )
    assert [d["command"] for d in hooks.DENIALS] == ["ssh probe-host echo hello"]

    hooks.DENIALS.clear()
    asyncio.run(
        hooks.pre_tool_use({"tool_name": "Bash", "tool_input": {"command": "pytest -q"}}, None, None)
    )
    assert hooks.DENIALS == []


def test_the_prompt_names_every_denied_command():
    """A command the gate always refuses, not named in the prompt, costs a turn."""
    from hooks import _DENIED_COMMANDS

    prompt = (paths.root().parent / "prompts" / "system.md")
    if not prompt.exists():  # GRAD_ROOT points at a temp dir during tests
        from pathlib import Path

        prompt = Path(__file__).resolve().parent.parent / "prompts" / "system.md"
    text = prompt.read_text(encoding="utf-8")
    missing = [name for name in _DENIED_COMMANDS if f"`{name}`" not in text]
    assert missing == []
