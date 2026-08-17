"""The project wiki: what is extracted, what is written, and the line between.

The design is half retrieved and half generated, and every test here is about
keeping that line where it is. The extracted half must be true by construction
-- read off disk, no model, no network, and above all no *import* of the code it
describes, since a research pipeline downloads a checkpoint at module scope. The
generated half must be unable to pass off prose as fact: it is handed a sheet,
its citations are checked against that sheet afterwards, and a page that fails
is a missing page rather than a wrong one.
"""

from __future__ import annotations

import argparse

import pytest

from core import budget as budget_mod, jsonl, paths, projects, projwiki, wikigen
from core.errors import GradError, NotFound
from tools import projwiki as projwiki_tool

SPEC = """\
# The comment that explains the choice, which is the point of keeping the text.
entrypoint   = "train.py"
image        = "pytorch/pytorch@sha256:0000"
metrics_file = "metrics.json"

[estimate]
hours = 2.0

[target]
platform    = "kaggle"
# T4, chosen on evidence: an L4 request came back as a P100 last time.
accelerator = "NvidiaTeslaT4"
"""

TRAIN = '''\
"""Train the thing, and write metrics.json."""

import data

STEPS = 3000


class Model:
    """The architecture under test."""

    def __init__(self, width):
        self.width = width

    def forward(self, x):
        return x

    def _internal(self):
        return None


def main(argv=None):
    """Entry point."""
    return Model(8)
'''

DATA = '''\
"""Build the corpus."""


def build_corpus(n_docs):
    """Tokenise and cache."""
    return []
'''


@pytest.fixture
def project(workspace):
    """A project with a pipeline, a spec and a prediction bound to a run."""
    budget_mod.create("demo", title="Demo", budget={})
    budget_mod.set_current("demo")
    projects.scaffold("demo")
    (projects.resolve_dir("demo") / "PLAN.md").write_text(
        "# Plan\n\nEstablish whether width helps at fixed compute.\n", encoding="utf-8"
    )
    pipeline = paths.root() / "pipelines" / "demo"
    pipeline.mkdir(parents=True, exist_ok=True)
    (pipeline / "spec.toml").write_text(SPEC, encoding="utf-8")
    (pipeline / "train.py").write_text(TRAIN, encoding="utf-8")
    (pipeline / "data.py").write_text(DATA, encoding="utf-8")
    (pipeline / "test_train.py").write_text(
        '"""Checks."""\n\n\ndef test_forward_is_causal():\n    assert True\n', encoding="utf-8"
    )
    (pipeline / "__pycache__").mkdir(exist_ok=True)
    (pipeline / "__pycache__" / "train.cpython-314.pyc").write_bytes(b"\x00")
    return "demo"


# ---------------------------------------------------------------------------
# the extracted half
# ---------------------------------------------------------------------------
def test_symbols_are_read_without_importing_the_module(workspace):
    """The load-bearing one. A pipeline's `train.py` imports torch and pulls a
    checkpoint at module scope, so introspecting it by importing it would run
    the experiment to write a page about it."""
    path = workspace / "explodes.py"
    path.write_text(
        'raise SystemExit("this module must never be executed")\n\n\ndef f(a, b=1):\n    """Doc."""\n',
        encoding="utf-8",
    )
    found, warnings = projwiki.symbols(path)
    assert warnings == []
    assert [s["name"] for s in found] == ["f"]
    assert found[0]["signature"] == "def f(a, b)"


def test_every_symbol_carries_the_line_it_is_on(project):
    """Line numbers are what make a citation checkable. Without them a page
    saying "`build_corpus` handles tokenisation" is unfalsifiable prose."""
    facts = projwiki.collect(project)
    module = _module(facts, "data.py")
    symbol = next(s for s in module["symbols"] if s["name"] == "build_corpus")
    assert symbol["line"] == 4
    assert symbol["doc"] == "Tokenise and cache."


def test_a_class_brings_its_public_methods(project):
    """A `nn.Module` listed without `__init__` and `forward` says nothing about
    the architecture, which is the one thing a reader came for."""
    facts = projwiki.collect(project)
    model = next(s for s in _module(facts, "train.py")["symbols"] if s["name"] == "Model")
    names = [m["name"] for m in model["methods"]]
    assert "Model.__init__" in names and "Model.forward" in names
    assert "Model._internal" not in names


def test_module_level_constants_are_extracted(project):
    """`STEPS = 3000` is configuration wearing a different hat, and it is what a
    reader looks for first."""
    facts = projwiki.collect(project)
    steps = next(s for s in _module(facts, "train.py")["symbols"] if s["name"] == "STEPS")
    assert steps["kind"] == "constant"
    assert steps["signature"] == "STEPS = 3000"


def test_a_file_that_will_not_parse_is_a_warning_not_a_failure(project):
    """Half-written code is the normal state of a pipeline the agent is editing.
    A wiki that refuses to build until the syntax is fixed is a wiki nobody
    builds."""
    (paths.root() / "pipelines" / "demo" / "broken.py").write_text("def (\n", encoding="utf-8")
    facts = projwiki.collect(project)
    assert any("broken.py" in w for w in facts["warnings"])
    assert _module(facts, "train.py")["symbols"]


def test_the_spec_is_kept_as_text_as_well_as_parsed(project):
    """The parsed form is what a page can be checked against; the comments are
    where the reasoning is, and they are the highest-signal thing in the tree."""
    facts = projwiki.collect(project)
    spec = facts["pipelines"][0]["specs"][0]
    assert spec["parsed"]["target"]["accelerator"] == "NvidiaTeslaT4"
    assert "came back as a P100" in spec["text"]


def test_the_import_graph_marks_what_no_entrypoint_reaches(project):
    """A module nothing imports is a fact about the pipeline, and usually an
    interesting one -- so it is listed and flagged, never dropped."""
    facts = projwiki.collect(project)
    assert _module(facts, "data.py")["reachable"] is True
    assert _module(facts, "test_train.py")["reachable"] is False
    assert _module(facts, "data.py")["imports"] == []
    assert _module(facts, "train.py")["imports"] == ["data"]


def test_generated_bytecode_is_not_part_of_the_pipeline(project):
    facts = projwiki.collect(project)
    assert not any("__pycache__" in row["path"] for row in facts["pipelines"][0]["tree"])
    assert not any("__pycache__" in k for k in facts["source"]["files"])


def test_a_project_that_does_not_exist_says_so(workspace):
    with pytest.raises(NotFound):
        projwiki.collect("nope")


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------
def test_the_hash_moves_when_the_pipeline_does(project):
    before = projwiki.source_hash(project)["hash"]
    (paths.root() / "pipelines" / "demo" / "train.py").write_text(
        TRAIN.replace("3000", "4000"), encoding="utf-8"
    )
    assert projwiki.source_hash(project)["hash"] != before


def test_a_new_run_does_not_make_the_wiki_stale(project):
    """A wiki is not stale because a result arrived. It is stale because the
    code it describes changed -- and folding the ledger into the digest would
    mark every page stale after every collect."""
    from core import ledger_store as ls

    before = projwiki.source_hash(project)["hash"]
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-x", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "project": "demo", "estimate_usd": 1.0}
    )
    assert projwiki.source_hash(project)["hash"] == before


def test_notes_and_papers_are_never_read(project):
    """The generated half ships what it is given to a model, so the scope is an
    allowlist -- the same discipline `tools/wiki.py` applies, for the same
    reason. Research notes are not documentation."""
    (paths.notes_dir()).mkdir(parents=True, exist_ok=True)
    (paths.notes_dir() / "private.md").write_text("unpublished idea", encoding="utf-8")
    facts = projwiki.collect(project)
    blob = wikigen.overview_sheet(facts)
    assert "unpublished idea" not in blob
    assert not any("notes" in k for k in facts["source"]["files"])


# ---------------------------------------------------------------------------
# the line between the halves
# ---------------------------------------------------------------------------
def test_a_page_with_no_citations_is_rejected_before_it_is_stored(project):
    """A returned error is what makes the model try again. Enforcing this in the
    prompt alone would make unsourced prose a thing that happens sometimes."""
    assert wikigen._validate_page(
        {"title": "t", "summary": "s", "sections": [{"heading": "h", "body": "b", "refs": []}]}
    )
    assert (
        wikigen._validate_page(
            {"title": "t", "summary": "s",
             "sections": [{"heading": "h", "body": "b", "refs": ["train.py:4"]}]}
        )
        is None
    )


def test_citations_are_checked_against_the_extracted_facts(project):
    """The mechanism that separates this from a plausible essay. A page may cite
    only what was extracted, and what it cites that was not is reported."""
    facts = projwiki.collect(project)
    unresolved = wikigen.verify_refs(
        facts,
        {
            "sections": [
                {"refs": ["train.py:20", "spec.toml", "PLAN.md"]},
                {"refs": ["optimiser.py:12", "run-invented"]},
            ]
        },
    )
    assert unresolved == ["optimiser.py:12", "run-invented"]


def test_a_line_number_that_has_moved_still_resolves(project):
    """Strictness here would mark the whole wiki unverified after a one-line
    edit, which would train the reader to ignore the marking."""
    facts = projwiki.collect(project)
    assert wikigen.verify_refs(facts, {"sections": [{"refs": ["train.py:999"]}]}) == []


def test_a_ledger_id_is_citable_and_an_invented_one_is_not(project):
    from core import ledger_store as ls

    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-real", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "project": "demo", "estimate_usd": 1.0}
    )
    facts = projwiki.collect(project)
    assert wikigen.verify_refs(facts, {"sections": [{"refs": ["run-real"]}]}) == []
    assert wikigen.verify_refs(facts, {"sections": [{"refs": ["run-fake"]}]}) == ["run-fake"]


def test_the_fact_sheet_holds_the_ledger_the_page_will_cite(project):
    from core import ledger_store as ls

    ls.append_expectation(
        {"id": "exp-1", "task": "t", "created_at": ls.now_iso(), "quantity": "val_loss",
         "project": "demo", "predicted": {"low": 2.9, "high": 3.2, "direction": None},
         "basis": [{"paper": "arXiv:2405.21060", "locator": "fig 3", "value": "3.0",
                    "conditions": "same setup"}],
         "comparability": "same", "confidence": "medium"}
    )
    facts = projwiki.collect(project)
    sheet = wikigen.overview_sheet(facts)
    assert "exp-1" in sheet
    assert "arXiv:2405.21060" in sheet
    assert "Establish whether width helps" in sheet  # PLAN.md, the authored intent


def test_a_module_sheet_carries_the_module_and_not_the_whole_pipeline(project):
    facts = projwiki.collect(project)
    pipeline = facts["pipelines"][0]
    module = _module(facts, "data.py")
    sheet = wikigen.module_sheet(facts, pipeline, module)
    assert "def build_corpus(n_docs)" in sheet
    assert "Imported by: train.py" in sheet
    # `train.py`'s own body is another page's subject.
    assert "class Model" not in sheet


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------
def _build(project_id, **kw):
    args = argparse.Namespace(
        project=project_id, no_prose=True, model=None, page=[], json=True, **kw
    )
    return projwiki_tool.cmd_build(args)


def test_no_prose_produces_the_whole_retrieved_wiki_without_a_model(project, monkeypatch):
    """Free, offline and useful on its own -- and the thing that makes the
    extraction testable without a network."""
    def refuse(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("--no-prose must not call a model")

    monkeypatch.setattr(wikigen, "write_page", refuse)
    out = _build(project)

    assert out["pages_planned"] >= 3
    assert out["pages_written"] == 0
    assert (projwiki_tool.output_dir(project) / "facts.json").is_file()


def test_the_pages_planned_cover_the_modules_but_not_the_tests(project, monkeypatch):
    """Tests are described in the page for the code they test. A wiki whose back
    half is one page per `test_*.py` buries the pages worth reading."""
    monkeypatch.setattr(wikigen, "write_page", lambda *a, **k: {})
    _build(project)
    listed = projwiki_tool.cmd_show(
        argparse.Namespace(project=project, page=None, facts=False, json=True)
    )["pages"]
    ids = {p["id"] for p in listed}
    assert "overview" in ids
    assert "run-path-demo" in ids
    assert "module-demo-train.py" in ids
    assert not any("test_train" in i for i in ids)


def test_one_page_failing_does_not_lose_the_others(project, monkeypatch):
    """Half a wiki whose gaps are visible is worth more than a whole one with an
    invented page in it -- and much more than no wiki because call four of nine
    timed out."""
    calls = {"n": 0}

    def flaky(facts, page, *, model, log_name):
        calls["n"] += 1
        if page["kind"] == "module":
            raise RuntimeError("upstream said no")
        return {**page, "summary": "s", "sections": [{"heading": "h", "body": "b", "refs": ["spec.toml"]}],
                "open_questions": [], "unverified_refs": []}

    monkeypatch.setattr(wikigen, "write_page", flaky)
    out = projwiki_tool.cmd_build(
        argparse.Namespace(project=project, no_prose=False, model=None, page=[], json=True)
    )

    assert out["pages_written"] >= 2
    assert [f["page"] for f in out["failed"]]
    assert all("upstream said no" in f["error"] for f in out["failed"])
    # And the failure is legible in the page itself, not only in the envelope.
    pages = jsonl.read_json(projwiki_tool.output_dir(project) / "pages.json")
    assert any(p.get("error") for p in pages)


def test_rebuilding_one_page_keeps_the_rest(project, monkeypatch):
    monkeypatch.setattr(
        wikigen, "write_page",
        lambda facts, page, **k: {**page, "summary": "first", "sections": [
            {"heading": "h", "body": "b", "refs": ["spec.toml"]}], "open_questions": [],
            "unverified_refs": []},
    )
    projwiki_tool.cmd_build(
        argparse.Namespace(project=project, no_prose=False, model=None, page=[], json=True)
    )
    monkeypatch.setattr(
        wikigen, "write_page",
        lambda facts, page, **k: {**page, "summary": "second", "sections": [
            {"heading": "h", "body": "b", "refs": ["spec.toml"]}], "open_questions": [],
            "unverified_refs": []},
    )
    projwiki_tool.cmd_build(
        argparse.Namespace(project=project, no_prose=False, model=None, page=["overview"], json=True)
    )

    pages = {p["id"]: p for p in jsonl.read_json(projwiki_tool.output_dir(project) / "pages.json")}
    assert pages["overview"]["summary"] == "second"
    assert pages["run-path-demo"]["summary"] == "first"


def test_a_project_with_no_pipeline_gets_an_honest_refusal(workspace):
    """A project whose code has not been written yet is a normal thing to find,
    and an overview page about nothing would be worse than saying so."""
    budget_mod.create("empty", title="E", budget={})
    projects.scaffold("empty")
    with pytest.raises(GradError) as exc:
        _build("empty")
    assert exc.value.code == "no_pipeline"


def test_check_reports_stale_once_the_code_moves(project, monkeypatch):
    monkeypatch.setattr(wikigen, "write_page", lambda *a, **k: {})
    _build(project)
    assert projwiki_tool.cmd_check(argparse.Namespace(project=project, json=True))["current"] is True

    (paths.root() / "pipelines" / "demo" / "train.py").write_text(
        TRAIN.replace("3000", "4000"), encoding="utf-8"
    )
    with pytest.raises(GradError) as exc:
        projwiki_tool.cmd_check(argparse.Namespace(project=project, json=True))
    assert exc.value.code == "wiki_stale"
    assert "pipelines/demo/train.py" in exc.value.detail["changed"]


def test_check_refuses_before_anything_is_built(project):
    with pytest.raises(GradError) as exc:
        projwiki_tool.cmd_check(argparse.Namespace(project=project, json=True))
    assert exc.value.code == "no_wiki"


def test_the_markdown_marks_citations_that_could_not_be_resolved(project):
    """A reader deciding how far to trust a paragraph is entitled to know which
    of its citations matched nothing."""
    text = wikigen.as_markdown(
        {
            "title": "T", "summary": "S",
            "sections": [{"heading": "H", "body": "B", "refs": ["train.py:4", "ghost.py:1"]}],
            "open_questions": ["Was the grid submitted?"],
            "unverified_refs": ["ghost.py:1"],
        }
    )
    assert "matched nothing in the extracted facts" in text
    assert "ghost.py:1" in text
    assert "Was the grid submitted?" in text


def _module(facts, name):
    return next(m for m in facts["pipelines"][0]["modules"] if m["path"] == name)
