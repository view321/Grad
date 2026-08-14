"""The report and its gate (HANDOFF-2 §22).

§24: "8 needs a fixture ledger with a known-good and a known-bad claim set."
Both are built here, and every rule `check` enforces is tested from both sides --
because a gate that has only ever been shown to pass is a gate nobody has
tested.

The rule worth reading twice is rule 3:

    "no cited run has an unjudged deviation [...] Rule 3 is the one most in the
     spirit of this system: **you should not be able to write up a result you
     have not judged.**"
"""

from __future__ import annotations

import argparse
import json

import pytest

from core import budget, ledger_store as ls, report as report_lib
from core.errors import GradError
from tools import report


# ---------------------------------------------------------------------------
# fixtures: a real ledger
# ---------------------------------------------------------------------------
def make_run(project="proj-1", *, results, deviations, run_id=None, judged=False):
    run_id = run_id or ls.new_id("run")
    expectation = ls.append_expectation(
        {
            "id": ls.new_id("exp"),
            "task": "scaling",
            "created_at": ls.now_iso(),
            "quantity": next(iter(results), "val_loss"),
            "claim": "val loss should land between 2.9 and 3.2",
            "predicted": {"low": 2.9, "high": 3.2, "direction": None},
            "basis": [{"paper": "arXiv:2001.08361", "locator": "Fig 3", "value": 3.05,
                       "conditions": "1.3B params"}],
            "comparability": "our tokenizer differs",
            "confidence": "medium",
        }
    )
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": run_id, "task": "scaling",
            "status": "in_flight", "submitted_at": ls.now_iso(),
            "project": project, "estimate_usd": 1.0,
            "expectation_id": expectation["id"],
        }
    )
    ls.append_run_event(
        {
            "type": ls.T_RUN_COLLECTED, "id": run_id, "status": "completed",
            "collected_at": ls.now_iso(), "cost_usd_actual": 1.0,
            "results": results, "deviations": deviations,
        }
    )
    if judged:
        for dev in deviations:
            ls.append_run_event(
                {
                    "type": ls.T_VERDICT, "id": run_id, "quantity": dev["quantity"],
                    "verdict": "real", "note": "checked the schedule", "judged_at": ls.now_iso(),
                }
            )
    return run_id, expectation["id"]


@pytest.fixture
def project(workspace):
    budget.create("proj-1", title="scaling study", budget={})
    budget.set_current("proj-1")
    return "proj-1"


def in_range_run(project_id="proj-1"):
    return make_run(
        project_id,
        results={"val_loss": 3.05},
        deviations=[{"expectation_id": "e", "quantity": "val_loss", "actual": 3.05,
                     "in_range": True, "expected": {"low": 2.9, "high": 3.2}}],
    )


def unjudged_run(project_id="proj-1"):
    return make_run(
        project_id,
        results={"val_loss": 4.10},
        deviations=[{"expectation_id": "e", "quantity": "val_loss", "actual": 4.10,
                     "in_range": False, "expected": {"low": 2.9, "high": 3.2}}],
    )


def args(**kw):
    base = dict(project=None, json=True)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def stub_resolver(monkeypatch):
    """Resolve every placeholder to one corpus-backed entry.

    The draft emits `[CITE:<paper>]` for each basis entry, and `check` refuses
    while any placeholder survives -- so a test that wants to exercise a *later*
    rule has to run `cite` first, exactly as a real pipeline does.
    """
    monkeypatch.setattr(
        report, "_resolve_citation",
        lambda keyword, context, use_s2: {
            "key": "basis2026", "type": "article", "title": "Scaling Laws",
            "author": "Kaplan", "year": "2020", "gradsource": "corpus",
        },
    )


def draft_and_cite():
    report.cmd_draft(args())
    report.cmd_cite(args(context_chars=200, no_s2=True))


# ---------------------------------------------------------------------------
# draft: deterministic and free
# ---------------------------------------------------------------------------
def test_draft_needs_no_model_and_costs_nothing(workspace, project):
    run_id, _ = in_range_run()
    result = report.cmd_draft(args())
    tex = report_lib.paths_for(project)["tex"].read_text(encoding="utf-8")

    assert result["claim_count"] == 1
    assert r"\gradnum{" in tex
    assert run_id in tex


def test_draft_records_every_number_as_a_claim(workspace, project):
    run_id, _ = in_range_run()
    report.cmd_draft(args())
    claims = report_lib.load_claims(project)
    entry = next(iter(claims.values()))
    assert entry["run_id"] == run_id
    assert entry["quantity"] == "val_loss"
    assert entry["value"] == 3.05


def test_draft_surfaces_unjudged_deviations_rather_than_hiding_them(workspace, project):
    unjudged_run()
    result = report.cmd_draft(args())
    assert result["unjudged"]
    tex = report_lib.paths_for(project)["tex"].read_text(encoding="utf-8")
    assert "NOT YET JUDGED" in tex


def test_draft_lists_runs_with_no_bound_expectation(workspace, project):
    """"a skeleton that omits the runs that failed is a skeleton that invites
    writing up only the ones that worked." """
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": "run-orphan", "task": "t",
            "status": "in_flight", "submitted_at": ls.now_iso(), "project": "proj-1",
            "estimate_usd": 1.0, "expectation_id": None,
        }
    )
    ls.append_run_event(
        {
            "type": ls.T_RUN_COLLECTED, "id": "run-orphan", "status": "completed",
            "collected_at": ls.now_iso(), "cost_usd_actual": 1.0,
            "results": {"x": 1}, "deviations": [],
        }
    )
    report.cmd_draft(args())
    tex = report_lib.paths_for(project)["tex"].read_text(encoding="utf-8")
    assert "run-orphan" in tex


def test_draft_on_an_empty_project_says_so_rather_than_inventing(workspace, project):
    result = report.cmd_draft(args())
    assert result["claim_count"] == 0
    tex = report_lib.paths_for(project)["tex"].read_text(encoding="utf-8")
    assert "empty on purpose" in tex


# ---------------------------------------------------------------------------
# rule 1: claims
# ---------------------------------------------------------------------------
def test_a_gradnum_with_no_claims_entry_fails(workspace, project):
    findings = report_lib.check_claims(r"loss was \gradnum{missing}.", {})
    assert findings[0]["rule"] == "claims"
    assert "no entry in claims.json" in findings[0]["problem"]


def test_a_claim_pointing_at_a_nonexistent_run_fails(workspace, project):
    findings = report_lib.check_claims(
        r"\gradnum{k}", {"k": {"run_id": "run-nope", "quantity": "val_loss"}}
    )
    assert "not in the ledger" in findings[0]["problem"]


def test_a_claim_naming_a_quantity_the_run_never_reported_fails(workspace, project):
    run_id, _ = in_range_run()
    findings = report_lib.check_claims(
        r"\gradnum{k}", {"k": {"run_id": run_id, "quantity": "perplexity"}}
    )
    assert "reports no quantity" in findings[0]["problem"]


def test_a_claim_whose_value_does_not_match_the_ledger_fails(workspace, project):
    """The failure a citation-checker would miss: a real run, a real quantity,
    and a different number in the prose."""
    run_id, _ = in_range_run()
    findings = report_lib.check_claims(
        r"\gradnum{k}", {"k": {"run_id": run_id, "quantity": "val_loss", "value": 2.10}}
    )
    assert "recorded 3.05" in findings[0]["problem"]


def test_a_correct_claim_passes(workspace, project):
    run_id, _ = in_range_run()
    assert report_lib.check_claims(
        r"\gradnum{k}", {"k": {"run_id": run_id, "quantity": "val_loss", "value": 3.05}}
    ) == []


def test_rounding_in_the_prose_is_tolerated(workspace, project):
    run_id, _ = in_range_run()
    assert report_lib.check_claims(
        r"\gradnum{k}", {"k": {"run_id": run_id, "quantity": "val_loss", "value": 3.0500001}}
    ) == []


# ---------------------------------------------------------------------------
# rule 2: citations
# ---------------------------------------------------------------------------
def test_a_cite_key_with_no_bib_entry_fails(workspace):
    findings = report_lib.check_citations(r"as shown \cite{ghost}.", {})
    assert findings[0]["rule"] == "citations"
    assert "no entry in references.bib" in findings[0]["problem"]


def test_a_bib_entry_with_no_verified_provenance_fails(workspace):
    """"every bib entry came from the corpus or a verified S2 id." A
    hand-written entry is exactly the hallucinated citation this stops."""
    bib = {"invented2026": {"type": "article", "key": "invented2026", "title": "A Paper"}}
    findings = report_lib.check_citations(r"\cite{invented2026}", bib)
    assert any("verified provenance" in f["problem"] for f in findings)


def test_a_corpus_backed_entry_passes(workspace):
    bib = {"real2026": {"type": "article", "key": "real2026", "gradsource": "corpus"}}
    assert report_lib.check_citations(r"\cite{real2026}", bib) == []


def test_an_s2_verified_entry_passes(workspace):
    bib = {"real2026": {"type": "article", "key": "real2026", "gradsource": "s2"}}
    assert report_lib.check_citations(r"\cite{real2026}", bib) == []


def test_multi_key_cites_are_all_checked(workspace):
    bib = {"a": {"type": "article", "key": "a", "gradsource": "corpus"}}
    findings = report_lib.check_citations(r"\cite{a,b}", bib)
    assert [f["key"] for f in findings] == ["b"]


def test_citep_and_citet_are_recognised(workspace):
    findings = report_lib.check_citations(r"\citep{ghost} and \citet{ghost2}", {})
    assert {f["key"] for f in findings} == {"ghost", "ghost2"}


def test_the_generated_bib_carries_provenance(workspace):
    text = report._render_bib(
        {"k": {"type": "article", "key": "k", "title": "T", "author": "A",
               "year": "2026", "gradsource": "corpus"}}
    )
    parsed = report_lib.parse_bib(text)
    assert parsed["k"]["gradsource"] == "corpus"
    assert report_lib.check_citations(r"\cite{k}", parsed) == []


def test_an_unresolvable_placeholder_is_left_in_place_and_refused(workspace, project, monkeypatch):
    """"a citation quietly deleted is worse than one that fails loudly, because
    the sentence it supported survives without support." """
    in_range_run()
    report.cmd_draft(args())
    monkeypatch.setattr(report, "_resolve_citation", lambda *a, **k: None)

    with pytest.raises(GradError) as exc:
        report.cmd_cite(args(context_chars=200, no_s2=True))
    assert exc.value.code == "citations_unresolved"

    tex = report_lib.paths_for(project)["tex"].read_text(encoding="utf-8")
    assert "[CITE:" in tex, "the placeholder must survive so `check` refuses on it"


def test_a_resolved_placeholder_becomes_a_cite(workspace, project, monkeypatch):
    in_range_run()
    report.cmd_draft(args())
    monkeypatch.setattr(
        report, "_resolve_citation",
        lambda keyword, context, use_s2: {
            "key": "real2026", "type": "article", "title": "T", "author": "A",
            "year": "2026", "gradsource": "corpus",
        },
    )
    result = report.cmd_cite(args(context_chars=200, no_s2=True))
    tex = report_lib.paths_for(project)["tex"].read_text(encoding="utf-8")
    assert result["entries"] == 1
    assert r"\cite{real2026}" in tex
    assert "[CITE:" not in tex


# ---------------------------------------------------------------------------
# rule 3: unjudged deviations -- the one most in the spirit of the system
# ---------------------------------------------------------------------------
def test_a_cited_run_with_an_unjudged_deviation_refuses(workspace, project):
    """"You should not be able to write up a result you have not judged." """
    unjudged_run()
    report.cmd_draft(args())
    with pytest.raises(GradError) as exc:
        report.cmd_check(args())
    detail = exc.value.detail
    assert detail["by_rule"]["unjudged"] == 1
    unjudged = [f for f in detail["findings"] if f["rule"] == "unjudged"][0]
    assert "tools.ledger verdict" in unjudged["fix"]


def test_supplying_the_verdict_clears_the_refusal(workspace, project, stub_resolver):
    run_id, _ = make_run(
        results={"val_loss": 4.10},
        deviations=[{"expectation_id": "e", "quantity": "val_loss", "actual": 4.10,
                     "in_range": False, "expected": {"low": 2.9, "high": 3.2}}],
        judged=True,
    )
    draft_and_cite()
    result = report.cmd_check(args())
    assert result["ok"] is True
    assert run_id in result["cited_runs"]


def test_an_in_range_result_needs_no_verdict(workspace, project, stub_resolver):
    in_range_run()
    draft_and_cite()
    assert report.cmd_check(args())["ok"] is True


def test_an_unjudged_run_that_is_not_cited_does_not_block(workspace, project, stub_resolver):
    """The rule is about *cited* runs. An unrelated open verdict elsewhere in
    the project is `ledger query --pending`'s business, not this report's."""
    in_range_run()
    draft_and_cite()
    unjudged_run()  # after the draft, so no \gradnum points at it
    assert report.cmd_check(args())["ok"] is True


# ---------------------------------------------------------------------------
# rule 4: LaTeX hygiene
# ---------------------------------------------------------------------------
def test_unmatched_braces_are_found(workspace):
    findings = report_lib.check_latex("\\section{Results\n\nsome text\n")
    assert any("unmatched opening brace" in f["problem"] for f in findings)


def test_a_closing_brace_with_no_opener_is_found(workspace):
    findings = report_lib.check_latex("text }\n")
    assert any("no opener" in f["problem"] for f in findings)


def test_escaped_braces_do_not_count(workspace):
    assert report_lib.check_latex(r"a \{ literal \} brace" + "\n") == []


def test_duplicate_labels_are_found(workspace):
    findings = report_lib.check_latex("\\label{a}\ntext\n\\label{a}\n")
    assert any("duplicate" in f["problem"] for f in findings)


def test_leftover_placeholders_are_found(workspace):
    findings = report_lib.check_latex("as shown [CITE:transformers].\n")
    assert any("unresolved [CITE:" in f["problem"] for f in findings)


def test_comments_do_not_confuse_the_brace_counter(workspace):
    assert report_lib.check_latex("% a comment with { an unmatched brace\ntext\n") == []


# ---------------------------------------------------------------------------
# the pipeline, and where the gate sits
# ---------------------------------------------------------------------------
def test_check_reports_every_rule_it_ran(workspace, project, stub_resolver):
    in_range_run()
    draft_and_cite()
    result = report.cmd_check(args())
    assert set(result["by_rule"]) == {"claims", "citations", "unjudged", "latex"}


def test_check_refuses_while_a_placeholder_survives(workspace, project, monkeypatch):
    """The pipeline order is enforced, not assumed: `cite` runs before `check`,
    and an unresolved placeholder is a hard failure rather than cosmetic."""
    in_range_run()
    report.cmd_draft(args())
    with pytest.raises(GradError) as exc:
        report.cmd_check(args())
    assert exc.value.detail["by_rule"]["latex"] == 1


def test_check_exits_9(workspace, project, capsys):
    unjudged_run()
    report.cmd_draft(args())
    assert report.cli.run(["check", "--project", "proj-1", "--json"]) == 9
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] is False


def test_check_refuses_it_does_not_warn(workspace, project):
    """"`check` refuses; it does not warn. A report generator is where this
    system's epistemics either hold or collapse." """
    unjudged_run()
    report.cmd_draft(args())
    with pytest.raises(GradError):
        report.cmd_check(args())


def test_write_is_denied_while_over_budget(workspace):
    """§23 item 5, "currently specified as denied by the §15 hook" -- and the
    CLI agrees with the hook rather than contradicting it."""
    budget.create("proj-1", title="t", budget={"gpu_usd": 1.0})
    budget.set_current("proj-1")
    in_range_run()
    report.cmd_draft(args())
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": ls.new_id("run"), "status": "in_flight",
            "submitted_at": ls.now_iso(), "project": "proj-1", "estimate_usd": 50.0,
        }
    )
    with pytest.raises(GradError) as exc:
        report.cmd_write(args(section=[], dry_run=False))
    assert exc.value.exit_code == 12
    # And it points at the free command that answers "what did the spend buy".
    assert "report draft" in (exc.value.fix or "")


def test_write_refuses_without_a_draft(workspace, project):
    with pytest.raises(GradError) as exc:
        report.cmd_write(args(section=[], dry_run=False))
    assert exc.value.code == "no_draft"


def test_the_bundle_hands_the_model_keys_not_numbers(workspace, project):
    """"handing it more numbers than it has keys for is how a number ends up in
    the prose without a \\gradnum around it." """
    in_range_run()
    report.cmd_draft(args())
    result = report.cmd_write(args(section=[], dry_run=True))
    bundle = result["bundle"]
    assert result["sent"] is False
    for entry in bundle["claims"].values():
        assert "value" not in entry


def test_a_report_needs_a_project(workspace):
    from core.errors import UsageError

    with pytest.raises(UsageError) as exc:
        report.cmd_draft(args())
    assert "tools.budget use" in (exc.value.fix or "")


# ---------------------------------------------------------------------------
# review fixes
# ---------------------------------------------------------------------------
def test_rerunning_cite_preserves_earlier_entries(workspace, project, stub_resolver):
    r"""`cite` is naturally re-run -- after a new section, or after ingesting a
    paper that failed to resolve last time. By then the earlier placeholders are
    already `\cite{}` keys, so a second pass finds nothing to resolve; rewriting
    the bib from scratch deleted every entry the first pass earned and left
    `check` refusing on citations that were fine a moment ago.
    """
    in_range_run()
    report.cmd_draft(args())
    first = report.cmd_cite(args(context_chars=200, no_s2=True))
    assert first["entries"] == 1

    second = report.cmd_cite(args(context_chars=200, no_s2=True))
    assert second["entries"] == 1, "the earlier entry must survive"
    assert second["kept_from_previous_run"] == ["basis2026"]
    assert report.cmd_check(args())["ok"] is True


def test_a_weakly_related_s2_hit_is_rejected(workspace, project, monkeypatch):
    """An 8% word overlap was close to a rubber stamp: any two ML papers share
    enough vocabulary to clear it. A citation wrongly accepted is a claim
    silently attributed to a paper that does not support it."""
    class FakeS2:
        def __init__(self, cfg): ...
        def paper_search(self, keyword, limit=5):
            return [{"paper_id": "abc", "title": "Convolutional Networks for Image Segmentation",
                     "abstract": "We segment images using convolutions and pooling layers.",
                     "year": 2015}]

    monkeypatch.setattr(report.http, "SemanticScholar", FakeS2)
    context = (
        "The scaling behaviour of transformer language models under a fixed token "
        "budget follows a power law in parameter count across pretraining corpora."
    )
    assert report._from_s2("scaling laws", context) is None


def test_a_genuinely_matching_s2_hit_is_accepted_and_scored(workspace, project, monkeypatch):
    class FakeS2:
        def __init__(self, cfg): ...
        def paper_search(self, keyword, limit=5):
            return [{"paper_id": "abc",
                     "title": "Scaling Laws for Neural Language Models",
                     "abstract": "We study empirical scaling laws for language model "
                                 "performance as a function of parameter count and "
                                 "pretraining token budget, finding power-law behaviour.",
                     "year": 2020}]

    monkeypatch.setattr(report.http, "SemanticScholar", FakeS2)
    context = (
        "Empirical scaling laws for language models describe performance as a "
        "power-law function of parameter count and pretraining token budget."
    )
    entry = report._from_s2("scaling laws", context)
    assert entry is not None
    assert entry["gradsource"] == "s2"
    # Both scores recorded, so a borderline resolution is auditable.
    assert entry["gradmatch"] >= report.S2_MIN_CONTEXT_OVERLAP
    assert entry["gradtitlematch"] >= report.S2_MIN_TITLE_OVERLAP


def test_shared_jargon_alone_does_not_clear_the_title_test(workspace, project, monkeypatch):
    """The abstract can overlap on generic research vocabulary; the title is
    where a paper's actual subject lives."""
    class FakeS2:
        def __init__(self, cfg): ...
        def paper_search(self, keyword, limit=5):
            return [{"paper_id": "abc", "title": "A Dataset of Annotated Radiographs",
                     "abstract": "scaling laws parameter count pretraining token budget "
                                 "power law transformer language models corpora",
                     "year": 2021}]

    monkeypatch.setattr(report.http, "SemanticScholar", FakeS2)
    context = ("Scaling laws relate parameter count and pretraining token budget "
               "to transformer language model loss following a power law.")
    assert report._from_s2("scaling laws", context) is None


def test_the_right_paper_is_not_taken_down_by_a_keyword_stuffed_one(workspace, project, monkeypatch):
    """Both gates are applied before ranking, not to the winner afterwards.

    Ranking first let a loosely-related paper with a keyword-heavy abstract win
    on context overlap, fail the title gate, and reject the genuinely correct
    paper sitting next to it in the candidate list.
    """
    class FakeS2:
        def __init__(self, cfg): ...
        def paper_search(self, keyword, limit=5):
            return [
                {"paper_id": "generic", "title": "Miscellaneous Notes",
                 "abstract": "scaling laws parameter count pretraining token budget power "
                             "transformer language corpora empirical performance",
                 "year": 2021},
                {"paper_id": "right", "title": "Scaling Laws for Neural Language Models",
                 "abstract": "empirical scaling laws parameter count pretraining token budget",
                 "year": 2020},
            ]

    monkeypatch.setattr(report.http, "SemanticScholar", FakeS2)
    context = ("Empirical scaling laws relate parameter count and pretraining token "
               "budget to transformer language model performance.")
    entry = report._from_s2("scaling laws", context)
    assert entry is not None
    assert entry["note"] == "S2:right"


def test_partial_usage_keeps_cache_counters(workspace, project):
    """A turn that dies before the result message still spent cache traffic, and
    a long prompt spends most of its tokens there."""
    import inspect

    source = inspect.getsource(report._generate_prose)
    assert "cache_read_input_tokens" in source
    assert "cache_creation_input_tokens" in source


def test_report_usage_is_charged_to_the_reported_project(workspace, monkeypatch):
    """`--project` can name a project other than the selected one; charging the
    report's tokens to whichever happened to be current attributes the spend to
    the wrong allocation."""
    budget.create("proj-a", title="a", budget={})
    budget.create("proj-b", title="b", budget={})
    budget.set_current("proj-a")

    recorded: dict = {}

    def capture(stage, usage, **kw):
        recorded.update(kw)
        return None

    monkeypatch.setattr(report.quota_log, "from_sdk_usage", capture)
    monkeypatch.setattr(
        report, "_generate_prose",
        lambda bundle, *, model, project=None: (
            capture("report.write", {}, project=project, model=model, role="report")
            or r"\section{Results}" + "\nbody text long enough to pass validation." * 5
        ),
    )

    in_range_run("proj-b")
    report.cmd_draft(args(project="proj-b"))
    report.cmd_write(args(project="proj-b", section=[], dry_run=False))
    assert recorded["project"] == "proj-b"
