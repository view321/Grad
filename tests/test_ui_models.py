"""What each window shows, as plain data.

Everything interesting about a window is a decision -- which expectation counts
as broken, what the band strip does when the band is a single point, whether an
uncollected run is "running" or "waiting" -- and none of those decisions need a
browser. These tests run with the `ui` extra uninstalled, which is also what
keeps `ui/models.py` honest about never importing NiceGUI.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core import ledger_store as ls, paths
from ui import models


# ---------------------------------------------------------------------------
# the rule that makes the rest of this file possible
# ---------------------------------------------------------------------------
def test_the_model_layer_does_not_drag_in_nicegui():
    """`ui/models.py` and `ui/layout.py` are the tested half of the UI. If either
    grows a NiceGUI import the tests stop being runnable without the extra, and
    the layering claim in `ui/__init__.py` stops being true.

    The rule is about *importing* NiceGUI, not about naming it. `ui/tokens.py`
    has to write `.nicegui-content` into the stylesheet -- that wrapper is what
    the shell is nested inside, and its padding has to be zeroed out from CSS --
    and a selector for someone else's class name costs nothing at import time.
    So this matches import statements rather than the bare substring.
    """
    imports = re.compile(
        r"""^\s*(?:from|import)\s+nicegui\b"""      # from nicegui import ... / import nicegui
        r"""|__import__\(\s*['"]nicegui"""          # the dynamic spellings
        r"""|import_module\(\s*['"]nicegui""",
        re.MULTILINE,
    )
    for module in ("ui.models", "ui.layout", "ui.registry", "ui.tokens", "ui.fonts"):
        source = __import__(module, fromlist=["__file__"]).__file__
        assert source
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
        found = imports.search(text)
        assert not found, f"{module} must not import nicegui: {found.group(0).strip()!r}"


# ---------------------------------------------------------------------------
# the band strip
# ---------------------------------------------------------------------------
def test_a_closed_band_puts_the_observed_value_where_it_belongs():
    geometry = models.band_geometry(low=2.9, high=3.2, actual=3.05)
    assert geometry is not None
    assert geometry["in_band"] is True
    assert 0.0 < geometry["band_start"] < geometry["actual"] < geometry["band_end"] < 1.0


def test_an_observed_value_outside_the_band_is_not_in_band():
    geometry = models.band_geometry(low=2.9, high=3.2, actual=4.1)
    assert geometry["in_band"] is False
    assert geometry["actual"] > geometry["band_end"]


def test_a_half_open_band_reaches_the_axis_edge():
    """`compute_deviations` treats a missing bound as infinity, so "below 3.2"
    has to draw as a block reaching the wall -- not as nothing at all."""
    geometry = models.band_geometry(low=None, high=3.2, actual=3.0)
    assert geometry["band_start"] == 0.0
    assert geometry["open_low"] is True
    assert geometry["in_band"] is True

    other = models.band_geometry(low=2.9, high=None, actual=9.0)
    assert other["band_end"] == 1.0
    assert other["in_band"] is True


def test_a_point_prediction_still_renders():
    geometry = models.band_geometry(low=3.0, high=3.0, actual=3.0)
    assert geometry is not None
    assert 0.0 < geometry["actual"] < 1.0
    assert geometry["axis_min"] < 3.0 < geometry["axis_max"]


def test_an_inverted_band_is_read_the_right_way_round():
    assert models.band_geometry(low=3.2, high=2.9, actual=3.05)["in_band"] is True


def test_a_relational_prediction_has_no_band_to_draw():
    """§7 prefers relational expectations. A degenerate strip for them would be
    worse than none: it would assert a comparison the ledger never made."""
    assert models.band_geometry(low=None, high=None, actual=3.0) is None


def test_a_non_numeric_result_has_no_band_either():
    assert models.band_geometry(low=1, high=2, actual="diverged") is None
    assert models.band_geometry(low=1, high=2, actual=None) is None


def test_every_position_stays_inside_the_strip():
    geometry = models.band_geometry(low=0.0, high=1.0, actual=1e6)
    for key in ("band_start", "band_end", "actual"):
        assert 0.0 <= geometry[key] <= 1.0


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
def _expect(workspace, **overrides):
    record = {
        "id": ls.new_id("exp"),
        "task": "t",
        "created_at": ls.now_iso(),
        "quantity": "val_loss",
        "claim": "val loss lands between 2.9 and 3.2",
        "predicted": {"low": 2.9, "high": 3.2, "direction": None},
        "basis": [{"paper": "arXiv:2001.08361", "locator": "Table 3", "value": 3.05, "conditions": "1.3B"}],
        "comparability": "our tokenizer differs",
        "confidence": "medium",
    }
    record.update(overrides)
    ls.append_expectation(record)
    return record


def _run(run_id: str, expectation_id: str | None, actual, in_range, **overrides):
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "task": "t",
            "status": "in_flight",
            "submitted_at": ls.now_iso(),
            "estimate_usd": 4.0,
            "estimated_duration_s": 60,
            **overrides,
        }
    )
    if actual is not None or in_range is not None:
        ls.append_run_event(
            {
                "type": ls.T_RUN_COLLECTED,
                "id": run_id,
                "status": "completed",
                "collected_at": ls.now_iso(),
                "cost_usd_actual": 3.5,
                "results": {"val_loss": actual},
                "deviations": [
                    {
                        "expectation_id": expectation_id,
                        "quantity": "val_loss",
                        "expected": {"low": 2.9, "high": 3.2, "direction": None},
                        "actual": actual,
                        "in_range": in_range,
                    }
                ],
            }
        )


def test_an_expectation_with_no_run_is_open(workspace):
    _expect(workspace)
    model = models.ledger_model()
    assert model["counts"]["open"] == 1
    assert model["entries"][0]["band"] is None


def test_an_expectation_whose_run_landed_in_band_is_met(workspace):
    record = _expect(workspace)
    _run("run-1", record["id"], 3.05, True)
    model = models.ledger_model()
    assert model["counts"]["met"] == 1
    assert model["entries"][0]["accent"] == "ok"
    assert model["entries"][0]["unjudged"] is False


def test_an_expectation_whose_run_missed_is_broken(workspace):
    record = _expect(workspace)
    _run("run-1", record["id"], 4.4, False)
    entry = models.ledger_model()["entries"][0]
    assert entry["state"] == "broken"
    assert entry["band"]["in_band"] is False


def test_an_unsettleable_deviation_is_flagged_unjudged(workspace):
    """`in_range` is None for the cases no program can settle. Those need a
    human, and §7's argument is that they otherwise accumulate quietly."""
    record = _expect(workspace)
    _run("run-1", record["id"], 3.05, None)
    entry = models.ledger_model()["entries"][0]
    assert entry["unjudged"] is True
    assert entry["state"] == "open"


def test_an_explicit_falsification_outranks_the_arithmetic(workspace):
    """A human's judgement beats the comparison, so it is checked first."""
    record = _expect(workspace)
    _run("run-1", record["id"], 3.05, True)
    ls.append_expectation_event(
        {"type": ls.T_EXPECTATION_FALSIFIED, "id": record["id"], "at": ls.now_iso(), "reason": "bad eval"}
    )
    assert models.ledger_model()["entries"][0]["state"] == "broken"


def test_an_empty_ledger_carries_the_command_that_ends_it(workspace):
    model = models.ledger_model()
    assert model["entries"] == []
    assert "tools.ledger expect" in model["empty_fix"]


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------
def test_an_uncollected_run_counts_at_its_estimate(workspace):
    """A job that has not been collected yet is not free."""
    _run("run-1", None, None, None)
    row = models.queue_model()["rows"][0]
    assert row["cost"] == "$4.00"


def test_the_status_the_ledger_actually_writes_reads_as_running(workspace):
    """`core/submit.py` writes exactly `in_flight`. An unrecognised status falls
    through to "waiting gate", so getting this wrong renders a fleet of running
    jobs as a queue waiting on a human who has nothing to approve."""
    _run("run-1", None, None, None)  # submitted with status "in_flight"
    row = models.queue_model()["rows"][0]
    assert row["state"] == "RUNNING"
    assert row["tone"] == "running"
    assert models.queue_model()["running"] == 1


def test_a_submit_failure_reads_as_failed(workspace):
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-1", "task": "t", "status": "submit_failed",
         "submitted_at": ls.now_iso(), "estimate_usd": 0.0, "estimated_duration_s": 1}
    )
    assert models.queue_model()["rows"][0]["tone"] == "failed"


def test_the_status_bar_and_the_queue_agree_about_what_is_outstanding(workspace):
    """Two counters that disagree about the same runs are worse than either."""
    _run("run-1", None, None, None)
    _run("run-2", None, None, None)
    rows = models.queue_model()["rows"]
    outstanding = len([r for r in rows if r["state"] != "DONE"])
    assert models.status_model()["queued"] == outstanding == 2


def test_a_collected_run_reads_done_and_costs_its_actual(workspace):
    record = _expect(workspace)
    _run("run-1", record["id"], 3.05, True)
    row = models.queue_model()["rows"][0]
    assert row["state"] == "DONE"
    assert row["cost"] == "$3.50"
    assert row["tone"] == "done"


def test_an_abandoned_run_does_not_read_as_done(workspace):
    """`ledger abandon` finalises the run -- that is how it stops holding the
    ceiling -- so it is collected in the fold's sense and would have fallen into
    the DONE branch. A run that never started, labelled DONE, sends whoever
    reads the queue looking for metrics that do not exist."""
    from core import submit as submit_lib

    _run("run-1", None, None, None)
    submit_lib.abandon("run-1", reason="the submitter was killed mid-submit")

    row = models.queue_model()["rows"][0]
    assert row["state"] == "ABANDONED"
    assert row["accent"] == "dashed"
    assert row["cost"] == "$0.00"


def test_a_failed_run_names_the_error_in_its_chip(workspace):
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-1", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 1.0, "estimated_duration_s": 10}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-1", "status": "failed", "collected_at": ls.now_iso(),
         "results": {}, "deviations": [], "error": {"type": "KeyError"}}
    )
    row = models.queue_model()["rows"][0]
    assert row["state"] == "FAILED · KeyError"
    assert row["accent"] == "broken"


def test_an_open_campaign_appears_in_the_queue(workspace):
    """Candidates spend the same GPU dollars against the same ceiling; a queue
    that showed only runs.jsonl would render a campaign as idle."""
    from core import campaign as campaign_mod

    campaign_mod.append_campaign(
        {"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": "open",
         "task_dir": "tasks/x", "at": ls.now_iso(), "generations_run": 3}
    )
    rows = models.queue_model()["rows"]
    assert any(r["kind"] == "campaign" and r["job"] == "camp-1" for r in rows)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def _preflight(workspace, checks, **extra):
    from core import jsonl

    record = {"submission_hash": "abc123", "spec": "specs/x.json",
              "verified_at": ls.now_iso(), "checks": checks, **extra}
    paths.preflight_dir().mkdir(parents=True, exist_ok=True)
    jsonl.write_json(paths.preflight_record("abc123"), record)
    return record


def test_proceed_is_disabled_while_anything_is_failing(workspace):
    _preflight(workspace, {
        "tests": {"ok": True, "duration_s": 3.0},
        "dry_run": {"ok": False, "duration_s": 1.0, "reason": "shape mismatch",
                    "fix": "python -m tools.preflight run --spec specs/x.json --json"},
    })
    model = models.preflight_model()
    assert model["blocking"] == 1
    assert model["can_proceed"] is False
    assert "preflight run" in model["remedy"]


def test_a_clean_checklist_can_proceed(workspace):
    _preflight(workspace, {"tests": {"ok": True, "duration_s": 3.0}})
    model = models.preflight_model()
    assert model["can_proceed"] is True
    assert model["blocking"] == 0


def test_a_check_that_never_ran_is_neither_passing_nor_blocking(workspace):
    _preflight(workspace, {"smoke": {"ok": None}})
    row = models.preflight_model()["current"]["rows"][0]
    assert row["state"] == "attention"
    assert models.preflight_model()["blocking"] == 0


def test_an_unreadable_record_is_reported_not_swallowed(workspace):
    """`jsonl.read_json` returns None for missing and malformed, but lets
    `UnicodeDecodeError` through -- it is a sibling of `JSONDecodeError` under
    `ValueError`, not a subclass. Left to escape, `Workspace.rebuild` catches it
    upstream and the window renders "No preflight records yet.", which is the
    one wrong answer: it says nothing is there exactly when something is there
    and cannot be read."""
    paths.preflight_dir().mkdir(parents=True, exist_ok=True)
    (paths.preflight_dir() / "bad.json").write_bytes(b"\xff\xfe{}")
    model = models.preflight_model()
    assert model["current"] is None
    assert "bad.json" in model["error"]
    assert "UnicodeDecodeError" in model["error"]


def test_one_unreadable_record_does_not_hide_the_readable_ones(workspace):
    _preflight(workspace, {"tests": {"ok": True, "duration_s": 1.0}})
    (paths.preflight_dir() / "bad.json").write_bytes(b"\xff\xfe{}")
    model = models.preflight_model()
    assert model["can_proceed"] is True
    assert model["error"]


def test_a_vanished_record_does_not_break_the_listing(workspace, monkeypatch):
    """`preflight run` writes atomically, so a path returned by `glob` can be
    gone by the time it is stat'd."""
    _preflight(workspace, {"tests": {"ok": True}})
    monkeypatch.setattr(
        models.Path, "stat", lambda self, **k: (_ for _ in ()).throw(FileNotFoundError(self))
    )
    assert isinstance(models.preflight_model(), dict)


def test_an_unreadable_wiki_manifest_is_not_reported_as_never_built(workspace):
    from core import budget as budget_mod
    from tools import projwiki as projwiki_tool

    budget_mod.create("proj", title="P", budget={})
    budget_mod.set_current("proj")
    projwiki_tool.output_dir("proj").mkdir(parents=True, exist_ok=True)
    (projwiki_tool.output_dir("proj") / "manifest.json").write_bytes(b"\xff\xfe{}")
    model = models.wiki_model()
    assert model["built"] is False
    assert "UnicodeDecodeError" in model["error"]


def test_the_wiki_window_says_so_when_no_project_is_selected(workspace):
    """A wiki is written *about* a project, so there is nothing to show without
    one -- and "no wiki built yet" would send the reader to a build command that
    would refuse for a different reason."""
    model = models.wiki_model()
    assert model["built"] is False
    assert model["project"] is None
    assert "budget use" in model["empty_fix"]


def test_hash_warnings_survive_into_the_window(workspace):
    """The known gaps in the submission hash: dynamic imports and runtime-loaded
    files. Shown, not swallowed."""
    _preflight(workspace, {"tests": {"ok": True}}, warnings=["dynamic import in train.py"])
    assert models.preflight_model()["current"]["warnings"] == ["dynamic import in train.py"]


# ---------------------------------------------------------------------------
# funnel
# ---------------------------------------------------------------------------
def _trace(workspace, name="q", **extra):
    directory = paths.notes_dir() / "funnel"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "does depth help at fixed compute?",
        "stages": {
            "0_expand": {"queries": ["scaling laws depth"], "hyde_words": 84},
            "1_retrieve": {"candidates": 400, "corpus_chunks": 12000},
            "2_rerank": {"out": 50},
            "3_triage": {"returned": 15},
        },
        "survivors": [{"id": "a", "title": "A", "rerank_score": 0.9, "reason": "states the ratio"}],
        "dropped": [{"id": "b", "title": "B", "reason": "different tokenizer"}],
        **extra,
    }
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_the_funnel_bars_are_the_counts_and_not_a_silhouette(workspace):
    """They were the literals 1.0, 0.82, 0.64 and 0.46: the same four bars whether
    a stage kept everything or nothing, in the window that exists to show which.

    Scaled to the 400 candidates the first stage retrieved rather than to the
    12,000 chunks in the index -- both are honest denominators and only one is
    legible, since against the corpus every bar below the first is a fraction of a
    pixel wide.
    """
    _trace(workspace)
    trace = models.funnel_model()["trace"]
    bars = trace["bars"]
    assert bars[0]["label"].startswith("CORPUS · 12000")
    # The index is the frame around the measurement, not a stage inside it.
    assert bars[0]["width"] is None
    assert [b["width"] for b in bars[1:]] == [1.0, 50 / 400, 15 / 400]
    assert [b["share"] for b in bars[1:]] == ["", "13%", "4%"]
    # A proportion whose base is unnamed is a decoration again.
    assert "400 candidates" in trace["scale"]


def test_a_stage_that_kept_nothing_draws_no_bar(workspace):
    """The screenshot that prompted this showed three stages reading `-> 0` at
    decorative width. Zero is the count this window is opened to look at."""
    _trace(workspace, stages={
        "1_retrieve": {"candidates": 400, "corpus_chunks": 12000},
        "2_rerank": {"out": 0},
        "3_triage": {"returned": 0},
    }, survivors=[])
    bars = models.funnel_model()["trace"]["bars"]
    assert bars[2]["width"] is None and bars[2]["tone"] == "empty"
    # And the last stage is the failure rather than a quiet grey row: nothing
    # reached the model.
    assert bars[3]["tone"] == "broken"


def test_retrieving_nothing_leaves_no_scale_to_draw_against(workspace):
    _trace(workspace, stages={"1_retrieve": {"candidates": 0}}, survivors=[])
    trace = models.funnel_model()["trace"]
    assert [b["width"] for b in trace["bars"]] == [None, None, None, None]
    assert "no scale" in trace["scale"]


def test_dropped_chunks_survive_with_their_reason(workspace):
    """A funnel that shows only survivors cannot answer the question you have
    when retrieval goes wrong."""
    _trace(workspace)
    dropped = models.funnel_model()["trace"]["dropped"]
    assert dropped[0]["reason"] == "different tokenizer"


def test_an_unknown_trace_name_falls_back_to_the_newest(workspace):
    _trace(workspace, name="a")
    _trace(workspace, name="b")
    assert models.funnel_model("nope")["trace"]["name"] == "b"


def test_a_corrupt_trace_reports_rather_than_raises(workspace):
    directory = paths.notes_dir() / "funnel"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bad.json").write_text("{not json", encoding="utf-8")
    model = models.funnel_model("bad")
    assert model["error"]
    assert model["trace"]["survivors"] == []


# ---------------------------------------------------------------------------
# papers
# ---------------------------------------------------------------------------
def test_a_papers_basis_matches_its_directory_despite_the_naming(workspace):
    """A basis cites "arXiv:2001.08361", the directory is `2001.08361`, the
    corpus stores `arxiv_2001.08361`. Comparing them raw yields a claim count of
    zero, which looks like working software."""
    (paths.papers_dir() / "2001.08361").mkdir(parents=True, exist_ok=True)
    _expect(workspace)
    row = models.papers_model(filter_name="all")["all"][0]
    assert len(row["claims"]) == 1
    assert any("CLAIMS DEPEND" in c["text"] for c in row["chips"])


def test_an_unread_paper_says_so_rather_than_being_hidden(workspace):
    (paths.papers_dir() / "2401.00001").mkdir(parents=True, exist_ok=True)
    model = models.papers_model(filter_name="queued")
    assert model["rows"][0]["read"] is False
    assert any("NOT READ" in c["text"] for c in model["rows"][0]["chips"])


@pytest.mark.parametrize(
    "raw,expected",
    [("arXiv:2001.08361", "2001.08361"), ("arxiv_2001.08361", "2001.08361"),
     ("2001.08361", "2001.08361"), ("", "")],
)
def test_paper_keys_normalise(raw, expected):
    assert models._paper_key(raw) == expected


def test_a_workspace_with_no_corpus_is_not_an_error(workspace):
    """A fresh install has no local index. Reporting that in the error strip
    would put a red card on the papers window of every new workspace, which
    teaches people to ignore the strip."""
    (paths.papers_dir() / "2001.08361").mkdir(parents=True, exist_ok=True)
    assert models.papers_model()["error"] is None


def test_an_unpressed_filter_lands_on_one_with_rows(workspace):
    """`cited` is zero until the first expectation is registered, so defaulting
    to it opened this window on "Nothing matches cited" with every paper in the
    workspace one unmotivated click away."""
    paper = paths.papers_dir() / "2001.08361"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "source.tex").write_text("x", encoding="utf-8")

    model = models.papers_model(filter_name=None)
    assert model["counts"] == {"cited": 0, "read": 1, "queued": 0}
    assert model["filter"] == "read"
    assert len(model["rows"]) == 1


def test_a_filter_the_user_pressed_is_left_alone_even_when_empty(workspace):
    """The other half, and the one that matters more: being moved to READ after
    pressing CITED is the window arguing with the click. The builder passes
    `None` rather than a default precisely so the two can be told apart."""
    paper = paths.papers_dir() / "2001.08361"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "source.tex").write_text("x", encoding="utf-8")

    model = models.papers_model(filter_name="cited")
    assert model["filter"] == "cited"
    assert model["rows"] == []


def test_the_window_asks_for_no_filter_until_one_is_chosen(workspace):
    """The state builder has to pass `None`, not `"cited"` -- a default there
    makes "nobody pressed a chip" and "someone pressed CITED" the same value,
    and the fallback then cannot fire for the case it exists for.

    Asserted as the concrete answer rather than "one of the three": with a read
    paper and nothing cited, `read` is the only correct one, and a membership
    check would have passed on the `cited` this is meant to catch.
    """
    from ui import state as state_mod

    paper = paths.papers_dir() / "2001.08361"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "source.tex").write_text("x", encoding="utf-8")

    class _Space:
        selection: dict = {}

    assert state_mod.MODEL_BUILDERS["papers"](_Space())["filter"] == "read"


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------
def _campaign(workspace, scores, status="open"):
    from core import campaign as campaign_mod

    campaign_mod.append_campaign(
        {"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": status,
         "task_dir": "tasks/x", "at": ls.now_iso(), "objective": "maximise acc"}
    )
    for index, score in enumerate(scores):
        campaign_mod.append_candidate(
            {"type": campaign_mod.T_CANDIDATE, "id": f"cand-{index}", "campaign": "camp-1",
             "generation": index, "metrics": {"combined_score": score},
             "cost_usd": 0.02, "at": ls.now_iso()}
        )


def test_the_score_is_read_where_the_campaign_ledger_writes_it(workspace):
    """`metrics.combined_score`, which is Shinka's contract and what `top_k`
    sorts on. Reading a top-level key does not raise -- it silently produces an
    empty lineage and no champion, which looks exactly like a campaign that has
    not evaluated anything yet."""
    _campaign(workspace, [0.41, 0.52, 0.71, 0.58])
    campaign = models.evolve_model()["campaign"]
    assert len(campaign["bars"]) == 4
    assert campaign["champion"]["id"] == "cand-2"
    assert campaign["champion_score"] == 0.71
    assert campaign["delta"] == pytest.approx(0.30)


def test_the_lineage_marks_new_bests_and_one_champion(workspace):
    _campaign(workspace, [0.41, 0.52, 0.71, 0.58])
    bars = models.evolve_model()["campaign"]["bars"]
    assert [b["tone"] for b in bars] == ["best", "best", "champion", "ordinary"]
    assert all(0 < b["height"] <= 1 for b in bars)


def test_the_worst_candidate_still_gets_a_visible_bar(workspace):
    """A zero-height rectangle reads as a missing generation, not a bad one."""
    _campaign(workspace, [0.1, 0.9])
    assert min(b["height"] for b in models.evolve_model()["campaign"]["bars"]) >= 0.08


def test_top_is_flattened_so_the_window_never_touches_metrics(workspace):
    _campaign(workspace, [0.41, 0.71])
    top = models.evolve_model()["campaign"]["top"]
    assert top[0]["score"] == 0.71
    assert set(top[0]) == {"id", "generation", "score"}


def test_a_requested_halt_is_visible_before_the_loop_reaches_a_boundary(workspace):
    """The request is in the ledger but the generation is still running. The
    window has to show that, or the button looks unpressed and gets pressed
    again."""
    from core import campaign as campaign_mod

    _campaign(workspace, [0.4])
    assert models.evolve_model()["campaign"]["halt_requested"] is False
    campaign_mod.request_halt("camp-1", reason="too slow")
    campaign = models.evolve_model()["campaign"]
    assert campaign["halt_requested"] is True
    assert campaign["running"] is True


def test_an_unevaluated_campaign_has_no_champion_rather_than_a_fake_one(workspace):
    from core import campaign as campaign_mod

    campaign_mod.append_campaign(
        {"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": "open",
         "task_dir": "tasks/x", "at": ls.now_iso()}
    )
    campaign = models.evolve_model()["campaign"]
    assert campaign["bars"] == []
    assert campaign["champion"] is None


# ---------------------------------------------------------------------------
# quota
# ---------------------------------------------------------------------------
def _quota(workspace, stage, credits, **extra):
    from core import jsonl, quota_log

    jsonl.append(
        paths.quota_path(),
        {"at": ls.now_iso(), "stage": stage, "role": extra.pop("role", "sonnet"),
         "input_tokens": 1000, "output_tokens": 500, "credits_usd": credits, **extra},
    )


def test_the_session_meter_splits_chat_from_tools(workspace):
    """"you are near the cap" and "your retrieval is what put you there" are
    different pieces of information."""
    from core import quota_log

    _quota(workspace, quota_log.STAGE_MAIN, 2.0)
    _quota(workspace, quota_log.STAGE_RERANK, 1.0)
    session = models.quota_model()["session"]
    assert session["chat_usd"] == 2.0
    assert session["tool_usd"] == 1.0
    assert abs(session["chat_fraction"] - 2 / 3) < 1e-9


def test_spend_outside_the_five_hour_window_does_not_count(workspace):
    from core import jsonl, quota_log

    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=9)).isoformat()
    jsonl.append(paths.quota_path(), {"at": old, "stage": quota_log.STAGE_MAIN,
                                      "input_tokens": 1, "output_tokens": 1, "credits_usd": 99.0})
    assert models.quota_model()["session"]["credits_usd"] == 0.0


def test_the_honesty_note_is_in_the_window_not_in_a_docstring(workspace):
    honesty = models.quota_model()["honesty"]
    assert "not the provider's" in honesty
    assert "fuel gauge" in honesty


def test_gpu_spend_separates_collected_from_in_flight(workspace):
    _run("run-1", None, None, None)
    gpu = models.quota_model()["gpu"]
    assert gpu["in_flight_usd"] == 4.0
    assert gpu["actual_usd"] == 0.0


# ---------------------------------------------------------------------------
# notebook verify state
# ---------------------------------------------------------------------------
def _notebook(workspace, name="x.ipynb"):
    paths.notebooks_dir().mkdir(parents=True, exist_ok=True)
    path = paths.notebooks_dir() / name
    path.write_text('{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}', encoding="utf-8")
    return path


def test_a_never_verified_notebook_is_not_citable(workspace):
    _notebook(workspace)
    state = models.verify_state("x.ipynb")
    assert state["citable"] is False
    assert state["chip"] == "NOT CITABLE"


def test_a_clean_verification_is_citable(workspace):
    _notebook(workspace)
    models.write_verify_record("x.ipynb", {"ok": True, "at": ls.now_iso(),
                                           "cells_executed": 12, "duration_s": 41.8})
    state = models.verify_state("x.ipynb")
    assert state["citable"] is True
    assert state["chip"] == "CITABLE"
    assert "12 cells" in state["sentence"]


def test_an_edit_after_the_verification_makes_it_stale(workspace):
    """Including an edit made inside Lab that the host never saw -- which is the
    whole point: Lab and tools/nb.py are two kernel owners over one notebook."""
    path = _notebook(workspace)
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    models.write_verify_record("x.ipynb", {"ok": True, "at": past, "cells_executed": 3})
    path.write_text('{"cells": [1], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}', encoding="utf-8")
    state = models.verify_state("x.ipynb")
    assert state["state"] == "stale"
    assert state["citable"] is False
    assert state["chip"] == "RE-VERIFY"


def test_a_failed_verification_carries_its_fix(workspace):
    _notebook(workspace)
    models.write_verify_record("x.ipynb", {"ok": False, "at": ls.now_iso(), "message": "NameError",
                                           "cell_index": 4, "fix": "pip install torch"})
    state = models.verify_state("x.ipynb")
    assert state["state"] == "failed"
    assert state["cell_index"] == 4
    assert state["fix"] == "pip install torch"


# ---------------------------------------------------------------------------
# editor
# ---------------------------------------------------------------------------
def _report(workspace, tex: str, claims: dict | None = None):
    from core import budget as budget_mod, report as report_mod

    budget_mod.create("proj", title="t", payer="me", budget={"gpu_usd": 10.0})
    budget_mod.set_current("proj")
    targets = report_mod.paths_for("proj")
    targets["dir"].mkdir(parents=True, exist_ok=True)
    targets["tex"].write_text(tex, encoding="utf-8")
    targets["claims"].write_text(json.dumps(claims or {}), encoding="utf-8")
    return targets


def test_an_unbound_number_blocks_the_build(workspace):
    """`\\gradnum{}` resolving through claims.json to a run *and its value* is
    strictly stronger than the mock's `\\gradcite{}`: it catches a citation that
    points at the right run and prints the wrong number."""
    _report(workspace, "\\section{Results}\nWe reach \\gradnum{loss}.\n")
    model = models.editor_model("proj")
    assert model["exists"] is True
    assert model["blocking"] >= 1
    assert any(f["rule"] == "claims" for f in model["findings"])


def test_the_outline_reads_the_sections(workspace):
    _report(workspace, "\\section{Setup}\ntext\n\\section{Results}\nmore\n")
    titles = [s["title"] for s in models.editor_model("proj")["outline"]]
    assert titles == ["Setup", "Results"]


def test_no_draft_yields_the_command_that_writes_one(workspace):
    from core import budget as budget_mod

    budget_mod.create("proj", title="t", payer="me", budget={"gpu_usd": 1.0})
    budget_mod.set_current("proj")
    model = models.editor_model("proj")
    assert model["exists"] is False
    assert "tools.report draft" in model["empty_fix"]


def test_gradnum_macros_are_their_own_token_class():
    spans = models.highlight_tex("We reach \\gradnum{loss} on \\textbf{eval}.  % note")
    kinds = [s["kind"] for s in spans]
    assert "gradnum" in kinds
    assert "command" in kinds
    assert "comment" in kinds
    assert "".join(s["text"] for s in spans) == "We reach \\gradnum{loss} on \\textbf{eval}.  % note"


# ---------------------------------------------------------------------------
# message anatomy
# ---------------------------------------------------------------------------
def test_a_fenced_shell_block_becomes_a_tool_card():
    blocks = models.parse_message("before\n```bash\npython -m tools.nb verify x\n```\nafter")
    kinds = [b["kind"] for b in blocks]
    assert kinds == ["text", "tool", "text"]
    assert blocks[1]["title"] == "python -m tools.nb verify x"


def test_a_fenced_python_block_stays_code_not_a_tool_call():
    blocks = models.parse_message("```python\nimport torch\n```")
    assert blocks[0]["kind"] == "code"
    assert blocks[0]["language"] == "python"


def test_an_expectation_header_becomes_a_card_with_its_rows():
    text = "EXPECTATION REGISTERED exp-7\nclaim: loss lands in band\nband: 2.9 – 3.2\n\ntrailing prose"
    blocks = models.parse_message(text)
    card = next(b for b in blocks if b["kind"] == "expectation")
    assert card["id"] == "exp-7"
    assert ("claim", "loss lands in band") in card["rows"]
    assert any(b["kind"] == "text" and "trailing prose" in b["text"] for b in blocks)


def test_a_gate_header_becomes_a_gate_card():
    blocks = models.parse_message("GATE — YOUR CALL\ncost: $18.40\nresource: 4x A100\n")
    assert blocks[0]["kind"] == "gate"
    assert ("cost", "$18.40") in blocks[0]["rows"]


def test_prose_with_no_structure_stays_one_block():
    blocks = models.parse_message("just a sentence about gates and expectations")
    assert [b["kind"] for b in blocks] == ["text"]


def test_only_figures_that_exist_are_offered(workspace):
    paths.figures_dir().mkdir(parents=True, exist_ok=True)
    (paths.figures_dir() / "loss.png").write_bytes(b"")
    found = models.figures_in("see figures/loss.png and figures/missing.png")
    assert len(found) == 1
    assert found[0].endswith("loss.png")


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------
def test_the_header_never_invents_an_agent_state(workspace):
    assert models.header_model(agent_state="dancing")["agent_state"] == "idle"
    assert models.header_model(agent_state="running")["accent"] == "ok"
    assert models.header_model(agent_state="awaiting_gate")["accent"] == "attention"


def test_every_agent_state_has_exactly_one_accent():
    """"one accent per state, never two in the same element" is only enforceable
    if the mapping is total."""
    assert set(models.AGENT_ACCENT) == set(models.AGENT_STATES)


def test_the_status_bar_reads_the_workspace(workspace):
    status = models.status_model()
    assert str(paths.root()) == status["cwd"]
    assert status["queued"] == 0


# ---------------------------------------------------------------------------
# failure containment
# ---------------------------------------------------------------------------
def test_a_damaged_ledger_line_degrades_to_an_error_not_a_crash(workspace):
    paths.expectations_path().parent.mkdir(parents=True, exist_ok=True)
    paths.expectations_path().write_text("{not json at all\n", encoding="utf-8")
    model = models.ledger_model()
    assert isinstance(model, dict)
    assert model["entries"] == []


# ---------------------------------------------------------------------------
# the projects window
# ---------------------------------------------------------------------------
def test_a_project_with_no_ceilings_is_reported_as_unbounded(workspace):
    """It passes every gate that reads a ceiling, silently. `tools/budget.py`
    says so once at creation and nothing carried it further; the window puts it
    on the row it is true of, for as long as it stays true."""
    from core import budget as budget_mod

    budget_mod.create("proj-open", title="no ceilings", budget={})
    budget_mod.create("proj-bound", title="bounded", budget={"gpu_usd": 10.0})

    rows = {r["id"]: r for r in models.projects_model()["rows"]}
    assert rows["proj-open"]["unbounded"] is True
    assert rows["proj-bound"]["unbounded"] is False
    assert models.projects_model()["unbounded"] == ["proj-open"]


def test_a_closed_project_is_not_counted_as_unbounded(workspace):
    """Nothing will be charged to it, so an UNBOUNDED chip on a closed project is
    a warning about a thing that cannot happen."""
    from core import budget as budget_mod

    budget_mod.create("proj-done", title="finished", budget={})
    budget_mod.close("proj-done")

    model = models.projects_model()
    assert model["unbounded"] == []
    assert model["open_count"] == 0
    assert model["rows"][0]["status"] == "closed"


def test_every_project_carries_its_own_ceilings(workspace):
    """The reason this window exists. The menu's raise controls addressed the
    selected project only, so reading what bounded any other one meant switching
    to it -- which reloads every window in the app to answer a question about a
    number."""
    from core import budget as budget_mod

    budget_mod.create("proj-a", title="A", budget={"gpu_usd": 10.0})
    budget_mod.create("proj-b", title="B", budget={"quota_tokens": 5_000_000})
    budget_mod.set_current("proj-a")

    rows = {r["id"]: r for r in models.projects_model()["rows"]}
    by_resource = {c["resource"]: c for c in rows["proj-b"]["ceilings"]}
    assert by_resource["quota_tokens"]["set"] is True
    assert by_resource["gpu_usd"]["set"] is False
    # Not the selected one, and its ceiling is readable anyway.
    assert rows["proj-b"]["current"] is False


def test_tokens_are_counted_and_dollars_are_priced(workspace):
    """4.2M subscription tokens rendered as `$4,200,000.00` is the specific
    thing the per-resource formatter separates."""
    from core import budget as budget_mod

    budget_mod.create("proj-a", title="A", budget={"quota_tokens": 4_200_000, "gpu_usd": 12.0})
    row = models.projects_model()["rows"][0]
    by_resource = {c["resource"]: c for c in row["ceilings"]}
    assert "4.2M" in by_resource["quota_tokens"]["label"]
    assert "$" not in by_resource["quota_tokens"]["label"]
    assert "$12.00" in by_resource["gpu_usd"]["label"]


def test_a_project_whose_spend_will_not_compute_does_not_take_the_list_down(workspace, monkeypatch):
    """`status` folds the whole run ledger for one project. It is caught per row
    so the broken one says so in its own row."""
    from core import budget as budget_mod

    budget_mod.create("proj-a", title="A", budget={})
    budget_mod.create("proj-b", title="B", budget={})

    real_status = budget_mod.status

    def explode(project_id):
        if project_id == "proj-a":
            raise RuntimeError("this ledger is a lie")
        return real_status(project_id)

    monkeypatch.setattr(budget_mod, "status", explode)

    rows = {r["id"]: r for r in models.projects_model()["rows"]}
    assert "this ledger is a lie" in rows["proj-a"]["error"]
    assert rows["proj-b"]["error"] is None


def test_scaffolded_but_empty_is_not_the_same_as_never_scaffolded(workspace):
    """`budget new` guards the scaffold step so it cannot fail the creation. This
    is where that consequence becomes visible instead of being discovered by
    `project sync` weeks later."""
    from core import budget as budget_mod, projects as projects_mod

    budget_mod.create("proj-a", title="A", budget={})
    assert models.projects_model()["rows"][0]["memory"]["scaffolded"] is False

    projects_mod.scaffold("proj-a")
    memory = models.projects_model()["rows"][0]["memory"]
    assert memory["scaffolded"] is True
    assert "MEMORY.md" in memory["present"]


# ---------------------------------------------------------------------------
# the setup window
# ---------------------------------------------------------------------------
def test_a_token_in_the_environment_is_ready_but_not_durable(workspace, monkeypatch):
    """The distinction that only bites the installed app: a shell that exported
    the token has it, and the desktop shortcut launches from Explorer with
    whatever was made persistent -- usually nothing."""
    from core import credentials

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-test")
    monkeypatch.setattr(credentials, "status", lambda: dict.fromkeys(credentials.ALL, False))

    token = models.setup_model()["token"]
    assert token["state"] == "environment"
    assert token["ready"] is True
    assert token["durable"] is False


def test_a_stored_token_is_durable(workspace, monkeypatch):
    from core import credentials

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        credentials,
        "status",
        lambda: {**dict.fromkeys(credentials.ALL, False), credentials.CLAUDE_TOKEN: True},
    )
    token = models.setup_model()["token"]
    assert token["state"] == "stored"
    assert token["durable"] is True


def test_with_nothing_configured_the_first_step_is_the_one_that_blocks(workspace, monkeypatch):
    from core import credentials

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(credentials, "status", lambda: dict.fromkeys(credentials.ALL, False))

    model = models.setup_model()
    steps = {s["id"]: s for s in model["steps"]}
    assert steps["token"]["ready"] is False
    assert steps["backends"]["ready"] is False
    # Neither of these can be "unanswered": there are defaults for all six roles,
    # and an optional key is optional.
    assert steps["models"]["ready"] is True
    assert steps["extras"]["ready"] is True
    assert model["complete"] is False


def test_the_models_step_reports_where_each_role_resolved_from(workspace):
    from core import settings

    settings.set_models({"evolve": "claude-opus-5"})
    roles = {r["role"]: r for r in models.setup_model()["roles"]}
    assert roles["evolve"]["model"] == "claude-opus-5"
    assert roles["evolve"]["source"] == "setup"
    assert roles["evolve"]["overridden"] is True
    assert roles["cite"]["source"] in ("config", "default")
    assert roles["cite"]["overridden"] is False


def test_setup_is_needed_only_when_the_agent_cannot_authenticate(workspace, monkeypatch):
    """Narrow on purpose. An unconfigured backend means no remote training,
    which is a real limitation and not a reason to put a wizard in front of
    someone who opened the app to read a ledger."""
    from core import credentials

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(credentials, "present", lambda _name: False)
    assert models.setup_needed() is True

    monkeypatch.setattr(credentials, "present", lambda _name: True)
    assert models.setup_needed() is False


def test_an_unreachable_credential_store_does_not_decide_the_app_is_broken(workspace, monkeypatch):
    """`setup_needed` runs on the startup path. It must answer, not raise."""
    from core import credentials
    from core.errors import ConfigError

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    def boom(_name):
        raise ConfigError("no keyring here", fix="pip install keyring")

    monkeypatch.setattr(credentials, "present", boom)
    assert models.setup_needed() is True  # nothing configured is the same answer


def test_a_backend_credential_is_not_reported_as_unconditionally_required(workspace, monkeypatch):
    """`hf_token` used to be marked required, so a user who had chosen Kaggle --
    the free backend -- saw a red MISSING for a token they will never need. What
    is true is that HF Jobs needs it, which is a fact about a backend."""
    from core import credentials

    monkeypatch.setattr(credentials, "status", lambda: dict.fromkeys(credentials.ALL, False))
    rows = {r["name"]: r for r in models.credentials_model()["rows"]}
    assert rows["hf_token"]["required"] is False
    assert rows["hf_token"]["group"] == "backend"
    assert rows["hf_token"]["tone"] != "broken"
    # The one that genuinely is.
    assert rows["claude_oauth_token"]["required"] is True
    assert rows["claude_oauth_token"]["tone"] == "broken"


def test_the_header_carries_the_folder_basename_and_its_whole_path(workspace):
    """The appbar cell cannot hold an absolute path, and the tooltip has to."""
    model = models.header_model()
    assert model["root"] == str(paths.root())
    assert model["root_name"] == paths.root().name


def test_every_model_survives_a_completely_empty_workspace(workspace):
    """Eleven windows over eight ledgers is eight chances per refresh for one
    bad file to take the workspace down. None of them may raise."""
    for builder in (
        models.setup_model,
        models.projects_model,
        models.ledger_model,
        models.quota_model,
        models.preflight_model,
        models.funnel_model,
        models.queue_model,
        models.evolve_model,
        models.papers_model,
        models.wiki_model,
        models.notebook_model,
        models.editor_model,
        models.status_model,
        models.header_model,
    ):
        assert isinstance(builder(), dict)
