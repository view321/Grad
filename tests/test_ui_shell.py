"""The shell, rendered for real.

Everything else in `tests/test_ui_*.py` runs without NiceGUI, which is the point
of the layering. This file is the other half: it builds an actual element tree
for all eleven windows inside a real `Client`, so a typo in a window's render
path fails here rather than on someone's laptop at page-build time.

Two properties are worth holding still, and both are about `Element.move()`:

  * a window's root **survives** a retile -- otherwise every drag would wipe the
    chat transcript and the notebook's iframe anchor;
  * a closed window's root is **destroyed** -- otherwise the attic accumulates
    one detached subtree per window per session.

Skipped rather than failed when the `ui` extra is not installed: `core/` is
meant to run without it, and a test suite that cannot be run at all on a machine
without pywebview is a test suite that stops being run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui", reason="the ui extra is not installed")

from nicegui.client import Client  # noqa: E402
from nicegui.page import page  # noqa: E402

from ui import layout as layout_mod, registry, shell, state as state_mod  # noqa: E402


class FakeSession:
    """Enough of `ui.app.Session` to render. The SDK client is never started --
    `Session.start` only runs on the first `ask`, so a render touches nothing."""

    busy = False
    buffer = ""

    def __init__(self) -> None:
        self.settled: list[dict[str, str]] = []

    def interrupt(self) -> None:
        pass


@pytest.fixture
def rendered(workspace):
    """A built shell inside a real client, torn down afterwards."""
    clients: list[Client] = []

    def build(windows=None, project="proj"):
        client = Client(page("/"))
        clients.append(client)
        with client:
            space = state_mod.Workspace(FakeSession(), project)
            space.layout = layout_mod.Layout()
            for window in windows if windows is not None else registry.ids():
                space.layout.open(window)
            shell.build(space)
        return client, space

    yield build

    for client in clients:
        client.delete()


def html_of(client: Client) -> str:
    return " ".join(
        f"{element.tag} {' '.join(element.classes)} {getattr(element, 'content', '')}"
        for element in client.elements.values()
    )


# ---------------------------------------------------------------------------
# every window renders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", registry.ids())
def test_every_window_renders_on_an_empty_workspace(rendered, window):
    """The empty state is the state a new user sees, so it is the one most
    worth proving renders at all."""
    client, _ = rendered([window])
    assert len(client.elements) > 10


def test_all_eleven_render_together(rendered):
    client, space = rendered()
    assert len(space.layout.windows) == len(registry.ids())
    markup = html_of(client)
    assert "grad-shell" in markup
    assert "grad-tiles" in markup
    assert "grad-statusbar" in markup


def test_a_window_whose_render_raises_does_not_take_the_shell_down(rendered, monkeypatch):
    """Ten working windows and one broken one is a usable workspace; a traceback
    at page build time is not."""
    import ui.windows.funnel as funnel_window

    monkeypatch.setattr(
        funnel_window, "render", lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    client, _ = rendered(["funnel", "ledger"])
    assert "failed to render" in html_of(client)
    assert "grad-statusbar" in html_of(client)


# ---------------------------------------------------------------------------
# the chrome reflects the layout
# ---------------------------------------------------------------------------
def test_the_opener_marks_open_windows(rendered):
    client, space = rendered(["chat"])
    opener_cells = [
        e for e in client.elements.values() if "grad-opener-cell" in getattr(e, "classes", [])
    ]
    assert len(opener_cells) == len(registry.ids())
    assert len([c for c in opener_cells if "open" in c.classes]) == 1


def test_a_handle_sits_between_every_pair_of_columns(rendered):
    client, space = rendered(["chat", "ledger", "quota"])
    handles = [e for e in client.elements.values() if "grad-handle" in getattr(e, "classes", [])]
    columns = [e for e in client.elements.values() if "grad-column" in getattr(e, "classes", [])]
    assert len(columns) == 3
    assert len([h for h in handles if "row" not in h.classes]) == 2


def test_a_stacked_column_gets_a_row_handle(rendered):
    client, space = rendered(["chat", "notebook", "ledger", "quota"])
    handles = [e for e in client.elements.values() if "grad-handle" in getattr(e, "classes", [])]
    assert len([h for h in handles if "row" in h.classes]) == 1


def test_the_title_bar_tracks_the_model_not_just_the_layout(rendered):
    """The subtitle and the state chips are read from the model: `EVOLVING`
    becomes `HALTING`, a verify turns `NOT CITABLE` into `CITABLE`. Drawing them
    only when the panes are rebuilt leaves them stale until the next retile, and
    a chip that lags what it reports is worse than no chip."""
    from core import campaign as campaign_mod, ledger_store as ls

    campaign_mod.append_campaign(
        {"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": "open",
         "at": ls.now_iso(), "task_dir": "tasks/x"}
    )
    campaign_mod.append_candidate(
        {"type": campaign_mod.T_CANDIDATE, "id": "c0", "campaign": "camp-1", "generation": 0,
         "metrics": {"combined_score": 0.4}, "at": ls.now_iso()}
    )
    client, space = rendered(["evolve"])
    assert "EVOLVING" in html_of(client)

    campaign_mod.request_halt("camp-1", reason="from the workspace")
    space.tick()          # a poll, with no retile
    markup = html_of(client)
    assert "HALTING" in markup
    assert "EVOLVING" not in markup


def test_a_verify_flips_the_notebook_chip_without_a_retile(rendered):
    from core import ledger_store as ls
    from ui import models

    seed_everything()
    client, space = rendered(["notebook"])
    space.tick()
    assert "NOT CITABLE" in html_of(client)

    models.write_verify_record(
        "x.ipynb", {"ok": True, "at": ls.now_iso(), "cells_executed": 12, "duration_s": 4.0}
    )
    space.tick()
    markup = html_of(client)
    # "CITABLE" is a substring of "NOT CITABLE", so the negative is the
    # assertion that actually carries the test.
    assert "NOT CITABLE" not in markup
    assert "CITABLE" in markup


def test_the_focused_window_is_marked(rendered):
    client, space = rendered(["chat", "ledger"])
    space.focus("ledger")
    focused = [
        e for e in client.elements.values()
        if "grad-window" in getattr(e, "classes", []) and "focused" in e.classes
    ]
    assert len(focused) == 1


# ---------------------------------------------------------------------------
# roots survive retiling
# ---------------------------------------------------------------------------
def _root_ids(client: Client) -> dict[str, int]:
    """Window roots, by the id of the element carrying them."""
    out = {}
    for element in client.elements.values():
        if "grad-titlebar" in getattr(element, "classes", []):
            window = element.props.get("data-window")
            if window:
                out[window] = element.id
    return out


def test_a_window_root_is_reparented_not_rebuilt_on_retile(rendered):
    """`Element.move()` is what makes the window system practical: without it a
    drag would wipe the chat transcript and reload the Lab iframe."""
    client, space = rendered(["chat", "ledger"])
    body_before = {
        e.id for e in client.elements.values() if "grad-body" in getattr(e, "classes", [])
    }
    space.preset("stack")
    body_after = {
        e.id for e in client.elements.values() if "grad-body" in getattr(e, "classes", [])
    }
    assert body_before == body_after, "a retile rebuilt a window root"


def test_the_chat_transcript_survives_a_retile(rendered):
    client, space = rendered(["chat", "ledger"])
    transcripts = [
        e for e in client.elements.values() if "grad-transcript" in getattr(e, "classes", [])
    ]
    assert len(transcripts) == 1
    identity = transcripts[0].id
    space.preset("stack")
    space.preset("tile")
    space.retile("chat", 1)
    still = [e for e in client.elements.values() if "grad-transcript" in getattr(e, "classes", [])]
    assert [e.id for e in still] == [identity]


def test_closing_a_window_destroys_its_root(rendered):
    """Otherwise the attic accumulates a detached subtree per window per
    session, each one still bound to the poll."""
    client, space = rendered(["chat", "ledger"])
    before = len(client.elements)
    space.close("ledger")
    assert len(client.elements) < before
    assert "ledger" not in _root_ids(client)


def test_reopening_a_closed_window_builds_a_fresh_root(rendered):
    client, space = rendered(["chat", "ledger"])
    space.close("ledger")
    space.open("ledger")
    assert "ledger" in _root_ids(client)


# ---------------------------------------------------------------------------
# the whole lifecycle, in one pass
# ---------------------------------------------------------------------------
def test_the_full_gesture_sequence_leaves_a_consistent_tree(rendered):
    client, space = rendered()
    for action in (
        lambda: space.preset("stack"),
        lambda: space.preset("full"),
        lambda: space.preset("tile"),
        lambda: space.close("chat"),
        lambda: space.open("chat"),
        lambda: space.tick(),
        lambda: space.select("papers.filter", "queued"),
        lambda: space.select("funnel.trace", "nope"),
        lambda: space.retile("ledger", 0),
        lambda: space.resize("columns", [0.5] * len(space.layout.columns), total_px=1600),
        lambda: space.set_agent_state("running", step=14),
        lambda: space.say("verifying …"),
    ):
        action()

    windows = [e for e in client.elements.values() if "grad-window" in getattr(e, "classes", [])]
    assert len(windows) == len(space.layout.windows)
    assert "AGENT RUNNING · step 14" in html_of(client)
    assert "verifying" in html_of(client)


def test_a_tick_with_real_data_redraws_the_window(rendered):
    from core import ledger_store as ls

    client, space = rendered(["ledger"])
    ls.append_expectation(
        {"id": "exp-1", "task": "t", "created_at": ls.now_iso(), "quantity": "val_loss",
         "claim": "the loss lands between 2.9 and 3.2",
         "predicted": {"low": 2.9, "high": 3.2, "direction": None},
         "basis": [{"paper": "arXiv:1", "locator": "T3", "value": 3.0, "conditions": "1B"}],
         "comparability": "same eval", "confidence": "medium"}
    )
    space.tick()
    assert "the loss lands between 2.9 and 3.2" in html_of(client)


# ---------------------------------------------------------------------------
# populated: the paths an empty workspace never reaches
# ---------------------------------------------------------------------------
def seed_everything() -> None:
    """One of each thing the eleven windows read.

    The empty states are easy; the render paths that actually break are the ones
    behind a non-empty list -- a band strip, a lineage bar, a traceback, a
    reader rail. This seeds all of them.
    """
    import json

    from core import (
        budget as budget_mod, campaign as campaign_mod, jsonl,
        ledger_store as ls, paths, report as report_mod,
    )
    from tools import wiki as wiki_tool
    from ui import models

    ls.append_expectation(
        {"id": "exp-1", "task": "t", "created_at": ls.now_iso(), "quantity": "val_loss",
         "claim": "the loss lands in band", "predicted": {"low": 2.9, "high": 3.2, "direction": None},
         "basis": [{"paper": "arXiv:2001.08361", "locator": "T3", "value": 3.0, "conditions": "1B"}],
         "comparability": "same eval", "confidence": "medium"}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-1", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 4.0, "estimated_duration_s": 60}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-1", "status": "completed",
         "collected_at": ls.now_iso(), "cost_usd_actual": 3.5, "results": {"val_loss": 4.4},
         "deviations": [{"expectation_id": "exp-1", "quantity": "val_loss",
                         "expected": {"low": 2.9, "high": 3.2}, "actual": 4.4, "in_range": False}]}
    )

    budget_mod.create("proj", title="Scaling", payer="me", budget={"gpu_usd": 100.0})
    budget_mod.set_current("proj")
    targets = report_mod.paths_for("proj")
    targets["dir"].mkdir(parents=True, exist_ok=True)
    targets["tex"].write_text(
        "\\section{Setup}\nWe reach \\gradnum{loss} on eval. % note\n\\section{Results}\n",
        encoding="utf-8",
    )
    targets["claims"].write_text("{}", encoding="utf-8")

    for stage, role, credits in (("main", "opus", 2.0), ("funnel.rerank", "sonnet", 0.5)):
        jsonl.append(paths.quota_path(), {"at": ls.now_iso(), "stage": stage, "role": role,
                                          "input_tokens": 900, "output_tokens": 300,
                                          "credits_usd": credits, "project": "proj"})

    jsonl.write_json(paths.preflight_record("abc"), {
        "submission_hash": "abc", "spec": "specs/x.json", "verified_at": ls.now_iso(),
        "checks": {"tests": {"ok": True, "duration_s": 2.0},
                   "dry_run": {"ok": False, "duration_s": 1.0, "reason": "shape mismatch",
                               "output": "boom", "fix": "python -m tools.preflight run --spec specs/x.json"}},
        "warnings": ["dynamic import in train.py"]})

    funnel_dir = paths.notes_dir() / "funnel"
    funnel_dir.mkdir(parents=True, exist_ok=True)
    (funnel_dir / "q.json").write_text(json.dumps({
        "question": "does depth help at fixed compute?",
        "stages": {"0_expand": {"queries": ["depth scaling"], "hyde_words": 80},
                   "1_retrieve": {"candidates": 400, "corpus_chunks": 12000},
                   "2_rerank": {"out": 50}, "3_triage": {"returned": 1}},
        "survivors": [{"id": "a", "title": "A", "rerank_score": 0.9, "reason": "states the ratio"}],
        "dropped": [{"id": "b", "title": "B", "reason": "different tokenizer"}],
        "warnings": ["rerank fell back to lexical"]}), encoding="utf-8")

    paper = paths.papers_dir() / "2001.08361"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "meta.json").write_text(
        json.dumps({"title": "Scaling Laws", "authors": ["Kaplan"], "year": 2020}), encoding="utf-8"
    )

    campaign_mod.append_campaign({"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": "open",
                                  "task_dir": "tasks/x", "at": ls.now_iso(), "generations_run": 3,
                                  "project": "proj", "objective": "maximise accuracy", "islands": 4})
    for index, score in enumerate((0.1, 0.4, 0.3, 0.9)):
        campaign_mod.append_candidate({"type": campaign_mod.T_CANDIDATE, "id": f"cand-{index}",
                                       "campaign": "camp-1", "generation": index,
                                       "combined_score": score, "cost_usd": 0.02, "at": ls.now_iso()})

    paths.notebooks_dir().mkdir(parents=True, exist_ok=True)
    (paths.notebooks_dir() / "x.ipynb").write_text(
        '{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}', encoding="utf-8"
    )
    models.write_verify_record("x.ipynb", {"ok": False, "at": ls.now_iso(),
                                           "message": "NameError: torch is not defined",
                                           "cell_index": 4, "traceback": "Traceback ...",
                                           "fix": "pip install torch"})

    wiki_tool.output_dir().mkdir(parents=True, exist_ok=True)
    jsonl.write_json(wiki_tool.output_dir() / "manifest.json",
                     {"generated_at": ls.now_iso(), "output_dir": str(wiki_tool.output_dir()),
                      "source": {"hash": "stale00", "files": {"core/x.py": "aaa"}},
                      "scopes": {"core": 12, "tools": 9}})


def test_every_window_renders_with_real_data(rendered):
    seed_everything()
    client, space = rendered()
    space.select("papers.selected", "2001.08361", window="papers")
    space.select("ledger.filter", "broken")
    space.tick()
    markup = html_of(client)

    for expected in (
        "the loss lands in band",              # ledger, with a band strip
        "Scaling Laws",                        # papers, list and reader rail
        "does depth help at fixed compute?",   # funnel
        "different tokenizer",                 # funnel, dropped
        "dynamic import in train.py",          # preflight warnings
        "shape mismatch",                      # preflight failing row
        "NameError: torch is not defined",     # notebook failure detail
        "pip install torch",                   # notebook FIX box
        "camp-1",                              # evolve
        "gradnum",                             # editor source highlighting
        "different source tree",               # wiki staleness
        "run-1",                               # queue
    ):
        assert expected in markup, expected


def test_a_populated_workspace_survives_the_full_gesture_sequence(rendered):
    seed_everything()
    client, space = rendered()
    space.tick()
    for preset in ("stack", "full", "tile"):
        space.preset(preset)
    space.retile("evolve", 0)
    space.tick()
    windows = [e for e in client.elements.values() if "grad-window" in getattr(e, "classes", [])]
    assert len(windows) == len(space.layout.windows)
