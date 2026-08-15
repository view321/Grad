"""The context meter, the weighted quota ceiling, and compaction.

Three changes that arrived together because they are one argument. The ceiling
counted `input + output` and nothing else, which on the first fortnight of real
use was 149k tokens out of 12.5M actually moved; the missing 98% is cache reads;
cache reads are what a long context costs on every tool round-trip; and the only
lever on that is where the conversation gets compacted. So: measure all four
kinds, show how full the window is, and compact somewhere you chose.

The weighting tests use real records through `quota_log`, and the budget test
charges a real ceiling, for the reason `tests/test_budget.py` states -- a mock of
a gate proves nothing about the gate. The compaction tests are against the pure
half, which is deliberate: what is worth pinning down is *when* it fires and
*what survives*, and neither of those needs an SDK.
"""

from __future__ import annotations

import pytest

from core import budget, compaction, quota_log
from ui import models


# ---------------------------------------------------------------------------
# what a token counts as
# ---------------------------------------------------------------------------
def test_cache_reads_reach_the_ceiling_at_a_tenth_of_an_input_token():
    """The bug this whole change exists for, at its smallest.

    A turn that reads a large cached context and says little is the shape of
    every turn in a long session, and it used to register as almost nothing.
    """
    row = {
        "input_tokens": 100,
        "output_tokens": 100,
        "cache_read_tokens": 1_000_000,
        "cache_write_tokens": 0,
    }
    weights = {
        "weight_input": 1.0, "weight_output": 1.0,
        "weight_cache_read": 0.1, "weight_cache_write": 1.25,
    }
    assert quota_log.billable(row, weights) == pytest.approx(100_200.0)
    # What it did before: the same turn, counted as 200 tokens.
    assert row["input_tokens"] + row["output_tokens"] == 200


def test_the_weights_come_from_config_and_a_zero_restores_the_old_behaviour(workspace):
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "grad.toml").write_text(
        "[quota]\nweight_cache_read = 0.0\nweight_cache_write = 0.0\n", encoding="utf-8"
    )
    from core import config as config_mod

    config_mod._cache.clear()
    row = {"input_tokens": 5, "output_tokens": 7, "cache_read_tokens": 9_000, "cache_write_tokens": 900}
    assert quota_log.billable(row, quota_log.weights()) == pytest.approx(12.0)


@pytest.mark.parametrize("bad", ["-1.0", "'abc'", "nan"])
def test_an_unusable_weight_falls_back_rather_than_stranding_a_session(workspace, bad):
    """This is read on the path that decides whether a turn may be issued.

    A negative weight is refused specifically: it would make spending *lower*
    the measured total, which is the one error a ceiling cannot survive.
    """
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "grad.toml").write_text(
        f"[quota]\nweight_cache_read = {bad}\n", encoding="utf-8"
    )
    from core import config as config_mod

    config_mod._cache.clear()
    assert quota_log.weights()["weight_cache_read"] == quota_log.FALLBACK_WEIGHTS["weight_cache_read"]


def test_a_project_ceiling_is_charged_the_weighted_total(workspace):
    """Through `budget.status`, against a real ledger, not through the helper.

    The point is that the *gate* sees the cache traffic. Testing `billable` alone
    would have passed while `budget.py` went on adding two fields.
    """
    budget.create("proj-w", title="weighting", budget={"quota_tokens": 50_000})
    budget.set_current("proj-w")
    quota_log.record(
        quota_log.STAGE_MAIN,
        project="proj-w",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=1_000_000,
    )
    state = budget.status("proj-w")
    tokens = state["resources"]["quota_tokens"]
    assert tokens["spent"] == pytest.approx(100_020, rel=1e-6)
    assert tokens["over"] is True
    # And the four raw counts survive beside the one number, because a person
    # asking why they hit a ceiling needs them -- on `status`, which is the
    # payload every surface actually reads.
    assert state["quota_token_counts"]["cache_read_tokens"] == 1_000_000
    assert state["quota_weights"]["weight_cache_read"] == 0.1


def test_a_negative_count_cannot_buy_back_allocation(workspace):
    """`weights` already refuses a negative weight for this reason; the counts
    needed the same guard. `tools.quota record` refuses one at the CLI, but that
    is not the only door -- `from_sdk_usage` records what the SDK reports, and
    the ledger is a file on disk that can be edited."""
    budget.create("proj-neg", title="clamping", budget={"quota_tokens": 1_000})
    budget.set_current("proj-neg")
    quota_log.record(quota_log.STAGE_MAIN, project="proj-neg", output_tokens=900)
    quota_log.record(quota_log.STAGE_MAIN, project="proj-neg", output_tokens=-10_000)

    assert quota_log.counts({"output_tokens": -10_000})["output_tokens"] == 0
    assert budget.status("proj-neg")["resources"]["quota_tokens"]["spent"] == 900


def test_the_raw_counts_and_the_weighted_total_clamp_by_one_rule(workspace):
    """Read separately, a malformed row would be excluded from one and included
    in the other -- two numbers describing the same row and disagreeing, which is
    worse than either being wrong."""
    quota_log.record(quota_log.STAGE_MAIN, input_tokens=5, cache_read_tokens=-1_000)
    summary = quota_log.summarise()
    assert summary["totals"]["cache_read_tokens"] == 0
    assert summary["billable_tokens"] == 5


def test_the_weighted_total_is_rounded_once_rather_than_per_group(workspace):
    """Rounding each stage and then summing carries every group's error into the
    one number a ceiling is compared against."""
    # Three stages, each landing on exactly half a token once weighted.
    for stage in ("main", "funnel.expand", "funnel.triage"):
        quota_log.record(stage, cache_read_tokens=5)
    summary = quota_log.summarise()
    # 3 x 0.5 = 1.5 -> 2. Summing three separately-rounded 0.5s gives 0.
    assert summary["billable_tokens"] == 2
    assert sum(n["billable_tokens"] for n in summary["by_stage"].values()) == 0


def test_the_summary_reports_the_four_kinds_and_the_weighted_total(workspace):
    quota_log.record(quota_log.STAGE_MAIN, input_tokens=1, output_tokens=2,
                     cache_read_tokens=1_000, cache_write_tokens=100)
    summary = quota_log.summarise()
    assert summary["totals"] == {
        "input_tokens": 1, "output_tokens": 2,
        "cache_read_tokens": 1_000, "cache_write_tokens": 100,
    }
    assert summary["billable_tokens"] == round(1 + 2 + 100 + 125)
    # `total_tokens` keeps its old meaning; nothing that read it starts lying.
    assert summary["total_tokens"] == 3


# ---------------------------------------------------------------------------
# the context meter
# ---------------------------------------------------------------------------
def test_an_unknown_context_reads_as_unknown_rather_than_as_empty():
    """Zero and "no reading yet" look identical at a glance and only one of them
    is worth acting on."""
    model = models.context_model(None)
    assert model["known"] is False
    assert model["label"] == "ctx —"
    assert model["tone"] == ""


@pytest.mark.parametrize(
    "usage",
    [
        {"maxTokens": 1_000_000},                      # a reading with no total in it
        {"totalTokens": None, "maxTokens": 1_000_000},
        {"totalTokens": "lots", "maxTokens": 1_000_000},
    ],
)
def test_an_unreadable_total_is_unknown_rather_than_a_confident_zero(usage):
    """The failure this function's docstring is about, arriving through the
    function itself. "ctx 0 · 0%" for a session that could not be measured is
    worse than saying nothing, because it invites exactly the conclusion that
    there is plenty of room."""
    model = models.context_model(usage, compact_at=300_000)
    assert model["known"] is False
    assert model["label"] == "ctx —"


def test_a_category_that_cannot_be_read_is_skipped_rather_than_raising():
    """This is drawn from a timer, so one odd category would not produce one bad
    tooltip -- it would raise several times a second for the life of the
    session."""
    model = models.context_model(
        {
            "totalTokens": 5_000,
            "maxTokens": 1_000_000,
            "categories": [
                {"name": "Tools", "tokens": "unknown"},
                {"name": "Skills", "tokens": 1_469},
                "not even a dict",
            ],
        }
    )
    assert [c["name"] for c in model["categories"]] == ["Skills"]


def test_a_genuinely_empty_context_still_reads_as_known():
    """Zero is a real reading when the payload says so; only a missing or
    unparseable one is unknown."""
    model = models.context_model({"totalTokens": 0, "maxTokens": 1_000_000})
    assert model["known"] is True
    assert model["fraction"] == 0.0


def test_the_meter_measures_against_grads_threshold_when_there_is_one():
    """The whole point of the chip. Against the CLI's 967k the same session
    reads as nearly empty; against the threshold that will actually fire it
    reads as two thirds gone."""
    usage = {"totalTokens": 200_000, "maxTokens": 967_000, "categories": []}
    cli = models.context_model(usage)
    grad = models.context_model(usage, compact_at=300_000)
    assert cli["limit_source"] == "cli"
    assert cli["fraction"] == pytest.approx(200_000 / 967_000)
    assert grad["limit_source"] == "grad"
    assert grad["fraction"] == pytest.approx(2 / 3)


def test_a_threshold_above_the_models_window_does_not_win():
    """A threshold the conversation can never reach is not the binding one, and
    drawing against it would show a meter that never fills while the CLI
    compacts underneath."""
    usage = {"totalTokens": 500_000, "maxTokens": 967_000}
    model = models.context_model(usage, compact_at=2_000_000)
    assert model["limit_source"] == "cli"
    assert model["limit"] == 967_000


def test_the_chip_changes_tone_before_compaction_rather_than_after():
    near = models.context_model({"totalTokens": 295_000, "maxTokens": 1_000_000}, compact_at=300_000)
    warn = models.context_model({"totalTokens": 240_000, "maxTokens": 1_000_000}, compact_at=300_000)
    calm = models.context_model({"totalTokens": 10_000, "maxTokens": 1_000_000}, compact_at=300_000)
    assert near["tone"] == "attention"
    assert warn["tone"] == "warn"
    assert calm["tone"] == ""


def test_free_space_is_not_listed_as_something_using_the_context():
    """It is the complement of everything else: always the largest entry in the
    CLI's own breakdown, and never a consumer."""
    model = models.context_model(
        {
            "totalTokens": 1_469,
            "maxTokens": 1_000_000,
            "categories": [
                {"name": "Free space", "tokens": 998_531},
                {"name": "Skills", "tokens": 1_469},
            ],
        }
    )
    assert [c["name"] for c in model["categories"]] == ["Skills"]


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------
def test_compaction_is_off_unless_a_threshold_is_configured(workspace):
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "grad.toml").write_text(
        "[agent]\ncompact_at_tokens = 0\n", encoding="utf-8"
    )
    from core import config as config_mod

    config_mod._cache.clear()
    cfg = config_mod.load()
    assert compaction.threshold(cfg) == 0
    assert compaction.should_compact({"totalTokens": 10_000_000}, cfg) is False


@pytest.mark.parametrize("bad", ["-5", "nan", "'soon'"])
def test_an_unusable_threshold_disables_compaction_rather_than_firing_every_turn(workspace, bad):
    """A negative threshold is always already exceeded, which would compact
    after every single turn -- the most expensive possible reading of a typo."""
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "grad.toml").write_text(
        f"[agent]\ncompact_at_tokens = {bad}\n", encoding="utf-8"
    )
    from core import config as config_mod

    config_mod._cache.clear()
    assert compaction.threshold(config_mod.load()) == 0


def test_an_unreadable_context_never_triggers_a_compaction(workspace):
    """A missing measurement is not evidence of a large context, and compacting
    on the strength of one would discard a conversation for no reason."""
    cfg = _cfg(workspace, "[agent]\ncompact_at_tokens = 100\n")
    assert compaction.should_compact(None, cfg) is False
    assert compaction.should_compact({}, cfg) is False
    assert compaction.should_compact({"totalTokens": "lots"}, cfg) is False
    assert compaction.should_compact({"totalTokens": 101}, cfg) is True


def test_the_seed_says_it_is_a_reconstruction_and_carries_the_note():
    seed = compaction.seed_message("I was halfway through run-3.", tokens_before=412_000)
    assert "412,000" in seed
    assert "reconstruction" in seed
    assert "I was halfway through run-3." in seed


def test_an_empty_summary_seeds_a_warning_rather_than_a_blank_handover():
    """A blank handover looks to the model like a conversation that genuinely
    had nothing in it, which is the one reading that must not happen."""
    seed = compaction.seed_message("   ")
    assert "came back empty" in seed
    assert "re-read" in seed


def test_the_handoff_prompt_asks_for_the_ledger_state_a_generic_summary_drops():
    """The Grad-specific half. An expectation registered and not yet judged, or a
    run submitted and not yet collected, is state the next turn is expected to
    act on -- and losing it does not read as a bad summary, it reads as an agent
    that abandoned a run halfway."""
    prompt = compaction.HANDOFF_PROMPT
    for owed in ("expectation", "collected", "verdict", "gate"):
        assert owed in prompt


def test_a_compaction_leaves_a_record_the_chat_window_can_draw():
    from ui.app import ROLES

    record = compaction.record(
        tokens_before=310_000,
        tokens_after=0,
        note="what I was doing",
        cost={"output_tokens": 900, "cache_read_tokens": 300_000},
    )
    # `restore` keeps records whose role it knows and drops the rest, so a
    # marker with an unknown role would vanish on the next reload -- leaving a
    # transcript that reads as one continuous conversation beside a model that
    # remembers only the tail of it.
    assert record["role"] in ROLES
    assert record["kind"] == "compaction"
    assert "310,000" in record["text"]
    assert record["cost_tokens"]["cache_read_tokens"] == 300_000


def test_the_compaction_turn_is_charged_to_its_own_stage(workspace):
    """ml-intern's context manager carries a comment saying that not metering
    this "used to hide a significant share of hosted inference spend". The same
    hole was available here for free, and folding it into `main` would also hide
    the number that says whether the threshold is set right."""
    quota_log.record(quota_log.STAGE_COMPACT, output_tokens=1_200, cache_read_tokens=250_000)
    summary = quota_log.summarise()
    assert quota_log.STAGE_COMPACT in summary["by_stage"]
    assert quota_log.STAGE_COMPACT != quota_log.STAGE_MAIN
    assert summary["by_stage"][quota_log.STAGE_COMPACT]["billable_tokens"] == round(1_200 + 25_000)


def _cfg(workspace, text: str):
    (workspace / "config").mkdir(exist_ok=True)
    (workspace / "config" / "grad.toml").write_text(text, encoding="utf-8")
    from core import config as config_mod

    config_mod._cache.clear()
    return config_mod.load()
