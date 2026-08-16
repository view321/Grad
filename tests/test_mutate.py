"""The mutation operator's testable half (HANDOFF-2 §21).

Everything here runs without an SDK and without a model, which is deliberate:
the parts of `core/mutate.py` that decide whether a campaign is *safe* -- what
the operator is allowed to return, how a payload becomes a candidate, and what
the prompt actually contains -- are all pure, and the model call is a thin shell
around them.

The property this file exists to hold is the one that replaced §21's third
collision with a guarantee: **the operator never returns a file.** It returns the
contents of a mutable region, and `campaign.replace_blocks` splices it between
markers taken from the baseline, so code outside the region cannot change.
"""

from __future__ import annotations

from core import campaign as camp, evolution, mutate

BASELINE = """import json

# EVOLVE-BLOCK-START
def solve(x):
    return x * 2.0
# EVOLVE-BLOCK-END


def main():
    print(json.dumps({"ok": True}))
"""

TWO_BLOCKS = """import json

# EVOLVE-BLOCK-START
A = 1
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
B = 2
# EVOLVE-BLOCK-END
"""


# ---------------------------------------------------------------------------
# the structural guarantee
# ---------------------------------------------------------------------------
def test_the_outside_of_the_file_cannot_change():
    """The whole point. Whatever the operator returns goes *between* markers the
    baseline supplied, so the imports and the entry point are the baseline's by
    construction rather than by a check that runs afterwards."""
    payload = {"blocks": ["def solve(x):\n    import os\n    return os.getpid()"], "rationale": "r"}
    source, problems = mutate.apply_payload(
        payload, parent_full_source=BASELINE, patch_type=evolution.PATCH_FULL
    )
    assert not problems
    assert source.startswith("import json")
    assert 'print(json.dumps({"ok": True}))' in source
    assert camp.escaped_evolve_block(BASELINE, source)["escaped"] is False


def test_block_texts_and_replace_blocks_round_trip():
    assert camp.block_texts(BASELINE) == ["def solve(x):\n    return x * 2.0"]
    assert camp.replace_blocks(BASELINE, camp.block_texts(BASELINE)) == BASELINE


def test_several_regions_are_replaced_independently():
    assert camp.block_texts(TWO_BLOCKS) == ["A = 1", "B = 2"]
    out = camp.replace_blocks(TWO_BLOCKS, ["A = 9", "B = 2"])
    assert "A = 9" in out and "B = 2" in out
    assert out.count(camp.BLOCK_START) == 2


def test_a_wrong_number_of_replacements_is_refused():
    payload = {"blocks": ["A = 9"], "rationale": "r"}
    _, problems = mutate.apply_payload(
        payload, parent_full_source=TWO_BLOCKS, patch_type=evolution.PATCH_FULL
    )
    assert problems and "2 region" in problems[0]


# ---------------------------------------------------------------------------
# validation: what the tool handler refuses, so the model retries
# ---------------------------------------------------------------------------
def check(patch_type, args, blocks=1):
    return mutate._validator(patch_type, blocks)(args)


def test_marker_text_in_a_payload_is_refused_at_the_tool():
    """Caught here rather than downstream, and the difference is a whole
    candidate: rejected at the tool the model is told and writes it again;
    caught by the escape check it is recorded as an escape, never evaluated, and
    the generation is one proposal short."""
    problem = check(evolution.PATCH_FULL, {"blocks": [f"{camp.BLOCK_START}\nx = 1"], "rationale": "r"})
    assert problem and "marker" in problem


def test_a_rationale_is_required():
    """It is recorded beside the score and it is what makes a lineage readable
    afterwards."""
    assert check(evolution.PATCH_FULL, {"blocks": ["x = 1"], "rationale": "  "})
    assert check(evolution.PATCH_FULL, {"blocks": ["x = 1"], "rationale": "why"}) is None


def test_a_diff_needs_a_find_and_a_replace():
    assert check(evolution.PATCH_DIFF, {"edits": [], "rationale": "r"})
    assert check(evolution.PATCH_DIFF, {"edits": [{"find": "", "replace": "y"}], "rationale": "r"})
    assert check(
        evolution.PATCH_DIFF, {"edits": [{"find": "x", "replace": ""}], "rationale": "r"}
    ) is None


def test_a_diff_cannot_target_a_region_that_does_not_exist():
    problem = check(
        evolution.PATCH_DIFF, {"edits": [{"block": 4, "find": "x", "replace": "y"}], "rationale": "r"}
    )
    assert problem and "between 0 and 0" in problem


def test_the_wrong_number_of_blocks_is_refused_before_it_is_applied():
    assert check(evolution.PATCH_FULL, {"blocks": ["a", "b"], "rationale": "r"}, blocks=1)


# ---------------------------------------------------------------------------
# edits
# ---------------------------------------------------------------------------
def test_an_edit_must_match_exactly_once():
    """Not "at least once". A search string matching twice is a mutation whose
    effect depends on which occurrence the model meant, and applying it to both
    is a guess that silently edits code it was not looking at."""
    payload = {"edits": [{"find": "return x * 2.0", "replace": "return x * 3.0"}], "rationale": "r"}
    source, problems = mutate.apply_payload(
        payload, parent_full_source=BASELINE, patch_type=evolution.PATCH_DIFF
    )
    assert not problems
    assert "x * 3.0" in source

    twice = camp.replace_blocks(BASELINE, ["y = 1\ny = 1"])
    payload = {"edits": [{"find": "y = 1", "replace": "y = 2"}], "rationale": "r"}
    _, problems = mutate.apply_payload(
        payload, parent_full_source=twice, patch_type=evolution.PATCH_DIFF
    )
    assert problems and "2 times" in problems[0]


def test_an_edit_that_matched_nothing_says_why():
    """The ordinary LLM failure is whitespace reconstructed slightly
    differently, and a bare "0 matches" sends you looking in the wrong place."""
    payload = {"edits": [{"find": "return x*2.0", "replace": "z"}], "rationale": "r"}
    _, problems = mutate.apply_payload(
        payload, parent_full_source=BASELINE, patch_type=evolution.PATCH_DIFF
    )
    assert problems and "whitespace" in problems[0]


def test_a_failed_edit_produces_no_source_at_all():
    """Half-applied edits are worse than none: the candidate would be evaluated
    as something nobody wrote."""
    payload = {
        "edits": [
            {"find": "return x * 2.0", "replace": "return x * 3.0"},
            {"find": "nope", "replace": "z"},
        ],
        "rationale": "r",
    }
    source, problems = mutate.apply_payload(
        payload, parent_full_source=BASELINE, patch_type=evolution.PATCH_DIFF
    )
    assert source == ""
    assert problems


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------
def plan(**overrides):
    base = {
        "index": 0, "generation": 3, "island": 0, "patch_type": evolution.PATCH_DIFF,
        "parent": None, "mate": None, "elites": [], "failures": [],
    }
    base.update(overrides)
    return base


def build(**overrides):
    return mutate.build_prompt(
        plan(**overrides),
        baseline=BASELINE,
        parent_source=BASELINE,
        task_brief="Make it faster without changing the interface.",
        evaluator="print('score')",
        source_of=lambda c: c.get("_source"),
    )


def test_the_brief_and_the_evaluator_reach_the_operator():
    text = build()
    assert "Make it faster" in text
    assert "How it is scored" in text


def test_the_failures_section_is_in_the_prompt():
    """The single most common wasted generation is four candidates reproducing
    one crash the operator could not see. A change that quietly dropped this
    section would be invisible until that happened."""
    failure = {"candidate_id": "c9", "metrics": None, "error": "ImportError: no torch"}
    text = build(failures=[failure])
    assert "What has failed" in text
    assert "ImportError: no torch" in text


def test_generation_zero_says_it_is_mutating_the_baseline():
    assert "Generation 0" in build(generation=0, parent=None)


def test_the_parent_and_its_score_are_named():
    parent = {"candidate_id": "c3", "island": 1, "metrics": {"combined_score": -0.25}}
    text = build(parent=parent)
    assert "c3" in text and "-0.25" in text


def test_the_patch_type_decides_the_system_prompt():
    """Three operators with three different jobs. `diff` asks for exact
    find/replace pairs and `full` asks for a whole region, and giving one the
    other's instructions is a wasted candidate every time."""
    assert "find/replace" in mutate.system_prompt(evolution.PATCH_DIFF)
    assert "complete new contents" in mutate.system_prompt(evolution.PATCH_FULL)
    assert "TWO parents" in mutate.system_prompt(evolution.PATCH_CROSS)


def test_the_operator_is_told_not_to_tamper_with_the_measurement():
    """`combined_score` is a Goodhart machine and an LLM asked to raise a number
    will find the bug in the metric. The prompt is not the enforcement -- the
    verdict path is -- but it is free and it helps."""
    assert "Do not tamper with the measurement" in mutate.SYSTEM_PROMPT
