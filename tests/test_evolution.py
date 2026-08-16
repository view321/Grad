"""The search policy (HANDOFF-2 §21).

`core/evolution.py` is pure, which is what makes this file possible and what
makes it worth having. The failures it guards against are all *silent*: a search
that has collapsed onto one lineage still produces candidates, still fills the
ledger, and still reports a best score. Nothing about it looks wrong except that
it stops finding anything.

So the assertions here are about the properties rather than the outputs -- that
selection is not greedy, that islands stay apart, that the bandit spreads a
batch, that a duplicate is caught -- and every one of them is a thing that would
otherwise be noticed weeks later, if at all.
"""

from __future__ import annotations

import random

from core import evolution as ev


def candidate(cid, score, *, island=0, patch="full", **extra):
    return {
        "candidate_id": cid,
        "island": island,
        "patch_type": patch,
        "metrics": None if score is None else {"combined_score": score},
        **extra,
    }


POPULATION = [
    candidate("a", -1.0, island=0, patch="diff"),
    candidate("b", -2.0, island=1, patch="full"),
    candidate("c", -3.0, island=0, patch="full"),
    candidate("d", None, island=1, patch="diff", error="ImportError: no module named torch"),
]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_a_candidate_with_no_score_is_not_a_candidate_that_scored_zero():
    """The failure this prevents is a search that learns to crash.

    Every task in the scaffold reports a *negated error*, so real scores are
    negative -- and scoring a crash at 0.0 would make "fail immediately" the
    highest-scoring strategy available.
    """
    assert ev.score_of(candidate("x", None)) is None
    assert ev.score_of(candidate("x", 0.0)) == 0.0
    assert ev.score_of({"metrics": {"combined_score": True}}) is None
    assert ev.score_of({"metrics": {"combined_score": float("nan")}}) is None


def test_scored_orders_best_first_and_drops_the_unscored():
    assert [c["candidate_id"] for c in ev.scored(POPULATION)] == ["a", "b", "c"]


def test_reward_is_a_rank_so_it_needs_no_per_task_calibration():
    """`combined_score` is a negated error on one task and an accuracy on the
    next, so a bandit reward taken from its value would need tuning per task."""
    best = ev.reward_for(POPULATION[0], POPULATION)
    worst = ev.reward_for(POPULATION[2], POPULATION)
    assert best == 1.0 and worst == 0.0
    # An unscored candidate rewards nothing: the arm proposed something that did
    # not run.
    assert ev.reward_for(POPULATION[3], POPULATION) == 0.0


# ---------------------------------------------------------------------------
# islands
# ---------------------------------------------------------------------------
def test_islands_are_assigned_round_robin_so_none_is_left_empty():
    """Random assignment leaves an island empty about a third of the time with a
    population of four, and an empty island has no parent to breed from for the
    rest of the campaign."""
    assert [ev.assign_island(i, islands=3) for i in range(6)] == [0, 1, 2, 0, 1, 2]
    assert ev.assign_island(5, islands=1) == 0


def test_migration_copies_a_champion_rather_than_moving_it():
    """Moving it would empty the island it came from of its best member, which
    is the opposite of what migration is for."""
    moves = ev.migrate(POPULATION, islands=2)
    assert ("a", 0, 1) in moves
    migrated = [dict(c, migrated_to=[1]) if c["candidate_id"] == "a" else c for c in POPULATION]
    assert "a" in [c["candidate_id"] for c in ev.members(migrated, 1)]
    assert "a" in [c["candidate_id"] for c in ev.members(migrated, 0)]


def test_a_champion_is_not_migrated_to_an_island_it_is_already_on():
    """Otherwise a strong candidate writes one redundant event per interval for
    the rest of the campaign."""
    migrated = [dict(c, migrated_to=[1]) if c["candidate_id"] == "a" else c for c in POPULATION]
    assert all(cid != "a" for cid, _, to in ev.migrate(migrated, islands=2))


def test_migration_happens_on_the_interval_and_never_at_generation_zero():
    assert ev.should_migrate(0, interval=3) is False
    assert ev.should_migrate(3, interval=3) is True
    assert ev.should_migrate(4, interval=3) is False
    # Zero is how islands are switched off without a second flag.
    assert ev.should_migrate(9, interval=0) is False


def test_provenance_and_eligibility_are_different_questions():
    """`island_of` is where it came from; `islands_of` is where it may breed.
    Conflating them breaks migration in a way nothing reports."""
    migrant = candidate("m", -1.0, island=0, migrated_to=[1, 2])
    assert ev.island_of(migrant) == 0
    assert ev.islands_of(migrant) == [0, 1, 2]


# ---------------------------------------------------------------------------
# parents
# ---------------------------------------------------------------------------
def test_selection_is_not_greedy():
    """Always taking the argmax collapses the population onto one lineage;
    taking uniformly is a random walk. Rank weighting is neither."""
    rng = random.Random(0)
    picked = {
        ev.select_parent(POPULATION, island=0, rng=rng)["candidate_id"] for _ in range(200)
    }
    assert picked == {"a", "c"}, "island 0's whole scored population should be reachable"


def test_selection_prefers_the_better_parent():
    rng = random.Random(0)
    counts = {"a": 0, "c": 0}
    for _ in range(400):
        counts[ev.select_parent(POPULATION, island=0, rng=rng)["candidate_id"]] += 1
    assert counts["a"] > counts["c"] * 1.5


def test_zero_pressure_is_uniform_selection():
    weights = ev.rank_weights(4, pressure=0.0)
    assert weights == [1.0, 1.0, 1.0, 1.0]


def test_an_empty_island_falls_back_to_the_whole_population():
    """Islands only mean anything once something has been scored on each, and
    refusing to breed until then would stall the search at generation 0."""
    rng = random.Random(0)
    assert ev.select_parent(POPULATION, island=7, rng=rng) is not None
    assert ev.select_parent([], island=0, rng=rng) is None


def test_a_mate_comes_from_another_island():
    """Crossing two members of one island recombines the least: they share
    ancestry by construction."""
    rng = random.Random(0)
    parent = POPULATION[0]  # island 0
    for _ in range(50):
        mate = ev.select_mate(POPULATION, parent, rng=rng)
        assert ev.island_of(mate) != ev.island_of(parent)


def test_inspirations_carry_the_failures_as_well_as_the_elites():
    """The single most common wasted generation is four candidates reproducing
    one crash the operator could not see."""
    shown = ev.inspirations(POPULATION, parent=POPULATION[0])
    assert [c["candidate_id"] for c in shown["elites"]] == ["b", "c"]
    assert [c["candidate_id"] for c in shown["failures"]] == ["d"]
    # The parent is excluded: it is already in the prompt as the thing being
    # mutated, and listing it twice spends tokens to say one thing.
    assert all(c["candidate_id"] != "a" for c in shown["elites"])


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------
def test_reformatting_is_not_a_new_candidate():
    a = "def f():\n    return 1\n"
    b = "\n\ndef f():   \n    return 1\n\n"
    assert ev.source_key(a) == ev.source_key(b)


def test_indentation_is_the_program_so_it_is_not_normalised_away():
    assert ev.source_key("if x:\n    f()") != ev.source_key("if x:\n        f()")


def test_dedupe_removes_repeats_of_each_other_and_of_what_was_seen():
    seen = {ev.source_key("a = 1")}
    assert ev.dedupe(["a = 1", "b = 2", "b = 2", "c = 3"], seen=seen) == ["b = 2", "c = 3"]


def test_seen_sources_includes_failures():
    """A proposal identical to one that crashed is not worth a second evaluation
    -- and the operator shown that failure is the one most likely to repeat it."""
    rows = [{"source_key": "k1"}, {"source_key": "k2", "error": "boom"}, {"error": "no source"}]
    assert ev.seen_sources(rows) == {"k1", "k2"}


# ---------------------------------------------------------------------------
# the bandit
# ---------------------------------------------------------------------------
def test_cross_is_unavailable_until_two_parents_have_been_scored():
    """Offering it earlier spends the exploration budget on an arm that degrades
    to `full` every time, and then learns -- correctly and uselessly -- that
    `cross` is no better than `full`."""
    assert ev.arms_available([]) == [ev.PATCH_DIFF, ev.PATCH_FULL]
    assert ev.arms_available(POPULATION[:1]) == [ev.PATCH_DIFF, ev.PATCH_FULL]
    assert ev.PATCH_CROSS in ev.arms_available(POPULATION)


def test_the_bandit_spreads_a_batch_rather_than_giving_every_slot_one_arm():
    """The bug this is about: a generation is planned in one pass against one set
    of statistics, so without `pending` every slot picks the same arm and a
    population of four became four `cross` mutations."""
    stats = ev.bandit_from(POPULATION)
    plans = ev.plan_generation(
        POPULATION, generation=2, population=6, islands=2,
        stats=stats, rng=random.Random(3),
    )
    assert len({p["patch_type"] for p in plans}) > 1


def test_a_pending_pull_does_not_move_an_arms_mean():
    """It says "this arm is already being tried", not "this arm scored zero"."""
    stats = ev.bandit_update(ev.empty_bandit(), ev.PATCH_DIFF, 1.0)
    stats = ev.bandit_update(stats, ev.PATCH_FULL, 1.0)
    stats = ev.bandit_update(stats, ev.PATCH_CROSS, 0.0)
    rng = random.Random(0)
    # `diff` and `full` are tied and both beat `cross`; a pile of pending pulls
    # on `diff` should push the next draw elsewhere without making `diff` look
    # like it performed badly.
    assert ev.bandit_select(stats, list(ev.PATCH_TYPES), rng, pending={ev.PATCH_DIFF: 20}) != ev.PATCH_DIFF
    assert ev.bandit_report(stats)[0] == {"patch_type": "diff", "pulls": 1, "mean_reward": 1.0}


def test_the_bandit_rebuilds_from_the_records():
    """Derived rather than carried, so a crash cannot lose it and a reader can
    check it."""
    stats = ev.bandit_from(POPULATION)
    pulls = {row["patch_type"]: row["pulls"] for row in ev.bandit_report(stats)}
    assert pulls == {"diff": 2, "full": 2, "cross": 0}


def test_unpulled_arms_are_tried_in_a_random_order():
    """Fixed order means the first arm always gets generation 0's first
    candidate, which is a head start rather than an exploration."""
    empty = ev.empty_bandit()
    drawn = {ev.bandit_select(empty, list(ev.PATCH_TYPES), random.Random(s)) for s in range(30)}
    assert len(drawn) > 1


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def test_generation_zero_mutates_the_baseline_with_no_parent():
    plans = ev.plan_generation(
        [], generation=0, population=3, islands=2,
        stats=ev.empty_bandit(), rng=random.Random(0),
    )
    assert all(p["parent"] is None and p["patch_type"] == ev.PATCH_FULL for p in plans)


def test_a_plan_is_deterministic_in_its_seed():
    """The whole policy is a pure function of the ledger and the seed, which is
    what makes a campaign replayable -- and what lets the proposals be issued
    concurrently, since the plan does not depend on what it is planning."""
    # A population big enough for the draw to have somewhere to go. With three
    # scored candidates the rank weights make one parent so dominant that two
    # seeds agree by luck, which would make this pass for the wrong reason.
    big = [
        candidate(f"c{i}", -float(i), island=i % 2, patch=ev.PATCH_TYPES[i % 3])
        for i in range(12)
    ]

    def plan(seed):
        return [
            (p["island"], p["patch_type"], (p["parent"] or {}).get("candidate_id"))
            for p in ev.plan_generation(
                big, generation=4, population=6, islands=2,
                stats=ev.bandit_from(big), rng=random.Random(seed),
            )
        ]

    assert plan(99) == plan(99)
    assert any(plan(99) != plan(seed) for seed in range(100, 110))


def test_a_cross_with_no_mate_degrades_rather_than_mislabelling_itself():
    """One island, so there is nobody to cross with. Recording it as `cross`
    would be a `full` wearing the wrong label in the bandit's statistics."""
    one_island = [candidate("a", -1.0), candidate("b", -2.0)]
    plans = ev.plan_generation(
        one_island, generation=1, population=4, islands=1,
        stats=ev.bandit_from(one_island), rng=random.Random(1),
    )
    for p in plans:
        assert not (p["patch_type"] == ev.PATCH_CROSS and p["mate"] is None)
