"""The search policy: islands, archive, parents, novelty, and the patch bandit.

Pure functions over the candidate records `core/campaign.py` already stores. No
I/O, no model, no clock -- which is the point. This is the half of an
evolutionary algorithm that decides *what to try next*, and it is the half that
is worth testing, because its failures are silent: a search that has quietly
collapsed onto one lineage still produces candidates, still fills the ledger,
and still reports a best score. It just stops finding anything.

Everything randomised takes a `random.Random` explicitly rather than reaching
for the module-level one, so a campaign can be replayed from its seed and a test
can assert on an exact draw. The seed is recorded on the campaign record.

**Why this exists rather than ShinkaEvolve's version of it.** Grad needs the
budget gate between generations, and Shinka's async runner keeps N proposals in
flight and completes them out of order -- there is no generation boundary in it
to gate at. Its selection policy is the part worth keeping, and a selection
policy is data-structure work over records this project already writes. What was
*not* worth adopting was a second orchestrator: one that spends subscription
tokens through `npx @roberttlange/headless`, which `agent.drive_turn` never
issues and `ledger/quota.jsonl` therefore never sees. A loop designed to run
without a human in it is the last place to accept an unmetered model call.

The four policies, and what each is protecting against:

* **Islands** -- sub-populations that only occasionally exchange members. Without
  them a small population converges on the first decent basin it finds by about
  generation three, and every later mutation is a variation on one idea.
* **Rank-weighted parents** -- selection pressure without collapse. Always taking
  the argmax is the same failure as one island; taking uniformly is a random
  walk.
* **Novelty dedup** -- an LLM asked twice for a mutation frequently proposes the
  same one, and the evaluation is the expensive part. Two identical candidates
  are one candidate and one wasted GPU hour.
* **The patch bandit** -- `diff`, `full` and `cross` have very different costs
  and hit rates, and which one is working is task-dependent. UCB1 over three arms
  is the smallest thing that notices.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Iterable, Sequence

#: How a candidate is scored. Higher is better, matching Shinka's contract and
#: `campaign.top_k`'s sort. A task that wants to minimise something negates it in
#: `evaluate.py`, which keeps one convention rather than a direction flag that
#: every reader has to check.
METRIC = "combined_score"

#: The three patch types, which are also the bandit's arms.
#:
#: `diff` edits the block in place and is cheap and local; `full` rewrites it and
#: is the only one that can restructure; `cross` needs two parents and is what
#: recombines two partial ideas. A campaign with one parent cannot draw `cross`,
#: so `arms_available` narrows the set rather than the caller special-casing it.
PATCH_DIFF = "diff"
PATCH_FULL = "full"
PATCH_CROSS = "cross"
PATCH_TYPES = (PATCH_DIFF, PATCH_FULL, PATCH_CROSS)

#: UCB1's exploration constant. sqrt(2) is the textbook value for rewards in
#: [0, 1], and `reward_for` normalises into that range for exactly this reason.
UCB_C = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_of(candidate: dict[str, Any], *, metric: str = METRIC) -> float | None:
    """A candidate's score, or None if it never produced one.

    None and 0.0 are different answers and conflating them is how a search learns
    to prefer crashing: a candidate whose `evaluate.py` raised has no score, and
    scoring it zero makes "fails immediately" a competitive strategy on any task
    whose real scores are negative -- which is every task written as a negated
    error, including the one `evolve init` scaffolds.
    """
    value = (candidate.get("metrics") or {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if math.isnan(float(value)) else float(value)


def scored(
    candidates: Iterable[dict[str, Any]], *, metric: str = METRIC
) -> list[dict[str, Any]]:
    """Only the candidates that produced a usable score, best first."""
    keep = [c for c in candidates if score_of(c, metric=metric) is not None]
    keep.sort(key=lambda c: score_of(c, metric=metric), reverse=True)  # type: ignore[arg-type,return-value]
    return keep


def reward_for(
    candidate: dict[str, Any], population: Sequence[dict[str, Any]], *, metric: str = METRIC
) -> float:
    """A candidate's score as a reward in [0, 1], relative to what has been seen.

    The bandit needs a bounded reward and `combined_score` is unbounded and
    arbitrarily scaled -- it is a negated error on one task and an accuracy on
    the next. So the reward is the candidate's *rank* among everything scored so
    far rather than its value, which is scale-free and needs no per-task tuning.
    An unscored candidate rewards 0: the arm proposed something that did not run.
    """
    mine = score_of(candidate, metric=metric)
    if mine is None:
        return 0.0
    values = [s for s in (score_of(c, metric=metric) for c in population) if s is not None]
    if len(values) <= 1:
        return 1.0
    worse = sum(1 for v in values if v < mine)
    return worse / (len(values) - 1)


# ---------------------------------------------------------------------------
# islands
# ---------------------------------------------------------------------------
def island_of(candidate: dict[str, Any]) -> int:
    """Which island a candidate was *bred on*. Absent means island 0.

    Records written before islands existed read as island 0 rather than as an
    error, so an old campaign still folds. This is the home island and it never
    changes; migration adds eligibility elsewhere rather than moving it.
    """
    value = candidate.get("island")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def islands_of(candidate: dict[str, Any]) -> list[int]:
    """Every island this candidate may be bred *from*: its home plus migrations.

    Two functions rather than one because they answer different questions and
    conflating them breaks migration in a way nothing reports: `island_of` is
    provenance -- where it came from, what the lineage chart draws -- and this is
    eligibility. `camp.candidates` folds the migration events into
    `migrated_to`; a record that predates them has none and reads as its home
    island alone.
    """
    home = island_of(candidate)
    extra = candidate.get("migrated_to")
    if not isinstance(extra, list):
        return [home]
    return [home] + [i for i in extra if isinstance(i, int) and i != home]


def assign_island(index: int, *, islands: int) -> int:
    """Which island the `index`-th candidate of a generation is bred on.

    Round-robin rather than random: with a population of four and three islands,
    random assignment leaves an island empty about a third of the time, and an
    empty island has no parent to breed from for the rest of the campaign.
    """
    return index % max(1, islands)


def members(
    candidates: Iterable[dict[str, Any]], island: int, *, metric: str = METRIC
) -> list[dict[str, Any]]:
    """One island's scored population, best first -- migrants included."""
    return scored((c for c in candidates if island in islands_of(c)), metric=metric)


def should_migrate(generation: int, *, interval: int) -> bool:
    """Whether generation `generation` begins with a migration.

    Never at generation 0 -- there is nothing to migrate -- and never when the
    interval is zero, which is how islands are switched off without a second
    flag.
    """
    return interval > 0 and generation > 0 and generation % interval == 0


def migrate(
    candidates: Sequence[dict[str, Any]], *, islands: int, metric: str = METRIC
) -> list[tuple[str, int, int]]:
    """Copy each island's champion to the next island round.

    Returns `(candidate_id, from_island, to_island)` for the caller to record.

    A *copy*, not a move: the champion stays where it is and becomes eligible as
    a parent on its neighbour too. Moving it would empty the island it came from
    of its best member, which is the opposite of what migration is for.
    """
    if islands <= 1:
        return []
    out: list[tuple[str, int, int]] = []
    for island in range(islands):
        best = members(candidates, island, metric=metric)
        if not best:
            continue
        champion = best[0]
        target = (island + 1) % islands
        # A champion that has already spread to its neighbour -- because it won
        # two migrations running, which is what a strong candidate does -- would
        # otherwise write one redundant event per interval for the rest of the
        # campaign.
        if target in islands_of(champion):
            continue
        out.append((str(champion["candidate_id"]), island, target))
    return out


# ---------------------------------------------------------------------------
# parents and inspirations
# ---------------------------------------------------------------------------
def rank_weights(n: int, *, pressure: float = 1.0) -> list[float]:
    """Selection weights for `n` candidates ordered best-first.

    `1 / (rank + 1) ** pressure` -- a power law, so the best is preferred by a
    constant factor over the second rather than by whatever the score difference
    happens to be. Score-proportional weighting is unusable here for the reason
    `reward_for` gives: the scale is task-defined, so on one task the top
    candidate takes 99% of the mass and on the next it takes 51%.

    `pressure = 0` is uniform selection, which is what makes this one knob rather
    than a policy flag.
    """
    return [1.0 / ((rank + 1) ** max(0.0, pressure)) for rank in range(max(0, n))]


def weighted_choice(items: Sequence[Any], weights: Sequence[float], rng: random.Random) -> Any:
    """One item, sampled by weight. Empty in, None out."""
    if not items:
        return None
    total = float(sum(weights))
    if total <= 0:
        return items[0]
    draw = rng.random() * total
    upto = 0.0
    # `strict`: the two sequences come from `rank_weights(len(items))` at every
    # call site, so a length mismatch is a caller bug and silently dropping the
    # tail of the population is how it would otherwise show up -- as a selection
    # policy that never picks the last candidate.
    for item, weight in zip(items, weights, strict=True):
        upto += weight
        if draw < upto:
            return item
    return items[-1]


def select_parent(
    candidates: Sequence[dict[str, Any]],
    *,
    island: int,
    rng: random.Random,
    pressure: float = 1.0,
    metric: str = METRIC,
) -> dict[str, Any] | None:
    """The candidate a mutation is bred from, or None on an empty island.

    Falls back to the whole population when the island is empty, which is the
    first two generations of every campaign: islands only mean anything once
    something has been scored on each of them, and refusing to breed until then
    would stall the search at generation 0.
    """
    pool = members(candidates, island, metric=metric) or scored(candidates, metric=metric)
    if not pool:
        return None
    return weighted_choice(pool, rank_weights(len(pool), pressure=pressure), rng)


def select_mate(
    candidates: Sequence[dict[str, Any]],
    parent: dict[str, Any],
    *,
    rng: random.Random,
    pressure: float = 0.5,
    metric: str = METRIC,
) -> dict[str, Any] | None:
    """A second parent for a `cross`, drawn from *outside* the parent's island.

    Crossing two members of one island is the recombination that has the least to
    recombine -- they share ancestry by construction. Drawing across islands is
    where the operator earns its cost. Lower pressure than `select_parent`
    because the mate contributes an idea rather than a baseline.
    """
    home = island_of(parent)
    pool = [
        c
        for c in scored(candidates, metric=metric)
        if c.get("candidate_id") != parent.get("candidate_id") and island_of(c) != home
    ]
    if not pool:
        # One island, or nothing scored elsewhere yet. Anything but the parent.
        pool = [
            c
            for c in scored(candidates, metric=metric)
            if c.get("candidate_id") != parent.get("candidate_id")
        ]
    if not pool:
        return None
    return weighted_choice(pool, rank_weights(len(pool), pressure=pressure), rng)


def inspirations(
    candidates: Sequence[dict[str, Any]],
    *,
    parent: dict[str, Any] | None,
    k: int = 4,
    failures: int = 2,
    metric: str = METRIC,
) -> dict[str, list[dict[str, Any]]]:
    """What the mutation operator is shown besides its parent.

    Two lists, and the second is the one people leave out. `elites` are the best
    scored candidates in the campaign, which is in-context selection pressure.
    `failures` are recent candidates that scored nothing at all, with their
    error -- and they are worth their tokens because the single most common
    wasted generation is four candidates that all reproduce one crash the
    operator could not see. An operator shown "this import does not exist on the
    target image" stops writing it.

    The parent is excluded from `elites`: it is already in the prompt as the
    thing being mutated, and listing it twice spends tokens to say one thing.
    """
    parent_id = (parent or {}).get("candidate_id")
    elites = [c for c in scored(candidates, metric=metric) if c.get("candidate_id") != parent_id]
    broken = [
        c
        for c in candidates
        if score_of(c, metric=metric) is None and (c.get("error") or c.get("skipped"))
    ]
    return {
        "elites": elites[: max(0, k)],
        # Newest first: an error from generation 9 is about the code being
        # written now, and one from generation 0 is usually about the scaffold.
        "failures": list(reversed(broken))[: max(0, failures)],
    }


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------
def normalise_source(source: str) -> str:
    """Source reduced to what a duplicate check should care about.

    Blank lines and trailing whitespace go; nothing else does. Comments stay,
    deliberately -- an operator that changed only a comment changed only a
    comment, and treating that as a fresh candidate is exactly the wasted
    evaluation this exists to prevent. Indentation stays too, because in Python
    it is the program.
    """
    return "\n".join(line.rstrip() for line in source.splitlines() if line.strip())


def source_key(source: str) -> str:
    """A stable id for a candidate's source, for deduplication."""
    return hashlib.sha256(normalise_source(source).encode("utf-8")).hexdigest()[:16]


def is_duplicate(source: str, seen: Iterable[str]) -> bool:
    return source_key(source) in set(seen)


def dedupe(sources: Iterable[str], *, seen: Iterable[str] | None = None) -> list[str]:
    """`sources` with duplicates of each other and of `seen` removed, in order."""
    known = set(seen or ())
    out: list[str] = []
    for source in sources:
        key = source_key(source)
        if key in known:
            continue
        known.add(key)
        out.append(source)
    return out


# ---------------------------------------------------------------------------
# the patch bandit
# ---------------------------------------------------------------------------
def arms_available(candidates: Sequence[dict[str, Any]], *, metric: str = METRIC) -> list[str]:
    """Which patch types can be drawn right now.

    `cross` needs two distinct scored parents, so it is unavailable until
    generation 1 at the earliest. Offering it anyway would have the bandit spend
    its exploration budget on an arm that degrades to `full` every time and then
    learn -- correctly, and uselessly -- that `cross` is no better than `full`.
    """
    if len(scored(candidates, metric=metric)) >= 2:
        return list(PATCH_TYPES)
    return [PATCH_DIFF, PATCH_FULL]


def empty_bandit() -> dict[str, dict[str, float]]:
    return {arm: {"pulls": 0.0, "reward": 0.0} for arm in PATCH_TYPES}


def bandit_from(
    candidates: Sequence[dict[str, Any]], *, metric: str = METRIC
) -> dict[str, dict[str, float]]:
    """Rebuild the bandit's statistics from the candidate records.

    Derived rather than carried, for the reason the whole ledger is folded from
    events: the campaign loop lives in whatever process started it, and state
    held only in that process is state a crash loses. Rebuilding is also what
    makes the arm statistics *checkable* -- they are a function of records anyone
    can read, not a counter only the loop could see.

    Cheap enough to do every generation: a hundred candidates is a hundred dict
    lookups.
    """
    stats = empty_bandit()
    population = list(candidates)
    for candidate in population:
        arm = candidate.get("patch_type")
        if arm not in stats:
            continue
        stats = bandit_update(stats, arm, reward_for(candidate, population, metric=metric))
    return stats


def seen_sources(candidates: Iterable[dict[str, Any]]) -> set[str]:
    """Every source key already proposed, for `dedupe`.

    Includes candidates that failed or escaped. A proposal identical to one that
    crashed is not worth a second evaluation any more than one identical to a
    winner is -- and an operator shown that failure in its prompt is precisely
    the one most likely to propose it again.
    """
    return {str(c["source_key"]) for c in candidates if c.get("source_key")}


def bandit_select(
    stats: dict[str, dict[str, float]],
    available: Sequence[str],
    rng: random.Random,
    *,
    pending: dict[str, int] | None = None,
) -> str:
    """UCB1 over the patch types, restricted to those that can be drawn.

    Unpulled arms come first, in a random order rather than in declaration order:
    fixed order means the first arm always gets generation 0's first candidate,
    and on a two-candidate population that is a systematic head start rather than
    an exploration.

    **`pending` is what makes this work in batches, and without it the bandit is
    worse than no bandit.** A generation is planned in one pass against one set of
    statistics, so every slot sees the same numbers and picks the same arm --
    which turned a population of four into four `cross` mutations and removed
    exactly the diversity the bandit exists to supply. Counting the pulls already
    assigned *in this generation* as pulls, but with no reward, is the standard
    fix: it dilutes an arm's exploration bonus as the batch fills without letting
    an unmeasured outcome move its mean.
    """
    if not available:
        return PATCH_FULL
    pending = pending or {}

    def pulls_of(arm: str) -> float:
        return stats.get(arm, {}).get("pulls", 0.0) + pending.get(arm, 0)

    unpulled = [arm for arm in available if pulls_of(arm) <= 0]
    if unpulled:
        return rng.choice(unpulled)
    total = sum(pulls_of(arm) for arm in available)
    log_total = math.log(max(total, 1.0))

    def measured_mean(arm: str) -> float | None:
        """This arm's mean over *measured* pulls, or None before its first one."""
        node = stats.get(arm) or {}
        pulls = float(node.get("pulls", 0.0))
        return float(node.get("reward", 0.0)) / pulls if pulls > 0 else None

    # What an arm is worth before anything has come back from it. `unpulled`
    # above catches an arm with no pulls at all, so what reaches here with no
    # *measured* pulls is an arm already drawn once in this batch and not yet
    # evaluated -- and the old arithmetic scored it `0 / 1`, the worst mean
    # available. That is the pending pull being read as a failure, which is
    # exactly what the comment below says it must not be: an arm drawn once in a
    # generation was then the least likely to be drawn again in it, however
    # promising it was. Optimism in the face of uncertainty is the rule this
    # bandit is built on, so an unmeasured arm gets the best mean anyone has.
    observed = [m for m in (measured_mean(a) for a in available) if m is not None]
    optimistic = max(observed) if observed else 1.0

    def value(arm: str) -> float:
        # The mean is over *measured* pulls only; the confidence term is over all
        # of them. A pending pull says "this arm is already being tried", not
        # "this arm scored zero".
        mean = measured_mean(arm)
        pulls = max(pulls_of(arm), 1.0)
        return (optimistic if mean is None else mean) + UCB_C * math.sqrt(log_total / pulls)

    return max(available, key=value)


def bandit_update(
    stats: dict[str, dict[str, float]], arm: str, reward: float
) -> dict[str, dict[str, float]]:
    """Record one pull. Returns a new dict; the input is not mutated."""
    out = {name: dict(node) for name, node in stats.items()}
    node = out.setdefault(arm, {"pulls": 0.0, "reward": 0.0})
    node["pulls"] += 1.0
    node["reward"] += float(reward)
    return out


def bandit_report(stats: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """The bandit as rows, for the campaign record and the evolve window.

    Reported rather than kept internal because "which patch type is working" is a
    fact about the *task*, and it is the one thing a campaign learns that
    transfers to the next one.
    """
    rows = []
    for arm in PATCH_TYPES:
        node = stats.get(arm) or {"pulls": 0.0, "reward": 0.0}
        pulls = node["pulls"]
        rows.append(
            {
                "patch_type": arm,
                "pulls": int(pulls),
                "mean_reward": round(node["reward"] / pulls, 4) if pulls else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# one generation's plan
# ---------------------------------------------------------------------------
def plan_generation(
    candidates: Sequence[dict[str, Any]],
    *,
    generation: int,
    population: int,
    islands: int,
    stats: dict[str, dict[str, float]],
    rng: random.Random,
    pressure: float = 1.0,
    metric: str = METRIC,
) -> list[dict[str, Any]]:
    """What to propose this generation: one plan per candidate slot.

    Computed *before* any model is called, so the whole policy is one pure
    function of the ledger and the seed. That is what makes a campaign
    replayable, and it is what lets the proposals be issued concurrently -- the
    plan does not depend on the candidates it is planning.

    Generation 0 has nothing to breed from, so every slot is a `full` mutation of
    the baseline with no parent. That is stated here rather than special-cased in
    the driver, because "what does generation 0 do" is a policy question.
    """
    available = arms_available(candidates, metric=metric)
    pending: dict[str, int] = {}
    plans: list[dict[str, Any]] = []
    for index in range(max(1, population)):
        island = assign_island(index, islands=islands)
        parent = select_parent(
            candidates, island=island, rng=rng, pressure=pressure, metric=metric
        )
        patch = bandit_select(stats, available, rng, pending=pending) if parent else PATCH_FULL
        mate = (
            select_mate(candidates, parent, rng=rng, metric=metric)
            if parent is not None and patch == PATCH_CROSS
            else None
        )
        if patch == PATCH_CROSS and mate is None:
            # The arm was drawable when the generation was planned and is not for
            # this slot -- one island, or every other scored candidate is this
            # parent. Degrade rather than propose a cross with one parent, which
            # is a `full` wearing the wrong label in the bandit's statistics.
            patch = PATCH_FULL
        # Counted after the degrade, so the tally matches the arms actually
        # proposed. Counting the drawn arm would have a generation that degraded
        # every `cross` believe it had explored `cross`.
        pending[patch] = pending.get(patch, 0) + 1
        plans.append(
            {
                "index": index,
                "generation": generation,
                "island": island,
                "patch_type": patch,
                "parent": parent,
                "mate": mate,
                **inspirations(candidates, parent=parent, metric=metric),
            }
        )
    return plans
