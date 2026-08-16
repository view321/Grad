"""grad-paper-search -- the five-stage retrieval funnel (HANDOFF §5).

    | 0 | Query expansion | Haiku: 1 question -> 4-6 keyword queries + 1 HyDE abstract | quota  |
    | 1 | Retrieve        | tier 1 (pwc by default) + local index (RRF) + expansion    | free   |
    | 2 | Rerank          | voyageai/rerank-2.5 -> top ~50                             | credits|
    | 3 | Triage          | Haiku reads all 50 in one call, returns ~15 with a reason  | quota  |
    | 4 | Select          | The main agent reads the 15                                | quota  |

    Under the default `pwc` tier 1, "expansion" is a dense-neighbour walk rather
    than the citation graph, rows arrive title-only, and abstracts are backfilled
    from arXiv in one batched call -- see `tier1_clients` and `_fill_abstracts`.

The ordering is the design: each stage is cheaper per candidate than the one
after it, so the expensive stages only ever see filtered input. Stages 0 and 3
are optional at the flag level on purpose -- §12 step 9 adds them one at a time
and keeps whichever `evals/retrieval.jsonl` justifies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core import config as config_mod, corpus, credentials, haiku, http, paths, quota_log
from core.cli import Cli, main
from core.errors import GradError, UpstreamError, UsageError
from core.ledger_store import now_iso

cli = Cli(
    "grad-paper-search",
    "Search the literature (Papers with Code / Semantic Scholar) and the local index, rerank, triage.",
    epilog=(
        "Discovery and recall are different problems. Tier 1 finds papers you have not\n"
        "read; tier 2 (the local index) answers 'where did I see that lemma'.\n"
        "A local index cannot do discovery by construction, which is why both exist.\n\n"
        "Tier 1 defaults to pwc (Papers with Code): anonymous, fast, title-only rows\n"
        "with abstracts backfilled from arXiv. The two doors onto the Semantic Scholar\n"
        "corpus -- asta over MCP, s2 over the REST API -- are switched off in\n"
        "[retrieval] tier1_disabled because both answer in minutes rather than seconds;\n"
        "empty that list to have them back.\n\n"
        "The retriever sets the ceiling: query expansion and neighbour/citation\n"
        "expansion buy more than reranker shopping does."
    ),
)


#: `tier1` value -> the clients it selects, in the order they are queried.
#: `both` predates `pwc` and still means what it meant: the two Semantic Scholar
#: doors, for comparing them.
TIER1_SOURCES = ("pwc", "asta", "s2", "both", "all", "none")


class _Budget:
    """A wall clock over the whole of stage 1.

    `core/http.py` bounds one *request*; this bounds the run. They are different
    numbers because stage 0 turns one question into six queries and each is put
    to two endpoints, so a per-request deadline of five minutes is a
    one-hour stage. What made that concrete: a live `snippet_search` took 283
    seconds to report that Asta's own backend had refused a connection, and the
    funnel was killed by its caller long before the endpoints that *were*
    working could contribute anything.

    Stopping early is not the same as failing. What has been retrieved is kept,
    what was skipped is written into the trace, and the caller gets results --
    which is the whole difference between a slow funnel and a broken one.
    """

    def __init__(self, limit: float) -> None:
        self.limit = max(0.0, limit)
        self._started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    @property
    def spent(self) -> bool:
        return bool(self.limit) and self.elapsed >= self.limit


def _discover(
    queries: list[str],
    tier1: list[tuple[str, Any]],
    *,
    per_query: float,
    budget: _Budget,
    jobs: int,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]], int]:
    """Stage 1's calls, `jobs` at a time. Returns (results, dropped, queries tried).

    Every call here is an independent read-only HTTP request against a public
    endpoint, so this is the cheapest concurrency in the project and the one that
    buys the most: six expanded queries against two verbs on one client is twelve
    round trips, and at `pwc`'s one-to-two seconds that is the difference between
    a funnel that answers in three seconds and one that answers in twenty.

    Two invariants survive the change, and both were load-bearing:

    * **A source that fails once is dropped for the rest of the run.** An endpoint
      that is down is down for every query, and finding that out costs a request
      -- a live `snippet_search` took 283 seconds to report that Asta's backend
      had refused a connection. The check is racy under concurrency, in the
      benign direction: the calls already in flight when the first failure lands
      still run, and nothing after them does.
    * **The budget stops the tail.** Workers check it as they start, so a spent
      budget means the queued items return "not attempted" rather than being
      cancelled mid-request. Cancelling would leave sockets open and lose hits
      that had already been paid for.

    Results come back in submission order so the caller's merge is deterministic.
    """
    work: list[tuple[int, str, str, Any, str]] = []
    for q_index, query in enumerate(queries):
        for name, client in tier1:
            for verb in ("snippet_search", "paper_search"):
                work.append((q_index, query, name, client, verb))

    dropped: set[tuple[str, str]] = set()
    attempted_queries: set[int] = set()
    total = len(work)

    def one(item: tuple[int, str, str, Any, str]) -> dict[str, Any]:
        q_index, query, name, client, verb = item
        base = {"source": name, "verb": verb, "query": query, "hits": [], "error": None}
        if (name, verb) in dropped or budget.spent:
            return {**base, "skipped": True}
        attempted_queries.add(q_index)
        _progress(f"stage 1: {name}.{verb} q{q_index + 1}/{len(queries)}")
        try:
            hits = getattr(client, verb)(query, limit=per_query)
        except GradError as exc:
            dropped.add((name, verb))
            _progress(f"stage 1: {name}.{verb} failed; not retried this run")
            return {**base, "error": f"{name}.{verb}: {exc}"}
        _progress(f"stage 1: {name}.{verb} returned {len(hits)}")
        return {**base, "hits": hits}

    if jobs <= 1 or total <= 1:
        results = [one(item) for item in work]
    else:
        with ThreadPoolExecutor(
            max_workers=min(jobs, total), thread_name_prefix="grad-funnel"
        ) as pool:
            results = list(pool.map(one, work))
    return results, dropped, len(attempted_queries)


def _progress(message: str) -> None:
    """A line of progress on stderr, where a `--json` contract cannot see it.

    The funnel prints nothing until its envelope, which is minutes later, and
    everything that runs it reads a pipe: the tasks window streams the tail, and
    the agent's own Bash call gives up at 120 seconds and moves the command to
    the background. A run with no output is indistinguishable from a hung one in
    all three places, and that is how a slow stage got read as a broken tool.
    """
    print(message, file=sys.stderr, flush=True)


def disabled_tier1(cfg: Any) -> frozenset[str]:
    """Sources switched off in config, whatever asks for them.

    Asta and `s2` ship switched off, and the reason is latency rather than
    correctness -- both serve the Semantic Scholar corpus, and the corpus is
    fine. Measured live, Asta answers a search in ~121 seconds and takes ~283
    seconds to report that its own backend refused a connection; `s2`'s
    anonymous pool is shared and near-permanently rate limited. Stage 0 turns
    one question into six queries and each goes to two verbs, so either of them
    is twenty minutes of discovery before anything is ranked, and every caller
    -- the agent's Bash tool at 120s, a shell `timeout`, a person -- gives up
    first. Papers with Code answers the same query in one to two seconds.

    `_Budget` already bounds the damage, but bounding it still means spending
    the whole stage-1 budget discovering that a door is slow, on every run. This
    stops paying that at all, and it is a switch rather than a deletion: empty
    `[retrieval] tier1_disabled` and both are back, unchanged.
    """
    raw = cfg.get("retrieval", "tier1_disabled", ()) or ()
    if isinstance(raw, str):
        raw = [raw]
    return frozenset(str(name).lower() for name in raw)


def tier1_clients(cfg: Any, override: str | None = None) -> list[tuple[str, Any]]:
    """The discovery clients for this run, named so a trace can say which spoke.

    All three answer in the same vocabulary (`core/http.py:_row`, `_pwc_row`),
    so the funnel does not know which tier a candidate came from and does not
    have to. What differs is whether the door opens, and how fast:

    * **pwc** -- Papers with Code, as revived by Hugging Face. Anonymous, one to
      two seconds, and two genuinely different rankings (lexical and dense) that
      RRF was built to fuse. The default, because it is the one that answers.
    * **asta** -- the same Semantic Scholar corpus as `s2`, through a door that
      opens without an institutional address, and the only one of the three with
      real full-text snippets. Measured live at 121s for a search and 283s to
      report a backend failure, which is why it is no longer the default.
    * **s2** -- the REST API directly. Its own keys are only issued to
      institutional addresses, so a personal account falls back to the shared
      anonymous pool, which is near-permanently rate limited.
    """
    chosen = str(override or cfg.get("retrieval", "tier1", "pwc")).lower()
    if chosen not in TIER1_SOURCES:
        raise UsageError(
            f"unknown tier-1 source {chosen!r}",
            fix=f"one of: {', '.join(TIER1_SOURCES)}",
        )
    disabled = disabled_tier1(cfg)
    # Asked for by name, and switched off. Quietly substituting the default here
    # would answer a question nobody asked: `--tier1 asta` is how you compare
    # two retrievers, and a comparison that silently ran the same one twice is
    # worse than one that refused.
    if chosen in disabled:
        raise UsageError(
            f"tier-1 source {chosen!r} is switched off in config",
            fix=(
                f"remove {chosen!r} from [retrieval] tier1_disabled in config/grad.toml "
                "to use it, or --tier1 pwc"
            ),
        )
    out: list[tuple[str, Any]] = []
    if chosen in ("pwc", "all") and "pwc" not in disabled:
        out.append(("pwc", http.PapersWithCode(cfg)))
    if chosen in ("asta", "both", "all") and "asta" not in disabled:
        out.append(("asta", http.Asta(cfg)))
    if chosen in ("s2", "both", "all") and "s2" not in disabled:
        out.append(("s2", http.SemanticScholar(cfg)))
    # `both` means the two Semantic Scholar doors, and both of them ship shut.
    # Returning nothing would run the funnel with no tier 1 at all -- discovery
    # silently downgraded to the local index, which `--local-only` already says
    # explicitly and which nobody asking for `both` wanted.
    if not out and chosen != "none":
        raise UsageError(
            f"--tier1 {chosen} selects only sources that are switched off "
            f"({', '.join(sorted(disabled))})",
            fix=(
                "--tier1 pwc, or empty [retrieval] tier1_disabled in config/grad.toml. "
                "--local-only searches the papers already ingested."
            ),
        )
    return out


def _search_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("question", help="the research question, in words")
    p.add_argument(
        "--tier1",
        choices=list(TIER1_SOURCES),
        help=(
            "which discovery client to use (default from config; pwc unless changed). "
            "asta and s2 are switched off -- see [retrieval] tier1_disabled"
        ),
    )
    p.add_argument("--top", type=int, help="how many to return (default from config)")
    p.add_argument("--candidates", type=int, help="stage-1 candidate ceiling")
    p.add_argument("--no-expand", action="store_true", help="skip stage 0 (Haiku query expansion)")
    p.add_argument("--no-rerank", action="store_true", help="skip stage 2 (costs credits)")
    p.add_argument("--no-triage", action="store_true", help="skip stage 3 (Haiku triage)")
    p.add_argument("--local-only", action="store_true", help="tier 2 only: papers already read")
    p.add_argument("--no-local", action="store_true", help="tier 1 only: discovery")
    p.add_argument("--no-citations", action="store_true", help="skip citation-graph expansion")
    p.add_argument("--full", action="store_true", help="include full snippets in the output")
    p.add_argument(
        "--jobs",
        type=int,
        help=(
            "stage-1 requests in flight at once (default from [execution] default_jobs). "
            "Six expanded queries against two verbs is twelve round trips."
        ),
    )


@cli.command("search", "run the funnel end to end", setup=_search_args)
def cmd_search(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    paths.ensure_workspace()
    if args.local_only and args.no_local:
        raise UsageError("--local-only and --no-local contradict each other", fix="pick one")

    top = args.top or int(cfg.get("retrieval", "triage_top", 15))
    ceiling = args.candidates or int(cfg.get("retrieval", "candidates", 300))
    rerank_top = int(cfg.get("retrieval", "rerank_top", 50))
    log_name = _slug(args.question)
    trace: dict[str, Any] = {"question": args.question, "at": now_iso(), "log": log_name, "stages": {}}

    # -- stage 0: expansion --------------------------------------------------
    queries = [args.question]
    hyde = None
    if not args.no_expand:
        expansion = haiku.expand(
            args.question, model=cfg.model_for("expand"), log_name=log_name
        )
        queries = list(expansion["queries"])
        hyde = expansion["hyde"]
    trace["stages"]["0_expand"] = {"queries": queries, "hyde_words": len(hyde.split()) if hyde else 0}

    # -- stage 1: retrieve ---------------------------------------------------
    candidates: dict[str, dict[str, Any]] = {}
    rankings: list[list[dict[str, Any]]] = []
    #: Search calls that *failed*, kept apart from the trace's other warnings.
    #: An empty local index is not a failure and must not be reported as one.
    upstream_failures: list[str] = []

    tier1 = tier1_clients(cfg, args.tier1) if not args.local_only else []
    trace["stages"]["1_sources"] = [name for name, _ in tier1]

    if tier1:
        per_query = max(5, ceiling // max(1, len(queries) * 2 * len(tier1)))
        budget = _Budget(float(cfg.get("retrieval", "stage1_budget_s", 300)))
        jobs = args.jobs if args.jobs is not None else int(cfg.get("execution", "default_jobs", 4))
        found, dropped, searched = _discover(
            queries, tier1, per_query=per_query, budget=budget, jobs=jobs
        )
        for item in found:
            if item["error"]:
                trace.setdefault("warnings", []).append(item["error"])
                upstream_failures.append(item["error"])
                continue
            # Merged in *work-item order*, not completion order. `setdefault`
            # keeps the first row seen for an id, so merging as results arrive
            # would make which copy of a paper's metadata survives depend on
            # which endpoint happened to answer first -- a funnel whose output
            # changed run to run for no reason anyone could see.
            rankings.append(item["hits"])
            for hit in item["hits"]:
                candidates.setdefault(hit["id"], hit)
        # Both caps are reported rather than left to be inferred from a thin
        # result set: a funnel that quietly searched two of six queries looks
        # exactly like a corpus that had little to say.
        if dropped:
            trace.setdefault("warnings", []).append(
                "dropped for the rest of this run after one failure each: "
                + ", ".join(sorted(f"{n}.{v}" for n, v in dropped))
            )
        if budget.spent:
            trace.setdefault("warnings", []).append(
                f"tier-1 discovery stopped after {budget.limit:.0f}s having searched "
                f"{searched} of {len(queries)} queries — raise [retrieval] "
                "stage1_budget_s, or --no-expand to search one query instead of six"
            )
        trace["stages"]["1_discovery"] = {
            "queries_searched": searched,
            "queries": len(queries),
            "dropped": sorted(f"{n}.{v}" for n, v in dropped),
            "jobs": jobs,
            "seconds": round(budget.elapsed, 1),
        }
        if not args.no_citations and not budget.spent:
            # Twenty calls live under that single check -- five seeds by two
            # clients by two directions -- so testing the budget once at the top
            # bounds nothing: the stage can run minutes past `stage1_budget_s`
            # doing expansion after it has already decided it is out of time.
            # Rechecked per call, and the loop is left rather than skipped, so
            # what is already retrieved is kept and the trace still records how
            # far it got.
            seeds = [c for c in list(candidates.values())[:5] if c.get("paper_id")]
            expanded = 0
            for seed in seeds:
                if budget.spent:
                    break
                for name, client in tier1:
                    # A verb that was dropped in discovery was dropped because
                    # this endpoint refused it or timed out. Asking the same
                    # client again, per seed, spends the remaining budget
                    # rediscovering a failure already recorded above.
                    if (name, "neighbours") in dropped or budget.spent:
                        continue
                    for direction in ("citations", "references"):
                        if budget.spent:
                            break
                        try:
                            hits = client.neighbours(
                                seed["paper_id"], direction=direction, limit=10
                            )
                        except GradError:
                            dropped.add((name, "neighbours"))
                            continue
                        expanded += 1
                        rankings.append(hits)
                        for hit in hits:
                            candidates.setdefault(hit["id"], hit)
            trace["stages"]["1_discovery"]["expansions"] = expanded
            trace["stages"]["1_discovery"]["dropped"] = sorted(
                f"{n}.{v}" for n, v in dropped
            )

    if not args.no_local:
        local = _local_ranked(args.question, hyde, cfg, trace)
        if local:
            rankings.append(local)
            for hit in local:
                candidates.setdefault(hit["id"], hit)

    fused = corpus.rrf(rankings, k=int(cfg.get("retrieval", "rrf_k", 60)))
    pool = [candidates[f["id"]] | {"rrf": f["rrf"]} for f in fused if f["id"] in candidates][:ceiling]
    _fill_abstracts(pool, cfg, trace)
    quota_log.record(
        quota_log.STAGE_RETRIEVE, unit="quota", detail={"queries": len(queries), "candidates": len(pool)}
    )
    trace["stages"]["1_retrieve"] = {"rankings": len(rankings), "candidates": len(pool)}

    if not pool:
        # A run that found nothing is the run the funnel view exists to explain
        # -- "why is the obviously relevant paper not in here" -- so it gets a
        # trace like any other. Returning before writing one left exactly the
        # interesting failures invisible.
        _write_trace(log_name, trace, [])
        warnings = list(dict.fromkeys(trace.get("warnings") or []))
        if upstream_failures and not rankings:
            # Every retrieval call failed. That is an upstream failure, not an
            # empty result set, and the difference matters more here than
            # anywhere else in this tool: `ok: true` with no results reads as
            # "the literature has nothing on this", which is a conclusion nobody
            # should draw from a rate limit.
            raise UpstreamError(
                "every retrieval call failed, so the search returned nothing: "
                + "; ".join(dict.fromkeys(upstream_failures)),
                fix=_tier1_fix(
                    trace["stages"]["1_sources"], upstream_failures, disabled_tier1(cfg)
                ),
            )
        return {
            "question": args.question,
            "results": [],
            "trace": trace,
            "trace_log": str(paths.notes_dir() / "funnel" / f"{log_name}.json"),
            # The old note here recommended `--no-expand`, which is advice for a
            # cause this branch cannot distinguish and sent anyone following it
            # to a second empty run.
            "note": "; ".join(warnings) or "no candidates; widen --candidates or rephrase",
        }

    # -- stage 2: rerank -----------------------------------------------------
    ranked = pool
    if not args.no_rerank and len(pool) > 1:
        docs = [_document_text(c) for c in pool]
        try:
            scored = http.rerank(args.question, docs, cfg=cfg, top_n=min(rerank_top, len(docs)))
            ranked = apply_rerank(pool, scored)
            if not ranked:
                # Every index came back unusable. Dropping the pool here would
                # return nothing at all from a run that has already spent stage-0
                # quota -- an unordered pool is still better than no results.
                trace.setdefault("warnings", []).append(
                    "rerank returned no usable indices; falling back to the unreranked pool"
                )
                ranked = pool[:rerank_top]
        except GradError as exc:
            trace.setdefault("warnings", []).append(f"rerank unavailable: {exc}")
            ranked = pool[:rerank_top]
    else:
        ranked = pool[:rerank_top]
    trace["stages"]["2_rerank"] = {"in": len(pool), "out": len(ranked), "skipped": args.no_rerank}

    # -- stage 3: triage -----------------------------------------------------
    survivors = ranked[:top]
    if not args.no_triage and ranked:
        verdicts = haiku.triage(
            args.question, ranked, model=cfg.model_for("triage"), log_name=log_name
        )
        reasons = {v["id"]: v.get("reason", "") for v in verdicts if v.get("keep")}
        kept = [c for c in ranked if c["id"] in reasons]
        survivors = [{**c, "reason": reasons[c["id"]]} for c in kept][:top]
        trace["stages"]["3_triage"] = {"in": len(ranked), "kept": len(kept), "returned": len(survivors)}
    else:
        trace["stages"]["3_triage"] = {"skipped": True}

    _write_trace(log_name, trace, survivors)
    return {
        "question": args.question,
        "results": [_public(c, full=args.full) for c in survivors],
        "funnel": {
            "candidates": len(pool),
            "reranked": len(ranked),
            "returned": len(survivors),
        },
        "trace": trace,
        # `.json`, which is what `_write_trace` actually writes. The `.md` this
        # used to return was a path to nothing, and the empty-pool branch above
        # already got it right -- so the two exits from one function disagreed
        # about where the trace lives.
        "trace_log": str(paths.notes_dir() / "funnel" / f"{log_name}.json"),
    }


def apply_rerank(pool: list[dict[str, Any]], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder the pool by the reranker's verdict, ignoring unusable indices.

    The index comes from upstream and is used to subscript `pool`. An
    out-of-range value would raise IndexError and abandon a funnel run that has
    already spent stage-0 quota; a negative one would silently promote the wrong
    candidate, which is worse because nothing would look broken.
    """
    return [
        {**pool[i], "rerank_score": s.get("score")}
        for s in scored
        if isinstance(i := s.get("index"), int)
        and not isinstance(i, bool)
        and 0 <= i < len(pool)
    ]


def _tier1_fix(
    sources: list[str],
    failures: list[str] | None = None,
    disabled: frozenset[str] | None = None,
) -> str:
    """What to actually do when discovery is down, per source *and per failure*.

    This used to say "store a Semantic Scholar API key -- it is free", which
    stopped being true: Ai2 no longer accept key requests from free-domain email
    addresses, so for a personal account that instruction has no ending. Advice
    that cannot be followed is worse than no advice, because it is followed
    first and the real fix is found second.

    The Asta branch had the same problem one step down. It said "discovery is
    rate limited, not broken" whatever had happened, and pointed at a key -- but
    a live run failed with `ConnectionRefusedError` raised inside Asta's own
    backend, which no key affects. So the advice is now chosen by what the
    failures actually say rather than assumed.

    `disabled` is the third round of the same lesson. Every branch below that
    sends the reader to another tier-1 source has to know whether that source is
    switched off, because `tier1_clients` refuses one that is -- and advice
    whose next step is a usage error is exactly what the two paragraphs above
    are about.
    """
    joined = " ".join(failures or [])
    off = disabled or frozenset()
    local = "Meanwhile --local-only searches what is already ingested."

    def elsewhere(name: str) -> str:
        """Point at another source, or at the switch that would allow it."""
        if name in off:
            return (
                f"{name} serves the same literature more slowly and is switched off -- "
                f'remove "{name}" from [retrieval] tier1_disabled in config/grad.toml '
                "to try it"
            )
        return (
            f"python -m tools.paper_search search '<question>' --tier1 {name} --json   "
            f'(or set [retrieval] tier1 = "{name}" in config/grad.toml)'
        )
    # The cause first where there is one, because a rate limit has a fix of its
    # own and it is not the same fix as an endpoint being down.
    if "rate-limited" in joined or "429" in joined:
        if "asta" in sources:
            return (
                "retry -- discovery is rate limited, not broken. A key raises Asta's limits "
                "and is requested from a form rather than reviewed: "
                f"python -m tools.jobs credential set {credentials.ASTA_KEY}. {local}"
            )
        if "pwc" in sources:
            return (
                "retry -- discovery is rate limited, not broken. This catalogue is "
                "anonymous, so there is no key that raises it; the same literature is "
                "reachable more slowly through " + elsewhere("asta") + f". {local}"
            )
        return (
            "retry -- Semantic Scholar's anonymous pool is shared and its own keys are only "
            "issued to institutional addresses. Papers with Code is anonymous and answers in "
            "about a second: " + elsewhere("pwc") + f". {local}"
        )
    if sources == ["pwc"]:
        return (
            "this catalogue is anonymous, so there is no key to add and nothing to "
            "configure -- it is down or unreachable. Retry, or reach the same literature "
            "more slowly through " + elsewhere("asta") + f". {local}"
        )
    if sources in (["s2"], ["asta"]):
        # Both doors onto the Semantic Scholar corpus point at the one that is
        # neither rate limited nor minutes slow.
        return (
            "Semantic Scholar's own API only issues keys to institutional addresses, and "
            "Asta answers in minutes rather than seconds. Papers with Code is anonymous and "
            "answers in about a second: " + elsewhere("pwc") + f". {local}"
        )
    return (
        "the message above is the service's own and names the endpoint that refused -- no "
        f"key affects it. The endpoints fail independently, so retrying is worth it even "
        f"when one of them is down. {local}"
    )


def _fill_abstracts(pool: list[dict[str, Any]], cfg: Any, trace: dict[str, Any]) -> None:
    """Give the candidates that have no text an abstract, in one request.

    Stage 2 reranks on `title + snippet-or-abstract` and stage 3 triages on the
    same, so a pool of bare titles is a measurably worse funnel -- and the fast
    corpus does not return abstracts with its search results. Nearly every row
    it does return is an arXiv paper, and arXiv takes a hundred ids at once, so
    the whole pool costs one call rather than one per candidate.

    Degrades rather than fails: a candidate with no abstract ranks on its title,
    which is what would have happened without this. What it must not do is go
    unrecorded -- a funnel silently reranking titles looks exactly like one
    reranking abstracts, and only one of them is the retrieval §5 evaluates.
    """
    wanted = {
        c["external"]["ArXiv"]: c
        for c in pool[: http.ARXIV_BATCH]
        if not (c.get("snippet") or c.get("abstract"))
        and isinstance(c.get("external"), dict)
        and c["external"].get("ArXiv")
    }
    if not wanted:
        return
    _progress(f"stage 1: fetching {len(wanted)} abstracts from arXiv")
    found = http.arxiv_abstracts(list(wanted), cfg=cfg)
    for arxiv_id, abstract in found.items():
        if arxiv_id in wanted:
            wanted[arxiv_id]["abstract"] = abstract
    missing = len(wanted) - len(found)
    trace["stages"]["1_abstracts"] = {"wanted": len(wanted), "found": len(found)}
    if missing:
        trace.setdefault("warnings", []).append(
            f"{missing} of {len(wanted)} candidates were reranked and triaged on their "
            "title alone — arXiv returned no abstract for them"
        )


def _local_ranked(question: str, hyde: str | None, cfg: Any, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Tier 2, fused across FTS5 and vectors before joining the global pool."""
    path = paths.corpus_sqlite()
    if not path.exists():
        return []
    con = corpus.connect(path)
    try:
        lexical = corpus.fts_search(con, question, limit=100)
        dense: list[dict[str, Any]] = []
        bound = corpus.embedding_model(con)
        if hyde and bound:
            try:
                # The HyDE passage is embedded with the same model the index was
                # built with; a vector from another space is noise, not signal.
                vector = http.embed([hyde], cfg=cfg, input_type="query")[0]
                dense = corpus.vector_search(con, vector, limit=100)
            except GradError as exc:
                trace.setdefault("warnings", []).append(f"local vector search unavailable: {exc}")
        fused = corpus.rrf([lexical, dense], k=int(cfg.get("retrieval", "rrf_k", 60)))
        return [
            {
                "id": f"local:{row['doc_id']}#{row['id']}",
                "paper_id": None,
                "title": row.get("title"),
                "year": row.get("year"),
                "snippet": row.get("text", "")[:1500],
                "section": row.get("section"),
                "source": "local",
                "doc_id": row.get("doc_id"),
            }
            for row in fused[:100]
        ]
    finally:
        con.close()


def _document_text(c: dict[str, Any]) -> str:
    return f"{c.get('title') or ''}\n{(c.get('snippet') or c.get('abstract') or '')}"[:4000]


def _public(c: dict[str, Any], *, full: bool) -> dict[str, Any]:
    text = c.get("snippet") or c.get("abstract") or ""
    return {
        "id": c["id"],
        "title": c.get("title"),
        "year": c.get("year"),
        "source": c.get("source"),
        "arxiv": (c.get("external") or {}).get("ArXiv"),
        "doi": (c.get("external") or {}).get("DOI"),
        "reason": c.get("reason"),
        "rerank_score": c.get("rerank_score"),
        "text": text if full else text[:600],
    }


# The alphabet `_slug` produces, and the only thing `trace` will open.
SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")


def _slug(question: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60]
    return f"{now_iso()[:10]}-{base or 'query'}"


def _write_trace(log_name: str, trace: dict[str, Any], survivors: list[dict[str, Any]]) -> None:
    """The funnel view in §10 renders this; §12 step 3's week of real use reads it."""
    d = paths.notes_dir() / "funnel"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{log_name}.json").write_text(
        json.dumps({**trace, "survivors": survivors}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
@cli.command(
    "local",
    "search only the local index (papers read + own notes)",
    setup=lambda p: (
        p.add_argument("question"),
        p.add_argument("--top", type=int, default=15),
    ),
)
def cmd_local(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    trace: dict[str, Any] = {}
    rows = _local_ranked(args.question, None, cfg, trace)
    return {"question": args.question, "results": rows[: args.top], "warnings": trace.get("warnings", [])}


@cli.command("stats", "what the local index contains")
def cmd_stats(_: argparse.Namespace) -> dict[str, Any]:
    path = paths.corpus_sqlite()
    if not path.exists():
        return {"exists": False, "fix": "python -m tools.paper_ingest arxiv <id> --json"}
    con = corpus.connect(path)
    try:
        return {"exists": True, **corpus.stats(con)}
    finally:
        con.close()


@cli.command(
    "trace",
    "show a previous funnel run (300 -> 50 -> 15, with reasons)",
    setup=lambda p: p.add_argument("name", nargs="?", help="trace name; omit to list"),
)
def cmd_trace(args: argparse.Namespace) -> dict[str, Any]:
    d = paths.notes_dir() / "funnel"
    if not args.name:
        return {"traces": sorted(p.stem for p in d.glob("*.json"))} if d.exists() else {"traces": []}
    # The name is a slug this CLI minted, and the agent can invoke this command.
    # Without the pattern check a name like `../../../etc/some` would read and
    # print any JSON file the process can open, widening the deny-by-default
    # file boundary that the rest of §9 works to keep narrow.
    path = d / f"{args.name}.json"
    if not SLUG_RE.fullmatch(args.name) or not path.exists():
        raise GradError("not_found", f"no trace named {args.name!r}", exit_code=3,
                        fix="python -m tools.paper_search trace --json   # lists them")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(cli)
