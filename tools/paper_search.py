"""grad-paper-search -- the five-stage retrieval funnel (HANDOFF §5).

    | 0 | Query expansion | Haiku: 1 question -> ~5 keyword queries + 1 HyDE abstract | quota  |
    | 1 | Retrieve        | S2 snippets + local index (RRF) + citation expansion      | free   |
    | 2 | Rerank          | voyageai/rerank-2.5 -> top ~50                            | credits|
    | 3 | Triage          | Haiku reads all 50 in one call, returns ~15 with a reason | quota  |
    | 4 | Select          | The main agent reads the 15                               | quota  |

The ordering is the design: each stage is cheaper per candidate than the one
after it, so the expensive stages only ever see filtered input. Stages 0 and 3
are optional at the flag level on purpose -- §12 step 9 adds them one at a time
and keeps whichever `evals/retrieval.jsonl` justifies.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from core import config as config_mod, corpus, haiku, http, paths, quota_log
from core.cli import Cli, main
from core.errors import GradError, UsageError
from core.ledger_store import now_iso

cli = Cli(
    "grad-paper-search",
    "Search the literature (Semantic Scholar) and the local index, rerank, and triage.",
    epilog=(
        "Discovery and recall are different problems. Tier 1 (S2) finds papers you have\n"
        "not read; tier 2 (the local index) answers 'where did I see that lemma'.\n"
        "A local index cannot do discovery by construction, which is why both exist.\n\n"
        "The retriever sets the ceiling: expansion and citation expansion buy more than\n"
        "reranker shopping does."
    ),
)


def _search_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("question", help="the research question, in words")
    p.add_argument("--top", type=int, help="how many to return (default from config)")
    p.add_argument("--candidates", type=int, help="stage-1 candidate ceiling")
    p.add_argument("--no-expand", action="store_true", help="skip stage 0 (Haiku query expansion)")
    p.add_argument("--no-rerank", action="store_true", help="skip stage 2 (costs credits)")
    p.add_argument("--no-triage", action="store_true", help="skip stage 3 (Haiku triage)")
    p.add_argument("--local-only", action="store_true", help="tier 2 only: papers already read")
    p.add_argument("--no-local", action="store_true", help="tier 1 only: discovery")
    p.add_argument("--no-citations", action="store_true", help="skip citation-graph expansion")
    p.add_argument("--full", action="store_true", help="include full snippets in the output")


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

    if not args.local_only:
        s2 = http.SemanticScholar(cfg)
        per_query = max(5, ceiling // max(1, len(queries) * 2))
        for query in queries:
            for fn in (s2.snippet_search, s2.paper_search):
                try:
                    hits = fn(query, limit=per_query)
                except GradError as exc:
                    trace.setdefault("warnings", []).append(str(exc))
                    continue
                rankings.append(hits)
                for hit in hits:
                    candidates.setdefault(hit["id"], hit)
        if not args.no_citations:
            seeds = [c for c in list(candidates.values())[:5] if c.get("paper_id")]
            for seed in seeds:
                for direction in ("citations", "references"):
                    try:
                        hits = s2.neighbours(seed["paper_id"], direction=direction, limit=10)
                    except GradError:
                        continue
                    rankings.append(hits)
                    for hit in hits:
                        candidates.setdefault(hit["id"], hit)

    if not args.no_local:
        local = _local_ranked(args.question, hyde, cfg, trace)
        if local:
            rankings.append(local)
            for hit in local:
                candidates.setdefault(hit["id"], hit)

    fused = corpus.rrf(rankings, k=int(cfg.get("retrieval", "rrf_k", 60)))
    pool = [candidates[f["id"]] | {"rrf": f["rrf"]} for f in fused if f["id"] in candidates][:ceiling]
    quota_log.record(
        quota_log.STAGE_RETRIEVE, unit="quota", detail={"queries": len(queries), "candidates": len(pool)}
    )
    trace["stages"]["1_retrieve"] = {"rankings": len(rankings), "candidates": len(pool)}

    if not pool:
        return {"question": args.question, "results": [], "trace": trace,
                "note": "no candidates; try --no-expand to see the raw query, or widen --candidates"}

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
        "trace_log": str(paths.notes_dir() / "funnel" / f"{log_name}.md"),
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
    "show a previous funnel run (400 -> 50 -> 15, with reasons)",
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
