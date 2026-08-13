---
name: paper-corpus
description: How the retrieval funnel is organised, what each stage costs, search syntax, and how the local index is built. Load when tuning retrieval or debugging a search that missed something.
---

# The paper corpus

Two tiers, because discovery and recall are different problems.

**Tier 1 — discovery.** Semantic Scholar over HTTP: ~108M abstracts, ~12M full
texts via the snippet endpoint, plus citation-graph expansion. This finds papers
not yet read, which a local index cannot do by construction.

**Tier 2 — recall.** SQLite FTS5 + vectors over papers actually read and your
own notes. This answers "where did I see that lemma," which no external index
can.

## The funnel

| # | stage | mechanism | cost |
|---|---|---|---|
| 0 | expand | Haiku: 1 question → ~5 keyword queries + 1 HyDE abstract | quota |
| 1 | retrieve | S2 snippets + local index (RRF) + citations → 200–400 | free |
| 2 | rerank | `voyageai/rerank-2.5` → top ~50 | credits |
| 3 | triage | Haiku reads all 50 in one call → ~15 with a reason each | quota |
| 4 | select | the main agent reads the 15 | quota |

Each stage is cheaper per candidate than the one after it, so the expensive
stages only ever see filtered input.

**Expansion is retriever-specific.** HyDE works because a hypothetical answer's
*embedding* lands near the real answer's embedding — a dense gain. S2's snippet
endpoint is lexical/hybrid, and feeding it a synthetic abstract dilutes the
query terms. So stage 0 emits keyword queries for S2 and one HyDE passage for
the vector side of tier 2, and never crosses them.

**Stages 2 and 3 are not redundant.** The reranker is calibrated, fast, and
quota-free, but it scores against a query string. Haiku is uncalibrated and has
listwise position bias, but it judges against the actual research question.
Keeping the reranker in the middle means the quota-consuming stage never sees
the 350 candidates that were obviously wrong. Stage 3 is a funnel *widener*, not
a better ranker: the main agent can read 15 snippets, Haiku can read 50.

## Flags worth knowing

```bash
python -m tools.paper_search search "..." --json                 # everything
python -m tools.paper_search search "..." --no-expand --no-triage --json  # free path only
python -m tools.paper_search search "..." --local-only --json    # papers already read
python -m tools.paper_search search "..." --candidates 600 --json # widen stage 1
python -m tools.paper_search trace <name> --json                  # 400 → 50 → 15, with reasons
```

Every funnel run writes a trace to `notes/funnel/` — the full prompt, raw
response, and token counts for both Haiku stages, plus the surviving candidates.
Debugging a funnel whose middle is invisible is guesswork.

## Ingest

```bash
python -m tools.paper_ingest arxiv 2001.08361 --json
python -m tools.paper_ingest notes notes/ --json
```

Ingest from **LaTeX source, not PDF**. This is the single largest quality lever
in the stack: it preserves equations, theorem environments, and section
structure that PDF extraction destroys. The chunker keeps theorem and equation
environments whole and tags each chunk with its section.

The index records which embedding model built it and refuses vectors from any
other. Changing models means `reembed --model ... --yes`, deliberately, over the
whole corpus — a mixed vector space silently degrades every dense search after
it.

## Tuning

`evals/retrieval.jsonl` is the arbiter for any change here, including whether
stages 0 and 3 earn their quota. Two things that are easy to get wrong:

- **Size.** 20–30 query→paper pairs cannot separate rerank-only from
  rerank+triage. Target 40–60 queries, and report Recall@50 and graded relevance
  alongside Hit@10. Where the intervals overlap, keep the cheaper configuration —
  "no measurable difference" is a valid result and it favours dropping a stage.
- **Provenance.** Do not author the eval set cold. Harvest it from a written log
  of what retrieval was actually reached for. A benchmark of imagined queries
  measures the imagination.

And the ceiling fact: no reranker pushes Hit@10 past roughly 88%, because the
missing documents never entered the candidate set. Multi-query rewriting and
citation-graph expansion buy more than reranker shopping does.
