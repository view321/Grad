---
name: paper-corpus
description: How the retrieval funnel is organised, what each stage costs, search syntax, and how the local index is built. Load when tuning retrieval or debugging a search that missed something.
---

# The paper corpus

Two tiers, because discovery and recall are different problems.

**Tier 1 — discovery.** A remote index that finds papers not yet read, which a
local index cannot do by construction. Three doors, chosen by `--tier1` or
`[retrieval] tier1` in the config:

- `pwc` — Papers with Code (revived by Hugging Face). **The default.**
  Anonymous, answers in one to two seconds, and returns two genuinely different
  rankings (lexical and dense) that RRF was built to fuse. Rows are title-only;
  abstracts are backfilled from arXiv in one batched call, and anything not on
  arXiv is reranked on its title alone (the trace warns when that happens).
  "Expansion" here is a dense-neighbour walk, not the citation graph.
- `asta` — the Semantic Scholar corpus (~108M abstracts, ~12M full texts) over
  MCP, no institutional email needed, and the only door with real full-text
  snippets. Measured at 121s per search and 283s to report a backend failure,
  which is why it is no longer the default.
- `s2` — the Semantic Scholar REST API directly. Keys are only issued to
  institutional addresses, so a personal account shares the anonymous pool,
  which is near-permanently rate limited. (`both` = asta+s2, `all`, `none`.)

`asta` and `s2` are **switched off** in `[retrieval] tier1_disabled`, on those
latency numbers. Asking for one by name is refused rather than silently
answered by `pwc`; empty that list in `config/grad.toml` to use them.

**Tier 2 — recall.** SQLite FTS5 + vectors over papers actually read and your
own notes. This answers "where did I see that lemma," which no external index
can.

## The funnel

| # | stage | mechanism | cost |
|---|---|---|---|
| 0 | expand | Haiku: 1 question → 4–6 keyword queries + 1 HyDE abstract | quota |
| 1 | retrieve | tier 1 + local index (RRF) + neighbour/citation expansion → up to 300 | free |
| 2 | rerank | `voyageai/rerank-2.5` via OpenRouter → top ~50 | credits |
| 3 | triage | Haiku reads all 50 in one call → ~15 with a reason each | quota |
| 4 | select | the main agent reads the 15 | quota |

Each stage is cheaper per candidate than the one after it, so the expensive
stages only ever see filtered input. Stage 1 runs under a 300-second wall clock
(`[retrieval] stage1_budget_s`): what has been retrieved when it expires is
kept, what was skipped is written into the trace, and an endpoint that fails
once is dropped for the rest of the run.

**Expansion is retriever-specific.** HyDE works because a hypothetical answer's
*embedding* lands near the real answer's embedding — a dense gain. Feeding a
synthetic abstract to a lexical endpoint dilutes the query terms. So stage 0
emits keyword queries for the tier-1 retriever and one HyDE passage for the
vector side of tier 2, and never crosses them.

**Stages 2 and 3 are not redundant.** The reranker is calibrated, fast, and
quota-free, but it scores against a query string. Haiku is uncalibrated and has
listwise position bias, but it judges against the actual research question.
Keeping the reranker in the middle means the quota-consuming stage never sees
the candidates that were obviously wrong. Stage 3 is a funnel *widener*, not
a better ranker: the main agent can read 15 snippets, Haiku can read 50.

## Flags worth knowing

```bash
python -m tools.paper_search search "..." --json                 # everything
python -m tools.paper_search search "..." --no-expand --no-rerank --no-triage --json  # free path (rerank is the credits stage)
python -m tools.paper_search search "..." --tier1 asta --json    # refused while asta is in tier1_disabled
python -m tools.paper_search search "..." --local-only --json    # tier 2 only: papers read + own notes
python -m tools.paper_search search "..." --no-local --json      # tier 1 only: discovery
python -m tools.paper_search search "..." --no-citations --json  # skip neighbour/citation expansion
python -m tools.paper_search search "..." --candidates 600 --json # widen stage 1 (default ceiling 300)
python -m tools.paper_search search "..." --full --json          # include full snippets in the output
python -m tools.paper_search local "..." --json                  # tier 2 alone, no funnel
python -m tools.paper_search stats --json                        # what the local index contains
python -m tools.paper_search trace <name> --json                 # 300 → 50 → 15, with reasons
```

Every funnel run writes `notes/funnel/<slug>.json` — per-stage counts, warnings,
and the surviving candidates. Runs where the Haiku stages executed also write
`<slug>.md` beside it: the full prompt and raw response for each stage. Token
counts land in `ledger/quota.jsonl` (`python -m tools.quota summary --json`),
not in the notes. Debugging a funnel whose middle is invisible is guesswork.

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
neighbour/citation expansion buy more than reranker shopping does.
