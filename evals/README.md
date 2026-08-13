# evals/

`retrieval.jsonl` is the arbiter for any change to the retrieval stack in §5,
including whether the two Haiku stages earn their quota.

**It is deliberately almost empty.** §12 step 3 is a week of real use with a
written record of what retrieval was actually reached for, and the eval set is
harvested from that log. A benchmark of imagined queries measures the
imagination. The rows here are schema examples, marked `"seed": true`, and
should be replaced — not padded — with real ones.

## Schema

One JSON object per line:

```json
{"id": "q001",
 "question": "the real question, as it was actually asked",
 "asked_at": "2026-08-13",
 "relevant": [{"paper": "arXiv:2001.08361", "grade": 2, "why": "the scaling law itself"},
              {"paper": "arXiv:2203.15556", "grade": 1, "why": "revises the exponent"}],
 "notes": "what made this hard: the term of art changed between 2019 and 2022",
 "seed": false}
```

`grade` is 0/1/2 (irrelevant / useful / directly answers). Graded relevance and
Recall@50 extract far more signal per labelling hour than a binary Hit@10,
because most queries have several relevant documents and only one "hit".

## Target size, and why

40–60 queries. At n=25 a Hit@10 difference needs to be roughly 15–20 points
before it clears the noise, and the differences between rerank-only and
rerank+triage will not be that large. Where the confidence intervals overlap,
say so and keep the cheaper configuration: "no measurable difference" is a valid
and common result, and it favours dropping a stage.

Tune expansion before the reranker. The retriever sets the ceiling — no reranker
pushes Hit@10 past roughly 88%, because the missing documents never entered the
candidate set.
