# notes/

The research log. The agent appends here, greps here, and cites paths from here;
`paper_ingest.py notes notes/` puts it into the local index so past derivations
become retrievable alongside papers.

Two things are written here automatically:

- `notes/funnel/<date>-<slug>.md` — the full prompt, raw response, and token
  counts for both Haiku funnel stages, per query. Stages 0 and 3 are the one
  place this system uses subagents, and this log is the mitigation that makes
  the exception acceptable: debugging a funnel whose middle is invisible is
  guesswork.
- `notes/funnel/<date>-<slug>.json` — the machine-readable trace the UI's funnel
  view renders (400 → 50 → 15, with the reason each survivor was kept).

Everything else here is written by hand or by the agent during a session.

**Step 3 of the plan lives here too.** A week of real use, with a written record
of what retrieval was actually reached for — the real question, not a tidied
version. `evals/retrieval.jsonl` is harvested from that log rather than authored
cold.
