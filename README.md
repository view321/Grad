# Grad

A personal research agent for mathematics and machine learning. Runs on one
person's Windows desktop, backed by a Claude Max subscription.

This is the implementation of [`HANDOFF.md`](HANDOFF.md) and its extension
[`HANDOFF-2.md`](HANDOFF-2.md), which together remain the design documents of
record. Where this README and the handoffs disagree, the handoffs are the intent
and this file is the report.

The design temperament is borrowed from Mario Zechner's *pi*: trust the model,
keep the system prompt small, keep the tool surface small, prefer files and CLIs
over framework machinery. One rule overrides "trust the model":

> **Anything that spends money, destroys work, or must be true before the fact
> is enforced mechanically, not by prompt.**

Every row below is enforced by a program that refuses to proceed. None of them
is a sentence in `prompts/system.md`.

| Thing that must hold | Enforced by |
|---|---|
| Code passes QA before it costs money | `core/gates.py:check_preflight`, called by both submitters |
| Code runs on the *remote* before a full run | the `smoke` check, run through `--smoke` and folded into the preflight record |
| A prediction exists before the result does | `core/gates.py:check_expectation`, bound at submit time |
| Results get recorded at all | `collect` writes the run record; a stale uncollected run blocks new submissions |
| Cumulative spend stays bounded | `core/ledger_store.py:rolling_spend` — actuals for collected runs, estimates for in-flight ones |
| The smoke job cannot become a backdoor | `core/gates.py:check_smoke_caps` clamps steps and wall clock, and clamps the wall clock again against the target's hourly rate so the cost cap is arithmetic rather than a self-report |
| Notebooks run clean top-to-bottom | `tools/nb.py verify` on a fresh kernel |
| No general remote-execution capability | credentials in Windows Credential Manager, read only by `jobs.py` / `gpu.py` |
| Concurrent ledger writes don't corrupt | one locked `core/jsonl.py:append`; no CLI writes a ledger file directly |
| Token and credit spend stays bounded, not merely measured | `core/budget.py`, checked at every gateable event |
| An evolutionary campaign cannot outspend its allocation | the campaign gate in `tools/evolve.py`, before generation 0 and before each generation after it |
| A job submitted to an org is collectable from that org | the namespace is persisted on the run handle, not just passed at submit |
| Every number in a report traces to a run record | `tools/report.py check` refuses on an unresolved claim, on a `claims.tex` that has drifted from `claims.json`, and on a measured-looking number typed into the generated prose |
| Every citation in a report is a real paper | `report cite` resolves only against the corpus and verified S2 ids, and `check` re-resolves each entry's id rather than trusting its `gradsource` label |
| A result that has not been judged cannot be published | `report check` refuses while any cited run has an unjudged deviation |

### The one thing that is *not* fully mechanical, and why

**Subscription tokens are enforced to a granularity of one turn's overrun.**
Tokens are consumed continuously inside a turn and there is no way to refuse
mid-turn, so `agent.py` checks the remaining allocation *before* issuing the
next turn and `hooks.py` denies cost-bearing Bash once the project is over. The
turn that crosses the ceiling finishes.

Both surfaces run the same check because both run the same loop:
`agent.drive_turn` is the one place a turn is issued, and it checks the budget
before `query` and records the turn's usage after it. The CLI and the desktop
app called it independently for a while, and only the CLI accounted — so a
session held entirely in the app spent tokens that no ledger recorded and no
ceiling could see.

A second honesty note: subscription quota is not linear in tokens, and the real
limits are rolling windows (5-hour and weekly on Max) that the SDK does not
expose as a remaining balance. **A token ceiling is a proxy you control, not a
mirror of Anthropic's limit.** The meter says so on screen.

## Install

```bash
pip install -e ".[agent,notebook,retrieval,remote,ui,math,dev]"
```

`ui` brings the embedded JupyterLab with it (the `lab` extra, pinned exactly
because the 3→4 break is what killed the Tabnine extension). That is not a
convenience: the notebook window's interior *is* Lab, so an app installed
without it ships a window whose only content is a button that fails.

Optional extras, each pinned and each independently skippable:
`lab-extensions` (LSP, git — `python-lsp-server[all]` is heavier than everything
else here put together), `wiki` (RepoWiki), `evolve` (ShinkaEvolve).

The core — ledger, preflight, gates, submitters — needs only the standard
library plus a file lock. Everything heavier is optional and imported at the
point of use, so `preflight` can refuse a submission on a machine with no
NiceGUI and no SDK installed.

Then authenticate against the subscription, not the API:

```bash
claude setup-token
```

Export the result as `CLAUDE_CODE_OAUTH_TOKEN` and make sure `ANTHROPIC_API_KEY`
is **not** set — it outranks the OAuth token in the credential chain and will
silently bill the Developer Platform instead. `python agent.py --check` removes
it from the process environment and reports what it removed.

Store credentials once; they never enter the agent's environment:

```bash
python -m tools.jobs credential set hf_token
python -m tools.jobs credential set openrouter_key
python -m tools.jobs credential set voyage_key
python -m tools.jobs credential set claude_oauth_token
python -m tools.jobs credential set asta_api_key    # optional; raises rate limits
python -m tools.jobs credential set context7_key    # optional; raises rate limits
```

Or store them from the app: the workspace menu (`project ▾`) has a credentials
panel, which is the same command with `--stdin` instead of the `getpass` prompt.
That exists because the prompt needs a terminal, and needing one for this was
the only thing that forced a shell open beside the app on a fresh machine. The
value goes down a pipe rather than in an argument — an argv is visible to
anything that can list processes.

### Retrieval without an institutional email, and without waiting

Tier 1 defaults to **Papers with Code** (`paperswithcode.co/api/v1`) — the
catalogue as revived by Hugging Face, not the `.com` site Meta shut down. It is
anonymous and read-only: no account, no key, nothing to store. The endpoints are
those of [`huggingface/pwc-cli`][pwc-cli], which is the reference client for
this API.

The choice is about latency, and the numbers are measured rather than assumed:

| tier 1 | one search | key | text |
| --- | --- | --- | --- |
| `pwc` *(default)* | **1–2 s** | none | title only; abstracts fetched separately |
| `asta` | ~121 s, and ~283 s to report a backend failure | optional | full-text snippets |
| `s2` | fast when it answers | institutional addresses only | full-text snippets |

Stage 0 turns one question into ~6 queries and each goes to two endpoints, so
Asta's per-call latency is twenty minutes of discovery before anything is
ranked — and every caller gives up first: the agent's own Bash tool backgrounds
a command at 120 s, and a shell `timeout` kills it. Semantic Scholar
[no longer issues API keys to free-domain email addresses][s2-keys], leaving a
personal account on a shared anonymous pool that is rate limited often enough
that "no results" and "no key" are hard to tell apart.

**What the default costs.** Asta is the only one of the three with genuine
full-text snippets, which is what §5 designed stage-3 triage around. Under `pwc`,
triage reads the *abstract* instead — fetched from arXiv in one batched request
for the whole candidate pool (`core/http.py:arxiv_abstracts`), because nearly
every row is an arXiv paper and `id_list` takes a hundred ids at once. A
candidate that ends up with no abstract is reranked on its title, and the trace
says how many did. `pwc`'s expansion is also a *dense neighbour* rather than a
citation edge, and `neighbours` reports it as one rather than claiming the
citation graph §5 asks for.

Set `[retrieval] tier1` to `asta`, `s2`, `both` (the two Semantic Scholar doors)
or `all`, or pass `--tier1` per search. Two wall clocks bound a run either way:
`request_deadline_s` caps one request, and `stage1_budget_s` caps the whole of
stage 1 — when it is spent the funnel keeps what it retrieved and records how
many queries it actually searched.

[pwc-cli]: https://github.com/huggingface/pwc-cli
[s2-keys]: https://www.semanticscholar.org/product/api

## Run

```bash
python agent.py                      # interactive session
python agent.py "derive the update rule for ..."   # one turn
python agent.py --probe              # the §9 permission deny probe
python agent.py --ui                 # the NiceGUI desktop window
```

**Run `--probe` after every SDK upgrade.** The safety story rests on the exact
name and semantics of a deny-by-default permission mode, and those have changed
between releases. The probe attempts a call that should be denied and reports
whether it was *denied* — not prompted, not silently allowed.

## The tools

Each is a CLI with `--json` on every subcommand, a stable envelope, and errors
that carry the literal next command.

| CLI | What it does |
|---|---|
| `tools/paper_search.py` | the five-stage retrieval funnel: expand → retrieve → rerank → triage → select |
| `tools/paper_ingest.py` | arXiv LaTeX source → section-aware chunks → the local index |
| `tools/nb.py` | persistent Jupyter kernel: `exec` (timeout-bounded), `verify` (fresh kernel), `restart` |
| `tools/preflight.py` | run the QA gate, write `ledger/preflight/<hash>.json` |
| `tools/jobs.py` | Hugging Face Jobs: `submit` / `status` / `collect` / `ceilings` / `credential` |
| `tools/gpu.py` | the same verbs against a known SSH host |
| `tools/ledger.py` | `expect` / `query` / `verdict` / `falsify` / `verify` / `reindex` |
| `tools/quota.py` | measured token and credit usage, summarised by stage, role, and project |
| `tools/budget.py` | projects and their ceilings: `new` / `use` / `status` / `raise` / `close` |
| `tools/docs.py` | is this library call current? introspection first, then Context7 |
| `tools/evolve.py` | evolutionary search as a budgeted campaign, over ShinkaEvolve |
| `tools/report.py` | `draft` / `write` / `cite` / `check` / `build` — the report and its gate |
| `tools/lab.py` | the embedded JupyterLab server (human editing surface) |
| `tools/wiki.py` | RepoWiki over `core/` and `tools/` — **human-facing only**, not an agent tool |

### Exit codes

A usage error, a gate refusal, and an upstream failure are three different
things, and the model should not have to read prose to tell them apart.

| code | meaning |
|---|---|
| 0 | ok |
| 1 | internal error (a bug in the CLI) |
| 2 | usage error — bad or unknown flags |
| 3 | not found |
| 4 | **gate**: preflight missing or failing |
| 5 | **gate**: no open expectation |
| 6 | **gate**: spend ceiling exceeded |
| 7 | **gate**: stale uncollected run |
| 8 | upstream failure |
| 9 | a check ran and failed |
| 10 | job still running (not an error) |
| 11 | configuration or credential problem |
| 12 | **gate**: project budget exceeded |

12 is deliberately distinct from 6: "this research ran out of its allocation" is
not "the machine is out of money", and conflating them makes the wrong fix look
right.

## A full cycle

```bash
python -m tools.budget new --id proj-scaling-w2 --title "width vs depth" \
    --gpu-usd 50 --quota-tokens 5e6 --credits-usd 10 --payer hf:myorg --use --json
python -m tools.preflight run --spec pipeline/spec.toml --only tests,dry_run --json
python -m tools.jobs submit --spec pipeline/spec.toml --smoke --json
python -m tools.ledger expect --task scaling-w2 --quantity val_loss@1e9_tokens \
    --low 2.9 --high 3.2 \
    --basis 'arXiv:2001.08361|Fig 3|3.05|1.3B params, 100B tokens' \
    --comparability 'our tokenizer differs; eval is a 5k held-out subset' --json
python -m tools.jobs submit --spec pipeline/spec.toml --expect exp-... --json
python -m tools.jobs collect run-... --json
python -m tools.ledger verdict run-... --quantity val_loss@1e9_tokens \
    --verdict bug --note 'lr schedule off by one step' --json
python -m tools.report draft --project proj-scaling-w2 --json   # free, no model
python -m tools.report check --project proj-scaling-w2 --json   # the gate
```

Skip any of the preflight/expectation steps and `submit` refuses, with the
command you skipped in its `fix` field. Skip the verdict and `report check`
refuses. `skills/preflight/SKILL.md` documents the submission spec format and
what each check catches.

## Layout

```
agent.py              ClaudeSDKClient loop, permission configuration, the deny probe
hooks.py              PreToolUse gate (a speed bump) + Stop hook (budget warnings)
prompts/system.md     under 1000 tokens
core/                 the machinery the CLIs share, so no tool can forget a rule
  cli.py              the §8 CLI contract, implemented once
  jsonl.py            the single locked write path to the ledgers
  submission.py       the resolved submission and its hash
  gates.py            the submit gates and the smoke carve-out
  budget.py           the project dimension and its three ceilings
  ledger_store.py     event-folded runs, rolling spend, staleness, derived index
  submit.py           shared submitter machinery: record, collect, deviations
  campaign.py         campaigns, candidates, and the evolve-block escape check
  report.py           the claim and citation guarantees `report check` enforces
  corpus.py           FTS5 + vectors + reciprocal rank fusion
  haiku.py            funnel stages 0 and 3, via forced SDK tools
  http.py             Semantic Scholar, rerank, embeddings, Context7
tools/                the CLIs
ui/                   the NiceGUI workspace: a tiling shell over twelve windows
  tokens.py           the design tokens; the stylesheet is generated from them
  layout.py           the pane tree and the moves over it   -- pure, tested
  models.py           what each window shows, as plain data -- pure, tested
  registry.py         the one list of windows the shell is derived from
  shell.py            the chrome, and how a window survives a retile
  tasks.py            local commands run in the background, and how to stop one
  sessions.py         named chat sessions: a file each, listed by a glob
  windows/            twelve renderers, none of which read a ledger directly
  jupyter_theme.py    the same tokens, emitted as JupyterLab's custom.css
config/jupyter/       the Lab server config: framing headers, overrides, theme
skills/               loaded on demand, not into the default context
ledger/               expectations.jsonl, runs.jsonl, quota.jsonl, projects.jsonl,
                      campaigns.jsonl, candidates.jsonl, preflight records
reports/<project>/    main.tex, claims.json, references.bib, the PDF
evals/retrieval.jsonl the arbiter for any change to retrieval
```

## Status

Implemented and tested: the ledger, the submission hash, every gate, the smoke
caps, the CLI contract, the hook, the persistent kernel, notebook verification,
the project dimension and its three ceilings, HF organization namespaces,
library-currency checking, the campaign loop and its budget gate, and the report
generator with all four of its rules, the background task runner and its stop
path, and named chat sessions. `pytest` covers these — no network, no SDK
required, and the "no network" half is now enforced rather than intended: an
autouse fixture in `tests/conftest.py` replaces `core.http._httpx`, because a
suite that reaches the network does not fail, it *hangs*.

Implemented but not exercised against a live service: the HF Jobs backend, the
SSH backend, Semantic Scholar, the OpenRouter reranker, Voyage embeddings, the
two Haiku funnel stages, Context7, ShinkaEvolve, and RepoWiki. They are written
against the documented interfaces and fail with actionable errors rather than
tracebacks, but a real credential and a real run are what will find the
mismatches.

Tier-1 discovery is no longer in that list. A live run of the default funnel
returns 118 candidates from two rankings in 3.1 seconds, fills 99 abstracts in
one arXiv request, and hands 15 survivors to triage.

Papers with Code and Asta are both exercised against the live services now, and
what that found is worth recording, because none of it could have been reasoned
out from the documentation:

* **Asta's `search_papers_by_relevance` takes `keyword`** where `snippet_search`
  takes `query`, so tier 1 lost that endpoint on every search — the server
  answers in about a second with a validation error, which the slower endpoint's
  latency hid.
* **Asta answers `tools/call` with an event stream it holds open**, pinging every
  15 seconds. `httpx`'s timeout is per socket read, so every ping reset it: a
  buffered read waited for a close the server never promised, and discovery did
  not fail, it never returned. The client now streams the response, stops at the
  reply to its own request, and enforces a total deadline.
* **Asta wraps its hits under `result`** (singular), a key `_rows` did not know —
  so even with the argument fixed, every hit was discarded as an unrecognised
  envelope. And its `limit` must be ≤ 100, which `--no-expand` exceeded by
  dividing the candidate ceiling across one query instead of six.
* **Papers with Code returns no abstract with a search row**, which is why
  `arxiv_abstracts` exists.

The *shapes* the remaining tools answer in are still read defensively, and an
unrecognised one still raises rather than returning an empty list.

**Two of [HANDOFF-2 §23](HANDOFF-2.md)'s open questions are now closed:**

- **Context7's REST endpoints** (§23 item 2) are verified against the live API:
  `/api/v2/libs/search` returns `{"results": […]}`, and `/api/v2/context` with
  `type=json` returns `{"codeSnippets": […]}`. They stay in `[docs]` config
  because a third-party API can move; the client reads both the v2 and v1
  response keys so a config change is sufficient either way.
- **ShinkaEvolve exposes no per-candidate callback, and no per-generation entry
  point either** (§23 item 1). `ShinkaEvolveRunner` has `run` and `run_async`,
  both of which own the whole loop — which is the control the campaign budget
  gate needs in order to re-check between generations. **This is the evidence
  §21 said a fork should wait for.** `evolve run` refuses with that explanation
  rather than handing control away with the budget unchecked;
  `python -m tools.evolve capabilities --json` reports what it found.

**Still open:**

- **Whether `headless/claude` works against a Max subscription specifically** is
  reported in Shinka's release notes and untested here.
- **Historical records are left as `"unassigned"`** rather than retrofitted with
  a project. Cheap to change while the ledger is small.
- **Phase 2 of the campaign loop (remote evaluation) is not enabled.**
  `--remote` is refused: the gate is proven locally first, because doing the
  ledger work and the spend work simultaneously against live GPU jobs is how you
  learn about exit 7 the hard way.

**One correction to HANDOFF-2 itself.** §20 records `repowiki map` as taking
`--format html --open`. The 0.3.1 wheel's `map` takes exactly one `path`,
`--format text|json`, and has neither flag — so `tools/wiki.py` invokes it once
per scope directory asking for JSON and renders the HTML itself.

**`tools/docs.py check` imports the modules it inspects,** and importing runs
their top-level code. That is inherent to the introspection oracle — a checker
that does not import can only guess at what is installed. Run it on your own
pipeline, not on a repository you just downloaded; the module docstring and
`--help` both say so.

Four things worth knowing before trusting them:

- **The Agent SDK surface is version-sensitive.** `core/haiku.py` and
  `agent.py` are written against the interfaces described in the handoff
  (`ClaudeAgentOptions`, `create_sdk_mcp_server`, `@tool`, `HookMatcher`,
  `dontAsk`). Check them against the installed `claude-agent-sdk` before relying
  on them, and re-run `agent.py --probe`.
- **`gpu.py` materialises an SSH key to a mode-600 temp file** for the duration
  of one call, because `ssh` needs a key file. That is weaker than never
  materialising it. Prefer an SSH agent or a `~/.ssh/config` host entry and
  leave `key_credential` unset, in which case no key is ever written by us.
- **The preflight record is a plain JSON file the agent can write.** Gate 1
  reads `ledger/preflight/<hash>.json` and the model has `Write`. So the
  cheapest way past the most important gate is not an argument, it is a file —
  which puts it in the same class as the bypasses `core/credentials.py` already
  declares out of scope (an agent that can run Python can import `keyring`).
  Signing the record would not close it either, since the signing key would be
  readable by the same process. What actually bounds this is that the *spend*
  gates do not read agent-writable state: the ledger is append-only through one
  locked path, and `collect` prices runs from the platform's own timestamps.
- **The S2 half of the citation rule is weaker than the corpus half.** A
  `corpus` entry is verified by resolving its document id against the local
  index. An `s2` entry is verified by its `S2:<id>` shape and the overlap scores
  `report cite` recorded when it accepted the match — re-querying the live
  service inside a gate would make `check` require the network. Forging one is
  no longer a single line of BibTeX, but it is not impossible.

The order in §12 of the handoff is deliberate — build the agent, use it for a
week, *then* harvest `evals/retrieval.jsonl` from what retrieval was actually
reached for. The eval file here is a schema and a handful of seed rows, not a
benchmark; authoring it cold would measure the imagination rather than the
system.

## Tests

```bash
python -m pytest -q
```

The gate tests run against a real ledger in a temp workspace rather than against
mocks. A mock of a gate proves nothing about the gate, and these are the checks
that stand between an agent under deadline pressure and a GPU bill.
