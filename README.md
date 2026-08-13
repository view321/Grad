# Grad

A personal research agent for mathematics and machine learning. Runs on one
person's Windows desktop, backed by a Claude Max subscription.

This is the implementation of [`HANDOFF.md`](HANDOFF.md), which remains the
design document of record. Where this README and the handoff disagree, the
handoff is the intent and this file is the report.

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
| The smoke job cannot become a backdoor | `core/gates.py:check_smoke_caps` clamps steps, wall clock, and cost in code |
| Notebooks run clean top-to-bottom | `tools/nb.py verify` on a fresh kernel |
| No general remote-execution capability | credentials in Windows Credential Manager, read only by `jobs.py` / `gpu.py` |
| Concurrent ledger writes don't corrupt | one locked `core/jsonl.py:append`; no CLI writes a ledger file directly |

## Install

```bash
pip install -e ".[agent,notebook,retrieval,remote,ui,math,dev]"
```

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
```

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
| `tools/quota.py` | measured token and credit usage, summarised by stage |

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

## A full cycle

```bash
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
```

Skip any of the first three and the fourth refuses, with the command you skipped
in its `fix` field. `skills/preflight/SKILL.md` documents the submission spec
format and what each check catches.

## Layout

```
agent.py              ClaudeSDKClient loop, permission configuration, the deny probe
hooks.py              PreToolUse gate (a speed bump) + Stop hook (quota accounting)
prompts/system.md     under 1000 tokens
core/                 the machinery the CLIs share, so no tool can forget a rule
  cli.py              the §8 CLI contract, implemented once
  jsonl.py            the single locked write path to the ledgers
  submission.py       the resolved submission and its hash
  gates.py            the four submit gates and the smoke carve-out
  ledger_store.py     event-folded runs, rolling spend, staleness, derived index
  submit.py           shared submitter machinery: record, collect, deviations
  corpus.py           FTS5 + vectors + reciprocal rank fusion
  haiku.py            funnel stages 0 and 3, via forced SDK tools
  http.py             Semantic Scholar, rerank, embeddings
tools/                the CLIs
ui/                   NiceGUI app and the four widgets
skills/               loaded on demand, not into the default context
ledger/               expectations.jsonl, runs.jsonl, quota.jsonl, preflight records
evals/retrieval.jsonl the arbiter for any change to retrieval
```

## Status

Implemented and tested: the ledger, the submission hash, all four gates, the
smoke caps, the CLI contract, the hook, the persistent kernel, and notebook
verification. `pytest` covers these — 85 tests, no network, no SDK required.

Implemented but not exercised against a live service: the HF Jobs backend, the
SSH backend, Semantic Scholar, the OpenRouter reranker, Voyage embeddings, and
the two Haiku funnel stages. They are written against the documented interfaces
and fail with actionable errors rather than tracebacks, but a real credential
and a real run are what will find the mismatches.

Two things worth knowing before trusting them:

- **The Agent SDK surface is version-sensitive.** `core/haiku.py` and
  `agent.py` are written against the interfaces described in the handoff
  (`ClaudeAgentOptions`, `create_sdk_mcp_server`, `@tool`, `HookMatcher`,
  `dontAsk`). Check them against the installed `claude-agent-sdk` before relying
  on them, and re-run `agent.py --probe`.
- **`gpu.py` materialises an SSH key to a mode-600 temp file** for the duration
  of one call, because `ssh` needs a key file. That is weaker than never
  materialising it. Prefer an SSH agent or a `~/.ssh/config` host entry and
  leave `key_credential` unset, in which case no key is ever written by us.

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
