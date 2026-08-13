# GradientAgent — Handoff

A personal research agent for mathematics and machine learning. Runs on the user's
Windows desktop, backed by a Claude Max 5x subscription.

This document is the durable record of what was decided and why. Session context is
disposable; this file is not.

*Revision 2 (2026-08-13): closed the gaps where the doc's own "enforce mechanically"
principle was not applied — pre-registration, result collection, cumulative spend, notebook
reproducibility. Added the remote smoke job, reworked preflight staleness, and added §10 on
the interface. Interface framework decided: NiceGUI.*

*Revision 3 (2026-08-13): closed four gaps found in architecture review — the smoke check's
bootstrap circularity (§6), uncollected runs escaping the spend ceiling (§6, §7), credential
isolation as the real control behind §9's "no general remote execution" claim, and
multi-writer JSONL locking on Windows (§7). Dropped the expectation-predates-preflight
ordering rule (§7), named the tier-2 embedding model and hardened the triage tool (§5), made
`collect` non-blocking by default with a per-host rate table for SSH costs (§7), added the
preflight hash escape hatch (§6), and reworded the quota meter to what it can actually
measure (§10). Verified externally: OpenRouter does serve `voyageai/rerank-2.5` via a
dedicated rerank endpoint, and the SDK's `dontAsk` mode exists with the semantics §9 assumes.*

---

## 1. Goal

An agent that helps with math/ML research by:

- **Retrieving** — live search over the published literature for discovery, plus a local
  index over papers actually read.
- **Computing** — a persistent Jupyter kernel for exploratory analysis, plots, and small
  experiments.
- **Verifying** — a mandatory pre-flight gate (tests, dry run, remote smoke, symbolic/numeric
  checks) before any job that costs money.
- **Remembering** — an append-only ledger of predictions and outcomes that later sessions
  can query.
- **Offloading** — heavy compute to remote GPUs over SSH and to Hugging Face Jobs.

Design temperament is borrowed from Mario Zechner's **pi** (`badlogic/pi-mono`): trust the
model, keep the system prompt small, keep the tool surface small, prefer files and CLIs over
framework machinery. Do not build features we don't need.

One rule that overrides "trust the model": **anything that spends money, destroys work, or
must be true before the fact is enforced mechanically, not by prompt.** The model is trusted
to do research; it is not trusted to remember its own safety rails under deadline pressure.

### Where that rule applies

This list is the spine of the design. Every item is enforced by a program that refuses to
proceed, never by a sentence in `system.md`. If a future change adds something to this
category, it gets a submitter-side check, not a prompt line.

| Thing that must hold | Enforced by | Section |
|---|---|---|
| Code passes QA before it costs money | `preflight.py`, checked by `jobs.py` / `gpu.py` | §6 |
| Code runs on the *remote* before a full run | remote smoke check, part of preflight | §6 |
| A prediction exists before the result does | `jobs.py` / `gpu.py` refuse without an open expectation | §6, §7 |
| Results get recorded at all | `jobs.py collect` writes the run record; a stale uncollected run blocks new submissions | §6, §7 |
| Cumulative spend stays bounded | rolling total over `runs.jsonl` — actuals for collected runs, estimates for in-flight ones — checked at submit | §6 |
| The smoke job cannot become a backdoor | `submit --smoke` skips the gates but is hard-capped in code: one step, minutes of wall clock, cents | §6 |
| Notebooks run clean top-to-bottom | `nb.py verify` on a fresh kernel | §6 |
| No general remote-execution capability | credential isolation plus architecture — `gpu.py` / `jobs.py` are the only path that can authenticate | §9 |
| Concurrent ledger writes don't corrupt | one locked `append()` helper; no CLI writes a ledger file directly | §7 |

---

## 2. Verified facts about auth and billing

These were checked against primary sources during the design session. They corrected two
wrong assumptions, so they are worth keeping.

**The Claude Agent SDK can run on a Claude subscription.** Anthropic's support article
"Use the Claude Agent SDK with your Claude plan" states subscription plans "are now eligible
to receive a monthly Agent SDK credit" (Pro $20, Max 5x $100, Max 20x $200), and that the
credit applies to "Claude Agent SDK usage in your own projects, the `claude -p` command,
GitHub Actions integration, and third-party apps using the Agent SDK."

**That credit split is currently paused.** As of 2026-06-15: "For now, nothing has changed:
Claude Agent SDK, `claude -p`, and third-party app usage still draw from your subscription's
usage limits." So today it comes out of the normal Max 5x quota — the *same* quota as
interactive Claude Code use. Plan for the $100/month ceiling returning later.

**Auth mechanism.** `claude setup-token` mints a one-year OAuth token; export it as
`CLAUDE_CODE_OAUTH_TOKEN`. Per the authentication docs, it "authenticates with your Claude
subscription and requires a Pro, Max, Team, or Enterprise plan." The docs also confirm the
credential chain applies to "the CLI and the surfaces that wrap it, including the VS Code
extension, the Agent SDK, and GitHub Actions."

**Three gotchas, all confirmed in the docs:**

1. `ANTHROPIC_API_KEY` (precedence #3) outranks `CLAUDE_CODE_OAUTH_TOKEN` (#5) and `/login`
   subscription OAuth (#7). A stray exported key silently bills the API instead of the
   subscription. Unset it; verify with `/status`.
2. `--bare` mode "does not read `CLAUDE_CODE_OAUTH_TOKEN`" and never touches OAuth
   credentials or the keychain. Bare mode is API-key-only, so we run non-bare.
3. Managed Agents (the hosted agent platform) requires a Console API key and bills as
   Developer Platform usage. Not reachable from the subscription. Not needed here.

**Windows note.** Credentials live at `%USERPROFILE%\.claude\.credentials.json`.

Sources: [Agent SDK + Claude plan](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) ·
[Authentication](https://code.claude.com/docs/en/authentication) ·
[Headless](https://code.claude.com/docs/en/headless) ·
[Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview)

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Harness | Claude Agent SDK (Python), `ClaudeSDKClient` | Multi-turn loop, subscription auth, hooks, full system-prompt replacement |
| Auth | `CLAUDE_CODE_OAUTH_TOKEN` via `claude setup-token` | Subscription-backed; no API key |
| Tool surface | Built-ins only: Read, Write, Edit, Bash, Glob, Grep | pi-style minimalism |
| Tool restriction | `disallowed_tools` + `PreToolUse` hook + a deny-by-default permission mode | `allowed_tools` alone does **not** restrict anything — see §9 |
| Custom capability | Exposed as **CLIs invoked over Bash**, not MCP servers | Composable, portable, debuggable — see §8 |
| CLI contract | `--json` on every command, distinct exit codes, actionable errors | A CLI's failure mode is a stack trace the model retries blindly against — see §8 |
| Documentation channel | `--help` on each CLI + on-demand skills | Progressive disclosure |
| Discovery retrieval | Semantic Scholar / Ai2 Asta over HTTP | 200M+ papers, full-text snippets, free. Cannot be beaten locally |
| Local retrieval | SQLite FTS5 + `sqlite-vec`, papers read + own notes | Right scale for hundreds–thousands of docs; one file, one dependency |
| Hybrid fusion | Reciprocal rank fusion (k=60) over FTS5 and vector rankings | Score-free, needs no calibration between two incomparable scales |
| Tier-2 embeddings | Hosted Voyage (`voyage-4`) via the Voyage API | Same no-VRAM-contention logic as the reranker; model+version recorded in the index; `voyage-4-nano` local as fallback — see §5 |
| Reranker | `voyageai/rerank-2.5` via OpenRouter | Hosted; no VRAM contention with experiments; costs credits, not quota. Verified: OpenRouter serves this via a dedicated rerank endpoint |
| Query expansion + triage | Haiku 4.5 via the **Agent SDK** (`ClaudeAgentOptions(model=...)`) | Subscription-backed, native Python; widens the funnel past what the main loop can read |
| Haiku structured output | A single forced in-process SDK tool, not free-text JSON | Schema validation at the tool layer; retry handled by the SDK — see §5 |
| Direct Messages API calls | **Not used** | The `anthropic` SDK bills Developer Platform, not the subscription — see §2 and §5 |
| Notebook | **Jupyter** with a persistent kernel | Models are heavily trained on Jupyter; `.ipynb` is the format they expect |
| Kernel discipline | Wall-clock timeout on `nb.py exec`; long work goes to `jobs.py` | The kernel is the agent's only compute channel; a training cell blocks it |
| Math verification | **SymPy + mpmath in-kernel**, not SageMath | Zero new infrastructure; Sage's extra power is not what pipeline QA needs (§6) |
| Pre-flight gate | Mechanically enforced by `jobs.py` / `gpu.py`, not by prompt | The model cannot skip a check that the submitter refuses to run without |
| Preflight validity | Hash of the *resolved submission*, no time-based TTL | Time is not the risk; state change is — see §6 |
| Expectations ledger | Append-only JSONL (truth) + rebuildable SQLite (index) | Human-diffable, greppable, no migrations, still queryable |
| Result collection | `jobs.py collect` polls, fetches artifacts, writes the run record | A loop closed by the model is a loop that opens under pressure |
| Subagents | **Not used** for research; funnel stages 0 and 3 are the sole exception, and log their full I/O | Token cost and loss of observability; pi omits them deliberately |
| MCP servers | **None** as external servers; Asta's HTTP endpoint called from a CLI; one in-process SDK tool for Haiku triage | See §8 |
| Remote compute | Through `gpu.py` / `jobs.py` only; bare `ssh`/`hf` denied | See §9 |
| Credential storage | Windows Credential Manager via `keyring`; read only by `gpu.py` / `jobs.py` | A hook can be argued around; a token that is not in the environment cannot — see §9 |
| Interface | **NiceGUI**, single Python process, `native=True` desktop window | Async-native, one language, native ECharts/Plotly/Quasar tables for the widgets that matter — see §10 |

### Rejected, and why

- **marimo as host.** Reactive re-execution is hostile to long-running training cells, and
  models have far more Jupyter than marimo in their training data. Revisit only if
  notebook-as-plain-Python becomes a priority.
- **Managed Agents.** API-key-only billing; hosted sandboxes solve a problem we don't have.
- **Bulk local index of arXiv.** Superseded by the two-tier design in §5. A local index can
  only return what was already ingested, which makes it structurally useless for discovery.
- **LanceDB.** Premature at this corpus size. Revisit past ~100k chunks.
- **SageMath.** See §6. Deferred, not rejected.
- **Local embedding/reranking on the desktop GPU.** Competes for the same VRAM as the
  experiments the agent exists to run. `voyage-4-nano` (Apache-2.0, open-weight) is the
  fallback if hosted access ever becomes a problem.
- **Paid literature databases (Elicit, Undermind).** See §5.
- **GEPA prompt optimization.** Deferred, not rejected. See §11.
- **Building a notebook editor in the UI.** JupyterLab already exists and is better. The UI
  renders notebook *output*; editing links out to Lab. See §10.
- **Streamlit as the frontend.** Its rerun-per-interaction model fights a long-lived async
  agent session and a persistent kernel. See §10.
- **A separate React/Vite frontend.** Considered and not chosen: it wins on math rendering
  and streaming control and loses on the plot-heavy widgets, a second language, and a build
  step. NiceGUI instead, with named exit criteria in §10 rather than a standing option.
- **Electron / Tauri packaging.** `ui.run(native=True)` covers it. See §10.

---

## 4. Architecture

```
GradientAgent/
  agent.py            # ClaudeSDKClient loop, system prompt, hooks
  hooks.py            # PreToolUse gate: denies bare ssh/scp/hf/rm -rf
  prompts/
    system.md         # target: under 1000 tokens
  tools/
    paper_search.py   # CLI: the §5 funnel — expand, retrieve, rerank, triage
    paper_ingest.py   # CLI: arXiv id -> LaTeX source -> chunks -> local index
    nb.py             # CLI: exec (timeout-bounded) / verify (fresh kernel) / restart
    preflight.py      # CLI: run the QA gate, emit preflight.json
    jobs.py           # CLI: submit / status / collect for HF Jobs
    gpu.py            # CLI: submit / status / collect for known SSH hosts
    ledger.py         # CLI: append/query expectations, runs, and outcomes
    quota.py          # CLI: read the token/spend log, summarise by stage
  ui/
    app.py            # NiceGUI entrypoint: session view, streaming transcript
    widgets/          # preflight panel, expectation plot, quota meter, funnel view
    katex.py          # KaTeX head injection + render hook (see §10)
  skills/
    remote-gpu/       # SSH conventions, host inventory, job submission patterns
    hf-jobs/          # Hugging Face Jobs patterns
    paper-corpus/     # how the index is organised, search syntax
    preflight/        # what each check means, how to fix a failing one
  data/
    corpus.sqlite     # FTS5 + sqlite-vec over read papers and own notes
    papers/           # downloaded LaTeX sources and PDFs
  ledger/
    expectations.jsonl  # append-only, source of truth
    runs.jsonl          # append-only, source of truth
    preflight/          # one JSON per verified submission hash, plus logs
    quota.jsonl         # append-only token/credit spend, tagged by stage
    ledger.sqlite       # derived index, rebuildable from the JSONL at any time
  notes/              # markdown research log; the agent appends, greps, and cites it
  notebooks/          # .ipynb the agent creates and edits
  figures/            # kernel output images, referenced by path
  evals/
    retrieval.jsonl   # 40-60 query -> expected-paper pairs, harvested from real use
```

The agent's standing context is: a short system prompt, six built-in tool schemas, and a
one-line mention of each CLI. Everything else loads on demand — `--help` when the agent needs
an interface, a skill when it needs a workflow.

---

## 5. Retrieval: two tiers

Discovery and recall are different problems and were previously conflated.

**Tier 1 — discovery (hosted, free).** Finding papers not yet read. A local index cannot do
this by construction. Use Semantic Scholar's Academic Graph API directly:

- `/graph/v1/paper/search` — relevance search over ~108M abstracts.
- `/graph/v1/snippet/search` — **full-text** search over ~12M papers, returning ~500-word
  excerpts. This is the high-value endpoint and the reason to prefer S2 over arXiv's API.
- `/graph/v1/paper/{id}/citations` and `/references` — citation-graph expansion, which is
  worth more for recall than any reranker upgrade.

Ai2's **Asta** wraps the same corpus with better-tuned tooling at
`https://asta-tools.allen.ai/mcp/v1`. It is an MCP endpoint, but it speaks streamable HTTP,
so `paper_search.py` can call it over plain HTTP without adopting MCP as an architecture.
Free; an API key raises rate limits. Unauthenticated S2 is ~1 req/s, so cache aggressively.

**Tier 2 — recall (local, small).** Papers actually read, their LaTeX source, and the user's
own notes and derivations. This is for "where did I see that lemma," which no external index
can answer. SQLite FTS5 + `sqlite-vec` in a single file. Ingest from arXiv **LaTeX source,
not PDF** — this is the single largest quality lever in the retrieval stack, because it
preserves equations, theorem environments, and section structure that PDF extraction destroys.

The vector side needs an embedding model, and it is the same decision as the reranker:
hosted Voyage (`voyage-4`), because local embedding competes for the same VRAM as the
experiments (§3). The model name and version are recorded in `corpus.sqlite`;
`paper_ingest.py` refuses to add vectors from a model other than the one the index was
built with, so a model change means a deliberate re-embed of the corpus, never a silent mix
of incompatible vector spaces. Stage 0's HyDE passage is embedded with the same model — a
HyDE vector from a different embedding space is noise, not signal.

The two local rankings are combined by **reciprocal rank fusion** (`1/(k + rank)`, k=60,
summed across the FTS5 and vector rankings). RRF is used rather than a weighted score blend
because BM25 scores and cosine similarities are on incomparable scales and calibrating them
per-corpus is exactly the kind of tuning work this design is trying to avoid.

### The funnel

Five stages. The ordering is deliberate: each stage is cheaper per candidate than the one
after it, so the expensive stages only ever see filtered input.

| # | Stage | Mechanism | Cost |
|---|---|---|---|
| 0 | Query expansion | Haiku: 1 question → ~5 keyword queries for S2, plus 1 HyDE abstract for the local index only | quota |
| 1 | Retrieve | S2 snippets + local index (RRF) + citation expansion → 200–400 candidates | free |
| 2 | Rerank | `voyageai/rerank-2.5` → top ~50 | credits |
| 3 | Triage | Haiku reads all 50 in one call, returns ~15 with a one-line reason each | quota |
| 4 | Select | The main agent reads the 15 | quota (main window) |

**Expansion is retriever-specific, and this is easy to get wrong.** HyDE works by generating
a hypothetical answer whose *embedding* lands near the real answer's embedding — the gain is
a dense-retrieval gain. S2's snippet endpoint is lexical/hybrid, and feeding it a synthetic
abstract mostly dilutes the query terms. So stage 0 emits two different things: a set of
keyword/phrase queries aimed at S2, and a single HyDE passage aimed only at the `sqlite-vec`
side of tier 2.

**Why both stage 2 and stage 3.** They are not redundant. The reranker is calibrated — its
scores are comparable across queries, and it is fast and quota-free. Haiku is not calibrated
and has listwise position bias, but it can judge relevance *against the actual research
question* rather than against a query string. Keeping the reranker in the middle means the
quota-consuming stage never sees the 350 candidates that were obviously wrong.

**Stage 3 is a funnel widener, not a better ranker.** That is the whole argument for it. The
main agent can afford to read ~15 snippets; Haiku can afford 50. Since the retriever sets the
ceiling (below), widening the funnel is where the gains are.

**Both Haiku stages go through the Agent SDK, not the `anthropic` SDK.** This matters and is
easy to get wrong. `client.messages.create()` in the `anthropic` package resolves
`ANTHROPIC_API_KEY` → `ANTHROPIC_AUTH_TOKEN` → an `ant auth login` console profile — all
Developer Platform credentials that bill per token. The subscription-backed path is
`claude_agent_sdk`.

**Structured output without the Messages API.** `output_config.format` is a Messages API
feature and is not available here. Prompting for JSON and parsing it is the obvious fallback
and the wrong one — it fails silently on the tenth call, mid-funnel. Instead, register a
single in-process SDK tool and make it the only tool the triage call may use. The model is
then forced to emit a schema-validated payload, and the SDK handles the retry when it
doesn't. §8's cost argument against MCP does not apply: this is one tool, in-process, on a
call whose entire purpose is to return one structured object.

```python
from claude_agent_sdk import query, ClaudeAgentOptions, create_sdk_mcp_server, tool

captured = []   # the handler is a sink; the payload never round-trips through text

@tool("submit_triage", "Return the triage verdict for every candidate",
      {"verdicts": list})   # [{id: str, keep: bool, reason: str}]
async def submit_triage(args):
    captured.append(args["verdicts"])
    return {"content": [{"type": "text", "text": "recorded"}]}

opts = ClaudeAgentOptions(
    model="claude-haiku-4-5",
    system_prompt=TRIAGE_PROMPT,   # "call submit_triage exactly once, then stop"
    mcp_servers={"triage": create_sdk_mcp_server("triage", tools=[submit_triage])},
    allowed_tools=["mcp__triage__submit_triage"],
    disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
)
```

Sketch, not verified code — check the `@tool` schema format and result shape against the
installed `claude-agent-sdk` version before relying on it. The structural point stands
regardless: the payload arrives as validated tool input, not as text to be parsed.

Two hardening details the sketch glosses over, both cheap. First, `{"verdicts": list}`
validates that a list arrived, not the shape of its items — pass a full JSON Schema for the
item shape if the installed SDK supports it, and either way validate items in the handler
and return an error result on a malformed payload, since a returned error is what actually
triggers the model to retry. Second, the model can end the turn without calling the tool at
all, leaving `captured` empty — `paper_search.py` checks for that and retries the call once
before failing the stage loudly.

Three constraints that follow:

- **Quota, not dollars.** The Agent SDK drives the Claude Code CLI underneath, and it draws on
  the same Max 5x window as interactive use (§2). This is why stage 2 stays quota-free.
- **These are subagents, and §3 says we don't use subagents.** Stages 0 and 3 are the
  deliberate exception, so they carry the mitigation the general rule exists to avoid losing:
  every stage-0 and stage-3 call appends its full prompt, raw response, and token counts to
  `ledger/quota.jsonl` and a per-query log under `notes/`. Debugging a funnel whose middle is
  invisible is guesswork.
- **Don't call `set_model()` on the main client.** It exists, but prompt caches are
  model-scoped — switching the main agent to Haiku and back discards its cached prefix. Use a
  separate `query()` call.

Stage 3's per-candidate reason is not decoration: it is the provenance that populates the
ledger's `basis` field (§7).

### Two facts to keep in mind when tuning this

- **The retriever sets the ceiling.** Independent benchmarking finds no reranker pushes
  Hit@10 past ~88%, because the missing documents never entered the candidate set. Multi-query
  rewriting and citation-graph expansion buy more than reranker shopping does.
- **Free corpora are sufficient here.** Elicit (API since March 2026, ~$12/mo) and Undermind
  are good products, but they wrap comparable underlying data and sell report generation —
  which is the part this agent exists to do. Undermind additionally has no API and takes
  8–10 minutes per query, which makes it a fine human tool and a bad agent component.

### Evaluating the funnel honestly

`evals/retrieval.jsonl` is the arbiter for any change to this stack, including whether stages
0 and 3 earn their quota. Two things about it were previously understated:

**Size.** 20–30 query→paper pairs cannot separate rerank-only from rerank+triage. At n=25 a
Hit@10 difference needs to be roughly 15–20 points before it clears the noise, and the
differences here will not be that large. Target **40–60 queries**, and report **Recall@50**
and graded relevance alongside Hit@10 — Recall@50 has several relevant documents per query
instead of a single binary hit, so it extracts far more signal from the same labelling
effort. Where the confidence intervals overlap, say so and keep the cheaper configuration;
"no measurable difference" is a valid and common result, and it favours dropping a stage.

**Provenance.** Do not author the eval set cold. Step 3 of §12 is a week of real use with a
written record of what retrieval was actually reached for; the eval set is harvested from
that log. A benchmark of imagined queries measures the imagination.

---

## 6. The pre-flight gate

**No job that costs money runs until a machine-checkable artifact says it will work.**

The gate exists because remote-job failures are overwhelmingly boring — a shape mismatch, a
missing dependency, a bad path, a config that OOMs at step 0 — and they are boring at the
price of a GPU-hour and an hour of wall clock. Every one of them is catchable in advance for
cents.

### Enforcement

`preflight.py` runs the checks and writes `ledger/preflight/<submission_hash>.json`.
`jobs.py` and `gpu.py` refuse to submit unless four things hold:

1. A **passing preflight record** exists for the current submission hash (below).
2. An **open expectation** exists for this task in `ledger/expectations.jsonl` — no
   pre-registration, no submission (§7).
3. The **rolling spend total** plus this job's estimate stays under the monthly ceiling.
   The total counts collected runs at their actual cost and **in-flight runs at their
   estimates** — a job that has not been collected yet is not free. Without this, N jobs
   submitted before any is collected all pass the ceiling check.
4. **No stale uncollected run exists.** A run still uncollected well past its estimated
   duration (estimate plus a generous grace factor) blocks new submissions until
   `collect` is run. Spend only becomes "actual" at collect time; if collection were
   optional, the ceiling would be too (§7).

The hook in §9 denies bare `hf` and `ssh`, so there is no path around the submitter.

This is deliberate: a gate that lives in the system prompt is a gate the model will skip when
it is three steps into a plan and confident. A gate that lives in the submitter is not.

### What the hash covers, and why there is no TTL

The previous design hashed "the pipeline directory" and expired the record after 24 hours.
Both were wrong, in opposite directions.

A directory hash is simultaneously **too broad** — touching a note or writing a figure
invalidates a perfectly valid preflight, and a gate that fires spuriously is a gate that gets
argued around — and **too narrow**, because the most common real change is a config edit with
identical code, and configs do not always live under the pipeline directory.

A 24-hour TTL is a proxy for a risk that is not time. Nothing about a preflight record decays
by sitting still. What invalidates it is state change, and state change is exactly what a
hash is for. So: no TTL, and the hash covers the **resolved submission** —

- the entrypoint and every first-party module it imports (resolved by import graph, not by
  directory glob),
- the fully resolved config *after* CLI overrides are applied, serialised canonically,
- the dependency lock file,
- the dataset pointer and its revision/commit,
- the container image **digest**, not its tag — `:latest` is how remote environment drift
  sneaks past a hash that otherwise looks airtight,
- the entrypoint argv.

Anything not in that list cannot change the outcome of the job, and anything in it
invalidates the record immediately rather than at some arbitrary hour. Re-running preflight
after a real change costs about two minutes, which is the number that keeps the gate from
being resented.

Two known limits of the import-graph approach, handled explicitly rather than silently:
dynamic imports are invisible to static resolution, and files the pipeline loads at runtime
outside the config system — tokenizer files, data manifests, a hand-read YAML — are reached
by neither the import graph nor the resolved config. The pipeline config may declare
`extra_hash_paths`, which preflight folds into the hash. A runtime-loaded file not listed
there is a documented gap, not a silent guarantee.

### The checks

| Check | What it catches | Tool |
|---|---|---|
| `tests` | Regressions in pipeline code | `pytest` on the pipeline's unit tests |
| `dry_run` (local) | Logic errors, shape errors, config typos | Same entrypoint, 1 step, batch 2, 10 samples, tiny model, CPU or local GPU |
| `smoke` (remote) | Everything local cannot see — see below | Same entrypoint, 1 step, on the real target, at minimum billing granularity |
| `shapes` | Tensor rank/axis errors before compute | Named axes (`einops` / `jaxtyping`) asserted on the dry run |
| `grads` | Wrong hand-written gradients or custom autograd | `torch.autograd.gradcheck` against finite differences |
| `symbolic` | Implementation drifting from the derivation | SymPy — see below |
| `invariants` | Silent correctness bugs | Property tests (`hypothesis`): seed determinism, equivariance, loss ≥ 0, distributions normalize |
| `cost` | Surprise bills, this job and cumulatively | Estimated GPU-hours × rate vs a per-job ceiling, **and** rolling 30-day actual spend from `runs.jsonl` vs a monthly ceiling |

### The local dry run is not enough, and the previous version overclaimed

The earlier draft said the dry run catches "~most real job failures" and listed "a missing
dependency, a bad path, a config that OOMs at step 0" as examples. Those three are precisely
what a local CPU run with a tiny model **cannot** catch. A local dry run proves the code is
internally coherent. It says nothing about:

- the remote image missing a package the local venv happens to have,
- CUDA / driver / torch build mismatch on the target,
- distributed or multi-GPU initialisation, which is single-process locally,
- dataset not staged, or staged at a different path, on the remote filesystem,
- token or credential scope on the remote,
- OOM at the real batch size and real model size, which a tiny model by construction hides.

So the gate has **two dry runs, not one**:

**`dry_run` (local)** — seconds, free, run on every change. The fast filter.

**`smoke` (remote)** — the same entrypoint, one step, real model size for one micro-batch,
real image, real data path, on the actual target host or HF Jobs, at the minimum billing
granularity the platform allows. This costs cents and a couple of minutes. It is the single
highest-value check in the table, because it is the only one that exercises the environment
the real job will run in. Its result is recorded into the same preflight record, so the
submission hash covers it too — which is why the image **digest** is in the hash.

If only one check ever runs, it is `smoke`. If two, `dry_run` then `smoke`.

Batch-size OOM specifically deserves its own note: run the smoke job at the real per-device
batch size with a truncated sequence count rather than at batch 2, or it does not test the
thing that most often kills the real run.

### How smoke escapes its own gate

Smoke has a bootstrap problem the other checks do not: it is itself a paid remote job, it
must go through `jobs.py` / `gpu.py` (the §9 hook denies bare `ssh` and `hf`), and it runs
*before* a passing preflight record or a bound expectation can exist for the submission it
is validating. Left unspecified, this gets resolved ad hoc, and an ad-hoc bypass is the hole
in the fence. So the carve-out is explicit: `jobs.py submit --smoke` and `gpu.py submit
--smoke` are exempt from the preflight and expectation gates but hard-capped by the
submitter itself — one step, a wall-clock ceiling of a few minutes, a cost ceiling of cents,
no artifact upload — all enforced in code, not prompt. Smoke spend still lands in
`runs.jsonl` and counts toward the monthly ceiling. The result is written into the pending
preflight record for the submission hash. The caps are what keep the exemption from becoming
the way real jobs escape the gate: nothing useful can be trained inside them.

### Other mechanical gates

These are not about money, but they follow the same rule — enforced by a program, not a
prompt.

**`nb.py verify`.** A persistent kernel plus an agent that edits cells in place produces
notebooks that work live and fail on a clean run. The previous plan handled this with a
system-prompt line, which is the weak form by this document's own argument. Instead:
`nb.py verify <notebook>` restarts the kernel, executes every cell top to bottom, and exits
non-zero on the first failure. Run it before any notebook is cited in `notes/` or referenced
from a ledger entry.

**`nb.py exec` is timeout-bounded.** The kernel is the agent's only interactive compute
channel; a training loop in a cell blocks it indefinitely with no way to observe progress.
`exec` takes a wall-clock timeout (default a few minutes), and exceeding it is an error whose
message says to move the work to `jobs.py`. The kernel is for exploration; anything long is a
job.

### On symbolic math (why SymPy, not SageMath)

The genuinely valuable case for a CAS here is narrow but real: *you derived an update rule,
a bound, or a normalization constant on paper, and you want to confirm the code implements
that and not something adjacent.* SymPy handles this well — differentiate the loss
symbolically and compare against the implemented gradient; verify a schedule's closed form;
check that a density integrates to 1; confirm an algebraic simplification the code assumes.

SageMath is a multi-gigabyte install that on Windows realistically means WSL, and the power
it adds over SymPy (exact linear algebra over rings, algebraic geometry, serious number
theory) is not what verifies a training pipeline. Add it later if the *mathematics* being
researched needs it — not for QA. `mpmath` covers the high-precision numeric cases in the
meantime.

The honest caveat: most pipeline bugs are not symbolic-math bugs. Shape errors, dtype
promotion, off-by-one schedule indices, and environment drift dominate, and `dry_run`,
`shapes`, and `smoke` catch those far better than a CAS does. Weight the checks accordingly.

---

## 7. The expectations ledger

**Predict before you run. Record the prediction. Compare. Keep both.**

Before submitting an expensive job, the agent searches for the closest prior work, extracts
what those papers report for the quantity being measured, and writes a *pre-registered*
expectation. After the run, actual values are compared against it and the deviation is
recorded with a verdict.

Two things this buys, in order of importance:

1. **A surprise is an alarm.** A result far outside the predicted range is a bug hypothesis
   first and a discovery second. This is the check that stops "my method beats the baseline
   by 40%" from being celebrated instead of debugged.
2. **Cross-session memory.** A later session working on a similar task queries the ledger
   and starts from what was already learned instead of re-deriving it.

### Pre-registration is enforced by the submitter, not by the ledger

The previous design had `ledger.py expect` refuse to write once a run record with results
existed for that task. That is not enforcement. The model chooses the task id, so the
rationalisation path is one line long: run the job, read the logs, write the expectation
under a fresh task id, and the ledger records a confident prediction that was authored after
the fact. Nothing in the system would know.

The gate belongs where the preflight gate is:

- `jobs.py submit` and `gpu.py submit` **require** `--expect <expectation_id>`, and refuse if
  that id does not exist or is already bound to a run. The expectation is bound to the run
  at submit time, and the submitter writes the run record — in-flight status, bound
  expectation id, cost estimate — into `runs.jsonl` at that moment, not at collect time.
  This submit-time record is also what lets §6's spend ceiling count in-flight jobs.
- `ledger.py expect` still refuses to write against a task with results, which now serves as
  a second, weaker line rather than the only one.
- The expectation id is minted before submission and cannot be created retroactively for an
  existing run, because the run record already names the id it was submitted with.

This makes the two gates consistent: both live in the submitter, both are unskippable, and
both fail loudly with an actionable message.

An earlier revision also refused expectations created *after* the submission's preflight
record. Dropped. Binding-at-submit already closes the post-hoc path — results only come into
existence through `submit`, and the run record names the expectation it was submitted with,
so an expectation minted later can never claim an existing run. All the ordering rule
prevented beyond that was a prediction informed by the smoke run's single step, which leaks
almost nothing; what it cost was real — it refused the natural
preflight-passes-then-write-the-prediction workflow and forced a paid preflight re-run
purely to fix timestamps. A gate that fires spuriously is a gate that gets argued around, by
this document's own argument in §6.

### Results are collected by a program, never hand-written

The previous draft specified the run record's contents but not who writes them. In practice
that means the model writes them — from memory, at the end of a long session, after reading
scrolled-back logs. That is the failure mode this whole document is built to avoid, and it
silently voids the comparison step that justifies the ledger's existence.

So `jobs.py collect <run_id>` (and `gpu.py collect`) does it:

- reports the job's state and exits non-zero if it is still running — never blocking by
  default, because a two-hour poll inside the agent's only shell is a tool timeout waiting
  to happen; `--wait --timeout <seconds>` is the explicit opt-in,
- pulls stdout/stderr, the metrics file, and any artifacts into `ledger/runs/<run_id>/`,
- parses declared quantities out of the metrics file — the pipeline is required to emit a
  machine-readable metrics artifact (one JSON per eval, or a JSONL of scalar records), which
  is a cheap contract that removes all log-scraping,
- computes actual cost from the platform's own accounting on HF Jobs; SSH hosts have no
  billing API, so `gpu.py` prices wall-clock against a per-host rate in its inventory
  (rate 0 for hosts that are free to use) — never from the estimate,
- writes the completed run record to `runs.jsonl` including the `deviations` array, computed
  mechanically against the bound expectation,
- and leaves the **verdict** field (`bug | real | inconclusive`) unset. That is the one part
  requiring judgement, and `ledger.py verdict <run_id> --quantity ... --verdict ... --note ...`
  is how the model supplies it. Unset verdicts on out-of-range deviations are surfaced in the
  UI (§10) and by `ledger.py query --pending`, so they cannot quietly accumulate.

The division is the point: the machine records what happened, the model interprets it, and
the interpretation cannot overwrite the record.

One loop remains model-initiated: running `collect` at all. By this document's logic that is
a loop that opens under pressure, so the backstop is mechanical and lives in §6's gate — the
submit-time record marks the run in-flight at its estimate, the spend ceiling counts
in-flight runs, and a stale uncollected run blocks new submissions. Forgetting to collect
therefore costs the ability to submit, which is the one currency that reliably gets noticed.
`ledger.py query --pending` and the UI list uncollected runs alongside unjudged deviations.

### Records

**Expectation** (`ledger/expectations.jsonl`)

```json
{"id": "...", "task": "...", "created_at": "...",
 "quantity": "val_loss@1e9_tokens",
 "claim": "val loss should land between 2.9 and 3.2",
 "predicted": {"low": 2.9, "high": 3.2, "direction": null},
 "basis": [{"paper": "arXiv:xxxx.xxxxx", "locator": "Table 3, row 2",
            "value": 3.05, "conditions": "1.3B params, 100B tokens, SP tokenizer"}],
 "comparability": "our tokenizer differs; our eval set is a 5k held-out subset",
 "confidence": "medium"}
```

The `comparability` field is not optional decoration — it is the field that prevents the
whole system from generating confident nonsense. A number from a paper means nothing without
matching tokenizer, dataset, eval protocol, sequence length, and parameter count.

**For that reason, prefer relational expectations over absolute ones.** "Loss should drop
~X% when width doubles," "A should beat B on the same eval," "the curve should be roughly
power-law with exponent in [a, b]" all survive setup mismatch. Absolute numbers require a
populated `comparability` field to be recorded at all.

**Run** (`ledger/runs.jsonl`) — submission hash, config, command, preflight pointer, bound
expectation id, cost estimate and actual, results by quantity, and a `deviations` array
linking back to expectation ids with `{expected, actual, ratio, verdict, note}`. Written by
`collect`, except `verdict`/`note`.

### Storage

Append-only JSONL is the source of truth: human-diffable, greppable, survives every schema
change, and never needs a migration. `ledger.sqlite` is a derived index for structured
queries (`ledger.py query --quantity val_loss --model gpt2-small`) and is rebuildable from
the JSONL at any time. If they ever disagree, the JSONL wins.

The ledger files are multi-writer: the CLIs, the funnel stages inside `paper_search.py`, and
the Stop hook all append, while the UI reads concurrently — and Windows is less forgiving
about concurrent file access than POSIX (interleaved partial lines and sharing violations
are real outcomes, not theory). So there is exactly one write path: a shared
`append(path, record)` helper that every CLI imports, taking an exclusive lock
(`portalocker` / `msvcrt.locking`) around each line write. Readers tolerate a torn final
line. No CLI opens a ledger file for writing directly.

Every basis entry carries provenance (paper id + locator). Entries later shown wrong get
marked falsified rather than deleted — a wrong prediction with a recorded correction is more
useful to a future session than a gap.

### Failure mode to watch

This system's risk is accumulating low-quality entries that then poison future sessions. The
mitigations are: mandatory provenance, the `confidence` field, mandatory `comparability` for
absolute claims, `--pending` surfacing unjudged deviations, and the fact that the files are
small enough to read by hand. Review the ledger periodically. If it stops being worth
reading, it has stopped being worth writing.

---

## 8. CLIs, not MCP — the reasoning

"MCP" conflates two different things, and only one of them is expensive.

**External MCP servers** carry a real cost: every tool's schema sits in context on every
request, whether used or not. pi's measurement was 7–9% of context consumed by servers like
Playwright for tools that mostly go unused.

**The SDK's in-process mechanism** (`create_sdk_mcp_server`) is not that — it registers a
Python function as a tool, no subprocess, at roughly 100–200 tokens of schema each. This is
why §5's triage call uses one: on a subagent call whose entire job is to return a structured
object, a forced tool is strictly better than parsing free text, and the context cost lands
on a Haiku call rather than the main window.

For the main agent we use CLIs over Bash instead, for three reasons:

- **Composability.** `paper-search "..." | head -3`, loops, pipes into scripts. A tool call
  returns a blob; a CLI participates in the shell.
- **Portability.** The same CLIs work under the Agent SDK, `claude -p`, pi, or a human at a
  terminal. No harness lock-in.
- **Debuggability.** You can run exactly what the agent ran, which matters enormously when
  the thing being debugged is a job that cost $40.

**What this does not buy: zero token cost.** That argument was previously listed first and is
the weakest of the set. In practice `system.md` carries a compact usage line per CLI so the
agent knows when to reach for one, which costs roughly what a tool schema costs — and models
compose an unfamiliar CLI less reliably than they fill a schema. Accept the `--help` round
trip; take the three real wins above.

### The cost that makes or breaks this choice: error surface

A failed tool call returns a structured error the model can act on. A failed CLI returns a
stack trace on stderr and an exit code of 1, and the characteristic model response to that is
to retry with guessed flags — which burns quota and, worse, sometimes succeeds at doing the
wrong thing. This is the real tax on the CLI decision, and it is payable up front.

Every CLI in `tools/` therefore obeys the same contract:

- **`--json` on every subcommand**, emitting a stable envelope
  (`{"ok": bool, "data": ..., "error": {"code": ..., "message": ..., "fix": ...}}`) on stdout.
  Human-readable output goes to stderr or behind a flag; the agent always uses `--json`.
- **Distinct exit codes** with documented meanings — a usage error, a gate refusal, and an
  upstream failure are three different things and the model should not have to read prose to
  tell them apart. Gate refusals in particular (`preflight missing`, `no open expectation`,
  `spend ceiling exceeded`) get their own codes.
- **Errors state the fix, not just the fault.** `error.fix` is a literal next command where
  one exists: "run `preflight.py run --submission <hash>`". Never a bare traceback.
- **Unknown flags fail fast** with the closest valid flag named, rather than being ignored.

Without this, the portability and debuggability wins are real but the agent-facing ergonomics
are worse than tool schemas, and the decision does not hold up.

### Images

The one thing tools genuinely do better is returning images inline. The workaround is
adequate: the kernel saves to `figures/NNN.png`, the CLI prints the path, the agent calls
Read (which handles images). Two calls instead of one. The UI (§10) renders the figure
directly from that path, so the human cost is zero.

**Caveat to watch.** The context argument only holds if the always-on surface stays small. A
2000-token skill loaded every session is worse than three tool schemas. Keep skill *bodies*
out of the default context and let `--help` carry interface documentation.

---

## 9. Permissions

Previously listed as the largest open risk. It is now a decision.

**The correction that forced this:** in the Agent SDK, `allowed_tools` is an *auto-approve*
list, not a sandbox. Built-in tools remain in the model's toolset regardless of what is
listed, and `allowed_tools=["Read"]` together with `permission_mode="bypassPermissions"`
still approves Bash. The evaluation order is: PreToolUse hook → deny rules → allow rules →
ask rules → permission mode → `can_use_tool` → PostToolUse. Note that `can_use_tool` fires
*only* when evaluation resolves to a prompt, so it cannot be used as a universal gate.

**The configuration:**

- A **deny-by-default permission mode**, so anything not pre-approved is denied rather than
  prompted. This is a headless agent; there is nobody to answer a prompt.
- `disallowed_tools` for anything genuinely unwanted, since deny rules beat every other step.
- A `PreToolUse` hook matching `Bash` that denies bare `ssh`, `scp`, `hf`, `rm -rf`, and
  piped-curl-to-shell.

**Verify the mode empirically before trusting it.** The whole safety story rests on the exact
name and semantics of that mode in the installed SDK version (`dontAsk` at time of writing),
and mode names and behaviour have changed between releases. Step 1 of §12 includes a
deliberate probe: attempt a tool call that should be denied and confirm it is *denied*, not
prompted and not silently allowed. Do not take this document's word for it, and re-run the
probe after any SDK upgrade.

Three details from the current permissions docs, folded in so the probe covers them too: a
bare-name deny rule (e.g. `disallowed_tools: ["WebSearch"]`) removes the tool from the
model's context entirely rather than denying it at call time — which is the desired
behavior; in `dontAsk` mode `can_use_tool` is consulted only for tools already on the allow
list, so it cannot serve as a universal gate (consistent with the evaluation-order note
above); and the SDK does not load `settings.json` permission rules unless `setting_sources`
is set — keep the whole permission configuration in code, so a stray settings file cannot
add allow rules silently.

**Why command-string matching is not the security model.** Regexing shell commands is
defeated by `ssh host "cmd"`, `bash -c`, `$(...)`, aliases, and environment indirection. The
hook is a speed bump, not a wall. The actual control is architectural: the agent does not
have a general remote-execution capability. It has `gpu.py` and `jobs.py`, which carry a
hardcoded host inventory, a per-invocation spend ceiling **and a rolling monthly ceiling**
(§6 — a per-invocation cap alone does not stop twenty invocations), no destructive verbs, and
the preflight and pre-registration requirements from §6 and §7. A small allowlist over our
own CLIs is enforceable in a way that a large allowlist over a shell is not.

**Credentials are what make that claim true.** With unrestricted Bash, network access, and
an `HF_TOKEN` sitting in the environment, the agent *does* have general remote execution —
`python -c "import huggingface_hub; ..."` never contains the string `hf` and sails past the
hook. So the HF token and the SSH keys live in Windows Credential Manager, fetched via
`keyring` by `gpu.py` and `jobs.py` at the moment of use — never exported into the agent's
environment, never written to a file under the workspace, never in a `.env`. The honest
residual: a model determined to misbehave could call `keyring` from Python itself. The
threat model here is accidental or deadline-pressured spend, not an adversarial model, and
credential isolation is the right bar for that — it also means the spend ceilings guard the
only path that can actually authenticate.

---

## 10. The interface

The agent is a research instrument used daily by one person. That makes interface quality a
real requirement rather than a nicety: the things that make it pleasant — being able to see a
funnel's reasoning, a preflight's failing check, a prediction against its outcome — are the
same things that make it trustworthy.

**Decision: NiceGUI.** One Python process, no separate frontend build, no TypeScript. The
three properties that decided it:

- **Async-native.** NiceGUI is FastAPI + Quasar over a WebSocket, with `async def` handlers
  and a real event loop — `ClaudeSDKClient`'s async streaming maps onto it directly, with no
  bridging layer. This is also what disqualifies Streamlit, whose rerun-per-interaction model
  fights a long-lived session and a persistent kernel.
- **The plot widgets are easier here, not harder.** `ui.echart`, `ui.plotly`, `ui.pyplot`,
  and Quasar's `ui.table` are built in. Widgets 2 and 3 below — expectation-vs-outcome and
  the quota meter — are less work in NiceGUI than in a React app, and the funnel view is a
  dense sortable table, which `ui.table` already is. An earlier draft of this section argued
  the opposite; it was wrong about where the custom-widget effort actually lands.
- **`ui.run(native=True)`** gives a real desktop window via pywebview, so the packaging
  question (§ below) is answered for free rather than deferred to Tauri.

The cost is paid in two places, both known and both bounded — see *Known gaps* below.

The UI stays thin on purpose. It transports events, renders state, and calls the CLIs from
§8; it does not hold logic of its own. Anything the UI can do, the CLIs can already do, which
keeps the terminal path alive and keeps §8's portability claim honest.

**Rendering.** Figures served from `figures/` and rendered inline where the transcript
references a path. Code blocks via `ui.markdown(..., extras=['fenced-code-blocks'])`.
Notebooks: `nbformat` → `nbconvert` HTML → `ui.html`, read-only.

### Known gaps, and how they're handled

These are the two things a React app would have given for free. Neither is a blocker; both
are worth writing down so they aren't rediscovered as surprises.

**Math rendering.** `ui.markdown` has no KaTeX support. The fix is `ui.add_head_html` to load
KaTeX plus its auto-render extension, then call `renderMathInElement` on the transcript
container after each settled message (`ui.run_javascript`). Lives in `ui/katex.py` so it is
written once. Ship the KaTeX assets locally rather than from a CDN — this is a desktop tool
that should work offline.

**Streaming smoothness.** The obvious implementation — update a `ui.markdown` element per
token — re-renders and reflows the whole element on every token, which is exactly the jitter
that makes a streaming UI feel cheap. Two mitigations, both cheap:

- Buffer incoming tokens and flush on a `ui.timer` at ~15 Hz, not per token.
- Keep the streaming tail in its own element, separate from the settled transcript above it,
  so only the tail re-renders. Promote it into the transcript (and run KaTeX over it) once
  the message completes.

**Notebooks: render, don't rebuild.** JupyterLab already exists and is better at editing than
anything worth building here. Run Lab on a side port; the agent UI parses `.ipynb` with
`nbformat` and renders *outputs* — code, stdout, images, tracebacks — read-only, with an
"open in Lab" link per notebook. Building a notebook editor is the single easiest way to burn
a month on this project.

**The four widgets that earn their keep.** Generic chat UI is a solved, boring problem; the
value is in surfacing this system's own state, which is otherwise invisible:

1. **Preflight panel.** The checks from §6 as a live checklist. When a submission is blocked,
   the failing check, its output, and its `error.fix` command are one click away. A gate is
   only tolerable if it explains itself.
2. **Expectation vs. outcome.** The highest-value visual in the app: predicted range as a
   band, actual as a marker, in-range or not obvious at a glance, with the `basis` citations
   and the `comparability` note beside it. Unjudged deviations (§7) flagged for a verdict.
3. **Quota and spend meter.** Persistent in the header: measured token usage by stage,
   credits spent on reranking, and rolling 30-day GPU spend — in-flight runs at their
   estimates — against the monthly ceiling, with uncollected runs flagged. One honesty
   note: Anthropic exposes no remaining-quota API, and the Max 5x window (5-hour rolling
   plus weekly caps) is opaque, so the token meter is self-measured usage against an
   assumed budget and is labelled as such. That is exactly what the §5 stage-0/3 decision
   needs — relative attribution by stage — just not an authoritative fuel gauge. Reads
   `ledger/quota.jsonl` and `runs.jsonl`; this is the same data §12 step 4 collects.
4. **Funnel view.** 400 → 50 → 15 with stage-3's one-line reason per surviving candidate on
   hover. This is the debugging surface for retrieval, and it is what makes the stage-0/3
   evaluation in §5 interpretable rather than a pair of numbers.

Tool calls render as collapsible cards, not raw text.

**Feel, concretely.** Dark-first, one accent colour, and Quasar's defaults overridden rather
than accepted — the giveaway that something is a stock NiceGUI app is untouched Quasar
spacing and typography, and a `ui.add_head_html` block with a real type scale fixes most of
it. Keyboard-first: submit, interrupt, jump to the latest tool call, open the ledger, all
without the mouse. Persist and restore sessions so closing the window is not destructive.

**Packaging.** `ui.run(native=True)` for a desktop window, browser mode as the fallback when
pywebview misbehaves on Windows. No Electron, no Tauri, no separate build step.

### When to revisit

This is a one-way-ish door and worth naming the exit criteria rather than relitigating it
later. Move to a separate React frontend only if one of these actually happens:

- Streaming still feels bad after the buffered-flush and split-tail mitigations above.
- The transcript view becomes slow at length — NiceGUI holds element state server-side, so a
  very long session is the plausible scaling limit.
- A widget needs interaction that Quasar genuinely cannot express.

None of these are predictions. If none occur within a month of daily use, the question is
settled and this section can be trimmed.

**Build it last, and this is the important part.** §12 puts the UI after the agent works
headless. A terminal-driven agent is fully useful and the UI is a multiplier on something
that has to exist first; building it early also means designing widgets for a ledger and a
funnel whose real shape is not yet known.

---

## 11. Open questions

- **Corpus scope for tier 2.** Ingest everything read, or only what gets annotated? Current
  lean: everything read, since ingestion is cheap and the index is small either way.
- **How much the ledger actually gets queried.** The cross-session value is a hypothesis. If
  after a month no session has usefully queried it, cut the SQLite index and keep the JSONL
  as a plain research log.
- **Whether the metrics-artifact contract (§7) is a burden.** It requires every pipeline to
  emit machine-readable metrics. That is good practice anyway, but if it turns into friction
  on borrowed third-party training code, the fallback is a per-pipeline parser under
  `skills/`.
- **SageMath.** Deferred until the mathematics needs it, not the QA. See §6.
- **GEPA.** Deferred. Every metric call is a full agent rollout, and there is no metric for
  "good research assistance" yet. Revisit only for a narrow subcomponent with a real eval
  set — most likely the retrieval query rewriter, against `evals/retrieval.jsonl`.

*Quota observability was previously an open question. It is now step 4 of §12: it is a
prerequisite for the stage-0/stage-3 decision in §5, not a nice-to-have, because "does this
stage earn its quota" is unanswerable without measuring quota.*

---

## 12. Next steps

Ordered so that the agent is usable early and the retrieval system is specified by real
usage rather than by guesswork. The permission hook comes first because it is the only item
that can cause damage.

1. `agent.py` + `prompts/system.md` + `hooks.py` with the deny-by-default mode. No network,
   no SSH, no remote compute. Confirm with `/status` that auth is subscription-backed, not
   API key, and **run the deny probe from §9** — attempt a call that should be blocked and
   verify it is denied rather than prompted or allowed. Set up credential storage now,
   before anything uses it: HF token and SSH keys go into Windows Credential Manager, and
   no credential is ever exported into the agent's environment (§9). Everything downstream
   assumes this.
2. `nb.py` against a persistent `jupyter_client` kernel: `exec` (timeout-bounded), `verify`
   (fresh kernel, all cells, non-zero on failure), `restart`.
3. **Use it for a week.** Write down what retrieval was actually reached for, in `notes/` —
   this log becomes the eval set in step 8, so record the real question, not a tidied version.
4. `quota.py` + a `Stop` hook appending token counts to `ledger/quota.jsonl`, tagged by
   stage. Cheap, and it is the measurement instrument for every later cost decision.
5. `preflight.py` with `tests` + `dry_run` only, and the submission-hash scheme from §6. Add
   the remaining checks as real failures justify them.
6. `ledger.py expect` and the expectation record. It lands before the submitters because they
   will refuse to run without it.
7. `jobs.py` and `gpu.py`: `submit` (gated on preflight + open expectation + spend ceilings
   + no stale uncollected run, writing the in-flight run record), `submit --smoke`
   (gate-exempt, hard-capped — §6), `status`, and `collect` (non-blocking by default).
   **`smoke` lands with them**, not later — it is part of preflight but needs remote access
   to exist. Only now enable remote compute in the hook.
8. `paper_search.py` — S2/Asta discovery plus OpenRouter rerank (funnel stages 1–2). No local
   index, no Haiku stages yet: get the cheap path working before adding quota-consuming ones.
9. `evals/retrieval.jsonl` harvested from step 3's log — 40–60 queries, Recall@50 and graded
   relevance, not just Hit@10. Then add stage 0 (query expansion) and stage 3 (triage) one at
   a time and keep whichever the numbers justify, dropping either where the intervals
   overlap. Tune expansion before the reranker — the retriever sets the ceiling.
10. `paper_ingest.py` and the tier-2 index, plus `ledger.py verdict` and the derived SQLite.
    The ledger's basis entries need tier-2 papers to cite, so these land together.
11. `ui/` — the NiceGUI app from §10. Last, on purpose: by now the ledger, the funnel, and
    the preflight record have real shapes to design against. Build the transcript and the
    streaming mitigations first, then the four widgets in the order listed.
