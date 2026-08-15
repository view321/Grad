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
| Token and credit spend stays bounded, not merely measured | `core/budget.py`, checked at every gateable event, over all four kinds of token |
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

### The ceiling used to count about one per cent of the tokens

Worth recording, because the row above claimed the opposite for two months and
nothing in the tests caught it. `core/budget.py` charged a project's
`quota_tokens` ceiling with `input_tokens + output_tokens`. Over the first
fortnight of real use that came to **149,063 tokens, against 12,520,659 that had
actually moved**. The missing 98.8% is cache reads: a long conversation is
re-read from the prompt cache on every tool round-trip, so cache traffic
dominates everything else by two orders of magnitude. One turn in that ledger
read 10.1M cached tokens to produce 104k of output.

The counts were being recorded correctly the whole time — `quota_log.record` has
always stored all four — so this was never a measurement problem. It was one
line of arithmetic deciding which of the four a ceiling could see.

Now all four are weighted into one number by `core/quota_log.py:billable`, which
is the only place the four become one, so a change to what a cache read is worth
lands on the gate, the meters and the summaries together. The weights are
`[quota]` in `config/grad.toml`, as ratios against one input token:

| kind | weight | why |
|---|---|---|
| input | 1.0 | the unit |
| output | 1.0 | *not* its true multiple — see below |
| cache read | 0.1 | a tenth of an input token |
| cache write | 1.25 | a quarter more than one |

Output stays at 1.0 deliberately. Weighting it by its real price would have been
more accurate and would also have silently reduced every existing ceiling; this
change is meant to reveal the 98.8% that was invisible, not to reprice the 1.2%
that was not. On the measured ledger the correction is **12×**. Set
`weight_cache_read = 0` to get the old arithmetic back.

`python -m tools.quota summary --json` reports the four counts, the weighted
total and the weights it used, side by side, because a total that is mostly cache
traffic is unarguable with its components beside it and baffling without them.

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

## Update

```bash
grad --update
```

Or `python -m tools.update check` to see what it would do first, and
`python -m tools.update apply --to v0.2.0` / `--rollback` to pin a specific
release. The desktop app checks once a day in the background and shows an
`↑ v0.2.0` button in the title bar; the workspace menu (`project ▾`) has the
button that applies it. Nothing is ever applied on its own — a silent code
change in the middle of a campaign changes the code that produced a number.

**Updates move between release tags, not to whatever is on `main`.** A version
you can cite is worth more here than a version that is merely newest, and it is
what makes `--rollback` mean something: reproducing last month's result means
running last month's code.

**The dependency install is skipped unless it has to happen.** The install is
editable, so the interpreter imports the working tree and a release that changed
only Python code is live the moment the checkout moves. `pyproject.toml`
changing between the two commits is the only thing that can alter what must be
present in the environment, and that is exactly the condition on the reinstall
— which is the difference between a two-second update and a two-minute one.
When a reinstall *is* needed and Grad is running, the update refuses and says
so: replacing a dependency underneath a live `import nicegui` is not something
to do politely.

### Two folders, and why it is worth keeping them apart

The **installation** is the checkout: code, plus the `prompts/`, `skills/` and
`config/grad.toml` that ship with it. An update replaces it. The **workspace**
is the ledger, notebooks, notes, figures and reports — your research, which no
update touches.

They default to the same folder, which is the simplest thing that works and the
one thing that makes updating awkward: research committed into the same
repository as the code puts your notebooks on the same branch as upstream's
releases, and every update becomes a merge whose conflicts land in an
append-only ledger. The installers now ask for a separate folder, and an
existing single-folder install can separate them at any time:

```bash
python -m tools.workspace move ~/Grad
```

It copies, verifies every file arrived, and leaves the originals alone unless
you pass `--remove-originals`. The three shipped paths resolve workspace-first,
so a `prompts/system.md` in your workspace still overrides the installed one.

Grad refuses an update that would collide — a file the incoming release changes
*and* you have edited — rather than checking for a dirty tree in general: on the
default layout your own notebooks make the tree permanently dirty, and an
updater blocked forever by them would be useless to exactly the people who need
it.

### Which code produced a number

Every run record carries the version that submitted it (`code_version`: the
release tag, the commit, and whether the checkout was modified). The README's
claim is that every number in a report traces to a run record; that is only
complete if the record says which code produced it.

`python -m tools.report check` therefore has a fourth rule. A report whose cited
runs straddle two versions is a finding — a number from one version is not
comparable to a number from another — as is a run submitted from a checkout
with uncommitted edits, because the code it names cannot be resolved by anyone
else. Runs with no stamp at all pass silently: they predate the field, and
refusing a report because its evidence is old would make the rule a reason to
avoid updating.

### Where the conversation gets compacted, and who decides

The CLI underneath compacts on its own, and a live session reports the threshold
as **967,000 of a 1,000,000 window**. That is a ceiling in the sense that a wall
at the end of a runway is one: by the time it is reached, every tool round-trip
has spent a long time re-reading most of a million cached tokens — which, with
the accounting above fixed, is now visible as the dominant cost it always was.

There is no way to ask the SDK to compact, and no way to move its threshold from
here. The control protocol has ten subtypes — `initialize`, `mcp_status`,
`get_context_usage`, `interrupt`, `set_permission_mode`, `set_model`,
`rewind_files`, `mcp_reconnect`, `mcp_toggle`, `stop_task` — and none of them is
"compact"; the threshold comes from settings, and `agent.py` leaves
`setting_sources` unset on purpose so a stray `settings.json` cannot add
permission rules behind the code's back.

So `core/compaction.py` does it, at `[agent] compact_at_tokens` (300k by
default, 0 to disable). Being ours buys three things the CLI's version cannot:

* **It is visible.** A compaction performed in-band rewrites what the model
  remembers while the transcript on screen still shows every turn — the user is
  looking at evidence for a belief the agent no longer holds, and nothing says
  so. Grad's writes a marker into the transcript where it happened, with the
  handover note behind a disclosure.
* **It is metered.** The summary is charged to a `compaction` stage of its own,
  so "what does compacting cost" is a question the ledger answers rather than a
  cost folded into the conversation it was compacting.
* **It happens where you chose.**

The mechanism has no clever part: ask the session, while it still remembers
everything, to write a note to whoever picks it up next; drop the client; start a
fresh conversation; hand it the note in front of the next prompt rather than as a
turn of its own, so it costs nothing extra. The note is asked for in the first
person and asks for paths, commands, and the ledger state the next turn is
expected to act on — an expectation registered and not yet judged, a run
submitted and not yet collected. A generic "summarise the conversation" prompt
drops those every time, and losing them does not read as a bad summary. It reads
as an agent that abandoned a run halfway.

**Compacting is not obviously cheap, and the threshold is not a "lower is
better" dial.** The summary costs a turn, and the session it seeds starts with a
cold prompt cache — so the turn after a compaction pays cache *writes* at 1.25×
where it would have paid cache *reads* at 0.1×. There is a threshold below which
compacting costs more than not compacting. The `compaction` stage is what makes
that measurable, which is why the accounting split landed before this did.

The chat window's statusline carries a context meter, measured against whichever
limit will actually be reached first — Grad's threshold when one is set, the
CLI's otherwise — because a meter reading 40% means quite different things at
300k and at 967k. It reads `—` rather than `0` before the first reading: an
unknown context and an empty one look identical at a glance and only one of them
is worth acting on.

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
| `tools/traces.py` | tag stored sessions, and harvest eval candidates from real use — **human-facing only** |
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
  compaction.py       where a conversation is compacted, and what survives it
  traces.py           a session as tags a later query can slice on  -- pure, tested
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

That step had a prerequisite nobody wrote down: the week of use has to leave
something sliceable behind. A directory of transcripts is a record, but "every
session where a submitter refused" was a full-text search whose answer depended
on how the refusal happened to be phrased. `core/traces.py` tags each
trajectory — `tool:`, `gate:`, `ledger:`, `outcome:`, `turns:`, `cost:` — and
`python -m tools.traces list --json` reports what a week actually consisted of,
which is usually not what it felt like it consisted of.

`gate:` is the namespace worth having, and the one ml-intern's equivalent has no
reason to want. Every row of the table at the top of this file is a claim that
some gate refuses under some condition; a corpus of real sessions tagged by
which gate refused is the difference between believing that and knowing it. A
verb that was only asked about does not count — `ledger expect --help` tags the
module and not the verb, because on the real corpus four of the five `ledger:`
tags on the busiest session came from `--help` calls, and a corpus that cannot
tell reading an interface from using it would answer the question wrongly.

`python -m tools.traces harvest` turns the questions actually put to
`paper_search` into eval rows. They arrive **ungraded** — `relevant` is empty —
because which papers were the right answer is the one part of an eval row a
trace cannot recover, and a harvester that guessed would measure the guess. It
never rewrites an existing row and never appends a duplicate, so it is meant to
be re-run as the corpus grows.

## Tests

```bash
python -m pytest -q
```

The gate tests run against a real ledger in a temp workspace rather than against
mocks. A mock of a gate proves nothing about the gate, and these are the checks
that stand between an agent under deadline pressure and a GPU bill.

## Licence

MIT — see [`LICENSE`](LICENSE).

`pyproject.toml` claimed MIT from the first commit and the repository contained
no licence file, which is the one combination that is worse than saying nothing:
the package metadata grants a licence the repository does not. Both now say the
same thing, and `pyproject.toml` says it as an SPDX expression with
`license-files` rather than the deprecated free-text form.
