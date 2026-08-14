# GradientAgent — Handoff 2: the extension

Companion to [`HANDOFF.md`](HANDOFF.md), which remains the design record for everything
already built. Section numbering continues from it, so cross-references like "§6" and "§9"
mean what they mean there.

**§1–§12 of the original remain in force except where §14 below explicitly amends them.**

*Revision 1 (2026-08-14): scoped eight additions — HF organization namespaces, an embedded
JupyterLab surface, library-currency checking over Context7, a human-facing RepoWiki, an
evolutionary search loop over ShinkaEvolve, a report generator, per-role model selection,
and the project/budget dimension that three of the others turned out to need. Verified
against primary sources during the session: the `huggingface_hub` Jobs signatures, the
RepoWiki 0.3.1 wheel's internals, ShinkaEvolve's Headless provider, Context7's auth model,
and the death of the Tabnine JupyterLab extension.*

---

## 13. What this adds

Eight items. Three of them (§15, §16, §17) are foundations the rest lean on; two (§21, §22)
are substantial new capabilities; three (§18, §19, §20) are self-contained conveniences.

| # | Item | Section | Size |
|---|---|---|---|
| 1 | Models selected by role, Opus 5 as the research default | §16 | ~2 h |
| 2 | The `project` dimension and budget ceilings | §15 | ~2 d |
| 3 | HF Jobs under an organization namespace | §17 | ~0.5 d |
| 4 | `tools/docs.py` — library currency via Context7 | §18 | ~0.5 d |
| 5 | Embedded JupyterLab + `tools/lab.py` | §19 | ~1.5 d |
| 6 | RepoWiki, human-facing | §20 | ~1 d |
| 7 | `tools/evolve.py` — evolutionary search over ShinkaEvolve | §21 | ~3–5 d |
| 8 | `tools/report.py` — the scientific report | §22 | ~3–4 d |

### The rule still governs

§1's rule is unchanged: **anything that spends money, destroys work, or must be true before
the fact is enforced mechanically, not by prompt.** Two of the additions here are loops that
spend resources without a human between iterations, and one is a machine for asserting
results confidently. Those are precisely the shapes the rule exists for. New rows for §1's
table:

| Thing that must hold | Enforced by | Section |
|---|---|---|
| Token and credit spend stays bounded, not merely measured | `core/budget.py`, checked at every gateable event | §15 |
| An evolutionary campaign cannot outspend its allocation | campaign budget check before each generation, in `tools/evolve.py` | §15, §21 |
| A job submitted to an org is collectable from that org | the namespace is persisted on the run handle, not just passed at submit | §17 |
| Every number in a report traces to a run record | `report check`, refuses on an unresolved claim | §22 |
| Every citation in a report is a real paper | `report cite` resolves only against the corpus and verified S2 ids | §22 |
| A result that has not been judged cannot be published | `report check` refuses while any cited run has an unjudged deviation | §22 |

---

## 14. Amendments to §3

Fewer than expected. Most of this document is *additions* to the decision table rather than
reversals, which is a reasonable signal that the original design absorbed the new
requirements rather than fighting them.

### Rows that survive, and why that was a live question

**"Subagents: not used for research."** Survives. A QA subagent was designed and then
rejected in favour of a CLI (§18). The narrowing that made the subagent tractable — check
library currency and compatibility, nothing else — also made its agency unnecessary. Keeping
this row intact means `Task` stays in `DENIED_TOOLS` ([agent.py:39](agent.py:39)) and
`agent.py --probe` needs no changes.

**"MCP servers: none as external servers."** Survives. Context7 is reached over plain HTTP
from a CLI, which is the same move §5 already makes for Asta: *"It is an MCP endpoint, but it
speaks streamable HTTP, so `paper_search.py` can call it over plain HTTP without adopting MCP
as an architecture"* ([HANDOFF.md:229](HANDOFF.md:229)). This is a second instance of an
existing pattern, not a new one.

**"Direct Messages API calls: not used."** Survives, and §20 exists partly to keep it that
way — RepoWiki reads `ANTHROPIC_API_KEY` by default, which is exactly what
`credentials.scrub_environment()` deletes.

### The one real amendment

| Decision | Was | Now | Why |
|---|---|---|---|
| Notebook editing in the UI | "Rejected: building a notebook editor. The UI renders notebook *output*; editing links out to Lab" ([HANDOFF.md:156](HANDOFF.md:156), [:853](HANDOFF.md:853)) | **JupyterLab is embedded as a tab**, served by `tools/lab.py` | The rejection was of *building an editor*, and that still stands — we build none. What changed is that "links out to Lab" was never wired up: [ui/app.py:405](ui/app.py:405) points at a `localhost:8888` nobody starts. Embedding the real Lab honours the original reasoning better than the current stub does, and it is what makes arbitrary Lab extensions possible at all (§19). |

### New rows for §3

| Decision | Choice | Rationale |
|---|---|---|
| Model selection | By **role**, in a `[models]` config section | Five roles across four surfaces; scattering them across `[agent]` and `[retrieval]` does not survive the additions here — see §16 |
| Resource accounting | One `project` dimension on every cost-bearing record | HF payer attribution, evolutionary campaigns, and research budgets are three faces of one missing abstraction — see §15 |
| Evolutionary search | **ShinkaEvolve**, driven by `tools/evolve.py`, not forked | Its Python API plus the Headless provider covers what we need; a fork is a maintenance cost to defer until hook points prove insufficient — see §21 |
| Report generation | **Built here**, not adopted from a harness | Every existing harness reconstructs provenance from unstructured logs. Ours is already structured, and that is the whole advantage — see §22 |
| AI completion inside Lab | **None for now** | The only subscription-compatible path would be a second provider and a fifth credential; Tabnine's JupyterLab extension is dead (§19) |

### Additions to "Rejected, and why"

- **A reality-checker subagent.** Designed, scoped down, then rejected. Once the brief
  narrowed to "outdated libraries and compatibility issues", every claim it made became
  checkable against an oracle, and a CLI the main agent calls is strictly less invasive than
  re-enabling `Task`. See §18.
- **Pyright as a preflight gate.** Strict-mode Pyright over an ML codebase produces noise,
  and a check that gets ignored is worse than no check. Its deterministic strength
  (signatures, removed attributes) is covered by kernel introspection plus §18. Deferred, not
  rejected: revisit if `tools/docs.py` proves too soft.
- **Tabnine for JupyterLab.** `jupyterlab-tabnine` 0.0.24 was released 2021-08-24 with
  classifiers stopping at Python 3.9; the npm client is equally stale; the classic-Notebook
  variant last moved in March 2021. The JupyterLab 3→4 break is what killed it.
- **Forking ShinkaEvolve up front.** See §21 — a driver first, a fork only on evidence.
- **AI Scientist v2 / PaperOrchestra / Denario as the report harness.** See §22 — adopting
  one means discarding the ledger advantage and conforming to its log format.
- **Replacing the Voyage reranker with Haiku.** See §16 — worse at the task, and it moves
  load from credits onto the subscription quota this design is trying to protect.

---

## 15. The project dimension and the budget system

**Build this first.** §17, §21, and §22 all consume it, and building either loop (§21, §22)
before it exists is how a runaway campaign ends up blocking every future submission through
the §6 stale-run gate with no record of what consumed the quota.

### The gap

Three resources are consumed. They are tracked very unevenly:

| Resource | Measured today | Ceiling today |
|---|---|---|
| GPU dollars | yes — `runs.jsonl`, actuals and estimates | **yes** — `core/gates.py:check_spend`, per-job and monthly |
| API credits (Voyage embeddings, OpenRouter rerank) | yes — `quota.jsonl`, as `funnel.rerank` / `embed` stages | **no** |
| Subscription quota (tokens) | yes — `quota.jsonl`, by stage | **no** |

So [README.md:20](README.md:20)'s claim that "cumulative spend stays bounded" holds for one
resource in three. The other two are instrumented and unbounded. §21 in particular is a loop
that consumes all three.

### The insight: one dimension, three uses

Three separate requirements turned out to want the same thing — a dimension carried on every
cost-bearing record:

- **§17** needs to know which account paid for an HF job (personal vs. organization).
- **§21** needs to bound a campaign made of many runs.
- **The user's ask** is a budget for a piece of research.

Build it once. Every run record, every `quota.jsonl` entry, and every credit spend carries a
`project` id.

### Records

`ledger/projects.jsonl`, append-only and folded like `runs.jsonl`:

```json
{
  "id": "proj-scaling-w2",
  "created_at": "2026-08-14T09:00:00Z",
  "title": "width-vs-depth scaling under a fixed token budget",
  "payer": "hf:myorg",
  "budget": {"gpu_usd": 50.0, "quota_tokens": 5000000, "credits_usd": 10.0},
  "status": "open"
}
```

`payer` lives on the project rather than being invented per submission, so the org attribution
in §17 is a consequence of choosing a project rather than a separate flag to forget.

**Current project selection.** A file, `ledger/.current_project`, written by
`tools/budget.py use <id>` and read by every CLI, overridable per invocation with
`--project`. Deliberately *not* an environment variable: `credentials.scrub_environment()`
([core/credentials.py:103](core/credentials.py:103)) strips the agent's environment, and a
selection mechanism that the agent's own startup deletes is a bug waiting to happen.

Every run record gains `"project": "<id>"`. Every `quota_log` entry gains the same. This is
an additive schema change; `core/ledger_store.py` folds unknown-project records as
`"unassigned"` so existing ledgers keep loading.

### Enforcement, and where it is honest

Enforcement quality differs by resource, and the difference is structural rather than a
shortcoming of the implementation:

**GPU dollars — clean.** Submission is a discrete, gateable event. `check_spend` already
fires there; it gains a project-scoped check alongside the global one.

**Evolutionary campaigns — clean.** A generation boundary is a discrete event. §21 checks the
projected cost of generation *n+1* against remaining allocation before starting it.

**Subscription tokens — granular to one turn.** Tokens are consumed continuously inside a
turn and there is no way to refuse mid-turn. Two mechanisms, neither depending on SDK
behaviour we have not verified:

1. `agent.py`'s own turn loop checks remaining allocation *before* issuing the next turn and
   refuses with the overrun printed. This is our code end to end.
2. `hooks.py:pre_tool_use` denies cost-bearing Bash commands (`tools.jobs submit`,
   `tools.evolve run`, `tools.report write`) once the project is over budget. That hook
   already denies reliably ([hooks.py:133](hooks.py:133)) and is exercised by
   `agent.py --probe`.

The Stop hook ([hooks.py:145](hooks.py:145)) keeps its current job — recording usage — and
gains threshold warnings. It is deliberately *not* the enforcement point: its documented
`block` semantics force continuation rather than halting, which is the opposite of what is
wanted here.

**So the honest statement is: token budgets are enforced to a granularity of one turn's
overrun.** Write that in the CLI's `--help`, not just here.

### A second honesty note

Subscription quota is not linear in tokens, and the real limits are rolling windows (5-hour
and weekly on Max) which the SDK does not expose as a remaining balance. A token ceiling is
therefore a **proxy the user controls**, not a mirror of Anthropic's limit. Hitting the real
rate limit is an event the system can only observe after the fact. The meter must not imply
otherwise — this is the same discipline §10 already applied when it *"reworded the quota
meter to what it can actually measure"* ([HANDOFF.md:20](HANDOFF.md:20)).

### CLI — `tools/budget.py`

```bash
python -m tools.budget new --id proj-scaling-w2 --title "..." \
    --gpu-usd 50 --quota-tokens 5e6 --credits-usd 10 --payer hf:myorg --json
python -m tools.budget use proj-scaling-w2 --json
python -m tools.budget status --json          # current project, spend, remaining, per resource
python -m tools.budget raise --gpu-usd 75 --json   # deliberate, logged, never silent
python -m tools.budget close proj-scaling-w2 --json
```

`raise` appends an event rather than mutating; a ceiling that can be edited invisibly is not
a ceiling. Same argument as §7's append-only ledger.

New exit code: **12 — project budget exceeded**, distinct from 6 (global spend ceiling), so
"this research ran out of its allocation" is never confused with "the machine is out of
money".

### UI

The header meter ([ui/widgets/quota_meter.py](ui/widgets/quota_meter.py)) gains a project
selector and shows three bars rather than one. The Quota tab gains a per-project breakdown by
stage. Both read the ledger; no new logic in the UI, per §10.

---

## 16. Models by role

### The correction that motivates this

**Haiku is not the reranker.** The funnel is expand → retrieve → **rerank** → triage → select.
Haiku runs stages 0 and 3 (expand, triage). Stage 2's reranker is `voyageai/rerank-2.5` via
OpenRouter — a dedicated cross-encoder, not a generative model
([config/grad.toml:41](config/grad.toml:41), [HANDOFF.md:124](HANDOFF.md:124)).
[core/haiku.py:249](core/haiku.py:249) draws the line explicitly: triage is *"a funnel
widener, not a better ranker."*

Do not swap Voyage for Haiku. It is worse at pairwise relevance scoring, and — the reason
that matters here — Voyage costs **credits** while Haiku costs **subscription quota**. The
swap moves load onto the scarcer resource.

### Model facts

- The Claude 5 family is **Fable 5, Opus 5, Sonnet 5**. There is no Haiku 5; the latest Haiku
  is **4.5**.
- Ids: `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5-20251001`.
- `config/grad.toml` and `core/config.py` currently default to `claude-opus-4-5`
  ([config/grad.toml:55](config/grad.toml:55), [core/config.py:80](core/config.py:80)). Update
  to `claude-opus-5`.
- The existing `claude-haiku-4-5` entries are already correct.

### The `[models]` section

```toml
[models]
research = "claude-opus-5"      # the main loop (§3)
evolve   = "claude-sonnet-5"    # ShinkaEvolve mutation operators (§21)
expand   = "claude-haiku-4-5"   # funnel stage 0 (§5)
triage   = "claude-haiku-4-5"   # funnel stage 3 (§5)
report   = "claude-opus-5"      # prose synthesis (§22)
cite     = "claude-haiku-4-5"   # citation resolution — mechanical matching (§22)
```

`[retrieval] rerank_model` and `embed_model` **stay where they are**. They are a different
provider on a different billing rail, and folding them into `[models]` invites exactly the
substitution argued against above. `core/config.py` keeps the old `[retrieval] triage_model` /
`expand_model` keys readable as overrides for one release so existing configs do not break.

Every model call already routes through `quota_log.from_sdk_usage` with a stage; the stage
gains the role name so `tools/quota.py summary` can answer "what did Opus cost me this week"
without inference.

### On the evolve default

Sonnet 5 as the default is right for cost. But ShinkaEvolve's design is explicitly *an
ensemble of LLMs acting as mutation operators* — collapsing to a single model discards
diversity the algorithm is built around. **Default to an ensemble of Sonnet 5 (primary) and
Haiku 4.5 (cheap explorer)** and let Shinka's bandit allocate between them. Overridable with
`--set evo.llm_models=...`, which is Shinka's own mechanism.

---

## 17. HF Jobs under an organization namespace

### Verified API facts

Checked against the installed `huggingface_hub` **1.16.1** during the design session:

```
run_job(*, image, command, env, secrets, flavor, timeout, labels, volumes, namespace, token)
inspect_job(*, job_id, namespace, token)
fetch_job_logs(*, job_id, namespace, follow, tail, token)
cancel_job(*, job_id, namespace, token)
list_jobs(*, timeout, namespace, token)
whoami(token, *, cache=False) -> dict
```

`JobInfo` fields: `id, created_at, started_at, finished_at, docker_image, space_id, command,
arguments, environment, secrets, flavor, labels, volumes, status, durations, owner,
initiator, endpoint, url`.

### The trap

**`namespace` is a property of the job handle, not a submit-time parameter.** Adding it only
to `run_job` at [tools/jobs.py:148](tools/jobs.py:148) produces a job that cannot be found
again: `inspect_job` and `fetch_job_logs` would look under the personal namespace and 404. The
run never collects, goes stale, and then blocks *every* future submission through the §6
stale-run gate (exit 7). The failure appears far from its cause.

So the namespace must be persisted into the handle at
[tools/jobs.py:173](tools/jobs.py:173) (`attach_handle`) and threaded through every call
site that takes one:

- `_poll` — [tools/jobs.py:285](tools/jobs.py:285)
- `_logs` — [tools/jobs.py:307](tools/jobs.py:307)
- `cmd_status` — [tools/jobs.py:347](tools/jobs.py:347)
- `cmd_collect` — [tools/jobs.py:376](tools/jobs.py:376)
- `run_smoke` — [tools/jobs.py:199](tools/jobs.py:199)

### Membership validation

`whoami(token=...)` returns the organizations a token can act for. Validate the requested
namespace against it **before** `record_submission`, in the same place `_hub()` and `_token()`
are already called for exactly this reason ([tools/jobs.py:136](tools/jobs.py:136)): a
configuration problem must not leave a phantom estimate sitting on the ceiling. Surface it in
`credential status` too.

One network call per submit is acceptable on a path about to spend dollars. Do not cache it
aggressively — org membership changing is precisely the case worth catching.

### Resolution order

`--namespace` flag → spec `[target] namespace` → project `payer` (§15) → `[hf] namespace` →
`None` (personal). This mirrors how `flavor` already resolves at
[tools/jobs.py:128](tools/jobs.py:128).

### Interaction with the submission hash

The hash deliberately excludes `target` ([core/submission.py:299](core/submission.py:299)),
which is why `flavor` is not hashed. `namespace` follows the same rule for consistency — it is
not hashed.

The consequence is real and must be handled rather than ignored: a preflight whose `smoke`
check ran under personal credentials validates a job that will run in an organization. So
`run_smoke` records the namespace it used into the check result (which already flows into the
preflight record via `record_check_result`,
[tools/preflight.py:369](tools/preflight.py:369)), and `submit` **warns** when they differ.
Warn, not refuse — consistent with how `target` and `flavor` already behave.

### Spend

Costs are attributed to the project's `payer` (§15). An organization's budget and a personal
budget are separate allocations; without this, org runs silently consume the personal ceiling
and vice versa.

`hooks.py` is unchanged — bare `hf` stays denied.

---

## 18. Library currency: `tools/docs.py`

### Why a CLI and not a subagent

The original plan was a Haiku "reality checker" subagent with Context7 and Pyright. It was
rejected after the brief narrowed. The reasoning is worth keeping, because it will be
tempting to revisit:

A QA layer staffed by a weaker model than the one it checks is only sound when every claim it
makes is **checkable against an oracle**. Narrowing the brief to library currency and
compatibility achieved that. But once achieved, the agency was doing no work: "point it at a
file, get a verdict" is a tool, not an agent. The CLI form keeps `Task` denied, keeps
Context7's schemas out of the main loop's context entirely, and inherits `--json`, exit codes,
and `fix` fields from `core/cli.py` for free.

Revisit only if the iterate-and-recheck loop (introspect → hypothesise → run a counter-example
→ recheck) proves necessary. That is the one thing the CLI form cannot do.

### The two oracles

Context7 alone is not sufficient, and it is the weaker half:

1. **Introspection — what actually exists on this machine.** `importlib.metadata.version()`,
   `inspect.signature()`, `dir()`, run through `tools/nb.py exec`. Cheap, offline,
   definitive. This is how the `namespace` parameter in §17 was found, in about ten seconds.
2. **Context7 — what is current.** Deprecations, changed idioms, migration paths. Answers what
   introspection cannot see.

**Order matters: introspect first.** A checker relying on Context7 alone will confidently
describe an API version that is not installed.

The main agent runs both. `prompts/system.md`'s tool list gains one entry, and the "Habits
that matter" section gains a line: *check a library call against the installed signature
before trusting it, and against `docs.py` before assuming it is current.*

### Context7 facts

Verified: a REST API exists (reference at `context7.com/docs/api-guide`), auth is
`Authorization: Bearer <key>`, keys are free from their dashboard, and rate limits scale with
a registered key. The MCP tool names are `resolve-library-id` and **`query-docs`** — note that
`get-library-docs`, the name in older material, is stale. An official `ctx7` CLI also exists
(`ctx7 library <name> <query>`, `ctx7 docs <libraryId> <query>`).

**Not verified in session: the exact REST endpoint paths.** Read
`context7.com/docs/api-guide` before implementing.

### Why wrap rather than allowlist `ctx7`

The `--json` / exit-code / `fix`-field contract is what makes a tool legible to the model
(§8), and `ctx7` will not have it. The wrapper is also where the credential fetch and the
cache live, and caching matters more than it sounds: documentation lookups repeat heavily and
`core/http.py` already has the TTL cache and rate-limit machinery.

### Shape

```bash
python -m tools.docs resolve "huggingface_hub" --json
python -m tools.docs query <libraryId> "run_job namespace parameter" --json
python -m tools.docs check tools/jobs.py --json    # introspect imports, flag stale/deprecated calls
```

`check` is the interesting one: parse the file's imports, resolve installed versions, and
report calls whose signatures do not match what is installed, plus anything Context7 flags as
deprecated. Exit 9 (`a check ran and failed`) on findings, so it composes with preflight's
declared-check mechanism ([tools/preflight.py:315](tools/preflight.py:315)) if a pipeline
wants it as a gate later.

### Credential

Fifth Credential Manager entry, `context7_key`, added to `CREDENTIAL_NAMES`
([tools/jobs.py:477](tools/jobs.py:477)), fetched at point of use exactly like
[tools/jobs.py:62](tools/jobs.py:62), and added to `scrub_environment`'s list
([core/credentials.py:111](core/credentials.py:111)).

---

## 19. The notebook surface: embedded JupyterLab

Amends §10's notebook handling. See §14 for why this honours the original reasoning rather
than reversing it.

### Architecture

`tools/lab.py` manages a JupyterLab server as a subprocess on a side port with a token; the
UI adds a tab holding it in an iframe.

```bash
python -m tools.lab start --json      # returns port + token
python -m tools.lab status --json
python -m tools.lab extensions --json # what is installed, so the state is inspectable
python -m tools.lab stop --json
```

Two things that will otherwise cost an afternoon each:

**Framing.** JupyterLab ships `X-Frame-Options` / CSP headers that block embedding. It needs
a `tornado_settings` header override permitting the app's origin, in
`config/jupyter/jupyter_server_config.py`.

**The sandbox.** The existing notebook iframe is `sandbox=""` for a real reason — notebook
output is untrusted HTML that could otherwise run script in the page driving the agent
([ui/app.py:415](ui/app.py:415)). The Lab iframe **cannot** be sandboxed that way. It must be
a separate iframe, deliberately unsandboxed, pointed at a server we started ourselves. Keep
the existing read-only renderer as-is for the output view; do not merge the two.

### Kernel ownership — the rule that must not be lost

`tools/nb.py` spawns detached kernels via its own connection files
([tools/nb.py:66](tools/nb.py:66)). Lab has its own kernel manager. Two owners over one
notebook reproduces exactly the "works in the kernel that grew it" failure that `nb.py verify`
exists to catch ([HANDOFF.md:503](HANDOFF.md:503)).

**The discipline is unchanged: anything edited in Lab passes `nb verify` before it is cited in
`notes/` or referenced from a ledger entry.** The highest-value part of this whole item is a
per-notebook **Verify** button in the Notebooks tab that shells out to
`python -m tools.nb verify <path> --json` and renders the failing cell index and traceback.
Build that first; it is worth more than the embed.

`NotebookEdit` stays in `DENIED_TOOLS` ([agent.py:39](agent.py:39)). This item is about the
*human* editing by hand; the agent continues to edit notebooks through Write/Edit plus
`nb.py`, which works and does not depend on version-sensitive tool semantics.

### Arbitrary extensions

Lab already has a plugin system. Do not design one. What gets built is the reproducibility
layer:

- a pinned `lab` extra in `pyproject.toml`, so the extension set is declared rather than
  accumulated
- `config/jupyter/` holding `jupyter_server_config.py` and `overrides.json`
- `tools/lab.py extensions --json` so the installed set is inspectable like everything else

"Connect an arbitrary extension" then means: add a pin, reinstall, restart.

Three caveats to record before anything is installed:

1. **Server extensions run as you.** A frontend extension is confined to the browser. A server
   extension runs in the Lab process with your filesystem rights — it can read `ledger/` and
   `notes/`, and it can `import keyring` and reach the credential store. That is the same
   honest residual [core/credentials.py:9](core/credentials.py:9) already names. Frontend-only
   extensions are low risk; read a server extension before installing it.
2. **Origin.** An extension is code running in Lab's origin, and that iframe is unsandboxed.
   Keep Lab on its own port and do not share the storage secret from
   [ui/app.py:241](ui/app.py:241).
3. **Pin everything.** The JupyterLab 3→4 break is what killed the Tabnine extension. Pin
   `jupyterlab` itself and every extension, or an unrelated `pip install -U` takes the app
   down.

### AI completion inside Lab

Not for now. `jupyterlab-tabnine` is dead (§14). The live alternatives — `jupyter-ai` and
LSP-based copilots — route through an API key, which collides with §2 and would need a second
provider and a fifth credential decision. Revisit only if hand-editing in Lab becomes frequent
enough to justify it.

---

## 20. RepoWiki: the human's map

**Scope: human-facing only.** Not in the agent's tool list, not in `prompts/system.md`, no
context cost. Its job is letting a person reacquire the shape of a growing codebase quickly.
HANDOFF.md remains the design record and README.md the report; RepoWiki targets the third
thing — the module-level "what calls what, and where does this value come from" view nobody
wants to maintain by hand.

### Teardown facts (wheel 0.3.1, inspected in session)

- **All LLM coupling is in one 95-line file**, `repowiki/llm/client.py`, behind a two-method
  async interface: `complete(messages, *, temperature, max_tokens, response_format)` and
  `stream(...)`.
- Constructed at exactly **four call sites**: `cli.py:200`, `cli.py:402`,
  `server/routers/chat.py:65`, `server/routers/scan.py:117`.
- **`response_format` is never passed by any caller.** All four analyzer calls
  (`core/analyzer.py:116, 209, 237, 297`) use plain `complete(messages, max_tokens=4096)` and
  parse the returned text themselves. The structured-output problem that would have made a
  fork risky does not exist.
- Prompts are simple `[system, user]` pairs from `llm/prompts.py` (five builders). Flattening
  to `ClaudeAgentOptions(system_prompt=...)` + `query(prompt=user)` is exact, not lossy.
- **`repowiki map` is LLM-free.** `cli.py:40 repo_map()` never touches `LLMClient`, and
  `core/graph.py` has no LLM references. It needs no credential at all.

### The decision

**Try `repowiki map .` first.** It is free and may cover enough of the need to make the rest
unnecessary.

If more is wanted, **fork and replace `LLMClient`** with an Agent SDK implementation — roughly
120 lines, with `core/haiku.py:110` as the model for the plumbing. Token counters map onto
`quota_log.from_sdk_usage` with a new stage; `total_cost` should report 0 rather than a
fabricated number, since subscription usage is not priced per token.

**The fork is optional, not required, and it is worth knowing why before spending the day.**
`scrub_environment()` cleans only the *agent's* process ([agent.py:77](agent.py:77)). A human
running `repowiki scan` in their own shell with an API key set violates nothing technically.
The catch is that a key in the user profile will also be in the agent's environment and
trigger the scrub warning on every launch — safe, but noisy, and it erodes the §2 discipline
by habituation. The fork is cleaner. It is a preference.

While in there, one optional upgrade: those four analyzer calls parse free text, which is the
failure mode [core/haiku.py:12](core/haiku.py:12) documents — *"prompting for JSON and parsing
it fails silently on the tenth call."* Routing them through the forced-tool pattern removes a
class of silent corruption from the generated wiki.

### Scope and staleness

- Point it at `core/` and `tools/`. **Never** `ledger/`, `notes/`, or any papers directory —
  it ships content to a third party, and those hold research data.
- Output HTML (`--format html --open`), not markdown committed to the repo, so it never
  competes with the hand-written docs.
- A wiki behind the code is worse than none, because it is trusted. Record the source-tree
  hash in the output and provide a one-line staleness check — the pattern
  `core/submission.py` already implements.

---

## 21. Evolutionary search: `tools/evolve.py`

### What it is

[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana AI, arXiv:2509.19349)
maintains a population of programs evolved across generations, with an ensemble of LLMs acting
as mutation operators. Three patch types (`diff`, `full`, `cross`), island-based diversity,
parallel evaluation locally or on SLURM, and a `shinka_visualize` web UI.

**The Headless provider solves the auth problem.** A May 2026 release added CLI-backed
mutation models for subscription-backed agents — model strings like `headless/claude` and
`headless/codex@gpt-5.5?effort=high`, routed through `npx -y @roberttlange/headless`, with a
`headless --check` at startup. `examples/sine_approx_headless` is fully API-free. So this runs
on the Max subscription without bending §2.

### Why the contracts align

Shinka wants `evaluate.py` returning a metrics dict containing `combined_score`, and
`initial.py` with `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers around the mutable region.
Grad already has a submission spec with an entrypoint and a `metrics_file` whose parsed
contents flow through `submit_lib.parse_metrics` → `results` → `deviations`. These describe
the same object. Structurally this is "run the submit/collect loop N times with an LLM
choosing the next candidate" — an extension of §6/§7, not a bolt-on.

### Driver, not fork

Shinka exposes a Python API: `EvolutionConfig`, `LocalJobConfig`, `DatabaseConfig`,
`ShinkaEvolveRunner(...).run()`. What we need is **gating and ledger integration around the
loop**, which is a driver.

Fork only if per-candidate budget charging turns out to require intercepting inside the
generation loop and no hook point exists. **Not verified in session: whether Shinka exposes a
per-candidate callback.** Check before starting; it decides driver-vs-fork.

### The four collisions, and their resolutions

**1. An evolutionary loop is a machine for spending money with no human in it.** This is the
dangerous one. `check_spend` fires per submission, so the ceiling *would* stop it — at
generation 40, abandoning an in-flight run that then goes stale and blocks every future
submission (exit 7). Succeeding at the search would brick the system.

*Resolution:* a **campaign budget gate**. Before generation 0, refuse unless
`estimate_per_candidate × max_candidates` fits under the project's remaining allocation (§15).
Re-check before each generation. Shinka's own `max_api_costs` covers the LLM side; the compute
side is the expensive half and Grad must own it. **Do not run a single remote generation
before this exists.**

**2. The expectation gate is 1:1 with a run; evolution is 1:N.** You cannot pre-register a
prediction per candidate.

*Resolution:* the **campaign** is the unit of prediction — "the evolved variant beats baseline
X on metric Y by ≥ Z" — with candidate evaluations recorded as sub-runs exempt from the
per-run expectation gate. This is arguably more faithful to §7 than the current design, and it
is exactly the relational shape [prompts/system.md:16](prompts/system.md:16) already prefers.
Needs a `campaign` kind in the ledger and gates that understand campaign membership.

**3. Every mutation invalidates the preflight hash — correctly, and expensively.**
`Submission.hash()` covers the entrypoint's import graph, so each candidate needs a fresh
preflight, and `smoke` is a *paid remote job* ([tools/jobs.py:199](tools/jobs.py:199)). Naively
this doubles per-candidate cost.

*Resolution:* candidates run `--only tests,dry_run` — both local, both fast. Smoke is required
once per campaign at the baseline, and again whenever a mutation escapes the evolve-block. The
`EVOLVE-BLOCK` markers make "did it escape" mechanically checkable, which is convenient.

**4. `combined_score` is a Goodhart machine.** A search optimising a scalar will find the bug
in the metric. That is precisely the failure this codebase's temperament resists — *"a
surprise is an alarm... a bug hypothesis first"* ([prompts/system.md:15](prompts/system.md:15)).

*Resolution:* the campaign winner goes through the normal `verdict` path before it counts as a
result, and the campaign report surfaces top-K rather than the argmax.

### Phasing

**Phase 1: local only.** Shinka needs no GPU for many tasks and the headless example is
API-free. Run a campaign evaluated entirely through `tools/nb.py`, zero remote jobs. This
proves the campaign records, the sub-run bookkeeping, and the budget integration while the
blast radius is zero.

**Phase 2: remote**, behind the campaign budget gate.

Doing the ledger work and the spend work simultaneously against live GPU jobs is how you learn
about exit 7 the hard way.

### Shape

```bash
python -m tools.evolve init --task-dir pipeline/evolve-lr --json
python -m tools.evolve run --project proj-scaling-w2 --generations 20 --local --json
python -m tools.evolve status --campaign camp-... --json
python -m tools.evolve promote --campaign camp-... --candidate 47 --json   # into a normal run
```

Default models per §16: Sonnet 5 primary plus Haiku 4.5 explorer.

Sources: [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) ·
[releases](https://github.com/SakanaAI/ShinkaEvolve/releases) ·
[agentic_usage.md](https://github.com/SakanaAI/ShinkaEvolve/blob/main/docs/agentic_usage.md) ·
[arXiv:2509.19349](https://arxiv.org/abs/2509.19349) ·
[claude-evolve](https://github.com/samuelzxu/claude-evolve) (community Claude Code
reimplementation — considered, not chosen: Grad's value is the ledger and the gates, and
pointing a mature harness at them beats adopting a second harness with its own orchestration
opinions)

---

## 22. The report: `tools/report.py`

### Build, don't adopt

Surveyed: [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2),
[PaperOrchestra](https://arxiv.org/pdf/2604.05018),
[Jr. AI Scientist](https://arxiv.org/pdf/2511.04583),
[Denario](https://arxiv.org/pdf/2510.26887),
[Camyla](https://arxiv.org/pdf/2604.10696),
[CiteLLM](https://arxiv.org/html/2602.23075).

Every one of them **reconstructs provenance from unstructured experiment logs.** Grad's is
already structured: expectations with `basis` and `comparability`, runs with results and
`deviations`, verdicts with notes, figures, corpus paper ids. That is strictly better input
than any of these systems receive. Adopting one means discarding the advantage and conforming
to its log format.

### The two structural guarantees

The hard problem in machine-written papers is not prose. It is hallucinated citations and
unsupported claims. Both can be made **structurally impossible** here:

**Citations.** `report cite` resolves only against the local corpus (`core/corpus.py` has the
ids) plus S2-verified ids (`core/http.py` already talks to S2). A `\cite{}` key with no
resolved entry is a hard error, not a warning.

**Claims.** Every asserted number carries a `\gradnum{<key>}` macro backed by a `claims.json`
sidecar mapping each key to `(run_id, quantity)`. `report check` verifies each against the
ledger. A number that does not resolve fails the check.

No off-the-shelf harness can do either, because none of them know about the ledger.

### Subcommands

```bash
python -m tools.report draft --project proj-scaling-w2 --json  # deterministic, no LLM
python -m tools.report write --project proj-scaling-w2 --json  # prose + [CITE:...] placeholders
python -m tools.report cite  --project proj-scaling-w2 --json  # resolve, verify, emit .bib
python -m tools.report check --project proj-scaling-w2 --json  # the gate
python -m tools.report build --project proj-scaling-w2 --json  # PDF
```

`draft` emits the skeleton from the ledger — every expectation, its runs, its deviations, its
verdict, its figures — with no model in the loop. It is useful on its own and costs nothing.

`check` enforces, in order:

1. every `\gradnum{}` key resolves to a `(run_id, quantity)` present in the ledger, with a
   matching value;
2. every `\cite{}` key exists in `references.bib`, and every bib entry came from the corpus or
   a verified S2 id;
3. **no cited run has an unjudged deviation** — `collect` already computes `needs_verdict`
   ([tools/jobs.py:435](tools/jobs.py:435));
4. the LaTeX compiles clean — no unmatched braces, duplicate labels, or unescaped specials.

Rule 3 is the one most in the spirit of this system: **you should not be able to write up a
result you have not judged.**

`check` refuses; it does not warn. A report generator is where this system's epistemics either
hold or collapse — the whole design exists to stop the user believing results too easily, and
a paper generator is a machine for asserting them confidently.

### What to steal, and from whom

- **Camyla** — the two-pass citation flow. Write with `[CITE:keyword]` placeholders, then
  resolve by extracting a context window around each and verifying the S2 candidate's title
  and abstract against that context. Much better than citing inline.
- **PaperOrchestra** — the constraint set: cite keys must match `references.bib` exactly, no
  fabricated results, compile-clean LaTeX. Encode as validation, not as prompt text.
- **Denario** — progressive versions. It emits four because unattended LaTeX does not reliably
  compile. Copy the checkpointing; it is the honest response.
- **AI Scientist v2** — the role split (`--model_writeup` / `--model_citation` /
  `--model_review`), which maps onto §16's roles.
- **Jr. AI Scientist** — draft → reflect → adjust, working directly inside a conference
  template directory.

Template: vendor NeurIPS or ICML style. Not a decision worth deliberating.

---

## 23. Open questions

Things genuinely unresolved, listed so they are not mistaken for settled:

1. **Does ShinkaEvolve expose a per-candidate callback?** Decides driver-vs-fork (§21). Check
   `ShinkaEvolveRunner` before starting.
2. **Context7's exact REST endpoints.** Read `context7.com/docs/api-guide` (§18).
3. **Does `headless/claude` work against a Max subscription specifically?** Reported in
   release notes, not tested here (§21).
4. **What granularity should campaign sub-runs have in `runs.jsonl`?** One record per
   candidate is honest but will dominate the ledger — a 100-generation campaign is thousands
   of rows. Consider a separate `candidates.jsonl` folded into the campaign record, with only
   promoted candidates entering `runs.jsonl` (§21).
5. **Should `report write` be allowed to run at all while a project is over budget?** Argument
   for yes: the report is how you find out what the spend bought. Argument for no: it is a
   cost-bearing loop like any other. Currently specified as denied by the §15 hook; revisit
   after first use.
6. **Whether the `project` dimension should be retrofitted onto historical records** or left
   as `"unassigned"`. Specified as the latter; cheap to change while the ledger is small.

---

## 24. Build order

Ordering matters more than usual here, because §15 became a prerequisite for three others.

**Foundation — ~2.5 days**

1. **§16** — `[models]` section, Opus 5 default, role-tagged quota stages. ~2 hours. Unblocks
   everything and touches nothing risky.
2. **§15** — `project` dimension, `tools/budget.py`, quota and credit ceilings, exit code 12,
   UI meter. ~2 days. Prerequisite for 3, 7, 8.
3. **§17** — HF org namespaces, consuming the project `payer`. ~0.5 days.

**Tools — ~2.5 days**

4. **§18** — `tools/docs.py`. ~0.5 days.
5. **§19** — Verify button first, then `tools/lab.py` and the embed. ~1.5 days.
6. **§20** — `repowiki map` trial, then the fork if wanted. ~1 day.

**The big two — ~6–9 days**

7. **§21** — `tools/evolve.py`, local-only phase 1. ~3–5 days.
8. **§22** — `tools/report.py`. ~3–4 days.

Roughly 2.5–3 weeks. Items 1, 3, and 4 total about a day and a half and each stands alone, so
they are the sensible slice if value is wanted sooner.

**Testing.** Same discipline as the existing suite — a real ledger in a temp workspace, no
network, no SDK required ([README.md:190](README.md:190)). Straightforward for 1, 2, 3, and 6;
4 needs a faked HTTP layer; 7 needs a faked Shinka runner; 8 needs a fixture ledger with a
known-good and a known-bad claim set. The budget gates in 2 deserve the same treatment §6's
gates got: tested against a real ledger, not mocks, because they are what stands between a
loop and a bill.

---

*Written 2026-08-14. Verified in session: `huggingface_hub` 1.16.1 Jobs signatures; RepoWiki
0.3.1 wheel internals; ShinkaEvolve's Headless provider; Context7's auth model and current MCP
tool names; `jupyterlab-tabnine`'s release history. Unverified items are listed in §23.*
