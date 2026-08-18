"""Configuration: ceilings, caps, host inventory, model names.

Everything that a gate compares against lives here rather than in a prompt, and
the defaults are deliberately conservative -- a config file that fails to load
must not silently raise a ceiling.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import paths
from core.errors import ConfigError

DEFAULTS: dict[str, Any] = {
    "spend": {
        # HANDOFF §6: a per-invocation cap alone does not stop twenty invocations,
        # so both ceilings exist and both are checked at submit.
        "per_job_usd": 25.0,
        "monthly_usd": 200.0,
        "window_days": 30,
        # A run uncollected past estimate * grace + floor blocks new submissions.
        "stale_grace_factor": 3.0,
        "stale_grace_floor_s": 1800,
    },
    # What one token of each kind counts as against a `quota_tokens` ceiling.
    #
    # These exist because the ceiling used to count `input + output` and nothing
    # else, and on the first fortnight of real use that was 149k tokens out of
    # 12.5M actually moved -- 1.2% of the flow. The other 98.8% is cache reads,
    # which are what a long context costs on every tool round-trip, and the row
    # in the README promising that token spend is "bounded, not merely measured"
    # could not see any of it.
    #
    # The weights are ratios against one input token, taken from published
    # per-token pricing: a cache read is a tenth of an input token, a cache write
    # is 1.25 of one. Output is left at 1.0 rather than at its true multiple so
    # that an existing `quota_tokens` ceiling keeps roughly the meaning it had
    # for the two components it could already see -- this change is meant to
    # reveal the missing 98%, not to silently reprice the 1.2%.
    #
    # They are configuration and not constants for the reason §10 gives about
    # the meter as a whole: subscription quota is not linear in tokens and
    # Anthropic exposes no remaining balance, so this is a stated assumption you
    # control rather than a mirror of anyone's billing. Set them all to 1.0 for
    # a raw count, or `weight_cache_read = 0` to go back to what it did before.
    "quota": {
        "weight_input": 1.0,
        "weight_output": 1.0,
        "weight_cache_read": 0.1,
        "weight_cache_write": 1.25,
    },
    "smoke": {
        # HANDOFF §6: the carve-out is hard-capped in code, not in prose.
        # "nothing useful can be trained inside them".
        "max_steps": 1,
        "max_wall_clock_s": 600,
        "max_cost_usd": 0.50,
        "allow_artifact_upload": False,
    },
    "notebook": {
        "exec_timeout_s": 300,
        "verify_timeout_s": 1800,
        "kernel_name": "python3",
    },
    "retrieval": {
        "s2_base": "https://api.semanticscholar.org/graph/v1",
        "asta_base": "https://asta-tools.allen.ai/mcp/v1",
        # Papers with Code, as revived by Hugging Face. The `.com` site Meta
        # shut down is gone; this is the `.co` one, and its v1 API is anonymous,
        # read-only and documented by `github.com/huggingface/pwc-cli`.
        "pwc_base": "https://paperswithcode.co/api/v1",
        # arXiv's Atom API, used for one thing: abstracts in bulk. See
        # `arxiv_abstracts`.
        "arxiv_base": "https://export.arxiv.org/api/query",
        # Tier-1 sources switched off, whatever asks for them. Both doors onto
        # the Semantic Scholar corpus are shut here on latency grounds alone --
        # see `tools/paper_search.py:disabled_tier1`. Empty the list to have
        # them back; nothing else has to change.
        "tier1_disabled": ["asta", "s2"],
        "openrouter_base": "https://openrouter.ai/api/v1",
        # Which rail stage 2 rides: `auto` (Voyage if that key is stored, else
        # OpenRouter), or either name to pin it. Voyage serve this model
        # themselves and `embed_model` already needs that key, so the default
        # asks for one credential where it used to need two -- and OpenRouter
        # stays supported for anyone already billing through it. See
        # `core/http.py:rerank_provider`.
        "rerank_provider": "auto",
        # Read on both rails: `voyageai/` is stripped for Voyage's own API and
        # added back for OpenRouter's catalogue, so this one value is correct
        # either way. See `core/http.py:rerank_model`.
        "rerank_model": "voyageai/rerank-2.5",
        "embed_model": "voyage-4",
        "embed_dim": 1024,
        # Voyage bills per token and returns a token count but no price, so the
        # rate has to come from somewhere for `credits_usd` to be anything other
        # than structurally zero -- and a credit ceiling that cannot see one of
        # the two credit-spending paths is not a ceiling. Publisher's list price
        # per million tokens; wrong-but-present beats absent, and it is one line
        # to correct when the price moves.
        "embed_usd_per_1m_tokens": 0.06,
        # The same arithmetic for the same reason, one stage earlier. Only the
        # Voyage rail needs it: OpenRouter prices the call and reports it.
        "rerank_usd_per_1m_tokens": 0.05,
        # triage_model / expand_model moved to [models] triage / expand (§16).
        # They are still *readable* here as overrides -- see LEGACY_MODEL_KEYS --
        # but they are no longer defaulted here, so [models] is the one place a
        # role's default lives.
        "rrf_k": 60,
        "candidates": 300,
        "rerank_top": 50,
        "triage_top": 15,
        "cache_ttl_s": 604800,
        "request_timeout_s": 60,
        # The ceiling on ONE request end to end, as distinct from
        # `request_timeout_s`, which httpx applies per socket read.
        #
        # The distinction is the whole bug it exists to stop. Asta answers a
        # `tools/call` with an event stream and holds it open while it works,
        # sending `: ping` comments every 15s. Every ping resets the per-read
        # timeout, so a 60s read timeout never fires no matter how long the
        # server takes, and a buffered read of that stream waits for a close
        # that may never come. A total deadline is the only thing that bounds
        # it. 300s because a live `search_papers_by_relevance` measured 121s
        # and a bound below the real latency is just an outage.
        "request_deadline_s": 300,
        # Wall clock for the whole of stage 1, as distinct from the per-request
        # deadline above. Stage 0 turns one question into six queries and each
        # goes to two endpoints, so a five-minute request deadline is a one-hour
        # stage -- and the caller kills it long before the endpoints that work
        # can contribute anything. When this is spent the funnel stops issuing
        # tier-1 calls, keeps what it retrieved, and writes into the trace how
        # many queries it actually searched.
        "stage1_budget_s": 300,
        "min_request_interval_s": 1.1,  # unauthenticated S2 is ~1 req/s
        # Which tier-1 client does discovery: "pwc", "asta", "s2", "both"
        # (the two Semantic Scholar doors) or "all".
        #
        # Papers with Code by default because it is the one that *answers*.
        # Measured against the live services: pwc returns in 1-2s; Asta takes
        # ~121s for a search and ~283s to report that its own backend refused a
        # connection, and stage 0 multiplies that by six queries and two
        # endpoints, so every caller gives up before discovery finishes. S2's
        # own API stopped issuing keys to free-domain addresses, leaving a
        # personal account on the shared anonymous pool, which is rate limited
        # often enough that "no results" and "no key" are hard to tell apart.
        #
        # The trade is real and worth knowing: Asta is the only one of the three
        # with genuine full-text snippets, which is what §5 designed stage-3
        # triage around. Under pwc, triage reads the abstract -- fetched from
        # arXiv in one batched request, see `core/http.py:arxiv_abstracts`.
        "tier1": "pwc",
    },
    # HANDOFF-2 §18 listed the REST paths as unverified (§23 item 2). They are
    # now verified against the live API: `/api/v2/libs/search` returns
    # `{"results": [...]}` and `/api/v2/context` returns `{"codeSnippets": [...]}`
    # when `type=json` is passed. They stay configuration rather than constants
    # because a third-party API can move, and a 404 here should be a one-line
    # config edit rather than a code change.
    "docs": {
        "base": "https://context7.com",
        "resolve_path": "/api/v2/libs/search",
        "docs_path": "/api/v2/context",
        "request_timeout_s": 30,
        "cache_ttl_s": 86400,
        "min_request_interval_s": 0.5,
    },
    # HANDOFF-2 §22. The handoff says "vendor NeurIPS or ICML style. Not a
    # decision worth deliberating." Rather than commit a third-party .sty file
    # into this repo, the class and style are configuration: drop
    # `neurips_2024.sty` into reports/<project>/ and set
    # `documentclass = "article"` plus `style = "neurips_2024"` here. The default
    # compiles with a stock TeX installation and no vendored file.
    "report": {
        "documentclass": "article",
        "classoptions": "11pt",
        "style": "",
        "bibstyle": "plainnat",
    },
    "preflight": {
        "checks": ["tests", "dry_run", "smoke"],
        "test_command": ["pytest", "-q"],
        "dry_run_timeout_s": 900,
        "test_timeout_s": 900,
    },
    # How much may be in flight at once, and it is a ceiling for the same reason
    # every other number in this file is one.
    #
    # `max_concurrent_runs` is the one with teeth. `gates.check_stale` refuses
    # *every* later submission while *any* uncollected run is past its window, so
    # three parallel submissions open three collection windows at once and one
    # wedged job takes the other two down with it. Running things in parallel is
    # precisely what turns exit 7 from an occasional annoyance into the normal
    # state, and this is what stops it. Two is deliberately low: it is enough to
    # overlap a long job with a short one, which is the case worth having, and not
    # enough to lose track of what is out there.
    #
    # `max_concurrent_tasks` bounds local background commands (`tools/task.py`).
    # Different number because it bounds a different resource -- this machine's
    # CPU, not a backend's willingness to be polled.
    #
    # `default_jobs` is what `--jobs` defaults to where a tool parallelises inside
    # itself: the funnel's stage 1, preflight's independent checks. Network-bound
    # work, so it is the largest of the three.
    "execution": {
        "max_concurrent_runs": 2,
        "max_concurrent_tasks": 4,
        "default_jobs": 4,
    },
    "hf": {
        "default_flavor": "a10g-small",
        # HF Jobs report a job's start and end; the price of a flavor comes from
        # here. `collect` multiplies the two -- it never reuses the estimate.
        "flavor_rates": {
            "cpu-basic": 0.0,
            "cpu-upgrade": 0.03,
            "t4-small": 0.40,
            "t4-medium": 0.60,
            "a10g-small": 1.05,
            "a10g-large": 1.50,
            "a100-large": 4.13,
        },
    },
    # Modal Sandboxes: the fourth submitter, and the first whose billing model
    # the §6 dollar ceilings were already the right instrument for. Modal charges
    # per second against a published table, so unlike Kaggle there is nothing
    # here to ration in another unit -- `[spend]` is the gate, and `collect`
    # prices the elapsed time against `gpu_rates` below.
    "modal": {
        "default_gpu": "H100",
        "app_name": "grad",
        # A Volume, because a Sandbox's filesystem does not outlive it and
        # `collect` runs after it has exited. See `tools/modal.py`.
        "volume_name": "grad-runs",
        "mount_path": "/grad/out",
        "workdir": "/grad/pipeline",
        "poll_interval_s": 20,
        # Modal kills a Sandbox at 24 hours whatever it was doing, so a spec
        # asking for more is refused rather than started.
        "max_hours": 24.0,
        "timeout_margin": 1.25,
        # Dollars per hour, from Modal's published per-second prices. An
        # accelerator absent from this table is refused at submit rather than
        # booked at zero: `[spend]` is this backend's only gate, and a ceiling
        # that cannot price a run is not bounding it.
        "gpu_rates": {
            "T4": 0.5904,
            "L4": 0.7992,
            "A10G": 1.1016,
            "L40S": 1.9512,
            "A100-40GB": 2.0988,
            "A100-80GB": 2.4984,
            "H100": 3.9492,
            "H200": 4.5396,
            "B200": 6.2496,
        },
    },
    # Kaggle kernels: a third submitter, and the first whose scarce resource is
    # not money. Every run costs $0.00, so the §6 dollar ceilings can never
    # refuse one -- which would make them decoration on this backend rather than
    # a gate. What Kaggle actually rations is accelerator *hours*, weekly, so
    # that is what `core/kaggle_quota.py` counts and refuses against.
    "kaggle": {
        # Whose account. Not a credential: only the key is secret, and which
        # account is running the notebooks belongs somewhere you can read it.
        # `core/credentials.py:KAGGLE_KEY` holds the other half.
        #
        # Usually set by `python -m tools.kaggle account --set <username>`, which
        # writes it to the app directory instead of here -- this file is
        # hand-annotated and `tomllib` cannot write it back without discarding
        # every comment in it. That stored selection wins over this key; leaving
        # this empty is the normal case.
        "username": "",
        "default_accelerator": "NvidiaTeslaP100",
        # Kernel slugs are `<username>/<prefix>-<run_id>`, so every kernel this
        # tool pushes is identifiable as ours on the Kaggle side.
        "kernel_prefix": "grad",
        # Kaggle kernels are private by default here, and it is not a detail:
        # a public kernel publishes the pipeline, the data pointer and the
        # results to the internet the moment it is pushed.
        "is_private": True,
        # Off by default, and this one is a real trade. A kernel with internet
        # can `pip install` and pull weights; it can also exfiltrate anything it
        # can read. Turning it on is a per-spec decision (`[target] internet`),
        # not a default, because the default is the one nobody re-reads.
        "enable_internet": False,
        "poll_interval_s": 30,
        # Push and status calls are quick; `output` downloads every file the
        # kernel produced and is the one that can genuinely take minutes.
        "push_timeout_s": 900,
        "status_timeout_s": 120,
        "output_timeout_s": 1800,
        # How long a smoke run may sit queued before the poll gives up. Separate
        # from the wall-clock cap on the smoke itself: queue time is Kaggle being
        # busy, and charging it to a cap the spec asked for would fail the check
        # for a reason that has nothing to do with the code under test.
        "queue_grace_s": 900,
        # Which weekly pool each accelerator draws from. A table rather than a
        # constant for the reason `[docs]` gives: Kaggle adds hardware, and a new
        # id should be a one-line edit here rather than a code change. An id that
        # is not in this table is a configuration error and never a guess --
        # guessing "gpu" for a TPU would charge the wrong pool, and guessing
        # "cpu" would charge no pool at all, which is the failure that makes a
        # quota gate decoration.
        "accelerators": {
            "none": "cpu",
            "NvidiaTeslaP100": "gpu",
            "NvidiaTeslaT4": "gpu",
            "NvidiaTeslaT4Highmem": "gpu",
            "NvidiaTeslaA100": "gpu",
            "NvidiaL4": "gpu",
            "NvidiaL4X1": "gpu",
            "NvidiaH100": "gpu",
            "NvidiaRtxPro6000": "gpu",
            "TpuV38": "tpu",
            "Tpu1VmV38": "tpu",
            "TpuV5E8": "tpu",
            "TpuV6E8": "tpu",
        },
        # The published allowances, which Kaggle varies with demand and does not
        # expose as a remaining balance. So this is the same kind of number as
        # `[quota]`'s token ceiling: **a proxy you control, not a mirror of
        # Kaggle's limit.** Set it under the real allowance and it binds first,
        # which is the point -- a gate that refuses at 30.0h when Kaggle would
        # have allowed 34 costs you four hours; one that refuses at 34 when
        # Kaggle stops at 30 loses you a run mid-training.
        "quota": {
            "gpu_hours_per_week": 30.0,
            "tpu_hours_per_week": 20.0,
            "window_days": 7,
            # A single session's ceiling, which is a different failure from the
            # weekly one: Kaggle kills the kernel at the cap and the run dies
            # with whatever it had not checkpointed. Refusing before the push is
            # the only place that is cheap.
            "max_session_hours": 12.0,
            "max_tpu_session_hours": 9.0,
        },
    },
    # HANDOFF-2 §16: models are selected by *role*, not scattered across
    # [agent] and [retrieval]. Five surfaces, six roles; the rerank and embed
    # models deliberately stay in [retrieval] because they are a different
    # provider on a different billing rail, and folding them in here invites
    # the Voyage-for-Haiku substitution §16 argues against.
    "models": {
        "research": "claude-opus-5",        # the main loop (§3)
        "evolve": "claude-sonnet-5",        # ShinkaEvolve mutation operators (§21)
        "expand": "claude-haiku-4-5",       # funnel stage 0 (§5)
        "triage": "claude-haiku-4-5",       # funnel stage 3 (§5)
        "report": "claude-opus-5",          # prose synthesis (§22)
        "cite": "claude-haiku-4-5",         # citation resolution -- mechanical matching (§22)
        # One page of a project wiki (`core/wikigen.py`). Sonnet rather than
        # Opus because the job is bounded: explain a fact sheet that was handed
        # to you, in a schema that will not accept prose without citations. And
        # rather than Haiku because it is still *writing* -- the page has to say
        # why the pieces are arranged this way, which is the part of the job that
        # is not extraction, and a build is a handful of calls a person is
        # waiting on rather than a per-turn cost.
        "wiki": "claude-sonnet-5",
    },
    "agent": {
        "model": "claude-opus-5",
        "permission_mode": "dontAsk",
        "max_turns": 0,  # 0 = unbounded
        # Whether the reasoning arrives as *text*, which is what the chat
        # window's statusline switches on and off.
        #
        # This is not the same question as whether the model thinks. Opus 4.7+
        # defaults `display` to "omitted" and sends thinking blocks with a
        # signature and no text, so a client that captures reasoning correctly
        # still has nothing to show -- which is exactly what a toggle over an
        # empty transcript looks like. "summarized" is the only value that
        # actually produces text; "omitted" is the SDK's own default and is here
        # so turning the feature off is a config edit rather than a code one.
        "reasoning": "summarized",
        # Compact the conversation once it passes this many tokens of context.
        # 0 disables it and leaves the matter to the CLI underneath.
        #
        # There is a threshold either way -- the CLI autocompacts on its own, and
        # a live session reports it as 967,000 of a 1,000,000 window. That is a
        # ceiling in the sense that a wall at the end of a runway is: by the time
        # it is reached every tool round-trip has been re-reading the better part
        # of a million cached tokens for a long time. 300k is roughly a third of
        # the way in, which keeps a long session's per-turn cost bounded while
        # leaving room for the kind of turn this agent actually runs -- the
        # largest one in the ledger so far read 10.1M cached tokens.
        #
        # Compacting is not free and not obviously cheap: the summary costs a
        # turn, and the session it seeds starts with a cold prompt cache, so the
        # first turn after a compaction pays cache *writes* (1.25x) where it
        # would have paid cache *reads* (0.1x). Compacting too eagerly costs more
        # than not compacting. `python -m tools.quota summary --json` is where
        # that trade becomes visible, which is why the accounting split landed
        # before this did.
        "compact_at_tokens": 300_000,
        # How much of the selected project's `MEMORY.md` reaches the system
        # prompt. Characters rather than tokens: this is read while a session is
        # being built and loading a tokeniser there to make a bound exact would
        # cost more than the bound is worth. Roughly four characters to the
        # token, so this is about 4k.
        #
        # It is a ceiling on a *recurring* cost, which is what makes it worth
        # having at all. The system prompt is re-read from cache on every tool
        # round-trip of every turn, so a memory file that grows without bound
        # becomes the dominant line in `quota summary` without ever appearing as
        # a decision anyone made. 0 disables project memory entirely.
        "memory_max_chars": 16_000,
        # How hard the main agent thinks: auto | low | medium | high | xhigh |
        # max. "auto" passes nothing and leaves it to the CLI, which is what
        # every session did before the knob existed.
        #
        # This is only the *starting point*. The live selection is made from the
        # chat statusline and kept in the app directory, because the right level
        # is a property of the task and changes several times an afternoon --
        # and because this file is hand-annotated and cannot be written back
        # without losing its comments. See `core/effort.py`.
        "effort": "auto",
    },
    # The cross-workspace experiment archive (`core/experiments.py`).
    "experiments": {
        # Artifacts above this size are recorded with their length and no
        # digest. Hashing a multi-gigabyte checkpoint on the `collect` path
        # would turn bookkeeping into a visible stall; the size alone still
        # catches truncation, which is the common corruption.
        "hash_max_bytes": 64 * 1024 * 1024,
    },
    "hosts": {},
}

# One release of backwards compatibility, so an existing config keeps working.
# The old key wins over the [models] *default* but not over an explicit
# [models] entry -- see `model_for`.
LEGACY_MODEL_KEYS: dict[str, tuple[str, str]] = {
    "research": ("agent", "model"),
    "expand": ("retrieval", "expand_model"),
    "triage": ("retrieval", "triage_model"),
}

MODEL_ROLES = tuple(DEFAULTS["models"])


@dataclass(frozen=True)
class Host:
    """An SSH GPU host. `rate_usd_per_hour` may be 0 for hosts that are free."""

    name: str
    hostname: str
    user: str
    rate_usd_per_hour: float = 0.0
    workdir: str = "~/grad"
    key_credential: str | None = None  # keyring entry name; never a path to a key
    gpus: int = 1
    notes: str = ""


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    # What the file actually said, before the defaults were merged under it.
    # `model_for` needs the difference: an explicit [models] entry must beat a
    # legacy key, while the [models] *default* must not.
    user: dict[str, Any] = field(default_factory=dict)
    # `core/settings.py`: the writable half, chosen through the setup wizard and
    # stored under the app directory rather than in the hand-annotated TOML.
    # Outranks everything in the file, which is `kaggle account`'s rule and for
    # the same reason -- a command that silently did nothing because a config
    # file disagreed would be worse than one that overrides it, and
    # `settings.shadowing` is what stops that being a surprise.
    overlay: dict[str, Any] = field(default_factory=dict)
    # The selected project's own overrides (`core/budget.py:configure`). Kept
    # apart from `overlay` rather than merged into it, because "what would this
    # role be without the project" is a question the projects window has to
    # answer -- it draws every project, and only one of them is selected.
    project_overlay: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """One setting, overlay first.

        The overlay outranking the file is this dataclass's stated rule and was
        already true for the three things that had their own accessor -- models,
        the default backend, the host inventory. It was not true for anything
        read through `get`, which is every ordinary scalar, so a setup window
        that wrote `[agent] compact_at_tokens` into the overlay would have
        written a value nothing ever read.

        Safe to widen because the overlay is not free-form: `core/settings.py`
        is the only writer and every key it will write is on one of its
        allowlists. A section the overlay does not mention resolves exactly as
        it did before.
        """
        chosen = self.overlay.get(section)
        if isinstance(chosen, dict) and key in chosen:
            return chosen[key]
        return self.raw.get(section, {}).get(key, default)

    def model_for(self, role: str, *, project: bool = True) -> str:
        """The model for one role (HANDOFF-2 §16).

        Resolution, outermost first: the selected project's own override
        (`core/budget.py:configure`); then a `core/settings.py` overlay entry
        (what the setup window chose); then an explicit `[models] <role>`; then
        the legacy key the role replaced (`[agent] model`,
        `[retrieval] expand_model` / `triage_model`), readable "for one release
        so existing configs do not break"; then the `[models]` default.

        `project=False` answers the same question with the outermost layer
        removed -- "what would this be if the project said nothing". The
        projects window needs it because it draws every project and only one of
        them is selected, so for the rest the project layer in effect is not
        theirs.
        """
        if role not in DEFAULTS["models"]:
            raise ConfigError(
                f"unknown model role {role!r}",
                fix=f"roles are: {', '.join(MODEL_ROLES)}",
            )
        if project:
            chosen = (self.project_overlay.get("models") or {}).get(role)
            if chosen:
                return str(chosen)
        chosen = (self.overlay.get("models") or {}).get(role)
        if chosen:
            return str(chosen)
        explicit = (self.user.get("models") or {}).get(role)
        if explicit:
            return str(explicit)
        legacy = LEGACY_MODEL_KEYS.get(role)
        if legacy:
            value = (self.user.get(legacy[0]) or {}).get(legacy[1])
            if value:
                return str(value)
        return str(self.get("models", role, DEFAULTS["models"][role]))

    def models(self) -> dict[str, str]:
        return {role: self.model_for(role) for role in MODEL_ROLES}

    @property
    def hosts(self) -> dict[str, Host]:
        """The inventory, with malformed entries reported as ConfigError.

        A bad value here is a typo in a TOML file, not a bug, and it must not
        surface as a bare ValueError from inside a submitter -- `rate_usd_per_hour`
        in particular is what `collect` prices wall clock against, so getting it
        wrong is a spend-accounting problem and deserves a real message.
        """
        raw = self.raw.get("hosts", {})
        if not isinstance(raw, dict):
            raise ConfigError(
                f"[hosts] must be a table of host entries, not {type(raw).__name__}",
                fix=f"see the [hosts.*] example in {paths.config_path()}",
            )
        # The inventory has two sources and stays fixed: `[hosts.*]` in the TOML,
        # and whatever `setup host add` wrote. Combined by name, so a machine
        # that had hosts in its config keeps them, and validated below by the
        # same code either way -- a host added through the wizard is not a host
        # that skipped the rate check.
        #
        # **Whole entries, not `_merge`.** A recursive merge would have an
        # overlay host inherit the fields it omitted from the config host of the
        # same name -- including `key_credential`, which names the keyring entry
        # that authenticates the connection. Replacing `gpu-box` through the
        # wizard and getting the old box's credential, user and workdir attached
        # to the new hostname is a connection nobody described, and it is the one
        # field in this table where being wrong reaches a machine.
        overlay_hosts = self.overlay.get("hosts")
        if isinstance(overlay_hosts, dict) and overlay_hosts:
            raw = {**raw, **{k: v for k, v in overlay_hosts.items() if isinstance(v, dict)}}
        out: dict[str, Host] = {}
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                raise ConfigError(
                    f"host {name!r} must be a table, not {type(spec).__name__}",
                    fix=f"write it as [hosts.{name}] with hostname/user/rate_usd_per_hour keys",
                )
            rate = spec.get("rate_usd_per_hour", 0.0)
            try:
                # A negative rate would make `collect` book negative actuals,
                # which *reduce* rolling spend -- a typo that raises the ceiling.
                # Zero is legitimate (a host that is free to use is still
                # ledgered); below zero is not. Neither is nan, which fails
                # every comparison a gate makes against it, or inf, which is a
                # price no run can be under.
                if not math.isfinite(float(rate)):
                    raise ConfigError(
                        f"host {name!r} has a non-finite rate_usd_per_hour ({rate})",
                        fix="rate_usd_per_hour must be a finite number; use 0 for a free host",
                    )
                if float(rate) < 0:
                    raise ConfigError(
                        f"host {name!r} has a negative rate_usd_per_hour ({rate})",
                        fix="use 0 for a host that is free to use; negative spend is not a thing",
                    )
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"host {name!r} has a malformed rate_usd_per_hour: {rate!r}",
                    fix="rate_usd_per_hour must be a number",
                ) from exc
            try:
                out[name] = Host(
                    name=name,
                    hostname=str(spec.get("hostname", "")),
                    user=str(spec.get("user", "")),
                    rate_usd_per_hour=float(rate),
                    workdir=str(spec.get("workdir", "~/grad")),
                    key_credential=spec.get("key_credential"),
                    gpus=int(spec.get("gpus", 1)),
                    notes=str(spec.get("notes", "")),
                )
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"host {name!r} has a malformed value: {exc}",
                    fix="rate_usd_per_hour must be a number and gpus an integer",
                ) from exc
        return out

    def host(self, name: str) -> Host:
        """Hosts are a hardcoded inventory (HANDOFF §9). An unknown name is a
        configuration error, never an ad-hoc connection."""
        hosts = self.hosts
        if name not in hosts:
            known = ", ".join(sorted(hosts)) or "(none configured)"
            raise ConfigError(
                f"unknown host {name!r}; the inventory is fixed. known hosts: {known}",
                # Both halves named, because there are now two places a host can
                # be defined and the wrong guess costs an edit to a file that was
                # never going to be read.
                fix=(
                    f"python -m tools.setup host add --name {name} --hostname … --user … --json"
                    f"   # or add a [hosts.{name}] block to {paths.config_path()}"
                ),
            )
        return hosts[name]

    def accelerator_kind(self, name: str) -> str:
        """Which weekly pool a Kaggle accelerator draws from: gpu, tpu, or cpu.

        The same argument `host()` makes, for the same reason. An unrecognised
        accelerator id is a configuration error rather than a default, because
        every available default is wrong in a way that matters: falling back to
        `"gpu"` charges a TPU run to the GPU allowance, and falling back to
        `"cpu"` charges it to nothing at all -- and a run that draws no pool is
        exactly how a weekly ceiling quietly stops bounding anything.
        """
        table = self.get("kaggle", "accelerators", {}) or {}
        if name not in table:
            known = ", ".join(sorted(table)) or "(none configured)"
            raise ConfigError(
                f"unknown Kaggle accelerator {name!r}; known accelerators: {known}",
                fix=(
                    f"add `{name} = \"gpu\"` (or tpu/cpu) to [kaggle.accelerators] in "
                    f"{paths.config_path()} -- the value names the weekly pool it draws from"
                ),
            )
        return str(table[name])


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


_cache: dict[str, Config] = {}


def load(path: Path | None = None, *, reload: bool = False) -> Config:
    from core import budget as budget_mod, settings as settings_mod  # noqa: PLC0415

    path = Path(path) if path else paths.config_path()
    # The mtimes are part of the key, not just the path. `setup models` and
    # `budget configure` run as child processes and write files this process has
    # already read -- so without this the app goes on serving whatever it loaded
    # at startup and the setup window appears to do nothing. Three `stat` calls
    # and one tiny read per load, against a config that is read on the gate
    # path; the alternative is every writer remembering to clear a cache in a
    # module it does not import.
    project_id = budget_mod.current_project()
    key = f"{path}\x00{settings_mod.stamp()}\x00{project_id}\x00{budget_mod.selection_stamp()}"
    if not reload and key in _cache:
        return _cache[key]
    user: dict[str, Any] = {}
    if path.exists():
        try:
            user = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"{path} is not valid TOML: {exc}",
                fix=f"fix the syntax in {path}, or delete it to fall back to defaults",
            ) from exc
    cfg = Config(
        raw=_merge(DEFAULTS, user),
        user=user,
        overlay=settings_mod.load(),
        project_overlay=budget_mod.project_overrides(project_id),
    )
    _validate(cfg, path)
    _cache[key] = cfg
    return cfg


# The numbers a gate compares against. A string where a float belongs would
# otherwise surface as a TypeError from inside a ceiling check, which reads like
# a bug in the gate rather than a typo in a config file.
_NUMERIC = (
    ("spend", "per_job_usd"),
    ("spend", "monthly_usd"),
    ("spend", "window_days"),
    ("spend", "stale_grace_factor"),
    ("spend", "stale_grace_floor_s"),
    ("smoke", "max_steps"),
    ("smoke", "max_wall_clock_s"),
    ("smoke", "max_cost_usd"),
    # Both bound a wall clock, and both are read straight into arithmetic on
    # `time.monotonic()`. Unvalidated, a string here is a TypeError from inside
    # a retrieval loop and a negative is a deadline that has already expired --
    # every search failing instantly with a timeout message, which reads as an
    # outage at the endpoint rather than as a typo in a config file.
    ("retrieval", "request_deadline_s"),
    ("retrieval", "stage1_budget_s"),
    # All three are read straight into `max(1, int(...))` and into a ceiling
    # comparison. A string here is a TypeError from inside the gate that is
    # refusing a submission, which reads like a bug in the gate.
    ("execution", "max_concurrent_runs"),
    ("execution", "max_concurrent_tasks"),
    ("execution", "default_jobs"),
    ("kaggle", "poll_interval_s"),
    ("kaggle", "push_timeout_s"),
    ("kaggle", "status_timeout_s"),
    ("kaggle", "output_timeout_s"),
    ("kaggle", "queue_grace_s"),
)

# The nested ones. `_NUMERIC` addresses `raw[section][key]` and the Kaggle
# allowances live one table deeper, in `[kaggle.quota]` -- so they would have
# been validated by nothing at all, which is the wrong place for a string to
# surface: `"thirty"` there is a TypeError from inside the gate that is refusing
# a submission, and it reads like a bug in the gate.
_NUMERIC_NESTED = (
    ("kaggle", "quota", "gpu_hours_per_week"),
    ("kaggle", "quota", "tpu_hours_per_week"),
    ("kaggle", "quota", "window_days"),
    ("kaggle", "quota", "max_session_hours"),
    ("kaggle", "quota", "max_tpu_session_hours"),
)


def _check_number(value: Any, dotted: str, path: Path) -> None:
    """One config number, or a ConfigError naming it.

    Factored out rather than written twice: `[kaggle.quota]` needs exactly these
    three checks and a second copy is a second place for them to drift.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"[{dotted.rsplit('.', 1)[0]}] {dotted.rsplit('.', 1)[1]} must be a number, "
            f"not {type(value).__name__}",
            fix=f"fix {dotted} in {path}",
        )
    # TOML has literal `nan` and `inf`, so these reach the gates as ordinary
    # floats. Neither belongs in a ceiling: NaN fails every comparison, so a
    # gate written as `if spend > ceiling` waves everything through, and inf
    # is a ceiling that can never be reached. Both read as "no limit" while
    # looking like a number in the file.
    if not math.isfinite(value):
        raise ConfigError(
            f"{dotted} must be a finite number, not {value}",
            fix=(
                f"fix {dotted} in {path}; nan and inf are valid TOML floats "
                "but neither can bound anything"
            ),
        )
    if value < 0:
        raise ConfigError(
            f"{dotted} must not be negative",
            fix=f"fix {dotted} in {path}",
        )


def _validate(cfg: Config, path: Path) -> None:
    """Check the shapes a malformed file could break, before anything is cached."""
    # `Config.get` subscripts the section, so `spend = "lots"` in the file would
    # surface as an AttributeError from inside the loop below -- a traceback that
    # reads like a bug in the loader rather than a typo in a TOML file.
    for section in (*dict.fromkeys(s for s, _ in _NUMERIC), "hf", "models", "kaggle"):
        table = cfg.raw.get(section)
        if table is not None and not isinstance(table, dict):
            raise ConfigError(
                f"[{section}] must be a table, not {type(table).__name__}",
                fix=f"write it as a [{section}] section with its keys underneath, in {path}",
            )
    for section, key in _NUMERIC:
        value = cfg.get(section, key)
        if value is None:
            continue
        _check_number(value, f"{section}.{key}", path)
    for section, table_name, key in _NUMERIC_NESTED:
        table = cfg.get(section, table_name)
        if table is None:
            continue
        if not isinstance(table, dict):
            raise ConfigError(
                f"[{section}.{table_name}] must be a table, not {type(table).__name__}",
                fix=f"write it as a [{section}.{table_name}] section in {path}",
            )
        if table.get(key) is not None:
            _check_number(table[key], f"{section}.{table_name}.{key}", path)
    _validate_kaggle(cfg, path)
    # A model id is a string. An integer or a list here would surface as an
    # opaque SDK error on the first turn rather than as a typo in a TOML file,
    # and an unknown role name is a silently ignored setting -- which is worse,
    # because the model it names is never used and nothing says so.
    for role, value in (cfg.raw.get("models") or {}).items():
        if role not in DEFAULTS["models"]:
            raise ConfigError(
                f"[models] {role} is not a model role",
                fix=f"roles are: {', '.join(MODEL_ROLES)} (in {path})",
            )
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"[models] {role} must be a model id string, not {type(value).__name__}",
                fix=f'write it as {role} = "claude-opus-5" in {path}',
            )
    rates = cfg.get("hf", "flavor_rates", {})
    if not isinstance(rates, dict):
        raise ConfigError(
            "[hf.flavor_rates] must be a table of flavor -> dollars per hour",
            fix=f"fix the [hf.flavor_rates] section in {path}",
        )
    modal_rates = cfg.get("modal", "gpu_rates", {})
    if not isinstance(modal_rates, dict):
        raise ConfigError(
            "[modal.gpu_rates] must be a table of GPU name -> dollars per hour",
            fix=f"fix the [modal.gpu_rates] section in {path}",
        )
    for name, value in modal_rates.items():
        # Checked here rather than at submit, where the failure would be a run
        # that got as far as the ceiling before anything noticed the ceiling
        # could not be computed.
        #
        # `_check_number` rather than a bespoke test, for the reason its own
        # docstring gives about `[kaggle.quota]`: a second copy is a second
        # place to drift, and this one had already drifted -- it caught bools
        # and non-numbers and let `nan` through, which TOML has a literal for
        # and which makes every comparison against a ceiling false.
        _check_number(value, f"modal.gpu_rates.{name}", path)
        if value < 0:
            raise ConfigError(
                f"[modal.gpu_rates] {name} must not be negative",
                fix=f'write it as "{name}" = 3.9492 in {path}',
            )
    cfg.hosts  # noqa: B018 - raises ConfigError on a malformed inventory


#: The pools a Kaggle accelerator can draw from. `cpu` draws from none, which is
#: why it is a named kind rather than an absence: a CPU-only kernel is a real,
#: unmetered thing to run, and treating "no pool" as "unknown" would refuse it.
ACCELERATOR_KINDS = ("gpu", "tpu", "cpu")


def _validate_kaggle(cfg: Config, path: Path) -> None:
    """The accelerator inventory and the weekly allowances.

    Checked at load rather than at submit, for the reason the whole module
    exists: a malformed table must never be discovered by the gate that was
    about to use it. An accelerator mapped to `"GPU"` or to `"tpu "` is the case
    worth catching -- the lookup would miss, the run would be charged to no pool,
    and the weekly ceiling would silently stop counting the thing it exists to
    count.
    """
    table = cfg.get("kaggle", "accelerators", {})
    if not isinstance(table, dict):
        raise ConfigError(
            "[kaggle.accelerators] must be a table of accelerator id -> gpu|tpu|cpu",
            fix=f"fix the [kaggle.accelerators] section in {path}",
        )
    for name, kind in table.items():
        if not isinstance(kind, str) or kind not in ACCELERATOR_KINDS:
            raise ConfigError(
                f"[kaggle.accelerators] {name} must be one of "
                f"{', '.join(ACCELERATOR_KINDS)}, not {kind!r}",
                fix=(
                    f"fix kaggle.accelerators.{name} in {path}; the value names which weekly "
                    "pool the accelerator draws from, and an unrecognised one would charge none"
                ),
            )
    default = cfg.get("kaggle", "default_accelerator", "")
    if default and default not in table:
        raise ConfigError(
            f"[kaggle] default_accelerator {default!r} is not in [kaggle.accelerators]",
            fix=(
                f"add it to [kaggle.accelerators] in {path} with its pool (gpu/tpu/cpu), "
                f"or pick one of: {', '.join(sorted(table))}"
            ),
        )
