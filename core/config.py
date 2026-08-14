"""Configuration: ceilings, caps, host inventory, model names.

Everything that a gate compares against lives here rather than in a prompt, and
the defaults are deliberately conservative -- a config file that fails to load
must not silently raise a ceiling.
"""

from __future__ import annotations

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
        "openrouter_base": "https://openrouter.ai/api/v1",
        "rerank_model": "voyageai/rerank-2.5",
        "embed_model": "voyage-4",
        "embed_dim": 1024,
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
        "min_request_interval_s": 1.1,  # unauthenticated S2 is ~1 req/s
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
    },
    "agent": {
        "model": "claude-opus-5",
        "permission_mode": "dontAsk",
        "max_turns": 0,  # 0 = unbounded
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

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def model_for(self, role: str) -> str:
        """The model for one role (HANDOFF-2 §16).

        Resolution: an explicit `[models] <role>` wins; then the legacy key the
        role replaced (`[agent] model`, `[retrieval] expand_model` /
        `triage_model`), readable "for one release so existing configs do not
        break"; then the `[models]` default.
        """
        if role not in DEFAULTS["models"]:
            raise ConfigError(
                f"unknown model role {role!r}",
                fix=f"roles are: {', '.join(MODEL_ROLES)}",
            )
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
        out: dict[str, Host] = {}
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                raise ConfigError(
                    f"host {name!r} must be a table, not {type(spec).__name__}",
                    fix=f"write it as [hosts.{name}] with hostname/user/rate_usd_per_hour keys",
                )
            try:
                out[name] = Host(
                    name=name,
                    hostname=str(spec.get("hostname", "")),
                    user=str(spec.get("user", "")),
                    rate_usd_per_hour=float(spec.get("rate_usd_per_hour", 0.0)),
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
                fix=f"add a [hosts.{name}] block to {paths.config_path()}",
            )
        return hosts[name]


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
    path = Path(path) if path else paths.config_path()
    key = str(path)
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
    cfg = Config(raw=_merge(DEFAULTS, user), user=user)
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
)


def _validate(cfg: Config, path: Path) -> None:
    """Check the shapes a malformed file could break, before anything is cached."""
    # `Config.get` subscripts the section, so `spend = "lots"` in the file would
    # surface as an AttributeError from inside the loop below -- a traceback that
    # reads like a bug in the loader rather than a typo in a TOML file.
    for section in (*dict.fromkeys(s for s, _ in _NUMERIC), "hf", "models"):
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"[{section}] {key} must be a number, not {type(value).__name__}",
                fix=f"fix {section}.{key} in {path}",
            )
        if value < 0:
            raise ConfigError(
                f"[{section}] {key} must not be negative",
                fix=f"fix {section}.{key} in {path}",
            )
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
    cfg.hosts  # noqa: B018 - raises ConfigError on a malformed inventory
