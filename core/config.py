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
        "triage_model": "claude-haiku-4-5",
        "expand_model": "claude-haiku-4-5",
        "rrf_k": 60,
        "candidates": 300,
        "rerank_top": 50,
        "triage_top": 15,
        "cache_ttl_s": 604800,
        "request_timeout_s": 60,
        "min_request_interval_s": 1.1,  # unauthenticated S2 is ~1 req/s
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
    "agent": {
        "model": "claude-opus-4-5",
        "permission_mode": "dontAsk",
        "max_turns": 0,  # 0 = unbounded
    },
    "hosts": {},
}


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

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

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
    cfg = Config(raw=_merge(DEFAULTS, user))
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
    rates = cfg.get("hf", "flavor_rates", {})
    if not isinstance(rates, dict):
        raise ConfigError(
            "[hf.flavor_rates] must be a table of flavor -> dollars per hour",
            fix=f"fix the [hf.flavor_rates] section in {path}",
        )
    cfg.hosts  # noqa: B018 - raises ConfigError on a malformed inventory
