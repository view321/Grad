"""The writable settings overlay: what a wizard is allowed to change.

`config/grad.toml` is hand-annotated, and every value in it is annotated because
the reason for the value is worth more than the value. `tomllib` reads TOML and
does not write it, so a command that edited that file would reformat it and drop
every comment in it -- which is why nothing in this project ever has. The README
says so plainly, and `tools/kaggle.py` already solved the problem once: `account
--set` writes its choice to a file under the app directory, that choice *wins*
over `[kaggle] username`, and the command says when it is shadowing one.

This is that mechanism, generalised, so an interactive setup can answer "which
model for which role", "which backend by default" and "which SSH hosts exist"
without touching a line anybody wrote.

**Per workspace, not per machine.** `paths.config_path()` is
`_shipped("config", "grad.toml")` -- the workspace's copy when it has one, the
installation's otherwise -- so `grad.toml` is *already* overridable per
workspace. A machine-global overlay would silently flatten two workspaces that
had deliberately different model choices. An overlay has to mirror the scope of
the file it shadows, so this lives in `appdata.workspace_state_dir()`, keyed by
root, beside the window layouts.

It stays out of the workspace *folder* for the reason `core/workspace.py` keeps
the root pointer beside the code: these are answers about how this install is
wired up, and a research folder handed to a colleague should not carry this
machine's SSH inventory with it.

**What is not here.** Credentials: those are `core/credentials.py` and the
operating system's store, and a token in a JSON file under the app directory
would be exactly the environment-resident secret §9 argues against. Ceilings:
those are `core/budget.py`, append-only, because a ceiling that moved is an
event and not a setting.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any

from core import appdata, jsonl
from core.errors import UsageError

#: The schema version, for a migration that has not been needed yet. Written so
#: that the first one does not have to guess what it is reading.
VERSION = 1

#: Where a candidate can be evaluated. The same three names `tools/evolve.py`
#: uses for `--remote` and the run records use for `platform` -- one vocabulary,
#: and `tests/test_settings.py` asserts the two lists still agree.
#:
#: In `core/` rather than in the tool, because a *setting* naming a backend is
#: read by the config layer, and `core` importing `tools` is backwards.
BACKENDS: tuple[str, ...] = ("ssh", "hf_jobs", "kaggle", "modal")

#: The models the setup window offers as buttons. **Not a restriction.**
#: `set_models` takes any non-empty string, because a hardcoded list of model ids
#: in a UI ages the moment a new one ships -- and the one thing worse than not
#: offering the newest model is refusing it. These are the shortcut; the text
#: field beside them is the mechanism.
#:
#: The Claude 5 family is Fable 5 / Opus 5 / Sonnet 5. There is no Haiku 5, and
#: 4.5 is the latest Haiku -- the same note `config/grad.toml` carries, because
#: it is the thing people get wrong.
KNOWN_MODELS: tuple[str, ...] = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-haiku-4-5",
)

#: The keys an SSH host entry may carry, matching `config.Host`'s fields. A host
#: added here is merged into the `[hosts.*]` inventory, which is fixed by design
#: (`core/config.py:host`) -- this gives that inventory a second, writable
#: source; it does not make a connection ad-hoc.
HOST_FIELDS: tuple[str, ...] = (
    "hostname",
    "user",
    "rate_usd_per_hour",
    "workdir",
    "key_credential",
    "gpus",
    "notes",
)


def path(root: Path | None = None) -> Path:
    return appdata.workspace_state_dir(root) / "settings.json"


def stamp(root: Path | None = None) -> int:
    """A cheap marker that changes when this file does.

    `core/config.py` folds this into its cache key, so a wizard writing here is
    picked up by a process that has already loaded a `Config` -- without which
    the app would go on serving the models it read at startup and the setup
    window would appear to do nothing.

    Never raises: an unreadable app directory is not a reason to refuse to load
    a config, and the fallback -- treat it as absent -- is the same answer as a
    machine with nothing configured.
    """
    try:
        return path(root).stat().st_mtime_ns
    except OSError:
        return 0


def load(root: Path | None = None) -> dict[str, Any]:
    """The overlay, or an empty one. Never raises and never partially applies.

    `jsonl.read_json` returns None for a file that will not parse, and that is
    the right behaviour here rather than an error: a corrupt overlay should
    leave the app running on `grad.toml`, which is a working configuration, not
    stop it from starting.
    """
    raw = jsonl.read_json(path(root))
    return raw if isinstance(raw, dict) else {}


def _write(document: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    document["version"] = VERSION
    document["set_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    jsonl.write_json(path(root), document)
    return document


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def models(root: Path | None = None) -> dict[str, str]:
    """Only the roles that were actually chosen. A role absent here falls
    through to `grad.toml` and then to the shipped default, which is what makes
    this an overlay rather than a replacement."""
    chosen = load(root).get("models")
    if not isinstance(chosen, dict):
        return {}
    return {str(k): str(v) for k, v in chosen.items() if v}


def _check_roles(roles: dict[str, str]) -> dict[str, str]:
    from core import config as config_mod  # noqa: PLC0415 - `config` reads this module

    unknown = [r for r in roles if r not in config_mod.MODEL_ROLES]
    if unknown:
        raise UsageError(
            f"unknown model role(s): {', '.join(sorted(unknown))}",
            fix=f"roles are: {', '.join(config_mod.MODEL_ROLES)}",
        )
    cleaned = {}
    for role, value in roles.items():
        text = str(value or "").strip()
        if not text:
            raise UsageError(
                f"model for role {role!r} is empty",
                fix=f"pass a model id, or `setup models --clear {role}` to fall back to the config",
            )
        cleaned[role] = text
    return cleaned


def set_models(roles: dict[str, str], root: Path | None = None) -> dict[str, Any]:
    document = load(root)
    current = dict(document.get("models") or {})
    current.update(_check_roles(roles))
    document["models"] = current
    return _write(document, root)


def clear_models(roles: list[str], root: Path | None = None) -> dict[str, Any]:
    """Drop an override, so the role falls back through the layers again.

    Retiring by making optional rather than by deleting: a role that was set
    here and is now cleared resolves exactly as it did before anyone opened the
    wizard.
    """
    from core import config as config_mod  # noqa: PLC0415

    unknown = [r for r in roles if r not in config_mod.MODEL_ROLES]
    if unknown:
        raise UsageError(
            f"unknown model role(s): {', '.join(sorted(unknown))}",
            fix=f"roles are: {', '.join(config_mod.MODEL_ROLES)}",
        )
    document = load(root)
    current = dict(document.get("models") or {})
    for role in roles:
        current.pop(role, None)
    document["models"] = current
    return _write(document, root)


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
def default_backend(root: Path | None = None) -> str | None:
    value = str((load(root).get("backend") or {}).get("default") or "").strip()
    return value or None


def set_backend(name: str, root: Path | None = None) -> dict[str, Any]:
    chosen = str(name or "").strip()
    if chosen not in BACKENDS:
        raise UsageError(
            f"unknown backend {name!r}",
            fix=f"backends are: {', '.join(BACKENDS)}",
        )
    document = load(root)
    document["backend"] = {"default": chosen}
    return _write(document, root)


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------
#: The palettes `ui/tokens.py` ships. Spelled out here rather than imported,
#: because `core` importing `ui` is backwards -- the same reason `BACKENDS` is a
#: tuple here rather than a reference to `tools/evolve.py`. `tests/
#: test_settings.py` asserts the two lists still agree.
THEMES: tuple[str, ...] = ("light", "dark")


def theme(root: Path | None = None) -> str:
    """Which palette the workspace draws in. `light` unless it was changed.

    Per workspace, like everything else in this overlay, and that is the useful
    scope rather than an accident of where it landed: the theme is read by the
    splash process before any window exists, and a splash that flashed cream
    before a dark workspace painted would be the one frame the whole setting is
    judged on.

    Never raises and never returns something `ui/tokens.py` cannot resolve: a
    value written by a newer version falls back to the default there too.
    """
    chosen = str(load(root).get("theme") or "").strip().lower()
    return chosen if chosen in THEMES else "light"


def set_theme(name: str, root: Path | None = None) -> dict[str, Any]:
    chosen = str(name or "").strip().lower()
    if chosen not in THEMES:
        raise UsageError(
            f"unknown theme {name!r}",
            fix=f"themes are: {', '.join(THEMES)}",
        )
    document = load(root)
    document["theme"] = chosen
    return _write(document, root)


# ---------------------------------------------------------------------------
# the agent's own knobs
# ---------------------------------------------------------------------------
#: `[agent]` keys the setup window may write, with the bounds each is checked
#: against. An allowlist rather than "whatever is passed", because this overlay
#: outranks `config/grad.toml` for every reader: a typo here would not be a
#: setting that fails to apply, it would be a setting that applies and is wrong.
#:
#: `compact_at_tokens` is the only one so far, and it is the one people ask for.
#: The *context window* is the model's and is not ours to change -- a live
#: session reports 967,000 of 1,000,000 -- but where Grad compacts inside it is,
#: and that is what decides how much conversation the agent carries. 0 disables
#: compaction and leaves the matter to the CLI underneath.
#:
#: The floor is not cosmetic. Compaction costs a turn and the session it seeds
#: starts on a cold prompt cache, so a threshold small enough to trip after
#: every turn would compact continuously and cost more than it saves -- see
#: `core/compaction.py`. 20k is comfortably above a single large tool result.
AGENT_SETTINGS: dict[str, tuple[float, float]] = {
    "compact_at_tokens": (20_000, 10_000_000),
}


def agent(root: Path | None = None) -> dict[str, Any]:
    """Only the `[agent]` keys actually chosen here. A key absent falls through
    to `grad.toml` and then to the shipped default."""
    chosen = load(root).get("agent")
    if not isinstance(chosen, dict):
        return {}
    return {str(k): v for k, v in chosen.items() if k in AGENT_SETTINGS}


def _check_agent(values: dict[str, Any]) -> dict[str, int]:
    cleaned: dict[str, int] = {}
    for key, value in values.items():
        if key not in AGENT_SETTINGS:
            raise UsageError(
                f"unknown agent setting {key!r}",
                fix=f"settings are: {', '.join(sorted(AGENT_SETTINGS))}",
            )
        low, high = AGENT_SETTINGS[key]
        # Before `float`, because `bool` is an `int` in Python and both survive
        # the conversion: `True` became 1 and was refused for being under the
        # floor -- a confusing message for a wrong *kind* -- while `False` became
        # 0, which is the documented spelling of "off", so passing a boolean by
        # mistake silently disabled compaction.
        if isinstance(value, bool):
            raise UsageError(
                f"{key} must be a number of tokens, not {value!r}",
                fix=f"a whole number between {int(low):,} and {int(high):,}, or 0 to disable",
            )
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise UsageError(
                f"{key} must be a number, not {value!r}",
                fix=f"a whole number of tokens between {int(low):,} and {int(high):,}, or 0 to disable",
            ) from None
        # NaN fails every comparison below, so it would pass a naive range check
        # and land in the file as a threshold nothing is ever above.
        if not math.isfinite(number):
            raise UsageError(
                f"{key} must be a finite number, not {value!r}",
                fix=f"a whole number of tokens between {int(low):,} and {int(high):,}, or 0 to disable",
            )
        # 0 is not "below the floor", it is the documented way to turn the
        # feature off -- the same spelling `compaction.threshold` already reads.
        if number and not low <= number <= high:
            raise UsageError(
                f"{key} of {int(number):,} is outside the usable range",
                fix=(
                    f"between {int(low):,} and {int(high):,} tokens, or 0 to leave compaction "
                    "to the CLI underneath. Below the floor Grad would compact after almost "
                    "every turn, and a compaction costs a turn plus a cold prompt cache"
                ),
            )
        cleaned[key] = int(number)
    return cleaned


def set_agent(values: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    document = load(root)
    current = dict(document.get("agent") or {})
    current.update(_check_agent(values))
    document["agent"] = current
    return _write(document, root)


def clear_agent(keys: list[str], root: Path | None = None) -> dict[str, Any]:
    """Drop an override so the key resolves exactly as it did before anyone
    opened the wizard."""
    unknown = [k for k in keys if k not in AGENT_SETTINGS]
    if unknown:
        raise UsageError(
            f"unknown agent setting(s): {', '.join(sorted(unknown))}",
            fix=f"settings are: {', '.join(sorted(AGENT_SETTINGS))}",
        )
    document = load(root)
    current = dict(document.get("agent") or {})
    for key in keys:
        current.pop(key, None)
    document["agent"] = current
    return _write(document, root)


# ---------------------------------------------------------------------------
# ssh hosts
# ---------------------------------------------------------------------------
def hosts(root: Path | None = None) -> dict[str, dict[str, Any]]:
    stored = load(root).get("hosts")
    if not isinstance(stored, dict):
        return {}
    return {str(k): dict(v) for k, v in stored.items() if isinstance(v, dict)}


def add_host(name: str, spec: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Add or replace one host in the writable half of the inventory.

    The name is constrained the way a project id is, and for the same reason:
    it is used to look a host up and it ends up in a command line. A host called
    `-oProxyCommand=...` is not a naming problem.
    """
    chosen = str(name or "").strip()
    if not chosen or any(c.isspace() for c in chosen) or chosen.startswith("-"):
        raise UsageError(
            f"{name!r} is not a usable host name: no spaces, and it cannot start with a dash",
            fix="python -m tools.setup host add --name gpu-box --hostname … --user … --json",
        )
    unknown = [k for k in spec if k not in HOST_FIELDS]
    if unknown:
        raise UsageError(
            f"unknown host field(s): {', '.join(sorted(unknown))}",
            fix=f"fields are: {', '.join(HOST_FIELDS)}",
        )
    if not str(spec.get("hostname") or "").strip():
        raise UsageError(
            f"host {chosen!r} needs a hostname",
            fix="--hostname is what ssh connects to; --name is what Grad calls it",
        )
    rate = spec.get("rate_usd_per_hour", 0.0)
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        raise UsageError(
            f"host {chosen!r} has a malformed rate_usd_per_hour: {rate!r}",
            fix="a number; use 0 for a host that is free to use",
        ) from None
    if not math.isfinite(rate):
        # `nan` fails every comparison a gate makes against it and `inf` is a
        # price no run can be under, so both are ceilings that stop bounding
        # anything. `core/config.py` refuses them on the TOML side and this is
        # the writable side of the same inventory -- a check that exists in only
        # one of two entry points is a check with a way around it.
        raise UsageError(
            f"host {chosen!r} has a non-finite rate_usd_per_hour ({rate})",
            fix="rate_usd_per_hour must be a finite number; use 0 for a free host",
        )
    if rate < 0:
        # `collect` prices wall clock against this, and a negative rate books
        # negative actuals -- which *reduce* rolling spend. A typo that raises
        # the ceiling is the one shape of error worth refusing here rather than
        # at the point of accounting.
        raise UsageError(
            f"host {chosen!r} has a negative rate_usd_per_hour ({rate})",
            fix="use 0 for a host that is free to use; negative spend is not a thing",
        )
    document = load(root)
    inventory = dict(document.get("hosts") or {})
    entry = {k: v for k, v in spec.items() if v is not None}
    entry["rate_usd_per_hour"] = rate
    inventory[chosen] = entry
    document["hosts"] = inventory
    return _write(document, root)


def remove_host(name: str, root: Path | None = None) -> dict[str, Any]:
    document = load(root)
    inventory = dict(document.get("hosts") or {})
    if name not in inventory:
        raise UsageError(
            f"no host {name!r} was added here",
            fix="python -m tools.setup show --json   # lists both halves of the inventory",
        )
    inventory.pop(name)
    document["hosts"] = inventory
    return _write(document, root)


# ---------------------------------------------------------------------------
# what this is overriding
# ---------------------------------------------------------------------------
def shadowing(cfg: Any, root: Path | None = None) -> list[dict[str, Any]]:
    """Every value here that is overriding one in `grad.toml`.

    This is what keeps the whole arrangement honest. Someone edits `[models]
    evolve`, sees no change, and has no way to discover that a file they have
    never heard of outranks the file they were told to edit. `kaggle account`
    reports the same thing for the same reason, and the report is the price of
    being allowed to win.

    Only genuine conflicts: an overlay value that matches the config, or one for
    a key the config never set, is not shadowing anything.
    """
    out: list[dict[str, Any]] = []
    configured_models = (getattr(cfg, "user", None) or {}).get("models") or {}
    for role, value in models(root).items():
        configured = configured_models.get(role)
        if configured and str(configured) != value:
            out.append(
                {
                    "what": f"[models] {role}",
                    "config": str(configured),
                    "overlay": value,
                }
            )
    configured_agent = (getattr(cfg, "user", None) or {}).get("agent") or {}
    for key, value in agent(root).items():
        configured = configured_agent.get(key)
        if configured is not None and str(configured) != str(value):
            out.append({"what": f"[agent] {key}", "config": str(configured), "overlay": str(value)})
    configured_hosts = (getattr(cfg, "raw", None) or {}).get("hosts") or {}
    for name in hosts(root):
        if name in configured_hosts:
            out.append({"what": f"[hosts.{name}]", "config": "defined", "overlay": "replaced"})
    return out
