"""How hard the main agent thinks, and where that choice is kept.

`ClaudeAgentOptions.effort` takes `low | medium | high | xhigh | max`, and it is
a knob worth having on screen rather than in a file: the right setting is a
property of the *task*, and the tasks in one session range from "what is in the
ledger" to designing a campaign. A config value is the wrong shape for something
that should change three times an afternoon.

**It cannot be changed on a live client, and that is the fact the design is
built around.** The SDK's control protocol has exactly two subtypes that change
an option mid-session -- `set_model` and `set_permission_mode` -- and there is
nothing for effort. So a change means building a new client, which is the same
operation `Session.compact` performs and costs the same: the SDK subprocess is
restarted, and the conversation survives only because `resume` carries the
session id across.

Hence the selection is *recorded here* and applied lazily, at the start of the
next turn. Flipping the dial four times while idle costs nothing at all; the
rebuild happens once, at the moment it would have had to happen anyway. The
alternative -- rebuilding on every click -- would make an idle fiddle with a
control cost several seconds of subprocess startup each time.

The choice lives under the app directory rather than in `config/grad.toml`, for
the reason `tools/kaggle.py` keeps its account there: that file is hand-annotated
and `tomllib` reads it but cannot write it, so a command that edited it would
reformat it and drop every comment in it. `[agent] effort` is still read as the
default, so a workspace can ship a starting point.
"""

from __future__ import annotations

from typing import Any

from core import appdata, jsonl

#: The levels the SDK accepts, cheapest first. `AUTO` is ours: it means "pass
#: nothing and let the CLI decide", which is what every session did before this
#: existed and is therefore the only safe default. Keeping it in the cycle is
#: what makes the previous behaviour reachable rather than merely historical.
AUTO = "auto"
LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
CYCLE: tuple[str, ...] = (AUTO, *LEVELS)


def state_path():
    return appdata.state_dir() / "agent.json"


def _state() -> dict[str, Any]:
    record = jsonl.read_json(state_path())
    return record if isinstance(record, dict) else {}


def current(cfg: Any = None) -> str:
    """The selected level: the app-state choice, then `[agent] effort`, then auto.

    Never raises and never returns something the SDK would reject. An unreadable
    state file, a hand-edited one containing nonsense, or a config typo all
    resolve to `auto` -- which is the behaviour of every release before this
    knob existed, and the only wrong answer that cannot break a session.
    """
    chosen = _state().get("effort")
    if isinstance(chosen, str) and chosen.lower() in CYCLE:
        return chosen.lower()
    if cfg is None:
        try:
            from core import config as config_mod  # noqa: PLC0415

            cfg = config_mod.load()
        except Exception:  # noqa: BLE001 - see the docstring
            return AUTO
    try:
        configured = str(cfg.get("agent", "effort", AUTO)).lower()
    except Exception:  # noqa: BLE001
        return AUTO
    return configured if configured in CYCLE else AUTO


def set_current(level: str) -> str:
    """Record a level. Returns what was actually stored.

    Validated rather than trusted: this is written by a click handler today and
    by whatever calls it tomorrow, and an unknown string reaching
    `ClaudeAgentOptions` is a session that fails to start rather than a session
    that thinks slightly wrong.
    """
    value = (level or "").strip().lower()
    if value not in CYCLE:
        raise ValueError(f"unknown effort {level!r}; one of: {', '.join(CYCLE)}")
    appdata.ensure()
    jsonl.write_json(state_path(), {**_state(), "effort": value})
    return value


def cycle(level: str | None = None) -> str:
    """The next level round the ring, wrapping at the end.

    A cycle rather than a menu because the strip it lives on has room for one
    word, and because the levels are *ordered* -- the question a user asks of
    this control is "more" or "less", not "which of six".
    """
    now = level if level in CYCLE else current()
    return CYCLE[(CYCLE.index(now) + 1) % len(CYCLE)]


def option(cfg: Any = None, sdk: Any = None) -> dict[str, Any]:
    """The `ClaudeAgentOptions` fragment for the current level, or `{}`.

    Feature-detected against the installed SDK for the reason `agent.probe`
    exists and `agent.thinking_option` does the same: this field is newer than
    the permission mode, the options object has changed shape between releases,
    and an SDK without it should give a session with default effort rather than
    a `TypeError` before the first turn.
    """
    level = current(cfg)
    if level == AUTO:
        return {}
    if sdk is None:
        try:
            import claude_agent_sdk as sdk  # noqa: PLC0415
        except ImportError:
            return {}
    try:
        import dataclasses  # noqa: PLC0415

        fields = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    except (TypeError, AttributeError):
        # Not a dataclass, or no options class at all. Same degradation as
        # `thinking_option`: the settings this SDK does not understand are
        # dropped rather than passed and rejected.
        return {}
    return {"effort": level} if "effort" in fields else {}


def label(level: str | None = None) -> str:
    """What the statusline prints. ASCII, because it can reach a console."""
    return f"effort {level or current()}"
