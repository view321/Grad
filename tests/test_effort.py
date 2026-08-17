"""The agent's reasoning effort (`core/effort.py`) and how a change reaches it.

The interesting property is not the ring of levels, it is *when* a change
applies. The SDK fixes effort when a client is built and offers no control
request for it, so a change means a rebuild -- and a rebuild that forgets to
resume is a change of setting that silently costs the conversation.
"""

from __future__ import annotations

import dataclasses
import pytest

from core import effort


class _Options:
    """Stands in for `ClaudeAgentOptions`. A dataclass, because that is what
    `option()` introspects."""


@dataclasses.dataclass
class _WithEffort:
    effort: str | None = None


@dataclasses.dataclass
class _WithoutEffort:
    model: str | None = None


class _Sdk:
    def __init__(self, options_class):
        self.ClaudeAgentOptions = options_class


def test_the_default_is_auto_and_passes_nothing(workspace):
    """Auto is the behaviour of every release before this knob existed, which is
    the only safe thing for it to default to."""
    assert effort.current() == effort.AUTO
    assert effort.option(sdk=_Sdk(_WithEffort)) == {}


def test_a_level_round_trips_through_app_state(workspace):
    assert effort.set_current("xhigh") == "xhigh"
    assert effort.current() == "xhigh"
    assert effort.option(sdk=_Sdk(_WithEffort)) == {"effort": "xhigh"}


def test_the_cycle_wraps_and_reaches_auto_again(workspace):
    seen, level = [], effort.AUTO
    for _ in range(len(effort.CYCLE)):
        level = effort.cycle(level)
        seen.append(level)
    assert seen == [*effort.LEVELS, effort.AUTO]


def test_an_unknown_level_is_refused_rather_than_stored(workspace):
    """An unknown string reaching `ClaudeAgentOptions` is a session that fails to
    start, which is a worse outcome than a rejected click."""
    with pytest.raises(ValueError):
        effort.set_current("turbo")
    assert effort.current() == effort.AUTO


def test_junk_in_the_state_file_reads_as_auto(workspace):
    effort.set_current("high")
    effort.state_path().write_text('{"effort": "banana"}', encoding="utf-8")
    assert effort.current() == effort.AUTO
    effort.state_path().write_text("not json at all", encoding="utf-8")
    assert effort.current() == effort.AUTO


def test_config_supplies_the_starting_point(workspace, monkeypatch):
    class _Cfg:
        def get(self, section, key, default=None):
            return "medium" if (section, key) == ("agent", "effort") else default

    assert effort.current(_Cfg()) == "medium"
    # An explicit selection beats the config default, which is the whole reason
    # the selection is not in the config file.
    effort.set_current("low")
    assert effort.current(_Cfg()) == "low"


def test_an_sdk_without_the_field_gets_nothing(workspace):
    """Same degradation as `agent.thinking_option`: this field is newer than the
    permission mode and the options object has changed shape between releases."""
    effort.set_current("max")
    assert effort.option(sdk=_Sdk(_WithoutEffort)) == {}
    assert effort.option(sdk=_Sdk(_Options)) == {}


# ---------------------------------------------------------------------------
# when it applies
# ---------------------------------------------------------------------------
#: Stands in for a live SDK client. Module-level rather than `object()` in the
#: signature below, where it would be evaluated once at class-definition time
#: anyway -- so the default was already shared, and only looked as though a
#: fresh one arrived per instance.
_A_CLIENT = object()


class _FakeSession:
    """The two methods `apply_effort` drives, and a record of the calls."""

    def __init__(self, *, client=_A_CLIENT, sdk_session_id="sdk-1", client_effort="auto"):
        self.client = client
        self.sdk_session_id = sdk_session_id
        self.client_effort = client_effort
        self.calls: list[str] = []

    async def close(self):
        self.calls.append("close")
        self.client = None
        self.client_effort = None

    async def start(self):
        self.calls.append("start")
        self.client = object()
        self.client_effort = effort.current()


@pytest.mark.asyncio
async def test_a_change_rebuilds_the_client(workspace):
    from ui.app import Session

    session = _FakeSession(client_effort="auto")
    effort.set_current("high")
    assert await Session.apply_effort(session) is True
    assert session.calls == ["close", "start"]
    assert session.client_effort == "high"


@pytest.mark.asyncio
async def test_selecting_the_level_already_running_rebuilds_nothing(workspace):
    from ui.app import Session

    effort.set_current("high")
    session = _FakeSession(client_effort="high")
    assert await Session.apply_effort(session) is False
    assert session.calls == []


@pytest.mark.asyncio
async def test_nothing_is_rebuilt_when_no_client_is_connected(workspace):
    """The next `start` picks the level up for free."""
    from ui.app import Session

    effort.set_current("max")
    session = _FakeSession(client=None, client_effort=None)
    assert await Session.apply_effort(session) is False
    assert session.calls == []


@pytest.mark.asyncio
async def test_a_change_is_deferred_when_there_is_no_session_to_resume(workspace):
    """A rebuild with no resume id starts a *new* conversation.

    Changing a setting must never be the thing that discards the transcript, so
    the level waits for a rebuild that can carry the conversation across.
    """
    from ui.app import Session

    effort.set_current("low")
    session = _FakeSession(sdk_session_id=None, client_effort="auto")
    assert await Session.apply_effort(session) is False
    assert session.calls == []
