"""The two ways the retrieval funnel came back empty, and how each says so.

Both failures below were silent in the same expensive way: the CLI answered
`ok: true` with no results, or died with a message that named nothing. From the
agent's side either one reads as "there is no literature on this", which is the
one conclusion a research tool must never invite by accident.

  * **Stage 0 could not authenticate.** The Haiku stages are Agent SDK clients,
    and the agent reaches them the way it reaches every capability here: by
    running the CLI over Bash. That hop strips `CLAUDE_CODE_OAUTH_TOKEN` and
    nothing else, so a token in the environment works from a terminal and is
    gone under the agent. The CLI then answers "Not logged in" and exits
    non-zero, which the SDK reports as `Claude Code returned an error result:
    success` -- the CLI sends no `errors` array, so the SDK falls back to
    printing the result *subtype*.
  * **Stage 1 was rate limited on every call.** Semantic Scholar's anonymous
    pool is shared and near-permanently exhausted.

No SDK and no network: the SDK is faked, because what is under test is which
error comes out, not what Haiku says.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core import credentials, haiku, paths
from core.errors import ConfigError, UpstreamError

TOKEN = "sk-ant-oat-not-a-real-token"


# ---------------------------------------------------------------------------
# where stage 0's credentials come from
# ---------------------------------------------------------------------------
def test_the_environment_is_used_when_it_has_a_token(monkeypatch):
    """Running a stage by hand in a terminal has to keep working with no setup:
    there, the token *is* in the environment."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)
    monkeypatch.setattr(
        credentials, "get", lambda *a, **k: pytest.fail("the store was read before the environment")
    )
    assert haiku._credentials_env() == {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}


def test_the_credential_store_covers_the_hop_that_strips_the_environment(monkeypatch):
    """The case that was broken: under the agent there is no token to inherit."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        credentials, "get", lambda name, **k: TOKEN if name == credentials.CLAUDE_TOKEN else None
    )
    assert haiku._credentials_env() == {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}


def test_no_credentials_anywhere_names_both_halves_of_the_fix(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(credentials, "get", lambda *a, **k: None)
    with pytest.raises(ConfigError) as caught:
        haiku._credentials_env()
    assert "claude setup-token" in caught.value.fix
    assert "credential set claude_oauth_token" in caught.value.fix


def test_the_token_is_handed_to_the_subprocess_not_left_to_inheritance(monkeypatch):
    """`ClaudeAgentOptions.env` merges over the inherited environment, so this
    adds one variable and takes nothing away."""
    sdk = _FakeSdk([_expansion_message()])
    monkeypatch.setattr(haiku, "_sdk", lambda: sdk)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(credentials, "get", lambda *a, **k: TOKEN)

    haiku.expand("a question", model="claude-haiku-4-5", log_name="probe")
    assert sdk.options.env == {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}


# ---------------------------------------------------------------------------
# what an unauthenticated stage 0 says
# ---------------------------------------------------------------------------
def test_a_not_logged_in_turn_becomes_the_error_that_names_the_fix(monkeypatch):
    """Rather than `Claude Code returned an error result: success`, which is the
    result *subtype* and describes nothing."""
    sdk = _FakeSdk(
        [_message(text="Not logged in", error="authentication_failed")],
        raises=Exception("Claude Code returned an error result: success"),
    )
    monkeypatch.setattr(haiku, "_sdk", lambda: sdk)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

    with pytest.raises(ConfigError) as caught:
        haiku.expand("a question", model="claude-haiku-4-5", log_name="probe")
    assert "credential set claude_oauth_token" in caught.value.fix
    assert "success" not in caught.value.message


def test_an_authentication_failure_that_does_not_raise_is_still_a_failure(monkeypatch):
    """The CLI need not exit non-zero for the turn to have been refused; an
    empty transcript would otherwise be reported as "no structured output"."""
    sdk = _FakeSdk([_message(text="Not logged in", error="authentication_failed")])
    monkeypatch.setattr(haiku, "_sdk", lambda: sdk)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

    with pytest.raises(ConfigError):
        haiku.expand("a question", model="claude-haiku-4-5", log_name="probe")


def test_any_other_sdk_failure_is_left_alone(monkeypatch):
    """Only the authentication case is translated. Swallowing the rest would
    hide real upstream faults behind a credentials message."""
    sdk = _FakeSdk([], raises=RuntimeError("connection reset"))
    monkeypatch.setattr(haiku, "_sdk", lambda: sdk)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

    with pytest.raises(RuntimeError, match="connection reset"):
        haiku.expand("a question", model="claude-haiku-4-5", log_name="probe")


def test_a_healthy_stage_still_returns_its_payload(monkeypatch):
    sdk = _FakeSdk([_expansion_message()])
    monkeypatch.setattr(haiku, "_sdk", lambda: sdk)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

    out = haiku.expand("a question", model="claude-haiku-4-5", log_name="probe")
    assert out["queries"] == ["adaptive optimizers", "second order methods"]
    # The per-query log is the mitigation for these stages being subagents.
    assert (paths.notes_dir() / "funnel" / "probe.md").exists()


# ---------------------------------------------------------------------------
# what a rate-limited stage 1 says
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source, client, expected_fix",
    [
        # Each door's dead end points at the one that is not a dead end.
        ("s2", "SemanticScholar", "--tier1 asta"),
        ("asta", "Asta", "credential set asta_api_key"),
    ],
)
def test_a_search_whose_every_call_failed_is_a_failure_not_an_empty_result(
    workspace, capsys, monkeypatch, source, client, expected_fix
):
    """`ok: true` with no results reads as "the literature has nothing", which
    is not a conclusion anyone should draw from a rate limit."""
    from core import http
    from tools import paper_search

    monkeypatch.setattr(http, client, lambda cfg: _RateLimited())

    code = paper_search.cli.run(
        ["search", "efficient optimizers", "--tier1", source,
         "--no-expand", "--no-triage", "--no-local", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code != 0
    assert payload["ok"] is False
    assert "rate-limited" in payload["error"]["message"]
    assert expected_fix in payload["error"]["fix"]
    # The old advice, which sent anyone following it to a second empty run.
    assert "--no-expand" not in json.dumps(payload)


def test_the_s2_dead_end_does_not_recommend_a_key_that_cannot_be_obtained(
    workspace, capsys, monkeypatch
):
    """Ai2 stopped issuing Semantic Scholar keys to free-domain addresses, so
    "store an S2 API key -- it is free" has no ending for a personal account.
    Advice that cannot be followed is worse than none: it is followed first."""
    from core import http
    from tools import paper_search

    monkeypatch.setattr(http, "SemanticScholar", lambda cfg: _RateLimited())
    paper_search.cli.run(
        ["search", "efficient optimizers", "--tier1", "s2",
         "--no-expand", "--no-triage", "--no-local", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert "credential set s2_api_key" not in payload["error"]["fix"]
    assert "institutional" in payload["error"]["fix"]


def test_an_empty_run_still_writes_the_trace_the_funnel_view_reads(
    workspace, capsys, monkeypatch
):
    """"why is the obviously relevant paper not in here" is the question the
    funnel view exists for, and it was exactly the runs that answered it that
    returned before writing a trace."""
    from core import http
    from tools import paper_search

    # The default source, so this covers the path a real run takes.
    monkeypatch.setattr(http, "Asta", lambda cfg: _RateLimited())
    paper_search.cli.run(
        ["search", "efficient optimizers", "--no-expand", "--no-triage", "--no-local", "--json"]
    )
    capsys.readouterr()

    traces = list((paths.notes_dir() / "funnel").glob("*.json"))
    assert len(traces) == 1
    written = json.loads(traces[0].read_text(encoding="utf-8"))
    assert written["stages"]["1_retrieve"] == {"rankings": 0, "candidates": 0}
    # Which client was asked, so "no results" can be read against who was down.
    assert written["stages"]["1_sources"] == ["asta"]
    assert any("rate-limited" in w for w in written["warnings"])


def test_a_genuinely_empty_index_is_an_empty_result_not_a_failure(workspace, capsys):
    """The other side of it: nothing found and nothing broken is `ok: true`."""
    from tools import paper_search

    assert paper_search.cli.run(
        ["search", "efficient optimizers", "--local-only", "--no-expand", "--no-triage", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["data"]["results"] == []
    assert "widen --candidates" in payload["data"]["note"]


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _RateLimited:
    """Either tier-1 client, rate limited. They share a vocabulary, so one fake
    stands in for both -- which is the property `tests/test_asta.py` pins."""

    def _fail(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
        raise UpstreamError(
            "Semantic Scholar rate-limited the request",
            fix="wait a few seconds and retry, or store an S2 API key",
        )

    snippet_search = _fail
    paper_search = _fail
    neighbours = _fail


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str = "", error: str | None = None) -> None:
        self.content = [_Block(text)] if text else []
        self.error = error
        self.usage = None


def _message(*, text: str = "", error: str | None = None) -> _Message:
    return _Message(text, error)


def _expansion_message() -> _Message:
    """A turn that calls the tool, which is what the real one does."""
    message = _Message("calling submit_expansion")
    message.call = {
        "queries": ["adaptive optimizers", "second order methods"],
        "hyde": " ".join(["word"] * 40),
    }
    return message


class _FakeSdk:
    """Just enough of `claude_agent_sdk` for `haiku._call`.

    A message carrying `.call` invokes the registered tool with that payload,
    which is how the real SDK delivers validated tool input.
    """

    def __init__(self, messages: list[_Message], raises: Exception | None = None) -> None:
        self.messages = messages
        self.raises = raises
        self.options: Any = None
        self._handler: Any = None

    def tool(self, _name: str, _description: str, _schema: dict[str, Any]) -> Any:
        def decorate(fn: Any) -> Any:
            self._handler = fn
            return fn

        return decorate

    def create_sdk_mcp_server(self, name: str, tools: list[Any]) -> dict[str, Any]:
        return {"name": name, "tools": tools}

    def ClaudeAgentOptions(self, **kwargs: Any) -> Any:  # noqa: N802 - the SDK's name
        self.options = type("Options", (), kwargs)
        return self.options

    def query(self, *, prompt: str, options: Any) -> Any:  # noqa: ARG002
        async def stream() -> Any:
            for message in self.messages:
                payload = getattr(message, "call", None)
                if payload is not None and self._handler is not None:
                    await self._handler(payload)
                yield message
            if self.raises is not None:
                raise self.raises

        return stream()


def test_the_fix_the_error_prints_is_a_command_that_exists():
    """The error above tells the reader to run `credential set
    claude_oauth_token`. That is only useful if the CLI accepts the name."""
    from tools.jobs import CREDENTIAL_NAMES

    assert credentials.CLAUDE_TOKEN in CREDENTIAL_NAMES
    assert credentials.CLAUDE_TOKEN in credentials.status()
    assert credentials.CLAUDE_TOKEN in haiku.AUTH_FIX
