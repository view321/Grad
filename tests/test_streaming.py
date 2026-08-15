"""Assembling a turn's text from partial and finished messages (`agent.TextStream`).

`include_partial_messages` makes the SDK emit the same text twice over: once as
a run of `text_delta` events, and again as the finished `AssistantMessage`. The
whole job of `TextStream` is to show it once, and the failure it exists to
prevent -- every answer appearing twice -- is invisible until someone actually
runs the agent. So it is tested here, against fakes shaped like the SDK's own
messages, with no SDK and no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import agent


# --- fakes shaped like the SDK's messages -----------------------------------
@dataclass
class FakeBlock:
    text: str


@dataclass
class ThinkingBlock:
    """No `.text`, exactly like the real one -- which is why thinking never
    reaches the transcript."""

    thinking: str


@dataclass
class FakeMessage:
    """An `AssistantMessage`: a finished list of content blocks."""

    content: list[Any] = field(default_factory=list)


@dataclass
class FakeStreamEvent:
    """A `StreamEvent`: the raw Anthropic streaming event, and no `.content`."""

    event: dict[str, Any]


def delta(text: str) -> FakeStreamEvent:
    return FakeStreamEvent({"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}})


def message(*texts: str) -> FakeMessage:
    return FakeMessage([FakeBlock(t) for t in texts])


def drain(stream: agent.TextStream, messages: list[Any]) -> str:
    """Feed a whole turn, returning what a CLI would have printed."""
    return "".join(stream.feed(m) for m in messages)


# ---------------------------------------------------------------------------
# the rule the class exists for
# ---------------------------------------------------------------------------
def test_a_streamed_answer_is_not_also_appended_when_it_finishes():
    """The bug this is all for: deltas *and* the finished message both carry the
    text, so the obvious loop shows every answer twice."""
    stream = agent.TextStream()
    printed = drain(stream, [delta("Hello"), delta(" world"), message("Hello world")])
    assert stream.text == "Hello world"
    assert printed == "Hello world"  # printed once, as it arrived


def test_the_tokens_are_visible_before_the_message_finishes():
    """Streaming is the point: the text has to be readable mid-turn, not only
    once the message lands."""
    stream = agent.TextStream()
    stream.feed(delta("The loss "))
    assert stream.text == "The loss "
    stream.feed(delta("is 3.1"))
    assert stream.text == "The loss is 3.1"


def test_a_message_that_never_streamed_still_arrives():
    """Not every message comes with deltas -- a resumed turn, a cached reply, an
    SDK that stopped emitting them. Losing the text would be far worse than
    showing it late."""
    stream = agent.TextStream()
    printed = drain(stream, [message("no deltas for this one")])
    assert stream.text == "no deltas for this one"
    assert printed == "no deltas for this one"


def test_a_partially_streamed_message_is_completed_not_duplicated():
    stream = agent.TextStream()
    printed = drain(stream, [delta("half "), message("half the answer")])
    assert stream.text == "half the answer"
    assert printed == "half the answer"  # "half " streamed, "the answer" caught up


def test_the_finished_message_wins_when_the_two_disagree():
    """A dropped or reordered event must not leave a mangled transcript. The
    reconstruction is discarded; the message is authoritative."""
    stream = agent.TextStream()
    stream.feed(delta("garbled"))
    stream.feed(message("the real answer"))
    assert stream.text == "the real answer"


# ---------------------------------------------------------------------------
# a whole turn
# ---------------------------------------------------------------------------
def test_several_messages_in_one_turn_each_stream_and_settle():
    """A turn is prose, then a tool call, then more prose. Each finished message
    resets the run of deltas -- reset once at the end and the second message
    would overwrite the first."""
    stream = agent.TextStream()
    printed = drain(stream, [
        delta("Checking"), delta(" the ledger."), message("Checking the ledger."),
        FakeMessage([]),                       # the tool call itself: no text
        delta(" Found"), delta(" it."), message(" Found it."),
    ])
    assert stream.text == "Checking the ledger. Found it."
    assert printed == "Checking the ledger. Found it."


def test_a_message_carrying_no_text_does_not_disturb_a_stream_in_flight():
    """Tool results and system messages arrive mid-turn. Treating one as a
    finished assistant message would drop the deltas it interrupted."""
    stream = agent.TextStream()
    stream.feed(delta("half an answer"))
    stream.feed(FakeMessage([]))               # a tool result: content, but no text
    stream.feed(FakeMessage())                 # empty content
    assert stream.text == "half an answer"
    stream.feed(message("half an answer, finished"))
    assert stream.text == "half an answer, finished"


# ---------------------------------------------------------------------------
# what counts as visible text
# ---------------------------------------------------------------------------
def test_thinking_deltas_never_reach_the_transcript():
    """`_text_of` skips a finished `ThinkingBlock` because it has no `.text`. If
    the deltas did not skip thinking too, the stream would show reasoning that
    vanished the moment the message settled."""
    stream = agent.TextStream()
    stream.feed(FakeStreamEvent(
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}}
    ))
    assert stream.text == ""
    stream.feed(FakeMessage([ThinkingBlock("hmm"), FakeBlock("the answer")]))
    assert stream.text == "the answer"


def test_tool_input_deltas_are_not_answer_text():
    stream = agent.TextStream()
    stream.feed(FakeStreamEvent(
        {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}
    ))
    assert stream.text == ""


def test_non_delta_stream_events_are_ignored():
    stream = agent.TextStream()
    for event in (
        {"type": "message_start", "message": {}},
        {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ):
        assert stream.feed(FakeStreamEvent(event)) == ""
    assert stream.text == ""


def test_a_malformed_stream_event_is_not_a_crash():
    """`event` is a raw dict off the wire, so every field is untrusted. A turn
    must not die because one event was shaped unexpectedly."""
    stream = agent.TextStream()
    for event in (None, "nope", {}, {"type": "content_block_delta"},
                  {"type": "content_block_delta", "delta": None},
                  {"type": "content_block_delta", "delta": {"type": "text_delta"}},
                  {"type": "content_block_delta", "delta": {"type": "text_delta", "text": 7}}):
        assert stream.feed(FakeStreamEvent(event)) == ""
    assert stream.text == ""


def test_the_options_actually_ask_for_partial_messages(monkeypatch):
    """The whole feature is one flag on the options, and it defaults to off in
    the SDK. Every other test here would still pass with it dropped -- they feed
    deltas by hand -- so this is the one that notices."""
    sdk = pytest.importorskip("claude_agent_sdk", reason="the SDK is not installed")
    from core import config as config_mod

    # The prompt file lives in the real workspace, not the fixture's temp one;
    # this test is about the flag, not about reading it.
    monkeypatch.setattr(agent, "system_prompt", lambda: "prompt")
    options = agent.build_options(config_mod.load())
    assert isinstance(options, sdk.ClaudeAgentOptions)
    assert options.include_partial_messages is True
