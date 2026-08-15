"""Assembling a turn from partial and finished messages (`agent.TextStream`,
`agent.TurnStream`).

`include_partial_messages` makes the SDK emit the same text twice over: once as
a run of `text_delta` events, and again as the finished `AssistantMessage`. The
whole job of `TextStream` is to show it once, and the failure it exists to
prevent -- every answer appearing twice -- is invisible until someone actually
runs the agent. So it is tested here, against fakes shaped like the SDK's own
messages, with no SDK and no network.

`TurnStream` is the other half: the tool calls, which `TextStream` filters out
by construction. Its own rule is that **the order is the information** -- which
command ran before which claim -- so most of what is tested below is where a
block lands, not just that it exists.
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


# ---------------------------------------------------------------------------
# the tool calls (`TurnStream`)
# ---------------------------------------------------------------------------
@dataclass
class ToolUse:
    """A `ToolUseBlock`."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """A `ToolResultBlock`. These arrive on a `UserMessage`, not the assistant's."""

    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


def kinds(stream: agent.TurnStream) -> list[str]:
    return [b["kind"] for b in stream.blocks]


def test_a_call_becomes_a_block_of_its_own():
    """The gap this closes: every capability in this project is reached by a
    Bash into `tools/`, and none of it used to reach the transcript."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([
        FakeBlock("Checking the ledger."),
        ToolUse("tu_1", "Bash", {"command": "python -m tools.ledger show"}),
    ]))
    assert kinds(stream) == ["text", "tool"]
    call = stream.blocks[1]
    assert call["name"] == "Bash"
    assert call["title"] == "python -m tools.ledger show"
    assert call["status"] == "running"


def test_prose_after_a_call_lands_below_it():
    """The order is the whole point: a claim made *after* a command ran reads
    differently from one made before it."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([
        FakeBlock("Checking."),
        ToolUse("tu_1", "Bash", {"command": "ls"}),
    ]))
    stream.feed(FakeMessage([ToolResult("tu_1", "one\ntwo")]))
    stream.feed(delta(" Two entries."))
    stream.feed(FakeMessage([FakeBlock(" Two entries.")]))
    assert kinds(stream) == ["text", "tool", "text"]
    assert [b["text"] for b in stream.blocks if b["kind"] == "text"] == ["Checking.", " Two entries."]


def test_a_result_attaches_to_the_call_it_answers():
    """Two calls in flight and the results back in the other order is the case
    that matching by position would get exactly backwards."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([
        ToolUse("tu_1", "Read", {"file_path": "core/budget.py"}),
        ToolUse("tu_2", "Grep", {"pattern": "def status"}),
    ]))
    stream.feed(FakeMessage([ToolResult("tu_2", "core/budget.py:41"), ToolResult("tu_1", "…source…")]))
    first, second = stream.blocks
    assert (first["title"], first["result"]) == ("core/budget.py", "…source…")
    assert (second["title"], second["result"]) == ("def status", "core/budget.py:41")
    assert [b["status"] for b in stream.blocks] == ["ok", "ok"]


def test_a_refused_call_is_drawn_as_failed_not_as_output():
    """A `PreToolUse` denial comes back as an error result. Showing it as
    ordinary output would make a blocked spend look like a successful one."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "ssh probe-host echo hello"})]))
    stream.feed(FakeMessage([ToolResult("tu_1", "denied by the gate", is_error=True)]))
    assert stream.blocks[0]["status"] == "error"
    assert stream.blocks[0]["result"] == "denied by the gate"


def test_a_result_for_a_call_this_stream_never_saw_is_dropped():
    """A turn resumed from cache can carry a result whose call was never
    streamed. Inventing a card for it would claim an order we do not know."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolResult("tu_missing", "output")]))
    assert stream.blocks == []


def test_result_content_that_arrives_as_blocks_is_flattened():
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "ls"})]))
    stream.feed(FakeMessage([ToolResult("tu_1", [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])]))
    assert stream.blocks[0]["result"] == "one\ntwo"


def test_a_long_result_is_clipped_and_says_so():
    """A `Read` of a long file is held for the session, written to the
    transcript file and drawn again on restore. The card is a record that the
    call happened, not a second copy of its output."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Read", {"file_path": "notes/long.md"})]))
    stream.feed(FakeMessage([ToolResult("tu_1", "\n".join(f"line {i}" for i in range(500)))]))
    result = stream.blocks[0]["result"]
    assert len(result.splitlines()) <= agent.RESULT_LINES + 2
    assert "more lines" in result


def test_the_turns_text_is_the_prose_alone():
    """`text` is what a transcript with no cards should say. Tool output is not
    something the agent said, and folding it in would put a command's stdout in
    the agent's own voice."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([FakeBlock("Running it."), ToolUse("tu_1", "Bash", {"command": "ls"})]))
    stream.feed(FakeMessage([ToolResult("tu_1", "budget.py")]))
    assert stream.text == "Running it."


def test_a_card_is_titled_by_what_the_call_was_on():
    """Not by the whole input: an `Edit` carries its entire replacement text,
    and a head that is a wall of source is worse than one that is empty."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Edit", {
        "file_path": "core/budget.py",
        "old_string": "x" * 5000,
        "new_string": "y" * 5000,
    })]))
    call = stream.blocks[0]
    assert call["title"] == "core/budget.py"
    assert dict(call["rows"]).keys() == {"old_string", "new_string"}
    assert all(len(value) < 400 for value in dict(call["rows"]).values())


def test_a_multiline_command_is_one_line_in_the_head_and_whole_in_the_body():
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "cd tools \\\n&& python -m evolve"})]))
    call = stream.blocks[0]
    assert "\n" not in call["title"]
    assert "python -m evolve" in call["text"]


def test_the_call_in_flight_is_the_one_still_running():
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "first"})]))
    stream.feed(FakeMessage([ToolResult("tu_1", "done")]))
    stream.feed(FakeMessage([ToolUse("tu_2", "Bash", {"command": "second"})]))
    assert stream.active()["title"] == "second"
    stream.feed(FakeMessage([ToolResult("tu_2", "done")]))
    assert stream.active() is None


def test_a_cli_gets_a_line_naming_each_call_and_its_outcome():
    """`feed` returns what a terminal should print next -- the same stream the
    UI draws as cards, in one line each."""
    stream = agent.TurnStream()
    printed = drain(stream, [
        delta("Checking."), FakeMessage([FakeBlock("Checking."), ToolUse("tu_1", "Bash", {"command": "ls"})]),
        FakeMessage([ToolResult("tu_1", "one\ntwo")]),
    ])
    assert "Checking." in printed
    assert "[tool] Bash ls" in printed
    assert "[tool] Bash ok (2 lines)" in printed


def test_a_failed_call_says_why_on_the_command_line():
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "ssh probe-host"})]))
    printed = stream.feed(FakeMessage([ToolResult("tu_1", "denied: ssh is gated", is_error=True)]))
    assert "[tool] Bash failed: denied: ssh is gated" in printed


def test_a_printed_line_is_ascii_so_a_windows_console_survives_it():
    """`print` to a cp1252 console raises on a stray glyph, and that would take
    the whole turn down at the moment it was most worth watching."""
    stream = agent.TurnStream()
    printed = stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "ls"})]))
    printed += stream.feed(FakeMessage([ToolResult("tu_1", "x" * 6000, is_error=True)]))
    printed.encode("ascii")  # raises if anything above is not


def test_the_no_duplication_rule_still_holds_around_a_call():
    """`TurnStream` delegates text to `TextStream`, so the bug that class exists
    to prevent must not come back through the wrapper."""
    stream = agent.TurnStream()
    printed = drain(stream, [
        delta("Checking"), delta(" the ledger."),
        FakeMessage([FakeBlock("Checking the ledger."), ToolUse("tu_1", "Bash", {"command": "ls"})]),
        FakeMessage([ToolResult("tu_1", "one")]),
        delta(" Done."), FakeMessage([FakeBlock(" Done.")]),
    ])
    assert stream.text == "Checking the ledger. Done."
    assert printed.count("Checking the ledger.") == 1


def test_a_note_is_the_sessions_own_voice_at_the_end_of_the_turn():
    """How a turn that died says so: the prose that streamed before it is kept,
    and the card of the call it died on is left mid-flight."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([FakeBlock("Starting."), ToolUse("tu_1", "Bash", {"command": "ls"})]))
    stream.note("\n\n**the session failed:** `ConnectionError`")
    assert kinds(stream) == ["text", "tool", "text"]
    assert stream.blocks[1]["status"] == "running"
    assert stream.text.endswith("`ConnectionError`")


# ---------------------------------------------------------------------------
# what survives to the transcript file
# ---------------------------------------------------------------------------
def test_the_calls_survive_a_restart_and_a_transcript_without_them_still_opens():
    """`blocks` is optional on the way back in: transcripts written before tool
    calls were captured have none, and those still have to open."""
    app = pytest.importorskip("ui.app", reason="the ui extra is not installed")

    session = app.Session("test")
    session.settled = [
        {"role": "user", "text": "check the ledger"},
        {"role": "assistant", "text": "Checking.", "blocks": [
            {"kind": "text", "text": "Checking."},
            {"kind": "tool", "name": "Bash", "title": "ls", "status": "ok", "result": "one"},
        ]},
    ]
    session._persist()  # noqa: SLF001 - the round trip is the test

    reopened = app.Session("test")
    reopened.restore()
    assert reopened.settled[0] == {"role": "user", "text": "check the ledger"}
    assert reopened.settled[1]["blocks"][1]["name"] == "Bash"


def test_a_transcript_line_whose_blocks_are_junk_still_opens_the_window():
    """The file outlives the version that wrote it, so `blocks` is untrusted
    input. A block with no `kind` would reach the renderer's dispatch and take
    the page down at build time."""
    app = pytest.importorskip("ui.app", reason="the ui extra is not installed")

    session = app.Session("junk")
    session.path().parent.mkdir(parents=True, exist_ok=True)
    session.path().write_text(
        "\n".join([
            '{"role": "assistant", "text": "fine", "blocks": "not a list"}',
            '{"role": "assistant", "text": "fine", "blocks": [{"no": "kind"}, 7, {"kind": "tool"}]}',
        ]),
        encoding="utf-8",
    )
    session.restore()
    assert "blocks" not in session.settled[0]
    assert session.settled[1]["blocks"] == [{"kind": "tool"}]


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


# ---------------------------------------------------------------------------
# the reasoning (`TurnStream`, the third kind of block)
# ---------------------------------------------------------------------------
def thinking(text: str) -> FakeStreamEvent:
    return FakeStreamEvent(
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": text}}
    )


def test_the_reasoning_is_a_block_of_its_own_and_not_part_of_the_answer():
    """The chat window's statusline switches this on and off, so it has to be
    separable. `text` is what the agent *said*; the working is a different
    claim and the transcript records it as one."""
    stream = agent.TurnStream()
    stream.feed(thinking("the ceiling is the binding constraint"))
    stream.feed(delta("Raise the ceiling."))
    stream.feed(FakeMessage([
        ThinkingBlock("the ceiling is the binding constraint"),
        FakeBlock("Raise the ceiling."),
    ]))
    assert kinds(stream) == ["thinking", "text"]
    assert stream.text == "Raise the ceiling."
    assert stream.thinking == "the ceiling is the binding constraint"


def test_reasoning_is_not_printed_by_the_command_line():
    """`feed` returns what a CLI prints. The reasoning is a block a UI may draw,
    not something the terminal session starts emitting."""
    stream = agent.TurnStream()
    printed = stream.feed(thinking("hmm")) + stream.feed(delta("Yes."))
    assert printed == "Yes."


def test_a_finished_thinking_block_does_not_repeat_the_deltas_that_built_it():
    """The same trap `TextStream` exists for, one channel over: the SDK sends
    `thinking_delta` events *and* the `ThinkingBlock` containing all of them."""
    stream = agent.TurnStream()
    stream.feed(thinking("first "))
    stream.feed(thinking("second"))
    stream.feed(FakeMessage([ThinkingBlock("first second")]))
    assert stream.thinking == "first second"
    assert kinds(stream) == ["thinking"]


def test_reasoning_resumes_below_a_call_rather_than_above_it():
    """Interleaved thinking: the order is the information here too. Reasoning
    that happened *after* a command must not be appended to the block that was
    open before it ran."""
    stream = agent.TurnStream()
    stream.feed(FakeMessage([ThinkingBlock("check the ledger first")]))
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "ls"})]))
    stream.feed(FakeMessage([ToolResult("tu_1", "one")]))
    stream.feed(FakeMessage([ThinkingBlock("one entry, so the claim holds")]))
    assert kinds(stream) == ["thinking", "tool", "thinking"]
    assert [b["text"] for b in stream.blocks if b["kind"] == "thinking"] == [
        "check the ledger first",
        "one entry, so the claim holds",
    ]


def test_a_turn_that_only_reasoned_still_settles_as_something():
    """`Session.ask` keeps a turn when it produced blocks. Reasoning is blocks,
    so an interrupted turn that had only got as far as thinking is not recorded
    as a prompt that went unanswered."""
    stream = agent.TurnStream()
    stream.feed(thinking("still working out what to run"))
    assert stream.text == ""
    assert stream.blocks and stream.blocks[0]["kind"] == "thinking"


def test_a_call_is_stamped_so_the_tasks_window_can_age_it():
    """Wall clock, not monotonic: it goes into the session file and is read back
    in another process, where a monotonic reading means nothing."""
    import time

    stream = agent.TurnStream()
    stream.feed(FakeMessage([ToolUse("tu_1", "Bash", {"command": "sleep 60"})]))
    started = stream.blocks[0]["started"]
    assert isinstance(started, float)
    assert abs(started - time.time()) < 60


def test_the_reasoning_is_asked_for_as_text_rather_than_assumed(monkeypatch):
    """Capturing thinking blocks is not enough to have any: Opus 4.7+ defaults
    `display` to "omitted" and sends them with a signature and no text. The chat
    window's reasoning switch had nothing to reveal no matter how correctly the
    stream was read -- one flag away from the feature, and indistinguishable
    from a toggle that does nothing."""
    sdk = pytest.importorskip("claude_agent_sdk", reason="the SDK is not installed")
    from core import config as config_mod

    monkeypatch.setattr(agent, "system_prompt", lambda: "prompt")
    options = agent.build_options(config_mod.load(reload=True))
    assert options.thinking == {"type": "adaptive", "display": "summarized"}


def test_reasoning_can_be_turned_off_from_the_config(workspace, monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk", reason="the SDK is not installed")
    from core import config as config_mod, paths

    config = paths.root() / "config" / "grad.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[agent]\nreasoning = "omitted"\n', encoding="utf-8")
    monkeypatch.setattr(agent, "system_prompt", lambda: "prompt")
    options = agent.build_options(config_mod.load(reload=True))
    assert options.thinking == {"type": "adaptive", "display": "omitted"}


def test_an_sdk_without_the_option_still_builds_a_session(monkeypatch):
    """Feature-detected rather than assumed, for the same reason the deny probe
    exists: this option is newer than the permission mode, and the SDK's shape
    has changed between releases."""
    import dataclasses

    from core import config as config_mod

    class OldOptions:
        pass

    class OldSdk:
        ClaudeAgentOptions = dataclasses.make_dataclass("ClaudeAgentOptions", ["model"])

    assert agent.thinking_option(config_mod.load(reload=True), OldSdk) == {}
