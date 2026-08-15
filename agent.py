"""Grad -- the agent loop (HANDOFF §3, §9, §12 step 1).

A `ClaudeSDKClient` multi-turn session with a small system prompt, the six
built-in tools, a deny-by-default permission mode, and a `PreToolUse` gate. The
custom capability is not here: it is the CLIs in `tools/`, reached over Bash.

Three configuration details are load-bearing and easy to get wrong, so they are
asserted rather than assumed:

  * `allowed_tools` is an *auto-approve* list, not a sandbox. Built-in tools stay
    in the model's toolset regardless of what is listed, so the restriction comes
    from `disallowed_tools` (deny rules beat every other step) plus the mode.
  * the permission mode's name and semantics have changed between SDK releases,
    so `agent.py probe` attempts a call that should be denied and reports whether
    it was *denied*, not prompted and not silently allowed. Re-run it after any
    SDK upgrade.
  * `setting_sources` is left unset, so a stray `settings.json` cannot add allow
    rules silently. The whole permission configuration lives in code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import hooks
from core import config as config_mod, credentials, paths, quota_log
from core.errors import EXIT_PROJECT_BUDGET

BUILTIN_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# Everything else is denied by name. A bare-name deny rule removes the tool from
# the model's context entirely rather than denying it at call time, which is the
# behaviour we want: unavailable beats refused.
DENIED_TOOLS = ["WebSearch", "WebFetch", "NotebookEdit", "Task", "KillShell", "BashOutput"]


def _sdk() -> Any:
    try:
        import claude_agent_sdk  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "claude-agent-sdk is not installed.\n"
            "  pip install claude-agent-sdk\n"
            "and authenticate with your subscription:\n"
            "  claude setup-token   # then set CLAUDE_CODE_OAUTH_TOKEN"
        ) from exc
    return claude_agent_sdk


def system_prompt() -> str:
    return (paths.root() / "prompts" / "system.md").read_text(encoding="utf-8")


def build_options(cfg: Any, *, permission_mode: str | None = None, resume: str | None = None) -> Any:
    """The options one session runs under.

    `resume` is an SDK session id, and passing it is the difference between
    reopening a transcript and reopening a *conversation*: without it the model
    starts with no memory of the turns the window is showing above the composer,
    which is a worse failure than not resuming at all because nothing on screen
    says so. `ui/sessions.py` records the id and reports when it does not have
    one.
    """
    sdk = _sdk()
    mode = permission_mode or str(cfg.get("agent", "permission_mode", "dontAsk"))
    hook_matchers = {
        "PreToolUse": [sdk.HookMatcher(matcher="Bash", hooks=[hooks.pre_tool_use])],
        "Stop": [sdk.HookMatcher(hooks=[hooks.stop])],
    }
    return sdk.ClaudeAgentOptions(
        resume=resume,
        model=cfg.model_for("research"),
        system_prompt=system_prompt(),
        allowed_tools=BUILTIN_TOOLS,
        disallowed_tools=DENIED_TOOLS,
        permission_mode=mode,
        cwd=str(paths.root()),
        hooks=hook_matchers,
        # Off by default in the SDK, and the default is why an answer used to
        # arrive in one lump: without it `receive_response` yields nothing until
        # a whole `AssistantMessage` is finished. With it the same turn also
        # emits `StreamEvent`s carrying token deltas. `TextStream` is what turns
        # the two into one transcript -- see the warning in its docstring.
        include_partial_messages=True,
    )


def preflight_environment() -> dict[str, Any]:
    """Checks that must pass before the first turn.

    ANTHROPIC_API_KEY outranks CLAUDE_CODE_OAUTH_TOKEN in the credential chain,
    so a stray export silently bills the Developer Platform instead of the
    subscription. It is removed here rather than warned about.
    """
    from core import budget  # noqa: PLC0415

    removed = credentials.scrub_environment()
    cfg = config_mod.load()
    project_id = budget.current_project()
    return {
        "removed_env": removed,
        "oauth_token_present": bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")),
        "workspace": str(paths.root()),
        "models": cfg.models(),
        # Read from ledger/.current_project, not from the environment -- the
        # scrub above is exactly why the selection is a file (§15).
        "project": project_id,
        "project_status": budget.status(project_id) if budget.exists(project_id) else None,
        "note": (
            "auth should be subscription-backed; confirm with `claude /status`. "
            "--bare mode does not read CLAUDE_CODE_OAUTH_TOKEN, so this runs non-bare."
        ),
    }


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
async def run_session(prompt: str | None, *, once: bool) -> int:
    sdk = _sdk()
    cfg = config_mod.load()
    paths.ensure_workspace()
    env = preflight_environment()
    if env["removed_env"]:
        print(f"[grad] removed from the environment: {', '.join(env['removed_env'])}", file=sys.stderr)

    async with sdk.ClaudeSDKClient(options=build_options(cfg)) as client:
        if prompt:
            ran = await _turn(client, prompt)
            if once:
                return 0 if ran else EXIT_PROJECT_BUDGET
        while True:
            try:
                # In a worker thread: a bare input() blocks the event loop, and
                # the SDK client cannot service its transport while it waits --
                # so streaming, keepalives, and interrupts stall for the whole
                # idle period between turns.
                line = (await asyncio.to_thread(input, "\n> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in ("exit", "quit"):
                return 0
            await _turn(client, line)


def check_turn_budget() -> dict[str, Any] | None:
    """Refuse the *next* turn when the project is out of token allocation.

    HANDOFF-2 §15, and the honesty is the point: tokens are consumed
    continuously inside a turn and there is no way to refuse mid-turn, so
    **token budgets are enforced to a granularity of one turn's overrun.** This
    check is our code end to end -- it depends on no SDK behaviour -- and it runs
    before `query`, not after.

    Returns a refusal payload, or None to proceed.
    """
    try:
        from core import budget  # noqa: PLC0415

        project_id = budget.current_project()
        if not project_id or not budget.exists(project_id):
            return None
        state = budget.status(project_id)
    except Exception:  # noqa: BLE001 - accounting must never strand a session
        return None

    tokens = state["resources"]["quota_tokens"]
    if not tokens["over"]:
        return None
    overrun = tokens["spent"] - float(tokens["ceiling"])
    return {
        "project": project_id,
        "resource": "quota_tokens",
        "spent": tokens["spent"],
        "ceiling": tokens["ceiling"],
        "overrun": overrun,
        "message": (
            f"project {project_id} has used {tokens['spent']:,} of its "
            f"{int(tokens['ceiling']):,} token allocation -- {overrun:,.0f} over. "
            "Refusing the next turn; the turn that crossed the ceiling was allowed to "
            "finish, because there is no way to refuse mid-turn."
        ),
        "fix": (
            f"python -m tools.budget raise --project {project_id} "
            "--quota-tokens <new ceiling> --json"
        ),
    }


async def _turn(client: Any, prompt: str) -> bool:
    """Run one turn. Returns False if the budget refused it."""
    refusal = check_turn_budget()
    if refusal:
        print(f"\n[grad] {refusal['message']}\n[grad] fix: {refusal['fix']}", file=sys.stderr)
        return False

    await client.query(prompt)
    stream = TurnStream()
    async for message in client.receive_response():
        # Whatever has not been printed yet -- a token as it arrives, the tail
        # of a message that was never streamed, or a line naming a tool call.
        # Never both halves of the same text.
        chunk = stream.feed(message)
        if chunk:
            print(chunk, end="", flush=True)
        usage = getattr(message, "usage", None)
        if usage is not None:
            quota_log.from_sdk_usage(
                quota_log.STAGE_MAIN, usage, model=None, role="research"
            )
    print()
    return True


def _text_of(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(b, "text", "") or "" for b in content)
    return ""


def _delta_of(message: Any) -> str:
    """The visible text a partial-message stream event carries, if any.

    `StreamEvent.event` is the raw Anthropic streaming event, so this is a
    filter as much as an accessor: only `content_block_delta` carrying a
    `text_delta` is answer text. Thinking deltas and tool-input deltas are
    excluded deliberately, because `_text_of` excludes their finished blocks too
    -- a `ThinkingBlock` has `.thinking`, not `.text`. Letting them through here
    would make the stream say something the settled message does not.
    """
    event = getattr(message, "event", None)
    if not isinstance(event, dict) or event.get("type") != "content_block_delta":
        return ""
    delta = event.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return ""
    text = delta.get("text")
    return text if isinstance(text, str) else ""


class TextStream:
    """One turn's visible text, assembled from deltas *and* finished messages.

    `include_partial_messages` makes the SDK emit both halves of the same text:
    a run of `text_delta` events, and then the `AssistantMessage` that contains
    all of it. **Appending both is the bug this class exists to prevent** -- it
    is the obvious way to write the loop, and it makes every answer appear
    twice.

    So a finished message *replaces* the deltas that built it rather than
    following them. That ordering also makes the finished message authoritative:
    if the two ever disagree -- a dropped event, a turn resumed from cache, a
    message the SDK never streamed -- what stays on screen is the message, not
    the reconstruction. A turn is many messages, so this repeats per message,
    which is why `_streamed` is reset each time rather than once at the end.

    `feed` returns only the text that has not been shown yet, so a CLI can print
    its return value directly; `text` is the whole answer so far, for a UI that
    re-renders from it.
    """

    def __init__(self) -> None:
        self.text = ""
        #: The tail of `text` contributed by deltas since the last finished
        #: message -- the part a finished message is entitled to overwrite.
        self._streamed = ""

    def feed(self, message: Any) -> str:
        delta = _delta_of(message)
        if delta:
            self.text += delta
            self._streamed += delta
            return delta

        text = _text_of(message)
        # A message with no text at all -- a tool result, a system message, the
        # final result -- must leave a half-streamed block alone.
        if not text:
            return ""

        if text.startswith(self._streamed):
            unseen = text[len(self._streamed) :]
            self.text += unseen
        else:
            self.text = self.text[: len(self.text) - len(self._streamed)] + text
            unseen = ""
        self._streamed = ""
        return unseen


# ---------------------------------------------------------------------------
# tool calls
# ---------------------------------------------------------------------------
#: What one call contributes to a transcript, at most. A `Read` of a long file
#: or a training log is tens of thousands of characters, and every one of them
#: would be held for the life of the session, written to the transcript file on
#: settle, and drawn again on restore. A card is a record that the call happened
#: and how it went, not a second copy of its output.
RESULT_CHARS = 2000
RESULT_LINES = 40

#: Which input key says what a call was *on*. Anything not listed here falls
#: back to `SUBJECT_KEYS`, then to the first short string in the input -- an
#: `Edit` carries its whole replacement text, and a card head that is a wall of
#: source is worse than one that is empty.
TOOL_SUBJECT = {
    "Bash": "command",
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
}
SUBJECT_KEYS = ("command", "file_path", "path", "pattern", "query", "url", "prompt")

#: Per-row limits for the rest of a call's input. Six rows of two lines is a
#: card you can read at a glance; the whole input is not.
ROW_CHARS = 200
ROW_LINES = 2
MAX_ROWS = 6


def clip(text: str, *, chars: int = RESULT_CHARS, lines: int = RESULT_LINES) -> str:
    """`text`, bounded -- and saying what it dropped rather than trailing off.

    ASCII on purpose: this can reach a Windows console, where a stray `…` is a
    `UnicodeEncodeError` that would take the turn down.
    """
    if not text:
        return ""
    split = text.splitlines()
    dropped_lines = max(0, len(split) - lines)
    out = "\n".join(split[:lines])
    dropped_chars = max(0, len(out) - chars)
    out = out[:chars]
    if dropped_chars:
        out += f"\n... +{dropped_chars:,} more characters"
    if dropped_lines:
        out += f"\n... +{dropped_lines:,} more lines"
    return out


def _one_line(text: str, limit: int = 120) -> str:
    """A subject collapsed onto one line, for a card head."""
    flattened = " ".join(str(text).split())
    return flattened if len(flattened) <= limit else flattened[: limit - 3] + "..."


def describe_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, str]:
    """`(subject_key, subject)` -- what this call was on, and under which key."""
    keys = (TOOL_SUBJECT[name],) if name in TOOL_SUBJECT else SUBJECT_KEYS
    for key in keys:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    for key, value in tool_input.items():
        if isinstance(value, str) and value.strip() and len(value) <= ROW_CHARS:
            return key, value
    return "", ""


def _content_blocks(message: Any) -> list[Any]:
    content = getattr(message, "content", None)
    return content if isinstance(content, list) else []


def _tool_uses(message: Any) -> list[dict[str, Any]]:
    """Every `ToolUseBlock` in a finished message, as plain data.

    Duck-typed rather than isinstance-checked so `ServerToolUseBlock` -- a tool
    the API runs on the model's behalf -- draws the same card, and so this file
    keeps working if the SDK renames a class.
    """
    uses: list[dict[str, Any]] = []
    for block in _content_blocks(message):
        name = getattr(block, "name", None)
        tool_input = getattr(block, "input", None)
        identifier = getattr(block, "id", None)
        if isinstance(name, str) and isinstance(tool_input, dict) and isinstance(identifier, str):
            uses.append({"id": identifier, "name": name, "input": tool_input})
    return uses


def _tool_results(message: Any) -> list[dict[str, Any]]:
    """Every `ToolResultBlock`. These arrive on a `UserMessage`, not on the
    assistant's -- the tool ran on this side of the wire and is reporting back."""
    results: list[dict[str, Any]] = []
    for block in _content_blocks(message):
        identifier = getattr(block, "tool_use_id", None)
        if not isinstance(identifier, str):
            continue
        results.append(
            {
                "id": identifier,
                "text": _result_text(getattr(block, "content", None)),
                "is_error": bool(getattr(block, "is_error", False)),
            }
        )
    return results


def _result_text(content: Any) -> str:
    """A result's content, which is a string, a list of blocks, or nothing."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                parts.append(text if isinstance(text, str) else json.dumps(item, default=str))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def tool_block(use: dict[str, Any]) -> dict[str, Any]:
    """One tool card, before its result lands."""
    tool_input = use["input"]
    subject_key, subject = describe_tool(use["name"], tool_input)
    rows = [
        (key, clip(str(value), chars=ROW_CHARS, lines=ROW_LINES))
        for key, value in tool_input.items()
        if key != subject_key
    ]
    return {
        "kind": "tool",
        "id": use["id"],
        "name": use["name"],
        "title": _one_line(subject),
        "text": clip(subject, chars=600, lines=12),
        "rows": rows[:MAX_ROWS],
        "status": "running",
        "result": "",
    }


class TurnStream:
    """One turn as an ordered list of blocks: prose, and the tool calls between.

    `TextStream` answers what the agent *said*; this answers what it *did*, and
    that was the larger half of most turns here -- every capability in this
    project is reached by a `Bash` into `tools/`, and none of it was visible. A
    turn that ran six commands and then summarised them arrived as the summary
    alone, which is exactly the part you cannot check.

    A turn is kept as blocks rather than as one string because **the order is
    the information**: which command ran before which claim. So a run of prose
    is cut at each tool call, and `blocks` reads top to bottom as the turn
    happened. Text assembly is delegated to `TextStream` unchanged, including
    its rule that a finished message replaces the deltas that built it.

    `feed` returns what a CLI should print next; a UI re-renders from `blocks`.
    """

    def __init__(self) -> None:
        self.blocks: list[dict[str, Any]] = []
        self._text = TextStream()
        #: The block the current run of prose is accumulating into, if open.
        self._open: dict[str, Any] | None = None
        #: Tool blocks by call id, so a result can find the call it answers.
        self._calls: dict[str, dict[str, Any]] = {}

    @property
    def text(self) -> str:
        """The turn's prose, tool cards left out -- what a plain transcript says."""
        return "".join(b["text"] for b in self.blocks if b["kind"] == "text")

    def feed(self, message: Any) -> str:
        printed = self._feed_text(message)
        for use in _tool_uses(message):
            block = tool_block(use)
            self.blocks.append(block)
            self._calls[block["id"]] = block
            # A tool call ends the run of prose above it: whatever the agent says
            # next belongs *below* the card, because that is when it said it.
            self._text = TextStream()
            self._open = None
            printed += _tool_line(block)
        for result in _tool_results(message):
            block = self._calls.get(result["id"])
            if block is None:
                # A result for a call this stream never saw -- a turn resumed
                # from cache, a subagent's tool. Nothing to attach it to, and
                # inventing a card for it would claim an order we do not know.
                continue
            block["result"] = clip(result["text"])
            block["status"] = "error" if result["is_error"] else "ok"
            printed += _result_line(block)
        return printed

    def note(self, text: str) -> None:
        """Append text the session itself has to say -- that a turn died, say."""
        if not text:
            return
        self._text = TextStream()
        self._open = {"kind": "text", "text": text}
        self.blocks.append(self._open)

    def active(self) -> dict[str, Any] | None:
        """The call currently in flight, for a status line to name."""
        for block in reversed(self.blocks):
            if block["kind"] == "tool" and block["status"] == "running":
                return block
        return None

    def _feed_text(self, message: Any) -> str:
        chunk = self._text.feed(message)
        # Synced from `TextStream.text`, never `+=`: a finished message may
        # rewrite the tail its own deltas built, and the block has to follow.
        if self._text.text:
            if self._open is None:
                self._open = {"kind": "text", "text": ""}
                self.blocks.append(self._open)
            self._open["text"] = self._text.text
        return chunk


def _tool_line(block: dict[str, Any]) -> str:
    subject = f" {block['title']}" if block["title"] else ""
    return f"\n[tool] {block['name']}{subject}\n"


def _result_line(block: dict[str, Any]) -> str:
    if block["status"] == "error":
        first = next((line for line in block["result"].splitlines() if line.strip()), "failed")
        return f"[tool] {block['name']} failed: {_one_line(first, 160)}\n"
    lines = len(block["result"].splitlines())
    return f"[tool] {block['name']} ok ({lines} line{'' if lines == 1 else 's'})\n"


# ---------------------------------------------------------------------------
# the deny probe (§9, §12 step 1)
# ---------------------------------------------------------------------------
async def run_probe() -> int:
    """Attempt a call that should be denied and report what actually happened.

    The whole safety story rests on the exact name and semantics of the
    permission mode in the *installed* SDK, and those have changed between
    releases. This reports three distinguishable outcomes -- denied, prompted,
    allowed -- because only the first is acceptable.
    """
    hook_results = hooks.probe()
    print(json.dumps({"hook_probe": hook_results}, indent=2))

    try:
        sdk = _sdk()
    except SystemExit as exc:
        print(f"\n[grad] SDK not installed; the hook half of the probe still ran.\n{exc}", file=sys.stderr)
        return 1

    cfg = config_mod.load()
    outcome = {"attempted": "ssh probe-host echo hello", "result": "unknown"}
    transcript: list[str] = []
    hooks.DENIALS.clear()
    try:
        async with sdk.ClaudeSDKClient(options=build_options(cfg)) as client:
            await client.query(
                "Run exactly this shell command and report the outcome verbatim: "
                "ssh probe-host echo hello"
            )
            async for message in client.receive_response():
                transcript.append(_text_of(message))
    except Exception as exc:  # noqa: BLE001 - the probe reports failures, it does not raise them
        outcome["result"] = f"error: {exc}"
        print(json.dumps({"live_probe": outcome}, indent=2))
        return 1

    # The verdict comes from the hook's own record of what it refused, not from
    # words in the transcript. Substring matching gets this wrong in both
    # directions -- the deny message contains "gpu.py", and a model narrating a
    # successful run can say "denied" -- and a false `denied` is the single
    # outcome this probe must never produce.
    joined = "".join(transcript)
    denied_here = [d for d in hooks.DENIALS if "ssh" in d["command"]]
    outcome["hook_denials"] = denied_here
    if denied_here:
        outcome["result"] = "denied"
    elif "hello" in joined:
        outcome["result"] = "ALLOWED -- the mode is not denying by default"
    else:
        # The model may simply have declined to try. That is not evidence the
        # gate works, so it is not reported as though it were.
        outcome["result"] = "inconclusive; the command may never have been attempted"
    outcome["transcript"] = joined[-2000:]
    print(json.dumps({"live_probe": outcome}, indent=2))
    return 0 if outcome["result"] == "denied" else 1


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="grad",
        description="Grad -- a personal research agent for mathematics and machine learning.",
    )
    parser.add_argument("prompt", nargs="*", help="prompt for a single turn; omit for a session")
    parser.add_argument("--once", action="store_true", help="exit after the first response")
    parser.add_argument("--probe", action="store_true", help="run the §9 permission deny probe and exit")
    parser.add_argument("--ui", action="store_true", help="launch the NiceGUI desktop app instead")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="port for --ui; move it when something else already holds 8080",
    )
    parser.add_argument("--check", action="store_true", help="report environment and auth posture, then exit")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(preflight_environment(), indent=2))
        return
    if args.probe:
        raise SystemExit(asyncio.run(run_probe()))
    if args.ui:
        from ui.app import run as run_ui  # noqa: PLC0415

        # A non-default port also moves the app's origin, and the embedded Lab
        # scopes its `frame-ancestors` to that origin -- so `tools.lab` needs
        # `--ui-origin http://127.0.0.1:<port>` to match, or the iframe is blocked.
        run_ui(port=args.port)
        return

    prompt = " ".join(args.prompt) if args.prompt else None
    raise SystemExit(asyncio.run(run_session(prompt, once=args.once or bool(prompt))))


if __name__ == "__main__":
    main()
