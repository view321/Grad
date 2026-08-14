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


def build_options(cfg: Any, *, permission_mode: str | None = None) -> Any:
    sdk = _sdk()
    mode = permission_mode or str(cfg.get("agent", "permission_mode", "dontAsk"))
    hook_matchers = {
        "PreToolUse": [sdk.HookMatcher(matcher="Bash", hooks=[hooks.pre_tool_use])],
        "Stop": [sdk.HookMatcher(hooks=[hooks.stop])],
    }
    return sdk.ClaudeAgentOptions(
        model=cfg.model_for("research"),
        system_prompt=system_prompt(),
        allowed_tools=BUILTIN_TOOLS,
        disallowed_tools=DENIED_TOOLS,
        permission_mode=mode,
        cwd=str(paths.root()),
        hooks=hook_matchers,
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
    async for message in client.receive_response():
        text = _text_of(message)
        if text:
            print(text, end="", flush=True)
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
    parser.add_argument("--check", action="store_true", help="report environment and auth posture, then exit")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(preflight_environment(), indent=2))
        return
    if args.probe:
        raise SystemExit(asyncio.run(run_probe()))
    if args.ui:
        from ui.app import run as run_ui  # noqa: PLC0415

        run_ui()
        return

    prompt = " ".join(args.prompt) if args.prompt else None
    raise SystemExit(asyncio.run(run_session(prompt, once=args.once or bool(prompt))))


if __name__ == "__main__":
    main()
