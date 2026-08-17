"""Funnel stages 0 and 3: Haiku via the Agent SDK (HANDOFF §5).

Three things this module exists to get right.

**The SDK, not the `anthropic` package.** `client.messages.create()` resolves
Developer Platform credentials and bills per token; the subscription-backed path
is `claude_agent_sdk`. This is easy to get wrong and expensive when you do.

**Structured output without the Messages API.** `output_config.format` is a
Messages API feature and is not available here. Prompting for JSON and parsing
it fails silently on the tenth call, mid-funnel. Instead a single in-process SDK
tool is registered and made the only tool the call may use, so the payload
arrives as validated tool input rather than as text to be parsed. Two failure
modes the sketch in the handoff glosses over are handled here: the handler
validates item *shape* and returns an error result (a returned error is what
actually makes the model retry), and a turn that ends without calling the tool
at all is retried once and then fails loudly.

**Observability.** These are subagents, and §3 says we don't use subagents.
Stages 0 and 3 are the deliberate exception, so they carry the mitigation the
general rule exists to preserve: every call appends its full prompt, raw
response, and token counts to `ledger/quota.jsonl` and to a per-query log under
`notes/`. Debugging a funnel whose middle is invisible is guesswork.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from core import credentials, paths, quota_log
from core.errors import ConfigError, UpstreamError
from core.ledger_store import now_iso


def _sdk() -> Any:
    try:
        import claude_agent_sdk  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "claude-agent-sdk is not installed, so the Haiku funnel stages cannot run",
            fix="pip install claude-agent-sdk   (or run the funnel with --no-expand --no-triage)",
        ) from exc
    # Same seam as `agent.py:_sdk`: without it, a funnel run from the desktop
    # app opens a console window per Haiku stage.
    from core import spawn  # noqa: PLC0415

    spawn.mask_sdk_console()
    return claude_agent_sdk


EXPAND_PROMPT = """You expand a research question into retrieval queries.

Call submit_expansion exactly once, then stop. Do not explain yourself.

Two different retrievers need two different things, and conflating them is the
common mistake:

- `queries`: 4-6 short keyword/phrase queries for a lexical/hybrid search over
  paper full text. Use the terminology the literature actually uses, including
  synonyms and the older name for the idea if it has one. No sentences.
- `hyde`: ONE hypothetical abstract (80-150 words) that would answer the
  question if it existed. This is embedded and compared against a dense index,
  so write it as prose in the register of a real abstract. It is never sent to
  the lexical retriever, where a synthetic abstract only dilutes the query terms.
"""

TRIAGE_PROMPT = """You triage retrieved candidates against a research question.

Call submit_triage exactly once with a verdict for EVERY candidate id you were
given, then stop.

You are not re-ranking. A calibrated reranker already ordered these; your job is
to judge relevance against the actual research question rather than against a
query string. Keep a candidate if it would plausibly change how the researcher
proceeds -- direct answers, close methods, strong baselines, contradicting
results. Drop restatements of background, wrong-domain matches, and papers whose
only connection is shared vocabulary.

`reason` is one line and must be specific to this paper. It becomes the
provenance recorded in the research ledger, so "relevant to the query" is a
useless answer.
"""


# Shared with `core/mutate.py`, which has exactly the same problem: an SDK client
# inside a tool CLI the agent reached over Bash. The wording lives in
# `core/credentials.py` so the two cannot drift into saying different things
# about the same failure.
NOT_AUTHENTICATED = credentials.NOT_AUTHENTICATED
AUTH_FIX = credentials.AUTH_FIX


def _credentials_env() -> dict[str, str]:
    return credentials.sdk_env()


def _validate_expansion(args: dict[str, Any]) -> str | None:
    queries = args.get("queries")
    hyde = args.get("hyde")
    if not isinstance(queries, list) or not queries:
        return "queries must be a non-empty list of strings"
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        return "every entry in queries must be a non-empty string"
    if not isinstance(hyde, str) or len(hyde.split()) < 30:
        return "hyde must be a single hypothetical abstract of at least 30 words"
    return None


def _validate_triage(args: dict[str, Any]) -> str | None:
    verdicts = args.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        return "verdicts must be a non-empty list"
    for i, v in enumerate(verdicts):
        if not isinstance(v, dict):
            return f"verdicts[{i}] must be an object with id, keep, reason"
        if not isinstance(v.get("id"), str) or not v["id"]:
            return f"verdicts[{i}].id must be a non-empty string"
        if not isinstance(v.get("keep"), bool):
            return f"verdicts[{i}].keep must be a boolean"
        if v["keep"] and not str(v.get("reason", "")).strip():
            return f"verdicts[{i}].reason is required when keep is true"
    return None


async def _call(
    *,
    stage: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
    validate: Callable[[dict[str, Any]], str | None],
    system_prompt: str,
    user_prompt: str,
    model: str,
    role: str,
    log_name: str,
) -> dict[str, Any]:
    sdk = _sdk()
    captured: list[dict[str, Any]] = []

    @sdk.tool(tool_name, tool_description, tool_schema)
    async def _submit(args: dict[str, Any]) -> dict[str, Any]:
        problem = validate(args)
        if problem:
            # A returned error is what triggers the retry. Raising here, or
            # accepting the payload and fixing it up afterwards, both lose that.
            return {"content": [{"type": "text", "text": f"invalid payload: {problem}"}], "is_error": True}
        captured.append(args)
        return {"content": [{"type": "text", "text": "recorded"}]}

    options = sdk.ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        mcp_servers={"funnel": sdk.create_sdk_mcp_server("funnel", tools=[_submit])},
        allowed_tools=[f"mcp__funnel__{tool_name}"],
        disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
        env=_credentials_env(),
    )

    transcript: list[str] = []
    usage: Any = None
    unauthenticated = False
    try:
        async for message in sdk.query(prompt=user_prompt, options=options):
            # An unauthenticated CLI answers with a synthetic "Not logged in"
            # turn and then exits non-zero. The SDK reports the exit as
            # "Claude Code returned an error result: success" -- the CLI sends no
            # `errors` array, so the SDK falls back to printing the result
            # *subtype*, which is `success`. That message is worse than useless
            # here, so the reason is taken from the message that carries it.
            if getattr(message, "error", None) == "authentication_failed":
                unauthenticated = True
            text = _text_of(message)
            if text:
                transcript.append(text)
            usage = getattr(message, "usage", None) or usage
    except Exception as exc:  # noqa: BLE001 - re-raised unless we know better
        if unauthenticated:
            raise ConfigError(NOT_AUTHENTICATED, fix=AUTH_FIX) from exc
        raise
    if unauthenticated:
        raise ConfigError(NOT_AUTHENTICATED, fix=AUTH_FIX)

    quota_log.from_sdk_usage(
        stage, usage, model=model, role=role,
        detail={"tool": tool_name, "captured": bool(captured)},
    )
    _log_io(log_name, stage, system_prompt, user_prompt, transcript, captured)

    if not captured:
        raise _NoToolCall("the model ended its turn without calling " + tool_name)
    return captured[-1]


class _NoToolCall(RuntimeError):
    pass


def _text_of(message: Any) -> str:
    blocks = getattr(message, "content", None)
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        return "".join(getattr(b, "text", "") or "" for b in blocks)
    return ""


def _log_io(name: str, stage: str, system: str, user: str, transcript: list[str], captured: list[Any]) -> None:
    """Full prompt and raw response, per query, under notes/."""
    d = paths.notes_dir() / "funnel"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.md"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n## {stage} @ {now_iso()}\n\n")
        fh.write(f"### system\n\n```\n{system.strip()}\n```\n\n")
        fh.write(f"### user\n\n```\n{user.strip()[:8000]}\n```\n\n")
        fh.write(f"### response\n\n```\n{''.join(transcript).strip()[:8000]}\n```\n\n")
        fh.write(f"### captured\n\n```json\n{json.dumps(captured, indent=2, default=str)[:8000]}\n```\n")


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("call the async form from inside an event loop")


def _with_retry(make_coro: Callable[[], Any]) -> dict[str, Any]:
    try:
        return _run(make_coro())
    except _NoToolCall:
        # One retry, then fail the stage loudly rather than proceeding with a
        # silently empty result.
        try:
            return _run(make_coro())
        except _NoToolCall as exc:
            raise UpstreamError(
                f"the funnel stage produced no structured output: {exc}",
                fix="re-run the search; if it repeats, run with --no-expand/--no-triage and open an issue",
            ) from exc


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def structured(
    *,
    stage: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
    validate: Callable[[dict[str, Any]], str | None],
    system_prompt: str,
    user_prompt: str,
    model: str,
    role: str,
    log_name: str,
) -> dict[str, Any]:
    """One structured call through the SDK, retried once if it produces nothing.

    The named entry point for `_call`, which is the only thing in this system
    that knows how to get validated JSON out of a model without the Messages
    API, how to tell an unauthenticated CLI from a failed one, and where the
    tokens are booked. `core/wikigen.py` needs all three, and the alternative --
    a second copy of that hundred lines -- is how two code paths end up
    disagreeing about what "not logged in" looks like.

    Not restricted to Haiku despite the module name: `model` is the caller's,
    and the funnel's two stages happen to pass a Haiku id.
    """
    return _with_retry(
        lambda: _call(
            stage=stage,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema=tool_schema,
            validate=validate,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            role=role,
            log_name=log_name,
        )
    )


def expand(question: str, *, model: str, log_name: str) -> dict[str, Any]:
    """Stage 0: one question -> keyword queries for the tier-1 retriever, plus
    one HyDE abstract.

    Expansion is retriever-specific and this is easy to get wrong: HyDE is a
    dense-retrieval gain, and feeding a synthetic abstract to a lexical endpoint
    mostly dilutes the query terms. So the two outputs go to two different
    places, and the HyDE passage is embedded with the *same* model the index was
    built with -- a HyDE vector from another embedding space is noise.
    """
    schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 8},
            "hyde": {"type": "string"},
        },
        "required": ["queries", "hyde"],
    }
    return _with_retry(
        lambda: _call(
            stage=quota_log.STAGE_EXPAND,
            tool_name="submit_expansion",
            tool_description="Return the expanded queries and the HyDE passage",
            tool_schema=schema,
            validate=_validate_expansion,
            system_prompt=EXPAND_PROMPT,
            user_prompt=f"Research question:\n\n{question}",
            model=model,
            role="expand",
            log_name=log_name,
        )
    )


def triage(question: str, candidates: list[dict[str, Any]], *, model: str, log_name: str) -> list[dict[str, Any]]:
    """Stage 3: read ~50 candidates in one call, return ~15 with a reason each.

    A funnel widener, not a better ranker: the main agent can afford to read 15
    snippets, Haiku can afford 50. The per-candidate reason is not decoration --
    it is the provenance that populates the ledger's `basis` field.
    """
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "keep": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "keep"],
                },
            }
        },
        "required": ["verdicts"],
    }
    listing = "\n\n".join(
        f"[{c['id']}] {c.get('title', '(untitled)')} ({c.get('year', '?')})\n{(c.get('snippet') or c.get('abstract') or '')[:1200]}"
        for c in candidates
    )
    payload = _with_retry(
        lambda: _call(
            stage=quota_log.STAGE_TRIAGE,
            tool_name="submit_triage",
            tool_description="Return the triage verdict for every candidate",
            tool_schema=schema,
            validate=_validate_triage,
            system_prompt=TRIAGE_PROMPT,
            user_prompt=f"Research question:\n\n{question}\n\nCandidates:\n\n{listing}",
            model=model,
            role="triage",
            log_name=log_name,
        )
    )
    return list(payload.get("verdicts", []))
