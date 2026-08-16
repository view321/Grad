"""The mutation operator: Sonnet 5 through the Agent SDK (HANDOFF-2 §21).

This is the half of an evolutionary search that proposes, and it replaces
ShinkaEvolve's. Three reasons, and only the first was in the original plan:

1. **Loop ownership.** `ShinkaEvolveRunner.run()` owns the generation loop, and
   the campaign budget gate needs the boundary between generations. That was
   §23 item 1, and `tools/evolve.py capabilities` still reports it.
2. **There is no generation boundary to own.** The 0.0.7 runner is async:
   `max_proposal_jobs` and `max_evaluation_jobs` keep several proposals in
   flight and they complete out of order. Forking it to yield one generation at
   a time means removing the concurrency that makes it fast.
3. **The token rail, which is the one that would have cost money quietly.**
   Shinka's subscription path is `headless/claude`, which shells out to
   `npx @roberttlange/headless` and drives the Claude Code CLI. Every mutation
   would be a model call `agent.drive_turn` never issued, spending Max quota
   `ledger/quota.jsonl` cannot see and `budget.check` cannot bound -- on the one
   loop in this system explicitly designed to run with no human in it. The
   README's whole argument about the ceiling that could see one per cent of the
   tokens is about exactly this class of blindness.

So the operator is here, it goes through the SDK the way `core/haiku.py` does,
and `quota_log.from_sdk_usage` records what it actually spent. `evolve.mutate`
is a stage `tools.quota summary` totals like any other.

**The model never returns a file.** It returns the contents of the mutable
region, and `campaign.replace_blocks` splices them between markers taken from
the baseline. Code outside the region cannot change, because nothing outside the
region was ever in the model's output -- the imports, the entry point and the
I/O the campaign was preflighted against are the baseline's by construction.
That turns §21's third collision from a check into a property. The escape check
still runs on the result, and still earns its place: a replacement that
*contains* marker text would move real code across the boundary, and at that
point the markers are data rather than structure.

Structured output is forced the way `core/haiku.py` forces it -- one in-process
SDK tool, made the only tool the call may use -- for the reason given there: a
prompt asking for JSON fails silently on the tenth call, in the middle of a
campaign nobody is watching.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from core import campaign as camp, credentials, evolution, quota_log
from core.errors import ConfigError, GradError, UpstreamError

#: How much of an inspiration's source is shown. A campaign shows four elites
#: plus a parent every proposal, so this multiplies by five against a prompt that
#: is re-sent for every candidate of every generation. Enough to see the shape of
#: an idea; not enough to be a second copy of the program.
INSPIRATION_CHARS = 1200

#: The parent and the baseline are shown whole -- they are what is being edited,
#: and an operator working from a truncated parent writes `find` strings that
#: match nothing.
MAX_SOURCE_CHARS = 24_000

SYSTEM_PROMPT = """You are a mutation operator in an evolutionary program search.

You are given one program and asked to propose ONE variant of it. Call the tool
exactly once, then stop. Do not explain yourself outside the tool call.

## What you may change

Only the contents of the mutable region. You are shown the whole file so you can
see what is around it -- the imports, the entry point, how the region is called
-- but everything outside the markers is fixed and you cannot edit it. Do not
emit the marker lines themselves; you are writing what goes between them.

If the change you want needs a new import or a different entry point, say so in
`rationale` and make the best change you can inside the region. A mutation that
escapes the region costs a fresh remote smoke run, which is the one expensive
thing in this loop.

## What you are optimising

`combined_score`, higher is better. Read the evaluator if you were shown it.

**Do not tamper with the measurement.** Special-casing the evaluator's inputs,
memoising its expected outputs, catching the exception it uses to detect
failure, or writing to files it reads are all ways to raise the score without
improving the program, and all of them are treated as bugs when they are found.
The search records the components of the score alongside the scalar precisely so
this shows up. Optimise the thing being measured.

## How to propose

One idea per mutation, carried through properly, rather than three hedged
changes at once -- the search recombines ideas itself, and it can only do that
if each candidate carries one. Say what you expect the change to do and why in
`rationale`; that line is kept in the ledger next to the score, and it is what
makes a lineage readable afterwards.

You are shown candidates that already ran, with their scores, and candidates
that failed, with their errors. Both are evidence. A failure that recurs across
several candidates is usually a fact about the environment rather than about the
code -- read it before writing the same thing again."""

PATCH_GUIDANCE = {
    evolution.PATCH_DIFF: """## This mutation: a targeted edit

Return `edits`: exact find/replace pairs against the parent's region. Each
`find` must appear EXACTLY ONCE in the region, copied character for character
including indentation -- an edit that matches zero times or twice is rejected
and the candidate is wasted. Include enough surrounding context to be unique.

Use this for a change you can point at: a constant, a formula, a condition, one
loop body.""",
    evolution.PATCH_FULL: """## This mutation: a rewrite

Return `blocks`: the complete new contents of each mutable region, in order.
Whatever you do not want to change, reproduce unchanged.

Use this for a change of approach -- a different algorithm, a restructured
computation. If the change is a constant, you have been given the wrong operator
and should still return the whole region.""",
    evolution.PATCH_CROSS: """## This mutation: a recombination

You are given TWO parents from different sub-populations. Return `blocks`: the
complete new contents of each mutable region, combining what is working in each.

This is not "average the two". Identify what each parent does that the other
does not, and write a version that has both. If the two ideas are genuinely
incompatible, say so in `rationale` and carry the better one further rather than
producing a compromise that has neither.""",
}


def _sdk() -> Any:
    try:
        import claude_agent_sdk  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "claude-agent-sdk is not installed, so there is no mutation operator",
            fix="pip install -e '.[agent]'",
        ) from exc
    # The same seam `agent.py` and `core/haiku.py` take: without it, a campaign
    # run from the desktop app opens a console window per proposal -- and a
    # campaign is the one caller that makes dozens of them.
    from core import spawn  # noqa: PLC0415

    spawn.mask_sdk_console()
    return claude_agent_sdk


class NoToolCall(RuntimeError):
    """The turn ended without calling the tool. Retried once, then reported."""


# ---------------------------------------------------------------------------
# schemas and validation
# ---------------------------------------------------------------------------
def _schema(patch_type: str, block_count: int) -> dict[str, Any]:
    if patch_type == evolution.PATCH_DIFF:
        return {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "block": {
                                "type": "integer",
                                "description": "which mutable region, 0-based",
                            },
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                        },
                        "required": ["find", "replace"],
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": ["edits", "rationale"],
        }
    return {
        "type": "object",
        "properties": {
            "blocks": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": block_count,
                "maxItems": block_count,
            },
            "rationale": {"type": "string"},
        },
        "required": ["blocks", "rationale"],
    }


def _markers_in(text: str) -> bool:
    return camp.BLOCK_START in text or camp.BLOCK_END in text


def _validator(patch_type: str, block_count: int) -> Callable[[dict[str, Any]], str | None]:
    """The tool handler's check. A returned error is what makes the model retry.

    Marker text is rejected *here* rather than left to the escape check
    downstream, and the difference is a whole candidate: rejected here the model
    is told and writes it again without the markers; caught downstream the
    candidate is recorded as an escape, never evaluated, and the generation is
    one proposal short. `escaped_evolve_block` remains the backstop for the case
    this cannot see, which is a payload that arrives past a retry.
    """

    def check(args: dict[str, Any]) -> str | None:
        if not str(args.get("rationale", "")).strip():
            return "rationale is required: it is recorded beside the score and is what makes a lineage readable"
        if patch_type == evolution.PATCH_DIFF:
            edits = args.get("edits")
            if not isinstance(edits, list) or not edits:
                return "edits must be a non-empty list of {block, find, replace}"
            for i, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return f"edits[{i}] must be an object with find and replace"
                find, replace = edit.get("find"), edit.get("replace")
                if not isinstance(find, str) or not find.strip():
                    return f"edits[{i}].find must be a non-empty string copied from the region"
                if not isinstance(replace, str):
                    return f"edits[{i}].replace must be a string (empty to delete)"
                index = edit.get("block", 0)
                if not isinstance(index, int) or not 0 <= index < block_count:
                    return f"edits[{i}].block must be between 0 and {block_count - 1}"
                if _markers_in(find) or _markers_in(replace):
                    return (
                        f"edits[{i}] contains an EVOLVE-BLOCK marker; you edit the contents "
                        "of a region, never its boundaries"
                    )
            return None
        blocks = args.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != block_count:
            return f"blocks must be a list of exactly {block_count} string(s), one per mutable region"
        for i, block in enumerate(blocks):
            if not isinstance(block, str):
                return f"blocks[{i}] must be a string"
            if _markers_in(block):
                return (
                    f"blocks[{i}] contains an EVOLVE-BLOCK marker; return only what goes "
                    "between the markers, which are added for you"
                )
        return None

    return check


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [+{len(text) - limit:,} characters not shown]"


def _describe(candidate: dict[str, Any], *, source: str | None, metric: str) -> str:
    score = evolution.score_of(candidate, metric=metric)
    head = f"[{candidate.get('candidate_id')}] island {evolution.island_of(candidate)}"
    if score is not None:
        head += f" · {metric} = {score:.6g}"
    metrics = {k: v for k, v in (candidate.get("metrics") or {}).items() if k != metric}
    if metrics:
        head += " · " + ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:6])
    if candidate.get("rationale"):
        head += f"\n  intent: {candidate['rationale']}"
    if candidate.get("error"):
        head += f"\n  FAILED: {candidate['error']}"
    if source:
        head += f"\n```python\n{_clip(source, INSPIRATION_CHARS)}\n```"
    return head


def build_prompt(
    plan: dict[str, Any],
    *,
    baseline: str,
    parent_source: str,
    task_brief: str,
    evaluator: str,
    source_of: Callable[[dict[str, Any]], str | None],
    metric: str = evolution.METRIC,
) -> str:
    """The user turn for one proposal.

    Assembled here rather than inline so it is testable without an SDK, which is
    the same reason `hooks.evaluate_bash` is a pure function: the prompt is the
    operator's entire interface to the search, and a change to it that quietly
    drops the failures section is invisible until a campaign spends a generation
    reproducing one crash.
    """
    parent = plan.get("parent")
    mate = plan.get("mate")
    parts: list[str] = []

    if task_brief.strip():
        parts.append(f"# The task\n\n{task_brief.strip()}")

    parts.append(
        "# The program\n\nEverything between the EVOLVE-BLOCK markers is yours to change; "
        "everything else is fixed.\n\n```python\n"
        + _clip(baseline, MAX_SOURCE_CHARS)
        + "\n```"
    )
    if evaluator.strip():
        parts.append(
            "# How it is scored\n\nThis is the evaluator. You cannot change it; you are "
            "shown it so you know what is being measured.\n\n```python\n"
            + _clip(evaluator, MAX_SOURCE_CHARS)
            + "\n```"
        )

    if parent is None:
        parts.append(
            "# Where you are\n\nGeneration 0: nothing has been evaluated yet, so you are "
            "mutating the baseline above. Propose the change you think most likely to help, "
            "and make it one change."
        )
    else:
        score = evolution.score_of(parent, metric=metric)
        header = (
            f"# The parent\n\n`{parent.get('candidate_id')}` on island "
            f"{evolution.island_of(parent)}"
            + (f", {metric} = {score:.6g}" if score is not None else "")
            + ". This is the region you are editing:"
        )
        parts.append(header + f"\n\n```python\n{_clip(parent_source, MAX_SOURCE_CHARS)}\n```")

    if mate is not None:
        mate_source = source_of(mate) or ""
        parts.append(
            "# The second parent\n\n"
            + _describe(mate, source=mate_source, metric=metric)
        )

    elites = [c for c in plan.get("elites") or [] if c.get("candidate_id") != (parent or {}).get("candidate_id")]
    if elites:
        parts.append(
            "# What has scored best so far\n\n"
            + "\n\n".join(_describe(c, source=source_of(c), metric=metric) for c in elites)
        )

    failures = plan.get("failures") or []
    if failures:
        parts.append(
            "# What has failed\n\nThese produced no score. Read the errors before writing "
            "something that would hit the same one.\n\n"
            + "\n\n".join(_describe(c, source=None, metric=metric) for c in failures)
        )

    parts.append(
        f"Propose one variant now. Patch type: `{plan.get('patch_type')}`. "
        "Call the tool exactly once."
    )
    return "\n\n".join(parts)


def system_prompt(patch_type: str) -> str:
    return SYSTEM_PROMPT + "\n\n" + PATCH_GUIDANCE.get(patch_type, PATCH_GUIDANCE[evolution.PATCH_FULL])


# ---------------------------------------------------------------------------
# applying the payload
# ---------------------------------------------------------------------------
def apply_payload(
    payload: dict[str, Any], *, parent_full_source: str, patch_type: str
) -> tuple[str, list[str]]:
    """Turn a validated tool payload into a candidate's full source.

    Returns `(source, problems)`. Problems are per-candidate failures, not
    campaign failures: a `find` that matched twice is one wasted proposal, and
    the campaign records it with the reason and carries on.
    """
    blocks = camp.block_texts(parent_full_source)
    if patch_type == evolution.PATCH_DIFF:
        edited = list(blocks)
        problems: list[str] = []
        by_block: dict[int, list[dict[str, Any]]] = {}
        for edit in payload.get("edits", []):
            by_block.setdefault(int(edit.get("block", 0)), []).append(edit)
        for index, edits in sorted(by_block.items()):
            if not 0 <= index < len(edited):
                problems.append(f"edit targets region {index}, which does not exist")
                continue
            edited[index], trouble = camp.apply_edits(edited[index], edits)
            problems.extend(trouble)
        if problems:
            return "", problems
        return camp.replace_blocks(parent_full_source, edited), []

    new_blocks = [str(b) for b in payload.get("blocks", [])]
    if len(new_blocks) != len(blocks):
        return "", [f"expected {len(blocks)} region(s), got {len(new_blocks)}"]
    return camp.replace_blocks(parent_full_source, new_blocks), []


# ---------------------------------------------------------------------------
# one proposal
# ---------------------------------------------------------------------------
async def _call_once(
    *,
    plan: dict[str, Any],
    user_prompt: str,
    block_count: int,
    model: str,
    project: str | None,
) -> tuple[dict[str, Any], str]:
    sdk = _sdk()
    patch_type = str(plan.get("patch_type") or evolution.PATCH_FULL)
    captured: list[dict[str, Any]] = []
    validate = _validator(patch_type, block_count)

    @sdk.tool("submit_mutation", "Return one proposed variant", _schema(patch_type, block_count))
    async def _submit(args: dict[str, Any]) -> dict[str, Any]:
        problem = validate(args)
        if problem:
            return {"content": [{"type": "text", "text": f"invalid payload: {problem}"}], "is_error": True}
        captured.append(args)
        return {"content": [{"type": "text", "text": "recorded"}]}

    options = sdk.ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt(patch_type),
        mcp_servers={"evolve": sdk.create_sdk_mcp_server("evolve", tools=[_submit])},
        allowed_tools=["mcp__evolve__submit_mutation"],
        # The operator writes code; it does not run any. Everything it needs is
        # in the prompt, and a mutation operator with `Bash` is an evolutionary
        # loop with a shell in it -- which is the one thing §21's first collision
        # says must not exist.
        disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "Task"],
        env=credentials.sdk_env(),
    )

    transcript: list[str] = []
    usage: Any = None
    unauthenticated = False
    try:
        async for message in sdk.query(prompt=user_prompt, options=options):
            if getattr(message, "error", None) == "authentication_failed":
                unauthenticated = True
            text = _text_of(message)
            if text:
                transcript.append(text)
            usage = getattr(message, "usage", None) or usage
    except Exception as exc:  # noqa: BLE001 - re-raised unless we know better
        if unauthenticated:
            raise ConfigError(credentials.NOT_AUTHENTICATED, fix=credentials.AUTH_FIX) from exc
        raise
    if unauthenticated:
        raise ConfigError(credentials.NOT_AUTHENTICATED, fix=credentials.AUTH_FIX)

    # In every path that reached the model, including one that then fails to
    # produce a payload. A proposal that ended without calling the tool spent its
    # tokens, and `hooks.stop`'s docstring is about exactly what happens when the
    # accounting misses the calls most worth accounting for.
    quota_log.from_sdk_usage(
        quota_log.STAGE_EVOLVE,
        usage,
        model=model,
        role="evolve",
        project=project,
        detail={
            "patch_type": patch_type,
            "generation": plan.get("generation"),
            "index": plan.get("index"),
            "island": plan.get("island"),
            "parent": (plan.get("parent") or {}).get("candidate_id"),
            "captured": bool(captured),
        },
    )
    joined = "".join(transcript)
    if not captured:
        raise NoToolCall("the operator ended its turn without calling submit_mutation")
    return captured[-1], joined


def _text_of(message: Any) -> str:
    blocks = getattr(message, "content", None)
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        return "".join(getattr(b, "text", "") or "" for b in blocks)
    return ""


async def propose(
    plan: dict[str, Any],
    *,
    baseline: str,
    task_brief: str,
    evaluator: str,
    source_of: Callable[[dict[str, Any]], str | None],
    model: str,
    project: str | None = None,
    log_dir: Path | None = None,
    metric: str = evolution.METRIC,
) -> dict[str, Any]:
    """One proposal. Never raises for a per-candidate failure.

    A malformed payload, an edit that did not match, an operator that would not
    call the tool: all of these are one wasted candidate and are returned as
    `{"error": ...}` for the driver to record. Only a *campaign* failure --
    missing credentials, no SDK -- is raised, because carrying on would produce
    the same failure for every remaining candidate of every remaining generation.

    The parent's source is resolved here rather than passed in, because every
    plan in a generation may have a different parent: `propose_all` fans these
    out concurrently, and a single `parent_source` argument would have every
    candidate in the generation edit whichever one the caller happened to read.
    """
    parent = plan.get("parent")
    parent_source = (source_of(parent) if parent else None) or baseline
    block_count = len(camp.block_texts(parent_source))
    user_prompt = build_prompt(
        plan,
        baseline=baseline,
        parent_source=parent_source,
        task_brief=task_brief,
        evaluator=evaluator,
        source_of=source_of,
        metric=metric,
    )
    patch_type = str(plan.get("patch_type") or evolution.PATCH_FULL)

    payload: dict[str, Any] | None = None
    transcript = ""
    error: str | None = None
    for attempt in (1, 2):
        try:
            payload, transcript = await _call_once(
                plan=plan,
                user_prompt=user_prompt,
                block_count=block_count,
                model=model,
                project=project,
            )
            break
        except NoToolCall as exc:
            # One retry, matching `core/haiku.py`. A second failure is a
            # candidate that produced nothing, not a campaign that must stop.
            error = str(exc)
            if attempt == 2:
                payload = None
        except GradError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad proposal is not a bad campaign
            error = f"{type(exc).__name__}: {exc}"
            payload = None
            break

    result: dict[str, Any] = {
        "patch_type": patch_type,
        "island": plan.get("island"),
        "index": plan.get("index"),
        "generation": plan.get("generation"),
        "parent_id": (plan.get("parent") or {}).get("candidate_id"),
        "mate_id": (plan.get("mate") or {}).get("candidate_id"),
        "source": "",
        "rationale": "",
        "error": error,
    }
    if payload is not None:
        source, problems = apply_payload(
            payload, parent_full_source=parent_source, patch_type=patch_type
        )
        result["rationale"] = str(payload.get("rationale", "")).strip()
        if problems:
            result["error"] = "; ".join(problems)
        else:
            result["source"] = source
            result["error"] = None

    if log_dir is not None:
        _log(log_dir, plan=plan, system=system_prompt(patch_type), user=user_prompt,
             transcript=transcript, payload=payload, result=result)
    return result


def _log(
    log_dir: Path,
    *,
    plan: dict[str, Any],
    system: str,
    user: str,
    transcript: str,
    payload: Any,
    result: dict[str, Any],
) -> None:
    """The full prompt and raw response, beside the candidate they produced.

    Under the candidate's own artifacts directory rather than in `notes/`, which
    is where `core/haiku.py` puts the funnel's equivalent. A funnel query is
    something a person asked and will want to reread; a mutation is one row of a
    campaign that may have hundreds, and piling those into the notes directory
    would bury the notes. Here it sits next to `initial.py` and `evaluate.log` --
    the three files that together say what was proposed, why, and what happened.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        body = [
            f"# {plan.get('generation')}.{plan.get('index')} · {plan.get('patch_type')}",
            "",
            f"island {plan.get('island')} · parent {result.get('parent_id')}"
            + (f" · mate {result['mate_id']}" if result.get("mate_id") else ""),
            "",
            "## system", "", "```", system.strip(), "```", "",
            "## user", "", "```", user.strip()[:20000], "```", "",
            "## response", "", "```", (transcript or "").strip()[:8000], "```", "",
            "## payload", "", "```json",
            json.dumps(payload, indent=2, default=str)[:20000] if payload else "null",
            "```", "",
            f"## outcome\n\n{result.get('error') or 'applied'}",
        ]
        (log_dir / "mutation.md").write_text("\n".join(body), encoding="utf-8")
    except OSError:
        # A log that cannot be written must not lose the candidate it describes.
        pass


# ---------------------------------------------------------------------------
# a generation's worth, concurrently
# ---------------------------------------------------------------------------
async def propose_all_async(
    plans: Sequence[dict[str, Any]],
    *,
    jobs: int,
    log_dir_for: Callable[[dict[str, Any]], Path | None],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Every plan in a generation, at most `jobs` in flight.

    The proposals within a generation are independent by construction --
    `evolution.plan_generation` computes the whole plan from the ledger before
    any model is called, precisely so that they are -- so this is the one place
    in the campaign where concurrency is free of ordering questions. It is also
    where the wall clock actually goes: a population of four at forty seconds a
    proposal is a bit under three minutes serially and about forty seconds at
    `--jobs 4`.

    The semaphore is not decoration. Each proposal spawns a `claude` process, and
    an unbounded generation of thirty-two would spawn thirty-two of them.
    """
    semaphore = asyncio.Semaphore(max(1, jobs))

    async def one(plan: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await propose(plan, log_dir=log_dir_for(plan), **kwargs)

    settled = await asyncio.gather(*(one(plan) for plan in plans), return_exceptions=True)
    out: list[dict[str, Any]] = []
    fatal: BaseException | None = None
    for plan, item in zip(plans, settled):
        if isinstance(item, BaseException):
            # A campaign-level failure stops the campaign, but only after every
            # sibling has settled -- `gather` with `return_exceptions` is what
            # keeps the other proposals from being cancelled mid-call, leaving
            # `claude` processes behind with no one to reap them.
            if fatal is None and isinstance(item, (GradError, KeyboardInterrupt, SystemExit)):
                fatal = item
            out.append(
                {
                    "patch_type": plan.get("patch_type"),
                    "island": plan.get("island"),
                    "index": plan.get("index"),
                    "generation": plan.get("generation"),
                    "parent_id": (plan.get("parent") or {}).get("candidate_id"),
                    "source": "",
                    "rationale": "",
                    "error": f"{type(item).__name__}: {item}",
                }
            )
            continue
        out.append(item)
    if fatal is not None:
        raise fatal
    return out


def propose_all(plans: Sequence[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    """Synchronous entry point, for a driver that is not already async."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(propose_all_async(plans, **kwargs))
    raise UpstreamError(
        "propose_all was called from inside an event loop",
        fix="await propose_all_async instead",
    )
