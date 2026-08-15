"""When a conversation is compacted, and what survives it.

The SDK offers no way to ask for this. Its control protocol has ten subtypes --
`initialize`, `mcp_status`, `get_context_usage`, `interrupt`,
`set_permission_mode`, `set_model`, `rewind_files`, `mcp_reconnect`,
`mcp_toggle`, `stop_task` -- and none of them is "compact". The CLI underneath
does compact on its own, and a live session reports the threshold as 967,000 of
a 1,000,000 window, but that number is not reachable from here either: it comes
from settings, and `agent.py` deliberately leaves `setting_sources` unset so
that a stray `settings.json` cannot add permission rules behind the code's back.

So this is ours, and being ours is worth more than the convenience would have
been:

* **It is visible.** A compaction that the CLI performs in-band rewrites what
  the model remembers while the transcript on screen still shows every turn.
  Nothing says so, and the divergence is invisible until the model fails to
  remember something the user can see. A compaction performed here writes a
  record, and the chat window draws it in the transcript where it happened.
* **It is metered.** The summary is a model call and it is charged to
  `quota_log.STAGE_COMPACT`, so "what does compacting cost" is a question the
  ledger answers. ml-intern's context manager carries a comment saying that not
  doing this "used to hide a significant share of hosted inference spend"; the
  same hole was available here for free.
* **It happens where we choose.** 967k is a wall at the end of a runway. By the
  time it is reached, every tool round-trip has spent a long time re-reading
  most of a million cached tokens.

**Compacting is not obviously cheap.** The summary costs a turn, and the session
it seeds starts with a cold prompt cache -- so the turn after a compaction pays
cache *writes* at 1.25x where it would have paid cache *reads* at 0.1x. There is
a threshold below which compacting costs more than not compacting, this module
cannot know where it is, and `[quota]` weights plus the `compaction` stage are
what make it measurable. Do not lower `compact_at_tokens` on the theory that
less context is always cheaper.

The mechanism is the one thing here with no clever part: ask the session, while
it still remembers everything, to write a note to whoever picks it up next; drop
the client; start a fresh conversation; hand that note to it as the first thing
it reads. The note is written in the first person and asks for specifics,
because the failure mode of a summary is that it reads well and contains
nothing actionable.
"""

from __future__ import annotations

from typing import Any

from core import quota_log

#: What the outgoing session is asked to leave behind.
#:
#: First person, and explicitly not a précis. A compaction summary is read by a
#: model that has to *continue* the work, not by a person deciding whether to,
#: and the two want opposite things: the reader of a précis wants the shape of
#: what happened, while the continuer wants the paths, the ids and the half
#: -finished intention. The instruction to be specific rather than brief is
#: doing the load-bearing work here.
#:
#: The Grad-specific paragraph is the reason this is not ml-intern's prompt
#: verbatim. An expectation that was registered and not yet judged, a run that
#: was submitted and not yet collected, a project that is selected: these are
#: pieces of state that live in the ledger, that the next turn is expected to
#: act on, and that a general "summarise the conversation" prompt drops on the
#: floor every time. Losing them does not read as a bad summary -- it reads as
#: an agent that abandoned a run halfway.
HANDOFF_PROMPT = """\
You are about to be restored into a fresh session that has no memory of the \
conversation above. Write a first-person note to your future self so you can \
carry on exactly where you left off. This note is the only thing you will have.

Cover, specifically and with real values rather than descriptions:

  * What was asked for, and what has actually been done about it so far.
  * Every file you wrote or changed, by path.
  * The commands you ran that mattered, and what each one returned.
  * Decisions you made and *why* -- especially the ones you would otherwise \
have to make again.
  * What you were about to do next.

Then, separately, the ledger state you were holding in your head: the project \
selected, any expectation registered and not yet judged, any run submitted and \
not yet collected, any deviation awaiting a verdict, and anything a gate has \
already refused and why. If there is none of this, say so in one line.

Do not be brief and do not be graceful. Be specific. Anything you leave out is \
gone.
"""


def threshold(cfg: Any = None) -> int:
    """Where Grad compacts, in tokens of context. 0 disables it.

    Read through the same tolerant path as the quota weights, and for the same
    reason: this is consulted on the turn path, and a typo in `grad.toml` should
    degrade to "do not compact" rather than take a session down. A negative or
    non-finite value is treated as 0 -- disabled -- because the alternative is a
    threshold that is always already exceeded, which would compact after every
    single turn.
    """
    value = _number(cfg, "compact_at_tokens", 0)
    return int(value) if value > 0 else 0


def keep_turns(cfg: Any = None) -> int:
    """How many recent turns survive verbatim under the summary.

    Clamped to at least 0 and at most 10. The upper bound is not fussiness: the
    turns kept are kept in full, and this agent's turns carry tool output, so a
    generous number here is a compaction that does not compact.
    """
    return max(0, min(10, int(_number(cfg, "compact_keep_turns", 2))))


def _number(cfg: Any, key: str, default: float) -> float:
    if cfg is None:
        try:
            from core import config as config_mod  # noqa: PLC0415

            cfg = config_mod.load()
        except Exception:  # noqa: BLE001 - see `threshold`
            return default
    try:
        value = float(cfg.get("agent", key, default))
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return value


def context_tokens(usage: Any) -> int:
    """The context size out of a `get_context_usage` reading, or 0.

    0 for an unreadable reading rather than a raise, and the caller treats 0 as
    "do not compact" -- a missing measurement is not evidence of a large
    context, and compacting on the strength of one would throw away a
    conversation for no reason.
    """
    if not isinstance(usage, dict):
        return 0
    try:
        return max(0, int(usage.get("totalTokens") or 0))
    except (TypeError, ValueError):
        return 0


def should_compact(usage: Any, cfg: Any = None) -> bool:
    """Is this conversation over the threshold? Pure, so it is testable."""
    limit = threshold(cfg)
    return bool(limit) and context_tokens(usage) > limit


def seed_message(note: str, *, tokens_before: int = 0) -> str:
    """The first thing the fresh conversation reads.

    Framed as a handover rather than presented as though the model wrote it,
    which is the honest description and also the useful one: a model told that
    this is a reconstruction knows to distrust it where it is thin, and knows it
    may have to re-read a file rather than assume it remembers the contents.

    The token figure is included because it is the one piece of context about
    the compaction that the note itself cannot contain.
    """
    note = (note or "").strip()
    if not note:
        # A summary that came back empty is not a summary. Saying so beats
        # seeding a session with a blank handover, which would look to the model
        # like a conversation that genuinely had nothing in it.
        return (
            "[Grad compacted this conversation to keep it inside its context "
            "budget, and the summary came back empty. Assume you have lost the "
            "earlier turns entirely: re-read anything you need from disk and "
            "check the ledger for open expectations and uncollected runs before "
            "continuing.]"
        )
    size = f" (it had reached {tokens_before:,} tokens)" if tokens_before else ""
    return (
        f"[This session was compacted to keep it inside its context budget{size}. "
        "The earlier turns are gone; what follows is the note the previous "
        "session left for you. Treat it as a reconstruction rather than as a "
        "record -- where it is thin, re-read the file or re-run the query "
        "instead of assuming.]\n\n"
        f"{note}"
    )


async def write_handoff(client: Any, drive: Any, *, session: str | None = None) -> dict[str, Any]:
    """Ask the live session for its handoff note, and meter the asking.

    `drive` is `agent.drive_turn`, passed in rather than imported: `agent` is a
    top-level module that imports `core`, and reaching back the other way would
    make the dependency circular for the sake of one call.

    This runs while the outgoing client is still connected, which is what makes
    it cheap: the whole conversation is already in that session's prompt cache,
    so the summary is one more read of context that has been read many times
    already, rather than a fresh upload of the transcript.

    Returns the note and what it cost. Raises nothing of its own -- a caller
    that cannot get a summary should carry on with the conversation it has, and
    the decision about whether that is acceptable is not this function's.
    """
    from agent import TurnStream  # noqa: PLC0415 - see the docstring

    stream = TurnStream()
    result = await drive(
        client,
        HANDOFF_PROMPT,
        stream,
        session=session,
        stage=quota_log.STAGE_COMPACT,
        role="compaction",
    )
    return {
        "note": stream.text,
        "quota": (result or {}).get("quota"),
        "sdk_session_id": (result or {}).get("sdk_session_id"),
    }


def record(*, tokens_before: int, tokens_after: int, note: str, cost: Any = None) -> dict[str, Any]:
    """The transcript entry a compaction leaves behind.

    A record rather than a silent replacement, because the alternative -- what
    the CLI does on its own -- is a transcript that still shows twenty turns
    beside a model that remembers one. The user is looking at the evidence for a
    belief the agent no longer holds, and nothing on screen distinguishes that
    from the agent having forgotten something it should not have.

    `role` is `system`: `ui/app.py:restore` keeps records whose role it knows and
    drops the rest, so this has to be one the chat window will draw.
    """
    counts = quota_log.counts(cost or {})
    return {
        "role": "system",
        "kind": "compaction",
        "text": (
            f"**Compacted.** The conversation reached {tokens_before:,} tokens and was "
            "summarised into a handover note; the turns above are still here to read, "
            "but the agent's memory of them is now that note."
        ),
        "note": note,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "cost_tokens": counts,
    }
