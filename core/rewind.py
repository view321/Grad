"""Undoing a turn: what a rewind drops, what it keeps, and what it leaves behind.

A turn can end with nothing worth keeping. A 500 from the API is the case that
prompted this -- the prompt goes, an error comes back, and the composer is
usable again with the failure sitting in the transcript. Say "continue" and it
happens again, and now there are three dead exchanges the model will re-read on
every turn for the rest of the session. Nothing was wrong with the question; the
only way to ask it cleanly was to start a new session and lose the thread.

**A rewind is three operations and they fail independently.** Dropping records
from the transcript is ours and always works. Putting the *model's* memory back
is the SDK's: it needs an anchor -- the uuid of the last transcript entry of the
last turn being kept -- handed to `resume_session_at` when the client is
rebuilt. When there is no usable anchor the transcript rewinds anyway and the
agent goes on remembering the dropped turns.

Putting the *work* back is the SDK's too, by a different route: with
`enable_file_checkpointing` on, `client.rewind_files(uuid)` restores files to
their state at a given user message. It is a control request rather than an
option, so it is the one half that has to happen while the client is still
alive -- `ui/app.py:rewind_to` does it before `close()`, and doing it after
would have made it the half that silently never ran.

That third one is narrower than the other two and the wording everywhere is
careful about it. The CLI checkpoints around its own editing tools, so a rewind
returns what `Write` and `Edit` changed; a file a `Bash` command wrote stays
written. For this agent that is most of them, and it is the right boundary
rather than a shortfall: the ledger is append-only precisely so a run cannot be
un-recorded, and an undo that reached into `ledger/runs.jsonl` would be erasing
evidence rather than work.

That degradation is deliberate, and it is reported rather than hidden, because
it is the same split `ui/sessions.py` draws between resuming a conversation and
redisplaying a transcript: two promises, and the one nobody can see is the one
that matters. A rewind that silently only cleaned the screen would leave the
user believing the model had forgotten something it is still being charged for.

**An anchor belongs to exactly one SDK session.** A uuid names an entry inside
the conversation that issued it, so a uuid stamped by a different session is not
an anchor here -- it names an entry the resumed conversation does not contain.
That is why turns record which session named them and `anchor_in` checks, rather
than reaching for the most recent uuid it can find: the SDK is free to hand a
resumed conversation a new id (see `agent.drive_turn`), and reusing an id across
that boundary is how a rewind would appear to work and do nothing.

**Nothing leaves the file.** The dropped records are carried inside the marker
this leaves in their place, the way `core/compaction.py:record` carries the
handover note for the turns it discarded. A rewind is reached for when a session
has gone wrong, which is exactly when being able to read what happened is worth
most -- and the transcript is where the reasoning behind an expectation lives.
The point is to take the dead turns out of the *conversation*, not out of the
record.
"""

from __future__ import annotations

from typing import Any

#: The `kind` on the record a rewind leaves behind. Its `role` is `system` for
#: the reason `core/compaction.py` gives: `ui/app.py:restore` keeps records whose
#: role it knows and drops the rest, so a marker has to be one of them.
MARK_KIND = "rewind"


def points(settled: list[dict[str, Any]]) -> list[int]:
    """Where a rewind may start: the things the user said.

    A rewind means "drop this prompt and everything after it", so it anchors on
    a user message. Anchoring on an answer would leave a question standing with
    its reply gone, which reads as an agent that ignored it.
    """
    return [i for i, record in enumerate(settled) if record.get("role") == "user"]


def anchor_in(records: list[dict[str, Any]], sdk_session_id: str | None) -> str | None:
    """The SDK entry to resume at, out of the turns being kept.

    Walked backwards: `resume_session_at` wants the *last* transcript entry of
    the turn being kept, and the most recent turn that recorded one is it.
    Records without a uuid are stepped over rather than treated as the end of the
    search -- a user's own message never has one, and neither does a turn that
    died before the SDK named anything.

    `None` is a complete answer and not an error: the caller rewinds the
    transcript and says the memory did not move with it.
    """
    if not sdk_session_id:
        return None
    for record in reversed(records):
        uuid = record.get("uuid")
        if not (isinstance(uuid, str) and uuid):
            continue
        # Same session, or it is not an anchor. See the module docstring.
        if record.get("sdk_session_id") == sdk_session_id:
            return uuid
    return None


def plan(
    settled: list[dict[str, Any]],
    index: int,
    *,
    sdk_session_id: str | None = None,
) -> dict[str, Any]:
    """What rewinding to `index` would do, worked out before anything changes.

    Pure, and separate from the doing, so the two questions a rewind raises --
    "is this a thing I can rewind to" and "will the agent's memory come with it"
    -- are answerable in a test without an SDK, a client, or a browser.

    `index` reaches this out of a click handler on a transcript that may have
    been rebuilt underneath it, so it is validated rather than trusted.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        return {"ok": False, "reason": f"not a message index: {index!r}"}
    if not 0 <= index < len(settled):
        return {"ok": False, "reason": "that message is no longer in this session"}

    target = settled[index]
    if target.get("role") != "user":
        return {"ok": False, "reason": "a rewind starts at something you asked"}

    keep = list(settled[:index])
    dropped = list(settled[index:])
    anchor = anchor_in(keep, sdk_session_id)
    return {
        "ok": True,
        "keep": keep,
        "dropped": dropped,
        #: Handed back to the composer so the prompt can be edited and re-sent
        #: rather than retyped. Restored into the box rather than sent again on
        #: its own: a rewind is most often reached for because the prompt itself
        #: wanted changing, and re-issuing it unasked would spend a turn to
        #: reproduce the answer that was just discarded.
        "prompt": str(target.get("text") or ""),
        "anchor": anchor,
        #: How many exchanges go. Counted here because the marker says it and
        #: because `resume_drops_turn` only applies when it is exactly one.
        "turns": sum(1 for record in dropped if record.get("role") == "user"),
    }


def record(
    *,
    dropped: list[dict[str, Any]],
    resumed: bool,
    anchor: str | None = None,
    files: bool = False,
) -> dict[str, Any]:
    """The transcript entry a rewind leaves in place of what it dropped.

    A record rather than a silent truncation, for the reason a compaction leaves
    one: a transcript that quietly loses turns is indistinguishable from one that
    never had them, and the next confusing answer has nothing to point at. This
    also makes a rewind honest about the halves of itself that can fail -- the
    marker says whether the agent's memory came back with the screen, and whether
    the work did.

    `files` is stated only when it is true. The common rewind restores none --
    most turns edit nothing -- and a marker reading "no files were restored"
    describes a failure of something that was never attempted. What it claims is
    also deliberately narrow: the CLI checkpoints around its own editing tools,
    so this is `Write` and `Edit`, not what a `Bash` command wrote and not
    anything a submitter did on a backend. See `agent.checkpointing_option`.

    `dropped` is carried whole, blocks and tool calls included, so the file keeps
    everything the conversation no longer does.
    """
    turns = sum(1 for entry in dropped if entry.get("role") == "user")
    count = "one exchange" if turns == 1 else f"{turns} exchanges"
    if resumed:
        tail = (
            "the agent's memory of them was put back to this point as well, so the "
            "next turn starts as though they had not happened."
        )
    else:
        tail = (
            "**only the transcript moved** — there was no live conversation to put "
            "back, so the agent still remembers them and the next turn still pays "
            "for them."
        )
    if files:
        tail += (
            " Files the agent edited were restored to their state before the first "
            "dropped prompt; anything a command wrote was not."
        )
    return {
        "role": "system",
        "kind": MARK_KIND,
        "text": f"**Rewound.** {count} was dropped from here; {tail}",
        # Kept, not discarded. See the module docstring.
        "dropped": list(dropped),
        # Only for reading a session file after the fact -- nothing loads it
        # back. It is what says *which* conversation the rewind was against when
        # a transcript has been through several.
        "anchor": anchor,
        #: Whether the work moved with the conversation. Third of the three
        #: claims a rewind makes, and the newest.
        "files": bool(files),
    }


def pending_anchor(settled: list[dict[str, Any]]) -> str | None:
    """The rewind this transcript is still waiting to have applied to it.

    A rewind arms the *next* client, because `resume_session_at` is fixed when
    one is built -- so between the click and the turn after it there is a window
    where the transcript has moved and the conversation has not. Closing the app
    in that window used to lose the half that had not happened yet: the file came
    back short, the SDK session resumed whole, and the marker sat there claiming
    the agent had forgotten turns it was still being charged for.

    Nothing extra is stored to fix that. A trailing marker *is* the state -- it
    means a rewind happened and no turn has run since, because a turn that ran
    would have appended after it -- and the anchor it needs is already written on
    it. Once there is a turn below it the transcript is what it says it is, and
    this correctly finds nothing.
    """
    if not settled:
        return None
    last = settled[-1]
    if last.get("kind") != MARK_KIND:
        return None
    anchor = last.get("anchor")
    return anchor if isinstance(anchor, str) and anchor else None


def dropped_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The turns inside a marker, defensively.

    This file outlives the version that wrote it and the chat window renders
    whatever comes back, so a `dropped` that is not a list of records is treated
    as an empty one rather than raised on at build time -- the same contract
    `ui/app.py:_drawable_blocks` has with `blocks`.
    """
    value = record.get("dropped")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict) and entry.get("role")]
