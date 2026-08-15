"""Named chat sessions: starting a new one, and coming back to an old one.

There was one conversation per client key, in one file, forever. Everything the
agent had ever been asked was in it, and the only way to start clean was to
delete the file -- which took the record with it. That is the wrong trade for
this project in particular: a session is where the reasoning behind an
expectation lives, and the ledger entry it produced points back at nothing.

**A session is a file, and the file is the record.** No index, no database. The
id is the filename, so listing is a glob and nothing can disagree with anything.
A session's first line is a `meta` record and the rest are transcript records;
`app.Session.restore` already ignores any line whose `role` is not a real role,
so the meta line costs nothing there and the format stays one thing.

**The legacy transcript is already a session.** It was `ui_session-default.jsonl`
and the scheme here is `ui_session-<id>.jsonl`, so the file that exists on an
upgraded machine is a session called `default` with no migration step at all.
Its title is derived from its first user message, which is what a session that
was never named should be called anyway.

**Two ids, and they are not interchangeable.** Ours names the file. The SDK's --
recorded here when a turn reports one -- is what `ClaudeAgentOptions.resume`
takes, and it is what makes reopening a session continue the *conversation*
rather than merely redisplay it. A session whose SDK id is unknown (an older
transcript, a session whose first turn failed) still opens: the transcript is
shown and the next turn starts a fresh conversation under the same file. That
degradation is deliberate and is reported, because "the agent remembers this"
and "you can read this" are different promises.
"""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Any

from core import appdata
from core.ledger_store import now_iso

PREFIX = "ui_session"
#: The id of the conversation that existed before sessions did.
LEGACY_ID = "default"
#: What an id may contain. Ids reach a filename, and one of them comes from a
#: file already on disk rather than from `new_id`.
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
#: How much of the first user message becomes the fallback title.
TITLE_CHARS = 60


def sessions_dir() -> Path:
    """Transcripts, under the app directory but keyed to this workspace.

    Both halves matter. They are private, so they do not belong in a folder that
    gets committed; and `ui/app.py:rebind` promises that switching the workspace
    root switches which conversation is on screen, which one flat directory
    shared by every folder ever opened would quietly break. See
    `core/appdata.py:workspace_state_dir`.
    """
    d = appdata.workspace_state_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_id() -> str:
    """Sortable, so a glob comes back in a sensible order before anything reads
    an mtime, plus four hex digits because two sessions in one second is a
    double-click rather than an impossibility."""
    stamp = re.sub(r"[^0-9]", "", now_iso())[:14]
    return f"{stamp}-{secrets.token_hex(2)}"


def is_id(value: Any) -> bool:
    return isinstance(value, str) and bool(ID_RE.fullmatch(value))


def path_for(session_id: str) -> Path:
    """The file behind an id, refusing anything that is not one.

    The id reaches a filename and can come off disk or out of a click handler,
    so it is validated here rather than trusted -- the same reason
    `state.layout_path` sanitises a project id.
    """
    if not is_id(session_id):
        raise ValueError(f"not a session id: {session_id!r}")
    return sessions_dir() / f"{PREFIX}-{session_id}.jsonl"


def id_of(path: Path) -> str:
    return path.name[len(PREFIX) + 1 : -len(".jsonl")]


def read_meta(path: Path) -> dict[str, Any]:
    """The header record, the title, and how many messages are in the file.

    Stops at the header when the header carries everything -- which it does for
    any session this version wrote, because `write` records the count alongside
    the title. That matters because `listing()` calls this once per session: a
    scan-to-the-end here would make listing cost the total size of every
    transcript in the workspace, and transcripts grow with the conversations
    that are most worth coming back to.

    The full scan is still the fallback, for the one case that needs it: a file
    written before this format existed has no header, so its title is its first
    user message and its count is its lines. That is one file, once.
    """
    meta: dict[str, Any] = {}
    title = ""
    messages = 0
    if not path.exists():
        return meta
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("type") == "meta":
                    meta = {k: v for k, v in record.items() if k != "type"}
                    if meta.get("title") and isinstance(meta.get("messages"), int):
                        return meta
                    continue
                messages += 1
                if not title and record.get("role") == "user":
                    title = str(record.get("text") or "").strip()
    except OSError:
        return meta
    meta.setdefault("title", "")
    if not meta["title"]:
        meta["title"] = title_from(title) or "empty session"
    meta["messages"] = messages
    return meta


def title_from(text: str) -> str:
    """A session's name, taken from the first thing asked in it."""
    flattened = " ".join(text.split())
    if len(flattened) <= TITLE_CHARS:
        return flattened
    return flattened[: TITLE_CHARS - 1] + "…"


def listing() -> list[dict[str, Any]]:
    """Every session, most recently written first.

    Sorted on mtime rather than on the id, because `default` predates the
    timestamped scheme and because resuming a session is the thing that makes it
    recent. A file that cannot be read is skipped rather than raised on -- one
    damaged transcript must not make the picker unopenable.
    """
    out: list[dict[str, Any]] = []
    directory = sessions_dir()
    if not directory.exists():
        return out
    for path in directory.glob(f"{PREFIX}-*.jsonl"):
        session_id = id_of(path)
        if not is_id(session_id):
            continue
        meta = read_meta(path)
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        out.append(
            {
                "id": session_id,
                "title": meta.get("title") or "empty session",
                "created_at": meta.get("created_at"),
                "sdk_session_id": meta.get("sdk_session_id"),
                # Whether reopening continues the conversation or only shows it.
                "resumable": bool(meta.get("sdk_session_id")),
                "messages": int(meta.get("messages") or 0),
                "modified": modified,
            }
        )
    out.sort(key=lambda s: s["modified"], reverse=True)
    return out


def delete(session_id: str) -> bool:
    path = path_for(session_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def write(
    session_id: str,
    records: list[dict[str, Any]],
    *,
    title: str = "",
    created_at: str | None = None,
    sdk_session_id: str | None = None,
) -> None:
    """The whole session: one meta line, then the transcript.

    Rewritten wholesale rather than appended to, because that is what the
    transcript already did and because a session is small. It is *not* routed
    through `core/jsonl.py`: that module's contract is the append-only ledgers,
    which are multi-writer and must never lose a line. A transcript has exactly
    one writer -- the client that owns the session -- and is replaced in full.
    """
    path = path_for(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": "meta",
        "id": session_id,
        "title": title,
        "created_at": created_at or now_iso(),
        "sdk_session_id": sdk_session_id,
        # Written here so `read_meta` can answer from the header alone. The
        # count is free at this point -- the records are in hand -- and it is
        # what keeps listing a workspace from costing a read of every transcript
        # in it.
        "messages": len(records),
    }
    lines = [json.dumps(meta, ensure_ascii=False)]
    lines += [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# who has which session open
# ---------------------------------------------------------------------------
#: Sessions held by a live client, in this process, keyed by **path**. A session
#: is written wholesale by the client that owns it, so two clients on one
#: session is two writers on one file and the later `_persist` silently discards
#: the other's turns. `app.Session` was always one-per-client for exactly this
#: reason; the claim keeps that true now that the *file* is chosen rather than
#: derived from the client's own key.
#:
#: By path rather than by id because an id is only unique within a workspace:
#: every root has a `default`, and one client switching roots would otherwise
#: find another client's claim on a file it has never seen.
_claimed: dict[str, str] = {}

#: Owners with a live connection. A claim held by an owner that is not in here
#: is stale -- its client is gone -- and may be taken over.
#:
#: This exists because the owner key used to be the *browser* id, which is one
#: cookie shared by every tab: two tabs of one browser claimed the same session
#: successfully (`held == owner`), which is precisely the two-writers case the
#: claim was built to prevent. Per-connection owners fix that and break reload,
#: where the new page connects before the old one's disconnect fires and finds
#: its own session held by a client that no longer exists. Liveness is what
#: separates "another window is in this" from "the window that was in this is
#: gone", and those need different answers.
_live: set[str] = set()


def _key(session_id: str) -> str:
    return str(path_for(session_id))


def register(owner: str) -> None:
    _live.add(owner)


def claim(session_id: str, owner: str) -> bool:
    """Take a session for one client. False if another *live* client has it."""
    key = _key(session_id)
    held = _claimed.get(key)
    if held is not None and held != owner and held in _live:
        return False
    _claimed[key] = owner
    _live.add(owner)
    return True


def release(owner: str) -> None:
    # Only the claims this owner still holds. A reloaded page re-claims its
    # session under a new owner, and the old client's late disconnect must not
    # then drop the live page's claim -- which is what happened when release
    # matched on owner alone and the two shared a browser-id key.
    for key in [k for k, held in _claimed.items() if held == owner]:
        del _claimed[key]
    _live.discard(owner)


def held_by_other(session_id: str, owner: str) -> bool:
    """Does a *different, live* client hold this session?

    The question `_persist` needs, and it is not "do I hold it": an unclaimed
    session is safe to write -- there is no one to overwrite -- while one held
    by another live client is exactly the file that must not be replaced. A
    claim left behind by a client that has gone is not a writer either.
    """
    held = _claimed.get(_key(session_id))
    return held is not None and held != owner and held in _live


def holder(session_id: str) -> str | None:
    return _claimed.get(_key(session_id))


def reset_claims() -> None:
    """Drop every claim. For tests -- module state outlives a fixture."""
    _claimed.clear()
    _live.clear()


def most_recent(owner: str | None = None) -> str | None:
    """The session to open on a cold start, or None for a new workspace.

    With an `owner`, the most recent one *nobody else is in*. A second window
    opening the first one's conversation would not merely show it twice: both
    would write the whole file on every turn, and the loser's turns would
    disappear with nothing to say they had.
    """
    for row in listing():
        if owner is None or claim(row["id"], owner):
            return row["id"]
    return None
