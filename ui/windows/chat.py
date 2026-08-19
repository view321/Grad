"""Window 1 — the agent session.

The two implementation details that are the difference between this feeling
like a tool and feeling like a demo survive the redesign unchanged, because
they were never about the visuals:

  * **Buffered flush.** Updating a `ui.markdown` per token re-renders and
    reflows the whole element on every token. Tokens accumulate in the turn's
    blocks and a `ui.timer` flushes at ~15 Hz.
  * **Split tail.** The streaming turn lives in its own element, separate from
    the settled transcript above it, so only the tail re-renders -- and inside
    the tail, only the block that moved. It is promoted into the transcript (and
    KaTeX runs over it) once complete.

What the redesign adds is anatomy: a turn is parsed into prose, tool calls,
expectation cards and gate cards, each with its own shape, rather than being one
markdown blob. `ui/models.py:parse_message` does the parsing, so what counts as
a gate is testable without a browser.

The tool cards are the agent's *actual* calls, read off the SDK stream by
`agent.TurnStream` rather than inferred from what the agent said about them. It
is the same window either way, which is the point: a command the agent ran and a
command it claimed to run should not look alike, and before this only the second
one could ever appear.

**The statusline is the fourth thing, and it is a control as well as a report.**
It sits above the composer and is always there: what the agent is doing, which
call is in flight, how long the turn has been going. The strip it replaced
appeared only while a turn ran and said "running …", so the state worth reading
at a glance was the one that came and went. Clicking it shows or hides the
agent's reasoning, which is a fifth kind of block and is drawn whether or not it
is switched on -- a class on the chat root decides whether it is painted, so the
toggle costs no re-render and takes no scroll position with it.

This window is never redrawn by the poll. Its state is the live session, not a
file, and a redraw would take the transcript's scroll position with it.
"""

from __future__ import annotations

import asyncio
import html
import time
from typing import Any

from core import rewind
from ui import katex, kit, models
from ui.state import FLUSH_HZ

#: A call's status -> the one accent it is drawn in. Dashed while it runs, for
#: the same reason the design uses a dashed border everywhere else it means
#: "pending": an outcome that has not happened yet is not a green one.
STATUS_TONE = {"running": "dashed", "ok": "ok", "error": "broken"}

#: Seconds between context readings. Four is slow enough that the call is
#: invisible next to a turn and fast enough that the chip is never far behind
#: what is actually in the window.
CONTEXT_POLL_S = 4.0


def compact_threshold() -> int:
    """Where Grad will compact, for the meter to measure against.

    Here rather than read from `core.compaction` at the call site so the import
    stays lazy: `ui/windows/chat.py` is imported by the registry on a machine
    that may have no config yet, and `config.load()` is cached, so the cost of
    asking on every poll is a dict lookup.
    """
    from core import compaction  # noqa: PLC0415

    try:
        return compaction.threshold()
    except Exception:  # noqa: BLE001 - a meter must not be able to take the window down
        return 0


def subtitle(workspace: Any) -> str:
    session = getattr(workspace, "session", None)
    settled = len(getattr(session, "settled", []) or [])
    title = getattr(session, "title", "") or "new session"
    return f"{title} · {settled} messages · {workspace.agent_state}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    if workspace.agent_state == "awaiting_gate":
        return [("GATE", "broken")]
    if getattr(workspace.session, "busy", False):
        return [("STREAMING", "ok")]
    return []


def _sessions(ui: Any, workspace: Any) -> None:
    """The session bar: which conversation this is, and how to leave it.

    There was one conversation per client, in one file, forever -- so the only
    way to start clean was to delete the record, and in this project the record
    is where the reasoning behind an expectation lives.

    Reopening one is a *resume* where the SDK id is known and a redisplay where
    it is not, and the row says which. Those are different promises: a session
    that only redisplays shows the transcript above a composer whose next turn
    the agent has no memory of, and a picker that presented both the same way
    would be lying about the more important half.

    This was a Quasar `select` -- the one control left in the workspace still
    wearing NiceGUI's own look, in an app whose stylesheet bypasses Quasar
    rather than overriding it (see the note at the top of `ui/tokens.py`). It was
    not only out of place: a `select` has one string per option, so the two
    things a session row has to say arrived as ` · ` fragments glued onto the
    title, and the row that *cannot* be opened looked exactly like the rows that
    can. It is now the same menu the windows and the workspace use.
    """
    session = workspace.session
    menu = kit.menu(lambda body, m: _draw_session_menu(workspace, body, m), width=520)

    with kit.row("grad-pad", gap=6).style("border-bottom: var(--grad-border); flex: 0 0 auto"):
        kit.button(
            "+ NEW",
            tone="primary",
            title="start a clean conversation, keeping this one",
            on_click=lambda: workspace.spawn(_fresh(workspace), "new session"),
        )
        model = workspace.sessions()
        title = getattr(session, "title", "") or "new session"
        kit.button(
            f"{title}  ▾",
            tone="ghost",
            classes="grad-session-btn",
            title="sessions in this workspace, most recent first",
            on_click=menu.open,
        )
        kit.spacer()
        kit.text(f"{model['count']} stored", "grad-caption", tag="span")
        if not getattr(session, "sdk_session_id", None) and getattr(session, "settled", []):
            # Said once, where the decision is made, rather than left to be
            # discovered when the agent answers as though nothing was discussed.
            kit.chip("TRANSCRIPT ONLY", "attention")


async def _fresh(workspace: Any) -> None:
    workspace.say(await workspace.session.new_session())
    _left_for_another_session(workspace)


async def _switch(workspace: Any, session_id: str) -> None:
    workspace.say(await workspace.session.open_session(session_id))
    _left_for_another_session(workspace)


def _left_for_another_session(workspace: Any) -> None:
    """Redraw for a conversation that is not the one just left.

    The draft goes with it, and that is the whole reason this is a function
    rather than two calls to `rebuild_chat`. `chat_draft` is workspace state
    rather than window state -- it has to be, because its job is to survive the
    rebuild a rewind triggers -- and the composer is seeded from it on every
    draw. So a rewound prompt that was never sent stayed in it across a session
    switch and reappeared in the next conversation's box, where it reads as
    something typed there and is one Enter away from being asked of the wrong
    agent. Cleared here because this is where "a different conversation" is
    decided; `new_session` and `open_session` belong to the session and cannot
    see the workspace holding the draft.
    """
    workspace.chat_draft = ""
    workspace.rebuild_chat()


async def _rewind(workspace: Any, index: int) -> None:
    """Take the conversation back to before one prompt, and offer it again.

    The whole transcript is rebuilt rather than surgically edited. A rewind
    changes the index of nothing above the cut and removes everything below it,
    so the click handlers bound to the removed elements are exactly what must not
    survive -- and `rebuild_chat` is already the tested path for replacing a
    transcript wholesale, which is what a session switch does.

    The scroll position goes with it, which is a real cost the session switch
    pays for the same reason. Here it lands where it should anyway: the rewind
    point is the new end of the conversation.
    """
    outcome = await workspace.session.rewind_to(index)
    workspace.say(outcome.get("message"))
    if not outcome.get("ok"):
        return
    # Back in the box rather than re-sent. See `Session.rewind_to`.
    workspace.chat_draft = outcome.get("prompt") or ""
    # A gate that was awaiting a decision may have been part of what was just
    # dropped, and a titlebar left claiming GATE would be asking about a turn
    # that no longer exists.
    workspace.set_agent_state("idle")
    workspace.rebuild_chat()


def _draw_session_menu(workspace: Any, body: Any, menu: Any) -> None:
    """Every stored conversation, and what opening it would actually do."""
    model = workspace.sessions()
    current = model["current"]
    body.clear()

    with body:
        with kit.row("head ink", gap=9):
            kit.text("SESSIONS", "", tag="span")
            kit.spacer()
            kit.text(f"{model['count']} in this workspace", "", tag="span")

        with kit.el("div", "body"):
            kit.error_strip(model.get("error"))
            for row in model["rows"]:
                is_current = row["id"] == current
                held = row["held_elsewhere"]
                hint = f"{row['messages']} msg"
                if held:
                    hint += " · open in another window"
                elif not row["resumable"]:
                    # The more important half of the promise, and the half a
                    # one-string picker could not distinguish from decoration.
                    hint += " · transcript only"
                menu_row = kit.menu_row(
                    "■" if is_current else "□",
                    row["title"],
                    hint,
                    open=is_current,
                    wide=True,
                    disabled=held or is_current,
                    title=(
                        "another window has this one open"
                        if held
                        else "the conversation continues where it left off"
                        if row["resumable"]
                        else "the transcript opens; the agent has no memory of it"
                    ),
                )
                if not (held or is_current):
                    menu_row.on(
                        "click",
                        lambda _=None, sid=row["id"]: (
                            menu.close(),
                            workspace.spawn(_switch(workspace, sid), "session switch"),
                        ),
                    )

            if model["transcript_only"]:
                kit.text(
                    f"{model['transcript_only']} of these predate the id `resume` takes — "
                    "they open as a transcript, and the next turn starts fresh",
                    "grad-caption",
                ).style("margin-top: 12px")


def render(workspace: Any) -> None:
    from nicegui import ui

    session = workspace.session

    with kit.column(_chat_classes(workspace), gap=0).classes("h-full") as root:
        _sessions(ui, workspace)
        # One scrolling region, not two. The tail used to be a sibling *below*
        # the scroller, which was survivable when it held a single markdown
        # element and is not now that it holds cards: three tool calls and the
        # turn in flight grows past the pane, scrolls `.grad-body` instead, and
        # paints over the composer. So the tail lives inside the scroller, as
        # its last child, and the whole conversation scrolls as one thing.
        scroller = kit.column("grad-transcript", gap=0)
        scroller.props('id="grad-transcript"')
        scroller.style("flex: 1 1 auto; overflow-y: auto; min-height: 0")
        with scroller:
            # `flex: 0 0 auto` on both, and it is not decoration. The scroller is
            # a *flex column*, so these two are flex items and arrive with
            # `flex-shrink: 1`; `kit.column` then sets `min-height: 0`, which is
            # what a nested scroll container needs and which also removes the
            # automatic minimum size that otherwise stops a flex item shrinking
            # below its own content.
            #
            # So once the conversation grew past the pane, the browser did not
            # scroll it -- it shrank these two boxes to fit and let their
            # messages spill out, one turn painted over the next, with no
            # scrollbar to say anything had overflowed. It showed up "sometimes
            # while moving a tile around" because that is what changes the pane's
            # height: the same transcript overlaps at one pane size and is fine
            # at another. Refusing to shrink makes the overflow the scroller's,
            # which is the one element here equipped to have any.
            transcript = kit.column("", gap=0)
            transcript.style("flex: 0 0 auto")
            tail_root = kit.column("", gap=0)
            tail_root.props('id="grad-tail"')
            tail_root.style("flex: 0 0 auto")
        tail = _Tail(tail_root)

        with transcript:
            for index, message in enumerate(session.settled):
                # The index is the position in `settled`, which is what a rewind
                # cuts at. Passed rather than recomputed at the click because the
                # click handler has no way back to the list it was drawn from --
                # and `rewind.plan` re-validates it anyway, since a transcript
                # can be rebuilt underneath a handler that is still bound.
                _message(message, workspace, index=index)

        statusline = _Statusline(workspace, root)
        _composer(ui, workspace, transcript, tail, statusline)

    def flush() -> None:
        # ~15 Hz, not per token: only the tail re-renders, and inside it only
        # the block that actually moved.
        blocks = getattr(session, "blocks", None) or []
        tail.sync(blocks)
        statusline.sync(blocks)

    ui.timer(1 / FLUSH_HZ, flush)

    async def poll_context() -> None:
        """Ask the CLI how big the context is, and redraw the chip.

        On its own timer, several hundred times slower than the flush. The call
        costs no tokens -- it is a control request answered from state the CLI
        already holds -- but it is still a round-trip over the transport a turn
        is streaming on, and there is nothing to learn from asking sixty times a
        second about a number that moves once a turn.

        The interval is chosen for the case that matters: a turn that runs for
        forty minutes, where the point of the meter is watching the context
        climb while there is still time to do something about it.
        """
        await session.read_context()
        statusline.sync_context()

    ui.timer(CONTEXT_POLL_S, poll_context)
    # Keep the transcript pinned to the bottom while a turn streams. Doing it
    # from the flush instead would be a `run_javascript` fifteen times a second.
    #
    # This is the *registration*, and it is no longer the only arming: the node
    # behind this id is replaced whenever the pane tree is rebuilt, so
    # `ui/shell.py:retile` re-arms through `gradRearm` afterwards. What this call
    # does that the retile cannot is name the id in the window that owns it.
    kit.run_js("window.gradStickBottom && window.gradStickBottom('grad-transcript')")


def _chat_classes(workspace: Any) -> str:
    return "grad-chat" + (" reasoning-on" if workspace.show_reasoning else "")


#: What the statusline says the agent is doing, per workspace state. `running`
#: is absent on purpose: while a turn is in flight the activity is read off the
#: blocks, which is more specific than any fixed word.
STATE_CAPTION = {
    "idle": ("IDLE", "waiting for you"),
    "awaiting_gate": ("YOUR CALL", "the turn ended by asking for a decision"),
    "paused": ("PAUSED", ""),
}


class _Statusline:
    """The agent's own status line, and the switch for its reasoning.

    Two jobs in one strip, and they belong together. The strip that was here
    before appeared only while a turn ran, said "running …" and vanished --
    so an idle session had nothing saying so, and the one piece of state worth
    reading at a glance (what is it doing *right now*) was the one that came and
    went. This is always on screen, and it names the call in flight rather than
    spinning, which is the difference between a spinner and knowing a
    forty-minute job is still going.

    Clicking it shows or hides the reasoning. A toggle rather than a second
    control because the reasoning is *about* what this line reports, and because
    the line is already the widest click target in the window. Nothing is
    redrawn: the blocks are in the DOM either way and a class on the chat root
    decides whether they are painted -- see `Workspace.toggle_reasoning`.
    """

    def __init__(self, workspace: Any, root: Any) -> None:
        self.workspace = workspace
        self.root = root
        self._started: float | None = None
        self._last: tuple[Any, ...] | None = None

        bar = kit.el("button", "grad-statusline")
        bar.props('title="what the agent is doing — click to show or hide its reasoning"')
        bar.on("click", self.toggle)
        with bar:
            self.block = kit.el("span", "block")
            self.state = kit.text("IDLE", "state", tag="span")
            self.activity = kit.text("waiting for you", "activity", tag="span")
            kit.spacer()
            # Before the clock rather than after it: the clock and the reasoning
            # switch are about the turn in flight, and this is about the session
            # as a whole -- it is the one thing on this strip that is still true
            # when nothing is running.
            self.context = kit.text("", "context", tag="span")
            self.clock = kit.text("", "clock", tag="span")
            # Its own click, and `stopPropagation` is what keeps it separate:
            # the whole strip is one button, so without it every change of
            # effort would also toggle the reasoning panel. NiceGUI has no Vue
            # modifier passthrough -- `on("click.stop")` is camel-cased into an
            # event name nothing fires -- so the stop happens in `js_handler`,
            # which then emits to the Python handler as usual.
            self.effort = kit.text("", "effort", tag="span")
            self.effort.props(
                'title="how hard the agent thinks -- click to change" '
                # Still a span, deliberately: the strip around it is itself a
                # click target, and a real <button> nested inside one is invalid
                # markup that browsers resolve by unnesting it -- which moves the
                # control out of the strip it belongs to. So the semantics are
                # spelled out instead, and the keyboard handler below is what
                # makes them true rather than merely announced.
                'role="button" tabindex="0"'
            )
            self.effort.on(
                "click",
                self.cycle_effort,
                js_handler="(e) => { e.stopPropagation(); emit(); }",
            )
            self.effort.on(
                "keydown",
                self.cycle_effort,
                # Enter and Space, which is what `role="button"` promises. The
                # same `stopPropagation` as the click, and for the same reason:
                # the strip would otherwise toggle the reasoning panel too.
                js_handler=(
                    "(e) => { if (e.key === 'Enter' || e.key === ' ') {"
                    " e.preventDefault(); e.stopPropagation(); emit(); } }"
                ),
            )
            self.reasoning = kit.text("", "reasoning", tag="span")
        self.bar = bar
        self._context_mark: tuple[Any, ...] | None = None
        self._paint_reasoning()
        self._paint_effort()
        self.sync_context()

    def cycle_effort(self) -> None:
        self.workspace.cycle_effort()
        self._paint_effort()

    def _paint_effort(self) -> None:
        from core import effort as effort_mod  # noqa: PLC0415

        level = effort_mod.current()
        kit.set_text(self.effort, effort_mod.label(level))
        # `auto` is the absence of a choice, and the chip says so by staying
        # dashed. Without the distinction the strip reads as though someone had
        # deliberately selected "auto", which is the one level nobody selects.
        if level == effort_mod.AUTO:
            self.effort.classes(remove="set")
        else:
            self.effort.classes(add="set")

    def toggle(self) -> None:
        showing = self.workspace.toggle_reasoning()
        if showing:
            self.root.classes(add="reasoning-on")
        else:
            self.root.classes(remove="reasoning-on")
        self._paint_reasoning()
        if showing and not _has_reasoning(self.workspace.session):
            # Switching on something that reveals nothing is indistinguishable
            # from a switch that does not work, and there are two ordinary
            # reasons for it: a transcript written before reasoning was captured
            # at all, and a model that was never asked for the text. Both are
            # worth saying once, at the click, rather than leaving to be
            # guessed at.
            self.workspace.say(
                "no reasoning in this session yet — it is captured from the next turn on, "
                'and needs [agent] reasoning = "summarized" in config/grad.toml'
            )


    def _paint_reasoning(self) -> None:
        showing = self.workspace.show_reasoning
        kit.set_text(self.reasoning, f"reasoning {'■ on' if showing else '□ off'}")

    def sync_context(self) -> None:
        """Redraw the context chip from the session's last reading.

        Called on its own timer rather than on the 15 Hz flush: the underlying
        number changes once per control request, and repainting it a hundred
        times between two readings is a hundred DOM writes that say the same
        thing. Like `sync`, it touches nothing unless the line changed.
        """
        session = self.workspace.session
        model = models.context_model(
            getattr(session, "context", None),
            compact_at=compact_threshold(),
        )
        mark = (model["label"], model["tone"], model["detail"])
        if mark == self._context_mark:
            return
        self._context_mark = mark
        kit.set_text(self.context, model["label"])
        # Both removed before either is added: a chip that crossed from warn to
        # attention would otherwise carry the old class as well as the new one,
        # and the pair have different accents by design.
        self.context.classes(remove="warn attention")
        if model["tone"]:
            self.context.classes(add=model["tone"])
        self.context.props(f'title="{kit.attr(model["detail"])}"')

    def sync(self, blocks: list[dict[str, Any]]) -> None:
        """Called at the flush rate, so nothing here touches the DOM unless the
        line it draws actually changed."""
        busy = bool(getattr(self.workspace.session, "busy", False))
        if not busy:
            self._started = None
        elif self._started is None:
            self._started = time.monotonic()

        state = self.workspace.agent_state
        if busy:
            caption, activity = "RUNNING", _activity(blocks)
        else:
            caption, activity = STATE_CAPTION.get(state, ("IDLE", "waiting for you"))
        clock = _elapsed(self._started)

        mark = (busy, caption, activity, clock)
        if mark == self._last:
            return
        self._last = mark
        if busy:
            self.bar.classes(add="running")
        else:
            self.bar.classes(remove="running")
        kit.set_text(self.state, caption)
        kit.set_text(self.activity, activity)
        kit.set_text(self.clock, clock)


def _has_reasoning(session: Any) -> bool:
    """Is there any reasoning in this session to show?

    Scanned at the click rather than on the flush, because this is the one
    moment the answer changes what the user is looking at -- and a scan of every
    settled turn fifteen times a second to decide the wording of one chip is the
    kind of thing `ui/state.py` exists to avoid.
    """
    live = getattr(session, "blocks", None) or []
    if any(b.get("kind") == "thinking" for b in live):
        return True
    for record in getattr(session, "settled", None) or []:
        if any(b.get("kind") == "thinking" for b in record.get("blocks") or []):
            return True
    return False


def _elapsed(started: float | None) -> str:
    if started is None:
        return ""
    seconds = max(0.0, time.monotonic() - started)
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def _activity(blocks: list[dict[str, Any]]) -> str:
    """What the statusline says while a turn runs.

    Read backwards off the blocks, because the last thing that moved is what is
    happening: a call still running is the answer whenever there is one, and
    when there is not, whether the agent is reasoning or writing is the next
    most specific thing that can honestly be said.
    """
    for block in reversed(blocks):
        if block.get("kind") == "tool" and block.get("status") == "running":
            return f"running {block.get('name') or 'tool'} {block.get('title') or ''}".strip()[:90]
    for block in reversed(blocks):
        if not (block.get("text") or "").strip():
            continue
        return "thinking" if block.get("kind") == "thinking" else "writing"
    return "waiting for the model"


class _Tail:
    """The turn in flight, drawn block by block.

    The split-tail rule holds -- the settled transcript above is never touched
    while a turn runs -- but the tail is no longer one markdown element, so
    rebuilding it at 15 Hz would re-render every card already in it. This
    appends what is new and updates in place the only two things that change: the
    open prose block's text, and a tool card's chip and output once its result
    lands.
    """

    def __init__(self, container: Any) -> None:
        self.container = container
        self._drawn: list[dict[str, Any]] = []

    def sync(self, blocks: list[dict[str, Any]]) -> None:
        # A shorter list is a new turn; a changed kind at a drawn index is a
        # stream that did not grow the way this assumes. Both are cheap to
        # detect and neither can be repaired by appending.
        if len(blocks) < len(self._drawn) or any(
            drawn["kind"] != block.get("kind")
            for drawn, block in zip(self._drawn, blocks)
        ):
            self.clear()
        for index, block in enumerate(blocks):
            if index < len(self._drawn):
                self._update(self._drawn[index], block)
            else:
                self._drawn.append(self._draw(block))

    def clear(self) -> None:
        self.container.clear()
        self._drawn = []

    def _draw(self, block: dict[str, Any]) -> dict[str, Any]:
        from nicegui import ui

        kind = block.get("kind")
        with self.container:
            if kind == "tool":
                return {"kind": "tool", **_tool_card(block)}
            text = block.get("text") or ""
            if kind == "thinking":
                # Drawn while it streams like everything else, and hidden or
                # shown by a class on the chat root rather than by whether it
                # was built -- so the statusline's switch costs no re-render and
                # takes no scroll position with it.
                with _reasoning_card() as card:
                    body = ui.markdown(text).classes("body")
                return {"kind": kind, "text": text, "body": body, "card": card}
            body = ui.markdown(text).classes("grad-msg grad")
            return {"kind": kind, "text": text, "body": body}

    def _update(self, drawn: dict[str, Any], block: dict[str, Any]) -> None:
        if drawn["kind"] != "tool":
            text = block.get("text") or ""
            if drawn["text"] != text:
                drawn["body"].content = text
                drawn["text"] = text
            return
        if drawn.get("status") != block.get("status"):
            _paint_status(drawn, block)
        if drawn.get("result") != block.get("result"):
            _paint_output(drawn, block)


def _reasoning_card() -> Any:
    """The box a run of reasoning is drawn in, head included."""
    card = kit.el("div", "grad-reasoning")
    with card:
        kit.text("reasoning", "head")
    return card


#: Block kinds that are already what they are and must not go through
#: `parse_message`. A tool card is structured; reasoning is the agent's own
#: working, and running the gate/expectation patterns over it would let a turn
#: *thinking about* a gate be mistaken for one asking for a decision.
STRUCTURED_KINDS = ("tool", "thinking")


def _has_gate(record: dict[str, Any]) -> bool:
    """Did this turn end by asking for a decision?

    Parsed the same way the transcript renders it, so the header and the card
    cannot disagree about what counts as a gate.
    """
    blocks = record.get("blocks") or [{"kind": "text", "text": record.get("text") or ""}]
    for block in blocks:
        if block.get("kind") in STRUCTURED_KINDS:
            continue
        for part in models.parse_message(block.get("text") or ""):
            if part.get("kind") == "gate":
                return True
    return False


def _rewind_control(workspace: Any, index: int) -> Any:
    """The ⟲ on a prompt: drop it and everything after it.

    On every prompt rather than only on the ones that failed. The turn worth
    undoing is not always the one that errored -- a question that sent the agent
    down a forty-minute path is the expensive case -- and a control that appeared
    only after a failure would be one nobody knew existed until something broke.

    No confirmation step. The action is recoverable by construction: the dropped
    turns are kept in the marker it leaves (`core/rewind.py`), so the worst a
    mis-click costs is a rebuilt prompt cache, and the status bar says exactly
    what happened. A modal on every one of these would make undoing three dead
    turns after an API error six clicks instead of three.
    """
    control = kit.text("⟲", "grad-rewind", tag="button")
    control.props(
        'title="rewind to here — drop this and everything after it, from the '
        'transcript and from what the agent remembers"'
    )
    control.on("click", lambda _=None: workspace.spawn(_rewind(workspace, index), "rewind"))
    return control


def _rewound(ui: Any, record: dict[str, Any], workspace: Any) -> None:
    """The line across the transcript where turns were taken back.

    The same shape as a compaction and for the same reason: both are boundaries
    rather than turns, both mean the conversation above is not what the agent is
    now working from, and a transcript that lost messages with no mark would read
    as one that never had them.

    What is behind the disclosure is the difference. A compaction keeps a summary
    of what it discarded; a rewind keeps the turns themselves, whole, because it
    has them -- so the exchange that was rewound past is still readable, with its
    tool calls, by anyone asking later what actually went wrong. They are drawn
    without a rewind control of their own: they are not in the conversation any
    more, so there is no position left to cut at.
    """
    with kit.el("div", "grad-compaction rewound"):
        with kit.row("head", gap=8):
            kit.text("⟲", "mark", tag="span")
            ui.markdown(record.get("text") or "rewound")
        dropped = rewind.dropped_of(record)
        if dropped:
            label = "message" if len(dropped) == 1 else "messages"
            with ui.expansion(f"the {len(dropped)} {label} this took back").classes("note"):
                for entry in dropped:
                    _message(entry, workspace)


def _compaction(ui: Any, record: dict[str, Any]) -> None:
    """The line across the transcript where the agent's memory was replaced.

    Drawn as a rule rather than as a message because that is what it is: nothing
    was said here, and everything above it is now something the agent knows only
    second-hand. Without a mark the transcript reads as one continuous
    conversation, and the first time the model fails to remember a detail that
    is plainly visible three turns up, the reasonable conclusion is that the
    agent is broken.

    The note is behind a disclosure. It is long by design -- `HANDOFF_PROMPT`
    asks for paths, commands and ledger state, not for brevity -- and it is
    exactly what someone will want to read when the answer after a compaction is
    worse than the answers before it.
    """
    with kit.el("div", "grad-compaction"):
        with kit.row("head", gap=8):
            kit.text("⊟", "mark", tag="span")
            ui.markdown(record.get("text") or "compacted")
        note = record.get("note")
        if isinstance(note, str) and note.strip():
            with ui.expansion("the handover note the previous session left").classes("note"):
                ui.markdown(note)


def _composer(ui: Any, workspace: Any, transcript: Any, tail: _Tail, statusline: Any) -> None:
    session = workspace.session

    async def settle(record: dict[str, Any]) -> None:
        tail.clear()
        # `awaiting_gate` was a state nothing ever entered: the header knew how
        # to render "AWAITING YOUR CALL" and the titlebar had a GATE chip, but
        # every path set running/paused/idle, so a turn that ended by asking for
        # a decision looked identical to one that ended by finishing. The turn
        # that just settled is exactly where that is known.
        workspace.set_agent_state("awaiting_gate" if _has_gate(record) else "idle")
        if record.get("blocks") or record.get("text"):
            with transcript:
                _message(record, workspace)
            await katex.render("#grad-transcript")

    async def send(prompt: str | None = None) -> None:
        prompt = (prompt if isinstance(prompt, str) else (entry.value or "")).strip()
        if not prompt:
            return
        if workspace.chat_sending or session.busy:
            # Said, not swallowed. This guard is what stops two turns
            # interleaving into one block list, but it used to return in
            # silence -- so a prompt typed while the previous turn was still
            # winding down simply did not happen, with the text left in the box
            # and nothing to say why. That is the same symptom as the interrupt
            # bug it usually followed, and it made that bug much harder to see.
            #
            # `chat_sending` rather than `session.busy` alone, because `busy` is
            # set inside `ask` and `_send` yields before it gets there: two
            # Enters in the same tick both passed a `busy` that was still False.
            # See `ui/state.py` for what the second one then did.
            workspace.say("a turn is still running — stop it first (Esc)")
            return
        # Set before anything is cleared or drawn, so the window a second Enter
        # could land in is closed rather than merely narrowed, and released in
        # `finally` so an exception anywhere below -- including one from
        # `maybe_compact` -- cannot leave the composer refusing every prompt
        # after it. That is the failure this guard would otherwise trade for:
        # a flag that is only cleared on the success path turns one bad turn
        # into a dead composer, which is worse than the race.
        workspace.chat_sending = True
        try:
            await _send(prompt)
        finally:
            workspace.chat_sending = False

    async def _send(prompt: str) -> None:
        """The send itself, with the guard held by its caller."""
        entry.value = ""
        # The draft is the composer's state across a rebuild, so it has to be
        # cleared where the box is -- otherwise the next redraw would put the
        # sent prompt back in.
        workspace.chat_draft = ""
        with transcript:
            # The index this prompt is about to take in `settled`: `ask` appends
            # it, so it lands exactly at the current length. Computed here rather
            # than after the turn because the element is drawn now, and a control
            # bound to the wrong index would cut in the wrong place.
            _message({"role": "user", "text": prompt}, workspace, index=len(session.settled))
        workspace.set_agent_state("running")
        # Hand the frame to the browser *before* the turn starts, which is the
        # difference between an app that is working and an app that looks
        # broken.
        #
        # Drawing an element does not send it. NiceGUI queues it and a per-client
        # outbox task emits it when the event loop next runs something else --
        # and from here to the SDK subprocess there was nothing for the loop to
        # run. `_stopped` returns without awaiting when no interrupt is pending,
        # `apply_effort` and `apply_model` both return early while `self.client`
        # is None, and `start` then does `config.load()` and
        # `preflight_environment()` synchronously before it awaits anything. So
        # on a cold session the prompt sat in the outbox for the whole spawn --
        # measured at five to seven seconds -- with the composer cleared and
        # nothing on screen to show the message had been sent at all.
        #
        # One tick is enough *because* the blocking half of that spawn now runs
        # in a worker thread (`Session.start`); without that fix this would hand
        # the outbox a loop that is about to stop running again immediately.
        await asyncio.sleep(0)
        # The prompt goes to the agent exactly as it was typed. There was once a
        # mode chip here that prefixed it with `[plan]` or `[run]`, but nothing
        # downstream ever gave those tokens a meaning -- not the system prompt,
        # not the gates, not any CLI -- so the control promised a behaviour the
        # system does not have. Whether to plan first is something you say in
        # words, and the gates are what actually stop a spend.
        await session.ask(prompt, settle)
        # After the turn has settled, never during it: compacting drops the
        # client, and dropping it underneath a live `receive_response` is the
        # failure `_stop_turn` exists to clean up after. Here rather than inside
        # `ask` because the record has to be *drawn*, and the transcript is the
        # window's to write to.
        outcome = await session.maybe_compact()
        if outcome is None:
            return
        if outcome.get("record"):
            with transcript:
                _message(outcome["record"], workspace)
        else:
            # A compaction that could not happen is worth saying out loud. The
            # session carries on oversized, which is survivable, but silence
            # here would leave a meter pinned at the threshold with nothing
            # explaining why nothing is being done about it.
            workspace.say(outcome.get("message") or "could not compact this session")

    with kit.el("div", "grad-composer"):
        # Right-aligned by the row rather than by a leading spacer: with the mode
        # chips gone there is nothing on the left for a spacer to push against.
        with kit.row("", gap=6).style("justify-content: flex-end"):
            kit.text("@notebook @paper @wiki", "grad-mention")

        with kit.row("", gap=6, align="flex-end").style("margin-top: 8px"):
            entry = (
                # No `autogrow`. It grows by CSS now -- see `field-sizing` in
                # `ui/tokens.py`, which is also where the measurement that
                # removed this prop is written down. Quasar's version reads
                # `scrollHeight` back inside the input handler, and that read is
                # a full-document layout whose cost is the transcript's size.
                # Seeded from the workspace rather than left empty: a rewind puts
                # the prompt it dropped back here to be edited, and this window
                # is rebuilt between the two.
                ui.textarea(
                    placeholder="ask, or paste a result to interrogate",
                    value=workspace.chat_draft,
                )
                .props("borderless dense")
                .classes("field")
                .style("flex: 1 1 auto; padding: 0 8px")
            )
            entry.on("keydown.enter.prevent", send)
            kit.button("SEND ⏎", tone="primary", on_click=send)
            kit.button(
                "■",
                tone="neutral",
                title="interrupt (Esc)",
                # Through the workspace, so what the session says about the
                # interrupt reaches the status bar. Bound straight to
                # `session.interrupt`, the three answers it can give -- nothing
                # is running, one is already in flight, one has been asked for --
                # all went to the same place: nowhere.
                on_click=workspace.interrupt_turn,
            )

    workspace.chat_send = send

    ui.keyboard(
        on_key=lambda e: workspace.interrupt_turn()
        if (e.key == "Escape" and e.action.keydown)
        else None
    )


# ---------------------------------------------------------------------------
# message anatomy
# ---------------------------------------------------------------------------
def _message(record: dict[str, Any], workspace: Any, index: int | None = None) -> None:
    """One settled message: a user's prompt, or a turn with its calls in it.

    A record carries `blocks` when the stream produced them and only `text` when
    it did not -- an older transcript, or a user's own message -- so the fallback
    is the whole message as one run of prose, which is exactly what this drew
    before tool calls were captured at all.

    `index` is the record's position in `session.settled`, and passing it is what
    makes a prompt rewindable. It is absent for the two kinds of message that are
    not a rewind point: an answer, and anything drawn inside a rewind marker's
    own disclosure -- those turns are already gone, and offering to drop them
    again would cut at a position they no longer occupy.
    """
    from nicegui import ui

    text = record.get("text") or ""
    if record.get("role") == "system":
        if record.get("kind") == rewind.MARK_KIND:
            _rewound(ui, record, workspace)
        else:
            _compaction(ui, record)
        return
    if record.get("role") == "user":
        with kit.el("div", "grad-msg user"):
            with kit.row("role", gap=6):
                kit.text("you", "", tag="span")
                if index is not None:
                    _rewind_control(workspace, index)
            with kit.el("div", "bubble"):
                ui.markdown(text)
        return

    with kit.el("div", "grad-msg grad"):
        with kit.row("role", gap=6):
            kit.text("∇", "grad-avatar", tag="span")
            kit.text("grad", "", tag="span")
        for block in record.get("blocks") or [{"kind": "text", "text": text}]:
            if block.get("kind") in STRUCTURED_KINDS:
                _block(ui, block, workspace)
                continue
            # Prose is parsed on the way *in* to the transcript rather than on
            # the way through the tail: gates and expectation cards are worth a
            # re-render once each, and not worth one 15 times a second.
            for part in models.parse_message(block.get("text") or ""):
                _block(ui, part, workspace)


def _block(ui: Any, block: dict[str, Any], workspace: Any) -> None:
    kind = block["kind"]
    if kind == "text":
        ui.markdown(block["text"], extras=["fenced-code-blocks", "tables"]).classes("bubble")
    elif kind == "code":
        with kit.el("div", "grad-card"):
            kit.text(block.get("language", "text").upper(), "head ink")
            with kit.el("div", "body"):
                kit.pre(block["text"])
    elif kind == "tool":
        _tool_card(block)
    elif kind == "thinking":
        with _reasoning_card():
            ui.markdown(block.get("text") or "").classes("body")
    elif kind == "figure":
        _figure(ui, block)
    elif kind == "expectation":
        with kit.el("div", "grad-card"):
            with kit.row("head attention", gap=9):
                kit.text("EXPECTATION REGISTERED", "", tag="span")
                kit.spacer()
                kit.text(block.get("id") or "", "", tag="span")
            with kit.el("div", "body"):
                kit.kv(_rows(block))
    elif kind == "gate":
        _gate_card(block, workspace)


def _figure(ui: Any, block: dict[str, Any]) -> None:
    """A figure the agent drew, at the point in the message where it drew it.

    A bare `<img>`, not `ui.image`. `ui.image` is Quasar's `QImg`, which wraps
    the picture in a fixed-ratio box, cross-fades it in and shows a spinner
    first -- three things the design rules out in one component, and the ratio
    box letterboxes a plot whose aspect nobody chose. It also lazy-loads, which
    read as "the figure did not render" every time one sat below the fold.

    A **URL**, not a path: `ui.image(Path)` copies the file into NiceGUI's media
    registry under a content-addressed url, so a transcript would keep showing
    the plot as it was when the message was first drawn -- and a figure is
    exactly the file the next run of the same cell overwrites.
    """
    # `html.escape`, not `kit.attr`: this is interpolated into markup rather than
    # into a NiceGUI props string, and the two want opposite treatments -- attr
    # swaps the double quote for an apostrophe and doubles backslashes for
    # `literal_eval`, both of which would reach the screen here. The alt text is
    # whatever the agent wrote between the brackets.
    src = html.escape(block["src"], quote=True)
    alt = html.escape(block.get("alt") or "figure", quote=True)
    ui.html(
        f'<img class="grad-figure-img" src="{src}" alt="{alt}" decoding="async">',
        sanitize=False,
    )


def _tool_card(block: dict[str, Any]) -> dict[str, Any]:
    """A call: what it was, what it was on, how it went, and what it said.

    Returns the handles the tail updates in place. A settled render throws them
    away -- nothing about a finished call changes again.
    """
    with kit.el("div", "grad-card tool"):
        with kit.row("head ink", gap=9):
            kit.text("TOOL", "", tag="span")
            if block.get("name"):
                kit.text(block["name"], "", tag="span")
            if block.get("title"):
                # Shortened for the head, whole in the tooltip -- see `_call` in
                # ui/windows/tasks.py, which shows the same subject.
                subject = kit.text(kit.shorten_path(block["title"]), "subject", tag="span")
                subject.props(f'title="{kit.attr(block["title"])}"')
            kit.spacer()
            state = kit.el("span", "state")
        with kit.el("div", "body"):
            # Only when the head could not show all of it: a short command is
            # already up there in full, and printing it twice makes the card
            # look like it is saying two things when it is saying one.
            if block.get("text") and block.get("text") != block.get("title"):
                kit.pre(block["text"])
            rows = _rows(block)
            if rows:
                kit.kv(rows)
            output = kit.el("div", "out")

    handles = {"state": state, "output": output}
    _paint_status(handles, block)
    _paint_output(handles, block)
    return handles


def _paint_status(handles: dict[str, Any], block: dict[str, Any]) -> None:
    """The chip, rebuilt on a state change rather than updated.

    A chip is a span with a class and a dot in it; there is no leaf to set. The
    rebuild happens twice per call -- running, then the outcome -- not per tick.
    """
    # Blocks that came out of `parse_message` are transcript prose, not a live
    # call, and have no status: they finished long ago.
    status = str(block.get("status") or "ok")
    handles["state"].clear()
    with handles["state"]:
        kit.chip(status.upper(), STATUS_TONE.get(status, "neutral"), dot=status == "running")
    handles["status"] = block.get("status")


def _paint_output(handles: dict[str, Any], block: dict[str, Any]) -> None:
    result = str(block.get("result") or "")
    handles["output"].clear()
    if result:
        with handles["output"]:
            kit.sublabel("output")
            kit.pre(result, "broken" if block.get("status") == "error" else "neutral")
    handles["result"] = block.get("result")


def _rows(block: dict[str, Any]) -> list[tuple[str, str]]:
    """`rows` as pairs, whatever the source.

    A block that came back off disk has been through JSON, so its pairs are
    lists rather than tuples -- and a transcript file written by another version
    can hold anything at all. `kit.kv` unpacks, so a row of the wrong width has
    to be dropped here rather than raised on at build time.
    """
    rows = block.get("rows")
    if not isinstance(rows, list):
        return []
    return [
        (str(row[0]), str(row[1]))
        for row in rows
        if isinstance(row, (list, tuple)) and len(row) == 2
    ]


def _gate_card(block: dict[str, Any], workspace: Any) -> None:
    """A gate, and the three answers to it.

    The answer is a message back into the same session rather than a separate
    control channel, because that is the mechanism this agent actually has:
    `core/gates.py` refuses at the CLI, and the conversational gate is the agent
    asking and then continuing. Sending the decision as a turn keeps the reason
    in the transcript, which is where anyone auditing the decision later will
    look for it.
    """

    def answer(text: str) -> None:
        send = workspace.chat_send
        if send is None:
            # Before the state change, not after: leaving `running` set here
            # would paint the title bar with a live agent and a PAUSE button
            # while nothing is running and nothing will start.
            workspace.say("no chat session is open to answer the gate")
            return
        workspace.set_agent_state("running")
        result = send(text)
        if hasattr(result, "__await__"):
            workspace.spawn(result, "gate answer")

    with kit.el("div", "grad-card gate"):
        with kit.row("head broken", gap=9):
            kit.text("GATE — YOUR CALL", "", tag="span")
            kit.spacer()
            kit.text(block.get("id") or "", "", tag="span")
        with kit.el("div", "body"):
            kit.kv(_rows(block))
            with kit.row("", gap=6).style("margin-top: 10px"):
                kit.button("✓ APPROVE", tone="ok", on_click=lambda: answer("approved — proceed"))
                kit.button(
                    "✎ EDIT PLAN",
                    tone="neutral",
                    on_click=lambda: answer("hold — revise the plan before spending anything"),
                )
                kit.button(
                    "✕ DENY",
                    tone="neutral",
                    on_click=lambda: answer("denied — do not spend this; explain the alternative"),
                )
