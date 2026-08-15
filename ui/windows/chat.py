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

import time
from typing import Any

from ui import katex, kit, models
from ui.state import FLUSH_HZ

#: A call's status -> the one accent it is drawn in. Dashed while it runs, for
#: the same reason the design uses a dashed border everywhere else it means
#: "pending": an outcome that has not happened yet is not a green one.
STATUS_TONE = {"running": "dashed", "ok": "ok", "error": "broken"}


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
    workspace.rebuild_chat()


async def _switch(workspace: Any, session_id: str) -> None:
    workspace.say(await workspace.session.open_session(session_id))
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
            transcript = kit.column("", gap=0)
            tail_root = kit.column("", gap=0)
            tail_root.props('id="grad-tail"')
        tail = _Tail(tail_root)

        with transcript:
            for message in session.settled:
                _message(message, workspace)

        statusline = _Statusline(workspace, root)
        _composer(ui, workspace, transcript, tail, statusline)

    def flush() -> None:
        # ~15 Hz, not per token: only the tail re-renders, and inside it only
        # the block that actually moved.
        blocks = getattr(session, "blocks", None) or []
        tail.sync(blocks)
        statusline.sync(blocks)

    ui.timer(1 / FLUSH_HZ, flush)
    # Once, at build: keep the transcript pinned to the bottom while a turn
    # streams. Doing it from here instead would be a `run_javascript` per flush.
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
            self.clock = kit.text("", "clock", tag="span")
            self.reasoning = kit.text("", "reasoning", tag="span")
        self.bar = bar
        self._paint_reasoning()

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

    def _paint_reasoning(self) -> None:
        showing = self.workspace.show_reasoning
        kit.set_text(self.reasoning, f"reasoning {'■ on' if showing else '□ off'}")

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
        if session.busy:
            # Said, not swallowed. This guard is what stops two turns
            # interleaving into one block list, but it used to return in
            # silence -- so a prompt typed while the previous turn was still
            # winding down simply did not happen, with the text left in the box
            # and nothing to say why. That is the same symptom as the interrupt
            # bug it usually followed, and it made that bug much harder to see.
            workspace.say("a turn is still running — stop it first (Esc)")
            return
        entry.value = ""
        with transcript:
            _message({"role": "user", "text": prompt}, workspace)
        workspace.set_agent_state("running")
        # The prompt goes to the agent exactly as it was typed. There was once a
        # mode chip here that prefixed it with `[plan]` or `[run]`, but nothing
        # downstream ever gave those tokens a meaning -- not the system prompt,
        # not the gates, not any CLI -- so the control promised a behaviour the
        # system does not have. Whether to plan first is something you say in
        # words, and the gates are what actually stop a spend.
        await session.ask(prompt, settle)

    with kit.el("div", "grad-composer"):
        # Right-aligned by the row rather than by a leading spacer: with the mode
        # chips gone there is nothing on the left for a spacer to push against.
        with kit.row("", gap=6).style("justify-content: flex-end"):
            kit.text("@notebook @paper @wiki", "grad-mention")

        with kit.row("", gap=6, align="flex-end").style("margin-top: 8px"):
            entry = (
                ui.textarea(placeholder="ask, or paste a result to interrogate")
                .props("autogrow borderless dense")
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
def _message(record: dict[str, Any], workspace: Any) -> None:
    """One settled message: a user's prompt, or a turn with its calls in it.

    A record carries `blocks` when the stream produced them and only `text` when
    it did not -- an older transcript, or a user's own message -- so the fallback
    is the whole message as one run of prose, which is exactly what this drew
    before tool calls were captured at all.
    """
    from nicegui import ui

    text = record.get("text") or ""
    if record.get("role") == "user":
        with kit.el("div", "grad-msg user"):
            kit.text("you", "role")
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
        for figure in models.figures_in(text):
            ui.image(figure).classes("w-full max-w-2xl")


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
                kit.text(block["title"], "subject", tag="span")
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
            kit.text("OUTPUT", "grad-label")
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
