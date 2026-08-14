"""Window 1 — the agent session.

The two implementation details that are the difference between this feeling
like a tool and feeling like a demo survive the redesign unchanged, because
they were never about the visuals:

  * **Buffered flush.** Updating a `ui.markdown` per token re-renders and
    reflows the whole element on every token. Tokens go into a buffer and a
    `ui.timer` flushes at ~15 Hz.
  * **Split tail.** The streaming message lives in its own element, separate
    from the settled transcript above it, so only the tail re-renders. It is
    promoted into the transcript (and KaTeX runs over it) once complete.

What the redesign adds is anatomy: a turn is parsed into prose, tool calls,
expectation cards and gate cards, each with its own shape, rather than being one
markdown blob. `ui/models.py:parse_message` does the parsing, so what counts as
a gate is testable without a browser.

This window is never redrawn by the poll. Its state is the live session, not a
file, and a redraw would take the transcript's scroll position with it.
"""

from __future__ import annotations

from typing import Any

from ui import katex, kit, models
from ui.state import FLUSH_HZ

MODES = ("ask", "plan", "run")


def subtitle(workspace: Any) -> str:
    session = getattr(workspace, "session", None)
    settled = len(getattr(session, "settled", []) or [])
    return f"{settled} messages · {workspace.agent_state}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    if workspace.agent_state == "awaiting_gate":
        return [("GATE", "broken")]
    if getattr(workspace.session, "busy", False):
        return [("STREAMING", "ok")]
    return []


def render(workspace: Any) -> None:
    from nicegui import ui

    session = workspace.session

    with kit.column("", gap=0).classes("h-full"):
        transcript = kit.column("grad-transcript", gap=0)
        transcript.props('id="grad-transcript"')
        transcript.style("flex: 1 1 auto; overflow-y: auto; min-height: 0")
        with transcript:
            for message in session.settled:
                _message(message["role"], message["text"], workspace)

        tail = kit.column("", gap=0)
        with tail:
            tail_body = ui.markdown("").classes("grad-msg grad")

        streaming = kit.el("div", "grad-streaming")
        streaming.style("display: none")
        with streaming:
            kit.el("span", "block")
            kit.text("running …", "", tag="span")
            kit.spacer()
            kit.text("esc to interrupt", "", tag="span")

        _composer(ui, workspace, transcript, tail_body, streaming)

    def flush() -> None:
        # ~15 Hz, not per token: only the tail element re-renders.
        busy = bool(session.busy)
        streaming.style(f"display: {'flex' if busy else 'none'}")
        if session.buffer and tail_body.content != session.buffer:
            tail_body.content = session.buffer
        elif not busy and tail_body.content and not session.buffer:
            tail_body.content = ""

    ui.timer(1 / FLUSH_HZ, flush)


def _composer(ui: Any, workspace: Any, transcript: Any, tail_body: Any, streaming: Any) -> None:
    session = workspace.session
    mode = {"value": "ask"}

    async def settle(text: str) -> None:
        tail_body.content = ""
        streaming.style("display: none")
        workspace.set_agent_state("idle")
        if text:
            with transcript:
                _message("assistant", text, workspace)
            await katex.render("#grad-transcript")

    async def send(prompt: str | None = None) -> None:
        prompt = (prompt if isinstance(prompt, str) else (entry.value or "")).strip()
        if not prompt or session.busy:
            return
        entry.value = ""
        with transcript:
            _message("user", prompt, workspace)
        workspace.set_agent_state("running")
        # The mode is a prefix rather than a separate code path: `plan` and
        # `run` are instructions to the agent, not to the UI, and encoding them
        # here would put behaviour in the window (§10 says it does not live
        # here).
        prefixed = prompt if mode["value"] == "ask" else f"[{mode['value']}] {prompt}"
        await session.ask(prefixed, settle)

    with kit.el("div", "grad-composer"):
        with kit.row("", gap=6):
            chips_row = kit.row("", gap=0)
            with chips_row:
                buttons: dict[str, Any] = {}
                for name in MODES:
                    buttons[name] = kit.button(
                        name.upper(),
                        tone="active" if name == "ask" else "neutral",
                        on_click=lambda _=None, n=name: _set_mode(mode, buttons, n),
                    )
            kit.spacer()
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
            kit.button("■", tone="neutral", title="interrupt (Esc)", on_click=session.interrupt)

    workspace.chat_send = send

    ui.keyboard(
        on_key=lambda e: session.interrupt() if (e.key == "Escape" and e.action.keydown) else None
    )


def _set_mode(mode: dict[str, str], buttons: dict[str, Any], name: str) -> None:
    mode["value"] = name
    for key, element in buttons.items():
        element.classes(remove="active")
        if key == name:
            element.classes(add="active")


# ---------------------------------------------------------------------------
# message anatomy
# ---------------------------------------------------------------------------
def _message(role: str, text: str, workspace: Any) -> None:
    from nicegui import ui

    if role == "user":
        with kit.el("div", "grad-msg user"):
            kit.text("you", "role")
            with kit.el("div", "bubble"):
                ui.markdown(text)
        return

    with kit.el("div", "grad-msg grad"):
        with kit.row("role", gap=6):
            kit.text("∇", "grad-avatar", tag="span")
            kit.text("grad", "", tag="span")
        for block in models.parse_message(text):
            _block(ui, block, workspace)
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
        with kit.el("div", "grad-card"):
            with kit.row("head ink", gap=9):
                kit.text("TOOL", "", tag="span")
                kit.text(block["title"], "", tag="span")
                kit.spacer()
                kit.chip("OK", "ok")
            with kit.el("div", "body"):
                kit.pre(block["text"])
    elif kind == "expectation":
        with kit.el("div", "grad-card"):
            with kit.row("head attention", gap=9):
                kit.text("EXPECTATION REGISTERED", "", tag="span")
                kit.spacer()
                kit.text(block.get("id") or "", "", tag="span")
            with kit.el("div", "body"):
                kit.kv(block.get("rows") or [])
    elif kind == "gate":
        _gate_card(block, workspace)


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
        workspace.set_agent_state("running")
        if send is None:
            workspace.say("no chat session is open to answer the gate")
            return
        result = send(text)
        if hasattr(result, "__await__"):
            import asyncio

            # Consumed rather than fire-and-forget: a bare `create_task` drops
            # any SDK error and Python logs "Task exception was never retrieved".
            task = asyncio.create_task(result)
            task.add_done_callback(lambda t: t.exception())

    with kit.el("div", "grad-card gate"):
        with kit.row("head broken", gap=9):
            kit.text("GATE — YOUR CALL", "", tag="span")
            kit.spacer()
            kit.text(block.get("id") or "", "", tag="span")
        with kit.el("div", "body"):
            kit.kv(block.get("rows") or [])
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
