"""The NiceGUI desktop app (HANDOFF §10).

    "The things that make it pleasant -- being able to see a funnel's reasoning,
     a preflight's failing check, a prediction against its outcome -- are the
     same things that make it trustworthy."

Two implementation details are the difference between this feeling like a tool
and feeling like a demo, and both are cheap:

  * **Buffered flush.** Updating a `ui.markdown` per token re-renders and
    reflows the whole element on every token. Tokens go into a buffer and a
    `ui.timer` flushes at ~15 Hz.
  * **Split tail.** The streaming message lives in its own element, separate
    from the settled transcript above it, so only the tail re-renders. It is
    promoted into the transcript (and KaTeX runs over it) once the message
    completes.

Notebooks render, they do not rebuild: JupyterLab already exists and is better
at editing. Building a notebook editor is the single easiest way to burn a month
on this project.
"""

from __future__ import annotations

import asyncio
import html
import json
from pathlib import Path
from typing import Any

from core import config as config_mod, paths
from ui import katex
from ui.widgets import expectation_panel, funnel_view, preflight_panel, quota_meter, quota_panel

FLUSH_HZ = 15
SESSION_FILE = "ui_session.jsonl"

# Quasar's defaults, overridden rather than accepted -- untouched spacing and
# typography is the giveaway that something is a stock NiceGUI app.
THEME = """
<style>
  :root {
    --grad-accent: #38bdf8;
    --grad-bg: #0b0f14;
    --grad-panel: #121820;
  }
  body { background: var(--grad-bg); }
  .grad-transcript { font-size: 15px; line-height: 1.65; letter-spacing: 0.005em; }
  .grad-transcript h1 { font-size: 1.5rem; margin: 1.2rem 0 .5rem; font-weight: 600; }
  .grad-transcript h2 { font-size: 1.2rem; margin: 1rem 0 .4rem; font-weight: 600; }
  .grad-transcript p  { margin: .5rem 0; }
  .grad-transcript code { font-size: 13px; }
  .grad-panel { background: var(--grad-panel); border: 1px solid #1e2732; border-radius: 10px; }
  .grad-user { border-left: 2px solid var(--grad-accent); padding-left: .75rem; opacity: .9; }
  .q-tab { text-transform: none; letter-spacing: 0; }
</style>
"""


class Session:
    """Owns the `ClaudeSDKClient` and the token buffer.

    The UI holds no logic of its own beyond this: everything else it shows is
    read from the ledger or produced by the CLIs.
    """

    def __init__(self) -> None:
        self.client: Any = None
        self.buffer: str = ""
        self.settled: list[dict[str, str]] = []
        self.busy = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.client is not None:
            return
        import agent  # noqa: PLC0415 - imported here so the UI can load without the SDK
        from claude_agent_sdk import ClaudeSDKClient  # noqa: PLC0415

        cfg = config_mod.load()
        agent.preflight_environment()
        self.client = ClaudeSDKClient(options=agent.build_options(cfg))
        await self.client.__aenter__()

    async def close(self) -> None:
        """Exit the client context entered by `start`.

        The client owns a CLI subprocess and transport tasks. Entering the
        context by hand and never exiting it leaves those alive for the life of
        the app, and leaks a whole set on any later reconnect.
        """
        client, self.client = self.client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - shutdown must not raise on the way out
                pass

    async def ask(self, prompt: str, on_settle: Any) -> None:
        await self.start()
        import agent  # noqa: PLC0415

        self.settled.append({"role": "user", "text": prompt})
        self.busy = True
        self.buffer = ""
        try:
            await self.client.query(prompt)
            async for message in self.client.receive_response():
                text = agent._text_of(message)  # noqa: SLF001 - one helper, deliberately shared
                if text:
                    self.buffer += text
        except Exception as exc:  # noqa: BLE001 - the transcript must say why a turn died
            # Otherwise the turn settles as an empty message: the prompt looks
            # unanswered and the reason is only in the server log.
            self.buffer += f"\n\n**the session failed:** `{type(exc).__name__}: {exc}`"
        finally:
            self.busy = False
            settled_text = self.buffer
            self.buffer = ""
            if settled_text:
                self.settled.append({"role": "assistant", "text": settled_text})
            self._persist()
            await on_settle(settled_text)

    def interrupt(self) -> None:
        """Interrupt the turn in flight, if there is one.

        Bound to a button and to Escape, so it fires when nothing is running.
        The result has to be consumed: a bare `create_task` drops any SDK error
        and Python logs "Task exception was never retrieved".
        """
        if not (self.busy and self.client and hasattr(self.client, "interrupt")):
            return

        async def _interrupt() -> None:
            try:
                await self.client.interrupt()
            except Exception:  # noqa: BLE001 - a failed interrupt must not kill the app
                pass

        self._task = asyncio.create_task(_interrupt())

    def _persist(self) -> None:
        """Closing the window should not be destructive."""
        path = paths.data_dir() / SESSION_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in self.settled), encoding="utf-8"
        )

    def restore(self) -> None:
        path = paths.data_dir() / SESSION_FILE
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                self.settled.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def build() -> None:
    from nicegui import app as nicegui_app, ui

    ui.add_head_html(THEME)
    katex.install(nicegui_app)
    ui.dark_mode(True)

    session = Session()
    session.restore()
    nicegui_app.on_shutdown(session.close)

    with ui.header().classes("items-center justify-between px-4 py-2 grad-panel"):
        with ui.row().classes("items-center gap-2"):
            ui.label("Grad").classes("text-lg font-semibold")
            ui.label("research instrument").classes("text-xs opacity-50")
        quota_meter()

    with ui.tabs().classes("w-full") as tabs:
        tab_chat = ui.tab("Session")
        tab_preflight = ui.tab("Preflight")
        tab_expect = ui.tab("Expectations")
        tab_funnel = ui.tab("Funnel")
        tab_quota = ui.tab("Quota")
        tab_nb = ui.tab("Notebooks")

    with ui.tab_panels(tabs, value=tab_chat).classes("w-full"):
        with ui.tab_panel(tab_chat):
            _chat_panel(ui, session)
        with ui.tab_panel(tab_preflight):
            _refreshable(ui, preflight_panel)
        with ui.tab_panel(tab_expect):
            _refreshable(ui, expectation_panel)
        with ui.tab_panel(tab_funnel):
            _refreshable(ui, funnel_view)
        with ui.tab_panel(tab_quota):
            _refreshable(ui, quota_panel)
        with ui.tab_panel(tab_nb):
            _notebook_panel(ui)


def _refreshable(ui: Any, render: Any) -> None:
    container = ui.column().classes("w-full")

    def draw() -> None:
        container.clear()
        with container:
            render()

    ui.button(icon="refresh", on_click=draw).props("flat dense").classes("self-end")
    draw()


def _chat_panel(ui: Any, session: Session) -> None:
    transcript = ui.column().classes("w-full gap-3 grad-transcript").props('id="grad-transcript"')
    with transcript:
        for message in session.settled:
            _bubble(ui, message["role"], message["text"])

    tail = ui.markdown("").classes("w-full grad-transcript opacity-90")

    def flush() -> None:
        # ~15 Hz, not per token: only the tail element re-renders.
        if session.buffer and tail.content != session.buffer:
            tail.content = session.buffer

    ui.timer(1 / FLUSH_HZ, flush)

    async def settle(text: str) -> None:
        tail.content = ""
        if text:
            with transcript:
                _bubble(ui, "assistant", text)
            await katex.render("#grad-transcript")

    async def send() -> None:
        prompt = entry.value.strip()
        if not prompt or session.busy:
            return
        entry.value = ""
        with transcript:
            _bubble(ui, "user", prompt)
        await session.ask(prompt, settle)

    with ui.row().classes("w-full items-end gap-2 mt-2"):
        entry = ui.textarea(placeholder="ask, or paste a result to interrogate").classes("flex-grow").props(
            "autogrow outlined dense"
        )
        entry.on("keydown.enter.prevent", send)
        ui.button("Send", on_click=send).props("unelevated")
        ui.button(icon="stop", on_click=session.interrupt).props("flat dense").tooltip("interrupt (Esc)")

    # Keyboard-first: submit, interrupt, jump to the latest tool call.
    ui.keyboard(
        on_key=lambda e: session.interrupt() if (e.key == "Escape" and e.action.keydown) else None
    )


def _bubble(ui: Any, role: str, text: str) -> None:
    if role == "user":
        ui.markdown(text).classes("grad-user w-full")
        return
    with ui.column().classes("w-full gap-1"):
        for block in _split_tool_calls(text):
            if block["kind"] == "tool":
                # Tool calls render as collapsible cards, not raw text.
                with ui.expansion(block["title"], icon="terminal").classes("w-full grad-panel"):
                    ui.code(block["text"], language="bash").classes("w-full")
            else:
                ui.markdown(block["text"], extras=["fenced-code-blocks", "tables"]).classes("w-full")
        for figure in _figures_in(text):
            ui.image(figure).classes("w-full max-w-2xl rounded")


def _split_tool_calls(text: str) -> list[dict[str, str]]:
    """Very small parser: fenced bash blocks become cards, prose stays prose."""
    out: list[dict[str, str]] = []
    parts = text.split("```")
    for index, part in enumerate(parts):
        if index % 2 == 0:
            if part.strip():
                out.append({"kind": "text", "text": part})
            continue
        lang, _, body = part.partition("\n")
        if lang.strip() in ("bash", "sh", "console"):
            first = body.strip().splitlines()[0] if body.strip() else "command"
            out.append({"kind": "tool", "title": first[:80], "text": body.strip()})
        else:
            out.append({"kind": "text", "text": f"```{part}```"})
    return out


def _figures_in(text: str) -> list[str]:
    """Figures are referenced by path; the UI renders them from that path, so
    the two-call workaround in §8 costs the human nothing."""
    found = []
    for token in text.replace("(", " ").replace(")", " ").split():
        if token.endswith(".png") and "figures" in token.replace("\\", "/"):
            path = Path(token)
            if path.exists():
                found.append(str(path))
    return found


def _notebook_panel(ui: Any) -> None:
    """Render notebook *outputs*, read-only, with a link out to JupyterLab."""
    notebooks = sorted(paths.notebooks_dir().glob("*.ipynb")) if paths.notebooks_dir().exists() else []
    if not notebooks:
        ui.label("No notebooks yet.").classes("text-sm opacity-60")
        return

    container = ui.column().classes("w-full")

    def show(name: str) -> None:
        container.clear()
        path = paths.notebooks_dir() / name
        with container:
            with ui.row().classes("items-center gap-3"):
                ui.link("open in JupyterLab", f"http://localhost:8888/lab/tree/notebooks/{name}").classes("text-sm")
                ui.code(f"python -m tools.nb verify notebooks/{name} --json", language="bash")
            try:
                import nbformat  # noqa: PLC0415
                from nbconvert import HTMLExporter  # noqa: PLC0415

                nb = nbformat.read(path, as_version=4)
                body, _ = HTMLExporter(template_name="basic").from_notebook_node(nb)
                # Sandboxed iframe, not ui.html: notebook outputs are untrusted
                # HTML and can carry <script> or event handlers. Injecting them
                # into the app origin would let a notebook from a downloaded
                # repository run script in the page that drives the agent.
                ui.element("iframe").props(
                    f'sandbox="" srcdoc="{html.escape(body, quote=True)}"'
                ).classes("w-full h-96 bg-white rounded")
            except ImportError:
                ui.label("install nbformat and nbconvert to render notebooks").classes("text-sm opacity-60")
            except Exception as exc:  # noqa: BLE001 - one bad notebook must not break the panel
                ui.label(f"could not render {name}: {exc}").classes("text-sm text-red-400")

    ui.select([p.name for p in notebooks], value=notebooks[0].name, on_change=lambda e: show(e.value)).classes("w-full")
    show(notebooks[0].name)


def run(*, native: bool = True, port: int = 8080) -> None:
    """`ui.run(native=True)` gives a real desktop window via pywebview, so the
    packaging question is answered without Electron or Tauri. Browser mode is
    the fallback when pywebview misbehaves on Windows."""
    from nicegui import ui

    paths.ensure_workspace()
    build()
    ui.run(
        native=native,
        title="Grad",
        # Explicit localhost: in browser mode NiceGUI would otherwise bind
        # 0.0.0.0, and this is an unauthenticated UI that accepts prompts for an
        # agent with Bash access.
        host="127.0.0.1",
        port=port,
        reload=False,
        dark=True,
        window_size=(1400, 900),
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
