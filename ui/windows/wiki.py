"""Window 10 — the codebase wiki and its references rail.

The design shows chat on the left and a numbered references rail on the right.
The rail here holds what actually exists: the wiki's scope, when it was
generated, and -- when the source tree has moved underneath it -- exactly which
files differ.

There is no answer engine behind this window, and that is a decision rather than
a gap. `tools/wiki.py` deliberately does not enable `repowiki scan`, the LLM
half, because it reads `ANTHROPIC_API_KEY` by default -- the exact variable
`credentials.scrub_environment()` deletes -- and a key in the user profile is
also in the agent's environment, tripping the scrub warning on every launch.
Safe, but noisy, and habituation to that warning erodes the credential
discipline. So questions about the codebase go to the agent with an `@wiki`
mention, and this window is the map plus its staleness.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.tasks import start, task_message


def subtitle(workspace: Any) -> str:
    model = workspace.model("wiki") or {}
    if not model.get("built"):
        return "not generated"
    return f"generated {model.get('generated_at', '?')}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("wiki") or {}
    if not model.get("built"):
        return []
    if model.get("stale"):
        return [(f"STALE · {model.get('changed_total', 0)} FILES", "attention")]
    return [("CURRENT", "ok")]


def render(workspace: Any) -> None:
    model = workspace.model("wiki") or {}
    kit.error_strip(model.get("error"))

    def settled(task: Any) -> None:
        workspace.say(task_message(task))
        workspace.invalidate("wiki")
        workspace.tick()

    def rebuild() -> None:
        """RepoWiki walks `core/` and `tools/` and calls a model per scope, so
        this is minutes rather than seconds -- and nothing else in the workspace
        needs to wait for it."""
        start("wiki map", "tools.wiki", "map", "--json", on_done=settled)
        workspace.say("regenerating the wiki — see the tasks window")
        workspace.invalidate("tasks")
        workspace.tick()

    if not model.get("built"):
        kit.empty("No wiki has been generated yet.", model.get("empty_fix"))
        with kit.pad():
            kit.button("▶ GENERATE", tone="primary", on_click=rebuild)
        return

    with kit.row("", gap=0, align="stretch").style("min-height: 0; flex: 1 1 auto"):
        with kit.column("grad-pad", gap=10).style("flex: 1 1 auto; min-width: 0; overflow-y: auto"):
            with kit.row("", gap=9):
                kit.button("↻ REGENERATE", tone="primary", on_click=rebuild)
                kit.spacer()
                kit.text(model.get("source_hash") or "", "grad-caption", tag="span")

            if model.get("stale"):
                with kit.el("div", "grad-card"):
                    kit.text("STALE", "head attention")
                    with kit.el("div", "body"):
                        kit.text(
                            f"the wiki was generated from a different source tree: "
                            f"{model.get('changed_total', 0)} file(s) differ",
                            "",
                        )
                        for path in model.get("changed") or []:
                            kit.text(path, "grad-mono")
            else:
                kit.note("the wiki matches the current source tree")

            kit.label("ask about the codebase")
            kit.note(
                "Questions go to the agent with an @wiki mention rather than to a local "
                "answer engine: repowiki's scan half reads ANTHROPIC_API_KEY, the variable "
                "the credential scrub deletes."
            )
            _ask(workspace)

        _rail(model)


def _ask(workspace: Any) -> None:
    from nicegui import ui

    def send() -> None:
        question = (entry.value or "").strip()
        if not question:
            return
        entry.value = ""
        chat_send = workspace.chat_send
        if chat_send is None:
            workspace.open("chat")
            workspace.say("opened chat — ask again once it is up")
            return
        result = chat_send(f"@wiki {question}")
        if hasattr(result, "__await__"):
            workspace.spawn(result, "wiki question")
        workspace.focus("chat")

    # `grad-wiki-ask` so the composer's `field-sizing` rule reaches this box too.
    # The same `autogrow` was here, and a forced layout is document-wide however
    # small the window asking for it -- so typing a question in this pane paid
    # the size of the *chat* transcript in the pane beside it.
    with kit.row("grad-wiki-ask", gap=6, align="flex-end"):
        entry = (
            ui.textarea(placeholder="ask about this codebase")
            .props("borderless dense")
            .classes("field")
            .style("flex: 1 1 auto; border: var(--grad-border); background: var(--grad-paper-raised); padding: 0 8px")
        )
        entry.on("keydown.enter.prevent", send)
        kit.button("ASK ⏎", tone="primary", on_click=send)


def _rail(model: dict[str, Any]) -> None:
    from nicegui import ui

    with kit.column("grad-pad", gap=9).style(
        "flex: 0 0 440px; background: var(--grad-paper-sunk); "
        "border-left: var(--grad-border); overflow-y: auto"
    ):
        kit.label("references")
        for index, scope in enumerate(model.get("scopes") or [], start=1):
            with kit.row("grad-row", gap=9):
                kit.chip(str(index), "solid")
                kit.text(scope["name"], "grad-mono", tag="span")
                kit.spacer()
                if scope.get("entries") is not None:
                    kit.text(f"{scope['entries']} entries", "grad-caption", tag="span")
        if model.get("html"):
            kit.hr()
            kit.text(model["output_dir"], "grad-caption")
            kit.button(
                "↗ OPEN THE MAP",
                tone="neutral",
                on_click=lambda: ui.navigate.to("/grad-wiki/index.html", new_tab=True),
            )
