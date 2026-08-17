"""Window 10 — the project wiki and its page rail.

The subject is the research code the agent generated for the selected project:
`pipelines/<name>/` -- the training script, the model, the data loader, the
probe, the tests -- together with the spec that submits it and the ledger
entries that record what it found. Grad's *own* `core/` and `tools/` are a
different question with a different reader, and `tools/wiki.py` answers that one
from the command line.

The design's chat-plus-numbered-references becomes a page list beside a page,
which is the same shape doing the job it was drawn for: the rail holds what
there is to read, and the pane holds what is being read.

**Nothing is generated here.** `core/wikigen.py` writes the pages, and this
window starts a background task to do it -- a build is several model calls and a
window redrawn every two seconds must not be able to spend money. What is drawn
is what a build wrote, plus the one thing a reader has to know before trusting
it: whether the code has moved underneath.
"""

from __future__ import annotations

from typing import Any

from ui import kit, tasks as tasks_mod
from ui.tasks import start, task_message

#: The rail's label per page kind. The generated titles are long and specific --
#: `minimamba/model.py` -- so the kind is what makes the list scannable.
KIND_LABEL = {"overview": "OVERVIEW", "run-path": "RUN PATH", "module": "MODULE"}


def _building(project: str | None) -> bool:
    """Is a wiki build for this project already running?

    Off the task registry rather than a flag on the window: a window is rebuilt
    whenever its model changes and its subtree is rebuilt on every retile, so a
    flag held here would be a flag that resets under a background poll. The
    registry is process-wide and outlives both.
    """
    if not project:
        return False
    return any(
        task.argv[:2] == ("tools.projwiki", "build") and "--project" in task.argv
        and task.argv[task.argv.index("--project") + 1] == project
        for task in tasks_mod.running()
    )


def subtitle(workspace: Any) -> str:
    model = workspace.model("wiki") or {}
    if not model.get("project"):
        return "no project selected"
    if not model.get("built"):
        return f"{model['project']} · not built"
    written = len([p for p in model.get("pages") or [] if p["written"]])
    return f"{model['project']} · {written} page{'s' if written != 1 else ''} · {model.get('generated_at', '?')}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("wiki") or {}
    if not model.get("built"):
        return []
    out: list[tuple[str, str]] = []
    if model.get("stale"):
        out.append((f"STALE · {model.get('changed_total', 0)} FILES", "attention"))
    else:
        out.append(("CURRENT", "ok"))
    if not model.get("prose", True):
        out.append(("FACTS ONLY", "neutral"))
    if model.get("unverified_total"):
        out.append((f"{model['unverified_total']} UNVERIFIED REFS", "attention"))
    return out


def render(workspace: Any) -> None:
    model = workspace.model("wiki") or {}
    kit.error_strip(model.get("error"))

    def settled(task: Any) -> None:
        workspace.say(task_message(task))
        workspace.invalidate("wiki")
        workspace.tick()

    building = _building(model.get("project"))

    def build(*, prose: bool = True) -> None:
        """A background task, always. Nine pages is nine model calls, and the
        rest of the workspace has no reason to wait for them.

        One at a time per project. Two builds racing write the same `pages.json`
        and `manifest.json`, so the loser's pages are lost -- and they are lost
        *after* being paid for, which is the part that matters when a page is a
        model call. The buttons are disabled while one runs; this is the second
        check, because the poll that disables them is up to two seconds behind
        the click that starts one.
        """
        project = model.get("project")
        if not project:
            workspace.say("select a project first — a wiki is written about one")
            return
        if _building(project):
            workspace.say(f"a wiki build for {project} is already running — see the tasks window")
            return
        argv = ["tools.projwiki", "build", "--project", project, "--json"]
        if not prose:
            argv.append("--no-prose")
        start(f"wiki {project}", *argv, on_done=settled)
        workspace.say(
            f"building the wiki for {project} — see the tasks window"
            + ("" if prose else " (facts only, no model calls)")
        )
        workspace.invalidate("tasks")
        workspace.tick()

    if not model.get("built"):
        kit.empty(model.get("empty_message") or "No wiki has been built yet.", model.get("empty_fix"))
        if model.get("project"):
            with kit.pad():
                with kit.row("", gap=9):
                    kit.button(
                        "BUILDING…" if building else "▶ BUILD",
                        tone="primary",
                        disabled=building,
                        on_click=lambda: build(),
                    )
                    kit.button(
                        "FACTS ONLY",
                        tone="neutral",
                        disabled=building,
                        title="extract the tree, the spec, the symbols and the ledger without calling a model",
                        on_click=lambda: build(prose=False),
                    )
                kit.note(
                    "The wiki is half extracted and half written. The extracted half — the file "
                    "tree, the spec, what imports what, every function with its line number, the "
                    "predictions and the runs — is free and always true. The written half explains "
                    "that arrangement, one page per call, and every section cites the extracted "
                    "facts it rests on."
                )
        return

    with kit.row("grad-split", gap=0, align="stretch").style("min-height: 0; flex: 1 1 auto"):
        with kit.column("main grad-pad", gap=10):
            _header(model, build, building)
            if model.get("stale"):
                _stale(model)
            _page(model.get("page"))
            kit.hr()
            kit.label("ask about this project")
            kit.note(
                "A page is a document; a question is a conversation. Questions go to the agent "
                "with an @wiki mention, which can read the code this describes — and read the "
                "parts a page had to leave out."
            )
            _ask(workspace)
        _rail(workspace, model)


def _header(model: dict[str, Any], build: Any, building: bool) -> None:
    with kit.row("", gap=9).style("flex-wrap: wrap"):
        kit.button(
            "BUILDING…" if building else "↻ REBUILD",
            tone="primary",
            disabled=building,
            title="a build for this project is already running" if building else "",
            on_click=lambda: build(),
        )
        kit.button(
            "FACTS ONLY",
            tone="neutral",
            disabled=building,
            title="re-extract without calling a model",
            on_click=lambda: build(prose=False),
        )
        kit.spacer()
        if model.get("model"):
            kit.text(model["model"], "grad-caption", tag="span")
        kit.text(model.get("source_hash") or "", "grad-caption", tag="span")


def _stale(model: dict[str, Any]) -> None:
    with kit.el("div", "grad-card"):
        kit.text("STALE", "head attention")
        with kit.el("div", "body"):
            kit.text(
                f"this wiki was written from a different source tree: "
                f"{model.get('changed_total', 0)} file(s) differ. A wiki behind the code is "
                f"worse than none, because it is trusted.",
                "",
            )
            for path in model.get("changed") or []:
                kit.text(path, "grad-mono")


def _page(page: dict[str, Any] | None) -> None:
    """One page: its summary, its sections, and the citations under each.

    The refs are drawn *with* the section rather than collected at the bottom,
    because their job is to be checked while the sentence they support is still
    on screen. A ref that resolved to nothing is marked here rather than
    silently dropped -- a reader deciding how far to trust a paragraph is
    entitled to know which of its citations could not be found.
    """
    from nicegui import ui

    if page is None:
        kit.empty("Select a page.")
        return
    if page.get("error"):
        with kit.el("div", "grad-card"):
            kit.text("THIS PAGE WAS NOT WRITTEN", "head broken")
            with kit.el("div", "body"):
                kit.text(str(page["error"]), "grad-mono")
                kit.note(
                    "The rest of the wiki was written anyway. Rebuild to try this page again — "
                    "half a wiki whose gaps are visible is worth more than a whole one with an "
                    "invented page in it."
                )
        return
    if not page.get("sections"):
        kit.empty(
            "This page has been planned but not written — the last build ran facts-only.",
            "python -m tools.projwiki build --project <id> --json",
        )
        return

    kit.text(page.get("title") or "", "grad-serif").style("font-size: 28px; line-height: 1.15")
    if page.get("summary"):
        kit.text(page["summary"], "grad-caption").style("font-size: 14px; line-height: 1.6")

    for section in page["sections"]:
        with kit.el("div", "grad-wiki-section"):
            kit.text(section["heading"], "grad-serif").style("font-size: 19px; margin-bottom: 4px")
            ui.markdown(section["body"], extras=["fenced-code-blocks", "tables"]).classes("bubble")
            unverified = set(page.get("unverified_refs") or [])
            with kit.row("", gap=5).style("flex-wrap: wrap; margin-top: 5px"):
                for ref in section.get("refs") or []:
                    bad = str(ref).strip().strip("`") in unverified
                    kit.chip(
                        str(ref),
                        "attention" if bad else "neutral",
                    ).props(
                        'title="this citation matched nothing in the extracted facts"'
                        if bad
                        else ""
                    )

    if page.get("open_questions"):
        with kit.el("div", "grad-card"):
            kit.text("OPEN QUESTIONS", "head attention")
            with kit.el("div", "body"):
                kit.text(
                    "What the extracted facts did not settle. Named rather than written around.",
                    "grad-caption",
                )
                for question in page["open_questions"]:
                    kit.text(f"— {question}", "").style("margin-top: 6px; line-height: 1.55")


def _rail(workspace: Any, model: dict[str, Any]) -> None:
    with kit.column("rail grad-pad", gap=9):
        kit.label("pages")
        selected = model.get("selected")
        for index, page in enumerate(model.get("pages") or [], start=1):
            classes = "grad-row" + (" striped selected" if page["id"] == selected else "")
            row = kit.row(classes, gap=9, align="flex-start")
            row.on(
                "click",
                lambda _=None, pid=page["id"]: workspace.select("wiki.page", pid, window="wiki"),
            )
            with row:
                kit.chip(str(index), "solid" if page["written"] else "dashed")
                with kit.column("", gap=3).style("min-width: 0; flex: 1 1 auto"):
                    kit.text(page["title"], "grad-mono").style("font-size: 12px")
                    with kit.row("", gap=5).style("flex-wrap: wrap"):
                        kit.chip(KIND_LABEL.get(page["kind"], page["kind"]), "neutral")
                        if page.get("error"):
                            kit.chip("FAILED", "broken")
                        elif not page["written"]:
                            kit.chip("NOT WRITTEN", "dashed")
                        if page["unverified_refs"]:
                            kit.chip(f"{len(page['unverified_refs'])} UNVERIFIED", "attention")
        kit.hr()
        kit.text(model.get("output_dir") or "", "grad-caption")
        kit.note(
            "Every page was written against facts extracted from the code, not from memory: "
            "the file tree, the spec, the symbols with their line numbers, the expectations and "
            "the runs. `facts.json` beside these pages is exactly what each one was shown."
        )


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
            ui.textarea(placeholder="ask about this project's code")
            .props("borderless dense")
            .classes("field")
            .style("flex: 1 1 auto; border: var(--grad-border); background: var(--grad-paper-raised); padding: 0 8px")
        )
        entry.on("keydown.enter.prevent", send)
        kit.button("ASK ⏎", tone="primary", on_click=send)
