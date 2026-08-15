"""Window 11 — the LaTeX paper editor.

Three panes: outline, source, preview.

The design's mock invents `\\gradcite{run-…}` and `\\gradexp{exp-…}`. The macro
that already exists in `core/report.py` is `\\gradnum{key}`, resolving through
`claims.json` to a `(run_id, quantity)` **and its recorded value** -- strictly
stronger than what the mock drew, because it catches the failure a citation
checker misses: a citation that points at the right run and prints the wrong
number. So this window renders the real macro, and the outline's warning box
counts findings from `report.check_claims`, `check_citations` and `check_latex`
rather than from a regex of its own.

This is a viewer, not an editor. `main.tex` is written by `tools/report.py`
draft/write and edited in a text editor; putting a text area over a file that
another process is generating is how you lose a paragraph.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.tasks import envelope_message, run_tool, start, task_message


def subtitle(workspace: Any) -> str:
    model = workspace.model("editor") or {}
    if not model.get("exists"):
        return "no draft"
    return f"{model.get('project')} · {len(model.get('lines') or [])} lines"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("editor") or {}
    blocking = model.get("blocking", 0)
    if blocking:
        return [(f"{blocking} UNBOUND", "broken")]
    return [("BINDINGS CLEAN", "ok")] if model.get("exists") else []


def render(workspace: Any) -> None:
    model = workspace.model("editor") or {}
    kit.error_strip(model.get("error"))
    if not model.get("exists"):
        kit.empty(
            "No draft for this project yet."
            if model.get("project")
            else "No project selected — a report belongs to one.",
            model.get("empty_fix"),
        )
        return

    def settled(task: Any) -> None:
        workspace.say(task_message(task))
        workspace.invalidate("editor")
        workspace.tick()

    def build() -> None:
        """LaTeX runs more than once and can pull fonts, so this is minutes."""
        start(
            f"build {model['project']}",
            "tools.report", "build", "--project", str(model["project"]), "--json",
            on_done=settled,
        )
        workspace.say("building the PDF — see the tasks window")
        workspace.invalidate("tasks")
        workspace.tick()

    async def check() -> None:
        payload = await run_tool("tools.report", "check", "--project", str(model["project"]), "--json")
        workspace.say(envelope_message(payload))
        workspace.invalidate("editor")
        workspace.tick()

    with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
        kit.text(model["tex_path"], "grad-caption", tag="span")
        kit.spacer()
        kit.button("✓ CHECK", tone="neutral", on_click=check)
        kit.button("BUILD PDF", tone="primary", on_click=build,
                   disabled=bool(model.get("blocking")),
                   title="blocked while any asserted number is unbound")

    with kit.row("", gap=0, align="stretch").style("min-height: 0; flex: 1 1 auto"):
        _outline(model)
        _source(model)
        _preview(model)


def _outline(model: dict[str, Any]) -> None:
    with kit.column("grad-pad", gap=6).style(
        "flex: 0 0 190px; background: var(--grad-paper-sunk); "
        "border-right: var(--grad-border); overflow-y: auto"
    ):
        kit.label("outline")
        for section in model.get("outline") or []:
            with kit.row("grad-row", gap=6):
                kit.text(section["title"], "", tag="span")
                kit.spacer()
                kit.caption(section["line"])

        findings = model.get("findings") or []
        if findings:
            with kit.el("div").style(
                "border: 2px solid var(--grad-broken); background: var(--grad-broken-tint); "
                "padding: 9px; margin-top: 10px"
            ):
                kit.text(model.get("warning") or "", "").style(
                    "font-size: 12px; color: var(--grad-broken-ink)"
                )


def _source(model: dict[str, Any]) -> None:
    from ui import models as models_mod

    flagged = model.get("flagged_lines") or set()
    with kit.column("", gap=0).style("flex: 1 1 auto; min-width: 0; overflow: auto"):
        with kit.el("pre", "grad-pre").style("border: 0; background: var(--grad-paper)"):
            for number, line in enumerate(model.get("lines") or [], start=1):
                row = kit.el("div")
                if number in flagged:
                    row.style("background: var(--grad-broken-tint)")
                with row:
                    kit.text(f"{number:>4} ", "", tag="span").style(
                        "opacity: 0.35; user-select: none"
                    )
                    for span in models_mod.highlight_tex(line):
                        element = kit.text(span["text"], "", tag="span")
                        if span["kind"] == "gradnum":
                            element.style(
                                "background: var(--grad-attention); "
                                "border: 1px solid var(--grad-ink); padding: 0 2px"
                            )
                        elif span["kind"] == "command":
                            element.style("color: var(--grad-link)")
                        elif span["kind"] == "comment":
                            element.style("color: var(--grad-muted-2)")


def _preview(model: dict[str, Any]) -> None:
    with kit.column("grad-pad", gap=9).style(
        "flex: 0 0 420px; background: var(--grad-paper-raised); "
        "border-left: var(--grad-border); overflow-y: auto"
    ):
        kit.label("findings")
        findings = model.get("findings") or []
        if not findings:
            kit.note("every asserted number resolves to a run, a quantity and a matching value")
        for finding in findings:
            with kit.el("div", "grad-card"):
                with kit.row("head broken", gap=9):
                    kit.text(str(finding.get("rule", "?")).upper(), "", tag="span")
                    kit.spacer()
                    if finding.get("line"):
                        kit.text(f"line {finding['line']}", "", tag="span")
                with kit.el("div", "body"):
                    kit.text(finding.get("problem") or "", "")
                    if finding.get("fix"):
                        kit.pre(finding["fix"])

        if model.get("cited_runs"):
            kit.hr()
            kit.label("runs this paper depends on")
            for run_id in model["cited_runs"]:
                kit.chip(run_id, "outline")

        kit.hr()
        kit.label("pdf")
        kit.text(model["pdf_path"], "grad-caption")
        kit.chip("BUILT" if model.get("pdf_exists") else "NOT BUILT",
                 "ok" if model.get("pdf_exists") else "dashed")
