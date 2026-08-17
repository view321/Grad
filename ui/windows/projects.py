"""Window 13 — projects: what the research is divided into, and what bounds it.

A project is the unit three separate requirements turned out to share (§15): HF
payer attribution, the bound on an evolutionary campaign, and a budget for a
piece of research. Every run, every expectation and every report is keyed by
one. It was also, until this window, the only first-class concept in the app
with no window — it lived in a section of the `project ▾` dialog, between a
folder picker and a credentials panel.

Two things the dialog could not do, and this exists for both:

**Ceilings for a project you are not on.** The menu's raise controls addressed
the *selected* project only, so reading what bounded any other one meant
switching to it first — which charges nothing, and reloads every window in the
app to answer a question about a number.

**Saying that a project bounds nothing.** A project with no ceilings passes
every gate that reads one, silently. `tools/budget.py` returns a warning saying
so at the moment of creation and nothing carried it any further; here it is a
chip on the row it is true of, for as long as it stays true.

Closing is offered and deleting is not, because nothing here deletes: `close`
appends an event and the records stay (`core/budget.py`).
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.models import CEILINGS


def subtitle(workspace: Any) -> str:
    model = workspace.model("projects") or {}
    current = model.get("current_project") or "none selected"
    return f"{model.get('open_count', 0)} open · on {current}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("projects") or {}
    out: list[tuple[str, str]] = []
    unbounded = model.get("unbounded") or []
    if unbounded:
        out.append((f"{len(unbounded)} UNBOUNDED", "attention"))
    over = [r["id"] for r in model.get("rows") or [] if r.get("over_budget")]
    if over:
        out.append((f"{len(over)} OVER BUDGET", "broken"))
    return out


def render(workspace: Any) -> None:
    model = workspace.model("projects") or {}
    kit.error_strip(model.get("error"))

    rows = model.get("rows") or []
    if not rows:
        kit.empty(
            "No projects in this folder yet.",
            "python -m tools.budget new --id proj-scaling-w2 --title '...' --use --json",
        )
        _new_project(workspace, model)
        return

    for row in rows:
        _project(workspace, model, row)
    _new_project(workspace, model)


def _project(workspace: Any, model: dict[str, Any], row: dict[str, Any]) -> None:
    """One project: who it is, what it has spent, and the two controls that
    change either."""
    closed = row["status"] == "closed"
    accent = "broken" if row.get("over_budget") else ("attention" if row["unbounded"] else "")

    with kit.el("div", "grad-card").style("margin: 9px"):
        with kit.row(f"head {accent}".strip(), gap=9):
            kit.text(row["id"], "", tag="span").style("font-weight: 700")
            if row["current"]:
                kit.chip("IN USE", "ok")
            if closed:
                kit.chip("CLOSED", "neutral")
            if row["unbounded"] and not closed:
                kit.chip("UNBOUNDED", "attention")
            for resource in row.get("over_budget") or []:
                kit.chip(f"OVER {resource.replace('_', ' ').upper()}", "broken")
            kit.spacer()
            kit.text(row["created"], "grad-caption", tag="span")

        with kit.el("div", "body"):
            kit.error_strip(row.get("error"))
            if row["title"]:
                kit.text(row["title"], "").style("font-size: 13.5px; margin-bottom: 8px")

            kit.kv(
                [
                    ("payer", row["payer"] or "—"),
                    ("runs", row["run_count"]),
                    ("ceiling raises", row["raise_count"] or "none"),
                    ("memory", _memory_line(row["memory"])),
                ]
            )

            _ceilings(workspace, row)
            _models(workspace, model, row)

            with kit.row("", gap=6).style("margin-top: 10px"):
                kit.button(
                    "IN USE" if row["current"] else "USE",
                    tone="active" if row["current"] else "primary",
                    disabled=row["current"] or closed,
                    title=(
                        "closed projects cannot be selected"
                        if closed
                        else "charge runs and tokens to this project"
                    ),
                    on_click=lambda _=None, pid=row["id"]: workspace.spawn(
                        workspace.use_project(pid), "project switch"
                    ),
                )
                kit.spacer()
                kit.button(
                    "CLOSE",
                    tone="neutral",
                    disabled=closed,
                    title="append a close event — nothing is deleted, and the records stay readable",
                    on_click=lambda _=None, pid=row["id"]: workspace.spawn(
                        workspace.run_and_reload("tools.budget", "close", pid, "--json"),
                        "project close",
                    ),
                )


def _memory_line(memory: dict[str, Any]) -> str:
    """"Scaffolded but empty" and "never scaffolded" are different answers."""
    if memory.get("error"):
        return "unreadable"
    if not memory.get("scaffolded"):
        return "not scaffolded — python -m tools.project init"
    missing = memory.get("missing") or []
    present = len(memory.get("present") or [])
    return f"{present} file(s)" + (f" · {len(missing)} missing" if missing else "")


def _ceilings(workspace: Any, row: dict[str, Any]) -> None:
    """The three ceilings, and the one control that moves them.

    Drawn for every project rather than only the selected one, which is the
    reason this window exists. A raise is a logged event and not a setting
    (`core/budget.py`), so the reason field is offered here — the CLI defaults it
    to empty and this does not force one, but a ceiling that moved without a
    reason is unarguable with six months later.
    """
    from nicegui import ui

    kit.label("ceilings").style("margin-top: 10px")
    fields: dict[str, Any] = {}

    for ceiling in row["ceilings"]:
        with kit.row("", gap=9).style("margin: 5px 0"):
            kit.text(ceiling["caption"], "grad-mono", tag="span").style("min-width: 84px")
            if not ceiling["set"]:
                # "unbounded" and "nothing spent" must not look the same in a
                # meter, so an unset ceiling gets no bar at all.
                kit.text(ceiling["label"], "grad-caption", tag="span").style("flex: 1 1 auto")
            else:
                kit.bar(
                    [(ceiling["fraction"] or 0.0, "broken" if ceiling["over"] else "base", "")]
                ).style("flex: 1 1 auto")
                kit.text(ceiling["label"], "grad-mono", tag="span")

            field = (
                ui.input(placeholder=ceiling["caption"])
                .props("borderless dense")
                .classes("field")
                .style("flex: 0 0 110px; padding: 0 8px")
            )
            field.props(f'title="{kit.attr(ceiling["hint"])}"')
            fields[ceiling["flag"]] = field

    with kit.row("", gap=6).style("margin-top: 6px"):
        reason = (
            ui.input(placeholder="why it moved — optional, and it ages badly without one")
            .props("borderless dense")
            .classes("field")
            .style("flex: 1 1 auto; padding: 0 8px")
        )

        def raise_them(_=None, pid: str = row["id"]) -> None:
            # `--project <id>` as a flag, not a positional: `tools.budget raise`
            # takes it as a flag, and passing it positionally is what made this
            # control dead on every click for a release. `tests/test_ui_argv.py`
            # runs this argv through the real parser.
            argv = ["tools.budget", "raise", "--project", pid]
            base = len(argv)
            for flag, field in fields.items():
                if (field.value or "").strip():
                    argv += [f"--{flag}", str(field.value).strip()]
            if len(argv) == base:
                workspace.say("no ceiling given — fill one of the three fields")
                return
            if (reason.value or "").strip():
                argv += ["--reason", str(reason.value).strip()]
            workspace.spawn(workspace.run_and_reload(*argv, "--json"), "ceiling raise")

        kit.button(
            "RAISE",
            tone="primary",
            title="a logged event — leave a field blank to leave that ceiling alone",
            on_click=raise_them,
        )


def _models(workspace: Any, model: dict[str, Any], row: dict[str, Any]) -> None:
    """What this project overrides about how it is run.

    Collapsed until asked for, because six roles times every project is a window
    nobody can read, and the common case is a project that overrides nothing.
    The expanded project id lives in `workspace.selection` rather than in a
    closure — this window is rebuilt whenever a run lands.

    Beside each role is what it *would* be without the override, and that is the
    workspace's answer rather than this project's: the window draws every
    project and only one of them is selected, so for all the others the project
    layer in effect belongs to somebody else.
    """
    from nicegui import ui

    expanded = workspace.selection.get("projects.models") == row["id"]
    overrides = [m for m in row["models"] if m["override"]]

    with kit.row("", gap=6).style("margin-top: 10px"):
        kit.label("models").style("min-width: 84px")
        if not overrides:
            kit.text("workspace defaults", "grad-caption", tag="span")
        for entry in overrides:
            kit.chip(f"{entry['role']} → {entry['override']}", "ok")
        if row["backend"]:
            kit.chip(f"backend {row['backend']}", "outline")
        kit.spacer()
        kit.button(
            "HIDE" if expanded else "CHANGE",
            tone="neutral",
            on_click=lambda _=None, pid=row["id"]: workspace.select(
                "projects.models", None if expanded else pid, window="projects"
            ),
        )

    if not expanded:
        return

    for entry in row["models"]:
        with kit.row("", gap=6).style("margin: 4px 0"):
            kit.text(entry["role"], "grad-mono", tag="span").style("min-width: 84px")
            kit.text(
                entry["effective"],
                "grad-mono" if entry["override"] else "grad-caption",
                tag="span",
            ).style("min-width: 150px")
            if not entry["override"]:
                kit.text(
                    "from the workspace", "grad-caption", tag="span",
                    style="flex: 1 1 auto; min-width: 0",
                )
            else:
                kit.text(
                    f"workspace says {entry['workspace']}", "grad-caption", tag="span",
                    style="flex: 1 1 auto; min-width: 0",
                )
            field = (
                ui.input(placeholder="model id")
                .props("borderless dense")
                .classes("field")
                .style("flex: 0 0 170px; padding: 0 8px")
            )
            def set_model(
                _=None, pid: str = row["id"], role: str = entry["role"], f: Any = field
            ) -> None:
                # An empty field means "I clicked the wrong button", not "clear
                # it": `configure_project` reads an empty model for a named role
                # as `--clear`, so SET on a blank field silently dropped the
                # override that ✕ is there to drop deliberately.
                chosen = (f.value or "").strip()
                if not chosen:
                    workspace.say("no model given — type an id, or use ✕ to drop the override")
                    return
                workspace.spawn(
                    workspace.configure_project(pid, role=role, model=chosen), "project model"
                )

            kit.button("SET", tone="primary", on_click=set_model)
            kit.button(
                "✕",
                tone="neutral",
                disabled=not entry["override"],
                title="drop the override — the role resolves as the workspace's does",
                on_click=lambda _=None, pid=row["id"], r=entry["role"]: workspace.spawn(
                    workspace.configure_project(pid, role=r, model=""), "project model"
                ),
            )

    with kit.row("", gap=6).style("margin-top: 6px; flex-wrap: wrap"):
        kit.text("backend", "grad-caption", tag="span").style("min-width: 84px")
        for backend in model.get("known_backends") or []:
            kit.button(
                backend,
                tone="active" if backend == row["backend"] else "neutral",
                disabled=backend == row["backend"],
                on_click=lambda _=None, pid=row["id"], b=backend: workspace.spawn(
                    workspace.configure_project(pid, backend=b), "project backend"
                ),
            )
    kit.text(
        "a preference, not a restriction — --remote still names one per campaign, and a spec's "
        "[target] wins over both",
        "grad-caption",
    )


def _new_project(workspace: Any, model: dict[str, Any]) -> None:
    """Create, with the ceilings in the same form.

    This is the project half of setup, and it is the whole of it: everything
    else a wizard would ask -- the token, the six model roles, which backends
    exist -- is a fact about this machine, answered once in the setup window and
    not re-asked per project.

    The ceilings are here rather than in a second step because of what the old
    form did without them. It created with none and said "set them below once it
    is selected" in a caption under the button, so the common path produced a
    project that bounds nothing and every gate that reads a ceiling passed
    silently on it.
    """
    from nicegui import ui

    with kit.el("div", "grad-card").style("margin: 9px"):
        kit.text("NEW PROJECT", "head")
        with kit.el("div", "body"):
            with kit.row("", gap=6):
                project_id = (
                    ui.input(placeholder="id, e.g. proj-scaling-w2")
                    .props("borderless dense")
                    .classes("field")
                    .style("flex: 0 0 220px; padding: 0 8px")
                )
                title = (
                    ui.input(placeholder="what this research is")
                    .props("borderless dense")
                    .classes("field")
                    .style("flex: 1 1 auto; padding: 0 8px")
                )

            kit.label("ceilings").style("margin-top: 10px")
            fields: dict[str, Any] = {}
            with kit.row("", gap=6):
                for _resource, flag, caption_text, hint in CEILINGS:
                    field = (
                        ui.input(placeholder=caption_text)
                        .props("borderless dense")
                        .classes("field")
                        .style("flex: 1 1 0; padding: 0 8px")
                    )
                    field.props(f'title="{kit.attr(hint)}"')
                    fields[flag] = field
                payer = (
                    ui.input(placeholder="payer, e.g. hf:myorg")
                    .props("borderless dense")
                    .classes("field")
                    .style("flex: 1 1 0; padding: 0 8px")
                )
                payer.props(
                    'title="who pays. hf:&lt;org&gt; attributes HF jobs to that organisation"'
                )

            with kit.row("", gap=6).style("margin-top: 10px"):
                kit.button(
                    "CREATE",
                    tone="primary",
                    on_click=lambda: workspace.spawn(
                        workspace.create_project(
                            project_id.value or "",
                            title.value or "",
                            ceilings={f: (i.value or "") for f, i in fields.items()},
                            payer=payer.value or "",
                        ),
                        "project create",
                    ),
                )
                kit.text(
                    "set here, they are what the project was allowed from the start — a raise "
                    "afterwards records a ceiling that moved, which is a different claim",
                    "grad-caption",
                    style="flex: 1 1 auto; min-width: 0",
                )

            if model.get("needs_setup"):
                # The machine half, pointed at rather than repeated. It opens on
                # its own after a create, too -- but a form that is about to
                # produce an unrunnable project should say so before the click,
                # not after it.
                kit.note(
                    "This machine has no subscription credentials yet, so a new project will have "
                    "nothing to run. Setup asks for those once, not per project."
                )
                kit.button(
                    "OPEN SETUP",
                    tone="neutral",
                    on_click=lambda: workspace.open("setup"),
                )
