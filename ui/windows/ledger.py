"""Window 3 — the ledger: expectations against outcomes.

The highest-value visual in the app, and the reason it is a band strip rather
than a table: predicted range as a band, observed value as a marker, in-range or
not obvious at a glance. A number in a cell requires the reader to do the
comparison; a tick outside a block does the comparison for them.

Unjudged deviations are flagged rather than left to accumulate quietly, which is
the whole argument of §7. A deviation is unjudged when it is not confirmed in
range *and* carries no verdict -- `in_range` is `False` for a numeric miss and
`None` for the cases no program can settle (a relational prediction, a
non-numeric result), and both need a human.

Four states, and the fourth is the other half of that argument. `open`, `met`
and `broken` are what the arithmetic can say; `judged` is what a verdict says,
and without it every expectation the machine could not settle stayed `open`
after a human had settled it -- the one window whose job is to show that the
loop closed, showing that it never did.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.tasks import envelope_message, run_tool

FILTERS = (("open", "OPEN"), ("met", "MET"), ("judged", "JUDGED"), ("broken", "BROKEN"))


def subtitle(workspace: Any) -> str:
    model = workspace.model("ledger") or {}
    counts = model.get("counts") or {}
    return (
        f"{counts.get('open', 0)} open · {counts.get('met', 0)} met · "
        f"{counts.get('judged', 0)} judged · {counts.get('broken', 0)} broken"
    )


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("ledger") or {}
    unjudged = len([e for e in model.get("entries", []) if e.get("unjudged")])
    return [(f"{unjudged} UNJUDGED", "attention")] if unjudged else []


def render(workspace: Any) -> None:
    model = workspace.model("ledger") or {}
    kit.error_strip(model.get("error"))
    entries = model.get("entries") or []
    if not entries:
        kit.empty("No expectations registered yet.", model.get("empty_fix"))
        return

    active = workspace.selection.get("ledger.filter", "open")
    counts = model.get("counts") or {}

    with kit.row("grad-pad", gap=6).style("border-bottom: var(--grad-border)"):
        for key, caption in FILTERS:
            kit.button(
                f"{caption} {counts.get(key, 0)}",
                tone="active" if key == active else "neutral",
                on_click=lambda _=None, k=key: workspace.select("ledger.filter", k),
            )
        kit.spacer()
        kit.text(f"{model.get('total', 0)} total", "grad-caption")

    visible = [e for e in entries if e["state"] == active]
    if not visible:
        kit.empty(f"Nothing is {active}.")
        return

    for entry in visible:
        _entry(workspace, entry)


def _entry(workspace: Any, entry: dict[str, Any]) -> None:
    with kit.column(f"grad-row striped {entry['accent']}", gap=0):
        with kit.row("", gap=9):
            kit.text(entry["id"], "grad-mono", tag="span")
            kit.chip(entry["state"].upper(), entry["accent"])
            if entry.get("unjudged"):
                kit.chip("UNJUDGED", "attention")
            kit.spacer()
            kit.text(entry.get("at") or "", "grad-caption", tag="span")

        kit.text(entry["claim"], "").style("font-size: 13.5px; margin: 6px 0")

        if entry["state"] == "open" or entry.get("band"):
            kit.band_strip(
                entry.get("band"),
                reason=(
                    "no result yet — the band is drawn when the run is collected"
                    if entry["state"] == "open" and not entry.get("band")
                    else ""
                ),
            )

        if entry.get("verdict"):
            # What closed it, in the words of whoever closed it. The ledger is
            # read months later by someone who no longer remembers which of the
            # three it was, which is the same reason `_verdict` refuses an empty
            # note when writing one.
            reason = entry.get("verdict_note") or ""
            kit.note(f"verdict {entry['verdict']}{' — ' + reason if reason else ''}")

        if entry.get("comparability"):
            kit.note(f"comparability — {entry['comparability']}")

        for basis in entry.get("basis") or []:
            kit.text(
                f"· {basis.get('paper')} — {basis.get('locator')} = {basis.get('value')} "
                f"({basis.get('conditions')})",
                "grad-caption",
            )

        if entry.get("runs"):
            with kit.row("", gap=6).style("margin-top: 6px"):
                kit.text("runs", "grad-caption", tag="span")
                for run_id in entry["runs"][:6]:
                    kit.chip(run_id, "outline")

        if entry.get("unjudged"):
            _verdict(workspace, entry)


#: The three verdicts `tools.ledger verdict` accepts, and what each one claims.
#: Kept in the same order as the CLI's `choices` so the two cannot drift apart
#: into a UI that offers a judgement the ledger will not take.
VERDICTS = (
    ("bug", "✎ BUG", "ok", "the deviation is ours — the code was wrong"),
    ("real", "⚠ REAL", "danger", "the deviation is the world's — the prediction was wrong"),
    ("inconclusive", "? INCONCLUSIVE", "neutral", "neither, yet — say why in the note"),
)


def _verdict(workspace: Any, entry: dict[str, Any]) -> None:
    """The one field a program does not fill in.

    This was a copyable command, on the argument that a ledger write belongs to
    the CLI. It still does -- the buttons run exactly that command -- but the
    reason to type it by hand was never a good one: `report check` refuses while
    any cited run carries an unjudged deviation, so this is on the critical path
    between a result and a paper, and it was the only step on that path that
    required leaving the app.

    The note is not optional here even though the CLI defaults it to empty. A
    verdict with no reasoning ages badly, and the ledger is read months later by
    someone who no longer remembers which of the three it was.
    """
    from nicegui import ui

    run_id = (entry.get("runs") or [None])[-1]
    quantity = entry.get("quantity")
    if not run_id or not quantity:
        # `verdict` is keyed on a run and a quantity. Without both there is
        # nothing to write against, and a button that cannot work should say so
        # rather than fail on click.
        kit.note("no run and quantity to judge against yet")
        return

    async def judge(verdict: str) -> None:
        reason = (note.value or "").strip()
        if not reason:
            workspace.say("say why — a verdict with no reasoning ages badly")
            return
        note.value = ""
        payload = await run_tool(
            "tools.ledger", "verdict", str(run_id),
            "--quantity", str(quantity),
            "--verdict", verdict,
            "--note", reason,
            "--json",
        )
        workspace.say(f"{run_id} {quantity}: {envelope_message(payload)}")
        workspace.invalidate("ledger")
        workspace.tick()

    with kit.el("div", "grad-card"):
        with kit.row("head attention", gap=9):
            kit.text("UNJUDGED DEVIATION", "", tag="span")
            kit.spacer()
            kit.text(f"{run_id} · {quantity}", "", tag="span")
        with kit.el("div", "body"):
            note = (
                ui.input(placeholder="why — this is what the ledger is read for later")
                .props("borderless dense")
                .classes("field")
                .style("width: 100%; padding: 0 8px")
            )
            with kit.row("", gap=6).style("margin-top: 8px"):
                for verdict, caption, tone, hint in VERDICTS:
                    kit.button(
                        caption,
                        tone=tone,
                        title=hint,
                        on_click=lambda _=None, v=verdict: workspace.spawn(judge(v), "verdict"),
                    )
