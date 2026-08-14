"""Window 8 — Evolve (ShinkaEvolve campaigns).

Three panes: population stats, the lineage chart, the champion diff.

The lineage is drawn from `ledger/candidates.jsonl` rather than from a Shinka
export, so the window works whether or not the optional dependency is installed
-- the campaign bookkeeping and the budget gate, the parts that matter, are ours
either way.

`top` is a top-K rather than the argmax, on purpose and all the way through to
this window: a search optimising a scalar will find the bug in the metric, and a
UI that only ever shows you the winner is a UI that hides that.
"""

from __future__ import annotations

from typing import Any

from ui import kit
from ui.state import envelope_message, run_tool


def subtitle(workspace: Any) -> str:
    model = workspace.model("evolve") or {}
    campaign = model.get("campaign")
    if not campaign:
        return "no campaigns"
    return f"{campaign['id']} · gen {campaign.get('generations_run', 0)} · {campaign.get('candidates', 0)} candidates"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("evolve") or {}
    campaign = model.get("campaign") or {}
    if not campaign.get("running"):
        return []
    if campaign.get("halt_requested"):
        return [("HALTING", "attention")]
    return [("EVOLVING", "ok")]


def render(workspace: Any) -> None:
    from nicegui import ui

    model = workspace.model("evolve") or {}
    kit.error_strip(model.get("error"))
    campaign = model.get("campaign")
    if not campaign:
        kit.empty("No campaigns yet.", model.get("empty_fix"))
        return

    if len(model.get("campaigns") or []) > 1:
        with kit.row("grad-pad", gap=9).style("border-bottom: var(--grad-border)"):
            ui.select(
                model["campaigns"],
                value=campaign["id"],
                on_change=lambda e: workspace.select("evolve.campaign", e.value),
            ).props("dense borderless").style("min-width: 220px")
            kit.spacer()

    with kit.row("", gap=0, align="stretch").style("min-height: 0; flex: 1 1 auto"):
        _stats(workspace, campaign)
        _lineage(campaign)
        _champion(workspace, campaign)


def _stats(workspace: Any, campaign: dict[str, Any]) -> None:
    spend = campaign.get("spend") or {}
    with kit.column("grad-pad", gap=10).style(
        "flex: 0 0 300px; background: var(--grad-paper-sunk); "
        "border-right: var(--grad-border); overflow-y: auto"
    ):
        kit.label("population")
        kit.kv(
            [
                ("status", campaign.get("status")),
                ("generations", campaign.get("generations_run")),
                ("candidates", campaign.get("candidates")),
                ("islands", campaign.get("islands")),
                ("migrations", campaign.get("migrations")),
                ("novelty", campaign.get("novelty")),
                ("project", campaign.get("project")),
                ("spend", f"${float(spend.get('cost_usd', 0.0)):,.4f}"),
            ]
        )
        kit.note(f"objective — {campaign.get('objective')}")
        if campaign.get("running"):
            async def halt() -> None:
                workspace.say(f"requesting halt of {campaign['id']} …")
                payload = await run_tool(
                    "tools.evolve", "halt",
                    "--campaign", campaign["id"],
                    "--reason", "halted from the workspace",
                    "--json",
                )
                workspace.say(envelope_message(payload))
                workspace.invalidate("evolve")
                workspace.tick()

            if campaign.get("halt_requested"):
                # The request is in the ledger but the loop has not reached a
                # boundary yet. Saying so beats a button that looks unpressed.
                kit.chip("HALT REQUESTED", "attention")
                kit.caption("stops before the next generation")
            else:
                kit.button("■ HALT", tone="danger", on_click=halt,
                           title="stop at the next generation boundary, "
                                 "with every candidate collected")


def _lineage(campaign: dict[str, Any]) -> None:
    bars = campaign.get("bars") or []
    with kit.column("grad-pad", gap=6).style("flex: 1 1 auto; min-width: 0; overflow: auto"):
        kit.label("lineage")
        if not bars:
            kit.text("no scored candidates yet", "grad-caption")
            return
        with kit.el("div", "grad-lineage"):
            for bar in bars:
                element = kit.el("div", f"bar {bar['tone'] if bar['tone'] != 'ordinary' else ''}".strip())
                element.style(f"height: {bar.get('height', 0.5) * 100:.1f}%")
                element.props(f'title="gen {bar.get("generation")} · {bar["score"]:.6g} · {bar.get("id")}"')
        with kit.row("", gap=9):
            kit.caption(f"gen {bars[0].get('generation')}")
            kit.spacer()
            kit.caption(f"gen {bars[-1].get('generation')}")
        with kit.row("", gap=9):
            kit.chip("NEW BEST", "attention")
            kit.chip("CHAMPION", "ok")


def _champion(workspace: Any, campaign: dict[str, Any]) -> None:
    champion = campaign.get("champion")
    with kit.column("grad-pad", gap=9).style(
        "flex: 0 0 420px; background: var(--grad-paper-raised); "
        "border-left: var(--grad-border); overflow-y: auto"
    ):
        kit.label("champion")
        if not champion:
            kit.text("nothing has scored yet", "grad-caption")
            return

        with kit.row("", gap=9):
            kit.text(str(champion.get("id")), "grad-mono", tag="span")
            if campaign.get("delta") is not None:
                kit.chip(f"Δ {campaign['delta']:+.6g}", "ok")

        for row in campaign.get("top") or []:
            with kit.row("grad-row", gap=9):
                kit.text(str(row.get("id")), "grad-mono", tag="span")
                kit.caption(f"gen {row.get('generation')}")
                kit.spacer()
                score = row.get("score")
                kit.text(f"{score:.6g}" if isinstance(score, (int, float)) else "—",
                         "grad-mono", tag="span")

        diff = champion.get("diff") or champion.get("patch")
        if diff:
            kit.hr()
            with kit.el("div", "grad-diff"):
                for line in str(diff).splitlines():
                    kind = "meta"
                    if line.startswith("+") and not line.startswith("+++"):
                        kind = "add"
                    elif line.startswith("-") and not line.startswith("---"):
                        kind = "del"
                    elif not line.startswith(("+", "-", "@@")):
                        kind = "same"
                    kit.text(line, kind)
        else:
            kit.note(
                "no diff recorded for this candidate — `tools.evolve promote` writes the "
                "promoted source, and the patch is against the task directory"
            )

        async def promote() -> None:
            workspace.say(f"promoting {champion.get('id')} …")
            payload = await run_tool(
                "tools.evolve", "promote",
                "--campaign", campaign["id"],
                "--candidate", str(champion.get("id")),
                "--json",
            )
            workspace.say(envelope_message(payload))
            workspace.invalidate("evolve")
            workspace.tick()

        with kit.row("", gap=6).style("margin-top: 10px"):
            kit.button("✓ ADOPT INTO MAIN", tone="primary", on_click=promote)
