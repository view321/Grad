"""Window 4 — quota and budget.

Three ceilings, drawn separately because they fail differently: the rolling
5-hour session window (what stops the conversation), the project allocation
(what stops the research), and the machine's rolling GPU ceiling (what stops the
money). One combined bar would hide two of them.

The honesty note is part of the window, not a footnote. Anthropic exposes no
remaining-quota API and the Max 5x window is opaque, so the token meter is
self-measured usage against an assumed budget, and it says so on screen. GPU
dollars are different: those are real, and they count in-flight runs at their
estimates, because a job that has not been collected yet is not free.
"""

from __future__ import annotations

from typing import Any

from ui import kit


def subtitle(workspace: Any) -> str:
    model = workspace.model("quota") or {}
    session = model.get("session") or {}
    gpu = model.get("gpu") or {}
    return f"session {session.get('credits_usd', 0):.2f} · gpu ${gpu.get('total_usd', 0):.2f}"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("quota") or {}
    out: list[tuple[str, str]] = []
    if (model.get("session") or {}).get("fraction", 0) >= 0.9:
        out.append(("SESSION NEAR CAP", "attention"))
    project = model.get("project") or {}
    for resource in project.get("over_budget") or []:
        out.append((f"OVER {resource.upper()}", "broken"))
    return out


def render(workspace: Any) -> None:
    model = workspace.model("quota") or {}
    kit.error_strip(model.get("error"))
    session = model.get("session") or {}
    gpu = model.get("gpu") or {}

    with kit.pad():
        kit.label("5-hour window")
        used = float(session.get("credits_usd", 0.0))
        ceiling = float(session.get("ceiling_usd") or 0.0) or 1.0
        kit.bar(
            [
                (session.get("chat_usd", 0.0) / ceiling, "chat", "chat"),
                (session.get("tool_usd", 0.0) / ceiling, "tool", "tools"),
            ]
        )
        with kit.row("", gap=14).style("margin-top: 6px"):
            kit.text(f"${used:,.2f} / ${ceiling:,.2f}", "grad-mono", tag="span")
            kit.text(f"resets {session.get('resets_in', '—')}", "grad-caption", tag="span")
            kit.spacer()
            kit.text(f"{session.get('tokens', 0):,} tokens", "grad-caption", tag="span")

        kit.hr()

        # The four kinds, spelled out. The ceiling is charged one weighted
        # number, and the first question that number provokes is "why is it
        # twelve times what I expected" -- which is unanswerable without this
        # row and obvious with it. Cache reads are nearly always the largest
        # entry, and that is the point rather than a defect.
        kit.label(f"tokens · {model.get('days', 1)}d")
        counts = model.get("token_counts") or {}
        weights = model.get("token_weights") or {}
        if not counts:
            kit.text("nothing recorded in this window", "grad-caption")
        else:
            kit.kv([
                (label, f"{counts.get(field, 0):,}  × {weights.get(key, 1.0):g}")
                for field, key, label in (
                    ("input_tokens", "weight_input", "input"),
                    ("output_tokens", "weight_output", "output"),
                    ("cache_read_tokens", "weight_cache_read", "cache read"),
                    ("cache_write_tokens", "weight_cache_write", "cache write"),
                )
            ])
            with kit.row("", gap=14).style("margin-top: 6px"):
                kit.text(
                    f"{model.get('billable_tokens', 0):,} charged to the ceiling",
                    "grad-mono",
                    tag="span",
                )

        kit.hr()

        kit.label(f"spend today · {model.get('days', 1)}d")
        roles = model.get("roles") or []
        if not roles:
            kit.text("nothing recorded in this window", "grad-caption")
        for role in roles:
            biggest = max((r["credits_usd"] for r in roles), default=0.0) or 1.0
            with kit.row("", gap=9).style("margin: 5px 0"):
                kit.text(role["role"], "grad-mono", tag="span").style("min-width: 120px")
                kit.bar([(role["credits_usd"] / biggest, role["tone"], "")]).style("flex: 1 1 auto")
                kit.text(f"${role['credits_usd']:,.4f}", "grad-mono", tag="span")
                kit.text(f"{role['calls']} calls", "grad-caption", tag="span")

        kit.hr()

        kit.label(f"gpu · rolling {gpu.get('window_days', 30)}d")
        kit.bar(
            [
                (gpu.get("actual_usd", 0.0) / (gpu.get("monthly_usd") or 1.0), "ink", "collected"),
                (gpu.get("in_flight_usd", 0.0) / (gpu.get("monthly_usd") or 1.0), "tool", "in flight"),
            ]
        )
        with kit.row("", gap=14).style("margin-top: 6px"):
            kit.text(
                f"${gpu.get('total_usd', 0):,.2f} / ${gpu.get('monthly_usd', 0):,.2f}",
                "grad-mono",
                tag="span",
            )
            kit.text(f"${gpu.get('in_flight_usd', 0):,.2f} uncollected", "grad-caption", tag="span")

        project = model.get("project")
        if project:
            kit.hr()
            kit.label(f"project · {project.get('project')}")
            for resource, node in (project.get("resources") or {}).items():
                ceiling_value = node.get("ceiling")
                with kit.row("", gap=9).style("margin: 5px 0"):
                    kit.text(resource, "grad-mono", tag="span").style("min-width: 120px")
                    if ceiling_value is None:
                        # "unbounded" and "a very large number" must not look
                        # the same in a meter.
                        kit.text("no ceiling set", "grad-caption", tag="span")
                    else:
                        kit.bar(
                            [(node.get("fraction") or 0.0, "broken" if node.get("over") else "ink", "")]
                        ).style("flex: 1 1 auto")
                        kit.text(
                            f"{node.get('spent'):,.4g} / {ceiling_value:,.4g}", "grad-mono", tag="span"
                        )

        kit.hr()
        kit.note(model.get("honesty", ""))
