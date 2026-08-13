"""Widget 1: the preflight panel (HANDOFF §10).

    "When a submission is blocked, the failing check, its output, and its
     error.fix command are one click away. A gate is only tolerable if it
     explains itself."
"""

from __future__ import annotations

from typing import Any

from core import jsonl, paths


def _records() -> list[dict[str, Any]]:
    out = []
    for path in sorted(paths.preflight_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        rec = jsonl.read_json(path)
        if rec:
            out.append(rec)
    return out


def preflight_panel() -> None:
    from nicegui import ui

    records = _records()
    if not records:
        ui.label("No preflight records yet.").classes("text-sm opacity-60")
        ui.code("python -m tools.preflight run --spec <spec> --json", language="bash")
        return

    for record in records[:20]:
        checks: dict[str, Any] = record.get("checks", {})
        failing = [n for n, r in checks.items() if r.get("ok") is False]
        passing = [n for n, r in checks.items() if r.get("ok")]
        colour = "text-red-400" if failing else "text-emerald-400"
        title = f"{record.get('submission_hash', '?')} — {len(passing)} passing, {len(failing)} failing"

        with ui.expansion(title, value=bool(failing)).classes("w-full"):
            ui.label(str(record.get("spec", ""))).classes("text-xs opacity-60")
            ui.label(f"verified {record.get('verified_at', '?')}").classes("text-xs opacity-60")

            for name, result in checks.items():
                ok = result.get("ok")
                icon = "check_circle" if ok else ("cancel" if ok is False else "help")
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.icon(icon).classes(colour if ok is not None else "opacity-50")
                    ui.label(name).classes("font-mono text-sm")
                    ui.label(f"{result.get('duration_s', '—')}s").classes("text-xs opacity-50")
                if not ok:
                    # The failing check, its output, and its fix -- one click away.
                    with ui.column().classes("pl-8 w-full gap-1"):
                        if result.get("reason"):
                            ui.label(result["reason"]).classes("text-sm text-red-300")
                        if result.get("output"):
                            ui.code(result["output"], language="text").classes("w-full text-xs")
                        if result.get("fix"):
                            ui.code(result["fix"], language="bash").classes("w-full")

            for warning in record.get("warnings", []) or []:
                # The known gaps in the hash: dynamic imports and runtime-loaded
                # files. Shown, not swallowed.
                ui.label(f"⚠ {warning}").classes("text-xs text-amber-400")
