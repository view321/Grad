"""The window registry: the one list the whole shell is derived from.

The opener strip, the layout presets, the `⌘K` palette, the persisted layout's
validation and the status bar's count all read this tuple. Adding a twelfth
window is adding one `WindowSpec` and one module -- if it is ever more than
that, something has grown a second list and the two will drift.

`module` is resolved with `importlib` at first render rather than imported here,
for the reason the rest of the app imports lazily: `ui.registry` has to stay
importable without NiceGUI so the layout and model tests can validate window ids
against it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WindowSpec:
    """One window. `id` is what persists; `name` is what the opener shows."""

    id: str
    name: str
    module: str
    hint: str
    #: In the arrangement a fresh workspace opens with. The mock's opening
    #: state: chat and notebook side by side, ledger over quota on the right.
    default: bool = False
    #: Windows that own a heavy embedded document (the Lab iframe) and must not
    #: be rebuilt on every refresh tick.
    persistent: bool = False


WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("chat", "chat", "ui.windows.chat", "the agent session", default=True, persistent=True),
    WindowSpec("notebook", "notebook", "ui.windows.notebook", "JupyterLab, and the verify banner", default=True, persistent=True),
    WindowSpec("ledger", "ledger", "ui.windows.ledger", "expectations against outcomes", default=True),
    WindowSpec("quota", "quota", "ui.windows.quota", "the 5-hour window and today's spend", default=True),
    WindowSpec("wiki", "wiki", "ui.windows.wiki", "the generated codebase wiki"),
    WindowSpec("papers", "papers", "ui.windows.papers", "papers, and what depends on them"),
    WindowSpec("evolve", "evolve", "ui.windows.evolve", "ShinkaEvolve campaigns"),
    WindowSpec("editor", "editor", "ui.windows.editor", "the LaTeX paper and its claim bindings"),
    WindowSpec("preflight", "preflight", "ui.windows.preflight", "the checklist that blocks a submission"),
    WindowSpec("funnel", "funnel", "ui.windows.funnel", "retrieval, stage by stage"),
    WindowSpec("queue", "queue", "ui.windows.queue", "runs and GPU jobs"),
)

BY_ID: dict[str, WindowSpec] = {w.id: w for w in WINDOWS}


def ids() -> tuple[str, ...]:
    return tuple(w.id for w in WINDOWS)


def spec(window_id: str) -> WindowSpec:
    try:
        return BY_ID[window_id]
    except KeyError:  # pragma: no cover - the shell never asks for an unknown id
        raise KeyError(f"unknown window {window_id!r}; known: {', '.join(ids())}") from None


def defaults() -> tuple[str, ...]:
    return tuple(w.id for w in WINDOWS if w.default)


def renderer(window_id: str) -> Callable[[Any], None]:
    """`render(context)` for one window, imported on first use."""
    return _entry(window_id, "render")


def subtitle(window_id: str, context: Any) -> str:
    """The 55%-opacity mono line beside the name in the title bar.

    A window that does not define one gets its hint, so the bar is never empty.
    """
    fn = _entry(window_id, "subtitle", required=False)
    if fn is None:
        return spec(window_id).hint
    try:
        return str(fn(context) or "")
    except Exception:  # noqa: BLE001 - a subtitle must never break a title bar
        return spec(window_id).hint


def chips(window_id: str, context: Any) -> list[tuple[str, str]]:
    """`(text, tone)` state chips for the title bar. Optional per window."""
    fn = _entry(window_id, "chips", required=False)
    if fn is None:
        return []
    try:
        return list(fn(context) or [])
    except Exception:  # noqa: BLE001 - same reason as `subtitle`
        return []


def _entry(window_id: str, attribute: str, *, required: bool = True) -> Any:
    module = importlib.import_module(spec(window_id).module)
    fn = getattr(module, attribute, None)
    if fn is None and required:
        raise AttributeError(f"{spec(window_id).module} defines no {attribute}()")
    return fn
