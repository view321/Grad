"""The primitives every window is built from.

Quasar is bypassed here rather than overridden -- see the note at the top of
`tokens.py`. Everything below emits a plain element carrying a class from the
project stylesheet, so a chip is a `<span class="grad-chip ok">` and not a
`q-chip` with four defaults fighting the design.

Three consequences worth knowing before extending this:

* **Text is escaped here, once.** `ui.html` is the only way to get an arbitrary
  tag out of NiceGUI, and it renders raw HTML. Everything on screen in this app
  comes from a ledger, a traceback or a notebook -- all of which can contain
  angle brackets, and one of which can contain a downloaded repository's idea of
  a filename. So `text()` escapes server-side and then turns *off* the
  client-side sanitiser, because escaped text cannot contain markup and running
  DOMPurify over every label on every refresh is real work for no benefit. The
  one place raw HTML is intentional is `ui.markdown`, which keeps its own
  sanitiser on.
* **No element in here holds state.** A widget that owned its own value would
  need clearing and rebuilding on every refresh, and the design's motion rule is
  "instant state swaps, progress bars update in place". Windows update leaf
  properties instead; see `ui/state.py`.
* **`nicegui` is imported inside the functions, never at module scope**, so the
  models and layout tests run with the UI extra uninstalled.
"""

from __future__ import annotations

import html as _html
from typing import Any, Callable, Iterable, Sequence

from ui import tokens

# Tone -> the class suffix the stylesheet uses. Keeping the mapping here rather
# than letting callers pass raw class names is what makes "one accent per state"
# enforceable: a tone that is not in this dict does not render an accent.
TONES = {
    "ok": "ok",
    "attention": "attention",
    "broken": "broken",
    "neutral": "",
    "solid": "solid",
    "outline": "outline",
    "dashed": "dashed",
}

BUTTON_TONES = {
    "primary": "primary",
    "ok": "ok",
    "danger": "danger",
    "active": "active",
    "dashed": "dashed",
    "ghost": "ghost",
    "neutral": "",
}


def _ui() -> Any:
    from nicegui import ui  # noqa: PLC0415 - import at use, not at module scope

    return ui


def _tone_class(tone: str | None) -> str:
    return TONES.get(tone or "neutral", "")


def escape(value: Any) -> str:
    """HTML-escape, for text going into an element's content."""
    return _html.escape("" if value is None else str(value), quote=False)


def attr(value: Any) -> str:
    """A value safe to interpolate into a `props('name="…"')` string.

    Deliberately *not* `escape`. Props are parsed by NiceGUI and then bound as
    attributes client-side, so nothing decodes entities on the way: an escaped
    apostrophe would reach the screen as a literal `&#x27;` in the tooltip.

    What actually has to go is the double quote, which closes the value early
    and takes every attribute after it with it -- silently, because the parser
    has no reason to complain. That matters because these values are not all
    constants: a preflight remedy and a lineage bar's candidate id are ledger
    text, and ledger text can hold a quote. Newlines go for the same reason.
    """
    collapsed = " ".join(str("" if value is None else value).split())
    return collapsed.replace('"', "'")


def el(tag: str, classes: str = "", *, style: str = "") -> Any:
    """A bare container element with our classes on it."""
    element = _ui().element(tag)
    if classes:
        element.classes(classes)
    if style:
        element.style(style)
    return element


def text(value: Any, classes: str = "", *, tag: str = "div", style: str = "") -> Any:
    """A leaf carrying escaped text. Update it later with `.set_content()`."""
    element = _ui().html(escape(value), tag=tag, sanitize=False)
    if classes:
        element.classes(classes)
    if style:
        element.style(style)
    return element


def set_text(element: Any, value: Any) -> None:
    """In-place update, which is what the design's no-fades rule requires."""
    element.set_content(escape(value))


def label(value: Any, classes: str = "") -> Any:
    """An 11px mono uppercase label at the design's letter-spacing."""
    return text(value, f"grad-label {classes}".strip())


def caption(value: Any, classes: str = "") -> Any:
    return text(value, f"grad-caption {classes}".strip())


def mono(value: Any, classes: str = "") -> Any:
    return text(value, f"grad-mono {classes}".strip())


def chip(value: Any, tone: str = "neutral", *, dot: bool = False) -> Any:
    with el("span", f"grad-chip {_tone_class(tone)}".strip()) as element:
        if dot:
            el("span", "dot")
        text(value, tag="span")
    return element


def button(
    value: Any,
    *,
    tone: str = "neutral",
    on_click: Callable[..., Any] | None = None,
    disabled: bool = False,
    title: str = "",
    classes: str = "",
) -> Any:
    """A 2px-bordered square button. `tone` picks the fill, never a gradient."""
    element = text(value, f"grad-btn {BUTTON_TONES.get(tone, '')} {classes}".strip(), tag="button")
    if title:
        element.props(f'title="{attr(title)}"')
    if disabled:
        element.props("disabled").classes("disabled")
    elif on_click is not None:
        element.on("click", on_click)
    return element


def group(children: Sequence[tuple[str, str, Callable[..., Any] | None]]) -> Any:
    """A joined button group: 2px borders, `border-left: 0` on the joins."""
    with el("div", "grad-group") as element:
        for value, tone, handler in children:
            button(value, tone=tone, on_click=handler)
    return element


def kv(rows: Iterable[tuple[str, Any]]) -> Any:
    with el("div", "grad-kv") as element:
        for key, value in rows:
            text(key, "k")
            text("—" if value is None else value, "v")
    return element


def bar(segments: Sequence[tuple[float, str, str]], *, thin: bool = False) -> Any:
    """A segmented meter. `segments` is `(fraction, tone-class, inline label)`.

    Fractions are of the whole bar and need not sum to 1 -- the remainder is the
    unspent part, and showing it as empty paper rather than as a third colour is
    what keeps the meter readable at a glance.
    """
    with el("div", f"grad-bar {'thin' if thin else ''}".strip()) as element:
        for fraction, tone, caption_text in segments:
            width = max(0.0, min(1.0, float(fraction))) * 100
            if width <= 0:
                continue
            seg = el("div", f"seg {tone}", style=f"width: {width:.4f}%")
            if caption_text:
                with seg:
                    text(caption_text, tag="span")
    return element


def progress(fraction: float, tone: str = "running") -> Any:
    """The 12px queue progress bar. `tone` is running | done | failed | queued."""
    variant = {"running": "", "done": "done", "failed": "failed", "queued": "queued"}.get(tone, "")
    with el("div", f"grad-progress {variant}".strip()) as element:
        width = max(0.0, min(1.0, float(fraction))) * 100
        if width > 0:
            el("div", "fill", style=f"width: {width:.2f}%")
    return element


def status_square(state: str, glyph: str) -> Any:
    return text(glyph, f"grad-status-square {_tone_class(state)}".strip())


def band_strip(geometry: dict[str, Any] | None, *, unit: str = "") -> Any:
    """Predicted band, observed tick, falsifier bounds.

    `None` renders the honest thing -- a note saying the prediction is
    relational and has no band -- rather than an empty box that reads as a
    missing value.
    """
    if not geometry:
        return note("relational prediction — no numeric band to draw")
    with el("div") as element:
        with el("div", "grad-band"):
            start = geometry.get("band_start")
            end = geometry.get("band_end")
            if start is not None and end is not None:
                width = max(0.6, (end - start) * 100)
                el("div", "band", style=f"left: {start * 100:.4f}%; width: {width:.4f}%")
            for key in ("falsifier_low", "falsifier_high"):
                position = geometry.get(key)
                if position is not None:
                    el("div", "tick falsifier", style=f"left: {position * 100:.4f}%")
            actual = geometry.get("actual")
            if actual is not None:
                el("div", "tick", style=f"left: {actual * 100:.4f}%")
                text(f"{geometry.get('actual_value')}{unit}", "value", style=f"left: {actual * 100:.4f}%")
        with el("div", "grad-band-labels"):
            text(_number(geometry.get("axis_min")), tag="span")
            text("predicted band", tag="span")
            text(_number(geometry.get("axis_max")), tag="span")
    return element


def _number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    if abs(value) >= 1000 or (value and abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.4g}"


def note(value: Any) -> Any:
    """The 2px dashed box the design uses for anything qualifying a number."""
    return text(value, "grad-note")


def empty(message: str, fix: str | None = None) -> Any:
    """An empty window that says what would fill it.

    Every empty state in this app carries the command that ends it. A blank pane
    is the one thing a research instrument cannot afford to show.
    """
    with el("div", "grad-pad") as element:
        text(message, "grad-empty")
        if fix:
            pre(fix)
    return element


def pre(value: Any, tone: str = "neutral") -> Any:
    return text(value, f"grad-pre {_tone_class(tone)}".strip(), tag="pre")


def error_strip(message: str | None) -> None:
    """Shown above a window's content when its data source failed to read."""
    if not message:
        return
    with el("div", "grad-pad"):
        with el("div", "grad-card"):
            text("SOURCE UNREADABLE", "head broken")
            with el("div", "body"):
                text(message, "grad-mono")


def hr() -> Any:
    return el("div", "grad-hr")


def spacer() -> Any:
    return el("div", "", style="flex: 1 1 auto")


def row(classes: str = "", *, gap: int = 9, align: str = "center") -> Any:
    return el("div", classes, style=f"display: flex; align-items: {align}; gap: {gap}px;")


def column(classes: str = "", *, gap: int = 0) -> Any:
    return el("div", classes, style=f"display: flex; flex-direction: column; gap: {gap}px; min-height: 0;")


def pad(classes: str = "") -> Any:
    return el("div", f"grad-pad {classes}".strip())


def scroll_body(classes: str = "") -> Any:
    """The scrolling region under a window's title bar."""
    return el("div", f"grad-body {classes}".strip())


def figure_placeholder(caption_text: str, tag_text: str) -> Any:
    with el("div", "grad-figure") as element:
        chip(caption_text, "neutral")
        text(tag_text, "tag")
    return element


def blink_caret() -> Any:
    return el("span", "grad-caret")


def run_js(code: str) -> None:
    """Send JavaScript to the browser once, after the client is connected.

    `ui.run_javascript` called straight from a render is a message to a socket
    that may not exist yet: during a page build the client has not connected,
    and inside a detached element tree there is no event loop at all. A one-shot
    zero-delay timer is NiceGUI's own answer -- it defers to the first tick
    after the page is live, and it makes the render function synchronous and
    testable rather than quietly scheduling background tasks.
    """
    ui = _ui()
    ui.timer(0.05, lambda: ui.run_javascript(code), once=True)


def stylesheet_head(*, url_prefix: str = "/grad-static/fonts") -> str:
    """Everything that goes in `<head>`: fonts first, then the stylesheet."""
    from ui import fonts

    return f"{fonts.head_html(url_prefix=url_prefix)}\n<style>\n{tokens.stylesheet()}\n</style>"
