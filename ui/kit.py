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

    **And the backslash, which is worse than the quote.** NiceGUI parses a props
    string and hands each value to `ast.literal_eval`, so the text is read as a
    Python string literal and a backslash is an escape character in it. On
    Windows that is not a curiosity: `C:\\Users\\...` in a tooltip contains `\\U`,
    which begins a unicode escape, and `literal_eval` raises a SyntaxError from
    inside `element.props()` -- taking down not the tooltip but whatever was
    being built. It surfaced the first time a control put a *path* in a tooltip,
    which is to say the first time the appbar had to say which folder this is.
    """
    collapsed = " ".join(str("" if value is None else value).split())
    # Backslashes first: doing it after the quote swap would also escape the
    # apostrophes this puts in.
    return collapsed.replace("\\", "\\\\").replace('"', "'")


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


def sublabel(value: Any, classes: str = "") -> Any:
    """A 10px lowercase mono name for something *inside* a component.

    `label` shouts, which is right over a section of a window and wrong on the
    word `output` repeated once per tool call down a transcript. Kept as a
    separate primitive rather than a flag on `label` so the choice is visible at
    the call site.
    """
    return text(value, f"grad-sublabel {classes}".strip())


def caption(value: Any, classes: str = "") -> Any:
    return text(value, f"grad-caption {classes}".strip())


def shorten_path(value: str, *, keep: int = 2) -> str:
    """A path with its leading directories dropped, for a head that must fit.

    `C:\\Users\\vovas\\Grad\\projects\\proj-marl-agents\\MEMO.md` in a 410px pane
    is `C:\\Users\\vovas\\Grad\\projects\\proj-marl-…` once CSS has ellipsised it,
    which is every character except the informative ones. CSS can only truncate
    at the end, and for a path the end is the answer -- so the trimming has to
    happen here, where the separators are still legible as separators.

    Not applied to everything: a tool subject is a path for `Edit` and a *command*
    for `Bash`, and a command's first words are the informative ones. Whitespace
    is the test rather than the tool name, because it is a property of the string
    rather than of a caller's table -- `export PYTHONIOENCODING=utf-8; python -m
    pytest` keeps its head, `./scripts/run.sh` loses a leading `./` nobody reads.

    The full value belongs in a `title=` beside the call to this; nothing here
    should be the only place a path was written down.
    """
    if not value or any(c.isspace() for c in value):
        return value
    separator = "\\" if value.count("\\") >= value.count("/") else "/"
    parts = [p for p in value.split(separator) if p]
    if len(parts) <= keep:
        return value
    return "…" + separator + separator.join(parts[-keep:])


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


def band_strip(geometry: dict[str, Any] | None, *, unit: str = "", reason: str = "") -> Any:
    """Predicted band, observed tick, falsifier bounds.

    `None` renders the honest thing -- a note saying why there is no band --
    rather than an empty box that reads as a missing value. `reason` exists
    because there are two ways to have no geometry and only one of them was
    being reported: `band_geometry` also returns None for a numeric prediction
    with no result yet, and calling that "relational" told the reader something
    false about their own expectation.
    """
    if not geometry:
        return note(reason or "relational prediction — no numeric band to draw")
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
            # A command you are meant to run is a command you are meant to read.
            pre(fix, wrap=True)
    return element


def pre(value: Any, tone: str = "neutral", *, wrap: bool = False) -> Any:
    """A bordered mono block. `wrap` for prose-shaped machine text.

    Off by default because the default content is *aligned*: a table, a diff, a
    dataframe, a traceback with an indent that means something. Mono is assigned
    by the handoff to "anything the machine produced or that must align", and
    wrapping the second kind at the pane edge destroys the reason it is monospaced.

    On for the other kind -- a shell command, a URL, a one-line fix -- where the
    alternative is a horizontal scrollbar inside a card inside a scrolling pane,
    to read a line that was already there. Passed at the call site rather than
    inferred, because only the caller knows which kind it has.
    """
    classes = f"grad-pre {_tone_class(tone)} {'wrap' if wrap else ''}"
    return text(value, " ".join(classes.split()), tag="pre")


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


class Menu:
    """A dialog whose body is rebuilt each time it opens.

    `ui.dialog` builds its contents once. These menus list projects, folders,
    open windows and stored sessions, and all four change *because of* what the
    dialog does -- create a project and the list it was read from is already
    stale, open a window and the mark beside its name is wrong. Redrawing on open
    is cheaper than binding every row to the poll, and it cannot go stale between
    the click and the dialog appearing.

    `draw` is handed the menu so a control *inside* it can call `redraw` after
    changing what the menu is listing -- which is what lets the window menu stay
    open across several toggles instead of closing after each one.

    It lives here rather than in `ui/shell.py`, where it was written, because the
    chat window's session picker is the fourth of these: a Quasar `select` was
    the one control in the workspace still carrying NiceGUI's own look, and it
    could not say the two things a session row has to say -- that reopening one
    only redisplays it, and that another window already has it.
    """

    def __init__(self, dialog: Any, draw: Any) -> None:
        self._dialog = dialog
        self._draw = draw

    def open(self) -> None:
        self.redraw()
        self._dialog.open()

    def redraw(self) -> None:
        self._draw(self)

    def close(self) -> None:
        self._dialog.close()


def menu(draw: Callable[[Any, Any], None], *, width: int = 460) -> Menu:
    """A `Menu` over a card in the app's own paper, ready to be filled.

    `draw` is called with `(body, menu)` each time it opens; it is expected to
    clear the body itself, because a redraw from inside the menu is the same
    call.
    """
    ui = _ui()
    with ui.dialog() as dialog, el("div", "grad-app"):
        body = el("div", "grad-card", style=f"background: var(--grad-paper); min-width: {width}px")
    return Menu(dialog, lambda m: draw(body, m))


def steps(
    items: Sequence[dict[str, Any]],
    active: str,
    on_pick: Callable[[str], Any],
) -> Any:
    """A row of numbered steps, each one a way back to itself.

    Not a wizard that marches forward: every step stays reachable, because a
    setup that has to be restarted to change the answer to question two is a
    setup people abandon at question three. The mark is the step's state -- a
    tick when it is satisfied, its number when it is not -- so "what is left"
    is answerable without opening anything.

    `items` are `setup_model`'s steps; each needs `id`, `caption`, `ready`.
    """
    with el("div", "grad-steps") as element:
        for index, item in enumerate(items, start=1):
            current = item["id"] == active
            classes = "grad-step" + (" open" if current else "")
            step = el("button", classes)
            step.props(f'title="{attr(item.get("hint", ""))}"')
            step.on("click", lambda _=None, sid=item["id"]: on_pick(sid))
            with step:
                text("✓" if item.get("ready") else str(index), "mark", tag="span")
                text(item["caption"], "name", tag="span")
                text(item.get("detail", ""), "hint", tag="span")
    return element


class Confirm:
    """A yes/no dialog, built once and reused for every question.

    Built during the page and only *opened* later, for `_install_quit_guard`'s
    reason: a NiceGUI element belongs to the client whose slot context created
    it, and the handler that wants to ask is running long after that context has
    gone. So the shell constructs one of these while there is a client, and the
    control that needs an answer awaits `ask`.

    It exists because exactly one control in the app is destructive enough to
    need it. Switching *project* changes what spend is charged to; switching
    *workspace folder* replaces the ledger, the project list, the notebooks and
    the config under every open window at once. Those two sat six rows apart in
    one dialog, styled identically, and the difference was discoverable only by
    doing it.
    """

    def __init__(self, dialog: Any, card: Any) -> None:
        self._dialog = dialog
        self._card = card

    async def ask(
        self,
        title: str,
        body: str,
        *,
        confirm: str = "CONTINUE",
        cancel: str = "CANCEL",
        tone: str = "danger",
        note_text: str = "",
    ) -> bool:
        self._card.clear()
        with self._card:
            text(title, "grad-label")
            text(body)
            if note_text:
                note(note_text)
            with row("", gap=9):
                button(cancel, tone="primary", on_click=lambda: self._dialog.submit(False))
                button(confirm, tone=tone, on_click=lambda: self._dialog.submit(True))
        self._dialog.open()
        return bool(await self._dialog)


def confirm() -> Confirm:
    """A `Confirm` over the app's own paper. Call this during the page build."""
    ui = _ui()
    dialog = ui.dialog().props("persistent")
    with dialog, el("div", "grad-app"):
        card = column("grad-pad", gap=9).style(
            "background: var(--grad-paper); border: var(--grad-border); min-width: 440px"
        )
    return Confirm(dialog, card)


def menu_row(
    mark: str,
    name: str,
    hint: str,
    *,
    open: bool = False,
    title: str = "",
    wide: bool = False,
    disabled: bool = False,
) -> Any:
    """One row of a menu: a mark, a name, and what the row is for.

    A `<button>` so the keyboard reaches it, which is why the stylesheet resets
    four properties on it.

    `wide` gives the name the row instead of a fixed column -- a window is called
    `LEDGER` and a session is called whatever was first asked in it, and the same
    96px column cannot serve both.
    """
    classes = "grad-menu-row"
    if open:
        classes += " open"
    if wide:
        classes += " wide"
    if disabled:
        classes += " disabled"
    row = el("button", classes)
    if title:
        row.props(f'title="{attr(title)}"')
    if disabled:
        row.props("disabled")
    with row:
        text(mark, "mark", tag="span")
        text(name, "name", tag="span")
        text(hint, "hint", tag="span")
    return row


def run_js(code: str) -> None:
    """Send JavaScript to the browser once, after the client is connected.

    `ui.run_javascript` called straight from a render is a message to a socket
    that may not exist yet: during a page build the client has not connected,
    and inside a detached element tree there is no event loop at all. A one-shot
    zero-delay timer is NiceGUI's own answer -- it defers to the first tick
    after the page is live, and it makes the render function synchronous and
    testable rather than quietly scheduling background tasks.

    **It must be called with a live slot in scope, and that is a real trap.** The
    timer is an element, so it is created in the enclosing slot -- and inside an
    event handler the enclosing slot belongs to the element the handler was bound
    to, which a handler that rebuilds the UI has usually just deleted. There is
    no way to recover from here: `context.client` is itself reached *through* the
    current slot, so a deleted one takes the client with it and the code is never
    sent. All of it raises `RuntimeError: The parent element this slot belongs to
    has been deleted` before the socket is touched.

    It is also quiet. `ui/shell.py:retile` runs from a titlebar button and
    deletes every titlebar including the one that was clicked; the raise landed
    in `ui/state.py:_guard`, which logs and carries on, so the JavaScript simply
    never ran on any path but the first page build -- see `gradRearm`'s call
    site, which enters a long-lived container precisely for this reason.
    """
    ui = _ui()
    ui.timer(0.05, lambda: ui.run_javascript(code), once=True)


def stylesheet_head(*, url_prefix: str = "/grad-static/fonts") -> str:
    """Everything that goes in `<head>`: fonts first, then the stylesheet."""
    from ui import fonts

    return f"{fonts.head_html(url_prefix=url_prefix)}\n<style>\n{tokens.stylesheet()}\n</style>"
