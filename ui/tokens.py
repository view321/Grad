"""The design tokens, and the stylesheet generated from them.

Every colour, border, shadow and size in the window system comes from this
module. Nothing else in `ui/` is allowed to spell a hex code -- `tests/
test_ui_tokens.py` enforces that -- because the handoff's whole rule set is
stated in terms of these names ("one accent per state", "radius 0 everywhere",
"no blur shadows anywhere"), and a rule you can only check by eye is a rule that
drifts.

The stylesheet is *generated* rather than shipped as a `.css` file for the same
reason: a static file would be a second copy of the table, free to disagree with
this one. `stylesheet()` is a pure function of the constants below, so the test
that asserts the handoff's table is fully represented is testing the thing that
actually renders.

Quasar is bypassed, not overridden. Nearly every component NiceGUI would give us
carries a border-radius, a ripple, an elevation shadow and an uppercase
transform, all four of which this design contradicts; unwinding that per
component costs more than emitting a `<div>` with a class from here. The
`_quasar_reset()` block exists only for the handful of places a real Quasar
control is still the right answer (a textarea that autogrows, a select that
opens a menu).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------
# The handoff's table, verbatim. Keys are the token names it uses; they become
# `--grad-<key>` custom properties.
COLOUR: dict[str, str] = {
    "ink": "#14100C",
    "paper": "#F7F3E8",
    "paper-raised": "#FFFDF8",
    "paper-sunk": "#EFE8D8",
    "desk": "#E8E2D4",
    "rule-soft": "rgba(20,16,12,0.15)",
    "rule-mid": "rgba(20,16,12,0.3)",
    "attention": "#FFD400",
    "verified": "#12A594",
    "verified-ink": "#04302C",
    "verified-tint": "#DFF3EF",
    "broken": "#A3122F",
    "broken-tint": "#FDEEF1",
    "broken-ink": "#5A1020",
    "broken-ink-2": "#7A1024",
    "link": "#B04A2C",
    "muted": "#8A8272",
    "muted-2": "#9C9484",
    "literal": "#0B7B6E",
    "hatch-a": "#F1EADA",
    "hatch-b": "#E7DFCC",
    # Not in the table but used by it: the hairline that separates controls
    # inside the ink title bar, where `rule-soft` would be invisible.
    "ink-rule": "#4A443A",
    "row-alt": "#FDFAF2",
    "attention-row": "#FFFBE8",
}

# The four state accents, and the rule that governs them. `one accent per state,
# never two in the same element` is a property of this mapping being total: a
# state that is not here has no accent, rather than borrowing one.
STATE_ACCENT: dict[str, str] = {
    "ok": COLOUR["verified"],
    "attention": COLOUR["attention"],
    "broken": COLOUR["broken"],
    "neutral": COLOUR["ink"],
}

# ---------------------------------------------------------------------------
# type
# ---------------------------------------------------------------------------
FONT_SANS = "'Space Grotesk', Helvetica, Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"
FONT_SERIF = "'Instrument Serif', Georgia, 'Times New Roman', serif"

# Every size the handoff lists as in use. A size not in this tuple is a mistake,
# which is worth being able to assert.
TYPE_SCALE: tuple[float, ...] = (
    9, 10, 11, 12, 12.5, 13, 13.5, 14, 14.5, 15, 19, 21, 28, 30, 34,
)

# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
BORDER_STRUCTURAL = f"2px solid {COLOUR['ink']}"
BORDER_HAIRLINE = f"1px solid {COLOUR['rule-soft']}"
BORDER_SECONDARY = f"1px dashed {COLOUR['rule-mid']}"
BORDER_PENDING = f"2px dashed {COLOUR['ink']}"

SHADOW_SHELL = f"8px 8px 0 {COLOUR['ink']}"
SHADOW_CARD = f"6px 6px 0 {COLOUR['ink']}"

RADIUS = "0"
STRIPE_WIDTH = "6px"          # cell / ledger-entry state stripe
HANDLE_WIDTH = 8              # px, the pane split handle
TITLE_BAR_HEIGHT = 30         # px, every window's own bar
APP_BAR_HEIGHT = 42           # px, the workspace title bar
OPENER_HEIGHT = 30            # px, the window opener strip
STATUS_HEIGHT = 30            # px, the workspace status bar
GUTTER_PANE = 72              # px, notebook gutter inside a pane
GUTTER_STANDALONE = 84        # px, notebook gutter at full size
MIN_PANE_PX = 320             # the handoff's minimum pane width

# The only two motions in the system.
BLINK = "gradblink 1.1s steps(1) infinite"


def css_variables() -> str:
    """`:root` block. Everything else in the stylesheet reads from here."""
    lines = [f"    --grad-{name}: {value};" for name, value in COLOUR.items()]
    lines += [
        f"    --grad-font-sans: {FONT_SANS};",
        f"    --grad-font-mono: {FONT_MONO};",
        f"    --grad-font-serif: {FONT_SERIF};",
        f"    --grad-border: {BORDER_STRUCTURAL};",
        f"    --grad-hairline: {BORDER_HAIRLINE};",
        f"    --grad-secondary: {BORDER_SECONDARY};",
        f"    --grad-pending: {BORDER_PENDING};",
        f"    --grad-shadow-shell: {SHADOW_SHELL};",
        f"    --grad-shadow-card: {SHADOW_CARD};",
        f"    --grad-handle: {HANDLE_WIDTH}px;",
        f"    --grad-titlebar: {TITLE_BAR_HEIGHT}px;",
        f"    --grad-min-pane: {MIN_PANE_PX}px;",
    ]
    return ":root {\n" + "\n".join(lines) + "\n}"


def _quasar_reset() -> str:
    """Undo the four Quasar defaults that contradict the design.

    Scoped to the app root rather than global so the handful of real Quasar
    controls we keep (textarea, select) inherit the reset without needing a
    class each. Radius, ripple, elevation and the uppercase button transform are
    the whole list -- anything beyond that is bypassed by not using the
    component in the first place.
    """
    return """
.grad-app .q-field__control,
.grad-app .q-btn,
.grad-app .q-menu,
.grad-app .q-card,
.grad-app .q-chip,
.grad-app .q-item { border-radius: 0 !important; }
.grad-app .q-btn { text-transform: none; letter-spacing: 0; box-shadow: none !important;
                   min-height: 0; font-family: var(--grad-font-mono); }
.grad-app .q-btn .q-focus-helper, .grad-app .q-ripple { display: none !important; }
.grad-app .q-field--outlined .q-field__control:before { border: var(--grad-border); }
.grad-app .q-field--outlined .q-field__control:after { display: none; }
.grad-app .q-field__native, .grad-app .q-field__input {
    font-family: var(--grad-font-sans); color: var(--grad-ink); }
.grad-app .q-tab { text-transform: none; letter-spacing: 0; }
"""


def _base() -> str:
    return """
@keyframes gradblink { 0%,49% { opacity: 1 } 50%,100% { opacity: 0 } }

.grad-app, .grad-app * { box-sizing: border-box; border-radius: 0; }
body { margin: 0; background: var(--grad-desk); }

/* The page itself must never scroll: the shell is already sized to fill the
   window exactly (`100vh` less its 14px margins), so any scrollbar here means
   a wrapper is adding space the layout did not budget for. NiceGUI's
   `.nicegui-content` adds it twice over -- `padding: 16px` made the document
   32px taller than the window, and `align-items: flex-start` on the same
   flex wrapper let `.grad-app` size to its widest pane's max-content rather
   than to the window, which is what pushed the tiling area ~200px off the
   right edge. `align-self` answers the second; the width and height pin the
   app to the viewport so the shell's own arithmetic is the only thing
   deciding its size. */
html, body { height: 100%; overflow: hidden; }
.nicegui-content { padding: 0 !important; gap: 0 !important; width: 100%; }
.grad-app {
    width: 100%; height: 100vh; align-self: stretch; overflow: hidden;
    background: var(--grad-desk);
    color: var(--grad-ink);
    font-family: var(--grad-font-sans);
    font-weight: 500;
    font-size: 13px;
    line-height: 1.55;
}
.grad-app a { color: var(--grad-link); text-decoration: underline; text-underline-offset: 2px; }
.grad-app a:hover { color: var(--grad-ink); background: var(--grad-attention); }
.grad-app ::selection { background: var(--grad-attention); color: var(--grad-ink); }
.grad-app :focus-visible { outline: 2px solid var(--grad-ink); outline-offset: 2px; }

.grad-mono { font-family: var(--grad-font-mono); }
.grad-serif { font-family: var(--grad-font-serif); }
.grad-label {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
}
.grad-caption { font-family: var(--grad-font-mono); font-size: 10px; letter-spacing: 0.08em; }
.grad-blink { animation: gradblink 1.1s steps(1) infinite; }
.grad-caret { display: inline-block; width: 8px; height: 14px; background: var(--grad-ink);
              vertical-align: text-bottom; animation: gradblink 1.1s steps(1) infinite; }
"""


def _shell() -> str:
    return """
.grad-shell {
    border: var(--grad-border); box-shadow: var(--grad-shadow-shell);
    background: var(--grad-paper); display: flex; flex-direction: column;
    height: calc(100vh - 28px); margin: 14px; overflow: hidden;
}
.grad-appbar {
    display: flex; align-items: stretch; background: var(--grad-ink);
    color: var(--grad-paper); border-bottom: var(--grad-border); flex: 0 0 auto;
}
.grad-appbar-cell {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px;
    border-right: 1px solid var(--grad-ink-rule);
    font-family: var(--grad-font-mono); font-size: 12px;
}
.grad-appbar-cell.right { border-right: 0; border-left: 1px solid var(--grad-ink-rule); }
.grad-appbar-cell.brand { border-right: 2px solid var(--grad-paper); }
.grad-mark {
    width: 22px; height: 22px; background: var(--grad-attention);
    border: 2px solid var(--grad-paper); display: flex; align-items: center;
    justify-content: center; font-family: var(--grad-font-mono); font-size: 13px;
    font-weight: 700; color: var(--grad-ink);
}
.grad-wordmark { font-family: var(--grad-font-mono); font-size: 15px; font-weight: 700;
                 letter-spacing: 0.22em; }
.grad-appbar .dim { opacity: 0.55; }
.grad-appbar-btn {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    padding: 3px 8px; border: 1.5px solid var(--grad-paper); cursor: pointer;
    background: transparent; color: inherit;
}
.grad-appbar-btn:hover { background: var(--grad-paper); color: var(--grad-ink); }

/* The `⋯` button carries a menu, so it gets the caret's job: a little wider
   than a word button, and legible as a target rather than as punctuation. */
.grad-dots { font-size: 15px; line-height: 1; padding: 1px 9px 4px; letter-spacing: 0.1em; }

/* One row of the `⋯` menu: a mark, a name, and what the window is for. Rows are
   buttons so the keyboard reaches them, which means resetting the four
   properties a `<button>` brings with it. */
.grad-menu-row {
    display: flex; align-items: baseline; gap: 9px; width: 100%;
    padding: 6px 8px; cursor: pointer; text-align: left;
    background: transparent; color: var(--grad-ink); border: 0;
    font: inherit; font-family: var(--grad-font-mono); font-size: 12px;
}
.grad-menu-row:hover { background: var(--grad-paper-sunk); }
.grad-menu-row .mark { flex: 0 0 22px; opacity: 0.55; }
.grad-menu-row .name {
    flex: 0 0 96px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em;
}
.grad-menu-row .hint {
    flex: 1 1 auto; min-width: 0; opacity: 0.55;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
/* Open is the state worth reading off the list at a glance, so it is the one
   that gets ink -- the mark alone is too quiet at eleven rows. */
.grad-menu-row.open { background: var(--grad-ink); color: var(--grad-paper); }
.grad-menu-row.open .mark, .grad-menu-row.open .hint { opacity: 0.7; }
.grad-menu-row.open:hover { background: var(--grad-broken); color: #fff; }
/* A session is called whatever was first asked in it, so its name gets the row
   and the count beside it gets what it needs -- the reverse of a window list,
   where the names are one word and the hints are the sentence. */
.grad-menu-row.wide .name { flex: 1 1 auto; min-width: 0; text-transform: none;
                            letter-spacing: 0; overflow: hidden;
                            white-space: nowrap; text-overflow: ellipsis; }
.grad-menu-row.wide .hint { flex: 0 0 auto; }
/* A row that cannot be opened: the current one, and one another window holds.
   It is dimmed rather than hidden, because "in use elsewhere" is the answer to
   the question the list is being read to answer. */
.grad-menu-row.disabled { opacity: 0.45; cursor: default; }
.grad-menu-row.disabled:hover { background: transparent; }
.grad-menu-row.open.disabled { opacity: 1; }

.grad-statusbar {
    display: flex; align-items: center; gap: 14px; padding: 0 12px;
    background: var(--grad-ink); color: var(--grad-paper);
    font-family: var(--grad-font-mono); font-size: 11px;
    height: 30px; flex: 0 0 auto; border-top: var(--grad-border);
}
.grad-statusbar .dim { opacity: 0.55; }
.grad-statusbar .count {
    background: var(--grad-attention); color: var(--grad-ink);
    padding: 2px 7px; font-weight: 700;
}
"""


def _tiling() -> str:
    """The tiling area.

    Fractions live in CSS custom properties on the pane elements so the drag can
    write them from JS without a server round-trip; Python only ever reads them
    back at the end of a gesture.
    """
    return """
/* The one place left that may scroll, and only under duress: three columns
   hold a `--grad-min-pane` floor each, so under about 1000px of window there
   is genuinely not enough room for them. Clipping would put a pane out of
   reach, so the tiling area scrolls sideways instead -- the chrome above and
   below it stays put. Safe for the Lab overlay because `tiling.js` reflows
   the flown iframes on capture-phase scroll, not just on resize. */
.grad-tiles { display: flex; flex: 1 1 auto; min-height: 0; align-items: stretch;
              overflow-x: auto; overflow-y: hidden; }
.grad-column {
    display: flex; flex-direction: column; min-width: var(--grad-min-pane);
    flex: var(--grad-fraction, 1) 1 0; min-height: 0;
}
.grad-slot {
    display: flex; flex-direction: column; min-height: 0;
    flex: var(--grad-fraction, 1) 1 0; overflow: hidden;
}
.grad-handle {
    flex: 0 0 var(--grad-handle); background: var(--grad-ink); cursor: col-resize;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 3px; user-select: none;
}
.grad-handle.row { cursor: row-resize; flex-basis: var(--grad-handle);
                   flex-direction: row; }
.grad-handle span { width: 2px; height: 2px; background: var(--grad-paper); display: block; }
.grad-handle.dragging { background: var(--grad-broken); }

.grad-window { display: flex; flex-direction: column; min-height: 0; flex: 1 1 auto;
               background: var(--grad-paper); overflow: hidden; }
.grad-window.focused .grad-titlebar { background: var(--grad-ink); color: var(--grad-paper); }
.grad-window.focused .grad-titlebar .grad-winctl { color: var(--grad-paper); }
.grad-titlebar {
    display: flex; align-items: center; gap: 10px; padding: 0 10px;
    height: var(--grad-titlebar); flex: 0 0 var(--grad-titlebar);
    background: var(--grad-paper-sunk); border-bottom: var(--grad-border);
    cursor: grab; user-select: none;
}
.grad-titlebar .name {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
}
.grad-titlebar .subtitle {
    font-family: var(--grad-font-mono); font-size: 11px; opacity: 0.55;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.grad-winctl { opacity: 0.5; cursor: pointer; background: none; border: 0;
               font: inherit; color: inherit; padding: 0 3px; }
.grad-winctl:hover { opacity: 1; }
.grad-body { flex: 1 1 auto; min-height: 0; overflow: auto; }
.grad-pad { padding: 12px 14px; }

/* The Lab iframe lives outside the pane tree (see ui/static/tiling.js): a
   reparented iframe is destroyed and recreated by the browser, which would
   reload JupyterLab -- kernel and all -- on every retile. */
.grad-iframe-host { position: absolute; border: 0; background: #fff; z-index: 5; }
.grad-iframe-anchor { flex: 1 1 auto; min-height: 0; }

/* Dragging a title bar to retile. Three pieces of feedback, and none of them
   animates: the design's "no easing curves, no fades" applies to a gesture the
   pointer is already driving at frame rate.

   The ghost and the indicator sit above the flown Lab iframe (z-index 5) and
   are `pointer-events: none`, which is load-bearing rather than tidy --
   `tiling.js` hit-tests with `elementFromPoint`, and an indicator that could be
   hit would answer every query with itself. */
.grad-drop-indicator {
    position: fixed; z-index: 40; pointer-events: none;
    background: var(--grad-attention); outline: 1px solid var(--grad-ink);
}
.grad-drag-ghost {
    position: fixed; z-index: 41; pointer-events: none;
    background: var(--grad-ink); color: var(--grad-paper);
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em; padding: 3px 8px;
}
/* The pane being dragged stays exactly where it is until the drop lands: the
   layout is the server's, and moving it here would show an arrangement that
   does not exist yet. It is marked, not moved. */
.grad-window .grad-titlebar.grad-drag-source {
    outline: 2px dashed var(--grad-ink); outline-offset: -2px; opacity: 0.6;
}
.grad-window .grad-titlebar.grad-swap-target {
    background: var(--grad-attention); color: var(--grad-ink);
}
.grad-window .grad-titlebar.grad-swap-target .grad-winctl { color: var(--grad-ink); }
body.grad-dragging { cursor: grabbing; }
/* Text selection and iframe hit-testing both eat a drag that crosses a pane. */
body.grad-dragging * { user-select: none !important; }
body.grad-dragging .grad-iframe-host { pointer-events: none; }
"""


def _controls() -> str:
    return """
.grad-btn {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; padding: 9px 11px;
    border: var(--grad-border); background: var(--grad-paper); color: var(--grad-ink);
    cursor: pointer; line-height: 1;
}
.grad-btn:hover { background: var(--grad-paper-sunk); }
.grad-btn.primary { background: var(--grad-attention); }
.grad-btn.primary:hover { background: var(--grad-attention); filter: brightness(0.94); }
.grad-btn.ok { background: var(--grad-verified); color: var(--grad-verified-ink); }
.grad-btn.danger { background: var(--grad-broken); color: #fff; }
.grad-btn.active { background: var(--grad-ink); color: var(--grad-paper); }
.grad-btn.dashed { border: var(--grad-pending); background: transparent; opacity: 0.75; }
.grad-btn[disabled], .grad-btn.disabled {
    background: var(--grad-paper-sunk); opacity: 0.5; pointer-events: none;
}
.grad-btn.ghost { border: 0; background: transparent; padding: 6px 8px; }
.grad-btn.ghost:hover { background: var(--grad-paper-sunk); }
.grad-group { display: flex; }
.grad-group .grad-btn + .grad-btn { border-left: 0; }

.grad-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    padding: 3px 8px; border: var(--grad-border); background: var(--grad-paper);
    white-space: nowrap; text-transform: uppercase; letter-spacing: 0.06em;
}
.grad-chip.ok { background: var(--grad-verified); color: var(--grad-verified-ink); border-color: var(--grad-ink); }
.grad-chip.attention { background: var(--grad-attention); color: var(--grad-ink); }
.grad-chip.broken { background: var(--grad-broken); color: #fff; }
.grad-chip.solid { background: var(--grad-ink); color: var(--grad-paper); }
.grad-chip.outline { background: transparent; }
.grad-chip.dashed { border: var(--grad-pending); background: transparent; }
.grad-chip .dot { width: 7px; height: 7px; background: currentColor; }

.grad-kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px;
           font-family: var(--grad-font-mono); font-size: 12px; }
.grad-kv .k { opacity: 0.55; }
.grad-kv .v { font-weight: 700; }

.grad-empty { border: var(--grad-pending); padding: 14px; font-size: 13px; opacity: 0.75; }
.grad-note { border: var(--grad-pending); padding: 11px; font-size: 12px; line-height: 1.55; }
.grad-pre {
    font-family: var(--grad-font-mono); font-size: 12.5px; line-height: 1.7;
    background: var(--grad-paper-raised); border: 1.5px solid var(--grad-ink);
    padding: 11px; overflow: auto; white-space: pre; margin: 0;
}
.grad-pre.broken { background: var(--grad-broken-tint); border-color: var(--grad-broken);
                   color: var(--grad-broken-ink); }
.grad-hr { height: 0; border: 0; border-top: var(--grad-secondary); margin: 12px 0; }
"""


def _data() -> str:
    """Bars, tables, stripes -- the shapes the ten data windows share."""
    return """
.grad-bar { display: flex; border: 2px solid var(--grad-ink); background: var(--grad-paper-raised);
            height: 22px; overflow: hidden; }
.grad-bar .seg { display: flex; align-items: center; justify-content: center;
                 font-family: var(--grad-font-mono); font-size: 10px; font-weight: 700;
                 overflow: hidden; white-space: nowrap; }
.grad-bar .seg.chat { background: var(--grad-attention); color: var(--grad-ink); }
.grad-bar .seg.tool { background: var(--grad-verified); color: var(--grad-verified-ink); }
.grad-bar .seg.ink  { background: var(--grad-ink); color: var(--grad-paper); }
.grad-bar .seg.opus { background: var(--grad-link); color: #fff; }
.grad-bar.thin { height: 12px; border-width: 1.5px; }

.grad-table { width: 100%; border-collapse: collapse; font-family: var(--grad-font-mono);
              font-size: 12px; }
.grad-table thead th {
    background: var(--grad-ink); color: var(--grad-paper); text-align: left;
    padding: 7px 9px; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    font-weight: 700; white-space: nowrap;
}
.grad-table td { padding: 8px 9px; border-bottom: var(--grad-hairline); vertical-align: middle; }
.grad-table tbody tr:nth-child(even) { background: var(--grad-row-alt); }
.grad-table tbody tr.running { background: var(--grad-paper-raised); }

.grad-progress { height: 12px; border: 1.5px solid var(--grad-ink); display: flex;
                 background: var(--grad-paper-raised); min-width: 90px; }
.grad-progress .fill { background: var(--grad-verified); }
.grad-progress.done .fill { background: var(--grad-ink); }
.grad-progress.failed { border-color: var(--grad-broken); }
.grad-progress.failed .fill { background: var(--grad-broken); }
.grad-progress.queued { border-style: dashed; }

.grad-row { display: flex; align-items: flex-start; gap: 11px; padding: 11px 14px;
            border-bottom: var(--grad-hairline); }
.grad-row.striped { border-left: 6px solid var(--grad-ink); }
.grad-row.striped.ok { border-left-color: var(--grad-verified); }
.grad-row.striped.attention { border-left-color: var(--grad-attention); }
.grad-row.striped.broken { border-left-color: var(--grad-broken); }
.grad-row.selected { background: var(--grad-paper-raised); }
.grad-row.attention-bg { background: var(--grad-attention-row); }
.grad-row:hover { background: var(--grad-paper-raised); }

.grad-status-square { width: 18px; height: 18px; border: var(--grad-border);
                      display: flex; align-items: center; justify-content: center;
                      font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
                      flex: 0 0 18px; }
.grad-status-square.ok { background: var(--grad-verified); color: var(--grad-verified-ink); }
.grad-status-square.attention { background: var(--grad-attention); color: var(--grad-ink); }
.grad-status-square.broken { background: var(--grad-broken); color: #fff; }

/* Ledger band strip: predicted band, observed tick, falsifier bounds. */
.grad-band { position: relative; height: 30px; border: 1.5px solid var(--grad-ink);
             background: var(--grad-paper-raised); margin-top: 8px; }
.grad-band .band { position: absolute; top: 0; bottom: 0; background: var(--grad-verified);
                   opacity: 0.35; }
.grad-band .tick { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--grad-ink); }
.grad-band .tick.falsifier { background: var(--grad-broken); }
.grad-band .value { position: absolute; top: -15px; font-family: var(--grad-font-mono);
                    font-size: 10px; font-weight: 700; transform: translateX(-50%); }
.grad-band-labels { display: flex; justify-content: space-between;
                    font-family: var(--grad-font-mono); font-size: 10px; opacity: 0.6;
                    margin-top: 3px; }

/* Funnel: stage bars, progressively indented and narrowed. */
.grad-stage { height: 34px; border: var(--grad-border); display: flex; align-items: center;
              padding: 0 11px; font-family: var(--grad-font-mono); font-size: 11px;
              font-weight: 700; letter-spacing: 0.08em; margin-bottom: 6px;
              background: var(--grad-paper-sunk); text-transform: uppercase; }
.grad-stage.rerank { background: var(--grad-attention); }
.grad-stage.context { background: var(--grad-verified); color: var(--grad-verified-ink); }
.grad-dropped { opacity: 0.45; }

/* Evolve lineage bars. */
.grad-lineage { display: flex; align-items: flex-end; gap: 4px; height: 190px;
                padding: 10px 0; }
.grad-lineage .bar { flex: 1 1 0; border: 1.5px solid var(--grad-ink);
                     background: var(--grad-paper-sunk); min-width: 6px; }
.grad-lineage .bar.best { background: var(--grad-attention); }
.grad-lineage .bar.champion { background: var(--grad-verified); }

/* Unified diff. */
.grad-diff { font-family: var(--grad-font-mono); font-size: 12px; line-height: 1.65;
             background: var(--grad-paper-raised); border: 1.5px solid var(--grad-ink); }
.grad-diff div { padding: 1px 9px; white-space: pre-wrap; }
.grad-diff .add { background: var(--grad-verified-tint); color: var(--grad-verified-ink); }
.grad-diff .del { background: var(--grad-broken-tint); color: var(--grad-broken-ink-2); }
.grad-diff .meta { background: var(--grad-paper-sunk); opacity: 0.7; }

/* Paper covers and figure placeholders are CSS, per the handoff's assets note. */
.grad-cover { width: 70px; height: 92px; border: 1.5px solid var(--grad-ink);
              flex: 0 0 70px;
              background: repeating-linear-gradient(180deg,
                  var(--grad-hatch-a) 0 6px, var(--grad-hatch-b) 6px 12px); }
.grad-cover.unread { border-style: dashed; }
.grad-figure { border: var(--grad-border); min-height: 160px; position: relative;
               background: repeating-linear-gradient(135deg,
                   var(--grad-hatch-a) 0 8px, var(--grad-hatch-b) 8px 16px);
               display: flex; align-items: center; justify-content: center; }
.grad-figure .tag { position: absolute; left: 0; bottom: 0; background: var(--grad-ink);
                    color: var(--grad-paper); font-family: var(--grad-font-mono);
                    font-size: 10px; padding: 3px 7px; }
"""


def _chat() -> str:
    return """
.grad-msg { padding: 9px 14px; }
.grad-msg .role { font-family: var(--grad-font-mono); font-size: 10px; opacity: 0.5;
                  text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px; }
.grad-msg.user { display: flex; flex-direction: column; align-items: flex-end; }
.grad-msg.user .bubble { max-width: 88%; border: var(--grad-border);
                         background: var(--grad-paper-raised); padding: 11px; font-size: 14px; }
.grad-msg.grad .bubble { padding-left: 23px; font-size: 14px; }
.grad-avatar { width: 16px; height: 16px; background: var(--grad-attention);
               border: 1.5px solid var(--grad-ink); display: inline-flex;
               align-items: center; justify-content: center;
               font-family: var(--grad-font-mono); font-size: 10px; font-weight: 700; }
.grad-msg code { background: var(--grad-paper-sunk); font-family: var(--grad-font-mono);
                 font-size: 12.5px; padding: 1px 4px; }
.grad-msg pre { background: var(--grad-paper-raised); border: 1.5px solid var(--grad-ink);
                padding: 10px; overflow: auto; font-size: 12.5px; }
.grad-msg h1, .grad-msg h2, .grad-msg h3 { font-family: var(--grad-font-serif);
                                           font-weight: 400; margin: 12px 0 6px; }
.grad-msg h1 { font-size: 28px; } .grad-msg h2 { font-size: 21px; } .grad-msg h3 { font-size: 19px; }
.grad-msg sup.ref { color: var(--grad-link); font-family: var(--grad-font-mono);
                    font-size: 10px; font-weight: 700; }

.grad-card { border: var(--grad-border); margin: 9px 14px; }
.grad-card > .head { display: flex; align-items: center; gap: 9px; padding: 6px 10px;
                     font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
                     text-transform: uppercase; letter-spacing: 0.1em; }
.grad-card > .head.attention { background: var(--grad-attention); color: var(--grad-ink); }
.grad-card > .head.ink { background: var(--grad-ink); color: var(--grad-paper); }
.grad-card > .head.broken { background: var(--grad-broken); color: #fff; }
.grad-card > .body { padding: 11px; background: var(--grad-paper-raised); }
.grad-card.gate { border-color: var(--grad-broken); }

/* A tool card's head is TOOL · name · what it was on, and the subject is the
   one part of it that is not a label: a path or a command, in its own case,
   ellipsised rather than allowed to push the chip off the end. */
.grad-card.tool > .head .subject { font-weight: 400; text-transform: none; letter-spacing: 0;
                                   opacity: 0.72; min-width: 0; overflow: hidden;
                                   white-space: nowrap; text-overflow: ellipsis; }
.grad-card.tool > .head .state { flex: 0 0 auto; }
/* Output is clipped by `agent.clip` at capture, but a 40-line result is still
   taller than the card should be in a transcript you are scrolling. */
.grad-card.tool > .body .grad-pre { max-height: 240px; overflow: auto; }
.grad-card.tool > .body .out { margin-top: 9px; }
.grad-card.tool > .body .out .grad-label { opacity: 0.5; margin-bottom: 4px; }

/* The agent statusline: always on screen, and the switch for the reasoning
   below it. The strip it replaced was `.grad-streaming`, which appeared only
   while a turn ran -- so the one piece of state worth reading at a glance was
   the one that came and went, and an idle session had nothing saying so.

   A `<button>`, because the whole bar is the click target, which means undoing
   the four properties a `<button>` arrives with. */
.grad-statusline {
    display: flex; align-items: center; gap: 9px; width: 100%;
    padding: 7px 14px; cursor: pointer; text-align: left;
    border: 0; border-top: var(--grad-border); flex: 0 0 auto;
    background: var(--grad-paper-sunk); color: var(--grad-ink);
    font: inherit; font-family: var(--grad-font-mono); font-size: 11px;
}
.grad-statusline:hover { background: var(--grad-paper-raised); }
.grad-statusline .block { width: 8px; height: 8px; flex: 0 0 8px;
                          background: var(--grad-rule-mid); }
.grad-statusline.running .block { background: var(--grad-ink);
                                  animation: gradblink 1.1s steps(1) infinite; }
.grad-statusline .state { font-weight: 700; letter-spacing: 0.14em; flex: 0 0 auto; }
.grad-statusline .activity { opacity: 0.62; min-width: 0; overflow: hidden;
                             white-space: nowrap; text-overflow: ellipsis; }
.grad-statusline .clock { opacity: 0.62; flex: 0 0 auto; }
/* The one part of the bar that is a control rather than a report, so it is the
   one part drawn as one. */
.grad-statusline .reasoning { flex: 0 0 auto; border: 1.5px solid var(--grad-ink);
                              padding: 1px 7px; letter-spacing: 0.08em; }
.grad-chat.reasoning-on .grad-statusline .reasoning {
    background: var(--grad-ink); color: var(--grad-paper); }

/* Reasoning is drawn either way and painted only when it is switched on: a
   toggle that rebuilt the transcript would take its scroll position with it,
   which is the same reason the poll never touches this window. */
.grad-reasoning { display: none; border: var(--grad-secondary); margin: 9px 14px 9px 37px;
                  background: var(--grad-paper-sunk); }
.grad-chat.reasoning-on .grad-reasoning { display: block; }
.grad-reasoning > .head {
    font-family: var(--grad-font-mono); font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em; opacity: 0.5;
    padding: 5px 10px 0;
}
.grad-reasoning > .body { padding: 4px 10px 9px; font-size: 12.5px; opacity: 0.72; }
.grad-reasoning > .body > *:first-child { margin-top: 0; }
.grad-reasoning > .body > *:last-child { margin-bottom: 0; }

.grad-composer { border-top: var(--grad-border); background: var(--grad-paper-sunk);
                 padding: 10px 14px; flex: 0 0 auto; }
.grad-composer .field { border: var(--grad-border); background: var(--grad-paper-raised); }
.grad-mention { font-family: var(--grad-font-mono); font-size: 10px; opacity: 0.55; }

/* Which conversation this is, and the opener for the rest of them. */
.grad-session-btn {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    border: 1.5px solid var(--grad-ink); padding: 5px 9px;
    max-width: 320px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
"""


def stylesheet() -> str:
    """The whole project stylesheet, generated from the constants above."""
    return "\n".join(
        [
            css_variables(),
            _base(),
            _shell(),
            _tiling(),
            _controls(),
            _data(),
            _chat(),
            _quasar_reset(),
        ]
    )
