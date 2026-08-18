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
    # -- role tokens -------------------------------------------------------
    # Everything above names a *colour*. The seven below name a *job*, and they
    # exist because the light palette gets to conflate jobs that a dark one
    # cannot. In cream-and-ink, `ink` is simultaneously the text on paper, the
    # fill of the app bar, and the text on a yellow chip. Invert the ground and
    # those three want to move in different directions: the text goes light, the
    # app bar must stay dark or it becomes the brightest thing on screen, and
    # the text on yellow must not move at all, because the yellow did not.
    #
    # So a fill and its foreground are named as a pair, and `test_ui_tokens.py`
    # holds every pair to 4.5:1 *in both palettes*. That is the property that
    # makes a second palette safe to add: a contrast rule you can only check by
    # eye is a contrast rule that is already broken somewhere you have not
    # looked.
    #
    # In this palette each one equals the colour it replaced, so the light
    # stylesheet renders exactly as it did before they existed.
    #: The emphasis ground: app bar, status bar, table head, focused title bar,
    #: active button. "Inverted from the page", which is ink here and a raised
    #: dark grey in the dark palette -- not the light one `ink` becomes.
    "fill": "#14100C",
    #: Text and hairline borders on `fill`.
    "fill-ink": "#F7F3E8",
    #: Text on the yellow. Fixed across both palettes, because `attention` is
    #: fixed across both -- a brand mark that changed value with the theme would
    #: stop being one.
    "on-attention": "#14100C",
    #: Text on the crimson. The `#fff` the handoff names ("`#A3122F` fill, white
    #: text"), given a name so the dark sheet cannot be searched for stray
    #: literals and find the one that is deliberate.
    "on-broken": "#FFFFFF",
    #: Text on `verified-tint`, which is a *tint* and not the fill: the fill
    #: stays teal in both palettes and carries `verified-ink`, while the tint
    #: inverts with the ground and needs a foreground that inverts with it.
    "verified-tint-ink": "#04302C",
    #: The hard shadow. Ink here; **not** `ink` in the dark palette, where an
    #: 8px offset block of near-white is a glow rather than a shadow.
    "shadow-ink": "#14100C",
    #: What shows through behind the JupyterLab iframe before it paints. Was a
    #: `#fff` literal, which is a white flash on every retile in a dark theme.
    "iframe-ground": "#FFFFFF",
    #: One per chart series, because a series fill is a fill like any other and
    #: the segment labels sit inside it. `base` tracks `paper` in both palettes
    #: (the series is `ink`, so its foreground is whatever ink is legible on);
    #: the other two are pinned dark and light respectively by what their fills
    #: do when the ground inverts.
    "on-series-base": "#F7F3E8",
    "on-series-alt": "#14100C",
    "on-series-third": "#FFFFFF",
}

#: The dark palette: the same keys, none added and none missing.
#:
#: Not a computed inversion. Inverting lightness mechanically gives a blue-grey
#: screen and a muddy yellow, and the two colours this design is *about* --
#: `attention` and `verified` -- are the two an inversion damages most. These
#: are chosen against the same rules the light table was: warm greys rather than
#: neutral ones, one accent per state, and a foreground for every fill.
#:
#: `attention`, `verified` and `broken` keep their hues. They are the vocabulary
#: -- "yellow needs you, teal passed, red broke" -- and a theme that renegotiated
#: them would be a different design rather than the same one at night.
DARK: dict[str, str] = {
    "ink": "#EFE8DA",
    "paper": "#17140F",
    "paper-raised": "#211D16",
    "paper-sunk": "#100E0A",
    "desk": "#0A0806",
    "rule-soft": "rgba(239,232,218,0.16)",
    "rule-mid": "rgba(239,232,218,0.32)",
    "attention": "#FFD400",
    # Lifted from #12A594: the light palette's teal is a *fill* under dark text,
    # and here it also has to read as a stroke on a dark ground -- the band, the
    # progress fill and the `ok` chip's border all draw with it.
    "verified": "#1FC7B3",
    "verified-ink": "#04302C",
    "verified-tint": "#0E2E29",
    "broken": "#D8324F",
    "broken-tint": "#2C0F17",
    "broken-ink": "#FF9AAB",
    "broken-ink-2": "#FF8299",
    "link": "#E8845C",
    "muted": "#A79E8B",
    "muted-2": "#8F8776",
    "literal": "#45D6C2",
    "hatch-a": "#1C1913",
    "hatch-b": "#252118",
    "ink-rule": "#4A443A",
    "row-alt": "#1B1712",
    "attention-row": "#241F0E",
    # The emphasis ground goes *up* from paper here, not down to near-black.
    # Dark UI convention and the handoff's own logic agree for once: elevation
    # reads as light, and an app bar darker than the window it sits on would be
    # a hole rather than a bar.
    "fill": "#2E2822",
    "fill-ink": "#EFE8DA",
    "on-attention": "#14100C",
    "on-broken": "#FFFFFF",
    "verified-tint-ink": "#8AE6D6",
    "shadow-ink": "#000000",
    "iframe-ground": "#1E1B16",
    "on-series-base": "#17140F",
    "on-series-alt": "#14100C",
    # `link` is light enough here that white on it is 2.4:1. The other two
    # foregrounds are unchanged by the inversion; this one had to move.
    "on-series-third": "#2A1008",
}

#: The two palettes by name. `light` is the handoff's and is the default
#: everywhere -- a theme setting that has never been touched resolves to it.
PALETTES: dict[str, dict[str, str]] = {"light": COLOUR, "dark": DARK}
DEFAULT_THEME = "light"

#: Every ground in the system, and the token that is legible on it.
#:
#: This is the table that makes a second palette safe. The rule "one accent per
#: state" is enforceable because `STATE_ACCENT` is total; the rule "text on a
#: fill can be read" needs the same treatment, and until there were two palettes
#: it did not have it -- the light one is legible by inspection and inspection
#: does not scale to a ground somebody inverts.
#:
#: `test_ui_tokens.py` holds every pair here to WCAG 4.5:1 **in both palettes**,
#: which is what caught `on-series-third`: white on the light palette's `link`
#: is 5.6:1 and white on the dark one's is 2.4:1, and nothing else would have
#: said so until a spend meter was unreadable at night.
FOREGROUND: dict[str, str] = {
    "paper": "ink",
    "paper-raised": "ink",
    "paper-sunk": "ink",
    "desk": "ink",
    "row-alt": "ink",
    "attention-row": "ink",
    "fill": "fill-ink",
    "attention": "on-attention",
    "broken": "on-broken",
    "verified": "verified-ink",
    "verified-tint": "verified-tint-ink",
    "broken-tint": "broken-ink",
    "iframe-ground": "ink",
}

#: The same claim for the chart ramp, kept separate because the keys on the left
#: are series names rather than palette entries.
SERIES_FOREGROUND: dict[str, str] = {
    "base": "on-series-base",
    "alt": "on-series-alt",
    "third": "on-series-third",
}

# The four state accents, and the rule that governs them. `one accent per state,
# never two in the same element` is a property of this mapping being total: a
# state that is not here has no accent, rather than borrowing one.
#:
#: Named by palette *key* rather than by value, so the mapping means the same
#: thing in both palettes -- "ok is whatever `verified` is here". Resolved
#: against the light palette below for the callers that want a colour.
STATE_ACCENT_KEYS: dict[str, str] = {
    "ok": "verified",
    "attention": "attention",
    "broken": "broken",
    "neutral": "ink",
}

STATE_ACCENT: dict[str, str] = {
    "ok": COLOUR["verified"],
    "attention": COLOUR["attention"],
    "broken": COLOUR["broken"],
    "neutral": COLOUR["ink"],
}

# Chart series colours, which are emphatically *not* states.
#
# The rule above governs one axis and left the other unguarded. A spend segment,
# a lineage generation and a funnel stage each answer "which one is this", not
# "how is this going" -- and the only fills in the palette were the state
# accents, so the charts borrowed them. The result was `attention` meaning five
# things at once (the brand mark, a primary action, an open-window count, chat
# spend, a new best candidate) and `verified` meaning two (a passing check, and
# GPU spend). A colour that means everything means nothing, which is the whole
# reason the accent rule exists in the first place.
#
# None of these three are chromatic accents, so a chart can be read as a chart
# and a yellow fill can go back to meaning "this needs you".
# `test_ui_tokens.py` holds the two mappings disjoint.
# Keyed the same way `STATE_ACCENT_KEYS` is, and for the same reason: the claim
# "a series is never a state accent" is about the *mapping*, so it has to be
# checkable in whichever palette is on screen rather than in one of them.
SERIES_KEYS: dict[str, str] = {
    "base": "ink",
    "alt": "muted",
    "third": "link",
}

SERIES: dict[str, str] = {name: COLOUR[key] for name, key in SERIES_KEYS.items()}

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
# Written as `var()` references rather than as interpolated hex, and that is the
# one change that made a second palette possible at all.
#
# These used to read `f"2px solid {COLOUR['ink']}"`, evaluated at import -- so
# `--grad-border` reached the browser as a literal `2px solid #14100C` and no
# amount of re-declaring `--grad-ink` further down could move it. Every
# structural rule in the sheet is built from these four, which meant every
# border and both shadows were pinned to the light palette by an f-string.
#
# Indirection through the custom property costs nothing (the browser resolves it
# at use) and buys the whole feature: one `:root` block per theme, and every
# derived value follows.
BORDER_STRUCTURAL = "2px solid var(--grad-ink)"
BORDER_HAIRLINE = "1px solid var(--grad-rule-soft)"
BORDER_SECONDARY = "1px dashed var(--grad-rule-mid)"
BORDER_PENDING = "2px dashed var(--grad-ink)"

# `shadow-ink`, not `ink`. The offset block is near-black in both palettes: in
# the dark one `ink` is near-white, and an 8px white slab under every window is
# a glow, which is the opposite of what a shadow is for.
SHADOW_SHELL = "8px 8px 0 var(--grad-shadow-ink)"
SHADOW_CARD = "6px 6px 0 var(--grad-shadow-ink)"

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


def palette(theme: str | None = None) -> dict[str, str]:
    """One palette by name, falling back to light rather than raising.

    An unknown name is what a settings file written by a newer version looks
    like from an older one, and the answer to that is the design's default --
    not a stylesheet that fails to generate and takes the window with it.
    """
    return PALETTES.get(str(theme or DEFAULT_THEME).lower(), COLOUR)


def resolved_theme() -> str:
    """The palette this workspace is set to, or the default. Never raises.

    One copy, because there were three: `ui/render.py`, `ui/splash.py` and
    `ui/state.py` each had the same try-settings-except-default, and one of them
    fell back to a literal `"light"` rather than to `DEFAULT_THEME` -- which is
    the drift `_check_number`'s docstring warns about, arriving on schedule.

    Here rather than in `core/settings.py` because the fallback is a *design*
    fact: an unreadable setting resolves to the palette the design ships, and
    `palette()` already makes the same choice for an unknown name.
    """
    try:
        from core import settings  # noqa: PLC0415 - keeps `core` off the import path

        return settings.theme()
    except Exception:  # noqa: BLE001 - a theme is never worth failing to draw
        return DEFAULT_THEME


def colour_variables(theme: str | None = None) -> str:
    """Just the colours, for one palette. The part that differs between themes."""
    active = palette(theme)
    lines = [f"    --grad-{name}: {value};" for name, value in active.items()]
    lines += [
        f"    --grad-series-{name}: {active[key]};" for name, key in SERIES_KEYS.items()
    ]
    return "\n".join(lines)


def css_variables(theme: str | None = None) -> str:
    """`:root` block. Everything else in the stylesheet reads from here.

    The non-colour half is emitted once, with the light palette, because none of
    it varies: a border is two pixels in both themes and its colour is a `var()`
    reference resolved at use. That split is what lets `themed_variables` emit a
    second block containing only the table that actually changes.
    """
    lines = [colour_variables(theme)]
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


#: The attribute the shell sets on `<html>`, and the selector the dark block is
#: scoped by. One attribute rather than a class because `ui/static/tiling.js`
#: and the Lab iframe both need to read the current theme without knowing what
#: else is on the element.
THEME_ATTRIBUTE = "data-grad-theme"


def themed_variables() -> str:
    """The override block for every palette that is not the default.

    Both palettes ship in one stylesheet, which is what makes switching instant
    and reload-free: `ui/app.py` adds the sheet once, at import, with
    `shared=True`, so there is no second injection to make -- the switch is one
    attribute on the document element and the cascade does the rest. That also
    satisfies the design's motion rule for free, since an attribute flip is an
    instant state swap rather than a transition.
    """
    blocks = []
    for name in PALETTES:
        if name == DEFAULT_THEME:
            continue
        blocks.append(
            f':root[{THEME_ATTRIBUTE}="{name}"] {{\n{colour_variables(name)}\n}}'
        )
    return "\n\n".join(blocks)


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

/* The one rule in here that is deliberately *not* scoped to the app root.
   Quasar renders a select's popup into a portal at `<body>` level, so it is
   not inside `.grad-app` and nothing above reaches it -- which is why the
   squared corners declared for `.grad-app .q-menu` have never applied to the
   three selects in this app. That was invisible while Quasar's white default
   sat under a cream design; against a dark one it is a white card in the
   middle of the screen. `ui.run(dark=False)` stays as it is: Quasar's dark
   mode would fight every token here, and this is all it was needed for. */
.q-menu {
    border-radius: 0 !important; box-shadow: none !important;
    background: var(--grad-paper-raised); color: var(--grad-ink);
    border: var(--grad-border);
}
.q-menu .q-item { font-family: var(--grad-font-mono); font-size: 12px; }
.q-menu .q-item.q-manual-focusable--focused,
.q-menu .q-item:hover { background: var(--grad-paper-sunk); }
.q-menu .q-item.q-item--active { background: var(--grad-fill); color: var(--grad-fill-ink); }
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
.grad-app a:hover { color: var(--grad-on-attention); background: var(--grad-attention); }
.grad-app ::selection { background: var(--grad-attention); color: var(--grad-on-attention); }
.grad-app :focus-visible { outline: 2px solid var(--grad-ink); outline-offset: 2px; }

.grad-mono { font-family: var(--grad-font-mono); }
.grad-serif { font-family: var(--grad-font-serif); }
.grad-label {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
}
/* The label's quiet sibling, for a name *inside* a component rather than over a
   section of one. All-caps letterspaced mono is a spice: it earns its shout on
   `CHAT` and `SURVIVORS, IN RANK ORDER`, and spends it on the word `output`
   repeated once per tool call down a transcript. Lowercase mono at half ink
   reads calmer and, for machine chrome, more honestly -- a terminal does not
   shout the word `stdout` at you either. */
.grad-sublabel {
    font-family: var(--grad-font-mono); font-size: 10px; font-weight: 400;
    letter-spacing: 0.06em; opacity: 0.5;
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
    display: flex; align-items: stretch; background: var(--grad-fill);
    color: var(--grad-fill-ink); border-bottom: var(--grad-border); flex: 0 0 auto;
}
.grad-appbar-cell {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px;
    border-right: 1px solid var(--grad-ink-rule);
    font-family: var(--grad-font-mono); font-size: 12px;
}
.grad-appbar-cell.right { border-right: 0; border-left: 1px solid var(--grad-ink-rule); }
.grad-appbar-cell.brand { border-right: 2px solid var(--grad-fill-ink); }
.grad-mark {
    width: 22px; height: 22px; background: var(--grad-attention);
    border: 2px solid var(--grad-fill-ink); display: flex; align-items: center;
    justify-content: center; font-family: var(--grad-font-mono); font-size: 13px;
    font-weight: 700; color: var(--grad-on-attention);
}
.grad-wordmark { font-family: var(--grad-font-mono); font-size: 15px; font-weight: 700;
                 letter-spacing: 0.22em; }
.grad-appbar .dim { opacity: 0.55; }
/* Three classes deep on purpose. These buttons are built by `kit.button`, so
   they carry `.grad-btn` and `.ghost` as well -- and `_controls()` is
   concatenated after this block, so at equal specificity its `.grad-btn`
   rules win the cascade: `color: var(--grad-ink)` on an ink app bar was an
   invisible button, and `.ghost`'s `border: 0` took the outline with it. The
   extra ancestor is what keeps these rules winning without caring where in
   the stylesheet they sit. */
.grad-appbar .grad-btn.grad-appbar-btn {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    padding: 3px 8px; border: 1.5px solid var(--grad-fill-ink); cursor: pointer;
    background: transparent; color: inherit;
}
.grad-appbar .grad-btn.grad-appbar-btn:hover { background: var(--grad-fill-ink); color: var(--grad-fill); }

/* The `⋯` button carries a menu, so it gets the caret's job: a little wider
   than a word button, and legible as a target rather than as punctuation.
   Deeper than the rule above, which would otherwise re-shrink the glyph. */
.grad-appbar .grad-btn.grad-appbar-btn.grad-dots {
    font-size: 15px; line-height: 1; padding: 1px 9px 4px; letter-spacing: 0.1em;
}

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
.grad-menu-row.open { background: var(--grad-fill); color: var(--grad-fill-ink); }
.grad-menu-row.open .mark, .grad-menu-row.open .hint { opacity: 0.7; }
.grad-menu-row.open:hover { background: var(--grad-broken); color: var(--grad-on-broken); }
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

/* The setup window's step row. A tab strip rather than a wizard's forward
   march: every step stays clickable, because a setup that has to be restarted to
   change the answer to question two is a setup people abandon at question
   three. Sticky, so the steps do not scroll away underneath a long form. */
.grad-steps {
    display: flex; gap: 0; border-bottom: var(--grad-border);
    position: sticky; top: 0; z-index: 2; background: var(--grad-paper);
}
.grad-step {
    display: flex; flex-direction: column; align-items: flex-start; gap: 1px;
    flex: 1 1 0; min-width: 0; padding: 8px 10px; cursor: pointer;
    text-align: left; background: transparent; color: var(--grad-ink);
    border: 0; border-right: var(--grad-border);
    font: inherit; font-family: var(--grad-font-mono); font-size: 12px;
}
.grad-step:last-child { border-right: 0; }
.grad-step:hover { background: var(--grad-paper-sunk); }
.grad-step .mark { font-weight: 700; opacity: 0.55; }
.grad-step .name {
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
}
.grad-step .hint {
    opacity: 0.55; font-size: 11px; width: 100%;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
.grad-step.open { background: var(--grad-fill); color: var(--grad-fill-ink); }
.grad-step.open .mark, .grad-step.open .hint { opacity: 0.7; }

.grad-statusbar {
    display: flex; align-items: center; gap: 14px; padding: 0 12px;
    background: var(--grad-fill); color: var(--grad-fill-ink);
    font-family: var(--grad-font-mono); font-size: 11px;
    height: 30px; flex: 0 0 auto; border-top: var(--grad-border);
}
.grad-statusbar .dim { opacity: 0.55; }
/* How many windows are open. Outlined rather than filled: a count is not a
   state, and this one was wearing `attention` -- the colour that means "this
   needs you" -- to report a number that needs nothing. Inverting it to a paper
   fill would have been no quieter (paper is *brighter* than the yellow it
   replaced, 0.89 relative luminance against 0.69); what makes it recede is
   dropping the fill, and with it the hue's claim on the eye. */
.grad-statusbar .count {
    border: 1.5px solid var(--grad-fill-ink); padding: 1px 6px; font-weight: 700;
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
    flex: 0 0 var(--grad-handle); background: var(--grad-fill); cursor: col-resize;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 3px; user-select: none;
}
.grad-handle.row { cursor: row-resize; flex-basis: var(--grad-handle);
                   flex-direction: row; }
.grad-handle span { width: 2px; height: 2px; background: var(--grad-fill-ink); display: block; }
.grad-handle.dragging { background: var(--grad-broken); }

.grad-window { display: flex; flex-direction: column; min-height: 0; flex: 1 1 auto;
               background: var(--grad-paper); overflow: hidden; }
.grad-window.focused .grad-titlebar { background: var(--grad-fill); color: var(--grad-fill-ink); }
.grad-window.focused .grad-titlebar .grad-winctl { color: var(--grad-fill-ink); }
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
.grad-iframe-host { position: absolute; border: 0; background: var(--grad-iframe-ground); z-index: 5; }
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
    background: var(--grad-fill); color: var(--grad-fill-ink);
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
    background: var(--grad-attention); color: var(--grad-on-attention);
}
.grad-window .grad-titlebar.grad-swap-target .grad-winctl { color: var(--grad-on-attention); }
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
.grad-btn.primary { background: var(--grad-attention); color: var(--grad-on-attention); }
.grad-btn.primary:hover { background: var(--grad-attention); filter: brightness(0.94); }
.grad-btn.ok { background: var(--grad-verified); color: var(--grad-verified-ink); }
.grad-btn.danger { background: var(--grad-broken); color: var(--grad-on-broken); }
.grad-btn.active { background: var(--grad-fill); color: var(--grad-fill-ink); }
.grad-btn.dashed { border: var(--grad-pending); background: transparent; opacity: 0.75; }
.grad-btn[disabled], .grad-btn.disabled {
    background: var(--grad-paper-sunk); opacity: 0.5; pointer-events: none;
}
/* `active` + disabled is a *badge*, not a dead control: "IN USE" on the
   current project is disabled precisely because it is already in effect. The
   general disabled rule above swaps the fill to sunk paper but leaves
   `.active`'s paper text in place -- paper on paper, illegible. Keep the
   inverted ink fill and most of its contrast. */
.grad-btn.active[disabled], .grad-btn.active.disabled {
    background: var(--grad-fill); color: var(--grad-fill-ink); opacity: 0.85;
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
.grad-chip.attention { background: var(--grad-attention); color: var(--grad-on-attention); }
.grad-chip.broken { background: var(--grad-broken); color: var(--grad-on-broken); }
.grad-chip.solid { background: var(--grad-fill); color: var(--grad-fill-ink); }
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
/* See `kit.pre`. A command or a URL has no columns to keep, so it wraps rather
   than growing the scrollbar that made a one-line fix into a two-axis gesture. */
.grad-pre.wrap { white-space: pre-wrap; overflow-wrap: anywhere; overflow-x: hidden; }
.grad-hr { height: 0; border: 0; border-top: var(--grad-secondary); margin: 12px 0; }
"""


def _data() -> str:
    """Bars, tables, stripes -- the shapes the ten data windows share."""
    return """
/* Segmented meters. Every segment here names a *kind of spend* -- chat against
   tools, sonnet against opus against GPU -- so all of them draw from `SERIES`
   and none from the state accents. `broken` is the exception and the proof: a
   resource over its ceiling is a state, and it is the only fill in a meter that
   still gets an accent.

   `muted` carries ink text rather than paper: at 0.24 relative luminance it is
   5.5:1 against ink and 3.2:1 against paper, and the segment labels are 10px. */
.grad-bar { display: flex; border: 2px solid var(--grad-ink); background: var(--grad-paper-raised);
            height: 22px; overflow: hidden; }
.grad-bar .seg { display: flex; align-items: center; justify-content: center;
                 font-family: var(--grad-font-mono); font-size: 10px; font-weight: 700;
                 overflow: hidden; white-space: nowrap; }
.grad-bar .seg.base { background: var(--grad-series-base); color: var(--grad-on-series-base); }
.grad-bar .seg.chat { background: var(--grad-series-base); color: var(--grad-on-series-base); }
.grad-bar .seg.tool { background: var(--grad-series-alt); color: var(--grad-on-series-alt); }
.grad-bar .seg.opus { background: var(--grad-series-third); color: var(--grad-on-series-third); }
.grad-bar .seg.broken { background: var(--grad-broken); color: var(--grad-on-broken); }
.grad-bar.thin { height: 12px; border-width: 1.5px; }

.grad-table { width: 100%; border-collapse: collapse; font-family: var(--grad-font-mono);
              font-size: 12px; }
.grad-table thead th {
    background: var(--grad-fill); color: var(--grad-fill-ink); text-align: left;
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
.grad-status-square.attention { background: var(--grad-attention); color: var(--grad-on-attention); }
.grad-status-square.broken { background: var(--grad-broken); color: var(--grad-on-broken); }

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

/* Funnel: a stage per row, each a full-width track holding a fill whose length
   is the count.
 *
 * It was four bars at hardcoded widths -- 1.0, 0.82, 0.64, 0.46 -- indented to
 * suggest a funnel. A decorative bar chart is a strange thing to put in the one
 * window whose docstring calls itself "the debugging surface for retrieval": it
 * drew the same shape whether a stage kept everything or nothing, and the
 * screenshot that prompted this showed three stages reading `-> 0` at
 * full width.
 *
 * The track is drawn rather than implied, because the empty part of it is the
 * information the window exists for: what this stage threw away. So the funnel
 * still narrows down the rows, but the silhouette is now the data and the gap to
 * the right of each fill is the loss.
 *
 * The label sits *over* the fill instead of inside it. A fill can be a few
 * pixels wide -- 15 chunks against 300 candidates is 5% -- and a label inside it
 * would be clipped at exactly the moment it matters most. That is what rules the
 * fill out of the accents and into a 30% ink wash: over `paper-raised` it
 * resolves to about #B6B6B1, which carries ink text at 9.6:1, so one label
 * style reads across the boundary at any width.
 *
 * A stage is `absolute` inside a `relative` track and both label spans are
 * `relative`, so DOM order alone paints them above the fill. */
.grad-stage { height: 34px; border: var(--grad-border); display: flex; align-items: center;
              padding: 0 11px; font-family: var(--grad-font-mono); font-size: 11px;
              font-weight: 700; letter-spacing: 0.08em; margin-bottom: 6px;
              background: var(--grad-paper-raised); text-transform: uppercase;
              position: relative; overflow: hidden; }
.grad-stage .fill { position: absolute; left: 0; top: 0; bottom: 0;
                    background: var(--grad-rule-mid); }
.grad-stage .name { position: relative; min-width: 0; overflow: hidden;
                    white-space: nowrap; text-overflow: ellipsis; }
.grad-stage .share { position: relative; margin-left: auto; padding-left: 9px;
                     opacity: 0.55; letter-spacing: 0; flex: 0 0 auto; }
/* The corpus is the index the funnel drew from, not a stage it passed through --
   there is no survival rate to draw for it. Sunk paper marks it as the frame
   around the measurement rather than a bar in it. */
.grad-stage.corpus { background: var(--grad-paper-sunk); }
/* Nothing came out of this stage. No fill, because zero is zero, and dimmed
   because a stage that ran and returned nothing is not the row you should be
   reading first. */
.grad-stage.empty { opacity: 0.45; }
/* Except when it is the last one. "Nothing reached the context" is the failure
   this whole window is opened to look at, and a grey row is the wrong way to
   report it. */
.grad-stage.broken { background: var(--grad-broken-tint); border-color: var(--grad-broken);
                     color: var(--grad-broken-ink); opacity: 1; }
.grad-dropped { opacity: 0.45; }

/* Evolve lineage bars. Ordinary generation, then a new best, then the current
   champion: an *ordinal* distinction rather than three categories, so it reads
   as one ramp getting darker -- sunk paper, `muted`, ink -- instead of as two
   borrowed accents. Yellow said "this needs you" of a generation that needs
   nothing, and teal said "verified" of a champion nothing has verified. */
.grad-lineage { display: flex; align-items: flex-end; gap: 4px; height: 190px;
                padding: 10px 0; }
.grad-lineage .bar { flex: 1 1 0; border: 1.5px solid var(--grad-ink);
                     background: var(--grad-paper-sunk); min-width: 6px; }
.grad-lineage .bar.best { background: var(--grad-series-alt); }
.grad-lineage .bar.champion { background: var(--grad-series-base); }

/* Unified diff. */
.grad-diff { font-family: var(--grad-font-mono); font-size: 12px; line-height: 1.65;
             background: var(--grad-paper-raised); border: 1.5px solid var(--grad-ink); }
.grad-diff div { padding: 1px 9px; white-space: pre-wrap; }
.grad-diff .add { background: var(--grad-verified-tint); color: var(--grad-verified-tint-ink); }
.grad-diff .del { background: var(--grad-broken-tint); color: var(--grad-broken-ink-2); }
.grad-diff .meta { background: var(--grad-paper-sunk); opacity: 0.7; }

/* A list beside a detail rail, in a pane that may be 320px wide.
 *
 * The rail used to be `flex: 0 0 520px`, which does not shrink -- so in any pane
 * narrower than the rail the list column (`min-width: 0`) was squeezed to
 * *zero* and the window showed a detail pane and nothing to select in it. Panes
 * go down to `--grad-min-pane`, and three columns on a 1280px screen are 410px
 * each, so this was the normal case rather than the edge one.
 *
 * `flex-wrap` is the fix rather than a smaller rail: the two hypothetical sizes
 * below add up to 620px, and under that the browser stacks them, which is the
 * one arrangement where both halves are still readable. No media query, because
 * the constraint is the *pane* and a media query would read the window. */
/* One wiki section: prose, then the facts it rests on. The rule above the
   heading separates sections without the boxes a card would add -- a page of
   six cards reads as six unrelated things, and these are one argument. */
.grad-wiki-section { border-top: var(--grad-hairline); padding-top: 11px; margin-top: 11px; }
.grad-wiki-section:first-of-type { border-top: 0; margin-top: 0; }
.grad-wiki-section .bubble { font-size: 14px; line-height: 1.6; }

.grad-split { flex-wrap: wrap; align-content: flex-start; }
.grad-split > .main { flex: 1 1 300px; min-width: 0; overflow-y: auto; }
.grad-split > .rail { flex: 0 1 320px; min-width: 0; max-width: 520px; overflow-y: auto;
                      background: var(--grad-paper-sunk); border-left: var(--grad-border); }

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
.grad-figure .tag { position: absolute; left: 0; bottom: 0; background: var(--grad-fill);
                    color: var(--grad-fill-ink); font-family: var(--grad-font-mono);
                    font-size: 10px; padding: 3px 7px; }
"""


def _chat() -> str:
    return """
/* `content-visibility` is here for the composer, which is nowhere near it.
 *
 * The textarea autogrows, and Quasar spells that: set `height: 1px`, read
 * `scrollHeight`, put the height back. The write dirties the layout tree to the
 * root and the read forces it clean again -- a full-document synchronous layout,
 * on every keystroke, whose cost is the size of the transcript above it. A
 * settled research conversation is tens of thousands of nodes once KaTeX has
 * expanded the maths, and measured in the browser that is 30-40ms per key. At
 * that point the typing is behind the typist by a whole word, which is the
 * symptom this file is fixing and the composer is not where it was fixable.
 *
 * Skipping the layout of messages scrolled out of view roughly halves it. The
 * `auto` in `contain-intrinsic-size` is the load-bearing half: it makes the
 * browser remember each message's real height once it has been rendered once,
 * so `scrollHeight` stays honest and `gradStickBottom` keeps pinning to a
 * bottom that does not move underneath it. A bare placeholder height would make
 * every scroll past an unrendered message a small jump.
 *
 * `auto` and not `hidden`: these have to render when scrolled to, and be found
 * by the browser's own find-in-page. */
.grad-msg { padding: 9px 14px; content-visibility: auto; contain-intrinsic-size: auto 140px; }
.grad-msg .role { font-family: var(--grad-font-mono); font-size: 10px; opacity: 0.5;
                  text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px; }
.grad-msg.user { display: flex; flex-direction: column; align-items: flex-end; }
.grad-msg.user .bubble { max-width: 88%; border: var(--grad-border);
                         background: var(--grad-paper-raised); padding: 11px; font-size: 14px; }
.grad-msg.grad .bubble { padding-left: 23px; font-size: 14px; }
.grad-avatar { width: 16px; height: 16px; background: var(--grad-attention); color: var(--grad-on-attention);
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
/* A figure the agent drew, in the transcript at the point it drew it. Bordered
   like every other framed thing here, and bounded in *height* as well as width:
   a tall figure that filled the pane would push the paragraph arguing about it
   off the screen, which is the one thing a result and its reading must not do
   to each other. `contain-intrinsic-size` on `.grad-msg` reserves 140px, so an
   image whose dimensions arrive late does not jump the scroll position. */
.grad-msg .grad-figure-img { display: block; margin: 9px 0 9px 23px; max-width: min(100%, 640px);
                             max-height: 420px; border: var(--grad-border);
                             background: var(--grad-paper-raised); }

.grad-card { border: var(--grad-border); margin: 9px 14px; }
.grad-card > .head { display: flex; align-items: center; gap: 9px; padding: 6px 10px;
                     font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
                     text-transform: uppercase; letter-spacing: 0.1em; }
.grad-card > .head.attention { background: var(--grad-attention); color: var(--grad-on-attention); }
.grad-card > .head.ink { background: var(--grad-fill); color: var(--grad-fill-ink); }
.grad-card > .head.broken { background: var(--grad-broken); color: var(--grad-on-broken); }
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
   taller than the card should be in a transcript you are scrolling.
 *
 * Borderless, and on sunk paper instead. The card already carries a 2px ink rule
 * and the head above it is a solid ink bar; a third bordered rectangle inside
 * that costs a run of high-frequency edges to say something the background shift
 * says on its own. Edge density is most of what makes a dense UI feel busy, and
 * a transcript is a column of these.
 *
 * **Wrapped, and this is the half that removes an interaction rather than a
 * line.** `.grad-pre` is `white-space: pre; overflow: auto`, so every call whose
 * output ran past the pane -- which is most of them, in a 410px column -- grew a
 * chunky horizontal scrollbar *inside* a card inside a scrolling transcript.
 * Three nested scroll axes to read a log line is the worst interaction in the
 * window, and the line was already there.
 *
 * Scoped to the tool card on purpose. `.grad-pre` elsewhere holds notebook
 * output, tables and the shell command an empty state tells you to run, and the
 * handoff gives mono to "anything the machine produced *or that must align*":
 * wrapping a 200-column dataframe at the pane edge destroys the alignment that
 * is the reason it is monospaced. Here the content is log lines, where a wrapped
 * line is strictly better than a clipped one. `overflow-wrap: anywhere` is for
 * the case the wrap cannot help with -- one unbroken 300-character path or
 * base64 blob, which would otherwise still force the scrollbar. */
.grad-card.tool > .body .grad-pre {
    max-height: 240px; overflow-x: hidden; overflow-y: auto;
    white-space: pre-wrap; overflow-wrap: anywhere;
    border: 0; background: var(--grad-paper-sunk);
}
/* The exception, and the reason the rule above is not simply "no borders in
   cards": a failed call is the one you opened the window for. It keeps its
   frame while the twenty successful ones above it lose theirs. */
.grad-card.tool > .body .grad-pre.broken {
    border: 1.5px solid var(--grad-broken); background: var(--grad-broken-tint);
}
.grad-card.tool > .body .out { margin-top: 9px; }
.grad-card.tool > .body .out .grad-sublabel { display: block; margin-bottom: 4px; }
/* A task's tail is one `pre` *per line*, so it is a stack of blocks rather than
   one block of lines. The 11px each carried was box padding when each had a
   border; with the borders gone it is 22px of gap between consecutive log lines.
   Tightened to the same 1px/9px the diff rows use, which is what a run of
   adjacent rows wants. `.broken` keeps its frame, so a stderr line still stands
   out of the stdout around it. */
.grad-card.tool > .body .tail .grad-pre { padding: 1px 9px; }
.grad-card.tool > .body .tail .grad-pre.broken { padding: 1px 8px; }
/* `OK` is the boring outcome, so it is the quietest mark on the card.
 *
 * It was a filled `verified` chip -- correct as a state, wrong as a *frequency*.
 * Every settled call carries one, so a column of twenty of them glowed harder
 * than the single crimson `FAILED` that is the only row anybody is scanning for.
 * Outlined here and filled nowhere else: `.grad-chip.ok` still means passing on
 * a preflight row or a verify banner, where it appears once and answers the
 * question the pane was opened to ask. Paper rather than ink, because the chip
 * sits on the card's ink head. */
.grad-card.tool > .head .grad-chip.ok {
    background: transparent; color: var(--grad-fill-ink);
    border: 1.5px solid var(--grad-fill-ink); opacity: 0.7;
}

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
/* How much of the context window is in use, and how close compaction is.
   Dimmed like the clock until it matters, because for most of a session the
   honest answer is "not yet" and a meter that shouts throughout is one nobody
   reads by the time it should be shouting. `warn` is an outline and
   `attention` is a fill: the first says the end is in sight, the second says
   the next turn may be the one that triggers a compaction. */
.grad-statusline .context { opacity: 0.62; flex: 0 0 auto; padding: 1px 5px; }
.grad-statusline .context.warn {
    opacity: 1; border: 1.5px solid var(--grad-ink); }
.grad-statusline .context.attention {
    opacity: 1; background: var(--grad-attention); color: var(--grad-on-attention); }
/* The two parts of the bar that are controls rather than reports, so they are
   the parts drawn as such. Effort sits to the left of the reasoning switch:
   both are about the agent's thinking, and this one changes what it does while
   that one changes what you see. */
.grad-statusline .reasoning, .grad-statusline .effort {
    flex: 0 0 auto; border: 1.5px solid var(--grad-ink);
    padding: 1px 7px; letter-spacing: 0.08em; }
/* Dashed until it is set, because "auto" is the absence of a choice rather than
   a level -- a solid chip reading `effort auto` looks like a setting someone
   picked. */
.grad-statusline .effort { cursor: pointer; border-style: dashed; opacity: 0.62; }
.grad-statusline .effort.set { border-style: solid; opacity: 1; }
.grad-statusline .effort:hover { background: var(--grad-paper-raised); }
.grad-chat.reasoning-on .grad-statusline .reasoning {
    background: var(--grad-fill); color: var(--grad-fill-ink); }

/* The compaction marker: a rule across the transcript, not a message. Nothing
   was said at this point -- what happened is that everything above it stopped
   being something the agent remembers first-hand. Drawn as a rule so it reads
   as a boundary rather than as a turn, and dashed for the same reason the rest
   of the design uses a dashed border: what is above the line is no longer
   solid. */
.grad-compaction { margin: 14px 14px; border-top: 1.5px dashed var(--grad-rule-mid);
                   padding-top: 10px; font-size: 12px; }
.grad-compaction > .head { align-items: flex-start; }
.grad-compaction .mark { font-family: var(--grad-font-mono); opacity: 0.55;
                         flex: 0 0 auto; line-height: 1.5; }
.grad-compaction .head .q-markdown, .grad-compaction .head p { margin: 0; opacity: 0.75; }
.grad-compaction .note { margin-top: 6px; font-size: 11px;
                         background: var(--grad-paper-sunk); border: var(--grad-secondary); }
/* A rewind is the same kind of boundary as a compaction -- the conversation
   above is not what the agent is working from -- so it is the same rule with a
   solid border: what is above a rewind *is* still solid, it is simply shorter.
   The messages it took back are drawn inside the disclosure, so their own
   margins are pulled in to keep them within the note rather than beside it. */
.grad-compaction.rewound { border-top: 1.5px solid var(--grad-rule-mid); }
.grad-compaction.rewound .note .grad-msg { padding: 6px 8px; }
.grad-compaction.rewound .note .grad-card { margin-left: 0; margin-right: 0; }

/* The ⟲ on a prompt, drawn inside the role label and as quiet as it is. Always
   there rather than revealed on hover: a control nobody can see is one nobody
   knows exists, and the alternative -- fading it in -- is the easing curve the
   design rules out. Pointing at the message lifts the whole label instead,
   which is an instant swap and needs nothing on the button itself.

   The lift is on `.role` and not on the button because opacity on a parent caps
   its children: `.grad-msg .role` is already 0.5, and no rule on something
   inside it can paint above that. */
.grad-rewind { border: 0; background: none; padding: 0 3px; cursor: pointer;
               font-family: var(--grad-font-mono); font-size: 11px; line-height: 1;
               color: var(--grad-ink); }
.grad-msg.user:hover .role { opacity: 0.85; }
.grad-rewind:focus-visible { outline: 1.5px solid var(--grad-ink); outline-offset: 1px; }

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

/* The other half of the typing-lag fix, and the half that removes the cause
 * rather than reducing it.
 *
 * A textarea that grows with its content is the right control, and Quasar's
 * way of getting one is `autogrow`: on every input event, set `height: 1px`,
 * read `scrollHeight`, put the height back. The write invalidates layout to the
 * root and the read forces it clean again -- a full-document synchronous layout
 * *inside the keystroke handler*, so its cost is the size of the transcript and
 * it is paid once per key. Measured against a 77,700-node conversation: typing
 * twenty characters blocked the main thread for 289ms.
 *
 * `field-sizing: content` asks the browser for the same behaviour and lets it
 * do the sizing during its own layout pass, where it belongs. Nothing is read
 * back, so nothing is forced: the same twenty characters block for 0.2ms and
 * the layout they imply is one 13ms pass for the whole burst rather than twenty
 * separate ones. O(frames), not O(keystrokes) -- and a person typing quickly is
 * precisely the case where those two diverge.
 *
 * The `autogrow` prop is gone from both composers, because leaving it on would
 * keep the measurement above exactly as it was; this rule replaces it rather
 * than assisting it. `min-height` is the one row `rows="1"` used to give, and
 * `max-height` is what `autogrow` never had -- a pasted stack trace grew the
 * box until it ate the transcript.
 *
 * WebView2 is evergreen and the app's floor is well past this property, but a
 * browser without it would get a one-line box with an inner scrollbar, so the
 * fallback asks for a few rows and lets it scroll. Usable, not lovely, and not
 * reachable on the platform this ships to. */
.grad-composer .field textarea,
.grad-wiki-ask .field textarea {
    field-sizing: content;
    min-height: 1lh;
    max-height: 40vh;
}
@supports not (field-sizing: content) {
    .grad-composer .field textarea,
    .grad-wiki-ask .field textarea { min-height: 4lh; max-height: 40vh; overflow-y: auto; }
}
.grad-mention { font-family: var(--grad-font-mono); font-size: 10px; opacity: 0.55; }

/* Which conversation this is, and the opener for the rest of them.
 *
 * A session is named after the first thing asked in it, so this button holds a
 * sentence somebody wrote -- and it is built by `kit.button`, which means it
 * arrived carrying `.grad-btn`'s `text-transform: uppercase` and 0.08em tracking.
 * Those belong to buttons that hold *commands*: `SEND`, `STOP`, `+ NEW`. Applied
 * to prose they shout it and cost it legibility at the same time, which the
 * truncated `I WANT TO MAKE AN AGENTIC SYSTEM THAT U…` demonstrated in one line.
 * Both are undone here rather than in `.grad-btn`, where they are right. */
.grad-session-btn {
    font-family: var(--grad-font-mono); font-size: 11px; font-weight: 700;
    text-transform: none; letter-spacing: 0;
    border: 1.5px solid var(--grad-ink); padding: 5px 9px;
    max-width: 320px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
"""


def stylesheet() -> str:
    """The whole project stylesheet, generated from the constants above."""
    return "\n".join(
        [
            css_variables(),
            themed_variables(),
            _base(),
            _shell(),
            _tiling(),
            _controls(),
            _data(),
            _chat(),
            _quasar_reset(),
        ]
    )
