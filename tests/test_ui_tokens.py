"""The design system, as assertions.

The handoff states its rules in prose -- "radius: 0 everywhere", "no blur
shadows anywhere", "one accent per state, never two in the same element" -- and
a rule you can only check by eye is a rule that drifts on the third window
somebody adds. These tests make the checkable half checkable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ui import fonts, tokens

UI_DIR = Path(tokens.__file__).resolve().parent
HEX = re.compile(r"#[0-9A-Fa-f]{3,8}\b")


def ui_sources() -> list[Path]:
    return sorted(p for p in UI_DIR.rglob("*.py") if "__pycache__" not in p.parts)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token",
    [
        "ink", "paper", "paper-raised", "paper-sunk", "desk", "rule-soft", "rule-mid",
        "attention", "verified", "verified-ink", "verified-tint", "broken", "broken-tint",
        "broken-ink", "link", "muted", "muted-2", "literal", "hatch-a", "hatch-b",
    ],
)
def test_every_token_in_the_handoff_table_exists(token):
    assert token in tokens.COLOUR
    assert f"--grad-{token}:" in tokens.css_variables()


@pytest.mark.parametrize(
    "name,value",
    [("ink", "#14100C"), ("paper", "#F7F3E8"), ("attention", "#FFD400"),
     ("verified", "#12A594"), ("broken", "#A3122F"), ("link", "#B04A2C")],
)
def test_the_colours_are_the_ones_the_handoff_specifies(name, value):
    """Fidelity is stated as high and the colours as final, so a typo in a hex
    digit is a bug rather than a preference."""
    assert tokens.COLOUR[name] == value


def test_every_state_accent_is_a_real_token():
    assert set(tokens.STATE_ACCENT) == {"ok", "attention", "broken", "neutral"}
    assert set(tokens.STATE_ACCENT.values()) <= set(tokens.COLOUR.values())
    # Keyed rather than valued, so the mapping means the same thing in a palette
    # it was not written against.
    assert set(tokens.STATE_ACCENT_KEYS) == set(tokens.STATE_ACCENT)
    for theme in tokens.PALETTES:
        assert set(tokens.STATE_ACCENT_KEYS.values()) <= set(tokens.palette(theme))


def test_every_series_colour_is_a_real_token():
    assert set(tokens.SERIES.values()) <= set(tokens.COLOUR.values())
    for name in tokens.SERIES:
        assert f"--grad-series-{name}:" in tokens.css_variables()
    for theme in tokens.PALETTES:
        assert set(tokens.SERIES_KEYS.values()) <= set(tokens.palette(theme))


def test_no_chart_series_borrows_a_chromatic_state_accent():
    """The other half of "one accent per state".

    The rule governed which accent a *state* may use and said nothing about what a
    chart series may use, so the charts reached for the only fills in the palette
    and `attention` ended up meaning five things at once -- brand mark, primary
    action, open-window count, chat spend, new-best candidate. A spend segment is
    not a state, so it may not wear a state's colour.

    `neutral` is excluded because it is the absence of an accent: ink is the
    system's text and structure colour, and a series is welcome to it.
    """
    chromatic = {
        tokens.STATE_ACCENT["ok"],
        tokens.STATE_ACCENT["attention"],
        tokens.STATE_ACCENT["broken"],
    }
    assert set(tokens.SERIES.values()) & chromatic == set()
    # And in every palette, which is the version of the claim that survives a
    # second one being added: a dark `link` that happened to land on the dark
    # `broken` would put a state's colour in a chart without changing a rule.
    for theme in tokens.PALETTES:
        active = tokens.palette(theme)
        accents = {active[tokens.STATE_ACCENT_KEYS[s]] for s in ("ok", "attention", "broken")}
        series = {active[key] for key in tokens.SERIES_KEYS.values()}
        assert series & accents == set(), theme


@pytest.mark.parametrize(
    "selector",
    [
        # Every chart fill in the system, and what it must resolve through. A new
        # one added with a raw `var(--grad-attention)` fails here rather than in
        # somebody's eye three windows later.
        ".grad-bar .seg.base", ".grad-bar .seg.chat", ".grad-bar .seg.tool",
        ".grad-bar .seg.opus", ".grad-lineage .bar.best", ".grad-lineage .bar.champion",
    ],
)
def test_chart_fills_are_drawn_from_the_series_ramp(selector):
    sheet = tokens.stylesheet()
    block = sheet.split(selector, 1)[1].split("}", 1)[0]
    assert "--grad-series-" in block, f"{selector} does not use a series colour: {block}"


# ---------------------------------------------------------------------------
# the second palette
# ---------------------------------------------------------------------------
def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.1 relative luminance, for the contrast check below."""
    value = hex_colour.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    channels = []
    for i in (0, 2, 4):
        c = int(value[i : i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


def test_both_palettes_carry_exactly_the_same_tokens():
    """A key in one and not the other is a rule that renders as `unset` -- which
    is not a visible failure, it is the *inherited* colour, so the element looks
    plausible and is wrong."""
    assert set(tokens.DARK) == set(tokens.COLOUR)


@pytest.mark.parametrize("theme", sorted(tokens.PALETTES))
def test_every_ground_carries_legible_text_in_every_palette(theme):
    """The rule the light palette got to satisfy by inspection.

    Inspection does not survive a second palette: `on-series-third` was white on
    `link`, which is 5.6:1 in cream and 2.4:1 in the dark, and the first anyone
    would have known is an unreadable spend meter at night.
    """
    active = tokens.palette(theme)
    failures = []
    for ground, text in tokens.FOREGROUND.items():
        ratio = contrast(active[ground], active[text])
        if ratio < 4.5:
            failures.append(f"{ground}/{text} = {ratio:.2f}:1")
    for series, text in tokens.SERIES_FOREGROUND.items():
        ratio = contrast(active[tokens.SERIES_KEYS[series]], active[text])
        if ratio < 4.5:
            failures.append(f"series-{series}/{text} = {ratio:.2f}:1")
    assert not failures, f"{theme}: " + ", ".join(failures)


def hue_degrees(hex_colour: str) -> float:
    import colorsys

    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


@pytest.mark.parametrize("theme", sorted(tokens.PALETTES))
def test_the_three_state_accents_stay_distinguishable(theme):
    """"One accent per state" is only information if the states do not converge.

    Measured as **hue** separation and deliberately not as contrast ratio. The
    first version of this test used the luminance formula above and failed on a
    perfectly good pair -- the dark palette's teal and its yellow are 1.48:1 and
    are not remotely confusable, because a contrast ratio says how legible one
    is *on* the other and nothing about telling two fills apart side by side.
    Yellow, teal and crimson are read by hue; that is the thing to hold.
    """
    active = tokens.palette(theme)
    hues = {
        state: hue_degrees(active[tokens.STATE_ACCENT_KEYS[state]])
        for state in ("ok", "attention", "broken")
    }
    for a, b in (("ok", "attention"), ("ok", "broken"), ("attention", "broken")):
        gap = abs(hues[a] - hues[b]) % 360
        assert min(gap, 360 - gap) >= 30, f"{theme}: {a} and {b} share a hue"


def test_the_accents_keep_their_hues_across_the_two_palettes():
    """The vocabulary is "yellow needs you, teal passed, red broke". A dark theme
    that renegotiated that would be a different design rather than the same one
    at night -- so the values may move for legibility and the hues may not."""
    for state in ("ok", "attention", "broken"):
        key = tokens.STATE_ACCENT_KEYS[state]
        gap = abs(hue_degrees(tokens.COLOUR[key]) - hue_degrees(tokens.DARK[key])) % 360
        assert min(gap, 360 - gap) <= 20, state


def test_the_dark_palette_is_actually_dark():
    """Stated as an assertion because the failure mode is subtle: a palette that
    inverted the text and forgot a ground reads as light with white text."""
    dark = tokens.palette("dark")
    for ground in ("paper", "paper-raised", "paper-sunk", "desk"):
        assert relative_luminance(dark[ground]) < 0.08, ground
    assert relative_luminance(dark["ink"]) > 0.5


def test_the_emphasis_ground_does_not_become_the_brightest_thing_on_screen():
    """The whole reason `fill` exists. The app bar, the status bar and every
    table head are `background: var(--grad-fill)`, and they were
    `var(--grad-ink)` -- so a palette that only swapped ink and paper would have
    given the dark theme a white app bar and white table headers."""
    dark = tokens.palette("dark")
    assert relative_luminance(dark["fill"]) < relative_luminance(dark["ink"])
    # And still distinct from the page it sits on, or the bar stops being one.
    assert contrast(dark["fill"], dark["paper"]) > 1.15


def test_the_hard_shadow_never_becomes_a_glow():
    """`SHADOW_SHELL` is an 8px offset block. Drawn in `ink` it is near-black on
    cream and near-*white* in the dark palette, which is a glow."""
    assert "var(--grad-shadow-ink)" in tokens.SHADOW_SHELL
    for theme in tokens.PALETTES:
        assert relative_luminance(tokens.palette(theme)["shadow-ink"]) < 0.05, theme


def test_the_switch_is_one_attribute_and_ships_in_the_same_sheet():
    """Both palettes travel in one stylesheet because `ui/app.py` adds it once,
    at import, with `shared=True` -- there is no second injection to make, so a
    theme change is an attribute on `<html>` and the cascade does the rest. That
    is also what keeps it inside the design's motion rule: an attribute flip is
    an instant state swap, not a transition."""
    sheet = tokens.stylesheet()
    assert f':root[{tokens.THEME_ATTRIBUTE}="dark"]' in sheet
    for name, value in tokens.DARK.items():
        assert f"--grad-{name}: {value};" in sheet, name
    # The non-colour half is emitted once: a second copy would be a second place
    # for the handle width to disagree with `layout.py`.
    assert sheet.count("--grad-handle:") == 1


def test_every_palette_declares_the_scheme_the_browser_draws_in():
    """The scrollbars were the one part of the inversion no token could reach.

    Everything else here dresses something the app renders; a scrollbar is drawn
    by the browser, which takes its colour from `color-scheme` alone -- so
    without this the dark palette kept light scrollbars on every pane that
    overflowed. It is emitted from the *colour* half on purpose: the non-colour
    half ships once, with the light palette, and a declaration there would pin
    every theme to `light` with no way to override it."""
    for name in tokens.PALETTES:
        assert name in tokens.COLOR_SCHEME, name
        assert f"color-scheme: {tokens.COLOR_SCHEME[name]};" in tokens.colour_variables(name)
    # One per palette in the shipped sheet: `:root` and the dark override block.
    assert tokens.stylesheet().count("color-scheme:") == len(tokens.PALETTES)


def test_an_unknown_theme_falls_back_rather_than_failing():
    """What a settings file written by a newer version looks like from an older
    one. The answer is the design's default, not a stylesheet that will not
    generate and takes the window with it."""
    assert tokens.palette("solarized") == tokens.COLOUR
    assert tokens.palette(None) == tokens.COLOUR
    assert tokens.palette("DARK") == tokens.DARK


# ---------------------------------------------------------------------------
# the structural rules
# ---------------------------------------------------------------------------
def test_radius_is_zero_everywhere():
    sheet = tokens.stylesheet()
    for match in re.finditer(r"border-radius:\s*([^;!]+)", sheet):
        assert match.group(1).strip() in ("0", "0px"), match.group(0)


def test_no_shadow_in_the_system_has_a_blur():
    """`8px 8px 0` and `6px 6px 0`. A third length that is not zero is a blur,
    and there are none."""
    for shadow in (tokens.SHADOW_SHELL, tokens.SHADOW_CARD):
        parts = shadow.split()
        assert parts[2] == "0", shadow
    # The two custom properties are resolved rather than skipped, so a shadow
    # reached through `var()` is held to the same rule as a literal one.
    resolved = (
        tokens.stylesheet()
        .replace("var(--grad-shadow-shell)", tokens.SHADOW_SHELL)
        .replace("var(--grad-shadow-card)", tokens.SHADOW_CARD)
    )
    for match in re.finditer(r"box-shadow:\s*([^;]+)", resolved):
        value = match.group(1).replace("!important", "").strip()
        if value == "none":
            continue
        assert not value.startswith("var("), f"unresolved shadow variable: {value}"
        assert value.split()[2] == "0", value


def test_the_structural_border_is_two_pixels_of_ink():
    """Through the custom property rather than the literal.

    It used to interpolate `COLOUR['ink']` at import, which put `2px solid
    #14100C` in the sheet -- so `--grad-border` was pinned to the light palette
    by an f-string and no re-declaration of `--grad-ink` could move it. The
    claim is unchanged; what it resolves through is."""
    assert tokens.BORDER_STRUCTURAL == "2px solid var(--grad-ink)"
    assert "--grad-border: 2px solid var(--grad-ink);" in tokens.css_variables()


def test_the_minimum_pane_matches_the_layout_model():
    """Two modules enforce this number; they must agree or a drag settles to a
    width the server immediately rewrites."""
    from ui import layout

    assert tokens.MIN_PANE_PX == layout.MIN_PANE_PX


def test_the_handle_is_eight_pixels():
    assert tokens.HANDLE_WIDTH == 8
    assert "--grad-handle: 8px" in tokens.css_variables()


def test_the_blink_is_the_only_animation():
    """"only two [motions] -- the 1.1s step blink for carets/live indicators, and
    instant state swaps. No easing curves, no fades, no skeleton shimmer.\""""
    sheet = tokens.stylesheet()
    assert sheet.count("@keyframes") == 1
    assert "gradblink" in sheet
    for match in re.finditer(r"animation:\s*([^;]+)", sheet):
        assert "gradblink" in match.group(1)
    assert "transition:" not in sheet


# ---------------------------------------------------------------------------
# single source
# ---------------------------------------------------------------------------
def test_no_module_outside_tokens_spells_a_hex_colour():
    """The single-source rule. A window that hardcodes `#12A594` is a window
    that will not follow the next change to the palette."""
    offenders = {}
    for path in ui_sources():
        if path.name in ("tokens.py", "jupyter_theme.py"):
            continue
        found = HEX.findall(path.read_text(encoding="utf-8"))
        if found:
            offenders[path.name] = sorted(set(found))
    # `#fff` on a crimson fill is the one exception the design itself names
    # ("`#A3122F` fill, white text"), and it is not a palette entry.
    offenders = {k: [v for v in vals if v.lower() not in ("#fff", "#ffffff")]
                 for k, vals in offenders.items()}
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}


def test_the_generated_stylesheet_is_a_pure_function_of_the_tokens():
    assert tokens.stylesheet() == tokens.stylesheet()


def test_the_stylesheet_covers_every_component_class_the_kit_emits():
    """A primitive whose class has no rule renders as unstyled text, which is
    the failure mode hardest to notice in review."""
    sheet = tokens.stylesheet()
    for name in (
        "grad-shell", "grad-appbar", "grad-dots", "grad-menu-row", "grad-statusbar", "grad-tiles",
        "grad-column", "grad-slot", "grad-handle", "grad-window", "grad-titlebar",
        "grad-body", "grad-btn", "grad-chip", "grad-kv", "grad-bar", "grad-progress",
        "grad-status-square", "grad-band", "grad-stage", "grad-lineage", "grad-diff",
        "grad-cover", "grad-figure", "grad-table", "grad-row", "grad-card",
        "grad-composer", "grad-msg", "grad-pre", "grad-note", "grad-empty",
        "grad-iframe-host", "grad-iframe-anchor", "grad-caret", "grad-label",
        "grad-sublabel",
    ):
        assert f".{name}" in sheet, name


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------
def test_the_three_families_are_the_ones_the_design_names():
    assert set(fonts.FAMILIES) == {"Space Grotesk", "JetBrains Mono", "Instrument Serif"}


def test_nothing_vendored_means_a_google_fonts_link(tmp_path):
    html = fonts.head_html(tmp_path)
    assert "fonts.googleapis.com" in html
    assert "@font-face" not in html


def test_a_vendored_family_gets_a_font_face_rule(tmp_path):
    (tmp_path / "jetbrains-mono-400.woff2").write_bytes(b"")
    html = fonts.head_html(tmp_path)
    assert "@font-face" in html
    assert "jetbrains-mono-400.woff2" in html
    # The other two are still missing, so the link stays -- partial vendoring
    # cuts the dependency down rather than all-or-nothing.
    assert "fonts.googleapis.com" in html


def test_everything_vendored_means_no_network(tmp_path):
    for stem, weights in fonts.FAMILIES.values():
        for weight in weights:
            (tmp_path / f"{stem}-{weight}.woff2").write_bytes(b"")
    html = fonts.head_html(tmp_path)
    assert "fonts.googleapis.com" not in html
    assert fonts.vendored(tmp_path) == set(fonts.FAMILIES)
