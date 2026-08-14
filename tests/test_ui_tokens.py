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
    assert tokens.BORDER_STRUCTURAL == f"2px solid {tokens.COLOUR['ink']}"


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
        "grad-shell", "grad-appbar", "grad-opener", "grad-statusbar", "grad-tiles",
        "grad-column", "grad-slot", "grad-handle", "grad-window", "grad-titlebar",
        "grad-body", "grad-btn", "grad-chip", "grad-kv", "grad-bar", "grad-progress",
        "grad-status-square", "grad-band", "grad-stage", "grad-lineage", "grad-diff",
        "grad-cover", "grad-figure", "grad-table", "grad-row", "grad-card",
        "grad-composer", "grad-msg", "grad-pre", "grad-note", "grad-empty",
        "grad-iframe-host", "grad-iframe-anchor", "grad-caret", "grad-label",
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
