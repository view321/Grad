"""The JupyterLab half of the design.

The notebook window is a real Lab iframe on its own port, so its interior cannot
be styled from the host page. `ui/jupyter_theme.py` emits the same tokens as
JupyterLab's `custom.css`, and `tools/lab.py` starts Lab with the flag that
loads it. Three things have to hold or the seam becomes visible:

  1. the file on disk still matches the tokens it was generated from,
  2. Lab is actually started with `--custom-css`,
  3. the selected theme is a light one, since the sheet re-tokens a light base.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ui import jupyter_theme, tokens

REPO = Path(jupyter_theme.__file__).resolve().parents[1]


def config_dir() -> Path:
    return REPO / "config" / "jupyter"


def test_the_generated_sheet_is_checked_in_and_current():
    """Generated rather than hand-written so the notebook interior cannot drift
    from the chrome above it. Regenerate with:

        python -m ui.jupyter_theme --write
    """
    path = jupyter_theme.repo_path()
    assert path.exists(), "run `python -m ui.jupyter_theme --write`"
    assert path.read_text(encoding="utf-8") == jupyter_theme.stylesheet()


def test_it_lands_where_jupyterlab_looks():
    """JupyterLab 4 loads `{JUPYTER_CONFIG_DIR}/custom/custom.css`, and
    `tools/lab.py` sets `JUPYTER_CONFIG_DIR` to `config/jupyter`."""
    assert jupyter_theme.repo_path() == config_dir() / "custom" / "custom.css"


def test_lab_is_started_with_the_flag_that_loads_it():
    """Without `--custom-css` JupyterLab ignores the file entirely, and the
    iframe renders as stock Lab inside Grad's own chrome."""
    source = (REPO / "tools" / "lab.py").read_text(encoding="utf-8")
    assert '"--custom-css"' in source
    assert 'JUPYTER_CONFIG_DIR' in source


def test_the_selected_theme_is_a_light_one():
    """`custom.css` cannot register a named theme, so the sheet re-tokens a base
    -- and it re-tokens the light one. Leaving JupyterLab Dark selected would
    put dark defaults under cream overrides."""
    overrides = json.loads((config_dir() / "overrides.json").read_text(encoding="utf-8"))
    theme = overrides["@jupyterlab/apputils-extension:themes"]["theme"]
    assert "Dark" not in theme


def test_the_ruler_the_handoff_asks_to_leave_alone_is_left_alone():
    overrides = json.loads((config_dir() / "overrides.json").read_text(encoding="utf-8"))
    editor = overrides["@jupyterlab/fileeditor-extension:plugin"]["editorConfig"]
    assert editor["rulers"] == [88]


@pytest.mark.parametrize(
    "variable",
    [
        "--jp-layout-color0", "--jp-layout-color1", "--jp-layout-color2",
        "--jp-border-color0", "--jp-border-color1", "--jp-border-color2",
        "--jp-cell-editor-background", "--jp-code-font-family", "--jp-content-font-family",
    ],
)
def test_every_variable_the_handoff_names_is_set(variable):
    assert f"{variable}:" in jupyter_theme.stylesheet()


@pytest.mark.parametrize(
    "selector",
    [".jp-InputArea-prompt", ".jp-OutputArea-output", ".jp-RenderedText"],
)
def test_every_class_the_handoff_names_is_styled(selector):
    assert selector in jupyter_theme.stylesheet()


def test_the_sheet_uses_the_same_palette_as_the_host():
    sheet = jupyter_theme.stylesheet()
    for name in ("ink", "paper", "attention", "verified", "broken"):
        assert tokens.COLOUR[name] in sheet


def test_no_colour_in_the_sheet_is_outside_the_palette():
    """Same single-source rule as the host stylesheet: a hex here that is not a
    token is a colour that will not follow the next palette change."""
    allowed = {v.lower() for v in tokens.COLOUR.values()} | {"#fff", "#ffffff"}
    found = {m.group(0).lower() for m in re.finditer(r"#[0-9A-Fa-f]{3,8}\b", jupyter_theme.stylesheet())}
    assert found <= allowed, found - allowed


def test_radius_and_blur_are_gone_inside_lab_too():
    sheet = jupyter_theme.stylesheet()
    assert "--jp-border-radius: 0px;" in sheet
    assert "border-radius: 0 !important" in sheet
    for match in re.finditer(r"--jp-elevation-z\d+:\s*([^;]+)", sheet):
        assert match.group(1).strip() == "none"
