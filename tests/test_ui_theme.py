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


def test_the_checked_in_selection_matches_the_checked_in_sheet():
    """`custom.css` cannot register a named theme, so the sheet re-tokens a base
    and the base has to match the palette -- dark defaults under cream overrides
    is Lab's own chrome keeping the wrong theme in every corner the sheet does
    not name. The repository copy is the light one; `install()` is what writes a
    matching pair into a workspace."""
    overrides = json.loads((config_dir() / "overrides.json").read_text(encoding="utf-8"))
    theme = overrides["@jupyterlab/apputils-extension:themes"]["theme"]
    assert theme == jupyter_theme.LAB_BASE["light"]


def test_every_palette_names_a_base_to_re_token():
    """A palette with no base would install `custom.css` over whichever theme was
    selected last, which is the mismatch above with no way to notice it."""
    assert set(jupyter_theme.LAB_BASE) == set(tokens.PALETTES)


def test_installing_a_palette_writes_a_matching_pair(workspace):
    """The two files are read at different times by different loaders, and a
    workspace where they disagree is the failure this pairing exists to stop."""
    result = jupyter_theme.install("dark")
    assert result["error"] is None

    sheet = jupyter_theme.target().read_text(encoding="utf-8")
    assert tokens.DARK["paper"] in sheet
    assert tokens.COLOUR["paper"] not in sheet

    overrides = json.loads(
        (jupyter_theme.target().parent.parent / "overrides.json").read_text(encoding="utf-8")
    )
    assert overrides["@jupyterlab/apputils-extension:themes"]["theme"] == "JupyterLab Dark"


def test_installing_seeds_a_workspace_that_has_no_config_directory(workspace):
    """The recommended layout keeps the workspace out of the checkout, and
    `JUPYTER_CONFIG_DIR` points into the workspace -- so `--custom-css` and
    `--ServerApp.config_file` have both been aimed at files that do not exist on
    every install that took the advice."""
    result = jupyter_theme.install("light")

    assert result["error"] is None
    server_config = jupyter_theme.target().parent.parent / "jupyter_server_config.py"
    assert server_config.is_file(), "the framing-header config has to reach the workspace"
    assert jupyter_theme.target().is_file()


def test_installing_does_not_overwrite_an_edited_server_config(workspace):
    """`custom.css` is generated and is rewritten every time. The other two are
    documents somebody may have edited, so they are seeded and then left."""
    jupyter_theme.install("light")
    server_config = jupyter_theme.target().parent.parent / "jupyter_server_config.py"
    server_config.write_text("# mine\n", encoding="utf-8")

    jupyter_theme.install("dark")
    assert server_config.read_text(encoding="utf-8") == "# mine\n"


def test_installing_keeps_the_other_overrides(workspace):
    """`overrides.json` carries the handoff's 88-column ruler. Rewriting the file
    to hold one key would drop it, and nothing would say so."""
    jupyter_theme.install("light")
    path = jupyter_theme.target().parent.parent / "overrides.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["@jupyterlab/fileeditor-extension:plugin"] = {"editorConfig": {"rulers": [88]}}
    path.write_text(json.dumps(document), encoding="utf-8")

    jupyter_theme.install("dark")
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["@jupyterlab/fileeditor-extension:plugin"]["editorConfig"]["rulers"] == [88]
    assert after["@jupyterlab/apputils-extension:themes"]["theme"] == "JupyterLab Dark"


def test_an_unwritable_workspace_still_lets_lab_start(workspace, monkeypatch):
    """Unstyled Lab beats no Lab. The caller reports the error rather than
    failing on it -- `tools/lab.py` puts it in the start record."""
    monkeypatch.setattr(jupyter_theme, "write", _boom)
    result = jupyter_theme.install("dark")
    assert result["error"] and "Boom" in result["error"]


class _Boom(Exception):
    pass


def _boom(*_args, **_kwargs):
    raise _Boom("Boom")


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
