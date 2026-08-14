"""The NiceGUI desktop interface (HANDOFF §10).

The UI stays thin on purpose: it transports events, renders state, and calls the
CLIs from §8. It holds no logic of its own. Anything the UI can do, the CLIs can
already do, which keeps the terminal path alive and keeps the portability claim
honest.

Since the window-system redesign the package has a shape worth stating, because
the layering is what keeps that promise checkable:

    tokens.py        design tokens; the stylesheet is generated from them
    fonts.py         @font-face for whatever is vendored, Google Fonts for the rest
    layout.py        the pane tree and the moves over it        -- pure, tested
    models.py        what each window shows, as plain data      -- pure, tested
    registry.py      the one list of windows the shell derives from
    state.py         one poll, one snapshot, per-client workspace state
    kit.py           the primitives; Quasar is bypassed, not overridden
    shell.py         the chrome, and how a window survives a retile
    windows/         eleven renderers, none of which read a ledger directly
    jupyter_theme.py the same tokens, emitted as JupyterLab's custom.css
    katex.py         math in the transcript

`layout.py` and `models.py` import nothing from NiceGUI, at module scope or
inside a function. That is the rule that makes the interesting half of this
package testable with the `ui` extra uninstalled, and `tests/test_ui_*.py`
enforces it.
"""
