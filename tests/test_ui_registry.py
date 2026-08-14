"""The window registry, and the contract every window module signs.

The registry is the one list the opener strip, the layout presets, the command
palette, the persisted layout's validation and the status bar's count are all
derived from. If it is ever more than one list, two of those will drift.

These tests import the window modules, which imports `ui/kit.py` -- but not
NiceGUI, because `kit` imports it inside its functions. That is deliberate: it
means a window with a typo in its module scope fails here rather than at page
build time on someone's laptop.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from ui import layout as layout_mod, registry, state as state_mod


def test_ids_are_unique():
    assert len(registry.ids()) == len(set(registry.ids()))


def test_the_eleven_windows_the_handoff_lists_are_all_here():
    assert set(registry.ids()) == {
        "chat", "notebook", "wiki", "papers", "evolve", "editor",
        "ledger", "preflight", "quota", "funnel", "queue",
    }


@pytest.mark.parametrize("window", registry.WINDOWS, ids=lambda w: w.id)
def test_every_window_module_imports_and_defines_render(window):
    module = importlib.import_module(window.module)
    assert callable(getattr(module, "render", None)), f"{window.module}.render"
    signature = inspect.signature(module.render)
    assert len(signature.parameters) == 1, "render takes the workspace and nothing else"


@pytest.mark.parametrize("window", registry.WINDOWS, ids=lambda w: w.id)
def test_optional_title_bar_hooks_have_the_right_shape(window):
    module = importlib.import_module(window.module)
    for name in ("subtitle", "chips"):
        fn = getattr(module, name, None)
        if fn is None:
            continue
        assert len(inspect.signature(fn).parameters) == 1


@pytest.mark.parametrize("window", registry.WINDOWS, ids=lambda w: w.id)
def test_no_window_reads_a_ledger_directly(window):
    """A window renders a model; it does not read `runs.jsonl`. Keeping that
    true is what lets `tests/test_ui_models.py` be the whole specification for
    what the windows say."""
    module = importlib.import_module(window.module)
    source = inspect.getsource(module)
    for forbidden in ("ledger_store", "quota_log", "jsonl.read", "core.corpus"):
        assert forbidden not in source, f"{window.module} reaches past ui/models.py for {forbidden}"


def test_every_window_but_chat_has_a_model_builder():
    """`chat` is the exception on purpose: its state is the live SDK session,
    not a file, so the poll must not redraw it and take the transcript's scroll
    position with it."""
    assert set(state_mod.MODEL_BUILDERS) == set(registry.ids()) - {"chat"}


def test_the_defaults_reproduce_the_mocks_opening_arrangement():
    """chat | notebook | ledger-over-quota."""
    assert registry.defaults() == ("chat", "notebook", "ledger", "quota")
    layout = layout_mod.Layout.default(registry.defaults())
    assert [c.windows for c in layout.columns] == [["chat"], ["notebook"], ["ledger", "quota"]]


def test_the_default_layout_fits_the_minimum_pane_width():
    columns = len(layout_mod.Layout.default(registry.defaults()).columns)
    assert columns * layout_mod.MIN_PANE_PX <= 1600, "the default window would open below minimum"


def test_an_unknown_window_id_is_an_error_with_the_known_ones_in_it():
    with pytest.raises(KeyError) as excinfo:
        registry.spec("holodeck")
    assert "chat" in str(excinfo.value)


def test_a_subtitle_that_raises_falls_back_to_the_hint(monkeypatch):
    """A title bar must never be the thing that takes a window down."""
    import ui.windows.ledger as ledger_window

    monkeypatch.setattr(ledger_window, "subtitle", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert registry.subtitle("ledger", object()) == registry.spec("ledger").hint


def test_chips_that_raise_degrade_to_none(monkeypatch):
    import ui.windows.queue as queue_window

    monkeypatch.setattr(queue_window, "chips", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert registry.chips("queue", object()) == []


def test_persistent_windows_are_the_ones_that_own_a_document():
    """A persistent window is one whose root must survive a retile. Chat owns a
    transcript; notebook owns the Lab iframe's anchor."""
    assert {w.id for w in registry.WINDOWS if w.persistent} == {"chat", "notebook"}
