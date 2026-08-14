"""The tiling layout (design handoff, "Interactions & behaviour").

This is the half of the window system with actual rules, and none of them need a
browser: where a new window lands, what happens to the fractions when one
closes, which layouts survive a version that renamed a window. Every test here
runs with NiceGUI uninstalled.
"""

from __future__ import annotations

import math

import pytest

from ui import layout as L
from ui import registry


def fractions(layout: L.Layout) -> list[float]:
    return [c.fraction for c in layout.columns]


def sums_to_one(values: list[float]) -> bool:
    return math.isclose(sum(values), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# opening and closing
# ---------------------------------------------------------------------------
def test_the_first_window_takes_the_whole_shell():
    layout = L.Layout().open("chat")
    assert layout.windows == ["chat"]
    assert layout.focused == "chat"
    assert sums_to_one(fractions(layout))


def test_a_second_window_splits_sideways():
    layout = L.Layout().open("chat").open("ledger")
    assert len(layout.columns) == 2
    assert layout.columns[0].windows == ["chat"]
    assert layout.columns[1].windows == ["ledger"]


def test_clicking_four_windows_reproduces_the_mock():
    """chat | notebook | ledger-over-quota, by clicking, in order.

    This is the arrangement the design shows, and it has to be reachable without
    dragging anything: the fourth click stacks because the columns are full.
    """
    layout = L.Layout()
    for window in ("chat", "notebook", "ledger", "quota"):
        layout.open(window)
    assert [c.windows for c in layout.columns] == [["chat"], ["notebook"], ["ledger", "quota"]]
    assert sums_to_one([s.fraction for s in layout.columns[2].slots])


def test_opening_stops_growing_sideways_at_the_cap():
    layout = L.Layout()
    for window in registry.ids():
        layout.open(window)
    assert len(layout.columns) <= L.MAX_COLUMNS
    assert len(layout.windows) == len(registry.ids())


def test_opening_an_open_window_focuses_it_rather_than_duplicating():
    layout = L.Layout().open("chat").open("ledger")
    layout.open("chat")
    assert layout.windows.count("chat") == 1
    assert layout.focused == "chat"


def test_closing_the_last_window_in_a_column_drops_the_column():
    layout = L.Layout().open("chat").open("ledger")
    layout.close("ledger")
    assert len(layout.columns) == 1
    assert sums_to_one(fractions(layout))


def test_closing_the_focused_window_moves_focus_to_a_live_one():
    layout = L.Layout().open("chat").open("ledger").open("quota")
    layout.focus("quota").close("quota")
    assert layout.focused in layout.windows
    assert layout.focused is not None


def test_closing_the_only_window_leaves_no_focus():
    layout = L.Layout().open("chat")
    layout.close("chat")
    assert layout.windows == []
    assert layout.focused is None


def test_toggle_is_open_then_close():
    layout = L.Layout()
    layout.toggle("funnel")
    assert layout.is_open("funnel")
    layout.toggle("funnel")
    assert not layout.is_open("funnel")


# ---------------------------------------------------------------------------
# fractions
# ---------------------------------------------------------------------------
def test_opening_windows_produces_an_even_split():
    """A new column created at 1.0 sits beside neighbours that already
    normalised down, so it ends up with half the shell. Opening four windows has
    to give the even thirds the design shows, not 25/25/50."""
    layout = L.Layout()
    for window in ("chat", "notebook", "ledger", "quota"):
        layout.open(window)
    for fraction in fractions(layout):
        assert math.isclose(fraction, 1 / 3, abs_tol=1e-6)
    assert all(
        math.isclose(s.fraction, 0.5, abs_tol=1e-6) for s in layout.columns[2].slots
    )


def test_the_split_does_not_depend_on_the_order_windows_were_opened():
    a = L.Layout()
    for window in ("chat", "ledger", "quota"):
        a.open(window)
    b = L.Layout()
    for window in ("quota", "chat", "ledger"):
        b.open(window)
    assert sorted(fractions(a)) == pytest.approx(sorted(fractions(b)))


def test_fractions_always_sum_to_one_at_both_levels():
    layout = L.Layout()
    for window in ("chat", "ledger", "quota", "funnel", "queue", "papers"):
        layout.open(window)
    assert sums_to_one(fractions(layout))
    for column in layout.columns:
        assert sums_to_one([s.fraction for s in column.slots])


def test_no_pane_is_ever_below_the_floor():
    """A 2% column is not a pane, it is a rounding error with a title bar."""
    layout = L.Layout().open("chat").open("ledger").open("wiki")
    layout.resize_columns([0.98, 0.01, 0.01])
    assert all(c.fraction >= L.MIN_FRACTION - 1e-9 for c in layout.columns)
    assert sums_to_one(fractions(layout))


def test_the_floor_is_expressed_in_pixels_when_a_width_is_known():
    """320px of a 900px shell is 35%, which is a much stronger floor than 6%."""
    layout = L.Layout().open("chat").open("ledger")
    layout.resize_columns([0.95, 0.05], total_px=900)
    assert min(fractions(layout)) >= L.MIN_PANE_PX / 900 - 1e-6


def test_a_resize_with_the_wrong_arity_is_ignored():
    """The browser and the server can disagree for one frame after a close."""
    layout = L.Layout().open("chat").open("ledger")
    before = fractions(layout)
    layout.resize_columns([0.5, 0.3, 0.2])
    assert fractions(layout) == before


def test_resize_slots_targets_one_column():
    layout = L.Layout()
    for window in ("chat", "notebook", "ledger", "quota"):
        layout.open(window)
    layout.resize_slots(2, [0.8, 0.2])
    assert layout.columns[2].slots[0].fraction > layout.columns[2].slots[1].fraction
    assert sums_to_one([s.fraction for s in layout.columns[2].slots])


def test_normalise_leaves_a_valid_layout_alone():
    """Idempotence, which is what stops a saved layout drifting toward uniform
    over a week of opening and closing windows."""
    layout = L.Layout().open("chat").open("ledger").open("quota")
    layout.resize_columns([0.5, 0.3, 0.2])
    before = fractions(layout)
    for _ in range(20):
        layout.normalise()
    for a, b in zip(before, fractions(layout)):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_a_resize_is_preserved_across_a_save_and_load_cycle():
    layout = L.Layout().open("chat").open("ledger").open("quota")
    layout.resize_columns([0.5, 0.3, 0.2])
    before = fractions(layout)
    for _ in range(10):
        layout = L.Layout.from_dict(layout.to_dict(), known=registry.ids())
    for a, b in zip(before, fractions(layout)):
        assert math.isclose(a, b, abs_tol=1e-4)


def test_resize_slots_on_a_column_that_does_not_exist_is_a_no_op():
    layout = L.Layout().open("chat")
    layout.resize_slots(7, [1.0])
    assert layout.windows == ["chat"]


def test_negative_and_zero_fractions_do_not_produce_nan():
    layout = L.Layout().open("chat").open("ledger")
    layout.resize_columns([0.0, -3.0])
    assert all(f == f for f in fractions(layout))  # not NaN
    assert sums_to_one(fractions(layout))


# ---------------------------------------------------------------------------
# retiling
# ---------------------------------------------------------------------------
def test_move_pulls_a_window_into_another_column():
    layout = L.Layout().open("chat").open("ledger").open("quota")
    layout.move("quota", 0)
    assert "quota" in layout.columns[0].windows


def test_move_past_the_last_column_creates_one():
    """A drag past the right edge means "make a new column"."""
    layout = L.Layout().open("chat").open("ledger")
    layout.move("ledger", 5)
    assert len(layout.columns) == 2
    assert layout.columns[-1].windows == ["ledger"]


def test_move_of_an_unopened_window_is_a_no_op():
    layout = L.Layout().open("chat")
    layout.move("evolve", 0)
    assert layout.windows == ["chat"]


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
def test_tile_gives_each_window_its_own_column_up_to_the_cap():
    layout = L.Layout()
    for window in ("chat", "ledger", "quota"):
        layout.open(window)
    layout.apply_preset("tile")
    assert [c.windows for c in layout.columns] == [["chat"], ["ledger"], ["quota"]]


def test_stack_collapses_everything_into_one_column():
    layout = L.Layout().open("chat").open("ledger").open("quota")
    layout.apply_preset("stack")
    assert len(layout.columns) == 1
    assert set(layout.columns[0].windows) == {"chat", "ledger", "quota"}


def test_full_keeps_every_window_open():
    """`full` is a view, not a close: `⌥1` has to restore without reopening."""
    layout = L.Layout().open("chat").open("ledger").open("quota")
    layout.focus("ledger").apply_preset("full")
    assert set(layout.windows) == {"chat", "ledger", "quota"}
    assert layout.columns[0].windows == ["ledger"]
    assert layout.columns[0].fraction > layout.columns[1].fraction


def test_presets_round_trip():
    layout = L.Layout()
    for window in ("chat", "ledger", "quota"):
        layout.open(window)
    before = set(layout.windows)
    for preset in L.PRESETS:
        layout.apply_preset(preset)
        assert set(layout.windows) == before


def test_an_unknown_preset_is_an_error_not_a_silent_no_op():
    layout = L.Layout().open("chat")
    with pytest.raises(ValueError):
        layout.apply_preset("cascade")


def test_a_preset_on_an_empty_layout_does_nothing():
    assert L.Layout().apply_preset("tile").windows == []


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def test_a_layout_round_trips_through_json():
    layout = L.Layout().open("chat").open("ledger")
    layout.resize_columns([0.6, 0.4])
    restored = L.Layout.from_dict(layout.to_dict(), known=registry.ids())
    assert [c.windows for c in restored.columns] == [c.windows for c in layout.columns]
    assert restored.focused == layout.focused
    for a, b in zip(restored.columns, layout.columns):
        assert math.isclose(a.fraction, b.fraction, abs_tol=1e-5)


def test_a_layout_naming_a_window_this_version_does_not_have_still_opens():
    """A layout file outlives the window set that wrote it. The failure mode for
    "we renamed the funnel" must be a missing pane, not an app that will not
    start."""
    stored = {
        "columns": [
            {"fraction": 0.5, "slots": [{"window": "chat", "fraction": 1.0}]},
            {"fraction": 0.5, "slots": [{"window": "holodeck", "fraction": 1.0}]},
        ],
        "focused": "holodeck",
    }
    restored = L.Layout.from_dict(stored, known=registry.ids())
    assert restored.windows == ["chat"]
    assert restored.focused == "chat"


def test_a_duplicated_window_in_a_stored_layout_is_deduplicated():
    stored = {
        "columns": [
            {"slots": [{"window": "chat"}]},
            {"slots": [{"window": "chat"}]},
        ]
    }
    assert L.Layout.from_dict(stored, known=registry.ids()).windows == ["chat"]


@pytest.mark.parametrize(
    "garbage",
    [None, [], "chat", 7, {"columns": "chat"}, {"columns": [{"slots": "chat"}]}, {"columns": [7]}],
)
def test_hand_edited_garbage_yields_an_empty_layout_not_a_traceback(garbage):
    assert L.Layout.from_dict(garbage, known=registry.ids()).windows == []


def test_fractions_that_are_not_numbers_fall_back_to_even():
    stored = {
        "columns": [
            {"fraction": "wide", "slots": [{"window": "chat", "fraction": None}]},
            {"fraction": float("inf"), "slots": [{"window": "ledger"}]},
        ]
    }
    restored = L.Layout.from_dict(stored, known=registry.ids())
    assert sums_to_one(fractions(restored))


def test_the_default_layout_is_the_registrys_defaults():
    layout = L.Layout.default(registry.defaults())
    assert set(layout.windows) == set(registry.defaults())
    assert sums_to_one(fractions(layout))
