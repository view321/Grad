"""The tiling layout: columns of stacked windows, and the moves over them.

Pure Python on purpose. This is the part of the window system with actual rules
-- where a new window lands, what happens to the fractions when one closes,
which layouts survive a version that renamed a window -- and none of those rules
need a browser to check. `tests/test_ui_layout.py` is the whole specification.

**Shape.** The handoff's mock is a horizontal flex of panes where "a pane may
split vertically (the right pane stacks LEDGER over QUOTA, each keeping its own
title bar)". That is two levels, not arbitrary nesting, and modelling it as two
levels rather than as a general BSP tree is the difference between fractions
that are obviously correct and fractions that need a diagram. A `Column` holds
`Slot`s; a `Layout` holds `Column`s; both levels carry fractions summing to 1.

**Fractions, not pixels.** The drag handles write pixel deltas in the browser,
but what persists is a fraction, so a layout saved on a 3440px monitor still
opens sanely on a laptop. `MIN_PANE_PX` is enforced at the point of resize,
against the width the browser reports, rather than baked into the stored value.

**Forward compatibility.** `from_dict` drops window ids it does not recognise
instead of raising. A layout file on disk outlives the window set that wrote it,
and the failure mode for "we renamed the funnel window" must be a missing pane,
not an app that will not start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

MIN_PANE_PX = 320
# Below this a fraction is noise: two columns cannot both be meaningful if one
# is 2% of the shell. Enforced on every normalise, so it also bounds how many
# columns `open` will create.
MIN_FRACTION = 0.06
# Three, matching the mock -- and not by coincidence. Clicking four windows in a
# row has to be able to produce the arrangement the design shows (chat |
# notebook | ledger-over-quota), which it only does if the fourth click stacks
# rather than opening a fourth column. Three columns is also 960px of minimum
# width, which still fits a laptop; five would not.
MAX_COLUMNS = 3

PRESETS = ("tile", "stack", "full")


@dataclass
class Slot:
    """One window inside a column, with its share of the column's height."""

    window: str
    fraction: float = 1.0


@dataclass
class Column:
    """A vertical stack of windows, with its share of the shell's width."""

    slots: list[Slot] = field(default_factory=list)
    fraction: float = 1.0

    @property
    def windows(self) -> list[str]:
        return [s.window for s in self.slots]


def _normalise(values: list[float], *, minimum: float = MIN_FRACTION) -> list[float]:
    """Fractions that sum to 1, with nothing below `minimum`.

    **Idempotent, and that is the whole difficulty.** `normalise` runs on every
    open, close, move and load, so a version that nudged already-valid fractions
    would walk a saved layout toward uniform a little on each round trip -- the
    panes would visibly drift back to even over a week of use, for no reason the
    user could see. The naive "give everyone the floor, then share the slack in
    proportion" does exactly that: it re-applies the floor to values that already
    clear it.

    So the floor is only paid for by the panes that are *above* it, and only when
    someone is actually below it. When every value already clears the floor the
    function is a pure rescale, and rescaling something that already sums to 1
    returns it unchanged.

    No rounding here either -- `to_dict` rounds once, on the way to disk. Rounding
    on every call is the same drift by a different route.
    """
    count = len(values)
    if count == 0:
        return []
    floor = min(minimum, 1.0 / count)
    positive = [max(0.0, v) for v in values]
    total = sum(positive)
    if total <= 0:
        return [1.0 / count] * count

    scaled = [v / total for v in positive]
    # Bounded loop rather than `while`: lifting one pane can push another under
    # the floor, but each pass strictly reduces the number below it, so `count`
    # passes always suffice and nothing can spin.
    for _ in range(count):
        deficit = sum(floor - v for v in scaled if v < floor)
        if deficit <= 1e-12:
            break
        surplus = sum(v - floor for v in scaled if v > floor)
        if surplus <= deficit:
            # Not enough room to give everyone their minimum: the only honest
            # answer is an even split.
            return [1.0 / count] * count
        shrink = (surplus - deficit) / surplus
        scaled = [floor if v <= floor else floor + (v - floor) * shrink for v in scaled]
    return scaled


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 1.0


@dataclass
class Layout:
    columns: list[Column] = field(default_factory=list)
    focused: str | None = None

    # -- inspection ---------------------------------------------------------
    @property
    def windows(self) -> list[str]:
        return [s.window for c in self.columns for s in c.slots]

    def is_open(self, window: str) -> bool:
        return window in self.windows

    def locate(self, window: str) -> tuple[int, int] | None:
        for ci, column in enumerate(self.columns):
            for si, slot in enumerate(column.slots):
                if slot.window == window:
                    return ci, si
        return None

    def __iter__(self) -> Iterator[tuple[int, int, Slot]]:
        for ci, column in enumerate(self.columns):
            for si, slot in enumerate(column.slots):
                yield ci, si, slot

    # -- mutation -----------------------------------------------------------
    def normalise(self) -> Layout:
        """Drop empty columns, re-spread both levels, keep focus valid."""
        self.columns = [c for c in self.columns if c.slots]
        for column in self.columns:
            spread = _normalise([s.fraction for s in column.slots])
            for slot, value in zip(column.slots, spread):
                slot.fraction = value
        spread = _normalise([c.fraction for c in self.columns])
        for column, value in zip(self.columns, spread):
            column.fraction = value
        if self.focused not in self.windows:
            self.focused = self.windows[0] if self.windows else None
        return self

    def focus(self, window: str) -> Layout:
        if self.is_open(window):
            self.focused = window
        return self

    def open(self, window: str) -> Layout:
        """Into the focused column, splitting it; a new column if there is room.

        The handoff says "clicking a name opens it into the focused pane (or
        splits if the pane already holds one)". Taken literally that never grows
        the number of columns, which makes the three-column layouts in the mock
        unreachable by clicking. So: an empty layout starts a column, a
        single-column layout that is already occupied grows sideways up to
        `MAX_COLUMNS`, and past that it stacks into the focused column. Both
        directions of the mock are then reachable, and the cap is what keeps a
        click from producing a 40px-wide pane.
        """
        if self.is_open(window):
            return self.focus(window)
        if not self.columns:
            self.columns = [Column([Slot(window)])]
            self.focused = window
            return self.normalise()

        target = self._focused_column_index()
        if len(self.columns) < MAX_COLUMNS and len(self.columns[target].slots) == 1:
            # The *mean* of the existing fractions, not 1.0. A new pane created
            # at 1.0 sits alongside neighbours that already normalised down, so
            # normalising again hands it half the shell -- which is why opening
            # four windows used to give 25/25/50 rather than the even thirds the
            # design shows. The mean makes `open` produce an even split, and
            # makes it order-independent.
            self.columns.insert(target + 1, Column([Slot(window)], self._mean_column_fraction()))
        else:
            column = self.columns[target]
            column.slots.append(Slot(window, _mean([s.fraction for s in column.slots])))
        self.focused = window
        return self.normalise()

    def _mean_column_fraction(self) -> float:
        return _mean([c.fraction for c in self.columns])

    def close(self, window: str) -> Layout:
        found = self.locate(window)
        if not found:
            return self
        ci, si = found
        del self.columns[ci].slots[si]
        if self.focused == window:
            # Focus the neighbour that is still on screen, preferring the one
            # that took over the vacated space.
            column = self.columns[ci] if ci < len(self.columns) else None
            if column and column.slots:
                self.focused = column.slots[min(si, len(column.slots) - 1)].window
            else:
                self.focused = None
        return self.normalise()

    def toggle(self, window: str) -> Layout:
        return self.close(window) if self.is_open(window) else self.open(window)

    def move(
        self,
        window: str,
        column_index: int,
        slot_index: int | None = None,
        *,
        new_column: bool = False,
    ) -> Layout:
        """Retile: pull a window out and drop it at a named position.

        `slot_index` is where in the target column it lands -- `None` appends,
        which is what a drop with no vertical opinion means. `new_column` splits
        a fresh column in at `column_index` rather than adding to the one already
        there; `column_index == len(columns)` means the same thing at the right
        edge, which is where a drag past the last pane ends up.

        Two corrections that are invisible until they are wrong:

        * **The cap is counted after the pull, not before.** Dragging the only
          window out of a column empties it, and an empty column is dropped by
          `normalise` -- so that drag can create a column without ever exceeding
          `MAX_COLUMNS`. Counting the columns that still hold something is what
          lets the gesture through while still refusing a genuine fourth.
        * **Moving down inside one column shifts its own target.** The browser
          computes `slot_index` against a column that still contains the dragged
          window; by the time we insert, the pull has shifted everything after it
          left by one. Without the adjustment, dragging a pane one place down
          moves it two.
        """
        found = self.locate(window)
        if not found:
            return self
        ci, si = found
        slot = self.columns[ci].slots.pop(si)
        slot.fraction = 1.0

        column_index = max(0, min(column_index, len(self.columns)))
        wants_column = new_column or column_index == len(self.columns)
        # Columns that still hold a window -- see the docstring.
        live = sum(1 for c in self.columns if c.slots)

        if wants_column and live < MAX_COLUMNS:
            self.columns.insert(column_index, Column([slot]))
        else:
            # At the cap (or asked for a column we cannot make), the drop lands
            # in the nearest real column rather than being refused: a gesture
            # that visibly picked a pane up has to put it down somewhere.
            if not self.columns:
                self.columns = [Column([slot])]
            else:
                column_index = min(column_index, len(self.columns) - 1)
                target = self.columns[column_index].slots
                # Clamped against the column as the *browser* saw it -- one
                # longer when the window came from this same column, because it
                # was still in it when the boundary was picked. Clamping to the
                # shortened list first and then correcting for the pull applies
                # the same subtraction twice, and a drop past the last pane
                # lands second-to-last.
                limit = len(target) + (1 if ci == column_index else 0)
                index = limit if slot_index is None else max(0, min(slot_index, limit))
                if ci == column_index and si < index:
                    index -= 1
                target.insert(index, slot)
        self.focused = window
        return self.normalise()

    def swap(self, a: str, b: str) -> Layout:
        """Exchange two windows, leaving the panes where they are.

        The slots keep their fractions and the windows trade places, rather than
        each window carrying its size across with it. Dropping the ledger onto
        the chat should put the ledger where the chat was, at the chat's size --
        not reflow the whole shell around a pane that just arrived.
        """
        if a == b:
            return self
        first, second = self.locate(a), self.locate(b)
        if not first or not second:
            return self
        (ac, as_), (bc, bs) = first, second
        self.columns[ac].slots[as_].window = b
        self.columns[bc].slots[bs].window = a
        self.focused = a
        return self.normalise()

    def resize_columns(self, fractions: Iterable[float], *, total_px: int | None = None) -> Layout:
        values = list(fractions)
        if len(values) != len(self.columns):
            return self
        for column, value in zip(self.columns, _normalise(values, minimum=self._floor(total_px))):
            column.fraction = value
        return self.normalise()

    def resize_slots(
        self, column_index: int, fractions: Iterable[float], *, total_px: int | None = None
    ) -> Layout:
        if not 0 <= column_index < len(self.columns):
            return self
        column = self.columns[column_index]
        values = list(fractions)
        if len(values) != len(column.slots):
            return self
        for slot, value in zip(column.slots, _normalise(values, minimum=self._floor(total_px))):
            slot.fraction = value
        return self.normalise()

    def _floor(self, total_px: int | None) -> float:
        """`MIN_PANE_PX` expressed as a fraction of the space we were given."""
        if not total_px or total_px <= 0:
            return MIN_FRACTION
        return max(MIN_FRACTION, min(0.5, MIN_PANE_PX / total_px))

    def _focused_column_index(self) -> int:
        found = self.locate(self.focused) if self.focused else None
        return found[0] if found else len(self.columns) - 1

    # -- presets ------------------------------------------------------------
    def apply_preset(self, preset: str) -> Layout:
        """`⌥1` tile, `⌥2` stack, `⌥3` full. Membership never changes."""
        windows = self.windows
        if not windows:
            return self
        if preset == "tile":
            # One column each, up to the cap; the overflow stacks in the last.
            self.columns = [Column([Slot(w)]) for w in windows[: MAX_COLUMNS - 1]]
            rest = windows[MAX_COLUMNS - 1 :]
            if rest:
                self.columns.append(Column([Slot(w) for w in rest]))
        elif preset == "stack":
            self.columns = [Column([Slot(w) for w in windows])]
        elif preset == "full":
            # Everything stays open -- `full` is a view, not a close -- but the
            # focused window gets a column of its own and the rest stack behind
            # it at the floor, so `⌥1` restores without reopening anything.
            focused = self.focused or windows[0]
            others = [w for w in windows if w != focused]
            self.columns = [Column([Slot(focused)], fraction=0.9)]
            if others:
                self.columns.append(Column([Slot(w) for w in others], fraction=0.1))
            self.focused = focused
        else:
            raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
        return self.normalise()

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "focused": self.focused,
            "columns": [
                {
                    "fraction": round(c.fraction, 6),
                    "slots": [{"window": s.window, "fraction": round(s.fraction, 6)} for s in c.slots],
                }
                for c in self.columns
            ],
        }

    @classmethod
    def from_dict(cls, data: Any, *, known: Iterable[str] | None = None) -> Layout:
        """Rebuild from disk, dropping anything that no longer exists.

        Every field is treated as untrusted: this file is hand-editable and
        outlives the code that wrote it.
        """
        allowed = set(known) if known is not None else None
        layout = cls()
        if not isinstance(data, dict):
            return layout
        seen: set[str] = set()
        for raw_column in data.get("columns") or []:
            if not isinstance(raw_column, dict):
                continue
            slots: list[Slot] = []
            for raw_slot in raw_column.get("slots") or []:
                if not isinstance(raw_slot, dict):
                    continue
                window = raw_slot.get("window")
                if not isinstance(window, str) or window in seen:
                    continue
                if allowed is not None and window not in allowed:
                    continue
                seen.add(window)
                slots.append(Slot(window, _as_fraction(raw_slot.get("fraction"))))
            if slots:
                layout.columns.append(Column(slots, _as_fraction(raw_column.get("fraction"))))
        focused = data.get("focused")
        layout.focused = focused if isinstance(focused, str) and focused in seen else None
        return layout.normalise()

    @classmethod
    def default(cls, windows: Iterable[str]) -> Layout:
        """The mock's opening arrangement: chat, notebook, and a split rail."""
        layout = cls()
        for window in windows:
            layout.open(window)
        return layout.normalise()


def _as_fraction(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if number != number or number in (float("inf"), float("-inf")) or number <= 0:
        return 1.0
    return number
