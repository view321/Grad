"""Numbered, idempotent migrations of on-disk state.

`core/appdata.py:migrate_legacy` was the first of these and is the template for
all of them: it runs at startup, it is safe to run twice, and it cannot take the
app down. What it lacked was a *record* -- it re-derived "is there anything to
move?" from the filesystem on every launch, which works exactly once, for
exactly one migration, and only because "the destination exists" happens to be a
reliable proxy for "this already ran".

The second migration cannot borrow that trick, so the record becomes explicit:
`schema.json` in the installation's state directory holds the highest step that
has completed. `core/update.py` runs the pending ones after a fast-forward, and
startup runs them too, because an update applied from a terminal and an app
launched afterwards must not disagree about which shape the state is in.

**Three properties, and each of them is load-bearing.**

*Idempotent*, because startup and `grad update` both run this and neither knows
what the other did. *Non-fatal*, because a workspace on a read-only mount or a
directory held open by a running Lab server must not stop the app opening -- the
same reasoning `migrate_legacy` gives in its own docstring. *Ordered, and
stopping at the first failure*, because step N+1 may assume step N has run:
skipping ahead past a failure and recording success would leave state that no
later migration will ever revisit, which is worse than trying again tomorrow.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core import appdata

log = logging.getLogger("grad.migrate")

#: A migration returns what it changed, for the caller to log. Empty means
#: "nothing to do", which is the normal answer on every launch after the first.
Step = Callable[[], list[str]]


def _schema_path() -> Path:
    return appdata.state_dir() / "schema.json"


def _m1_app_state_out_of_workspace() -> list[str]:
    """Move layouts, transcripts, the Lab token and the cache out of `data/`.

    The migration `core/appdata.py` already implements. Registered here rather
    than reimplemented: it is called directly by two entry points that predate
    this module, and having two copies of a file-moving routine is exactly the
    situation where one of them quietly stops matching where the readers look.
    """
    return [f"data/{name}" for name in appdata.migrate_legacy()]


def _m2_record_installed_extras() -> list[str]:
    """Write `install.json` for a checkout installed before the updater existed.

    `core/update.py` reinstalls with the extras the user originally chose, and
    from this release the installers record them. An installation that predates
    that has no such file, and reinstalling with the wrong extras is not a
    cosmetic error -- dropping `ui` would take the desktop app off a machine
    whose owner asked only for an update.

    So the gap is filled by *observation* rather than by a guess: ask the
    environment which of the optional dependency sets actually import. Anything
    this cannot see stays out, and `grad update --extras` remains the override.
    """
    from core import update  # noqa: PLC0415 - circular at module scope

    path = update.install_record_path()
    if path.exists():
        return []
    extras = update.detect_extras()
    update.write_install_record(extras=extras, detected=True)
    if not path.exists():
        # `write_install_record` swallows OSError, which is right for a caller
        # that only wants the record kept up to date. Here it is the entire
        # migration, and recording a step that did not happen is the one thing
        # this module must not do: the number goes up, nothing ever revisits it,
        # and `grad update` reinstalls with guessed extras forever. Raising
        # leaves it pending -- see `run_pending`.
        raise OSError(f"could not write {path}")
    return [f"install.json (extras: {','.join(extras) or 'none detected'})"]


#: (number, name, step). Append only, and never renumber: the number is what a
#: user's `schema.json` already holds.
MIGRATIONS: tuple[tuple[int, str, Step], ...] = (
    (1, "app-state-out-of-workspace", _m1_app_state_out_of_workspace),
    (2, "record-installed-extras", _m2_record_installed_extras),
)

LATEST = max(number for number, _, _ in MIGRATIONS)


def current() -> int:
    """The highest completed step. Zero for an installation that has never run
    this, which is every installation that predates it -- and correct, because
    step 1 is idempotent and re-running it costs a stat per entry."""
    try:
        data = json.loads(_schema_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    value = data.get("version") if isinstance(data, dict) else None
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _record(number: int) -> None:
    appdata.ensure()
    try:
        _schema_path().write_text(
            json.dumps({"version": number}, indent=2), encoding="utf-8"
        )
    except OSError:
        # The migration ran; only the bookkeeping failed. Running it again next
        # launch is harmless by construction, so this is a debug line and not an
        # error the user has to act on.
        log.debug("could not record schema version %d", number)


def pending() -> list[tuple[int, str, Step]]:
    done = current()
    return [entry for entry in MIGRATIONS if entry[0] > done]


def run_pending() -> list[str]:
    """Apply the migrations this installation has not run. Never raises.

    Returns the human-readable list of what changed, for the caller to log. An
    empty list is the overwhelmingly common answer and means the state is
    already current.
    """
    changed: list[str] = []
    for number, name, step in pending():
        try:
            changed += step()
        except Exception:  # noqa: BLE001 - see the module docstring
            log.exception("migration %d (%s) failed; leaving it pending", number, name)
            break
        _record(number)
    return changed
