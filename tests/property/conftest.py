"""Hypothesis settings for the property suite.

These tests answer a different question from the ones next door. `tests/` says
"this input produces that output", which is what you write once you know which
input to worry about. The four commits before this directory existed were all
the same shape -- a shell string nobody had thought to type, waved through by a
parser that looked right -- so the question here is "is there *any* input that
breaks the rule", asked by a generator that has no idea what a reviewer expects.

Three profiles, because the cost of an answer and the value of one are not the
same in every setting:

  * `dev` (default) -- 50 examples. Adds about a second to a local run.
  * `ci` -- 300 examples, and no `derandomize`, so CI explores seeds a developer
    never will and a rare counterexample surfaces on somebody else's machine
    rather than never.
  * `deep` -- 2000 examples with a long deadline, for running deliberately
    against a module that has just changed.

Select with `HYPOTHESIS_PROFILE=deep python -m pytest tests/property`.

`derandomize` is on for `dev` so that a local run is reproducible: a property
suite that fails one time in five and passes when you re-run it teaches people
to re-run it.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Verbosity, settings

# `shellgrammar.py` is a helper, not a test, and it lives next to the tests that
# use it because it is about them and nothing else. Neither `tests/` nor this
# directory is a package -- adding `__init__.py` here would change how pytest
# imports the fifty modules next door, which is a large change to make for one
# import -- so the directory goes on the path the same way `tests/conftest.py`
# puts the repository root there.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The repository's `tests/conftest.py` installs four autouse, function-scoped
# fixtures -- the temp workspace, the temp app directory, the process-state
# reset and the network refusal. Hypothesis sets those up once and then runs
# every example inside them, which is what `function_scoped_fixture` warns
# about, and it is right to warn: an example that writes into the workspace can
# be read by the next one.
#
# Suppressed rather than worked around, because the properties here are either
# pure functions of their arguments or make their own per-example directory
# (see `tests/property/test_prop_jsonl.py`). What the fixtures are actually
# providing is the *negative* guarantee -- no network, no writes into the
# developer's real `%LOCALAPPDATA%` -- and that one does not decay across
# examples.
_COMMON = {
    "suppress_health_check": [HealthCheck.function_scoped_fixture],
    # Wall-clock deadlines and a Windows filesystem do not mix: the first
    # example to touch a cold path pays for the whole directory tree and gets
    # blamed for it. Coverage is bounded by `max_examples` here, not by time.
    "deadline": None,
}

settings.register_profile("dev", max_examples=50, derandomize=True, **_COMMON)
settings.register_profile("ci", max_examples=300, **_COMMON)
settings.register_profile(
    "deep", max_examples=2000, verbosity=Verbosity.normal, **_COMMON
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def fresh_workspace(tmp_path):
    """A whole workspace per *example*, for the properties that write a ledger.

    The suppression above is safe for a pure function and not safe for anything
    that appends to a file. `tests/conftest.py` points `GRAD_ROOT` at one
    `tmp_path` per test *function*, so a hundred examples would share one
    ledger: run counts would accumulate across examples, `budget.create` would
    refuse the second example's project as already existing, and "spend equals
    the sum of what was submitted" would be false for every example after the
    first. All three of those happened.

    Returns a callable rather than a path, because the reset has to happen at
    the top of each example and a fixture body runs once.

    `os.environ` is written directly rather than through `monkeypatch`, which is
    also function-scoped; the outer fixture's own monkeypatch still owns the
    original value and restores it at teardown.
    """
    from core import config, paths

    counter = itertools.count()

    def reset():
        root = tmp_path / f"ws-{next(counter)}"
        os.environ["GRAD_ROOT"] = str(root)
        os.environ["GRAD_CONFIG"] = str(root / "config" / "grad.toml")
        config._cache.clear()
        paths.ensure_workspace()
        return root

    return reset
