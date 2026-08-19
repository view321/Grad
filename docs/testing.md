# Testing

Three suites, answering three different questions. They are not tiers of the
same thing and none of them replaces another.

| Suite | Question | Cost | Runs in CI |
| --- | --- | --- | --- |
| `tests/*.py` | does this input produce that output? | ~3 min | yes, every push |
| `tests/property/` | is there *any* input that breaks the rule? | ~5 s | yes, every push |
| `mutmut` | would any test notice if this line were wrong? | CPU-hours | no, by hand |

```bash
python -m pytest -q
```

That is still the command. The generated suite is inside `tests/`, so it runs
with everything else and needs no separate invocation.

## Why the second suite exists

The four commits before `tests/property/` was written were four bugs in one
function — `hooks._segments`, the quote-aware splitter that decides whether a
shell command reaches the deny list. Each was found by a person typing one more
string, and each got an example-based test recording the string that found it.

That is a good record and a bad search. `tests/test_hooks.py` now pins nineteen
command lines; the shell accepts infinitely many, and the fifth bug was not in
the nineteen. It was `( ssh gpu-box nvidia-smi )` — three tokens, no quoting, no
substitution, and the shortest bypass the deny list ever had.

So `tests/property/shellgrammar.py` builds command lines from a grammar instead,
and carries the answer alongside the text: every node knows which heads the
shell would execute in it. The property is then one line — if the shell runs
`ssh`, the hook says so — and Hypothesis searches for a counterexample rather
than waiting for one to be reported. It found three in the first run:

- `( cmd )` and `{ cmd; }` were never read as starting a command;
- `if`, `then`, `do`, `else`, `!` and `time` were read *as* the command they
  introduce, so `for h in a b; do ssh $h; done` had a head of `do`;
- `rm --recursive -f` matched none of the six alternations in the `rm -rf` rule,
  which covered short-with-short and long-with-long and no mixed pair.

The other five modules under `tests/property/` are chosen on the same basis:
pure functions of their arguments, deciding something irreversible, where the
answer is constrained by an identity rather than by an example. A mean lies
between the extremes it was taken over; a rolling spend never falls when a run
is submitted; a document hashes the same after a round trip through the archive.

### Profiles

```bash
HYPOTHESIS_PROFILE=deep python -m pytest tests/property -q
```

- `dev` (default) — 50 examples, `derandomize=True`. About a second. A property
  suite that fails one time in five and passes when you re-run it teaches people
  to re-run it.
- `ci` — 300 examples, random seeds, so CI explores what a developer never will.
- `deep` — 2000 examples. Worth running deliberately against a module that has
  just changed.

### Writing one

Two rules, both learned the hard way in this directory:

**Fixtures come first in the signature.** `@given` binds positional strategies
to the *trailing* parameters, so `def test(rows, tmp_path)` hands the strategy to
`tmp_path` and asks pytest for a fixture called `rows`.

**A function-scoped fixture is set up once and shared by every example.** The
health check that says so is suppressed in `tests/property/conftest.py`, because
most properties here are pure — but anything that writes needs its own isolation
per example, or example 2 reads example 1's ledger. `test_prop_jsonl.py` uses a
module-level counter for a fresh file; `test_prop_ceilings.py` uses the
`fresh_workspace` fixture, which re-points `GRAD_ROOT` at a new directory. Both
of those exist because the first version of each test silently passed on
leftovers — a round trip that appended six records and read back 242.

## Mutation testing

Coverage says a line ran. It does not say anything would have failed if the line
were different, and those turn out to be very different questions: a line
executed by twenty tests that all assert on something else is covered and
unprotected.

`mutmut` changes one line at a time and runs the tests. A mutant that survives is
a change to the source that no test objected to.

```bash
mutmut run
mutmut results
mutmut show <mutant>
```

Configured in `pyproject.toml` under `[tool.mutmut]`, deliberately narrow:
`source_paths` is eight modules, not the project. Mutation testing costs about
one test run per mutant, so the useful version of it is aimed rather than
sprayed. These eight are the pure ones, deciding the irreversible things — what
gets denied, what a run measured, what is written to the ledger — and they are
the ones `tests/property/` already covers, which is what makes a surviving
mutant a finding rather than a to-do. Widen it one module at a time, when that
module is what changed.

It is **not** in CI. Hours per run is a thing to spend on a module you are
changing, not on every push.

### On Windows

mutmut refuses to run natively on Windows ([mutmut#397]) — it forks, and Windows
has no fork. Use WSL:

```bash
wsl -d <distro>
python3 -m venv ~/gradmut/.venv
~/gradmut/.venv/bin/pip install -e ".[dev]"
cd /path/to/checkout && ~/gradmut/.venv/bin/python -m mutmut run
```

Working from a copy inside the WSL filesystem rather than over `/mnt/d` is worth
the `tar` — mutmut copies the whole tree into `mutants/` and then runs pytest in
it several hundred times, and 9p is not the filesystem for that.

[mutmut#397]: https://github.com/boxed/mutmut/issues/397

## The three tests that fail for environmental reasons

Unchanged, and documented in [`CONTRIBUTING.md`](../CONTRIBUTING.md): two lock
tests in `tests/test_desktop_app.py` fail if a real Grad is running, and
`tests/test_wakeup.py::test_the_deadline_is_reported_as_a_deadline` depends on
the time of day. CI deselects the third and holds nothing.
