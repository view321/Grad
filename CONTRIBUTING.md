# Contributing

Grad is in development and its most valuable contribution right now is not a
pull request. It is a report from somebody who is not its author.

## The thing that helps most

Run it and say where you stopped:
[**I ran it — here is how far I got**](https://github.com/view321/Grad/issues/new?template=ran_it.yml).

Nothing has to have broken. "I read the README and did not install it" is a
finding about the README. "I got authenticated and then did not know what to
type" is a finding about the first screen. Both are more useful than silence,
and silence is what this project has had so far.

## Filing a bug

Use the [bug report form](https://github.com/view321/Grad/issues/new?template=bug_report.yml).
It asks for three things, and all three are load-bearing:

```bash
grad --version
grad --check
```

`--check` reports whether a token is present as a true/false and names any
environment variables it removed **without their values** — it carries no
credentials. It does include your workspace path, which contains your username;
redact that line if you would rather not post it.

Logs are at `%LOCALAPPDATA%\Grad\logs\grad.log` on Windows and
`~/.local/state/grad/logs/grad.log` elsewhere.

Before filing, check [Status](README.md#status). The Hugging Face Jobs and SSH
submitters, the evolutionary campaign loop, and the report generator's
model-driven half are implemented and tested but have never run against live
services. Breakage there is expected — still worth reporting, because "a real
credential and a real run are what find the mismatches" is the whole reason to
want users.

## Sending code

```bash
git clone https://github.com/view321/Grad && cd Grad
pip install -e ".[dev,agent,notebook,retrieval,remote,ui,lab,math]"
python -m pytest -q
```

Two failures on an unmodified tree are conditions rather than regressions, and
each has its own condition:

- the two lock tests in `tests/test_desktop_app.py` fail **if a real Grad
  instance is running**, because it is already holding the single-instance lock
  they are about. Quit the app and they pass.
- `tests/test_wakeup.py::test_the_deadline_is_reported_as_a_deadline` compares an
  ISO timestamp against a wake id suffix, so it **depends on the time of day**.
  It is a real defect and is deselected in CI rather than deleted, so that it
  stays visible locally where a human can tell it from a regression.

No fixed pass/fail count is given on purpose. The number moves with every test
added, and it is not portable anyway: the count you get depends on your platform
and Python version, which is exactly what CI exists to cover and a local run does
not. Compare a suspicious failure against a clean tree with `git stash`, not
against a number written down here.

CI runs with nothing holding the lock and deselects the time-of-day test, so it
expects a clean pass on every leg.

What the CI checks, and why it is shaped the way it is:

- the suite on Windows and Linux, on Python 3.11 and 3.13
- **a real wheel, installed into a venv outside the checkout.** Two packaging
  bugs in this project's history were invisible to an editable install, because
  an editable install puts the source tree on `sys.path` and `import agent`
  resolves whether or not the wheel contains it. If you change
  `[tool.setuptools]`, that job is the one to watch.

There is a second suite inside the first. `tests/property/` generates its inputs
with Hypothesis instead of listing them, and it is where a rule goes when the
examples keep running out — `hooks._segments` had four bugs in four commits and a
fifth that no example had reached. It runs with everything else and takes about a
second; `HYPOTHESIS_PROFILE=deep` turns it up when you have just changed one of
the modules it covers. Mutation testing is configured too, and is not in CI.
Both are in [`docs/testing.md`](docs/testing.md), including the two ways a
generated test can silently pass on the previous example's leftovers.

Some conventions worth knowing before a larger change:

- **Capability is a CLI, not a framework.** New agent-facing capability is a
  program with `--json` on every subcommand and a stable envelope, not a Python
  API the loop imports.
- **A gate is a program that refuses, not a sentence in a prompt.** Anything that
  must be true before money is spent belongs in code with its own exit code.
- **Errors carry the next command.** A refusal names the thing the caller
  skipped, literally enough to paste.

The reasoning behind all three, and what each one cost to learn, is in
[`CLAUDE_README.md`](CLAUDE_README.md).

## Licence

MIT. By contributing you agree your contribution is licensed under it.
