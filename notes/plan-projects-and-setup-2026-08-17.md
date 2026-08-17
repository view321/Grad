# Plan — the projects window and interactive setup, 2026-08-17

> **Implemented, 2026-08-17.** All five stages are in on `dev`, with 1550 tests
> green and the whole thing driven in a live browser preview against a scratch
> workspace. Three things came out differently from the plan and are recorded at
> the bottom under *What changed in the building*.

Scope: a first-class projects surface, and a guided setup that asks for the
Claude token, the six model roles, the backends and the ceilings — with setup
opening automatically when a project is created.

The feature as originally described is one wizard covering four subjects. The
central claim of this plan is that **it is two wizards**, because three of the
four subjects are machine- or workspace-scoped and only one is per-project. A
single wizard bolted to project creation re-asks for the Claude token every time
a project is made, and a user with six projects answers the same question six
times.

---

## 1. Five decisions everything else follows from

**1.1 Projects is a window, not a dialog.** `ui/registry.py`'s own docstring
says a thirteenth window is one `WindowSpec` and one module. The current project
surface is a 540px modal that closes on every action (`ui/shell.py:300`) and
cannot be tiled beside the ledger it explains. Projects keys every run, every
ceiling and every report; it is the only first-class concept in the app with no
window.

**1.2 Setup splits by scope, not by page count.**

| subject | scope | asked |
| --- | --- | --- |
| Claude OAuth token | machine | once, or when missing |
| six model roles | workspace | once, editable any time |
| backends + credentials | machine (credential) + workspace (inventory) | once per backend |
| ceilings, payer, default backend | **project** | every new project |

First launch runs the machine half. Project creation runs the project half and
*skips* anything already satisfied, with a link to review the rest.

**1.3 `config/grad.toml` is never machine-written.** README:216 states the
reason — it is hand-annotated, `tomllib` cannot write it, and an editing command
would reformat it and drop every comment in it. The precedent for a machine-set
value already exists: `tools/kaggle.py:179` writes
`appdata.state_dir()/kaggle.json`, that selection *wins* over `[kaggle] username`,
and `account` reports when it is shadowing a config value. Setup follows that
exactly.

**1.4 Resolution is layered, and the layers are already the shape `model_for`
uses.** `core/config.py:431` resolves explicit → legacy → default, and
`paths.config_path()` is `_shipped("config", "grad.toml")` — the *workspace's*
copy when it has one, the installation's otherwise. So two config layers already
exist. Setup adds two more above them:

```
project override → workspace overlay → workspace grad.toml → installed grad.toml → legacy → DEFAULTS
```

Every layer must be reportable. "Why is it using Sonnet for evolve?" has to have
an answer, the way `[kaggle] username` shadowing does.

**1.5 Budget stays an append-only event.** `core/budget.py:234` never mutates a
ceiling; it records `previous` and `reason`. Setup produces a create-time
`budget{}` through `budget.create`, and every later change is a raise with a
reason. A wizard that could be re-run and silently rewrite ceilings would
destroy the one property that module exists to hold.

---

## 2. The layers, bottom up

### 2.1 `core/settings.py` — the writable overlay (new)

One JSON document at `appdata.workspace_state_dir()/settings.json` — **per
workspace, not per machine.** An earlier draft of this plan put it in
`state_dir()`, and that was wrong: `paths.config_path()` already resolves
`grad.toml` per workspace, so a machine-global overlay would silently flatten
two workspaces that had deliberately different model choices. An overlay must
mirror the scope of the file it shadows.

It stays out of the workspace *folder* for the reason `core/workspace.py` keeps
the root pointer beside the code: these are answers about how this install is
wired up, and a research folder copied to a colleague's machine should not carry
this one's SSH inventory. `appdata.workspace_state_dir()` is exactly that seam —
keyed by workspace, stored with the installation — and it is already where the
window layouts live.

Note an existing inconsistency this makes visible: `tools/kaggle.py:179` writes
`kaggle.json` to the machine-global `state_dir()` while shadowing `[kaggle]
username`, which is per-workspace. Not urgent, and not this plan's to fix — but
`setup show` will report it, so it should be a deliberate answer rather than a
surprise.

```python
def load() -> dict[str, Any]
def models() -> dict[str, str]            # role -> model, only what was set
def set_models(**roles: str) -> dict      # validates against config.MODEL_ROLES
def default_backend() -> str | None
def set_backend(name: str) -> dict
def hosts() -> dict[str, dict]            # SSH inventory added through setup
def add_host(name, hostname, user, *, rate_usd_per_hour, workdir, gpus) -> dict
def shadowing(cfg: Config) -> list[dict]  # what here overrides what there
```

`shadowing()` is not optional decoration — it is the thing that keeps 1.3
honest. A user who edits `[models] evolve` in the TOML and sees no change needs
the app to tell them why.

**Validation belongs here, not in the UI.** Model role names against
`config.MODEL_ROLES`; host names against the same constraint `config.host()`
enforces; backend against `evolve.REMOTE_BACKENDS`. A bad value written by a
button is a bad value the next campaign reads.

### 2.2 `core/config.py` — teach the resolver about the overlay

`model_for` gains the overlay above `self.user`, and `hosts` merges
`settings.hosts()` into `[hosts.*]`. Both stay pure functions of their inputs;
the overlay is read once in `load()` and carried on `Config`, so nothing on the
gate path grows a file read.

**The `[hosts.*]` inventory rule survives this.** `core/config.py:520` refuses an
unknown host name — "the inventory is fixed, never an ad-hoc connection". Setup
does not weaken that; it gives the inventory a second, writable source. The
refusal message should name both places a host can be defined.

### 2.3 `core/budget.py` — per-project settings as an event

```python
T_PROJECT_CONFIGURED = "project_configured"

def configure(project_id, *, models=None, backend=None, reason="") -> dict
```

Folded in `projects()` beside `T_PROJECT_RAISED` (`core/budget.py:164`). An
event rather than a field for the same reason a raise is one: the record of what
this project was set to, and when, is worth more than the current value alone —
and a campaign whose model changed mid-flight is exactly the thing a ledger
should be able to answer.

**Confirmed as real work, not a maybe** (decision, 2026-08-17): the model chosen
per role is the main lever on both cost and quality, and it is exactly the thing
that should differ between a cheap exploratory project and one being written up.

Two consequences that are easy to miss:

**The running session has to be rebuilt when `research` changes.** `ui/app.py:172`
builds `ClaudeSDKClient` options from `cfg` once, at client start. Switching to a
project that overrides `research` while a session is live leaves the old model
answering — silently, and while the projects window shows the new one. The
precedent is already there: `client_effort` (`ui/app.py:178`) exists so an effort
change can decide whether a rebuild is needed. Model selection needs the same
recorded-and-compared treatment, and `use_project` becomes a rebuild trigger.

**Which roles may be overridden is a decision, not a default.** All six is the
obvious answer and probably right, but `research` is the one with the live-session
problem above, and `cite`/`triage` are described in `config/grad.toml:70-73` as
mechanical work where a cheaper model is the point. Start with all six overridable
and let the UI's ordering carry the advice.

### 2.4 The CLIs — because every button runs one

§10 and `tests/test_ui_argv.py:27`: the UI builds argv and a flag that does not
parse is a dead button. New commands:

```bash
python -m tools.setup show --json                    # every layer, and what wins
python -m tools.setup models --evolve claude-opus-5 --json
python -m tools.setup backend --default kaggle --json
python -m tools.setup host add --name gpu-box --hostname … --user … --rate 1.20 --json
python -m tools.setup check --json                   # what is missing for each backend
```

`setup check` is the one that earns its place: it answers "can I actually submit
to Kaggle right now" by testing the credential *pair*, the way
`kaggle account --check` already does. A wizard that stores a token and never
tries it produces a green checkmark and a failure an hour later.

Every new argv goes into `UI_COMMANDS`.

### 2.5 `ui/models.py` — two shaped models

- `projects_model()` — id, title, status, payer, ceilings *and* spend against
  each, memory-file freshness (`core/projects.py` knows whether the generated
  files are stale), last run. Every reader wrapped in `_safe`, per the rule
  `workspaces_model` already follows: this panel has to render when the
  workspace is wrong.
- `setup_model()` — per step: satisfied / missing / unknown, plus what each
  answer would unlock. This is where `hf_token` stops being unconditionally
  "required" (`ui/models.py:539`) and becomes required *for HF Jobs*, which is
  the actual claim.

### 2.6 The windows

`ui/windows/projects.py` — list, select, create, close, per-project ceilings
inline, and a button that opens setup for that project.

`ui/windows/setup.py` — the stepper.

**Step state cannot live in a Python local.** `ui/static/tiling.js:184` — the
pane tree is rebuilt by the server on every retile, and non-persistent windows
are rebuilt on refresh ticks. The step index and the in-progress answers belong
in workspace state keyed by window id. Note also that `WindowSpec.persistent` is
declared (`ui/registry.py:36`) and asserted in `tests/test_ui_registry.py:106`
but has **no Python-side consumer I could find** — confirm what actually honours
it before relying on it for this window.

`kit` needs one new primitive: a step header (n of m, back/next, a disabled next
until the step validates). Everything else — `kit.button`, `kit.chip`,
`kit.error_strip`, the password-shaped input from `_credentials` — already
exists.

### 2.7 Dismantle the junk drawer

Three scopes, three homes (decision, 2026-08-17):

| control | goes to |
| --- | --- |
| switch project | the projects window, and `project ▾` for the quick switch |
| switch workspace folder, recent list | a new `workspace ▾` appbar control |
| credentials | the setup window |
| version and update | the setup window (the `↑ v0.2.0` appbar button stays) |

`workspace ▾` shows the folder's **basename**, with the full path in the tooltip
— `model["root"]` is an absolute path and the appbar cannot carry one. It gets a
confirmation step, and it is the only control in the app that does: a project
switch changes what spend is charged to, while a folder switch replaces the
ledger, the project list, the notebooks and possibly the whole model config
under every open window. In the current menu those two sit six rows apart and
are styled identically (`ui/shell.py:342` and `:357`), which is the specific
thing this fixes.

This is the change that makes the projects window worth having rather than a
fourth place to look.

---

## 3. Sequence

Each stage is shippable and independently useful.

**Stage 1 — the projects window, and the `workspace ▾` split.** `projects_model`,
`ui/windows/projects.py`, the `WindowSpec`, the existing create/use/ceiling
actions moved into it, and the folder picker lifted into its own appbar control
(§2.7). No new storage — every action here already has a CLI behind it. Ends
with: projects is a tileable window, and switching folders no longer looks like
switching projects.

**Stage 2 — `core/settings.py` + `tools/setup.py` + the resolver layers.** No
UI. Ends with: `python -m tools.setup models --evolve …` works, `setup show`
explains what wins, `config.model_for` honours it, and none of it touched
`grad.toml`.

**Stage 3 — the setup window, machine half.** Token → models → backends, driven
by `setup_model`. Runs automatically on first launch when
`setup check` reports nothing configured. Ends with: a fresh machine is usable
without opening a terminal — which is the actual goal, and the README's install
section shrinks to `pip install` plus "open Grad".

**Stage 4 — the project half, and the create hook.** Ceilings, payer, default
backend, on create. Skips satisfied machine steps. Ends with: the "created with
no ceilings" hole (`ui/shell.py:402`) is closed.

**Stage 5 — per-project model overrides.** `T_PROJECT_CONFIGURED`, the top
resolution layer, and the session rebuild on project switch. Confirmed in scope.
Ends with: a project can be cheap or careful, and switching to it actually
changes which model answers.

---

## 4. Tests each stage owes

- Stage 1: `test_ui_registry` (id, defaults, the persistent set), `test_ui_argv`
  for every button, `projects_model` against a real ledger — including a
  workspace with an unparseable one, per `workspaces_model`'s rule.
- Stage 2: resolution order, with a case per layer; shadowing report; refusal on
  an unknown role and an unknown backend; **`grad.toml` byte-identical after
  every setup command** — that is the test 1.3 actually needs.
- Stage 3: `setup_model` step states; the credential-pair check surfacing a
  wrong Kaggle key as a failed step rather than a stored one.
- Stage 4: create-with-ceilings goes through `budget.create`, not a raise; a
  re-run wizard does not rewrite ceilings silently.
- Stage 5: the fold, and a project whose override survives a compaction.

---

## 5. Decisions and open questions

**Settled 2026-08-17:**

1. **Per-project models are real.** They move cost and quality, which is the
   whole reason to have them. Stage 5 is in scope; see §2.3 for the two
   consequences.
2. **A hand-edited `grad.toml` is read, shown, and left alone.** Setup presents
   its values as the current answers and writes to the overlay only when one is
   changed — so a user who configured everything by hand meets a wizard that
   agrees with them, and their comments survive (§1.3).

3. **The workspace folder gets its own `workspace ▾` control**, not a strip
   inside the projects window — because switching folders changes what *every*
   open window is showing, and a control that destructive should not sit inside
   a list of projects as one more row. See §2.7 for the full split.

**Still open:** nothing blocking. Stage 1 can start.

---

## 6. Fixed on the way here

Two bugs found during the audit, fixed before this plan was written:

- **`kaggle_key` had no purpose text.** It was in `credentials.ALL` and not in
  `ui/models.py:CREDENTIAL_NOTES`, so the panel drew the free backend's
  credential with an empty purpose column. Test asserts against `ALL` rather
  than by name, because the hole was drift between two hand-written lists.
- **A stored Claude token never reached the agent's own loop.** The main loop
  authenticates from ambient `CLAUDE_CODE_OAUTH_TOKEN` (`agent.py`), `sdk_env`
  reads the credential store, and nothing joined them — so a token stored
  through the credentials panel ran the funnel and the mutation operator and
  left the main loop unauthenticated. It bit the installed app and not the
  terminal, which is why it survived. `credentials.hydrate_environment()` now
  bridges them, after the scrub and never over an exported token.

The second one is why step 1 of the wizard is worth building: pasting a token
into the app is now sufficient, and before this it silently was not.

---

## 7. What changed in the building

**The overlay is per workspace, not per machine** (§2.1). `paths.config_path()`
is `_shipped(...)`, so `grad.toml` was already workspace-overridable and a
machine-global overlay would have flattened two workspaces that had deliberately
different model choices. Caught by reading the resolver rather than by a test,
which is the wrong order and worth saying.

**The two overlays stayed separate rather than merging.** `Config` carries
`overlay` and `project_overlay` as distinct fields and `model_for` takes
`project=False`. Merging them would have been less code, and the projects window
could not then have answered "what would this role be if this project said
nothing" — which it has to, because it draws every project and only one of them
is selected.

**`kit.attr` did not escape backslashes.** NiceGUI hands each props value to
`ast.literal_eval`, so `C:\Users\...` in a tooltip contains `\U`, begins a
unicode escape, and raises a SyntaxError out of `element.props()` — taking down
whatever was being built rather than the tooltip. It had never fired because no
control had ever put a *path* in a tooltip; `workspace ▾` is the first.

**One bug the suite could not see.** `_project()` referenced a name that was not
in its scope, and the whole suite stayed green: the empty-workspace render test
returns before reaching that branch, and the populated one caught the NameError
in the failure card `shell._render` deliberately draws. That card is the right
behaviour and it is also a blind spot, so
`test_every_window_renders_with_real_data` now asserts on *content* per window
and that the string "failed to render" appears nowhere. Verified by
reintroducing the bug and watching the test fail.

Two things the plan called for and did not need: a `persistent` flag on the
setup window (step state lives in `workspace.selection`, which survives both a
rebuild and a retile, so the flag's unclear Python-side consumer never became
load-bearing), and a `kit` stepper with back/next — the steps are tabs, because
none of them gates another and a wizard you cannot go back in is one people
abandon at question three.
