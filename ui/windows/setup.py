"""Window 14 — setup: the questions a fresh machine has to answer.

Four steps, and all four are about *this machine and this workspace*: the
subscription token, which model runs which role, where a training run executes,
and the optional keys that widen retrieval. None of them is about a project,
which is the whole reason this is a separate surface from the projects window --
a wizard that asked for a Claude token every time someone created a project
would ask a user with six projects the same question six times.

Two decisions worth knowing before changing anything here.

**The step is a tab, not a stage.** Every step stays reachable and none of them
gates another. A setup that has to be restarted to change the answer to question
two is a setup people abandon at question three, and there is nothing here whose
answer depends on an earlier one.

**The step index lives in the workspace, not in a closure.** A non-persistent
window is rebuilt whenever its model changes and its whole subtree is rebuilt on
every retile (`ui/static/tiling.js`), so a step held in a Python local is a step
that resets when a background poll notices a new run. `workspace.selection` is
the same mechanism the ledger's filter chips and the funnel's trace picker use.

Nothing here writes a credential to a file or reads one back: `credential set`
takes the value down a pipe (`ui/state.py:set_credential`), and
`credentials.status()` returns booleans.
"""

from __future__ import annotations

from typing import Any

from ui import kit


def subtitle(workspace: Any) -> str:
    model = workspace.model("setup") or {}
    steps = model.get("steps") or []
    done = len([s for s in steps if s["ready"]])
    return f"{done}/{len(steps)} answered" if steps else "nothing configured yet"


def chips(workspace: Any) -> list[tuple[str, str]]:
    model = workspace.model("setup") or {}
    if not (model.get("token") or {}).get("ready"):
        return [("NOT AUTHENTICATED", "broken")]
    if not model.get("complete"):
        return [("NO BACKEND", "attention")]
    return []


def render(workspace: Any) -> None:
    model = workspace.model("setup") or {}
    kit.error_strip(model.get("error"))

    steps = model.get("steps") or []
    if not steps:
        kit.empty("Setup could not read this machine's configuration.")
        return

    _first_run(workspace, model)

    active = workspace.selection.get("setup.step") or steps[0]["id"]
    if active not in {s["id"] for s in steps}:
        active = steps[0]["id"]

    kit.steps(steps, active, lambda step_id: workspace.select("setup.step", step_id))

    # `.get`, not `[]`. A step added to `SETUP_STEPS` without a body here would
    # otherwise raise a KeyError out of `render` and take the whole window down
    # -- the one window whose job is to be usable when nothing else works.
    body = {
        "token": _token,
        "models": _models,
        "context": _context,
        "backends": _backends,
        "extras": _extras,
    }.get(active)
    if body is None:
        kit.empty(f"the {active} step has no body yet")
    else:
        body(workspace, model)

    _installation(workspace)


# ---------------------------------------------------------------------------
# 0. the first run
# ---------------------------------------------------------------------------
def _first_run(workspace: Any, model: dict[str, Any]) -> None:
    """Three things to do, in order, on a machine that has never been set up.

    **Not a tour, and deliberately not one.** A guided overlay would have to
    dim the workspace and float a callout with a drop shadow over it -- three
    separate contradictions of a design whose motion rule is "instant state
    swaps" and whose shadow rule is "no blur anywhere", enforced by
    `tests/test_ui_tokens.py`. It would also have to know where a window *is*,
    in a workspace whose whole premise is that you drag windows wherever you
    want them, and it would have to be dismissable -- at which point it teaches
    nobody who dismissed it.

    A list of what is not done yet has none of those problems. It is drawn in
    the ordinary components, it survives being read twice, and it goes away by
    being satisfied rather than by being closed. See `ui/models.py:first_run`
    for why the state behind it is derived rather than stored.

    Rows are buttons: the point of putting this here is that the next action is
    one click away rather than one paragraph away.
    """
    run = model.get("first_run") or {}
    if not run.get("active"):
        return

    with kit.el("div", "grad-note").style("margin: 0 0 12px"):
        with kit.row("", gap=9):
            kit.text("FIRST RUN", "grad-label", tag="span")
            kit.spacer()
            kit.text(f"{run.get('done', 0)} of {run.get('total', 0)} done", "grad-caption", tag="span")
        kit.text(
            "Grad is not configured yet. These are the things it needs, in order.",
            "grad-caption",
        ).style("margin: 6px 0 9px")

        for step in run.get("steps") or []:
            _first_run_row(workspace, step)


def _first_run_row(workspace: Any, step: dict[str, Any]) -> None:
    done = bool(step.get("done"))
    # A filled square for done and an empty one for not, the same mark the
    # windows menu uses -- one vocabulary for "this is in effect".
    row = kit.menu_row(
        "■" if done else "□",
        step.get("title", ""),
        step.get("why", ""),
        open=done,
        title=step.get("why", ""),
        # A finished step has nothing to click, and a `<button>` with no handler
        # is still a tab stop -- so keyboard navigation through this panel used
        # to land on rows that do nothing. `disabled` takes them out of the
        # order and dims them, which is also the honest appearance.
        disabled=done,
    )
    if done:
        return

    def go() -> None:
        target = step.get("opens") or "setup"
        if target != "setup":
            workspace.open(target)
            workspace.focus(target)
            return
        # Same window: move the step tabs rather than opening anything.
        if step.get("step"):
            workspace.select("setup.step", step["step"])

    row.on("click", lambda _=None: go())


# ---------------------------------------------------------------------------
# 1. the subscription
# ---------------------------------------------------------------------------
def _token(workspace: Any, model: dict[str, Any]) -> None:
    from nicegui import ui

    token = model["token"]
    with kit.pad():
        kit.label("subscription token")
        if token["state"] == "stored":
            kit.chip("STORED", "ok")
            kit.text(
                "The agent's own loop, the funnel's Haiku stages and the mutation operator all "
                "authenticate with this.",
                "grad-caption",
            )
        elif token["state"] == "environment":
            kit.chip("ENVIRONMENT ONLY", "attention")
            # The distinction that actually bites, and only on the installed app:
            # a shell that exported the token has it, and the desktop shortcut
            # launches from Explorer with whatever was made persistent.
            kit.note(
                "A token is set in this process's environment, so the agent works right now — but "
                "it is not stored. Launched from the desktop shortcut, which inherits whatever "
                "Explorer had, there would be nothing to authenticate with. Paste it below to "
                "keep it."
            )
        else:
            kit.chip("MISSING", "broken")
            kit.note(
                "Nothing can reach a model without this. Everything else in this window is "
                "optional by comparison."
            )

        kit.text("mint one in a terminal, then paste the result:", "grad-caption").style(
            "margin-top: 10px"
        )
        kit.pre(token["mint"], wrap=True)

        with kit.row("", gap=6).style("margin-top: 8px"):
            value = (
                ui.input(placeholder="paste the token")
                .props("borderless dense type=password")
                .classes("field")
                .style("flex: 1 1 auto; padding: 0 8px")
            )

            def store() -> None:
                pasted, value.value = value.value or "", ""
                workspace.spawn(
                    workspace.set_credential(token["name"], pasted), "credential set"
                )

            kit.button("STORE", tone="primary", on_click=store)
            kit.button(
                "✕",
                tone="neutral",
                disabled=not token["stored"],
                title="forget it",
                on_click=lambda: workspace.spawn(
                    workspace.delete_credential(token["name"]), "credential delete"
                ),
            )
        kit.text(
            "stored in the OS credential store, never in the workspace and never in the agent's "
            "environment — it is fetched at the moment of use",
            "grad-caption",
        )


# ---------------------------------------------------------------------------
# 2. the six roles
# ---------------------------------------------------------------------------
def _models(workspace: Any, model: dict[str, Any]) -> None:
    from nicegui import ui

    with kit.pad():
        kit.label("one model per role")
        kit.text(
            "Chosen here, these outrank config/grad.toml — which is hand-annotated and cannot be "
            "machine-written without losing every comment in it.",
            "grad-caption",
        )

        for role in model["roles"]:
            with kit.el("div", "grad-card").style("margin: 9px 0"):
                with kit.row("head", gap=9):
                    kit.text(role["role"], "", tag="span").style("font-weight: 700")
                    kit.spacer()
                    kit.chip(role["source"], "ok" if role["overridden"] else "neutral")
                with kit.el("div", "body"):
                    kit.text(role["model"], "grad-mono")
                    with kit.row("", gap=6).style("margin-top: 8px; flex-wrap: wrap"):
                        for known in model["known_models"]:
                            kit.button(
                                known,
                                tone="active" if known == role["model"] else "neutral",
                                disabled=known == role["model"],
                                on_click=lambda _=None, r=role["role"], m=known: workspace.spawn(
                                    workspace.set_model(r, m), "model set"
                                ),
                            )
                    with kit.row("", gap=6).style("margin-top: 6px"):
                        # The list above ages the moment a new model ships. This
                        # is the mechanism; the buttons are the shortcut.
                        other = (
                            ui.input(placeholder="or any other model id")
                            .props("borderless dense")
                            .classes("field")
                            .style("flex: 1 1 auto; padding: 0 8px")
                        )
                        kit.button(
                            "SET",
                            tone="primary",
                            on_click=lambda _=None, r=role["role"], f=other: workspace.spawn(
                                workspace.set_model(r, (f.value or "").strip()), "model set"
                            ),
                        )
                        kit.button(
                            "RESET",
                            tone="neutral",
                            disabled=not role["overridden"],
                            title=f"fall back to the config, then to {role['default']}",
                            on_click=lambda _=None, r=role["role"]: workspace.spawn(
                                workspace.clear_model(r), "model reset"
                            ),
                        )

        _shadowing(model)


# ---------------------------------------------------------------------------
# 3. how much conversation the agent carries
# ---------------------------------------------------------------------------
def _context(workspace: Any, model: dict[str, Any]) -> None:
    """The context budget, stated as what it actually is.

    The window is careful not to promise what it cannot deliver. The *context
    window* belongs to the model -- the CLI underneath reports 1,000,000 with
    its own threshold at 967,000, and `core/compaction.py` explains why neither
    is reachable from here. What this sets is where Grad compacts inside it,
    which is the number that decides how much conversation the agent is
    carrying. Calling that "context length" in the caption and explaining the
    difference in the note is more honest than a slider labelled with a number
    nothing here controls.
    """
    from nicegui import ui

    context = model.get("context") or {}
    tokens = int(context.get("tokens") or 0)
    low, high = (context.get("bounds") or [0, 0])[:2]

    with kit.pad():
        kit.label("context budget")
        with kit.row("", gap=9).style("margin: 6px 0 2px"):
            kit.text(
                f"{tokens:,} tokens" if tokens else "compaction off",
                "grad-mono",
                tag="span",
            ).style("font-size: 19px; font-weight: 700")
            kit.chip(context.get("source", "default"), "ok" if context.get("overridden") else "neutral")
            kit.spacer()
        kit.text(
            "Where Grad compacts. Past this many tokens the conversation is replaced by a "
            "handover note the next session reads.",
            "grad-caption",
        )

        with kit.row("", gap=6).style("margin-top: 10px; flex-wrap: wrap"):
            for preset in context.get("presets") or []:
                kit.button(
                    f"{preset // 1000}K",
                    tone="active" if preset == tokens else "neutral",
                    disabled=preset == tokens,
                    on_click=lambda _=None, value=preset: workspace.spawn(
                        workspace.set_context_budget(str(value)), "context budget"
                    ),
                )
            kit.button(
                "OFF",
                tone="active" if not tokens else "neutral",
                disabled=not tokens,
                title="leave compaction to the CLI underneath, which compacts at its own threshold",
                on_click=lambda: workspace.spawn(
                    workspace.set_context_budget("0"), "context budget"
                ),
            )

        with kit.row("", gap=6).style("margin-top: 6px"):
            entry = (
                ui.input(placeholder=f"or any number between {int(low):,} and {int(high):,}")
                .props("borderless dense")
                .classes("field")
                .style("flex: 1 1 auto; padding: 0 8px")
            )
            entry.on(
                "keydown.enter.prevent",
                lambda _=None: workspace.spawn(
                    workspace.set_context_budget(entry.value or ""), "context budget"
                ),
            )
            kit.button(
                "SET",
                tone="primary",
                on_click=lambda _=None: workspace.spawn(
                    workspace.set_context_budget(entry.value or ""), "context budget"
                ),
            )
            kit.button(
                "RESET",
                tone="neutral",
                disabled=not context.get("overridden"),
                title=f"fall back to the config, then to {int(context.get('default') or 0):,}",
                on_click=lambda: workspace.spawn(
                    workspace.clear_context_budget(), "context budget reset"
                ),
            )

        kit.note(
            "The window itself is the model's, not Grad's: the CLI underneath runs a "
            "1,000,000-token context and compacts at 967,000 of it. This is where Grad "
            "compacts first, and it is a trade in both directions — a larger budget means "
            "every tool round-trip re-reads more cached context, while a smaller one "
            "compacts more often and each compaction costs a turn plus a cold prompt cache. "
            "`python -m tools.quota summary --json` is where the trade becomes visible."
        )
        _shadowing(model)


def _shadowing(model: dict[str, Any]) -> None:
    """What these choices are overriding, said out loud.

    The price of being allowed to win. Someone edits `[models] evolve`, sees no
    change, and has no way to discover that a file they have never heard of
    outranks the one they were told to edit -- unless it is written here.
    """
    rows = model.get("shadowing") or []
    if not rows:
        return
    kit.note(
        "These override values set in "
        + model.get("config_path", "config/grad.toml")
        + ", which still says something different:"
    )
    kit.kv([(row["what"], f"{row['config']} → {row['overlay']}") for row in rows])


# ---------------------------------------------------------------------------
# 3. where a run executes
# ---------------------------------------------------------------------------
#: What each backend is, in the one line that decides whether to bother with it.
BACKEND_NOTES = {
    "kaggle": "free GPU/TPU hours, rationed in hours rather than dollars",
    "hf_jobs": "Hugging Face Jobs, priced per hour against the GPU ceiling",
    "ssh": "your own machines, priced by the rate you record for each",
    "modal": "Modal sandboxes, H100s and up, billed per second against the spend ceiling",
}


def _backends(workspace: Any, model: dict[str, Any]) -> None:
    with kit.pad():
        kit.label("where a run executes")
        kit.text(
            "Not alternatives — the useful arrangement is a mixture, and the default below is a "
            "preference rather than a restriction. --remote still names one per campaign.",
            "grad-caption",
        )

        for backend in model["backends"]:
            name = backend["backend"]
            with kit.el("div", "grad-card").style("margin: 9px 0"):
                with kit.row("head " + ("" if backend["ready"] else "attention"), gap=9):
                    kit.text(name, "", tag="span").style("font-weight: 700")
                    kit.chip("READY" if backend["ready"] else "NOT CONFIGURED",
                             "ok" if backend["ready"] else "attention")
                    kit.spacer()
                    if model["default_backend"] == name:
                        kit.chip("DEFAULT", "ok")
                    else:
                        kit.button(
                            "MAKE DEFAULT",
                            tone="neutral",
                            on_click=lambda _=None, b=name: workspace.spawn(
                                workspace.set_backend(b), "backend default"
                            ),
                        )
                with kit.el("div", "body"):
                    kit.text(BACKEND_NOTES.get(name, ""), "grad-caption")
                    if backend["missing"]:
                        kit.text(
                            "missing: " + ", ".join(backend["missing"]), "grad-caption"
                        ).style("margin-top: 6px")
                    # The half that is *not* a credential, named explicitly
                    # rather than falling through to `_hosts`. A fourth backend
                    # would otherwise be handed an SSH host editor, which is not
                    # merely useless -- it invites someone to add an inventory
                    # entry that backend will never read.
                    if name == "kaggle":
                        _kaggle(workspace, model)
                    elif name == "ssh":
                        _hosts(workspace, model)
                    # The credentials themselves are not dispatched on at all.
                    # They used to be, and `modal` was the branch nobody wrote:
                    # the backend was registered in `settings.BACKENDS`, in
                    # `REQUIREMENTS`, and in `CREDENTIAL_NOTES`, so its card drew
                    # with NOT CONFIGURED and named both missing tokens -- above
                    # a body with nothing to type them into. `_extras` does not
                    # catch them either; it filters to the retrieval and extras
                    # groups, and a backend credential is in neither. There was
                    # no field anywhere in the app.
                    for credential in backend.get("credentials") or ():
                        _credential_field(workspace, credential)


def _kaggle(workspace: Any, model: dict[str, Any]) -> None:
    """Two halves, and only one of them is a secret.

    The username is stored where it can be read back, because "whose kernels are
    these?" deserves a file you can open. Only that half is here: the key is a
    credential, and `_backends` draws a field for every credential a backend
    declares. Drawing it here too would put two `kaggle_key` boxes on the card.
    """
    from nicegui import ui

    account = model.get("kaggle") or {}
    kit.kv([("username", account.get("username") or "—"), ("from", account.get("source") or "—")])
    with kit.row("", gap=6).style("margin-top: 6px"):
        username = (
            ui.input(placeholder="your kaggle username")
            .props("borderless dense")
            .classes("field")
            .style("flex: 1 1 auto; padding: 0 8px")
        )
        kit.button(
            "SET",
            tone="primary",
            on_click=lambda _=None, f=username: workspace.spawn(
                workspace.set_kaggle_account((f.value or "").strip()), "kaggle account"
            ),
        )


def _hosts(workspace: Any, model: dict[str, Any]) -> None:
    """The inventory is fixed by design; this is its writable half.

    A host that can be named ad-hoc is a general remote-execution capability, so
    one has to be added on purpose before anything can reach it. Adding it here
    rather than by editing TOML does not change that.
    """
    from nicegui import ui

    for name in model.get("hosts") or []:
        with kit.row("grad-row", gap=6):
            kit.chip(name, "outline")
            kit.spacer()
            kit.button(
                "✕",
                tone="neutral",
                title="remove it from the inventory added here",
                on_click=lambda _=None, h=name: workspace.spawn(
                    workspace.remove_host(h), "host remove"
                ),
            )

    fields: dict[str, Any] = {}
    with kit.row("", gap=6).style("margin-top: 6px; flex-wrap: wrap"):
        for key, placeholder, width in (
            ("name", "name, e.g. gpu-box", 140),
            ("hostname", "hostname ssh connects to", 200),
            ("user", "ssh user", 110),
            ("rate", "$/hour (0 if free)", 120),
        ):
            fields[key] = (
                ui.input(placeholder=placeholder)
                .props("borderless dense")
                .classes("field")
                .style(f"flex: 0 0 {width}px; padding: 0 8px")
            )
        kit.button(
            "ADD HOST",
            tone="primary",
            on_click=lambda: workspace.spawn(
                workspace.add_host(
                    (fields["name"].value or "").strip(),
                    (fields["hostname"].value or "").strip(),
                    (fields["user"].value or "").strip(),
                    (fields["rate"].value or "0").strip(),
                ),
                "host add",
            ),
        )
    kit.text(
        "the rate is what `collect` prices wall clock against — a wrong one is a spend-accounting "
        "problem, and 0 is correct for a machine you already pay for",
        "grad-caption",
    )


# ---------------------------------------------------------------------------
# 4. the optional keys
# ---------------------------------------------------------------------------
def _extras(workspace: Any, model: dict[str, Any]) -> None:
    rows = [
        r for r in model["credentials"]["rows"] if r["group"] in ("retrieval", "extras")
    ]
    with kit.pad():
        kit.label("optional keys")
        kit.text(
            "Everything here degrades rather than fails. Retrieval works without any of them; "
            "each one widens or speeds up a stage.",
            "grad-caption",
        )
        kit.error_strip(model["credentials"].get("error"))
        for row in rows:
            with kit.column("grad-row", gap=6):
                with kit.row("", gap=6).style("width: 100%"):
                    kit.chip(row["state"], row["tone"])
                    kit.text(row["name"], "grad-mono", tag="span")
                    kit.text(
                        row["purpose"], "grad-caption", tag="span",
                        style="flex: 1 1 auto; min-width: 0",
                    )
                _credential_field(workspace, row["name"])


def _credential_field(workspace: Any, name: str) -> None:
    """One paste-to-store row. The value goes down a pipe, never in an argv."""
    from nicegui import ui

    with kit.row("", gap=6).style("width: 100%; margin-top: 6px"):
        value = (
            ui.input(placeholder=f"paste {name}")
            .props("borderless dense type=password")
            .classes("field")
            .style("flex: 1 1 auto; padding: 0 8px")
        )

        def store(_=None) -> None:
            pasted, value.value = value.value or "", ""
            workspace.spawn(workspace.set_credential(name, pasted), "credential set")

        kit.button("SET", tone="neutral", on_click=store)
        kit.button(
            "✕",
            tone="neutral",
            title="forget it",
            on_click=lambda _=None: workspace.spawn(
                workspace.delete_credential(name), "credential delete"
            ),
        )


# ---------------------------------------------------------------------------
# the footer
# ---------------------------------------------------------------------------
def _installation(workspace: Any) -> None:
    """Which Grad this is, and the one button that changes it.

    Not a step -- it is never *answered* -- but it belongs on this surface for
    the same reason the credentials do: it is a fact about the installation, and
    it used to live behind a control labelled `project`.
    """
    model = workspace.update()

    kit.hr()
    with kit.pad():
        kit.label("this installation")
        kit.kv([("version", model["installed"]), ("last checked", model["checked"])])
        if not model["is_checkout"]:
            kit.note(
                "This copy was not installed from a git checkout, so it cannot update itself. "
                "Reinstall from the repository to get updates."
            )
            return
        for warning in model["warnings"]:
            kit.note(f"{warning['message']} — {warning['fix']}")
        for blocker in model["blockers"]:
            kit.error_strip(f"{blocker['message']} — {blocker['fix']}")
        with kit.row("", gap=6).style("margin-top: 8px"):
            if model["available"]:
                kit.chip(f"{model['target']} AVAILABLE", "attention")
                kit.button(
                    "UPDATE",
                    tone="primary",
                    title=(
                        "quit first: this release changes dependencies"
                        if model["needs_reinstall"]
                        else "fast-forward this installation and migrate its state"
                    ),
                    on_click=lambda: workspace.spawn(workspace.apply_update(), "update"),
                )
            kit.button(
                "CHECK NOW",
                tone="neutral",
                title="ask the remote whether there is a newer release",
                on_click=lambda: workspace.spawn(workspace.check_update(), "update check"),
            )
            kit.spacer()
        if model["dirty"]:
            kit.text(
                "the installation has uncommitted edits; runs submitted from it are stamped "
                "as modified and `report check` will say so",
                "grad-caption",
            )
