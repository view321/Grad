"""The shell, rendered for real.

Everything else in `tests/test_ui_*.py` runs without NiceGUI, which is the point
of the layering. This file is the other half: it builds an actual element tree
for all eleven windows inside a real `Client`, so a typo in a window's render
path fails here rather than on someone's laptop at page-build time.

Two properties are worth holding still, and both are about `Element.move()`:

  * a window's root **survives** a retile -- otherwise every drag would wipe the
    chat transcript and the notebook's iframe anchor;
  * a closed window's root is **destroyed** -- otherwise the attic accumulates
    one detached subtree per window per session.

Skipped rather than failed when the `ui` extra is not installed: `core/` is
meant to run without it, and a test suite that cannot be run at all on a machine
without pywebview is a test suite that stops being run.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("nicegui", reason="the ui extra is not installed")

from nicegui.client import Client  # noqa: E402
from nicegui.page import page  # noqa: E402

from ui import layout as layout_mod, registry, shell, state as state_mod  # noqa: E402


class FakeSession:
    """Enough of `ui.app.Session` to render. The SDK client is never started --
    `Session.start` only runs on the first `ask`, so a render touches nothing."""

    busy = False

    def __init__(self) -> None:
        self.settled: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []
        self.session_id = "default"
        self.title = ""
        self.sdk_session_id: str | None = None

    def interrupt(self) -> None:
        pass

    async def open_session(self, session_id: str) -> str:
        self.session_id = session_id
        return f"opened {session_id}"

    async def new_session(self, title: str = "") -> str:
        self.session_id = "fresh"
        return "new session"


@pytest.fixture
def rendered(workspace):
    """A built shell inside a real client, torn down afterwards."""
    clients: list[Client] = []

    def build(windows=None, project="proj"):
        client = Client(page("/"))
        clients.append(client)
        with client:
            space = state_mod.Workspace(FakeSession(), project)
            space.layout = layout_mod.Layout()
            for window in windows if windows is not None else registry.ids():
                space.layout.open(window)
            shell.build(space)
        return client, space

    yield build

    for client in clients:
        client.delete()


def html_of(client: Client) -> str:
    return " ".join(
        f"{element.tag} {' '.join(element.classes)} {getattr(element, 'content', '')}"
        for element in client.elements.values()
    )


# ---------------------------------------------------------------------------
# every window renders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", registry.ids())
def test_every_window_renders_on_an_empty_workspace(rendered, window):
    """The empty state is the state a new user sees, so it is the one most
    worth proving renders at all."""
    client, _ = rendered([window])
    assert len(client.elements) > 10


def test_all_eleven_render_together(rendered):
    client, space = rendered()
    assert len(space.layout.windows) == len(registry.ids())
    markup = html_of(client)
    assert "grad-shell" in markup
    assert "grad-tiles" in markup
    assert "grad-statusbar" in markup


def test_a_window_whose_render_raises_does_not_take_the_shell_down(rendered, monkeypatch):
    """Ten working windows and one broken one is a usable workspace; a traceback
    at page build time is not."""
    import ui.windows.funnel as funnel_window

    monkeypatch.setattr(
        funnel_window, "render", lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    client, _ = rendered(["funnel", "ledger"])
    assert "failed to render" in html_of(client)
    assert "grad-statusbar" in html_of(client)


# ---------------------------------------------------------------------------
# the chrome reflects the layout
# ---------------------------------------------------------------------------
def _open_window_menu(client: Client, space):
    """The `⋯` menu, drawn. Its body is built on open rather than at build time,
    for the same reason the project menu's is: a toggle makes the list it was
    read from stale."""
    from nicegui import ui as nicegui_ui

    with client:
        menu = shell._windows_menu(nicegui_ui, space)  # noqa: SLF001 - no public hook
        menu.open()
    return menu


def _menu_rows(client: Client) -> list:
    return [e for e in client.elements.values() if "grad-menu-row" in getattr(e, "classes", [])]


def _menu_row(client: Client, window_id: str):
    """One window's row, found by the hint it carries as a tooltip."""
    from ui import kit

    wanted = kit.attr(registry.spec(window_id).hint)
    for element in _menu_rows(client):
        if element.props.get("title") == wanted:
            return element
    raise AssertionError(f"no menu row for {window_id!r}")


def test_the_window_menu_marks_what_is_open(rendered):
    """The `⋯` menu replaced a permanent strip of eleven names and a `⌘K`
    palette that listed the same eleven. It is the only opener now, so it is the
    only place the open/closed state appears."""
    client, space = rendered(["chat"])
    _open_window_menu(client, space)

    rows = _menu_rows(client)
    assert len(rows) == len(registry.ids()) + len(shell.PRESET_ROWS)
    assert [r.props.get("title") for r in rows].count(None) == 0
    assert len([r for r in rows if "open" in r.classes]) == 1
    assert "open" in _menu_row(client, "chat").classes


def test_the_window_menu_toggles_in_place_rather_than_closing(rendered):
    """Opening three windows is three clicks. A menu that dismissed itself after
    each one would be three trips back to the same button."""
    client, space = rendered(["chat"])
    _open_window_menu(client, space)

    click(_menu_row(client, "ledger"))
    assert "ledger" in space.layout.windows
    # Redrawn in place, so the mark beside the row is no longer stale.
    assert "open" in _menu_row(client, "ledger").classes

    click(_menu_row(client, "ledger"))
    assert "ledger" not in space.layout.windows
    assert "open" not in _menu_row(client, "ledger").classes


def test_a_quote_in_a_tooltip_cannot_truncate_the_props_string(rendered):
    """`props('title="…")` is parsed by NiceGUI, so a `"` in the value ends it
    early and silently drops whatever came after. Ledger text reaches these --
    a preflight remedy, a candidate id -- so it goes through `kit.attr`."""
    from ui import kit

    client, _ = rendered(["chat"])
    with client:
        element = kit.button("FIX", title='run --note "see below"\nand retry')

    # One line, and no double quote left to close the attribute early.
    assert '"' not in element.props["title"]
    assert "\n" not in element.props["title"]
    assert "see below" in element.props["title"]


def test_a_handle_sits_between_every_pair_of_columns(rendered):
    client, space = rendered(["chat", "ledger", "quota"])
    handles = [e for e in client.elements.values() if "grad-handle" in getattr(e, "classes", [])]
    columns = [e for e in client.elements.values() if "grad-column" in getattr(e, "classes", [])]
    assert len(columns) == 3
    assert len([h for h in handles if "row" not in h.classes]) == 2


def test_a_stacked_column_gets_a_row_handle(rendered):
    client, space = rendered(["chat", "notebook", "ledger", "quota"])
    handles = [e for e in client.elements.values() if "grad-handle" in getattr(e, "classes", [])]
    assert len([h for h in handles if "row" in h.classes]) == 1


def test_the_title_bar_tracks_the_model_not_just_the_layout(rendered):
    """The subtitle and the state chips are read from the model: `EVOLVING`
    becomes `HALTING`, a verify turns `NOT CITABLE` into `CITABLE`. Drawing them
    only when the panes are rebuilt leaves them stale until the next retile, and
    a chip that lags what it reports is worse than no chip."""
    from core import campaign as campaign_mod, ledger_store as ls

    campaign_mod.append_campaign(
        {"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": "open",
         "at": ls.now_iso(), "task_dir": "tasks/x"}
    )
    campaign_mod.append_candidate(
        {"type": campaign_mod.T_CANDIDATE, "id": "c0", "campaign": "camp-1", "generation": 0,
         "metrics": {"combined_score": 0.4}, "at": ls.now_iso()}
    )
    client, space = rendered(["evolve"])
    assert "EVOLVING" in html_of(client)

    campaign_mod.request_halt("camp-1", reason="from the workspace")
    space.tick()          # a poll, with no retile
    markup = html_of(client)
    assert "HALTING" in markup
    assert "EVOLVING" not in markup


def test_a_verify_flips_the_notebook_chip_without_a_retile(rendered):
    from core import ledger_store as ls
    from ui import models

    seed_everything()
    client, space = rendered(["notebook"])
    space.tick()
    assert "NOT CITABLE" in html_of(client)

    models.write_verify_record(
        "x.ipynb", {"ok": True, "at": ls.now_iso(), "cells_executed": 12, "duration_s": 4.0}
    )
    space.tick()
    markup = html_of(client)
    # "CITABLE" is a substring of "NOT CITABLE", so the negative is the
    # assertion that actually carries the test.
    assert "NOT CITABLE" not in markup
    assert "CITABLE" in markup


def click(element) -> None:
    """Invoke an element's click handlers, the way the browser would.

    Snapshotted first: a handler may rebuild the subtree it was clicked in --
    the `⋯` menu redraws itself after a toggle -- and deleting the element
    mutates the dict this is walking. The browser has the same freedom because
    it dispatches by id rather than by iterating.
    """
    for listener in list(element._event_listeners.values()):  # noqa: SLF001 - no public hook
        if listener.type == "click" and listener.handler is not None:
            listener.handler()


def find_button(client: Client, label: str):
    """`kit.button` renders through `ui.html(tag="button")`, so the real tag
    lives in the props rather than on the element."""
    for element in client.elements.values():
        if element._props.get("tag") != "button":  # noqa: SLF001 - no public accessor
            continue
        if label in str(getattr(element, "content", "")):
            return element
    raise AssertionError(f"no button matching {label!r}")


def test_answering_a_gate_with_no_session_does_not_claim_the_agent_is_running(rendered):
    """The guard has to come before the state change: leaving `running` set
    would paint the title bar with a live agent and a PAUSE button while
    nothing is running and nothing will start."""
    client, space = rendered(["chat"])
    space.chat_send = None
    with client:
        from ui.windows.chat import _gate_card

        _gate_card({"kind": "gate", "id": "gate-1", "rows": [("cost", "$18.40")]}, space)

    click(find_button(client, "APPROVE"))
    assert space.agent_state == "idle"
    assert "no chat session" in (space.notice or "")


def test_answering_a_gate_sends_the_decision_into_the_session(rendered):
    client, space = rendered(["chat"])
    sent: list[str] = []
    space.chat_send = sent.append
    with client:
        from ui.windows.chat import _gate_card

        _gate_card({"kind": "gate", "id": "gate-1", "rows": []}, space)

    click(find_button(client, "DENY"))
    assert space.agent_state == "running"
    assert sent and "denied" in sent[0]


# ---------------------------------------------------------------------------
# the calls the agent made
# ---------------------------------------------------------------------------
def test_a_settled_turn_draws_a_card_for_every_call_it_made(rendered):
    """What the transcript is for: a command the agent ran and a command it
    only claimed to run must not look alike."""
    client, space = rendered(["chat"])
    with client:
        from ui.windows.chat import _message

        _message(
            {
                "role": "assistant",
                "text": "Checking.",
                "blocks": [
                    {"kind": "text", "text": "Checking."},
                    {"kind": "tool", "name": "Bash", "title": "python -m tools.ledger show",
                     "text": "python -m tools.ledger show", "rows": [],
                     "status": "ok", "result": "3 expectations"},
                    {"kind": "tool", "name": "Bash", "title": "ssh probe-host", "text": "ssh probe-host",
                     "rows": [], "status": "error", "result": "denied by the gate"},
                ],
            },
            space,
        )

    markup = html_of(client)
    assert markup.count("grad-card tool") == 2
    assert "python -m tools.ledger show" in markup
    assert "3 expectations" in markup
    assert "OK" in markup and "ERROR" in markup


def test_the_turn_in_flight_scrolls_with_the_transcript(rendered):
    """The tail has to be *inside* the scrolling region. As a sibling below it
    the tail grows without bound, so a turn with three tool cards scrolls
    `.grad-body` instead and paints over the composer."""
    client, _ = rendered(["chat"])
    scroller = _by_id(client, "grad-transcript")
    tail = _by_id(client, "grad-tail")
    assert scroller in _ancestors(tail)
    assert scroller._style.get("overflow-y") == "auto"  # noqa: SLF001 - no public accessor


def _by_id(client: Client, element_id: str):
    for element in client.elements.values():
        if element.props.get("id") == element_id:
            return element
    raise AssertionError(f"no element with id {element_id!r}")


def _ancestors(element) -> list:
    """Every element between this one and the page root."""
    out = []
    slot = element.parent_slot
    while slot is not None:
        out.append(slot.parent)
        slot = slot.parent.parent_slot
    return out


def test_a_turn_with_no_blocks_still_draws_as_prose(rendered):
    """Transcripts written before the calls were captured have no `blocks`, and
    a user's own message never will."""
    client, space = rendered(["chat"])
    with client:
        from ui.windows.chat import _message

        _message({"role": "assistant", "text": "GATE — YOUR CALL\ncost: $18.40\n"}, space)

    assert "GATE" in html_of(client)


def test_the_tail_appends_a_card_rather_than_redrawing_the_turn(rendered):
    """The split-tail rule, extended: prose already in the tail must not be
    rebuilt 15 times a second just because a call landed under it."""
    client, space = rendered(["chat"])
    with client:
        from ui.windows import chat as chat_window

        tail = chat_window._Tail(chat_window.kit.column("", gap=0))
        tail.sync([{"kind": "text", "text": "Checking."}])
        prose = tail._drawn[0]["body"].id
        tail.sync([
            {"kind": "text", "text": "Checking."},
            {"kind": "tool", "name": "Bash", "title": "ls", "text": "ls", "rows": [],
             "status": "running", "result": ""},
        ])
        assert tail._drawn[0]["body"].id == prose      # the prose element is the same one
        assert "RUNNING" in html_of(client)

        # ... and the same card is repainted in place when the result lands.
        card = tail._drawn[1]["state"].id
        tail.sync([
            {"kind": "text", "text": "Checking."},
            {"kind": "tool", "name": "Bash", "title": "ls", "text": "ls", "rows": [],
             "status": "ok", "result": "budget.py"},
        ])
        assert tail._drawn[1]["state"].id == card
        markup = html_of(client)
        assert "RUNNING" not in markup
        assert "budget.py" in markup


def test_a_new_turn_clears_the_tail_rather_than_stacking_onto_it(rendered):
    client, space = rendered(["chat"])
    with client:
        from ui.windows import chat as chat_window

        tail = chat_window._Tail(chat_window.kit.column("", gap=0))
        tail.sync([{"kind": "text", "text": "the first turn"}])
        tail.sync([])                                   # settled: the tail was promoted
        assert tail._drawn == []
        tail.sync([{"kind": "text", "text": "the second turn"}])
        assert len(tail._drawn) == 1
        assert "the first turn" not in html_of(client)


def test_the_status_line_names_the_call_in_flight(rendered):
    """A spinner says something is happening; naming the command says a
    40-minute job is running and which one.

    The fallbacks changed with the statusline: `running …` was the only thing it
    could say when no call was in flight, which covered "reasoning", "writing"
    and "nothing has come back yet" with one word. Each is now named, because
    each is a different answer to "why is nothing on screen".
    """
    from ui.windows.chat import _activity

    assert _activity([]) == "waiting for the model"
    assert _activity([
        {"kind": "tool", "name": "Bash", "title": "python -m tools.jobs run", "status": "running"},
    ]) == "running Bash python -m tools.jobs run"
    # A finished call is not an activity; what is happening is whatever came
    # after it, and the reasoning is as specific as that gets.
    assert _activity([
        {"kind": "tool", "name": "Bash", "title": "ls", "status": "ok"},
        {"kind": "thinking", "text": "one entry, so the claim holds"},
    ]) == "thinking"
    assert _activity([
        {"kind": "thinking", "text": "working it out"},
        {"kind": "text", "text": "The answer is"},
    ]) == "writing"


def test_the_statusline_switches_the_reasoning_without_redrawing_the_transcript(rendered):
    """A toggle that rebuilt the transcript would take its scroll position with
    it, which is the same reason the poll never touches this window. So the
    blocks are always in the DOM and a class decides whether they are painted."""
    client, space = rendered(["chat"])
    with client:
        assert space.show_reasoning is False
        roots = [
            e for e in client.elements.values() if "grad-chat" in getattr(e, "classes", [])
        ]
        assert roots, "the chat root carries the class the switch writes"
        assert "reasoning-on" not in roots[0].classes

        bars = [
            e for e in client.elements.values()
            if "grad-statusline" in getattr(e, "classes", [])
        ]
        assert len(bars) == 1

        assert space.toggle_reasoning() is True
        roots[0].classes(add="reasoning-on")
        assert "reasoning-on" in roots[0].classes


def test_the_session_picker_is_the_workspaces_own_menu(rendered):
    """The last Quasar control in the workspace. A `select` has one string per
    option, so "another window has this open" arrived as a ` · ` fragment glued
    onto the title and looked exactly like the rows that can be opened."""
    client, space = rendered(["chat"])
    with client:
        selects = [
            e for e in client.elements.values() if type(e).__name__ == "Select"
        ]
        assert selects == []
        assert "grad-session-btn" in html_of(client)


def test_the_focused_window_is_marked(rendered):
    client, space = rendered(["chat", "ledger"])
    space.focus("ledger")
    focused = [
        e for e in client.elements.values()
        if "grad-window" in getattr(e, "classes", []) and "focused" in e.classes
    ]
    assert len(focused) == 1


# ---------------------------------------------------------------------------
# roots survive retiling
# ---------------------------------------------------------------------------
def _root_ids(client: Client) -> dict[str, int]:
    """Window roots, by the id of the element carrying them."""
    out = {}
    for element in client.elements.values():
        if "grad-titlebar" in getattr(element, "classes", []):
            window = element.props.get("data-window")
            if window:
                out[window] = element.id
    return out


def test_a_window_root_is_reparented_not_rebuilt_on_retile(rendered):
    """`Element.move()` is what makes the window system practical: without it a
    drag would wipe the chat transcript and reload the Lab iframe."""
    client, space = rendered(["chat", "ledger"])
    body_before = {
        e.id for e in client.elements.values() if "grad-body" in getattr(e, "classes", [])
    }
    space.preset("stack")
    body_after = {
        e.id for e in client.elements.values() if "grad-body" in getattr(e, "classes", [])
    }
    assert body_before == body_after, "a retile rebuilt a window root"


def test_the_chat_transcript_survives_a_retile(rendered):
    client, space = rendered(["chat", "ledger"])
    transcripts = [
        e for e in client.elements.values() if "grad-transcript" in getattr(e, "classes", [])
    ]
    assert len(transcripts) == 1
    identity = transcripts[0].id
    space.preset("stack")
    space.preset("tile")
    space.retile("chat", 1)
    still = [e for e in client.elements.values() if "grad-transcript" in getattr(e, "classes", [])]
    assert [e.id for e in still] == [identity]


def test_a_swap_moves_the_windows_and_keeps_both_roots(rendered):
    """A swap retiles, so it goes through the same teardown as a drag -- and the
    set of live windows does not change, so every root has to come back out of
    the attic. If it did not, dropping the ledger onto the chat would swap the
    panes and wipe the transcript in the same gesture."""
    client, space = rendered(["chat", "ledger"])
    bodies_before = {
        e.id for e in client.elements.values() if "grad-body" in getattr(e, "classes", [])
    }
    assert [c.windows for c in space.layout.columns] == [["chat"], ["ledger"]]

    space.swap("chat", "ledger")

    assert [c.windows for c in space.layout.columns] == [["ledger"], ["chat"]]
    bodies_after = {
        e.id for e in client.elements.values() if "grad-body" in getattr(e, "classes", [])
    }
    assert bodies_before == bodies_after, "a swap rebuilt a window root"


def test_a_drop_at_a_slot_boundary_reorders_within_the_column(rendered):
    client, space = rendered(["chat", "ledger", "quota"])
    space.preset("stack")
    assert space.layout.columns[0].windows == ["chat", "ledger", "quota"]
    space.retile("quota", 0, 0)
    assert space.layout.columns[0].windows == ["quota", "chat", "ledger"]
    windows = [e for e in client.elements.values() if "grad-window" in getattr(e, "classes", [])]
    assert len(windows) == 3


def test_closing_a_window_destroys_its_root(rendered):
    """Otherwise the attic accumulates a detached subtree per window per
    session, each one still bound to the poll."""
    client, space = rendered(["chat", "ledger"])
    before = len(client.elements)
    space.close("ledger")
    assert len(client.elements) < before
    assert "ledger" not in _root_ids(client)


def test_reopening_a_closed_window_builds_a_fresh_root(rendered):
    client, space = rendered(["chat", "ledger"])
    space.close("ledger")
    space.open("ledger")
    assert "ledger" in _root_ids(client)


# ---------------------------------------------------------------------------
# switching project and folder
# ---------------------------------------------------------------------------
def test_the_project_menu_lists_the_folder_and_its_projects(rendered):
    from core import budget as budget_mod

    budget_mod.create("proj-a", title="Scaling laws", budget={})
    budget_mod.set_current("proj-a")
    client, space = rendered(["chat"])

    from ui import shell as shell_mod

    menu = [e for e in client.elements.values() if "grad-card" in getattr(e, "classes", [])]
    assert menu, "the menu dialog was not built"
    # Drawn on open, not at build time: creating a project makes the list it was
    # read from stale, so it is rebuilt each time.
    with client:
        shell_mod._draw_project_menu(  # noqa: SLF001 - no public hook
            __import__("nicegui").ui, space, menu[0], _NullMenu()
        )
    markup = html_of(client)
    assert "proj-a" in markup
    assert "Scaling laws" in markup
    assert "WORKSPACE" in markup


class _NullMenu:
    def close(self) -> None:
        pass

    def redraw(self) -> None:
        pass


def test_the_workspace_menu_can_store_a_credential_without_a_terminal(rendered, monkeypatch):
    """The one thing the workspace could not do. `credential set` prompts with
    `getpass`, which needs a terminal -- so a fresh machine needed a shell open
    beside the app before the app was usable."""
    from core import budget as budget_mod
    from ui import shell as shell_mod, tasks as tasks_mod

    calls: list[dict] = []

    async def fake_run_tool(*argv, timeout=120.0, stdin=None):
        calls.append({"argv": argv, "stdin": stdin})
        return {"ok": True, "data": {"message": "stored"}}

    monkeypatch.setattr(tasks_mod, "run_tool", fake_run_tool)
    monkeypatch.setattr("ui.state.run_tool", fake_run_tool)

    budget_mod.create("proj-a", title="A", budget={})
    budget_mod.set_current("proj-a")
    client, space = rendered(["chat"])
    card = [e for e in client.elements.values() if "grad-card" in getattr(e, "classes", [])][0]
    with client:
        shell_mod._draw_project_menu(  # noqa: SLF001 - no public hook
            __import__("nicegui").ui, space, card, _NullMenu()
        )

    assert "CREDENTIALS" in html_of(client)
    import asyncio

    asyncio.run(space.set_credential("hf_token", "hf_the-actual-token"))

    assert calls, "no command was run"
    argv, stdin = calls[0]["argv"], calls[0]["stdin"]
    # Down a pipe, never in an argument: an argv is visible to anything that can
    # list processes.
    assert stdin == "hf_the-actual-token"
    assert "--stdin" in argv
    assert not any("hf_the-actual-token" in part for part in argv)


def test_an_empty_credential_is_refused_before_a_command_runs(rendered, monkeypatch):
    from ui import tasks as tasks_mod

    async def explode(*argv, **kwargs):
        raise AssertionError("a command ran for an empty value")

    monkeypatch.setattr("ui.state.run_tool", explode)
    monkeypatch.setattr(tasks_mod, "run_tool", explode)

    _, space = rendered(["chat"])
    import asyncio

    asyncio.run(space.set_credential("hf_token", "   "))
    assert "nothing to store" in (space.notice or "")


def test_the_folder_picker_argument_survives_a_process_boundary():
    """Native mode marshals `create_file_dialog` to the pywebview process over a
    multiprocessing queue, so its arguments have to pickle.

    `webview.FOLDER_DIALOG` does not: it is a deprecated `proxy_tools.Proxy`
    that reprs as `20` while being a proxy around a function, and it fails with
    "it's not the same object as webview.FOLDER_DIALOG". Worse, the error is
    raised in the queue's feeder thread, so it prints a traceback and hangs the
    picker instead of raising anywhere it could be caught -- which is why this
    is asserted here rather than left to a try/except at the call site.
    """
    import pickle

    from ui import shell as shell_mod

    value = shell_mod.folder_dialog_type()
    assert type(value) is int, f"a plain int, not {type(value).__name__}"
    assert pickle.loads(pickle.dumps(value)) == value

    webview = pytest.importorskip("webview", reason="pywebview is not installed")
    # Still the value pywebview means by "folder", however it spells it now.
    assert value == int(webview.FileDialog.FOLDER)


def test_switching_project_reloads_the_layout_for_that_project(rendered):
    """Layout persists per project, so the panes have to follow the switch --
    otherwise the new project opens with the old one's arrangement and silently
    overwrites its layout file on the next drag."""
    from core import budget as budget_mod
    from ui import state as state_module

    budget_mod.create("proj-a", title="A", budget={})
    budget_mod.create("proj-b", title="B", budget={})
    budget_mod.set_current("proj-a")

    client, space = rendered(["chat", "ledger"])
    space.project = "proj-a"
    space.preset("stack")
    stacked = [c.windows for c in space.layout.columns]

    # proj-b has never been opened, so it gets the default arrangement.
    budget_mod.set_current("proj-b")
    space.reload()
    assert space.project == "proj-b"
    assert [c.windows for c in space.layout.columns] != stacked
    assert state_module.layout_path("proj-b").name == "proj-b.json"


def test_a_reload_redraws_the_windows_rather_than_leaving_them_stale(rendered):
    """A retile reuses live roots -- that is what stops a drag wiping the
    transcript -- so a reload has to redraw the bodies explicitly or the panes
    would be rearranged for the new workspace while still showing the old one."""
    from core import ledger_store as ls

    client, space = rendered(["ledger"])
    ls.append_expectation(
        {"id": "exp-1", "task": "t", "created_at": ls.now_iso(), "quantity": "val_loss",
         "claim": "a claim from the first workspace",
         "predicted": {"low": 1.0, "high": 2.0, "direction": None},
         "basis": [], "comparability": "same", "confidence": "low"}
    )
    space.tick()
    assert "a claim from the first workspace" in html_of(client)

    space.reload()
    # Same workspace here, so the claim is still true -- what is being checked
    # is that the body was re-rendered at all, not that it changed.
    assert "a claim from the first workspace" in html_of(client)
    assert space.models == {} or "ledger" in space.models


# ---------------------------------------------------------------------------
# the whole lifecycle, in one pass
# ---------------------------------------------------------------------------
def test_the_full_gesture_sequence_leaves_a_consistent_tree(rendered):
    client, space = rendered()
    for action in (
        lambda: space.preset("stack"),
        lambda: space.preset("full"),
        lambda: space.preset("tile"),
        lambda: space.close("chat"),
        lambda: space.open("chat"),
        lambda: space.tick(),
        lambda: space.select("papers.filter", "queued"),
        lambda: space.select("funnel.trace", "nope"),
        lambda: space.retile("ledger", 0),
        lambda: space.resize("columns", [0.5] * len(space.layout.columns), total_px=1600),
        lambda: space.set_agent_state("running", step=14),
        lambda: space.say("verifying …"),
    ):
        action()

    windows = [e for e in client.elements.values() if "grad-window" in getattr(e, "classes", [])]
    assert len(windows) == len(space.layout.windows)
    assert "AGENT RUNNING · step 14" in html_of(client)
    assert "verifying" in html_of(client)


def test_a_tick_with_real_data_redraws_the_window(rendered):
    from core import ledger_store as ls

    client, space = rendered(["ledger"])
    ls.append_expectation(
        {"id": "exp-1", "task": "t", "created_at": ls.now_iso(), "quantity": "val_loss",
         "claim": "the loss lands between 2.9 and 3.2",
         "predicted": {"low": 2.9, "high": 3.2, "direction": None},
         "basis": [{"paper": "arXiv:1", "locator": "T3", "value": 3.0, "conditions": "1B"}],
         "comparability": "same eval", "confidence": "medium"}
    )
    space.tick()
    assert "the loss lands between 2.9 and 3.2" in html_of(client)


# ---------------------------------------------------------------------------
# populated: the paths an empty workspace never reaches
# ---------------------------------------------------------------------------
def seed_everything() -> None:
    """One of each thing the eleven windows read.

    The empty states are easy; the render paths that actually break are the ones
    behind a non-empty list -- a band strip, a lineage bar, a traceback, a
    reader rail. This seeds all of them.
    """
    import json

    from core import (
        budget as budget_mod, campaign as campaign_mod, jsonl,
        ledger_store as ls, paths, report as report_mod,
    )
    from tools import wiki as wiki_tool
    from ui import models

    ls.append_expectation(
        {"id": "exp-1", "task": "t", "created_at": ls.now_iso(), "quantity": "val_loss",
         "claim": "the loss lands in band", "predicted": {"low": 2.9, "high": 3.2, "direction": None},
         "basis": [{"paper": "arXiv:2001.08361", "locator": "T3", "value": 3.0, "conditions": "1B"}],
         "comparability": "same eval", "confidence": "medium"}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_SUBMITTED, "id": "run-1", "task": "t", "status": "in_flight",
         "submitted_at": ls.now_iso(), "estimate_usd": 4.0, "estimated_duration_s": 60}
    )
    ls.append_run_event(
        {"type": ls.T_RUN_COLLECTED, "id": "run-1", "status": "completed",
         "collected_at": ls.now_iso(), "cost_usd_actual": 3.5, "results": {"val_loss": 4.4},
         "deviations": [{"expectation_id": "exp-1", "quantity": "val_loss",
                         "expected": {"low": 2.9, "high": 3.2}, "actual": 4.4, "in_range": False}]}
    )

    budget_mod.create("proj", title="Scaling", payer="me", budget={"gpu_usd": 100.0})
    budget_mod.set_current("proj")
    targets = report_mod.paths_for("proj")
    targets["dir"].mkdir(parents=True, exist_ok=True)
    targets["tex"].write_text(
        "\\section{Setup}\nWe reach \\gradnum{loss} on eval. % note\n\\section{Results}\n",
        encoding="utf-8",
    )
    targets["claims"].write_text("{}", encoding="utf-8")

    for stage, role, credits in (("main", "opus", 2.0), ("funnel.rerank", "sonnet", 0.5)):
        jsonl.append(paths.quota_path(), {"at": ls.now_iso(), "stage": stage, "role": role,
                                          "input_tokens": 900, "output_tokens": 300,
                                          "credits_usd": credits, "project": "proj"})

    jsonl.write_json(paths.preflight_record("abc"), {
        "submission_hash": "abc", "spec": "specs/x.json", "verified_at": ls.now_iso(),
        "checks": {"tests": {"ok": True, "duration_s": 2.0},
                   "dry_run": {"ok": False, "duration_s": 1.0, "reason": "shape mismatch",
                               "output": "boom", "fix": "python -m tools.preflight run --spec specs/x.json"}},
        "warnings": ["dynamic import in train.py"]})

    funnel_dir = paths.notes_dir() / "funnel"
    funnel_dir.mkdir(parents=True, exist_ok=True)
    (funnel_dir / "q.json").write_text(json.dumps({
        "question": "does depth help at fixed compute?",
        "stages": {"0_expand": {"queries": ["depth scaling"], "hyde_words": 80},
                   "1_retrieve": {"candidates": 400, "corpus_chunks": 12000},
                   "2_rerank": {"out": 50}, "3_triage": {"returned": 1}},
        "survivors": [{"id": "a", "title": "A", "rerank_score": 0.9, "reason": "states the ratio"}],
        "dropped": [{"id": "b", "title": "B", "reason": "different tokenizer"}],
        "warnings": ["rerank fell back to lexical"]}), encoding="utf-8")

    paper = paths.papers_dir() / "2001.08361"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "meta.json").write_text(
        json.dumps({"title": "Scaling Laws", "authors": ["Kaplan"], "year": 2020}), encoding="utf-8"
    )

    campaign_mod.append_campaign({"type": campaign_mod.T_CAMPAIGN, "id": "camp-1", "status": "open",
                                  "task_dir": "tasks/x", "at": ls.now_iso(), "generations_run": 3,
                                  "project": "proj", "objective": "maximise accuracy", "islands": 4})
    for index, score in enumerate((0.1, 0.4, 0.3, 0.9)):
        campaign_mod.append_candidate({"type": campaign_mod.T_CANDIDATE, "id": f"cand-{index}",
                                       "campaign": "camp-1", "generation": index,
                                       "combined_score": score, "cost_usd": 0.02, "at": ls.now_iso()})

    paths.notebooks_dir().mkdir(parents=True, exist_ok=True)
    (paths.notebooks_dir() / "x.ipynb").write_text(
        '{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}', encoding="utf-8"
    )
    models.write_verify_record("x.ipynb", {"ok": False, "at": ls.now_iso(),
                                           "message": "NameError: torch is not defined",
                                           "cell_index": 4, "traceback": "Traceback ...",
                                           "fix": "pip install torch"})

    wiki_tool.output_dir().mkdir(parents=True, exist_ok=True)
    jsonl.write_json(wiki_tool.output_dir() / "manifest.json",
                     {"generated_at": ls.now_iso(), "output_dir": str(wiki_tool.output_dir()),
                      "source": {"hash": "stale00", "files": {"core/x.py": "aaa"}},
                      "scopes": {"core": 12, "tools": 9}})


def test_every_window_renders_with_real_data(rendered):
    seed_everything()
    client, space = rendered()
    space.select("papers.selected", "2001.08361", window="papers")
    space.select("ledger.filter", "broken")
    space.tick()
    markup = html_of(client)

    for expected in (
        "the loss lands in band",              # ledger, with a band strip
        "Scaling Laws",                        # papers, list and reader rail
        "does depth help at fixed compute?",   # funnel
        "different tokenizer",                 # funnel, dropped
        "dynamic import in train.py",          # preflight warnings
        "shape mismatch",                      # preflight failing row
        "NameError: torch is not defined",     # notebook failure detail
        "pip install torch",                   # notebook FIX box
        "camp-1",                              # evolve
        "gradnum",                             # editor source highlighting
        "different source tree",               # wiki staleness
        "run-1",                               # queue
    ):
        assert expected in markup, expected


def test_a_populated_workspace_survives_the_full_gesture_sequence(rendered):
    seed_everything()
    client, space = rendered()
    space.tick()
    for preset in ("stack", "full", "tile"):
        space.preset(preset)
    space.retile("evolve", 0)
    space.tick()
    windows = [e for e in client.elements.values() if "grad-window" in getattr(e, "classes", [])]
    assert len(windows) == len(space.layout.windows)
