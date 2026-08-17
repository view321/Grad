"""The writable overlay, and the file it is not allowed to touch.

Every test here is about one of two claims. The first is that a value chosen
through setup wins over the same value in `config/grad.toml` -- because a
command that silently did nothing when a config file disagreed would be worse
than one that overrides it. The second is that overriding it never *edits* it:
that file is hand-annotated, `tomllib` cannot write TOML, and the comments in it
are worth more than the values.
"""

from __future__ import annotations

import pytest

from core import config as config_mod, paths, settings
from core.errors import UsageError

#: A config with a comment on every line that matters, so "the file survived"
#: is a claim about the annotations and not only about the values.
ANNOTATED = """\
# The main loop's model. Chosen deliberately -- see HANDOFF-2 §16.
[models]
research = "claude-opus-5"   # the expensive one, on purpose
evolve   = "claude-sonnet-5"

[hosts.from-config]
hostname = "config-box.example"
user = "researcher"
rate_usd_per_hour = 2.5
"""


@pytest.fixture
def annotated_config():
    """Write the config the workspace fixture points `GRAD_CONFIG` at."""
    path = paths.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ANNOTATED, encoding="utf-8")
    config_mod._cache.clear()
    return path


# ---------------------------------------------------------------------------
# the layers, one test per layer
# ---------------------------------------------------------------------------
def test_a_role_nobody_set_resolves_to_the_shipped_default(workspace):
    assert config_mod.load(reload=True).model_for("cite") == config_mod.DEFAULTS["models"]["cite"]


def test_the_config_beats_the_default(workspace, annotated_config):
    assert config_mod.load(reload=True).model_for("evolve") == "claude-sonnet-5"


def test_the_overlay_beats_the_config(workspace, annotated_config):
    """`kaggle account`'s rule, generalised: a stored selection wins over the
    config key it shadows, because a user who has just answered a wizard is
    entitled to expect the answer to take effect."""
    settings.set_models({"evolve": "claude-opus-5"})
    assert config_mod.load(reload=True).model_for("evolve") == "claude-opus-5"


def test_clearing_an_override_falls_back_through_the_layers_again(workspace, annotated_config):
    settings.set_models({"evolve": "claude-opus-5"})
    settings.clear_models(["evolve"])
    cfg = config_mod.load(reload=True)
    assert cfg.model_for("evolve") == "claude-sonnet-5"      # the config, again
    assert cfg.model_for("cite") == config_mod.DEFAULTS["models"]["cite"]


def test_an_overlay_write_is_seen_by_a_process_that_already_loaded(workspace):
    """The overlay's mtime is part of the cache key. Without it, `setup models`
    -- which the UI runs as a child process -- writes a file the app has already
    read, and the setup window appears to do nothing until a restart."""
    before = config_mod.load().model_for("evolve")
    settings.set_models({"evolve": "claude-fable-5"})
    assert config_mod.load().model_for("evolve") == "claude-fable-5"
    assert before != "claude-fable-5"


# ---------------------------------------------------------------------------
# the project layer, which sits above all of them
# ---------------------------------------------------------------------------
def test_a_project_override_beats_the_workspace_overlay(workspace, annotated_config):
    """The model per role is the main lever on cost and quality, which is exactly
    why a cheap exploratory project and one being written up should be able to
    differ."""
    from core import budget

    settings.set_models({"evolve": "claude-sonnet-5"})
    budget.create("proj-a", title="A", budget={})
    budget.set_current("proj-a")
    budget.configure("proj-a", models={"evolve": "claude-opus-5"}, reason="writing it up")

    assert config_mod.load(reload=True).model_for("evolve") == "claude-opus-5"


def test_a_project_overriding_one_role_leaves_the_others_alone(workspace, annotated_config):
    """An override, not a replacement. "Opus for the write-up, everything else
    as usual" has to be expressible."""
    from core import budget

    budget.create("proj-a", title="A", budget={})
    budget.set_current("proj-a")
    budget.configure("proj-a", models={"report": "claude-opus-5"})

    cfg = config_mod.load(reload=True)
    assert cfg.model_for("report") == "claude-opus-5"
    assert cfg.model_for("evolve") == "claude-sonnet-5"   # still the config's
    assert cfg.model_for("cite") == config_mod.DEFAULTS["models"]["cite"]


def test_the_project_layer_can_be_asked_to_stand_aside(workspace, annotated_config):
    """The projects window draws every project and only one is selected, so for
    all the others the project layer in effect belongs to somebody else."""
    from core import budget

    budget.create("proj-a", title="A", budget={})
    budget.set_current("proj-a")
    budget.configure("proj-a", models={"evolve": "claude-opus-5"})

    cfg = config_mod.load(reload=True)
    assert cfg.model_for("evolve") == "claude-opus-5"
    assert cfg.model_for("evolve", project=False) == "claude-sonnet-5"


def test_switching_project_changes_which_model_answers(workspace, annotated_config):
    """Without the selection in the cache key, the app goes on serving whatever
    it loaded at startup."""
    from core import budget

    budget.create("proj-cheap", title="cheap", budget={})
    budget.create("proj-careful", title="careful", budget={})
    budget.configure("proj-cheap", models={"research": "claude-haiku-4-5"})
    budget.configure("proj-careful", models={"research": "claude-opus-5"})

    budget.set_current("proj-cheap")
    assert config_mod.load().model_for("research") == "claude-haiku-4-5"
    budget.set_current("proj-careful")
    assert config_mod.load().model_for("research") == "claude-opus-5"


def test_clearing_a_project_override_falls_back_to_the_workspace(workspace, annotated_config):
    from core import budget

    budget.create("proj-a", title="A", budget={})
    budget.set_current("proj-a")
    budget.configure("proj-a", models={"evolve": "claude-opus-5"})
    budget.configure("proj-a", models={"evolve": None})

    assert config_mod.load(reload=True).model_for("evolve") == "claude-sonnet-5"
    assert budget.project_overrides("proj-a")["models"] == {}


def test_every_configure_is_kept_not_just_the_last(workspace):
    """The model a candidate was mutated by is part of what produced the numbers
    in the ledger beside it, so a field that could be edited in place would let a
    project's history claim it had always been on today's model."""
    from core import budget

    budget.create("proj-a", title="A", budget={})
    budget.configure("proj-a", models={"evolve": "claude-sonnet-5"}, reason="first pass")
    budget.configure("proj-a", models={"evolve": "claude-opus-5"}, reason="it was missing things")

    history = budget.projects()["proj-a"]["configured"]
    assert [h["reason"] for h in history] == ["first pass", "it was missing things"]
    assert budget.projects()["proj-a"]["models"]["evolve"] == "claude-opus-5"


def test_configure_refuses_a_role_and_a_backend_it_does_not_know(workspace):
    from core import budget

    budget.create("proj-a", title="A", budget={})
    with pytest.raises(UsageError):
        budget.configure("proj-a", models={"summarise": "claude-opus-5"})
    with pytest.raises(UsageError):
        budget.configure("proj-a", backend="slurm")
    with pytest.raises(UsageError):
        budget.configure("proj-a")           # nothing to change
    assert budget.projects()["proj-a"]["configured"] == []


def test_configure_refuses_a_project_that_does_not_exist(workspace):
    from core import budget
    from core.errors import NotFound

    with pytest.raises(NotFound):
        budget.configure("proj-nope", models={"evolve": "claude-opus-5"})


def test_overrides_for_an_unreadable_ledger_are_empty_not_an_exception(workspace, monkeypatch):
    """`project_overrides` is read on the config path, which every surface calls
    while rendering."""
    from core import budget

    monkeypatch.setattr(
        budget, "projects", lambda: (_ for _ in ()).throw(RuntimeError("torn ledger"))
    )
    assert budget.project_overrides("proj-a") == {"models": {}, "backend": None}
    assert budget.project_overrides(None) == {"models": {}, "backend": None}


# ---------------------------------------------------------------------------
# the file that must not be written
# ---------------------------------------------------------------------------
def test_no_setup_command_edits_the_annotated_config(workspace, annotated_config):
    """The whole reason this module exists. `tomllib` reads TOML and cannot
    write it, so an editing command would reformat the file and drop every
    comment in it -- and the comments are the reasoning."""
    from tools import setup as setup_tool

    before = annotated_config.read_bytes()

    setup_tool.cli.run(["models", "--evolve", "claude-opus-5", "--json"])
    setup_tool.cli.run(["backend", "--default", "kaggle", "--json"])
    setup_tool.cli.run(
        ["host", "add", "--name", "gpu-box", "--hostname", "box.example", "--user", "me", "--json"]
    )
    setup_tool.cli.run(["host", "remove", "--name", "gpu-box", "--json"])
    setup_tool.cli.run(["models", "--clear", "evolve", "--json"])
    setup_tool.cli.run(["show", "--json"])
    setup_tool.cli.run(["check", "--json"])

    assert annotated_config.read_bytes() == before


def test_the_overlay_lives_beside_the_layouts_not_in_the_workspace(workspace):
    """Per workspace, because `grad.toml` already is. Out of the workspace
    folder, because a research folder handed to a colleague should not carry
    this machine's SSH inventory."""
    from core import appdata

    assert settings.path().parent == appdata.workspace_state_dir()
    assert paths.root() not in settings.path().parents


# ---------------------------------------------------------------------------
# what it refuses
# ---------------------------------------------------------------------------
def test_an_unknown_role_is_refused_with_the_roles_in_the_message(workspace):
    with pytest.raises(UsageError) as caught:
        settings.set_models({"summarise": "claude-opus-5"})
    assert "research" in caught.value.fix


def test_an_empty_model_is_refused_rather_than_stored(workspace):
    """Stored, it would resolve to falsy and silently fall through -- a setting
    that is present, wrong, and invisible."""
    with pytest.raises(UsageError):
        settings.set_models({"evolve": "   "})
    assert settings.models() == {}


def test_an_unknown_backend_is_refused(workspace):
    with pytest.raises(UsageError) as caught:
        settings.set_backend("slurm")
    assert "kaggle" in caught.value.fix


def test_a_negative_host_rate_is_refused_on_the_writable_side_too(workspace):
    """`collect` prices wall clock against this, so a negative rate books
    negative actuals -- which *reduce* rolling spend. A typo that raises the
    ceiling is worth refusing at the point of entry, on both sides of the
    inventory."""
    with pytest.raises(UsageError) as caught:
        settings.add_host("gpu-box", {"hostname": "box.example", "rate_usd_per_hour": -1.0})
    assert "negative spend" in caught.value.fix


def test_a_host_name_that_could_pass_for_an_ssh_flag_is_refused(workspace):
    with pytest.raises(UsageError):
        settings.add_host("-oProxyCommand=curl evil.example", {"hostname": "box.example"})


def test_a_host_with_no_hostname_is_refused(workspace):
    with pytest.raises(UsageError) as caught:
        settings.add_host("gpu-box", {"user": "me"})
    assert "--hostname" in caught.value.fix


# ---------------------------------------------------------------------------
# the inventory has two sources and stays fixed
# ---------------------------------------------------------------------------
def test_an_added_host_joins_the_inventory_the_config_declared(workspace, annotated_config):
    settings.add_host(
        "gpu-box", {"hostname": "box.example", "user": "me", "rate_usd_per_hour": 1.25}
    )
    hosts = config_mod.load(reload=True).hosts
    assert set(hosts) == {"from-config", "gpu-box"}
    assert hosts["gpu-box"].rate_usd_per_hour == 1.25
    assert hosts["from-config"].hostname == "config-box.example"


def test_an_overlay_host_replaces_the_config_one_rather_than_inheriting_it(
    workspace, annotated_config
):
    """A recursive merge would have the overlay host inherit the fields it
    omitted -- including `key_credential`, the keyring entry that authenticates
    the connection. Replacing `from-config` through the wizard and getting the
    old box's credential and user attached to the new hostname is a connection
    nobody described, and it is the one field here where being wrong reaches a
    machine."""
    settings.add_host("from-config", {"hostname": "new-box.example", "rate_usd_per_hour": 0.0})

    host = config_mod.load(reload=True).hosts["from-config"]
    assert host.hostname == "new-box.example"
    assert host.user == "", "the config host's user must not have carried over"
    assert host.key_credential is None
    assert host.rate_usd_per_hour == 0.0


def test_a_non_finite_host_rate_is_refused_on_the_writable_side_too(workspace):
    """`nan` fails every comparison a gate makes against it and `inf` is a price
    no run can be under, so both are ceilings that stop bounding anything.
    `core/config.py` refuses them on the TOML side; a check that exists in only
    one of two entry points is a check with a way around it."""
    for rate in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(UsageError) as caught:
            settings.add_host("gpu-box", {"hostname": "box.example", "rate_usd_per_hour": rate})
        assert "finite" in caught.value.message or "negative" in caught.value.message
    assert settings.hosts() == {}


def test_an_unknown_host_names_both_places_it_could_have_been_added(workspace):
    """The inventory is fixed by design -- a host that can be named ad-hoc is a
    general remote-execution capability. It now has two sources, so a refusal
    that named one of them sent half the readers to the wrong file."""
    from core.errors import ConfigError

    with pytest.raises(ConfigError) as caught:
        config_mod.load(reload=True).host("nowhere")
    assert "tools.setup host add" in caught.value.fix
    assert "grad.toml" in caught.value.fix


# ---------------------------------------------------------------------------
# the report that makes winning acceptable
# ---------------------------------------------------------------------------
def test_shadowing_names_the_config_value_that_stopped_taking_effect(workspace, annotated_config):
    """Someone edits `[models] evolve`, sees no change, and has no way to
    discover that a file they have never heard of outranks the file they were
    told to edit."""
    settings.set_models({"evolve": "claude-opus-5"})
    report = settings.shadowing(config_mod.load(reload=True))
    assert report == [
        {"what": "[models] evolve", "config": "claude-sonnet-5", "overlay": "claude-opus-5"}
    ]


def test_an_overlay_that_agrees_with_the_config_shadows_nothing(workspace, annotated_config):
    settings.set_models({"evolve": "claude-sonnet-5"})
    assert settings.shadowing(config_mod.load(reload=True)) == []


def test_a_role_the_config_never_set_shadows_nothing(workspace, annotated_config):
    settings.set_models({"cite": "claude-opus-5"})
    assert settings.shadowing(config_mod.load(reload=True)) == []


# ---------------------------------------------------------------------------
# one vocabulary for the three backends
# ---------------------------------------------------------------------------
def test_the_backend_names_are_the_ones_evolve_submits_to():
    """`core/settings.py` names them because a setting is read by the config
    layer and `core` importing `tools` is backwards. Two lists is one list that
    has not drifted yet."""
    from tools import evolve as evolve_tool

    assert set(settings.BACKENDS) == set(evolve_tool.REMOTE_BACKENDS)


# ---------------------------------------------------------------------------
# a broken overlay is not a broken app
# ---------------------------------------------------------------------------
def test_an_unparseable_overlay_leaves_the_app_on_the_config(workspace, annotated_config):
    """`grad.toml` is a working configuration. An overlay that will not parse
    should cost its own contents, not the ability to start."""
    settings.path().parent.mkdir(parents=True, exist_ok=True)
    settings.path().write_text("{ not json at all", encoding="utf-8")
    assert config_mod.load(reload=True).model_for("evolve") == "claude-sonnet-5"


def test_readiness_names_what_each_backend_is_missing(workspace, monkeypatch):
    """`hf_token` is required *for HF Jobs*, which is a different claim from
    "required" -- and the credentials panel used to make the stronger one at a
    user who had chosen Kaggle."""
    from core import credentials
    from tools import setup as setup_tool

    monkeypatch.setattr(credentials, "status", lambda: dict.fromkeys(credentials.ALL, False))
    rows = {r["backend"]: r for r in setup_tool.readiness(config_mod.load(reload=True))}
    assert rows["hf_jobs"]["missing"] == ["hf_token"]
    assert "an ssh host" in rows["ssh"]["missing"]
    assert not any(r["ready"] for r in rows.values())


# ---------------------------------------------------------------------------
# the agent's context budget
# ---------------------------------------------------------------------------
def test_the_context_budget_is_read_back_through_the_ordinary_config_path(workspace):
    """The overlay outranking the file is this module's whole premise, and it was
    true only for the three settings that had their own accessor. Every ordinary
    scalar goes through `Config.get`, which read `raw` alone -- so a setup window
    writing here would have written a value nothing ever read."""
    from core import compaction

    settings.set_agent({"compact_at_tokens": 120_000})
    cfg = config_mod.load(reload=True)
    assert cfg.get("agent", "compact_at_tokens") == 120_000
    assert compaction.threshold(cfg) == 120_000


def test_clearing_it_falls_back_through_the_layers(workspace):
    """Retiring by making optional, not by deleting: a key set here and then
    cleared resolves exactly as it did before anyone opened the wizard."""
    from core import compaction

    settings.set_agent({"compact_at_tokens": 120_000})
    settings.clear_agent(["compact_at_tokens"])
    assert settings.agent() == {}
    assert compaction.threshold(config_mod.load(reload=True)) == (
        config_mod.DEFAULTS["agent"]["compact_at_tokens"]
    )


def test_zero_is_off_rather_than_below_the_floor(workspace):
    """0 is the documented way to disable compaction -- the same spelling
    `compaction.threshold` already reads -- not a value under the minimum."""
    from core import compaction

    settings.set_agent({"compact_at_tokens": 0})
    assert compaction.threshold(config_mod.load(reload=True)) == 0


def test_a_threshold_small_enough_to_compact_every_turn_is_refused(workspace):
    """A compaction costs a turn and seeds a session with a cold prompt cache.
    Below the floor it would run continuously and cost more than it saves."""
    with pytest.raises(UsageError) as exc:
        settings.set_agent({"compact_at_tokens": 500})
    assert "0 to leave compaction" in (exc.value.fix or "")


def test_a_value_that_is_not_a_finite_number_is_refused(workspace):
    """NaN fails every range comparison, so a naive check would let it through
    -- as a threshold nothing is ever above, which reads as "compaction is
    broken" rather than as a rejected setting."""
    for bad in ("soon", float("nan"), float("inf")):
        with pytest.raises(UsageError):
            settings.set_agent({"compact_at_tokens": bad})
    assert settings.agent() == {}


def test_an_unknown_agent_setting_is_refused_rather_than_stored(workspace):
    """This overlay outranks the file for every reader, so a typo would be a
    setting that applies and is wrong -- not one that fails to apply."""
    with pytest.raises(UsageError):
        settings.set_agent({"compact_at_toknes": 100_000})


def test_it_reports_what_it_is_shadowing(workspace, annotated_config):
    """The report is the price of being allowed to win."""
    path = paths.config_path()
    path.write_text(ANNOTATED + '\n[agent]\ncompact_at_tokens = 250000\n', encoding="utf-8")
    config_mod._cache.clear()
    settings.set_agent({"compact_at_tokens": 120_000})

    reported = settings.shadowing(config_mod.load(reload=True))
    assert {"what": "[agent] compact_at_tokens", "config": "250000", "overlay": "120000"} in reported


def test_a_section_the_overlay_never_mentions_is_untouched(workspace, annotated_config):
    """Widening `Config.get` must not change how anything else resolves."""
    settings.set_agent({"compact_at_tokens": 120_000})
    cfg = config_mod.load(reload=True)
    assert cfg.get("spend", "monthly_usd") == config_mod.DEFAULTS["spend"]["monthly_usd"]
    assert cfg.model_for("evolve") == "claude-sonnet-5"
