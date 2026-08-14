"""Models by role (HANDOFF-2 §16).

Cheap tests for a cheap change, but the resolution order carries a real
promise -- "`core/config.py` keeps the old `[retrieval] triage_model` /
`expand_model` keys readable as overrides for one release so existing configs do
not break" -- and a promise nothing checks is a promise that breaks silently on
the next edit.
"""

from __future__ import annotations

import pytest

from core import config as config_mod
from core.errors import ConfigError


def write_config(workspace, text: str):
    path = workspace / "config" / "grad.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    config_mod._cache.clear()
    return config_mod.load(path, reload=True)


def test_defaults_are_the_claude_5_family(workspace):
    cfg = write_config(workspace, "")
    assert cfg.model_for("research") == "claude-opus-5"
    assert cfg.model_for("evolve") == "claude-sonnet-5"
    assert cfg.model_for("report") == "claude-opus-5"
    # There is no Haiku 5; 4.5 is the latest.
    assert cfg.model_for("expand") == "claude-haiku-4-5"
    assert cfg.model_for("triage") == "claude-haiku-4-5"
    assert cfg.model_for("cite") == "claude-haiku-4-5"


def test_the_opus_4_5_default_is_gone(workspace):
    cfg = write_config(workspace, "")
    assert "claude-opus-4-5" not in cfg.models().values()


def test_explicit_models_entry_wins(workspace):
    cfg = write_config(workspace, '[models]\nresearch = "claude-sonnet-5"\n')
    assert cfg.model_for("research") == "claude-sonnet-5"


def test_legacy_keys_still_resolve_for_one_release(workspace):
    """An existing config must not break."""
    cfg = write_config(
        workspace,
        '[agent]\nmodel = "claude-opus-4-5"\n'
        '[retrieval]\ntriage_model = "old-haiku"\nexpand_model = "older-haiku"\n',
    )
    assert cfg.model_for("research") == "claude-opus-4-5"
    assert cfg.model_for("triage") == "old-haiku"
    assert cfg.model_for("expand") == "older-haiku"


def test_an_explicit_models_entry_beats_a_legacy_key(workspace):
    cfg = write_config(
        workspace,
        '[models]\nresearch = "claude-opus-5"\n[agent]\nmodel = "claude-opus-4-5"\n',
    )
    assert cfg.model_for("research") == "claude-opus-5"


def test_unknown_role_in_config_is_a_config_error(workspace):
    """A silently ignored setting is worse than a refusal: the model it names is
    never used and nothing says so."""
    with pytest.raises(ConfigError) as exc:
        write_config(workspace, '[models]\nrerank = "voyageai/rerank-2.5"\n')
    assert "not a model role" in str(exc.value)


def test_non_string_model_is_a_config_error(workspace):
    with pytest.raises(ConfigError):
        write_config(workspace, "[models]\nresearch = 5\n")


def test_unknown_role_lookup_raises(workspace):
    cfg = write_config(workspace, "")
    with pytest.raises(ConfigError):
        cfg.model_for("nonexistent")


def test_rerank_and_embed_stay_in_retrieval(workspace):
    """§16 is explicit that these must not move: a different provider on a
    different billing rail, and folding them in invites swapping Voyage for
    Haiku -- which is worse at the task and moves load onto the scarcer
    resource."""
    cfg = write_config(workspace, "")
    assert cfg.get("retrieval", "rerank_model") == "voyageai/rerank-2.5"
    assert cfg.get("retrieval", "embed_model") == "voyage-4"
    assert "rerank" not in config_mod.MODEL_ROLES
    assert "embed" not in config_mod.MODEL_ROLES
