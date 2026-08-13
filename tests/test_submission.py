"""The resolved-submission hash (HANDOFF §6).

The claims under test are the ones the whole gate rests on: the hash notices
what can change the outcome of a job, and ignores what cannot.
"""

from __future__ import annotations

import json

import pytest

from core.errors import ConfigError
from core.submission import Submission, import_graph


def _pipeline(root, *, extra_spec: str = "") -> object:
    d = root / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("import helper\nprint('train')\n", encoding="utf-8")
    (d / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (d / "requirements.lock").write_text("torch==2.4.0\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\n"
        "image = 'org/img@sha256:aaaa'\n"
        "lockfile = 'requirements.lock'\n"
        "argv = ['--config', 'base']\n" + extra_spec +
        "[dataset]\nname = 'org/ds'\nrevision = 'abc123'\n"
        "[config]\nlr = 0.001\n"
        "[estimate]\nhours = 2.0\nrate_usd_per_hour = 1.5\n",
        encoding="utf-8",
    )
    return d


def test_hash_is_stable_across_calls(workspace):
    d = _pipeline(workspace)
    a = Submission.load(d / "spec.toml", resolve_digest=False)
    b = Submission.load(d / "spec.toml", resolve_digest=False)
    assert a.hash() == b.hash()


def test_config_override_changes_the_hash(workspace):
    """The most common real change is a config edit with identical code."""
    d = _pipeline(workspace)
    base = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    overridden = Submission.load(d / "spec.toml", overrides={"lr": 0.01}, resolve_digest=False).hash()
    assert base != overridden


def test_imported_module_change_changes_the_hash(workspace):
    d = _pipeline(workspace)
    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    (d / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    assert before != after


def test_unrelated_file_does_not_change_the_hash(workspace):
    """A directory hash is too broad: touching a note or writing a figure would
    invalidate a perfectly valid preflight, and a gate that fires spuriously is
    a gate that gets argued around."""
    d = _pipeline(workspace)
    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    (d / "notes.md").write_text("scratch thoughts\n", encoding="utf-8")
    (d / "figure.png").write_bytes(b"\x89PNG")
    after = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    assert before == after


def test_lockfile_change_changes_the_hash(workspace):
    d = _pipeline(workspace)
    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    (d / "requirements.lock").write_text("torch==2.5.0\n", encoding="utf-8")
    assert Submission.load(d / "spec.toml", resolve_digest=False).hash() != before


def test_dataset_revision_change_changes_the_hash(workspace):
    d = _pipeline(workspace)
    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    spec = (d / "spec.toml").read_text(encoding="utf-8").replace("abc123", "def456")
    (d / "spec.toml").write_text(spec, encoding="utf-8")
    assert Submission.load(d / "spec.toml", resolve_digest=False).hash() != before


def test_extra_hash_paths_are_covered(workspace):
    """Runtime-loaded files are reached by neither the import graph nor the
    resolved config. They are a documented gap, not a silent guarantee."""
    d = _pipeline(workspace, extra_spec="extra_hash_paths = ['tokenizer.json']\n")
    (d / "tokenizer.json").write_text('{"vocab": 1}', encoding="utf-8")
    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    (d / "tokenizer.json").write_text('{"vocab": 2}', encoding="utf-8")
    assert Submission.load(d / "spec.toml", resolve_digest=False).hash() != before


def test_untagged_image_is_refused(workspace, monkeypatch):
    """':latest is how remote environment drift sneaks past a hash that
    otherwise looks airtight.'

    The resolver is stubbed out: left alone it shells out to `docker manifest
    inspect`, which on a machine that has docker contacts a registry and can
    add two minutes to a suite that is meant to need no network.
    """
    def _no_docker(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("core.submission.subprocess.run", _no_docker)
    d = _pipeline(workspace)
    spec = (d / "spec.toml").read_text(encoding="utf-8").replace(
        "'org/img@sha256:aaaa'", "'org/img:latest'"
    )
    (d / "spec.toml").write_text(spec, encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        Submission.load(d / "spec.toml", resolve_digest=True)
    assert "digest" in str(exc.value)


def test_dynamic_import_is_reported_not_ignored(workspace):
    d = _pipeline(workspace)
    (d / "train.py").write_text(
        "import importlib\nmod = importlib.import_module('helper')\n", encoding="utf-8"
    )
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    assert any("dynamic import" in w for w in sub.warnings)


def test_from_package_import_module_is_hashed(workspace):
    """`from pkg import mod` names a module, not just an attribute.

    Resolving only `node.module` stops at `pkg/__init__.py`, leaving `pkg/mod.py`
    outside the hash -- so editing it would not invalidate the preflight record,
    which is exactly the gap the import graph exists to close.
    """
    d = _pipeline(workspace)
    pkg = d / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "inner.py").write_text("RATE = 1\n", encoding="utf-8")
    (d / "train.py").write_text("from pkg import inner\n", encoding="utf-8")

    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    (pkg / "inner.py").write_text("RATE = 2\n", encoding="utf-8")
    assert Submission.load(d / "spec.toml", resolve_digest=False).hash() != before


def test_relative_from_import_is_hashed(workspace):
    d = _pipeline(workspace)
    (d / "sibling.py").write_text("VALUE = 1\n", encoding="utf-8")
    (d / "train.py").write_text("from . import sibling\n", encoding="utf-8")

    before = Submission.load(d / "spec.toml", resolve_digest=False).hash()
    (d / "sibling.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert Submission.load(d / "spec.toml", resolve_digest=False).hash() != before


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("org/name:2026-08", "org/name"),
        ("registry.local:5000/org/name:2026-08", "registry.local:5000/org/name"),
        ("registry.local:5000/org/name", "registry.local:5000/org/name"),
        ("org/name", "org/name"),
    ],
)
def test_tag_stripping_leaves_a_registry_port_alone(image, expected):
    """Cutting at the first colon turns `registry.local:5000/org/name:tag` into
    `registry.local`, putting a wrong image in the hash and the submission."""
    from core.submission import _strip_tag

    assert _strip_tag(image) == expected


def test_import_graph_ignores_third_party(workspace):
    d = _pipeline(workspace)
    (d / "train.py").write_text("import torch\nimport helper\n", encoding="utf-8")
    files, _ = import_graph(d / "train.py", [d])
    names = {f.name for f in files}
    assert names == {"train.py", "helper.py"}


def test_missing_dataset_revision_warns(workspace):
    d = _pipeline(workspace)
    spec = (d / "spec.toml").read_text(encoding="utf-8").replace("revision = 'abc123'\n", "")
    (d / "spec.toml").write_text(spec, encoding="utf-8")
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    assert any("revision" in w for w in sub.warnings)


def test_resolved_document_is_json_serialisable(workspace):
    d = _pipeline(workspace)
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    json.dumps(sub.resolved())
    assert sub.estimated_cost_usd() == pytest.approx(3.0)
