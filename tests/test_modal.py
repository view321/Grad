"""The Modal backend: the parts that have rules, without a live sandbox.

What is under test is everything that decides *before* the SDK is reached --
which GPU, at what rate, for how long, and what the gates say -- plus the two
refusals that only exist on this backend: an accelerator with no price, and a
run that asks for longer than Modal will let a Sandbox live.

The SDK is not installed in this suite and is not stubbed at the module level
either. `_modal()` raising a ConfigError with a `pip install` in it *is* the
behaviour for a machine without the extra, and it is asserted rather than
mocked away.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import config as config_mod, credentials, settings, submit as submit_lib
from core.errors import ConfigError, GradError
from core.submission import Submission
from tools import modal as modal_tool


def spec(tmp_path: Path, **overrides) -> Path:
    """A minimal submittable spec, digest-pinned so nothing reaches a registry."""
    (tmp_path / "train.py").write_text("print('hi')\n", encoding="utf-8")
    document = {
        "entrypoint": "train.py",
        "image": "nvcr.io/nvidia/pytorch@sha256:" + "a" * 64,
        "metrics_file": "metrics.json",
        "target": {"platform": "modal"},
        "estimate": {"hours": 2.0, "cost_usd": 8.0},
    }
    # Replaced, not merged. Merging looks helpful and made
    # `estimate={"cost_usd": 1.0}` keep the default `hours`, so the test for "a
    # spec with no estimate" was asserting against a spec that had one.
    document.update(overrides)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def submission(tmp_path: Path, **overrides) -> Submission:
    return Submission.load(spec(tmp_path, **overrides), resolve_digest=False)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def test_the_backend_is_registered_everywhere_a_backend_is_listed(workspace):
    """Four lists name the backends and they drift independently. The one that
    actually bit: `modal` in `REMOTE_BACKENDS` without a matching branch in the
    candidate dispatcher ran the whole campaign on Hugging Face Jobs."""
    from tools import evolve as evolve_tool

    assert "modal" in settings.BACKENDS
    assert "modal" in evolve_tool.REMOTE_BACKENDS
    assert submit_lib.COLLECTORS["modal"] == "python -m tools.modal collect"


def test_the_collect_command_a_refusal_names_is_this_one(workspace):
    """`submit_lib.collect_command` is what a stale-run refusal prints, and an
    unknown platform degrades to the ledger rather than to a wrong instruction."""
    from core import ledger_store as ls

    run = ls.Run("run-1", {"id": "run-1", "platform": "modal"})
    assert submit_lib.collect_command(run) == "python -m tools.modal collect run-1 --json"


def test_both_halves_of_the_credential_are_registered(workspace):
    """Both are secret, unlike Kaggle where the username is a name. A credential
    missing from `ALL` is one `scrub_environment` does not remove."""
    assert credentials.MODAL_TOKEN_ID in credentials.ALL
    assert credentials.MODAL_TOKEN_SECRET in credentials.ALL


# ---------------------------------------------------------------------------
# hardware and money
# ---------------------------------------------------------------------------
def test_the_gpu_resolves_flag_then_spec_then_config(workspace, tmp_path):
    cfg = config_mod.load()
    sub = submission(tmp_path, target={"gpu": "A100-80GB"})

    assert modal_tool.resolve_gpu("H200", sub, cfg) == "H200"
    assert modal_tool.resolve_gpu(None, sub, cfg) == "A100-80GB"
    assert modal_tool.resolve_gpu(None, submission(tmp_path), cfg) == "H100"


def test_a_count_suffix_multiplies_the_rate(workspace):
    """Modal spells eight H100s `H100:8`, and eight cards cost eight times as
    much. A table with a row per count would go stale one row at a time."""
    cfg = config_mod.load()
    single = modal_tool.gpu_rate("H100", cfg)
    assert single == pytest.approx(3.9492)
    assert modal_tool.gpu_rate("H100:8", cfg) == pytest.approx(single * 8)


def test_an_unpriced_gpu_is_refused_rather_than_booked_at_zero(workspace, tmp_path):
    """`[spend]` is this backend's only gate. A run it cannot price is a run it
    is not bounding, and $0 would make the ceiling decoration."""
    cfg = config_mod.load()
    assert modal_tool.gpu_rate("GB200", cfg) is None

    with pytest.raises(ConfigError) as caught:
        modal_tool._rate_or_refuse("GB200", cfg)
    assert "modal.gpu_rates" in caught.value.fix
    # The fix names what *is* priced, so the next command is obvious.
    assert "H100" in caught.value.fix


# ---------------------------------------------------------------------------
# the 24-hour ceiling
# ---------------------------------------------------------------------------
def test_the_timeout_comes_from_the_estimate_with_a_margin(workspace, tmp_path):
    """An estimate that was exactly right is the one case a job would be killed
    for being on time."""
    cfg = config_mod.load()
    seconds = modal_tool._timeout_seconds(submission(tmp_path, estimate={"hours": 2.0}), cfg)
    assert seconds == int(2.0 * 1.25 * 3600)


def test_a_spec_with_no_estimate_cannot_set_a_timeout(workspace, tmp_path):
    cfg = config_mod.load()
    with pytest.raises(ConfigError) as caught:
        modal_tool._timeout_seconds(submission(tmp_path, estimate={"cost_usd": 1.0}), cfg)
    assert "estimate" in str(caught.value).lower()


def test_a_run_longer_than_modal_allows_is_refused_before_it_starts(workspace, tmp_path):
    """Modal kills a Sandbox at 24 hours whatever it was doing. Starting a
    20-hour run at a 1.25 margin means 25 hours of sandbox, and the failure would
    arrive a day later with nothing collected."""
    cfg = config_mod.load()
    with pytest.raises(ConfigError) as caught:
        modal_tool._timeout_seconds(submission(tmp_path, estimate={"hours": 20.0}), cfg)
    assert "24" in str(caught.value)
    assert "checkpoint" in caught.value.fix


def test_the_local_ceiling_can_lower_modals_but_never_raise_it(workspace, tmp_path, monkeypatch):
    """A config asking for 48 hours does not get 48 hours -- the container is
    stopped at 24 either way, and the only thing a higher local number changes
    is when you find out."""
    cfg = config_mod.load()
    monkeypatch.setitem(cfg.raw.setdefault("modal", {}), "max_hours", 48.0)
    with pytest.raises(ConfigError):
        modal_tool._timeout_seconds(submission(tmp_path, estimate={"hours": 30.0}), cfg)


# ---------------------------------------------------------------------------
# what the sandbox is told to do
# ---------------------------------------------------------------------------
def test_the_metrics_file_points_into_the_volume(workspace, tmp_path):
    """A sandbox's disk is gone when it exits and `collect` runs afterwards by
    construction, so a metrics file written beside the entrypoint is unreadable
    by the time anyone looks."""
    env = modal_tool._job_env(submission(tmp_path), "/grad/out/run-1")
    assert env["GRAD_METRICS_FILE"] == "/grad/out/run-1/metrics.json"
    assert env[modal_tool.OUT_ENV] == "/grad/out/run-1"


def test_the_wrapper_copies_a_stray_metrics_file_into_the_volume(workspace, tmp_path):
    """A pipeline written for another harness writes `metrics.json` beside
    itself. Losing it means the money was spent and the result is gone."""
    sub = submission(tmp_path)
    command = modal_tool._wrapped_command(["python", "train.py"], sub, "/grad/out/run-1")
    assert command[:2] == ["sh", "-c"]
    script = command[2]
    assert "python train.py" in script
    assert "cp -f metrics.json" in script


def test_the_wrapper_preserves_the_exit_code_across_the_copy(workspace, tmp_path):
    """`cp` failing must not turn a failed run into a successful one, or the
    reverse. This is the whole reason the wrapper is not a one-liner."""
    script = modal_tool._wrapped_command(["python", "train.py"], submission(tmp_path), "/out")[2]
    assert "rc=$?" in script
    assert script.rstrip().endswith("exit $rc")


def test_a_spec_command_overrides_the_entrypoint(workspace, tmp_path):
    sub = submission(tmp_path, target={"command": ["torchrun", "--nproc", "8", "train.py"]})
    assert modal_tool._command_for(sub) == ["torchrun", "--nproc", "8", "train.py"]


def test_the_run_directory_is_named_for_the_run(workspace):
    """One Volume, many runs. A shared output directory would let a later run
    overwrite the metrics of an earlier one that had not been collected yet."""
    cfg = config_mod.load()
    assert modal_tool._run_dir(cfg, "run-abc") == "/grad/out/run-abc"


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def test_the_cost_is_wall_clock_and_says_so(workspace):
    """Modal bills per second from container start; what is measurable here is
    the ledger's own interval, which includes the image pull and any delay
    before collection. It is an upper bound and the record must not imply it is
    a measurement."""
    from core import ledger_store as ls

    cfg = config_mod.load()
    run = ls.Run("run-1", {
        "id": "run-1",
        "platform": "modal",
        "submitted_at": ls.now_iso(),
        "target": {"gpu": "H100"},
    })
    cost, warning = modal_tool._actual_cost(run, {"gpu": "H100"}, cfg)
    assert cost >= 0.0
    assert warning and "wall clock" in warning


def test_the_cost_cannot_exceed_the_sandbox_timeout(workspace):
    """Collecting a week later must not book a week of H100 time: the container
    cannot have run longer than Modal would let it."""
    from core import ledger_store as ls

    cfg = config_mod.load()
    run = ls.Run("run-1", {
        "id": "run-1",
        "platform": "modal",
        "submitted_at": "2020-01-01T00:00:00+00:00",
        "target": {"gpu": "H100"},
    })
    cost, _ = modal_tool._actual_cost(run, {"gpu": "H100", "timeout_s": 3600}, cfg)
    assert cost == pytest.approx(3.9492, rel=1e-3)


def test_an_unpriced_gpu_at_collect_books_zero_and_admits_it(workspace):
    """Different from the submit-time refusal: by now the money is spent, so the
    only useful thing is to say the number is not one."""
    from core import ledger_store as ls

    run = ls.Run(
        "run-1", {"id": "run-1", "submitted_at": ls.now_iso(), "target": {"gpu": "GB200"}}
    )
    cost, warning = modal_tool._actual_cost(run, {"gpu": "GB200"}, config_mod.load())
    assert cost == 0.0
    assert "no rate configured" in warning


# ---------------------------------------------------------------------------
# the SDK's absence
# ---------------------------------------------------------------------------
def test_a_machine_without_the_extra_is_told_which_extra(workspace):
    """Not mocked away: this is the behaviour on a machine that has not
    installed it, which is every machine until someone does."""
    try:
        import modal  # noqa: F401, PLC0415
    except ImportError:
        with pytest.raises(ConfigError) as caught:
            modal_tool._modal()
        assert "modal" in caught.value.fix
    else:
        pytest.skip("the modal SDK is installed here")


def test_a_missing_credential_names_both_halves(workspace, monkeypatch):
    """Both are secret and either can be absent. Naming only the first would
    send someone round the loop twice."""
    monkeypatch.setattr(modal_tool, "_modal", lambda: object())
    monkeypatch.setattr(credentials, "get", lambda *_a, **_k: None)

    with pytest.raises(ConfigError) as caught:
        modal_tool._client()
    assert credentials.MODAL_TOKEN_ID in str(caught.value)
    assert credentials.MODAL_TOKEN_SECRET in str(caught.value)


def test_the_token_is_never_put_in_the_environment(workspace):
    """The strongest form of §9 available in this project: `from_credentials`
    sends the pair as gRPC headers, so unlike every other backend there is
    nothing to scrub because nothing is exported."""
    import ast

    source = Path(modal_tool.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Parsed rather than grepped. Two earlier versions of this test matched the
    # module's own *explanation* of why the environment is untouched -- first in
    # the module docstring, then in a function docstring that line-based
    # stripping could not see. The AST cannot be fooled by prose about the code.
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "os" not in imported, "this module has no business reading the environment"

    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "from_credentials" in calls


# ---------------------------------------------------------------------------
# the gates still apply
# ---------------------------------------------------------------------------
def test_submitting_without_an_expectation_is_refused(workspace, tmp_path):
    """The §6 gates are `core/submit.py`'s and this backend does not get its own
    version of them -- but a backend that forgot to *call* them would look
    exactly like one that had, until the first unpredicted run."""
    import argparse

    args = argparse.Namespace(
        spec=str(spec(tmp_path)), expect=None, overrides=[], gpu=None,
        task=None, project=None, smoke=False, no_digest=True,
    )
    with pytest.raises(GradError) as caught:
        modal_tool.cmd_submit(args)
    # Exit 4 (no preflight) or 5 (no expectation) -- both are gate refusals and
    # both must arrive before anything reaches Modal.
    assert caught.value.exit_code in (4, 5)
