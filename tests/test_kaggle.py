"""The Kaggle backend: the weekly accelerator allowance and the staged notebook.

The quota gate is to this backend what the spend gate is to the other two, and
it is tested the same way -- against a real ledger rather than a mock, because a
mock of a gate proves nothing about the gate. The rest covers the two things
unique to Kaggle: one uploadable file, and an accelerator id that has to resolve
to a pool or refuse.
"""

from __future__ import annotations

import ast
import base64
import datetime as dt
import io
import json
import tarfile
from pathlib import Path

import pytest

from core import config as config_mod, jsonl, kaggle_quota, ledger_store as ls, paths
from core.errors import EXIT_QUOTA, ConfigError, GateRefusal, UsageError
from core.submission import Submission
from tools import kaggle


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def write_config(workspace, extra: str = "") -> None:
    path = workspace / "config" / "grad.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '[kaggle]\nusername = "someone"\ndefault_accelerator = "NvidiaTeslaP100"\n' + extra,
        encoding="utf-8",
    )
    config_mod._cache.clear()


def make_submission(workspace, *, hours: float = 1.0, extra_files: dict[str, str] | None = None) -> Submission:
    d = workspace / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("import helper\nprint('x')\n", encoding="utf-8")
    (d / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name, body in (extra_files or {}).items():
        target = d / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        f"[estimate]\nhours = {hours}\n",
        encoding="utf-8",
    )
    return Submission.load(d / "spec.toml", resolve_digest=False)


def record_kaggle_run(
    *, hours: float, kind: str = "gpu", collected: bool = False, actual: float | None = None,
    submitted_at: str | None = None,
) -> str:
    run_id = ls.new_id("run")
    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED,
            "id": run_id,
            "status": "in_flight",
            "platform": "kaggle",
            "submitted_at": submitted_at or ls.now_iso(),
            "estimate_usd": 0.0,
            "estimated_duration_s": hours * 3600,
            kaggle_quota.F_ACCELERATOR: "NvidiaTeslaP100",
            kaggle_quota.F_KIND: kind,
            kaggle_quota.F_ESTIMATE: hours,
        }
    )
    if collected:
        ls.append_run_event(
            {
                "type": ls.T_RUN_COLLECTED,
                "id": run_id,
                "status": "completed",
                "collected_at": ls.now_iso(),
                "cost_usd_actual": 0.0,
                "deviations": [],
                kaggle_quota.F_ACTUAL: hours if actual is None else actual,
            }
        )
    return run_id


# ---------------------------------------------------------------------------
# the weekly allowance
# ---------------------------------------------------------------------------
def test_in_flight_hours_count_against_the_allowance(workspace):
    """The whole point of counting estimates: N runs pushed before any is
    collected must not all pass a 30-hour ceiling on zero hours counted."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    for _ in range(3):
        record_kaggle_run(hours=9.0)

    with pytest.raises(GateRefusal) as exc:
        kaggle_quota.check_quota(cfg, "gpu", 9.0, accelerator="NvidiaTeslaP100")
    assert exc.value.exit_code == EXIT_QUOTA
    assert exc.value.code == "quota_weekly"
    assert "in flight" in exc.value.message


def test_collected_runs_count_at_their_actual_hours(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    # Estimated 10h, actually used 2h. The pool must believe the actual.
    record_kaggle_run(hours=10.0, collected=True, actual=2.0)

    state = kaggle_quota.check_quota(cfg, "gpu", 1.0, accelerator="NvidiaTeslaP100")
    assert state["used_hours"] == pytest.approx(2.0)
    assert state["projected_hours"] == pytest.approx(3.0)
    assert state["projected_remaining_hours"] == pytest.approx(27.0)


def test_hours_outside_the_window_do_not_count(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=9)).isoformat(timespec="seconds")
    record_kaggle_run(hours=29.0, submitted_at=old)

    state = kaggle_quota.check_quota(cfg, "gpu", 5.0, accelerator="NvidiaTeslaP100")
    assert state["used_hours"] == pytest.approx(0.0)


def test_gpu_and_tpu_draw_from_separate_pools(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    record_kaggle_run(hours=29.0, kind="gpu")

    # The GPU pool is nearly gone; the TPU pool is untouched.
    with pytest.raises(GateRefusal):
        kaggle_quota.check_quota(cfg, "gpu", 5.0, accelerator="NvidiaTeslaP100")
    state = kaggle_quota.check_quota(cfg, "tpu", 5.0, accelerator="TpuV38")
    assert state["used_hours"] == pytest.approx(0.0)


def test_session_cap_refuses_a_run_kaggle_would_stop(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    with pytest.raises(GateRefusal) as exc:
        kaggle_quota.check_session(cfg, "gpu", 20.0, accelerator="NvidiaTeslaP100")
    assert exc.value.code == "quota_session"
    assert "12.0h" in exc.value.message


def test_tpu_sessions_are_capped_shorter_than_gpu_ones(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    # 10h is fine on a GPU and past the cap on a TPU.
    assert kaggle_quota.check_session(cfg, "gpu", 10.0, accelerator="NvidiaTeslaP100")
    with pytest.raises(GateRefusal):
        kaggle_quota.check_session(cfg, "tpu", 10.0, accelerator="TpuV38")


def test_session_is_checked_before_the_weekly_pool(workspace):
    """A 20h run on an empty week fails both. The session message is the useful
    one, because it is about this run and true regardless of the week."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    with pytest.raises(GateRefusal) as exc:
        kaggle_quota.check(cfg, "gpu", 40.0, accelerator="NvidiaTeslaP100")
    assert exc.value.code == "quota_session"


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_estimate_cannot_disable_the_ceiling(workspace, value):
    """NaN fails every comparison, so `projected > allowance` would wave it
    through -- the one input that disables a ceiling while looking like a number."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    with pytest.raises(GateRefusal) as exc:
        kaggle_quota.check_quota(cfg, "gpu", value, accelerator="NvidiaTeslaP100")
    assert exc.value.code == "quota_value_invalid"


def test_cpu_kernels_are_unmetered_but_not_unknown(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    assert kaggle_quota.check_quota(cfg, "cpu", 5.0, accelerator="none") is None
    assert kaggle_quota.check(cfg, "cpu", 5.0, accelerator="none")["metered"] is False


def test_the_allowance_is_rechecked_inside_the_append_lock(workspace):
    """The gate reads the ledger and this run's hours land in it afterwards, so
    two submitters racing could both pass on the same stale read. The dollar
    ceilings get an in-lock re-check; the allowance must not be the exception."""
    from core import submit as submit_lib

    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace, hours=20.0)

    calls: list[str] = []

    def _competitor() -> None:
        # Stands in for the run that committed between the gate and this write.
        calls.append("checked")
        kaggle_quota.check(cfg, "gpu", 20.0, accelerator="NvidiaTeslaP100")

    record_kaggle_run(hours=20.0)  # 20h already in flight; 20 more is over 30

    with pytest.raises(GateRefusal) as exc:
        submit_lib.record_submission(
            sub,
            expectation_id=None,
            platform="kaggle",
            target={},
            command=["python", "train.py"],
            precondition=_competitor,
        )
    assert calls == ["checked"]
    assert exc.value.exit_code == EXIT_QUOTA

    # And the refused run left nothing behind holding hours against the pool.
    assert kaggle_quota.summary(cfg)["pools"]["gpu"]["total_hours"] == pytest.approx(20.0)


def test_quota_summary_reports_both_pools(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    record_kaggle_run(hours=4.0, kind="gpu", collected=True)
    summary = kaggle_quota.summary(cfg)
    assert summary["pools"]["gpu"]["remaining_hours"] == pytest.approx(26.0)
    assert summary["pools"]["tpu"]["remaining_hours"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# a spec with no duration cannot be counted
# ---------------------------------------------------------------------------
def test_a_spec_without_an_estimate_is_refused(workspace):
    """0 hours passes every allowance forever. On a dollar backend that is merely
    optimistic; here it is the difference between a ceiling and a decoration."""
    write_config(workspace)
    d = workspace / "nodur"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("print('x')\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n", encoding="utf-8"
    )
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    with pytest.raises(UsageError) as exc:
        kaggle.estimated_hours(sub)
    assert "[estimate] hours" in exc.value.fix


def test_a_non_numeric_estimate_is_a_usage_error_not_a_crash(workspace):
    """`hours = "two"` is a typo in a file the agent wrote, and reporting it as
    exit 1 sends the reader looking for a bug in the CLI."""
    write_config(workspace)
    d = workspace / "badest"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("print('x')\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        '[estimate]\nhours = "two"\n',
        encoding="utf-8",
    )
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    with pytest.raises(UsageError):
        kaggle.estimated_hours(sub)


# ---------------------------------------------------------------------------
# which account
# ---------------------------------------------------------------------------
def _account_args(**kw):
    import argparse

    return argparse.Namespace(**{"username": None, "clear": False, "check": False, **kw})


def test_the_account_command_sets_and_reports_the_username(workspace):
    write_config(workspace)
    out = kaggle.cmd_account(_account_args(username="somebody"))
    assert out["username"] == "somebody"
    assert out["source"] == "state"
    # Readable, not hidden in a credential store: only the key is secret.
    assert Path(out["path"]).is_file()
    assert kaggle.stored_username() == "somebody"


def test_the_stored_account_wins_over_the_config_and_says_so(workspace):
    """A command that silently does nothing because a config file disagrees is
    worse than one that overrides it -- but the shadowing has to be stated."""
    write_config(workspace)  # writes username = "someone"
    out = kaggle.cmd_account(_account_args(username="somebody"))
    assert out["shadowed_config_username"] == "someone"

    cfg = config_mod.load(reload=True)
    assert kaggle.resolve_username(cfg) == ("somebody", "state")
    assert kaggle._username(cfg) == "somebody"


def test_clearing_the_account_falls_back_to_the_config(workspace):
    write_config(workspace)
    kaggle.cmd_account(_account_args(username="somebody"))
    out = kaggle.cmd_account(_account_args(clear=True))
    assert out["username"] == "someone"
    assert out["source"] == "config"


def test_an_unset_account_names_the_command_that_sets_it(workspace):
    path = workspace / "config" / "grad.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[kaggle]\n", encoding="utf-8")
    config_mod._cache.clear()
    cfg = config_mod.load(reload=True)

    out = kaggle.cmd_account(_account_args())
    assert out["username"] is None and out["source"] == "unset"
    assert "account --set" in out["fix"]
    with pytest.raises(ConfigError) as exc:
        kaggle._username(cfg)
    assert "account --set" in exc.value.fix


@pytest.mark.parametrize("bad", ["", "   ", "some/body", "some body", "some\tbody"])
def test_a_username_that_would_break_the_kernel_ref_is_refused(workspace, bad):
    """A kernel reference is <username>/<slug>, so a slash silently re-points
    every push, status and collect at another account's path."""
    write_config(workspace)
    with pytest.raises(UsageError):
        kaggle.validate_username(bad)


def test_set_and_clear_together_is_a_usage_error(workspace):
    write_config(workspace)
    with pytest.raises(UsageError):
        kaggle.cmd_account(_account_args(username="somebody", clear=True))


def test_the_account_check_reports_a_bad_pair_rather_than_raising(workspace, monkeypatch):
    """Both halves fail the same way and have the same fix: look at the pair."""
    write_config(workspace)
    monkeypatch.setattr(kaggle, "_executable", lambda: "kaggle")

    def _boom(argv, cfg_arg, timeout):
        from core.errors import UpstreamError

        raise UpstreamError("401 Unauthorized", fix=None)

    monkeypatch.setattr(kaggle, "_run", _boom)
    out = kaggle.cmd_account(_account_args(check=True))
    assert out["check"]["ok"] is False
    assert "credential set kaggle_key" in out["check"]["fix"]


def test_the_account_check_passes_on_a_good_pair(workspace, monkeypatch):
    write_config(workspace)
    monkeypatch.setattr(kaggle, "_executable", lambda: "kaggle")
    monkeypatch.setattr(kaggle, "_run", lambda argv, cfg_arg, timeout: "ref  title\nx  y\n")
    out = kaggle.cmd_account(_account_args(check=True))
    assert out["check"]["ok"] is True


def test_the_account_lives_outside_the_workspace(workspace):
    """A Kaggle account belongs to whoever installed this, like the key it pairs
    with -- not to the research sitting in one folder."""
    from core import appdata

    assert kaggle.account_path() == appdata.state_dir() / "kaggle.json"
    assert workspace not in kaggle.account_path().parents


# ---------------------------------------------------------------------------
# the accelerator inventory
# ---------------------------------------------------------------------------
def test_an_unknown_accelerator_is_a_config_error_not_a_guess(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    with pytest.raises(ConfigError) as exc:
        cfg.accelerator_kind("NvidiaTeslaV100")
    assert "known accelerators" in exc.value.message


def test_accelerator_resolution_order(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    # config default, with nothing more specific
    assert kaggle.resolve_accelerator(None, sub, cfg) == "NvidiaTeslaP100"
    # the spec beats the default
    sub.target["accelerator"] = "TpuV38"
    assert kaggle.resolve_accelerator(None, sub, cfg) == "TpuV38"
    # the flag beats the spec
    assert kaggle.resolve_accelerator("NvidiaH100", sub, cfg) == "NvidiaH100"


def test_a_malformed_accelerator_pool_is_refused_at_load(workspace):
    write_config(workspace, '\n[kaggle.accelerators]\nNvidiaTeslaP100 = "GPU"\n')
    with pytest.raises(ConfigError) as exc:
        config_mod.load(reload=True)
    assert "gpu, tpu, cpu" in exc.value.message


def test_a_default_accelerator_outside_the_table_is_refused_at_load(workspace):
    write_config(workspace, "\n[kaggle.quota]\ngpu_hours_per_week = 30.0\n")
    path = workspace / "config" / "grad.toml"
    path.write_text(
        '[kaggle]\nusername = "someone"\ndefault_accelerator = "NotAThing"\n', encoding="utf-8"
    )
    config_mod._cache.clear()
    with pytest.raises(ConfigError) as exc:
        config_mod.load(reload=True)
    assert "not in [kaggle.accelerators]" in exc.value.message


def test_a_string_weekly_allowance_is_refused_at_load(workspace):
    """`[kaggle.quota]` is nested a table deeper than the flat numeric checks
    reach, so without its own validation a string there is a TypeError from
    inside the gate that is refusing a submission."""
    write_config(workspace, '\n[kaggle.quota]\ngpu_hours_per_week = "thirty"\n')
    with pytest.raises(ConfigError) as exc:
        config_mod.load(reload=True)
    assert "kaggle.quota.gpu_hours_per_week" in (exc.value.fix or "")


# ---------------------------------------------------------------------------
# one uploadable file
# ---------------------------------------------------------------------------
def _blob_from(notebook: dict) -> str:
    """Pull the base64 payload back out of the generated notebook.

    Walks the cell's source lines between `blob = (` and its closing paren. A
    regex over the joined source would cut at the wrong `)` -- the unpack code
    below the literal has several.
    """
    lines = notebook["cells"][0]["source"]
    start = next(i for i, line in enumerate(lines) if line.startswith("blob = (")) + 1
    end = next(i for i, line in enumerate(lines) if i >= start and line.strip() == ")")
    return "".join(ast.literal_eval(line.strip()) for line in lines[start:end])


def test_the_payload_is_inside_the_notebook_not_beside_it(workspace, tmp_path):
    """`kernels push` uploads the code file and the metadata and nothing else, so
    a payload written next to the notebook never leaves this machine."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    out = tmp_path / "stage"
    out.mkdir()
    kaggle._stage(
        cfg, sub, out, ref="someone/grad-run-1", slug="grad-run-1",
        accelerator="NvidiaTeslaP100", command=["python", "train.py"],
    )
    staged = {p.name for p in out.iterdir()}
    assert staged == {"grad-run-1.ipynb", "kernel-metadata.json"}

    notebook = json.loads((out / "grad-run-1.ipynb").read_text(encoding="utf-8"))
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_blob_from(notebook)))) as tar:
        names = set(tar.getnames())
    assert {"train.py", "helper.py", "spec.toml"} <= names


def test_junk_directories_are_not_packed(workspace, tmp_path):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(
        workspace, extra_files={"__pycache__/train.cpython-312.pyc": "stale", "data/x.csv": "a,b\n"}
    )
    out = tmp_path / "stage"
    out.mkdir()
    kaggle._stage(
        cfg, sub, out, ref="someone/grad-run-1", slug="grad-run-1",
        accelerator="NvidiaTeslaP100", command=["python", "train.py"],
    )
    notebook = json.loads((out / "grad-run-1.ipynb").read_text(encoding="utf-8"))
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_blob_from(notebook)))) as tar:
        names = set(tar.getnames())
    # A sibling data file is staged, exactly as `scp -r` would stage it.
    assert "data/x.csv" in names
    # A stale pycache from this machine's architecture is not.
    assert not any(n.startswith("__pycache__") for n in names)


def test_an_oversized_pipeline_is_refused_before_the_upload(workspace, monkeypatch):
    write_config(workspace)
    monkeypatch.setattr(kaggle, "MAX_PAYLOAD_B64", 256)
    sub = make_submission(workspace, extra_files={"big.bin": "x" * 100_000})
    with pytest.raises(UsageError) as exc:
        kaggle._payload_b64(sub)
    assert "dataset" in (exc.value.fix or "")


def test_the_marker_is_written_before_the_cell_can_fail(workspace):
    """The failing run is the one whose exit code matters most, so the marker
    must land before anything that can stop the cell."""
    write_config(workspace)
    sub = make_submission(workspace)
    notebook = kaggle._notebook_for(sub, ["python", "train.py"], payload="")
    marker_cell = "".join(notebook["cells"][2]["source"])
    assert marker_cell.index(kaggle.MARKER) < marker_cell.index("raise SystemExit")


def _execute_notebook(notebook: dict, work) -> dict:
    """Run the generated cells with `/kaggle/working` pointed at a temp dir.

    Structure tests cannot catch a notebook that unpacks nothing or loses the
    exit code, and this is the one artifact whose bugs are only observable on
    Kaggle -- where the feedback loop is a push, a queue, and a dead kernel.
    """
    namespace: dict = {}
    for i, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"]).replace("/kaggle/working", work.as_posix())
        try:
            exec(compile(source, f"<cell{i}>", "exec"), namespace)  # noqa: S102
        except SystemExit:
            pass
    return namespace


def test_the_generated_notebook_unpacks_and_runs_the_pipeline(workspace, tmp_path):
    write_config(workspace)
    d = workspace / "runnable"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text(
        "import json, helper\n"
        "print('ran with', helper.VALUE)\n"
        "json.dump({'val_loss': 3.0}, open('metrics.json', 'w'))\n",
        encoding="utf-8",
    )
    (d / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n[estimate]\nhours = 1.0\n",
        encoding="utf-8",
    )
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    payload, _ = kaggle._payload_b64(sub)
    work = tmp_path / "working"
    work.mkdir()

    _execute_notebook(kaggle._notebook_for(sub, ["python", "train.py"], payload=payload), work)

    # The import graph survived the round trip: train.py found helper.py.
    assert "ran with 42" in (work / "stdout.log").read_text(encoding="utf-8")
    assert json.loads((work / "metrics.json").read_text(encoding="utf-8"))["val_loss"] == 3.0
    assert json.loads((work / kaggle.MARKER).read_text(encoding="utf-8"))["exit_code"] == 0


def test_a_failing_entrypoint_still_leaves_its_exit_code_behind(workspace, tmp_path):
    """Kaggle's own status says `error` either way. The marker is what says
    *why*, so it has to survive the failure that makes it worth reading."""
    write_config(workspace)
    d = workspace / "failing"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text("import sys\nsys.exit(7)\n", encoding="utf-8")
    (d / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n[estimate]\nhours = 1.0\n",
        encoding="utf-8",
    )
    sub = Submission.load(d / "spec.toml", resolve_digest=False)
    payload, _ = kaggle._payload_b64(sub)
    work = tmp_path / "working"
    work.mkdir()

    _execute_notebook(kaggle._notebook_for(sub, ["python", "train.py"], payload=payload), work)

    marker = json.loads((work / kaggle.MARKER).read_text(encoding="utf-8"))
    assert marker["exit_code"] == 7
    assert marker["state"] == "finished"


def test_the_slug_never_contains_an_underscore(workspace):
    """A title with an underscore makes the run's status unobtainable, so collect
    would poll forever and the run would go stale and block later submissions."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    slug = kaggle._slug(cfg, "run_20260816T053000_a1b2c3")
    assert "_" not in slug
    assert slug.islower()


def test_metadata_sets_the_accelerator_booleans_from_the_pool(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    gpu = kaggle._metadata(cfg, sub, ref="a/b", slug="b", accelerator="NvidiaTeslaP100")
    tpu = kaggle._metadata(cfg, sub, ref="a/b", slug="b", accelerator="TpuV38")
    assert (gpu["enable_gpu"], gpu["enable_tpu"]) == (True, False)
    assert (tpu["enable_gpu"], tpu["enable_tpu"]) == (False, True)
    # Private and offline unless the spec asks otherwise.
    assert gpu["is_private"] is True
    assert gpu["enable_internet"] is False


def test_a_spec_can_ask_for_internet(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    sub.target["internet"] = True
    meta = kaggle._metadata(cfg, sub, ref="a/b", slug="b", accelerator="NvidiaTeslaP100")
    assert meta["enable_internet"] is True


# ---------------------------------------------------------------------------
# the smoke check
# ---------------------------------------------------------------------------
def test_preflight_dispatches_a_kaggle_spec_to_this_backend(workspace):
    """Gate 1's input is a passing `smoke` check, and preflight picks the backend
    by reading [target] platform. Without a branch here, `preflight run` reports
    "cannot smoke without a real target" and every Kaggle job is refused at gate
    1 for a reason that has nothing to do with the job."""
    from tools import preflight

    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    sub.target["platform"] = "kaggle"

    seen = {}

    def _fake(sub_arg, cfg_arg, **kwargs):
        seen["args"] = (sub_arg, cfg_arg, kwargs)
        return {"ok": True}

    import tools.kaggle as kaggle_mod

    original = kaggle_mod.run_smoke
    kaggle_mod.run_smoke = _fake
    try:
        assert preflight._check_smoke(sub, cfg) == {"ok": True}
    finally:
        kaggle_mod.run_smoke = original
    assert seen["args"][0] is sub
    # Positionally, with nothing backend-specific.
    assert seen["args"][2] == {}


def test_run_smoke_needs_nothing_preflight_cannot_give_it(workspace):
    """preflight calls every backend as `run_smoke(sub, cfg)`. A required
    keyword here would make this backend submittable but not preflightable."""
    import inspect

    params = inspect.signature(kaggle.run_smoke).parameters
    required = [
        name for name, p in params.items()
        if name not in ("sub", "cfg") and p.default is inspect.Parameter.empty
    ]
    assert required == []


def test_run_smoke_resolves_its_own_accelerator(workspace):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    sub.target["accelerator"] = "TpuV38"
    # The same resolution `submit` uses, reachable with no argument at all.
    assert kaggle.resolve_accelerator(None, sub, cfg) == "TpuV38"


def test_an_unknown_platform_still_names_kaggle_as_an_option(workspace):
    from tools import preflight

    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    sub.target["platform"] = "nowhere"
    result = preflight._check_smoke(sub, cfg)
    assert result["ok"] is False
    assert "kaggle" in result["fix"]


def test_the_smoke_cap_is_handed_to_kaggle_not_just_to_our_patience(workspace, monkeypatch):
    """A poll deadline stops us waiting; it does not stop the kernel. Without
    --timeout on the push, a smoke could run for hours against the weekly
    allowance after the check had already returned."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    seen: dict = {}

    monkeypatch.setattr(kaggle, "_executable", lambda: "kaggle")
    monkeypatch.setattr(
        kaggle, "_run", lambda argv, cfg_arg, timeout: seen.setdefault("argv", argv) and ""
    )
    kaggle._push(
        cfg, workspace, accelerator="NvidiaTeslaP100", timeout_s=900, kernel_timeout_s=600
    )
    argv = seen["argv"]
    assert "--timeout" in argv
    assert argv[argv.index("--timeout") + 1] == "600"
    assert argv[argv.index("--accelerator") + 1] == "NvidiaTeslaP100"


def test_a_real_submit_carries_no_kernel_timeout(workspace, monkeypatch):
    """The cap belongs to the carve-out. A real run is bounded by the session cap
    and the weekly allowance, and clamping it to the smoke's ten minutes would
    kill every job this backend exists to run."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    seen: dict = {}
    monkeypatch.setattr(kaggle, "_executable", lambda: "kaggle")
    monkeypatch.setattr(
        kaggle, "_run", lambda argv, cfg_arg, timeout: seen.setdefault("argv", argv) and ""
    )
    kaggle._push(cfg, workspace, accelerator="NvidiaTeslaP100", timeout_s=900)
    assert "--timeout" not in seen["argv"]


def test_a_spec_pointing_at_another_backend_is_refused(workspace):
    """Its preflight smoke ran somewhere else, and gate 1 would accept that
    record without rechecking where it came from."""
    write_config(workspace)
    sub = make_submission(workspace)
    sub.target["platform"] = "hf"
    with pytest.raises(UsageError) as exc:
        kaggle.require_platform(sub)
    assert 'platform = "kaggle"' in exc.value.fix


def _fake_cli(monkeypatch, *, status: str = "complete", exit_code: int = 0, elapsed_s: float = 42.0):
    """Stand in for the `kaggle` CLI at the subprocess boundary.

    Faked here rather than deeper, because everything above this line is the
    code under test: what gets pushed, what the marker says, how hours are read,
    and what happens to the kernel afterwards.
    """
    calls: list[list[str]] = []

    def _run(argv, cfg_arg, timeout):
        calls.append(argv)
        verb = argv[2] if len(argv) > 2 else ""
        if verb == "status":
            return f'{argv[3]} has status "{status}"'
        if verb == "output":
            dest = Path(argv[argv.index("-p") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / kaggle.MARKER).write_text(
                json.dumps({"state": "finished", "exit_code": exit_code, "elapsed_s": elapsed_s}),
                encoding="utf-8",
            )
            ref_slug = argv[3].split("/", 1)[-1]
            (dest / f"{ref_slug}.log").write_text(
                json.dumps(
                    [
                        {"stream_name": "stdout", "time": 0.0, "data": "start\n"},
                        {"stream_name": "stdout", "time": elapsed_s, "data": "smoke ok\n"},
                    ]
                ),
                encoding="utf-8",
            )
        return ""

    monkeypatch.setattr(kaggle, "_executable", lambda: "kaggle")
    monkeypatch.setattr(kaggle, "_run", _run)
    return calls


def test_a_smoke_check_runs_end_to_end(workspace, monkeypatch):
    """The whole carve-out, from push to run record, with only the CLI faked."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    calls = _fake_cli(monkeypatch)

    result = kaggle.run_smoke(sub, cfg)

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["cost_usd"] == 0.0
    assert result["timed_out"] is False
    # Hours came from the kernel log, not from our own wall clock.
    # `abs`, not `rel`: the record rounds hours to four places on the way in.
    assert result["accelerator_hours"] == pytest.approx(42.0 / 3600.0, abs=1e-4)
    assert "smoke ok" in result["output"]

    verbs = [c[2] for c in calls]
    assert verbs[0] == "push"
    # allow_artifact_upload is false by default, so the kernel does not outlive
    # the check that made it.
    assert "delete" in verbs

    # And it landed on the ledger as a smoke run holding its actual hours.
    run = next(r for r in ls.runs() if r.id == result["run_id"])
    assert run.is_smoke and run.collected
    assert run.get(kaggle_quota.F_ACTUAL) == pytest.approx(42.0 / 3600.0, abs=1e-4)


def test_a_failing_smoke_reports_the_exit_code(workspace, monkeypatch):
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    _fake_cli(monkeypatch, status="error", exit_code=7)

    result = kaggle.run_smoke(sub, cfg)
    assert result["ok"] is False
    assert result["exit_code"] == 7
    assert "smoke.log" in result["fix"]


def test_a_smoke_that_never_left_the_queue_is_not_a_pass(workspace, monkeypatch):
    """A kernel still queued when the grace ran out has no marker and no status
    worth believing. Calling that a pass certifies an environment it never
    reached -- and gate 1 would then accept it."""
    # The poll deadline is the smoke's wall-clock cap *plus* the queue grace, so
    # both have to shrink -- zeroing only the grace still waits out the cap.
    write_config(
        workspace,
        "queue_grace_s = 0\npoll_interval_s = 1\n\n[smoke]\nmax_wall_clock_s = 1\n",
    )
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    _fake_cli(monkeypatch, status="queued")

    result = kaggle.run_smoke(sub, cfg)
    assert result["ok"] is False
    assert result["timed_out"] is True
    assert "nothing was proved" in result["reason"]
    assert "queue_grace_s" in result["fix"]


def test_smoke_hours_count_against_later_submissions(workspace, monkeypatch):
    """Smoke skips the gates but not the ledger, and not the allowance either --
    otherwise the exemption is a hole in the pool as well as in the gate."""
    write_config(workspace)
    cfg = config_mod.load(reload=True)
    sub = make_submission(workspace)
    _fake_cli(monkeypatch, elapsed_s=3600.0)

    kaggle.run_smoke(sub, cfg)
    assert kaggle_quota.summary(cfg)["pools"]["gpu"]["total_hours"] == pytest.approx(1.0, rel=1e-3)


def test_a_spec_with_no_platform_is_left_to_preflight(workspace):
    """Not a mismatch -- a spec preflight refuses to smoke at all, with its own
    instruction. Refusing twice for the same thing helps nobody."""
    write_config(workspace)
    sub = make_submission(workspace)
    assert kaggle.require_platform(sub) is None


# ---------------------------------------------------------------------------
# status parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "output,expected",
    [
        ('someone/grad-run-1 has status "complete"', "complete"),
        ('someone/grad-run-1 has status "error"', "error"),
        ('someone/grad-run-1 has status "running"', "running"),
        ('{"status": "complete", "failureMessage": null}', "complete"),
        ("", "unknown"),
        ("something nobody has seen before", "unknown"),
    ],
)
def test_status_parsing(output, expected):
    assert kaggle._parse_status(output)["status"] == expected


def test_an_unreadable_status_is_never_read_as_complete():
    """Guessing `complete` collects a kernel that is still running and writes a
    run record with no results in it."""
    assert kaggle._parse_status("gibberish")["status"] not in kaggle._TERMINAL


# ---------------------------------------------------------------------------
# hours from the kernel log
# ---------------------------------------------------------------------------
def test_execution_hours_come_from_the_log_not_the_wall_clock(tmp_path):
    """Charging queue time to a weekly allowance would make the allowance shrink
    under Kaggle's load rather than under our use."""
    (tmp_path / "grad-run-1.log").write_text(
        json.dumps(
            [
                {"stream_name": "stdout", "time": 0.5, "data": "start\n"},
                {"stream_name": "stdout", "time": 3600.0, "data": "done\n"},
            ]
        ),
        encoding="utf-8",
    )
    hours, text = kaggle._log_hours(tmp_path, "grad-run-1")
    assert hours == pytest.approx(1.0)
    assert "done" in text


def test_an_unreadable_log_reports_no_hours_rather_than_a_guess(tmp_path):
    (tmp_path / "grad-run-1.log").write_text("not json", encoding="utf-8")
    assert kaggle._log_hours(tmp_path, "grad-run-1") == (None, "")


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------
def test_bare_kaggle_is_denied():
    import hooks

    denial = hooks.evaluate_bash("kaggle kernels push -p .")
    assert denial is not None
    assert "tools.kaggle" in denial.suggestion


def test_the_credential_pair_is_scrubbed_from_the_environment(monkeypatch):
    """The kaggle CLI reads these straight out of the environment, so leaving
    them there is the general remote-execution capability §9 denies."""
    from core import credentials

    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/somewhere")
    removed = credentials.scrub_environment()
    assert {"KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"} <= set(removed)


def test_a_compressible_directory_is_not_refused_before_it_is_packed(workspace, monkeypatch):
    """The early guard counts raw bytes; the limit bounds base64-of-gzip.

    Those units differ by whatever gzip achieves, so refusing on raw size alone
    turned a tree of repetitive source -- which packs to a fraction of itself --
    into a submission that never got the chance to be measured accurately.
    """
    write_config(workspace)
    monkeypatch.setattr(kaggle, "MAX_PAYLOAD_B64", 4_000)
    # 60 kB raw, far past the 4 kB limit, and hugely compressible.
    sub = make_submission(workspace, extra_files={"generated.py": "x = 1\n" * 10_000})
    blob, _ = kaggle._payload_b64(sub)
    assert len(blob) <= kaggle.MAX_PAYLOAD_B64


def test_a_directory_too_large_to_pack_is_still_refused_without_packing(workspace, monkeypatch):
    write_config(workspace)
    monkeypatch.setattr(kaggle, "MAX_PAYLOAD_B64", 256)
    monkeypatch.setattr(kaggle, "_PACK_MEMORY_SLACK", 2)
    sub = make_submission(workspace, extra_files={"big.bin": "x" * 100_000})
    with pytest.raises(UsageError) as exc:
        kaggle._payload_b64(sub)
    assert "dataset" in (exc.value.fix or "")


def test_a_credential_beside_the_entrypoint_refuses_the_whole_payload(workspace):
    """The spec directory is uploaded whole, so a `.env` here is a `.env` there."""
    write_config(workspace)
    sub = make_submission(workspace, extra_files={".env": "KAGGLE_KEY=hunter2\n"})
    with pytest.raises(UsageError) as exc:
        kaggle._payload_b64(sub)
    assert ".env" in exc.value.message
    # Refused, not quietly dropped: a run that fails remotely for a file sitting
    # right there locally is the bug report this backend exists to avoid.
    assert "move them outside" in (exc.value.fix or "")


def test_an_ordinary_pipeline_file_is_not_mistaken_for_a_credential(workspace):
    write_config(workspace)
    sub = make_submission(workspace, extra_files={"data/x.csv": "a,b\n1,2\n"})
    blob, packed = kaggle._payload_b64(sub)
    assert "data/x.csv" in packed and blob


@pytest.mark.parametrize("name", [".ssh/config", ".SSH/config", ".Aws/config", ".GNUPG/trustdb.gpg"])
def test_a_credential_directory_is_caught_whatever_its_casing(workspace, name):
    """Windows keeps the casing the author typed in `Path.parts` while treating
    the directory itself as case-insensitive, so `.SSH` and `.ssh` are one
    directory with two spellings -- and only one of them used to be checked."""
    write_config(workspace)
    sub = make_submission(workspace, extra_files={name: "secret\n"})
    with pytest.raises(UsageError) as exc:
        kaggle._payload_b64(sub)
    assert "credential" in exc.value.message


def test_a_file_merely_named_like_a_secret_directory_is_still_packed(workspace):
    """`Config.py` is source, not `.ssh/config`. The directory rule is about
    directories, and case-folding it must not widen it into the filenames."""
    write_config(workspace)
    sub = make_submission(workspace, extra_files={"src/Config.py": "X = 1\n"})
    _, packed = kaggle._payload_b64(sub)
    assert "src/Config.py" in packed
