"""Running one evolve candidate on the two job backends.

`test_candidate_hosts.py` covers the SSH adapter, where the remote is a machine
that stays up and a candidate is a directory copied onto it. These two are
shaped differently and the difference is the whole reason each backend owns its
own adapter:

  * **Kaggle** has no `scp`. The pipeline already travels as a base64 tar inside
    the generated notebook, so a candidate is that same payload with one file
    swapped -- which means the secret scan and the size refusals apply to a
    candidate exactly as they do to a submission.
  * **HF Jobs** has no upload step at all. The pipeline is *in the image* and
    `_command_for` just runs the entrypoint it contains, so a candidate -- a
    program the image by definition does not have -- needs a way in.

Neither test touches a network. The CLI runner, the Hub client and the pollers
are stubbed; what is under test is what each adapter asks them for.
"""

from __future__ import annotations

import base64
import io
import secrets
import tarfile

import pytest

from core import campaign as camp, config as config_mod, kaggle_quota
from core.errors import UsageError
from tools import jobs as jobs_tool, kaggle as kaggle_tool

from test_candidate_hosts import a_submission


def unpack(blob: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob))) as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member)
            out[member.name] = handle.read().decode("utf-8") if handle else ""
    return out


# ---------------------------------------------------------------------------
# Kaggle: the candidate rides inside the notebook's payload
# ---------------------------------------------------------------------------
def test_an_override_replaces_a_file_rather_than_shadowing_it(workspace):
    """Two entries with one name in a tar is a file whose contents depend on
    extraction order, which for a candidate means a score that depends on tar."""
    blob, packed = kaggle_tool._payload_b64(
        a_submission(workspace), overrides={"train.py": "print('mutated')"}
    )
    assert unpack(blob)["train.py"] == "print('mutated')"
    assert packed.count("train.py") == 1


def test_an_override_can_add_a_file_the_pipeline_does_not_have(workspace):
    """A candidate is `initial.py` and `evaluate.py`, which a pipeline built for
    ordinary submission has no reason to contain."""
    blob, _ = kaggle_tool._payload_b64(
        a_submission(workspace),
        overrides={"initial.py": "x = 1", "evaluate.py": "print('{}')"},
    )
    files = unpack(blob)
    assert files["initial.py"] == "x = 1"
    assert files["evaluate.py"] == "print('{}')"
    assert "train.py" in files, "the proven pipeline stopped travelling"


def test_the_payload_is_a_function_of_its_inputs(workspace):
    """Deterministic, so two pushes of one candidate are byte-identical and a
    difference between two blobs means a difference in the code."""
    sub = a_submission(workspace)
    first, _ = kaggle_tool._payload_b64(sub, overrides={"initial.py": "x = 1"})
    second, _ = kaggle_tool._payload_b64(sub, overrides={"initial.py": "x = 1"})
    assert first == second


def test_a_payload_entry_may_not_climb_out_of_the_pipeline(workspace):
    for bad in ("../escape.py", "/etc/passwd", "a/../../b.py"):
        with pytest.raises(UsageError):
            kaggle_tool._payload_b64(a_submission(workspace), overrides={bad: "x"})


def stub_kaggle(monkeypatch, *, status="complete", marker=None, hours=0.03, log=""):
    pushed: dict = {}
    monkeypatch.setattr(kaggle_tool, "_username", lambda cfg: "someone")
    monkeypatch.setattr(
        kaggle_tool, "_push", lambda cfg, workdir, **kw: pushed.update(kw) or "pushed"
    )
    monkeypatch.setattr(kaggle_tool, "_wait", lambda cfg, ref, deadline: {"status": status})
    monkeypatch.setattr(
        kaggle_tool,
        "_fetch_output",
        lambda cfg, ref, artifacts: (marker if marker is not None else {}, hours, log),
    )
    return pushed


def run_kaggle(workspace, **kwargs):
    options = {
        "candidate_id": "camp-1-g0-c0",
        "files": {"initial.py": "x = 1", "evaluate.py": "print('{}')"},
        "command": ["python", "evaluate.py"],
        "timeout_s": 1800,
        "artifacts": workspace / "artifacts",
    }
    options.update(kwargs)
    return kaggle_tool.evaluate_candidate(
        a_submission(workspace), config_mod.load(), **options
    )


def test_the_kernel_is_bounded_where_it_runs(workspace, monkeypatch):
    """Without `--timeout` the only limit is how long we choose to poll, which
    stops us waiting and does not stop the kernel -- and an abandoned kernel
    keeps spending the weekly allowance."""
    pushed = stub_kaggle(
        monkeypatch,
        marker={"exit_code": 0, "elapsed_s": 120},
        log='epoch 1\n{"combined_score": 1}',
    )
    result = run_kaggle(workspace, timeout_s=1800)

    assert pushed["kernel_timeout_s"] == 1800
    assert result["ok"] is True
    assert result["hours"] == 0.03
    assert result["output"].endswith('{"combined_score": 1}')


def test_a_kaggle_candidate_is_priced_in_hours_not_dollars(workspace, monkeypatch):
    """Kaggle rations hours. The zero is a fact about the backend rather than a
    missing measurement, and `hours` is what bounds a campaign here."""
    stub_kaggle(monkeypatch, marker={"exit_code": 0}, hours=1.25)
    result = run_kaggle(workspace)
    assert result["cost_usd"] == 0.0
    assert result["hours"] == 1.25
    assert result["accelerator_kind"] in ("gpu", "tpu", "cpu")


def test_a_kernel_with_no_marker_is_not_an_exit_code(workspace, monkeypatch):
    """A kernel that died before the marker cell has a real outcome and no exit
    code. Reporting one would be a number nobody measured."""
    stub_kaggle(monkeypatch, status="error", marker={}, hours=None, log="boom")
    result = run_kaggle(workspace)

    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "without recording an outcome" in result["error"]


def test_a_kernel_that_never_leaves_the_queue_is_not_a_bad_mutation(workspace, monkeypatch):
    stub_kaggle(monkeypatch, status="queued", marker={"exit_code": 0})
    result = run_kaggle(workspace, timeout_s=1)

    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "still queued" in result["error"]


def test_kaggle_candidates_count_against_the_weekly_allowance(workspace):
    """Candidates never reach `runs.jsonl`, so without the candidate fold a
    campaign would burn real GPU hours the allowance cannot see -- surfacing as
    an ordinary submission refused for hours nothing accounts for."""
    camp.append_candidate(
        {
            "campaign": "camp-1",
            "candidate_id": "camp-1-g0-c0",
            "generation": 0,
            "index": 0,
            "at": camp.now_iso(),
            "backend": "kaggle",
            "metrics": {"combined_score": 1.0},
            kaggle_quota.F_ACCELERATOR: "NvidiaTeslaP100",
            kaggle_quota.F_KIND: "gpu",
            kaggle_quota.F_ACTUAL: 3.5,
        }
    )

    pools = kaggle_quota.accelerator_hours(kind="gpu")["pools"]
    assert pools["gpu"]["total_hours"] == 3.5
    assert pools["gpu"]["runs"][0]["campaign"] == "camp-1"


def test_a_local_campaigns_candidates_do_not_touch_the_allowance(workspace):
    """The fold keys on the backend, so a local campaign -- which spends no
    Kaggle hours at all -- must not appear in the pool."""
    camp.append_candidate(
        {
            "campaign": "camp-2",
            "candidate_id": "camp-2-g0-c0",
            "generation": 0,
            "index": 0,
            "at": camp.now_iso(),
            "metrics": {"combined_score": 1.0},
            "duration_s": 3600,
        }
    )
    assert kaggle_quota.accelerator_hours(kind="gpu")["pools"] == {}


# ---------------------------------------------------------------------------
# HF Jobs: the pipeline is in the image, so the candidate needs a way in
# ---------------------------------------------------------------------------
def test_the_candidate_travels_in_an_environment_variable(workspace):
    blob = jobs_tool._candidate_blob({"initial.py": "x = 1", "evaluate.py": "print('{}')"})
    assert unpack(blob) == {"initial.py": "x = 1", "evaluate.py": "print('{}')"}


def test_the_hf_blob_is_a_function_of_its_inputs(workspace):
    files = {"initial.py": "x = 1"}
    assert jobs_tool._candidate_blob(files) == jobs_tool._candidate_blob(files)


def test_a_candidate_that_is_really_a_pipeline_is_refused(workspace):
    """This is a container environment variable, not a file. The ceiling is the
    platform's, undocumented, and discovering it by exceeding it means a job
    that fails for a reason with nothing to do with the research."""
    # Incompressible, so it cannot slip under the limit by gzipping well.
    bulk = secrets.token_hex(400_000)
    with pytest.raises(UsageError) as exc:
        jobs_tool._candidate_blob({"initial.py": bulk})
    assert "evolve block to code" in (exc.value.fix or "")


def test_an_hf_candidate_file_may_not_be_a_path(workspace):
    for bad in ("../escape.py", "/etc/passwd"):
        with pytest.raises(UsageError):
            jobs_tool._candidate_blob({bad: "x"})


def test_the_unpack_runs_before_the_command_and_gates_it(workspace):
    """A failed unpack has to fail the job. Otherwise it runs the image's *own*
    entrypoint and reports a score for the wrong program -- which would not look
    like an error, it would look like every candidate scoring the same."""
    argv = jobs_tool._candidate_command(["python", "evaluate.py"])
    assert argv[0] == "sh" and argv[1] == "-c"
    assert "&&" in argv[2]
    assert argv[2].index("tarfile") < argv[2].index("evaluate.py")


def stub_hf(monkeypatch, *, state="COMPLETED", logs="", raises=None):
    sent: dict = {}

    class FakeHub:
        def run_job(self, **kwargs):
            if raises is not None:
                raise raises
            sent.update(kwargs)
            return type("J", (), {"id": "job-1"})()

    monkeypatch.setattr(jobs_tool, "_hub", lambda: FakeHub())
    monkeypatch.setattr(jobs_tool, "_token", lambda: "t")
    monkeypatch.setattr(jobs_tool, "_ns_kwargs", lambda ns: {})
    monkeypatch.setattr(jobs_tool, "resolve_namespace", lambda *a, **k: "org")
    monkeypatch.setattr(jobs_tool, "flavor_rate", lambda flavor, cfg: 1.0)
    monkeypatch.setattr(jobs_tool, "_poll", lambda job_id, deadline, namespace=None: (state, {}))
    monkeypatch.setattr(jobs_tool, "_logs", lambda job_id, namespace=None: logs)
    monkeypatch.setattr(jobs_tool, "_actual_cost", lambda *a, **k: (0.4, None))
    return sent


def run_hf(workspace, **kwargs):
    options = {
        "candidate_id": "c1",
        "files": {"initial.py": "x = 1"},
        "command": ["python", "evaluate.py"],
        "timeout_s": 900,
        "artifacts": workspace / "artifacts",
    }
    options.update(kwargs)
    return jobs_tool.evaluate_candidate(a_submission(workspace), config_mod.load(), **options)


def test_the_blob_and_the_candidate_id_reach_the_job(workspace, monkeypatch):
    sent = stub_hf(monkeypatch, logs='{"combined_score": 2}')
    result = run_hf(workspace)

    assert jobs_tool.CANDIDATE_ENV in sent["env"]
    assert sent["env"]["GRAD_CANDIDATE"] == "c1"
    assert unpack(sent["env"][jobs_tool.CANDIDATE_ENV]) == {"initial.py": "x = 1"}
    assert result["ok"] is True
    assert result["output"].endswith('{"combined_score": 2}')


def test_an_hf_candidate_reports_a_state_not_an_invented_exit_code(workspace, monkeypatch):
    stub_hf(monkeypatch, state="ERROR", logs="Traceback")
    result = run_hf(workspace)

    assert result["ok"] is False
    assert result["exit_code"] is None, "HF reports a state; an exit code would be invented"
    assert result["job_state"] == "ERROR"
    assert result["cost_usd"] == 0.4


def test_the_job_is_bounded_where_it_runs(workspace, monkeypatch):
    sent = stub_hf(monkeypatch)
    run_hf(workspace, timeout_s=1200)
    assert sent["timeout"] == 1200


def test_a_refused_hf_submission_is_not_a_bad_mutation(workspace, monkeypatch):
    stub_hf(monkeypatch, raises=RuntimeError("402 payment required"))
    result = run_hf(workspace)

    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "could not be submitted" in result["error"]
