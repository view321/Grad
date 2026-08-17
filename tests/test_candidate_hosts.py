"""Running one evolve candidate on a real backend.

`test_evolve_remote.py` covers the driver with the backend stubbed. This is the
other side of that seam: what each backend's adapter actually does to get a
mutated program onto a machine, bound it, and read a score back.

The case being designed for is the one that matters -- a candidate is a changed
architecture or a changed optimiser, so evaluating it is a *training run* of
minutes to hours. Every property tested here follows from that: the job is
detached rather than held on a connection, it is bounded where it runs rather
than only where it is watched, and a candidate that overruns is killed rather
than left competing with its successor for the same GPU.

No network. `_ssh` and `_scp` are stubbed with a fake host that records what it
was asked to do.
"""

from __future__ import annotations

import pytest

from core.config import Host
from core.errors import GradError
from core.submission import Submission
from tools import gpu as gpu_tool


def a_host(rate: float = 2.0) -> Host:
    return Host(
        name="gpu-box",
        hostname="10.0.0.7",
        user="research",
        workdir="~/grad",
        rate_usd_per_hour=rate,
    )


def a_submission(workspace) -> Submission:
    directory = workspace / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.py").write_text("print('x')\n", encoding="utf-8")
    (directory / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        "[target]\nhost = 'gpu-box'\n"
        "[estimate]\nhours = 0.1\nrate_usd_per_hour = 1.0\n",
        encoding="utf-8",
    )
    return Submission.load(directory / "spec.toml", resolve_digest=False)


class FakeHost:
    """Records every ssh command, and answers the ones with a known shape.

    `markers` is the sequence `grad_status.json` reads return, so a test can say
    "not finished, not finished, finished" without any sleeping.
    """

    def __init__(self, *, markers=None, stdout="", stderr="", launch_pid="4242"):
        self.commands: list[str] = []
        self.copies: list[tuple[str, str]] = []
        self.markers = list(markers or ['{"state":"finished","exit_code":0}'])
        self.stdout = stdout
        self.stderr = stderr
        self.launch_pid = launch_pid

    def ssh(self, host, command, *, timeout=300.0):
        self.commands.append(command)
        if "grad_status.json" in command and command.startswith("cat "):
            return self.markers.pop(0) if len(self.markers) > 1 else self.markers[0]
        if "nohup" in command:
            return self.launch_pid
        if "stdout.log" in command and command.startswith("tail"):
            return self.stdout
        if "stderr.log" in command and command.startswith("tail"):
            return self.stderr
        return ""

    def scp(self, host, source, dest, **kwargs):
        self.copies.append((source, dest))

    def install(self, monkeypatch):
        monkeypatch.setattr(gpu_tool, "_ssh", self.ssh)
        monkeypatch.setattr(gpu_tool, "_scp", self.scp)
        monkeypatch.setattr(gpu_tool.time, "sleep", lambda _: None)
        return self

    def launched(self) -> str:
        return next(c for c in self.commands if "nohup" in c)


def run_one(workspace, fake, monkeypatch, **kwargs):
    from core import config as config_mod

    fake.install(monkeypatch)
    options = {
        "candidate_id": "camp-1-g0-c0",
        "files": {"initial.py": "print('mutated')", "evaluate.py": "print('{}')"},
        "command": ["python", "evaluate.py"],
        "timeout_s": 600,
    }
    options.update(kwargs)
    return gpu_tool.evaluate_candidate(
        a_submission(workspace), config_mod.load(), host=a_host(), **options
    )


# ---------------------------------------------------------------------------
# the shape of the run
# ---------------------------------------------------------------------------
def test_the_candidate_is_detached_not_held_on_the_connection(workspace, monkeypatch):
    """A training run is minutes to hours. A single ssh channel held open across
    that is one a NAT timeout, a sleeping laptop or a wifi handover will drop --
    and what that produces is not a failed candidate but a candidate that scored
    nothing because the network moved, which the search then selects against."""
    fake = FakeHost()
    run_one(workspace, fake, monkeypatch)

    launched = fake.launched()
    assert "nohup" in launched
    assert "grad_status.json" in launched
    # Nothing ran the evaluator inline.
    assert not any(
        "python" in c and "nohup" not in c and "base64" not in c for c in fake.commands
    )


def test_the_candidate_is_bounded_where_it_runs(workspace, monkeypatch):
    """Not only in the poll. The poll giving up ends the function; it does not
    end a detached training run, and an abandoned candidate keeps holding the
    GPU the next one is about to be measured on."""
    fake = FakeHost()
    run_one(workspace, fake, monkeypatch, timeout_s=1800)
    assert "timeout 1800" in fake.launched()


def test_the_mutated_files_are_written_after_the_pipeline_is_staged(workspace, monkeypatch):
    """Order matters: the stage copies the preflighted pipeline, and the
    candidate's own files go over the top of it."""
    fake = FakeHost()
    run_one(workspace, fake, monkeypatch)

    writes = [i for i, c in enumerate(fake.commands) if "base64 -d" in c]
    launch = fake.commands.index(fake.launched())
    assert writes, "the candidate's source never reached the host"
    assert fake.copies, "the pipeline was not staged"
    assert max(writes) < launch


def test_each_candidate_gets_its_own_directory(workspace, monkeypatch):
    """Candidate N cannot see what candidate N-1 left behind. A loop that can
    accumulate state across evaluations is one whose scores stop being
    comparable, and the failure looks like a real improvement."""
    fake = FakeHost()
    result = run_one(workspace, fake, monkeypatch, candidate_id="camp-1-g3-c2")
    assert result["where"].endswith("camp-1-g3-c2")
    assert any("camp-1-g3-c2" in c for c in fake.commands)


def test_the_directory_is_removed_afterwards(workspace, monkeypatch):
    fake = FakeHost()
    run_one(workspace, fake, monkeypatch)
    assert any(c.startswith("rm -rf") for c in fake.commands)


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------
def test_a_finished_candidate_reports_its_metrics_line_last(workspace, monkeypatch):
    """`tools/evolve.py:_metrics_from` reads the last line, and the evaluator's
    one JSON object is on stdout -- so stdout has to come after stderr, however
    odd that reads in a log."""
    fake = FakeHost(
        stdout='epoch 1\n{"combined_score": 2.5}',
        stderr="a warning about a deprecated flag",
    )
    result = run_one(workspace, fake, monkeypatch)

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["output"].strip().endswith('{"combined_score": 2.5}')
    assert "deprecated flag" in result["output"]


def test_the_exit_code_comes_from_the_marker(workspace, monkeypatch):
    """Not from scraping a log. The marker is written by the runner itself and
    is the one thing that knows how the process ended."""
    fake = FakeHost(markers=['{"state":"finished","exit_code":3}'], stdout="Traceback")
    result = run_one(workspace, fake, monkeypatch)

    assert result["ok"] is False
    assert result["exit_code"] == 3
    assert "exited 3" in result["error"]


def test_cost_is_wall_clock_against_the_host_rate(workspace, monkeypatch):
    fake = FakeHost()
    result = run_one(workspace, fake, monkeypatch)
    assert result["cost_usd"] >= 0.0
    assert result["host"] == "gpu-box"


def test_a_free_host_still_records_a_cost_of_zero(workspace, monkeypatch):
    """`rate_usd_per_hour = 0` means free to use and still ledgered."""
    from core import config as config_mod

    FakeHost().install(monkeypatch)
    result = gpu_tool.evaluate_candidate(
        a_submission(workspace),
        config_mod.load(),
        host=a_host(rate=0.0),
        candidate_id="c1",
        files={"initial.py": "x", "evaluate.py": "y"},
        command=["python", "evaluate.py"],
        timeout_s=60,
    )
    assert result["cost_usd"] == 0.0


def test_the_poll_survives_an_unreadable_marker(workspace, monkeypatch):
    """One failed read is a network hiccup, not a verdict -- and the job is
    still running on the host either way, which is the point of detaching it."""
    fake = FakeHost(markers=["not json at all", '{"state":"finished","exit_code":0}'])
    result = run_one(workspace, fake, monkeypatch)
    assert result["ok"] is True


def test_a_candidate_that_never_finishes_is_killed(workspace, monkeypatch):
    """The loop starts the next candidate the moment this returns. One left
    running competes with its own successor for the same GPU, so the next score
    would measure this one's overrun rather than the mutation."""
    # The grace is what the loop waits *past* the remote bound before deciding
    # the host has stopped answering. Shrunk here because `install` stubs
    # `time.sleep` to a no-op, so a minute of grace is a minute of real
    # busy-looping in the suite rather than a minute of waiting.
    monkeypatch.setattr(gpu_tool, "CANDIDATE_TIMEOUT_GRACE_S", 0.2)
    fake = FakeHost(markers=['{"state":"running"}'])
    result = run_one(workspace, fake, monkeypatch, timeout_s=0)

    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "did not finish" in result["error"]
    killed = [c for c in fake.commands if "kill" in c]
    assert killed, "the overrunning candidate was left running"
    # Children first: the pid `_launch` echoes is the wrapping shell's, and
    # killing only that orphans the training process on the GPU.
    assert "pkill -P 4242" in killed[0]


def test_a_host_that_refuses_the_stage_is_not_a_bad_mutation(workspace, monkeypatch):
    def refuse(host, command, *, timeout=300.0):
        raise GradError("ssh_failed", "ssh to gpu-box failed (exit 255)", exit_code=1)

    monkeypatch.setattr(gpu_tool, "_ssh", refuse)
    monkeypatch.setattr(gpu_tool, "_scp", lambda *a, **k: None)

    from core import config as config_mod

    result = gpu_tool.evaluate_candidate(
        a_submission(workspace),
        config_mod.load(),
        host=a_host(),
        candidate_id="c1",
        files={"initial.py": "x", "evaluate.py": "y"},
        command=["python", "evaluate.py"],
        timeout_s=60,
    )
    assert result["ok"] is False
    assert result["exit_code"] is None, "a transport failure must not look like an exit code"
    assert "ssh to gpu-box failed" in result["error"]


# ---------------------------------------------------------------------------
# what the host is allowed to be asked for
# ---------------------------------------------------------------------------
def test_a_candidate_file_may_not_be_a_path(workspace):
    """Written on a machine we do not own, from a name a caller supplies.
    Checked where it would be written rather than trusted to every caller."""
    for bad in ("../escape.py", "sub/dir.py", "..", ""):
        with pytest.raises(GradError):
            gpu_tool._write_remote(a_host(), "~/grad/c1", bad, "print(1)")


def test_source_travels_as_base64_not_as_a_heredoc(workspace, monkeypatch):
    """The content is a language model's Python. It can contain anything a
    heredoc terminator, a quote or a backtick means to a shell, and getting that
    wrong does not raise -- it delivers a file subtly different from the one
    that was scored."""
    fake = FakeHost().install(monkeypatch)
    nasty = "s = '''\nEOF\n`whoami`\n$(rm -rf /)\n'''\n"
    gpu_tool._write_remote(a_host(), "~/grad/c1", "initial.py", nasty)

    written = fake.commands[-1]
    assert "base64 -d" in written
    for fragment in ("EOF", "whoami", "rm -rf /"):
        assert fragment not in written
