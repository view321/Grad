"""Evolve phase 2: candidates evaluated on real hardware.

Phase 1 was local-only on purpose -- prove the campaign records, the sub-run
bookkeeping and the budget gate while the blast radius is zero -- and these are
the tests for the thing that was being deferred until then. Two properties carry
most of the weight:

  * **the gate holds before generation 0.** A search is a loop with no human in
    it, so the environment it lands in must already have a complete, passing
    preflight *including the smoke run*, checked once, before anything is spent.
  * **a candidate still never becomes a run.** §23 item 4 is the reason
    candidates live in `candidates.jsonl`, and it would be undone the moment the
    search left this machine if each remote evaluation wrote a ledger row.

Nothing here opens an ssh connection. `tools/gpu.py:evaluate_candidate` is the
seam, and it is stubbed -- the point under test is what the driver does with
what a host says, not `ssh` itself.
"""

from __future__ import annotations

import pytest

from core import campaign as camp, config as config_mod, jsonl, ledger_store as ls, paths
from core.errors import GateRefusal, GradError, UsageError
from core.submission import Submission
from tools import evolve

from test_evolve import FakeMutator, make_expectation, run_args, scaffold


# ---------------------------------------------------------------------------
# fixtures of the world a remote campaign needs
# ---------------------------------------------------------------------------
def remote_spec(
    workspace,
    *,
    host: str = "gpu-box",
    hours: float = 0.1,
    accelerator: str | None = None,
    flavor: str | None = None,
) -> Submission:
    """A pipeline spec, plus the config a campaign on it needs.

    One helper for all three backends: the spec names a host, and optionally an
    accelerator or a flavor, so a test can point the same pipeline at whichever
    backend it is about.
    """
    directory = workspace / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.py").write_text("print('x')\n", encoding="utf-8")
    target = [f"host = '{host}'"]
    if accelerator:
        target.append(f"accelerator = '{accelerator}'")
    if flavor:
        target.append(f"flavor = '{flavor}'")
    (directory / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        "[target]\n" + "\n".join(target) + "\n"
        f"[estimate]\nhours = {hours}\nrate_usd_per_hour = 1.0\n",
        encoding="utf-8",
    )
    config_path = workspace / "config" / "grad.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"[hosts.{host}]\nhostname = '10.0.0.7'\nuser = 'research'\n"
        "workdir = '~/grad'\nrate_usd_per_hour = 2.0\n"
        "\n[hf]\ndefault_flavor = 'a10g-small'\n"
        "\n[hf.flavor_rates]\n'a10g-small' = 1.0\n"
        "\n[kaggle]\ndefault_accelerator = 'NvidiaTeslaP100'\n"
        "\n[kaggle.quota]\ngpu_hours_per_week = 30.0\nmax_session_hours = 12.0\n"
        "\n[kaggle.accelerators]\nNvidiaTeslaP100 = 'gpu'\n",
        encoding="utf-8",
    )
    config_mod._cache.clear()
    return Submission.load(directory / "spec.toml", resolve_digest=False)


def resolve_target(workspace, sub, **overrides):
    """Just the gate and the resolution, without running a campaign."""
    args = campaign_args(workspace, sub, **overrides)
    return evolve._remote_target(args, config_mod.load())


def hostless_spec(workspace) -> Submission:
    directory = workspace / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.py").write_text("print('x')\n", encoding="utf-8")
    (directory / "spec.toml").write_text(
        "entrypoint = 'train.py'\nimage = 'org/img@sha256:aaaa'\n"
        "[estimate]\nhours = 0.1\nrate_usd_per_hour = 1.0\n",
        encoding="utf-8",
    )
    config_mod._cache.clear()
    return Submission.load(directory / "spec.toml", resolve_digest=False)


def preflight(sub: Submission, **checks: bool) -> None:
    """Write a preflight record for this exact submission hash."""
    results = {"tests": True, "dry_run": True, "smoke": True}
    results.update(checks)
    jsonl.write_json(
        paths.preflight_record(sub.hash()),
        {
            "submission_hash": sub.hash(),
            "verified_at": ls.now_iso(),
            "checks": {name: {"ok": ok} for name, ok in results.items()},
        },
    )


def missing_smoke(sub: Submission) -> None:
    jsonl.write_json(
        paths.preflight_record(sub.hash()),
        {
            "submission_hash": sub.hash(),
            "verified_at": ls.now_iso(),
            "checks": {"tests": {"ok": True}, "dry_run": {"ok": True}},
        },
    )


def host_answers(monkeypatch, answer):
    """Stand in for the whole ssh side. `answer(candidate_id) -> dict`.

    Returns the list of calls, so a test can assert on what actually reached the
    host rather than only on what came back.
    """
    from tools import gpu as gpu_tool

    seen: list[dict] = []

    def evaluate_candidate(sub, cfg, *, candidate_id, files, command, timeout_s, host=None):
        seen.append(
            {
                "candidate": candidate_id,
                "files": dict(files),
                "command": list(command),
                "timeout_s": timeout_s,
            }
        )
        return answer(candidate_id)

    monkeypatch.setattr(gpu_tool, "evaluate_candidate", evaluate_candidate)
    return seen


def scored(candidate_id: str, *, score: float = 1.5, cost: float = 0.05) -> dict:
    return {
        "ok": True,
        "exit_code": 0,
        "cost_usd": cost,
        "host": "gpu-box",
        "where": f"gpu-box:~/grad/{candidate_id}",
        "error": None,
        "output": f'training noise\n{{"combined_score": {score}, "abs_error": 2}}\nEXIT:0',
    }


def campaign_args(workspace, sub, **overrides):
    base = dict(
        remote="ssh",
        remote_spec=str(sub.spec_path),
        generations=1,
        population=2,
    )
    base.update(overrides)
    return run_args(scaffold(workspace), make_expectation(), **base)


def drive(workspace, sub, monkeypatch, **overrides):
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    return evolve.cmd_run(campaign_args(workspace, sub, **overrides))


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_a_remote_campaign_refuses_without_a_preflight(workspace):
    """The environment is proven before generation 0, not rediscovered forty
    times at a dollar apiece."""
    sub = remote_spec(workspace)
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(campaign_args(workspace, sub))
    assert exc.value.code == "preflight_missing"
    assert "preflight run" in (exc.value.fix or "")


def test_a_remote_campaign_refuses_without_a_smoke_run(workspace):
    """Specifically smoke, and specifically not read from `[preflight] checks`:
    a machine configured without it would otherwise let a campaign put every
    candidate it has on hardware nothing had ever run one step on."""
    sub = remote_spec(workspace)
    missing_smoke(sub)
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(campaign_args(workspace, sub))
    assert "smoke missing" in exc.value.message


def test_a_failing_smoke_is_not_a_passing_preflight(workspace):
    sub = remote_spec(workspace)
    preflight(sub, smoke=False)
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(campaign_args(workspace, sub))
    assert "smoke failed" in exc.value.message


def test_a_preflight_for_a_different_submission_does_not_transfer(workspace):
    """The record is keyed by submission hash, so editing the pipeline after
    proving it puts the campaign back behind the gate."""
    sub = remote_spec(workspace)
    preflight(sub)
    (workspace / "pipeline" / "train.py").write_text("print('changed')\n", encoding="utf-8")
    config_mod._cache.clear()

    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(campaign_args(workspace, sub))
    assert exc.value.code == "preflight_missing"


def test_the_gate_runs_before_the_expectation_is_bound(workspace):
    """A configuration refusal must not cost an expectation. They are
    single-use by §7, so being refused for a missing preflight would otherwise
    burn one and force a re-mint before anything could be retried."""
    sub = remote_spec(workspace)
    expectation = make_expectation()
    args = run_args(
        scaffold(workspace), expectation, remote="ssh", remote_spec=str(sub.spec_path)
    )

    with pytest.raises(GateRefusal):
        evolve.cmd_run(args)
    assert expectation not in ls.consumed_expectation_ids()


def test_a_spec_with_no_host_is_a_configuration_error(workspace):
    sub = hostless_spec(workspace)
    preflight(sub)
    with pytest.raises(GradError) as exc:
        evolve.cmd_run(campaign_args(workspace, sub))
    assert "no [target] host" in exc.value.message


def test_an_unknown_host_is_refused_before_generation_zero(workspace):
    """Not forty evaluations in. An unknown host is a configuration error and
    the campaign gate is where configuration errors belong."""
    sub = remote_spec(workspace, host="gpu-box")
    preflight(sub)
    (workspace / "config" / "grad.toml").write_text("", encoding="utf-8")
    config_mod._cache.clear()

    with pytest.raises(GradError):
        evolve.cmd_run(campaign_args(workspace, sub))


def test_an_hf_campaign_refuses_a_flavor_nothing_prices(workspace):
    """An unpriced flavor makes the campaign's projected cost a fiction, and the
    campaign budget gate is the only thing between a search and an allocation.
    Refused before generation 0 rather than booked as free."""
    sub = remote_spec(workspace, flavor="h200-quantum")
    preflight(sub)
    with pytest.raises(GradError) as exc:
        evolve.cmd_run(campaign_args(workspace, sub, remote="hf_jobs"))
    assert "flavor_rates" in exc.value.message


def test_a_kaggle_campaign_refuses_what_will_not_fit_the_week(workspace):
    """Kaggle rations hours, not money, so a campaign priced at zero sails
    through the dollar gate and would then spend the whole weekly allowance --
    surfacing as an ordinary submission refused for hours nothing accounts for."""
    sub = remote_spec(workspace, accelerator="NvidiaTeslaP100", hours=4.0)
    preflight(sub)
    with pytest.raises(GateRefusal) as exc:
        # 10 generations x 4 candidates x 4h = 160h against a 30h week.
        evolve.cmd_run(
            campaign_args(workspace, sub, remote="kaggle", generations=10, population=4)
        )
    assert exc.value.code == "quota_weekly"
    assert "does not fit the week" in exc.value.message
    # The campaign's own shape, because "this run estimates 160h" is a confusing
    # way to describe forty four-hour candidates.
    assert "40 candidates at 4.00h each" in exc.value.message


def test_the_session_cap_is_asked_about_one_candidate_not_the_campaign(workspace):
    """The two Kaggle ceilings take different numbers. Handing the session cap
    the campaign total would refuse an ordinary search of twenty one-hour
    candidates for exceeding a twelve-hour session."""
    sub = remote_spec(workspace, accelerator="NvidiaTeslaP100", hours=1.0)
    preflight(sub)

    # 20 candidates x 1h = 20h: inside the 30h week, and each one inside a 12h
    # session. A gate that conflated the two would refuse this.
    target = resolve_target(workspace, sub, remote="kaggle", generations=5, population=4)
    assert target["backend"] == "kaggle"
    assert target["accelerator_kind"] == "gpu"


def test_an_hf_campaign_resolves_and_prices_its_flavor(workspace):
    sub = remote_spec(workspace, flavor="a10g-small")
    preflight(sub)
    target = resolve_target(workspace, sub, remote="hf_jobs")
    assert target["flavor"] == "a10g-small"
    assert target["rate_usd_per_hour"] == 1.0


def test_a_single_candidate_past_the_session_cap_is_refused(workspace):
    """Kaggle stops the kernel at the cap and hands back whatever it wrote, so a
    candidate estimated past it has already been decided to fail."""
    sub = remote_spec(workspace, accelerator="NvidiaTeslaP100", hours=20.0)
    preflight(sub)
    with pytest.raises(GateRefusal) as exc:
        evolve.cmd_run(
            campaign_args(workspace, sub, remote="kaggle", generations=1, population=1)
        )
    # Passed through from `kaggle_quota` untouched: its message already names
    # the cap and says what happens to a kernel that hits it.
    assert exc.value.code == "quota_session"
    assert "single GPU session" in exc.value.message


def test_a_backend_without_a_spec_is_refused(workspace):
    with pytest.raises(UsageError) as exc:
        evolve.cmd_run(run_args(scaffold(workspace), make_expectation(), remote="ssh"))
    assert "--remote-spec" in (exc.value.fix or "")


def test_a_spec_without_a_backend_is_refused(workspace):
    """The mirror. A flag that names an environment and is then silently
    ignored is worse than one that refuses."""
    with pytest.raises(UsageError) as exc:
        evolve.cmd_run(
            run_args(scaffold(workspace), make_expectation(), remote_spec="pipeline/spec.toml")
        )
    assert "no --remote backend" in str(exc.value)


# ---------------------------------------------------------------------------
# what happens on the host
# ---------------------------------------------------------------------------
def test_the_mutated_source_is_what_reaches_the_host(workspace, monkeypatch):
    """The whole point: the *candidate's* program runs inside the environment
    the preflight proved."""
    sub = remote_spec(workspace)
    preflight(sub)
    seen = host_answers(monkeypatch, scored)

    drive(workspace, sub, monkeypatch)

    assert seen, "nothing reached the host"
    for call in seen:
        assert set(call["files"]) == {"initial.py", "evaluate.py"}
        assert "EVOLVE-BLOCK-START" in call["files"]["initial.py"]
        assert call["command"] == ["python", "evaluate.py"]


def test_a_remote_campaign_scores_what_the_host_printed(workspace, monkeypatch):
    sub = remote_spec(workspace)
    preflight(sub)
    host_answers(monkeypatch, scored)

    result = drive(workspace, sub, monkeypatch)
    candidates = [c for c in camp.candidates(result["campaign"]) if c.get("metrics")]

    assert candidates, "no candidate was scored from the host's output"
    assert candidates[0]["metrics"]["combined_score"] == 1.5
    assert candidates[0]["ran_on"].startswith("gpu-box:")
    assert candidates[0]["backend"] == "ssh"


def test_the_cost_recorded_is_measured_not_estimated(workspace, monkeypatch):
    """The estimate is what the budget gate projects with; this is what was
    actually spent, and on a remote campaign the two are not the same number."""
    sub = remote_spec(workspace)
    preflight(sub)
    host_answers(monkeypatch, lambda cid: scored(cid, cost=0.07))

    result = drive(workspace, sub, monkeypatch, estimate_per_candidate_usd=0.5)
    for candidate in camp.candidates(result["campaign"]):
        assert candidate["cost_usd"] == 0.07


def test_a_host_that_could_not_run_it_is_not_a_bad_mutation(workspace, monkeypatch):
    """A search that reads 'the host refused the connection' as 'this idea
    scored nothing' quietly selects against whatever was being proposed when
    the network wobbled."""
    sub = remote_spec(workspace)
    preflight(sub)
    host_answers(
        monkeypatch,
        lambda cid: {
            "ok": False,
            "exit_code": None,
            "cost_usd": 0.0,
            "host": "gpu-box",
            "where": f"gpu-box:~/grad/{cid}",
            "output": "",
            "error": "ssh to gpu-box failed (exit 255): connection refused",
        },
    )

    result = drive(workspace, sub, monkeypatch)
    candidates = camp.candidates(result["campaign"])
    assert candidates
    for candidate in candidates:
        assert candidate["metrics"] is None
        assert candidate["skipped"] is True
        assert "could not run it" in candidate["error"]


def test_a_candidate_that_crashed_is_recorded_with_its_reason(workspace, monkeypatch):
    sub = remote_spec(workspace)
    preflight(sub)
    host_answers(
        monkeypatch,
        lambda cid: {
            "ok": False,
            "exit_code": 1,
            "cost_usd": 0.01,
            "host": "gpu-box",
            "where": f"gpu-box:~/grad/{cid}",
            "error": "the candidate exited 1 on gpu-box",
            "output": "Traceback (most recent call last):\nValueError: nope\nEXIT:1",
        },
    )

    result = drive(workspace, sub, monkeypatch)
    candidates = camp.candidates(result["campaign"])
    assert all(c["metrics"] is None for c in candidates)
    assert any("did not print a JSON object" in (c.get("error") or "") for c in candidates)
    # Not skipped: it ran, it cost money, and it failed. That is a fact about
    # the mutation and the next generation's prompt should see it.
    assert not any(c.get("skipped") for c in candidates)


def test_a_remote_candidate_still_never_becomes_a_run(workspace, monkeypatch):
    """§23 item 4 holds when the search leaves this machine."""
    sub = remote_spec(workspace)
    preflight(sub)
    host_answers(monkeypatch, scored)

    before = len(ls.runs())
    drive(workspace, sub, monkeypatch)
    assert len(ls.runs()) == before


def test_the_campaign_record_says_where_it_ran(workspace, monkeypatch):
    sub = remote_spec(workspace)
    preflight(sub)
    host_answers(monkeypatch, scored)

    result = drive(workspace, sub, monkeypatch)
    record = camp.campaign(result["campaign"])
    assert record["mode"] == "remote"
    assert record["backend"] == "ssh"
    assert record["host"] == "gpu-box"
    assert record["submission_hash"] == sub.hash()
    assert "sub" not in record, "a live Submission object got into a JSON record"


def test_a_local_campaign_still_says_it_is_local(workspace, monkeypatch):
    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(
        run_args(scaffold(workspace), make_expectation(), generations=1, population=2)
    )
    record = camp.campaign(result["campaign"])
    assert record["mode"] == "local"
    assert "host" not in record


def test_each_backend_is_handed_the_candidate_by_its_own_adapter(workspace, monkeypatch):
    """One dispatcher, three modules. The adapters answer the same question and
    return the same shape, but they get there differently enough that a shared
    implementation would be a lie -- and this is the wire that would otherwise
    send every campaign to whichever one happened to be first."""
    calls: list[str] = []

    def spy(name):
        def evaluate_candidate(sub, cfg, **kwargs):
            calls.append(name)
            return {
                "ok": True, "exit_code": 0, "cost_usd": 0.0, "error": None,
                "where": f"{name}:x", "output": '{"combined_score": 1}',
            }

        return evaluate_candidate

    from tools import gpu as gpu_tool, jobs as jobs_tool, kaggle as kaggle_tool

    monkeypatch.setattr(gpu_tool, "evaluate_candidate", spy("ssh"))
    monkeypatch.setattr(kaggle_tool, "evaluate_candidate", spy("kaggle"))
    monkeypatch.setattr(jobs_tool, "evaluate_candidate", spy("hf_jobs"))

    for backend, sub in (
        ("ssh", remote_spec(workspace)),
        ("kaggle", remote_spec(workspace, accelerator="NvidiaTeslaP100", hours=0.5)),
        ("hf_jobs", remote_spec(workspace, flavor="a10g-small")),
    ):
        calls.clear()
        preflight(sub)
        evolve._run_on_backend(
            resolve_target(workspace, sub, remote=backend),
            config_mod.load(),
            candidate_id="c1",
            files={"initial.py": "x", "evaluate.py": "y"},
            timeout_s=60,
            artifacts=workspace / "artifacts",
        )
        assert calls == [backend]


def test_a_kaggle_campaign_records_the_hours_the_quota_fold_reads(workspace, monkeypatch):
    """The candidate row is the only place those hours exist -- they never reach
    `runs.jsonl` -- so the field names have to be the ones the fold looks for."""
    from core import kaggle_quota
    from tools import kaggle as kaggle_tool

    sub = remote_spec(workspace, accelerator="NvidiaTeslaP100", hours=0.5)
    preflight(sub)
    monkeypatch.setattr(
        kaggle_tool,
        "evaluate_candidate",
        lambda sub_, cfg, **kw: {
            "ok": True, "exit_code": 0, "cost_usd": 0.0, "error": None,
            "where": "kaggle:x", "output": '{"combined_score": 1}',
            "hours": 0.4, "accelerator": "NvidiaTeslaP100", "accelerator_kind": "gpu",
        },
    )

    monkeypatch.setattr(evolve, "_make_mutator", lambda *a, **k: FakeMutator())
    result = evolve.cmd_run(campaign_args(workspace, sub, remote="kaggle"))

    rows = camp.candidates(result["campaign"])
    assert rows and all(r[kaggle_quota.F_ACTUAL] == 0.4 for r in rows)
    pools = kaggle_quota.accelerator_hours(kind="gpu")["pools"]
    assert pools["gpu"]["total_hours"] == pytest.approx(0.4 * len(rows))


def test_the_remote_timeout_defaults_to_the_candidate_timeout(workspace, monkeypatch):
    sub = remote_spec(workspace)
    preflight(sub)
    seen = host_answers(monkeypatch, scored)

    drive(workspace, sub, monkeypatch, timeout_s=45, remote_timeout_s=0)
    assert seen and all(call["timeout_s"] == 45 for call in seen)


def test_the_remote_timeout_can_be_set_apart_from_the_local_one(workspace, monkeypatch):
    """A remote evaluation is not bounded by the same number as a local one --
    the host is a different machine with different hardware."""
    sub = remote_spec(workspace)
    preflight(sub)
    seen = host_answers(monkeypatch, scored)

    drive(workspace, sub, monkeypatch, timeout_s=45, remote_timeout_s=1800)
    assert seen and all(call["timeout_s"] == 1800 for call in seen)


# ---------------------------------------------------------------------------
# reading metrics out of a combined stream
# ---------------------------------------------------------------------------
def test_metrics_are_read_from_the_last_line_only():
    """The same rule the local path uses, deliberately. A search whose metric
    can be found anywhere in the output is a search that can be fed a number by
    a log line."""
    metrics, problem = evolve._metrics_from(
        '{"combined_score": 99}\nreal work happened\n{"combined_score": 1}\nEXIT:0'
    )
    assert problem is None
    assert metrics["combined_score"] == 1


def test_the_exit_marker_is_not_mistaken_for_output():
    metrics, problem = evolve._metrics_from('{"combined_score": 1}\nEXIT:0\n')
    assert problem is None
    assert metrics["combined_score"] == 1


def test_silence_from_the_host_is_a_named_failure():
    metrics, problem = evolve._metrics_from("EXIT:0\n")
    assert metrics is None
    assert "printed nothing" in problem


def test_a_traceback_is_reported_with_its_last_line():
    metrics, problem = evolve._metrics_from("Traceback...\nValueError: nope\nEXIT:1")
    assert metrics is None
    assert "ValueError: nope" in problem


def test_metrics_without_a_combined_score_are_still_refused_remotely():
    """The metric contract does not relax because the number arrived over ssh."""
    metrics, problem = evolve._metrics_from('{"abs_error": 2}\nEXIT:0')
    assert problem is not None


# ---------------------------------------------------------------------------
# the ssh adapter's own refusals
# ---------------------------------------------------------------------------
def test_a_candidate_file_may_not_be_a_path(workspace):
    """It is written on a machine we do not own, from a name this module's
    callers supply. Checked where it would be written rather than trusted to
    every future caller."""
    from core.config import Host
    from tools import gpu as gpu_tool

    host = Host(
        name="gpu-box", hostname="10.0.0.7", user="research", workdir="~/grad",
        rate_usd_per_hour=0.0,
    )
    for bad in ("../escape.py", "sub/dir.py", ".."):
        with pytest.raises(GradError) as exc:
            gpu_tool._write_remote(host, "~/grad/c1", bad, "print(1)")
        assert "plain name" in (exc.value.fix or "") or "refusing to write" in exc.value.message
