"""HF Jobs under an organization namespace (HANDOFF-2 §17).

The trap this section exists to avoid is worth restating, because it is the
reason most of these tests are about the *handle* rather than about the submit
call:

    "`namespace` is a property of the job handle, not a submit-time parameter.
     Adding it only to `run_job` produces a job that cannot be found again:
     `inspect_job` and `fetch_job_logs` would look under the personal namespace
     and 404. The run never collects, goes stale, and then blocks *every* future
     submission through the §6 stale-run gate (exit 7). The failure appears far
     from its cause."

So the tests that matter check that the namespace is persisted and re-read, not
that it was passed once.
"""

from __future__ import annotations

import types

import pytest

from core import budget, config as config_mod, ledger_store as ls, submit as submit_lib
from core.errors import ConfigError, UpstreamError
from tests.test_gates import make_submission


class FakeHub:
    """Records what every call was asked to look at.

    Deliberately mimics the real trap: a job submitted under a namespace is
    *only* findable when the same namespace comes back, so a driver that forgets
    to thread it gets a 404 here exactly as it would in production.
    """

    def __init__(self, *, user="me", orgs=("myorg", "otherorg")):
        self.user = user
        self.orgs = list(orgs)
        self.jobs: dict[tuple[str | None, str], dict] = {}
        self.calls: list[tuple[str, str | None]] = []

    def whoami(self, token=None, **_):
        return {"name": self.user, "orgs": [{"name": o} for o in self.orgs]}

    def run_job(self, *, image, command, flavor=None, env=None, secrets=None,
                token=None, timeout=None, namespace=None, **_):
        job_id = f"job-{len(self.jobs)}"
        self.jobs[(namespace, job_id)] = {"id": job_id, "status": {"stage": "COMPLETED"}}
        self.calls.append(("run_job", namespace))
        return types.SimpleNamespace(id=job_id)

    def inspect_job(self, *, job_id, token=None, namespace=None, **_):
        self.calls.append(("inspect_job", namespace))
        found = self.jobs.get((namespace, job_id))
        if found is None:
            raise RuntimeError(f"404: no job {job_id} under namespace {namespace!r}")
        return types.SimpleNamespace(
            id=job_id, status=types.SimpleNamespace(stage="COMPLETED"),
            started_at=None, created_at=None, ended_at=None,
        )

    def fetch_job_logs(self, *, job_id, token=None, namespace=None, **_):
        self.calls.append(("fetch_job_logs", namespace))
        if (namespace, job_id) not in self.jobs:
            raise RuntimeError(f"404: no job {job_id} under namespace {namespace!r}")
        return ["line one"]


@pytest.fixture
def hub(monkeypatch):
    from tools import jobs

    fake = FakeHub()
    monkeypatch.setattr(jobs, "_hub", lambda: fake)
    monkeypatch.setattr(jobs, "_token", lambda: "tok")
    return fake


def cfg():
    return config_mod.load(reload=True)


# ---------------------------------------------------------------------------
# resolution order
# ---------------------------------------------------------------------------
def test_resolution_order_flag_beats_everything(workspace, hub):
    from tools import jobs

    sub = make_submission(workspace)
    sub.target["namespace"] = "from-spec"
    budget.create("proj-1", title="t", budget={}, payer="hf:from-project")
    assert jobs.resolve_namespace("from-flag", sub, cfg(), "proj-1") == "from-flag"


def test_resolution_order_spec_beats_project(workspace, hub):
    from tools import jobs

    sub = make_submission(workspace)
    sub.target["namespace"] = "from-spec"
    budget.create("proj-1", title="t", budget={}, payer="hf:from-project")
    assert jobs.resolve_namespace(None, sub, cfg(), "proj-1") == "from-spec"


def test_resolution_order_project_payer_is_used(workspace, hub):
    from tools import jobs

    sub = make_submission(workspace)
    budget.create("proj-1", title="t", budget={}, payer="hf:from-project")
    assert jobs.resolve_namespace(None, sub, cfg(), "proj-1") == "from-project"


def test_resolution_falls_through_to_personal(workspace, hub):
    from tools import jobs

    sub = make_submission(workspace)
    assert jobs.resolve_namespace(None, sub, cfg(), None) is None


# ---------------------------------------------------------------------------
# membership validation
# ---------------------------------------------------------------------------
def test_a_namespace_the_token_cannot_act_for_is_refused(workspace, hub):
    from tools import jobs

    with pytest.raises(ConfigError) as exc:
        jobs.validate_namespace("not-my-org", "tok")
    assert "cannot act for" in str(exc.value)
    assert "myorg" in str(exc.value)


def test_an_org_the_token_belongs_to_passes(workspace, hub):
    from tools import jobs

    identity = jobs.validate_namespace("myorg", "tok")
    assert identity["user"] == "me"
    assert "myorg" in identity["orgs"]


def test_the_users_own_namespace_passes(workspace, hub):
    from tools import jobs

    assert jobs.validate_namespace("me", "tok")["namespace"] == "me"


def test_a_whoami_failure_is_an_upstream_error(workspace, monkeypatch):
    from tools import jobs

    class Broken(FakeHub):
        def whoami(self, token=None, **_):
            raise RuntimeError("network down")

    monkeypatch.setattr(jobs, "_hub", lambda: Broken())
    with pytest.raises(UpstreamError):
        jobs.validate_namespace("myorg", "tok")


def test_validation_runs_before_any_record_exists(workspace, hub, monkeypatch):
    """"a configuration problem must not leave a phantom estimate sitting on the
    ceiling." A bad namespace must produce no run record at all."""
    import argparse

    from tools import jobs
    from tests.test_gates import make_expectation, pass_preflight

    sub = make_submission(workspace)
    pass_preflight(sub)
    expectation_id = make_expectation()
    before = len(ls.runs())

    args = argparse.Namespace(
        spec=str(sub.spec_path), expect=expectation_id, overrides=[], flavor=None,
        task=None, project=None, namespace="not-my-org", smoke=False,
        no_digest=True, json=True,
    )
    with pytest.raises(ConfigError) as exc:
        jobs.cmd_submit(args)
    assert "cannot act for" in str(exc.value)
    # The four gates passed, so nothing but the namespace stopped this -- and
    # still no phantom estimate landed on the ceiling.
    assert len(ls.runs()) == before


# ---------------------------------------------------------------------------
# the handle -- the part that actually matters
# ---------------------------------------------------------------------------
def test_the_namespace_is_persisted_onto_the_handle(workspace, hub):
    """Not merely passed to run_job. This is the whole point of §17."""
    submit_lib.attach_handle("run-1", {"job_id": "job-0", "flavor": "t4-small", "namespace": "myorg"})
    events = [e for e in ls.runs_events() if e.get("type") == "run_handle"]
    assert events[-1]["handle"]["namespace"] == "myorg"


def test_collect_reads_the_namespace_from_the_handle(workspace, hub, monkeypatch):
    """A job submitted to an org is collectable from that org -- even if the
    config or the current project changed in between."""
    import argparse

    from tools import jobs

    ls.append_run_event(
        {
            "type": ls.T_RUN_SUBMITTED, "id": "run-1", "status": "in_flight",
            "submitted_at": ls.now_iso(), "project": "proj-1", "estimate_usd": 1.0,
            "metrics_file": "metrics.json", "expectation_id": None,
            "target": {"flavor": "t4-small", "namespace": "myorg"},
        }
    )
    hub.jobs[("myorg", "job-0")] = {"id": "job-0"}
    submit_lib.attach_handle("run-1", {"job_id": "job-0", "flavor": "t4-small", "namespace": "myorg"})

    args = argparse.Namespace(run_id="run-1", wait=False, timeout=1, json=True)
    jobs.cmd_collect(args)

    looked_under = {ns for call, ns in hub.calls if call in ("inspect_job", "fetch_job_logs")}
    assert looked_under == {"myorg"}, "collect must look under the submitted namespace"


def test_a_personal_job_passes_no_namespace_at_all(workspace, hub):
    """Omitted rather than passed as None, so an older huggingface_hub without
    the parameter still works for personal jobs."""
    from tools import jobs

    assert jobs._ns_kwargs(None) == {}
    assert jobs._ns_kwargs("myorg") == {"namespace": "myorg"}


def test_a_hub_without_namespace_support_refuses_loudly(workspace, monkeypatch):
    """Silently dropping the namespace would produce exactly the uncollectable
    job this section exists to prevent."""
    from tools import jobs

    class Old:
        def run_job(self, *, image, command, token=None):  # no namespace
            ...

        def inspect_job(self, *, job_id, token=None):
            ...

        def fetch_job_logs(self, *, job_id, token=None):
            ...

    monkeypatch.setattr(jobs, "_hub", lambda: Old())
    with pytest.raises(ConfigError) as exc:
        jobs._ns_kwargs("myorg")
    assert "could not be collected" in str(exc.value)


# ---------------------------------------------------------------------------
# the smoke/submit mismatch
# ---------------------------------------------------------------------------
def test_a_smoke_namespace_mismatch_warns_rather_than_refuses(workspace, hub):
    """"Warn, not refuse -- consistent with how `target` and `flavor` already
    behave." The hash excludes `target`, so namespace follows the same rule."""
    from tools import jobs, preflight

    sub = make_submission(workspace)
    preflight.record_check_result(sub.hash(), "smoke", {"ok": True, "namespace": None})

    warnings = jobs._namespace_warnings(sub, "myorg")
    assert len(warnings) == 1
    assert "personal" in warnings[0] and "myorg" in warnings[0]


def test_no_warning_when_the_namespaces_agree(workspace, hub):
    from tools import jobs, preflight

    sub = make_submission(workspace)
    preflight.record_check_result(sub.hash(), "smoke", {"ok": True, "namespace": "myorg"})
    assert jobs._namespace_warnings(sub, "myorg") == []


def test_no_warning_when_smoke_predates_the_namespace_field(workspace, hub):
    """An older preflight record has no `namespace` key; that is not a mismatch."""
    from tools import jobs, preflight

    sub = make_submission(workspace)
    preflight.record_check_result(sub.hash(), "smoke", {"ok": True})
    assert jobs._namespace_warnings(sub, "myorg") == []


def test_the_namespace_is_not_part_of_the_submission_hash(workspace):
    """Consistent with `flavor`: the hash deliberately excludes `target`."""
    sub = make_submission(workspace)
    before = sub.hash()
    sub.target["namespace"] = "myorg"
    assert sub.hash() == before
