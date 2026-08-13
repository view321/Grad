"""Importing a CLI must not touch the credential store (HANDOFF §9).

argparse setup runs at decoration time, so anything a `--help` string computes
runs on *every* invocation of that module -- including `collect` and `ceilings`,
which have no business reading Windows Credential Manager. Beyond the latency,
an unexplained credential-store access on an unrelated command is exactly the
kind of surprise the credential-isolation argument exists to avoid.
"""

from __future__ import annotations

import importlib
import sys


def test_importing_jobs_does_not_read_the_credential_store(workspace, monkeypatch):
    reads: list[str] = []

    from core import credentials

    def spy(name: str, *, required: bool = True):
        reads.append(name)
        return None

    monkeypatch.setattr(credentials, "get", spy)
    for module in ("tools.jobs", "tools.gpu", "tools.preflight", "tools.ledger", "tools.quota"):
        sys.modules.pop(module, None)
        importlib.import_module(module)

    assert reads == []


def test_credential_help_still_names_every_credential(workspace):
    from core import credentials
    from tools import jobs

    assert set(jobs.CREDENTIAL_NAMES) == set(credentials.status())
