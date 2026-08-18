"""What happens when the credential store will not open.

Every test here describes the same machine: a headless Linux host, where
`keyring` is installed and imports cleanly and every read raises because there
is no D-Bus session and therefore no Secret Service to hold anything. That is
not an exotic configuration -- it is a GPU box reached over SSH, which is where
this software is most likely to run and the one place it had never been run.

The bug these are the regression tests for: `credentials.get` raised on a store
it could not open even when the caller had said `required=False`, so three
optional lookups (`s2_api_key`, `asta_api_key`, `context7_key`) turned "no
keyring here" into a hard failure of anonymous retrieval. `core/http.py` caught
it at one of the three call sites, which is why it looked handled.
"""

from __future__ import annotations

import pytest

from core import credentials
from core.errors import ConfigError

keyring = pytest.importorskip("keyring", reason="the remote extra is not installed")
keyring_errors = pytest.importorskip("keyring.errors")


@pytest.fixture
def dead_store(monkeypatch):
    """A keyring whose backend refuses every operation."""

    def no_backend(*_a, **_k):
        raise keyring_errors.NoKeyringError("No recommended backend was available.")

    monkeypatch.setattr(keyring, "get_password", no_backend)
    monkeypatch.setattr(keyring, "set_password", no_backend)
    # The environment fallback is off by default and these tests rely on that
    # being the state under test -- a runner that exported it would otherwise
    # make every one of them pass for the wrong reason.
    monkeypatch.delenv("GRAD_ALLOW_ENV_CREDENTIALS", raising=False)
    return no_backend


def test_an_optional_credential_is_absent_rather_than_an_error(dead_store):
    """The whole bug, in one line. A caller that said it can proceed without the
    value cannot act on the difference between "not stored" and "store will not
    open", so it is not offered one."""
    assert credentials.get(credentials.S2_KEY, required=False) is None


def test_every_optional_lookup_degrades_the_same_way(dead_store):
    """Named individually because the original fix was applied to one of them
    and the other two were the ones that broke."""
    for name in (credentials.S2_KEY, credentials.ASTA_KEY, credentials.CONTEXT7_KEY):
        assert credentials.get(name, required=False) is None, name


def test_a_required_credential_still_refuses_and_says_where_to_look(dead_store):
    """Degrading here would be the opposite mistake: a token the caller cannot
    proceed without, reported as merely missing, sends the user to `credential
    set` -- which will fail in the same way and not say why."""
    with pytest.raises(ConfigError) as exc:
        credentials.get(credentials.HF_TOKEN)
    assert "store unavailable" in str(exc.value)
    assert exc.value.fix, "a refusal with no fix is the one nobody can act on"


def test_the_fix_text_names_this_platforms_store(dead_store):
    """"Check that Windows Credential Manager is reachable" is not advice on a
    Linux box; it is a instruction to go and look at something that is not
    there."""
    import os

    fix = credentials.store_fix()
    if os.name == "nt":
        assert "Credential Manager" in fix
    else:
        # The headless answer has to be present, because headless is the case.
        assert "GRAD_ALLOW_ENV_CREDENTIALS" in fix
    assert credentials.store_name()


def test_present_reports_false_rather_than_raising(dead_store):
    assert credentials.present(credentials.VOYAGE_KEY) is False
    assert set(credentials.status()) == set(credentials.ALL)
    assert not any(credentials.status().values())


def test_storing_reports_a_config_error_rather_than_a_backend_traceback(dead_store):
    """`credential set` is the step immediately after installing, and a raw
    `NoKeyringError` at that moment reads as "this software is broken" rather
    than "this host needs a backend"."""
    with pytest.raises(ConfigError) as exc:
        credentials.set_(credentials.VOYAGE_KEY, "value")
    assert exc.value.fix


def test_the_environment_route_still_works_when_it_is_allowed(dead_store, monkeypatch):
    """The documented way to authenticate on a host with no keyring at all. If
    this stops working there is no path left on such a machine."""
    monkeypatch.setenv("GRAD_ALLOW_ENV_CREDENTIALS", "1")
    monkeypatch.setenv("GRAD_VOYAGE_KEY", "from-the-environment")
    assert credentials.get(credentials.VOYAGE_KEY) == "from-the-environment"
    assert credentials.present(credentials.VOYAGE_KEY) is True


def test_a_missing_keyring_package_keeps_its_own_advice(monkeypatch):
    """`pip install keyring` is a better fix than "install a backend" when the
    package itself is what is absent, so that error is passed through rather than
    wrapped in the vaguer one."""
    monkeypatch.setattr(credentials, "_keyring", _raises_import_error)
    monkeypatch.delenv("GRAD_ALLOW_ENV_CREDENTIALS", raising=False)
    assert credentials.get(credentials.S2_KEY, required=False) is None
    with pytest.raises(ConfigError) as exc:
        credentials.get(credentials.HF_TOKEN)
    assert "pip install keyring" in (exc.value.fix or "")


def _raises_import_error():
    raise ConfigError("the `keyring` package is not installed", fix="pip install keyring")
