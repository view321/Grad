"""Credential access (HANDOFF §9).

    "With unrestricted Bash, network access, and an HF_TOKEN sitting in the
     environment, the agent *does* have general remote execution [...] So the HF
     token and the SSH keys live in Windows Credential Manager, fetched via
     keyring by gpu.py and jobs.py at the moment of use -- never exported into
     the agent's environment, never written to a file under the workspace."

The honest residual is recorded in the handoff and not papered over here: a
model determined to misbehave could import keyring itself. The threat model is
accidental or deadline-pressured spend, and this is the right bar for that --
it also means the spend ceilings guard the only path that can authenticate.
"""

from __future__ import annotations

import os
from typing import Any

from core.errors import ConfigError

SERVICE = "grad"

# Named entries, so a missing credential names itself in the error.
HF_TOKEN = "hf_token"
OPENROUTER_KEY = "openrouter_key"
VOYAGE_KEY = "voyage_key"
S2_KEY = "s2_api_key"


def _keyring() -> Any:
    try:
        import keyring  # noqa: PLC0415 - imported at point of use, on purpose
    except ImportError as exc:
        raise ConfigError(
            "the `keyring` package is not installed, so credentials cannot be read "
            "from Windows Credential Manager",
            fix="pip install keyring",
        ) from exc
    return keyring


def get(name: str, *, required: bool = True) -> str | None:
    """Read one credential. Never caches, never logs the value.

    GRAD_ALLOW_ENV_CREDENTIALS=1 permits an environment fallback; it exists for
    CI and for the first-run bootstrap, and it is off by default precisely
    because §9's argument is that the token must not be in the environment.
    """
    kr = None
    try:
        kr = _keyring()
    except ConfigError:
        if not _env_fallback_allowed():
            raise
    value = None
    if kr is not None:
        try:
            value = kr.get_password(SERVICE, name)
        except Exception as exc:  # noqa: BLE001 - backend errors vary wildly by platform
            if not _env_fallback_allowed():
                raise ConfigError(
                    f"credential store unavailable while reading {name!r}: {exc}",
                    fix="check that Windows Credential Manager is reachable for this user",
                ) from exc
    if not value and _env_fallback_allowed():
        value = os.environ.get(f"GRAD_{name.upper()}")
    if not value and required:
        raise ConfigError(
            f"credential {name!r} is not in the credential store",
            fix=f"python -m tools.jobs credential set {name}   # prompts, does not echo",
        )
    return value


def set_(name: str, value: str) -> None:
    _keyring().set_password(SERVICE, name, value)


def delete(name: str) -> None:
    try:
        _keyring().delete_password(SERVICE, name)
    except Exception:  # noqa: BLE001 - deleting a missing entry is not an error here
        pass


def present(name: str) -> bool:
    try:
        return bool(get(name, required=False))
    except ConfigError:
        return False


def status() -> dict[str, bool]:
    """Which credentials exist. Values are never returned."""
    return {n: present(n) for n in (HF_TOKEN, OPENROUTER_KEY, VOYAGE_KEY, S2_KEY)}


def _env_fallback_allowed() -> bool:
    return os.environ.get("GRAD_ALLOW_ENV_CREDENTIALS") == "1"


def scrub_environment() -> list[str]:
    """Remove credential-shaped variables from the agent's own environment.

    Called by `agent.py` at startup. ANTHROPIC_API_KEY is the important one:
    it outranks CLAUDE_CODE_OAUTH_TOKEN in the credential chain, so a stray
    export silently bills the API instead of the subscription (HANDOFF §2).
    """
    removed = []
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENROUTER_API_KEY",
        "VOYAGE_API_KEY",
        # The GRAD_* fallbacks too. They exist for CI and first-run bootstrap,
        # where no agent is running; leaving them in place under the agent would
        # hand it exactly the environment-resident credentials §9 argues must
        # not exist, and with them the ability to reach a remote without going
        # through the submitters that hold the spend ceilings.
        f"GRAD_{HF_TOKEN.upper()}",
        f"GRAD_{OPENROUTER_KEY.upper()}",
        f"GRAD_{VOYAGE_KEY.upper()}",
        f"GRAD_{S2_KEY.upper()}",
    ):
        if os.environ.pop(var, None) is not None:
            removed.append(var)
    return removed
