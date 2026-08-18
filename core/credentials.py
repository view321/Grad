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
import sys
from typing import Any

from core.errors import ConfigError

SERVICE = "grad"

# Named entries, so a missing credential names itself in the error.
HF_TOKEN = "hf_token"
OPENROUTER_KEY = "openrouter_key"
VOYAGE_KEY = "voyage_key"
S2_KEY = "s2_api_key"
# The fifth entry (HANDOFF-2 §18). Free from Context7's dashboard; raises rate
# limits rather than unlocking anything, so `tools/docs.py` treats it as
# optional and says so when it is missing.
CONTEXT7_KEY = "context7_key"

# The sixth. The funnel's Haiku stages (`core/haiku.py`) are Agent SDK clients
# in their own right, and they are reached the way every capability here is
# reached: the agent runs the CLI over Bash. That hop strips
# CLAUDE_CODE_OAUTH_TOKEN from the child environment -- deliberately, and only
# that variable; everything else in the environment survives it. So a token that
# lives in the environment authenticates the funnel from a terminal and leaves
# it unauthenticated under the agent, which is the only way it actually runs.
# The answer is the one §9 already gives for every other credential: keep it in
# the credential store and fetch it at the moment of use.
CLAUDE_TOKEN = "claude_oauth_token"

# The seventh, and the one that exists because the fourth cannot be obtained.
# Semantic Scholar stopped issuing API keys to free-domain email addresses, so
# `S2_KEY` is unreachable from a personal account and the anonymous pool is
# shared with everyone else in that position. Ai2 serve the same corpus over MCP
# at `asta-tools.allen.ai`, where a key is optional and raises rate limits
# rather than unlocking anything -- so this is treated like `CONTEXT7_KEY`: its
# absence is a note, not an error.
ASTA_KEY = "asta_api_key"

# The eighth, and the third that can reach a machine. Kaggle's API authenticates
# with a username/key pair; only the key is secret, so only the key is here --
# the username is `[kaggle] username` in the config, exactly as HF's token is a
# credential and its `[hf] namespace` is not. Which account runs the notebooks
# belongs in a file you can read; the thing that authorises it does not.
KAGGLE_KEY = "kaggle_key"

#: Every credential this project knows, in one tuple so nothing derived from it
#: can be added to and then forgotten. `status()` reports these,
#: `tools/jobs.py` accepts these, and `scrub_environment` removes the `GRAD_*`
#: fallback of each -- and it was that last one that drifted: two credentials
#: were added to the lookup and not to the scrub, leaving the agent's own
#: environment holding tokens §9 says must not be in it.
ALL: tuple[str, ...] = (
    HF_TOKEN,
    OPENROUTER_KEY,
    VOYAGE_KEY,
    S2_KEY,
    CONTEXT7_KEY,
    CLAUDE_TOKEN,
    ASTA_KEY,
    KAGGLE_KEY,
)


def store_name() -> str:
    """What the credential store is called on this machine.

    Named rather than hardcoded because the error that mentions it is the error
    a user acts on, and "check that Windows Credential Manager is reachable"
    told every Linux user to go and look at something that does not exist.
    """
    if os.name == "nt":
        return "Windows Credential Manager"
    if sys.platform == "darwin":
        return "the macOS Keychain"
    return "the Secret Service keyring"


def store_fix() -> str:
    """How to make credentials work here, for the platform actually in use.

    The POSIX branch has to answer for the headless case, because that is the
    normal case: a GPU box reached over SSH has no D-Bus session and therefore no
    Secret Service, and a research tool that cannot authenticate without a
    desktop session is a research tool that cannot run where the GPUs are. The
    environment route is the honest answer there, and it already exists --
    `GRAD_ALLOW_ENV_CREDENTIALS` was built for CI and the first-run bootstrap and
    is exactly as good a fit for a headless host.
    """
    if os.name == "nt":
        return "check that Windows Credential Manager is reachable for this user"
    return (
        "install a keyring backend (`pip install keyrings.alt`, or run a Secret Service "
        "provider such as gnome-keyring), or -- on a headless host, where there is no "
        "D-Bus session to hold one -- export the credential instead:\n"
        "    export GRAD_ALLOW_ENV_CREDENTIALS=1\n"
        "    export GRAD_<NAME>=...   # e.g. GRAD_VOYAGE_KEY, GRAD_CLAUDE_OAUTH_TOKEN"
    )


def _keyring() -> Any:
    try:
        import keyring  # noqa: PLC0415 - imported at point of use, on purpose
    except ImportError as exc:
        raise ConfigError(
            "the `keyring` package is not installed, so credentials cannot be read "
            f"from {store_name()}",
            fix="pip install keyring",
        ) from exc
    return keyring


def get(name: str, *, required: bool = True) -> str | None:
    """Read one credential. Never caches, never logs the value.

    GRAD_ALLOW_ENV_CREDENTIALS=1 permits an environment fallback; it exists for
    CI and for the first-run bootstrap, and it is off by default precisely
    because §9's argument is that the token must not be in the environment.

    **An unreachable store is an absent credential, not an error, when the
    caller said the credential was optional.** `required=False` is a caller
    stating it can proceed without this; a store that will not open is one more
    way for the value not to be there, and it is not a distinction that caller
    can act on. This mattered on exactly one platform and it was the one nobody
    ran: a headless Linux host has no Secret Service, so `keyring` raises
    `NoKeyringError` for every read, and three optional lookups
    (`s2_api_key`, `asta_api_key`, `context7_key`) turned that into a hard
    failure of anonymous retrieval -- discovery refusing to run for want of a key
    it does not need. `core/http.py` had already caught it at one of the three
    call sites; the other two are why this belongs here instead.

    A *required* credential still raises, and now says what the store is called
    on this platform and how to get one working.
    """
    kr = None
    #: The store failing to open, remembered rather than raised, because whether
    #: it is an error depends on `required` -- which is not known until the end.
    store_error: Exception | None = None
    try:
        kr = _keyring()
    except ConfigError as exc:
        store_error = exc
    value = None
    if kr is not None:
        try:
            value = kr.get_password(SERVICE, name)
        except Exception as exc:  # noqa: BLE001 - backend errors vary wildly by platform
            store_error = exc
    if not value and _env_fallback_allowed():
        value = os.environ.get(f"GRAD_{name.upper()}")
    if value:
        return value
    if not required:
        return None
    if store_error is not None:
        # The import failure already carries its own message and `pip install
        # keyring` fix; wrapping it would bury both under a vaguer sentence.
        if isinstance(store_error, ConfigError):
            raise store_error
        raise ConfigError(
            f"credential store unavailable while reading {name!r}: {store_error}",
            fix=store_fix(),
        ) from store_error
    raise ConfigError(
        f"credential {name!r} is not in the credential store",
        fix=f"python -m tools.jobs credential set {name}   # prompts, does not echo",
    )


def set_(name: str, value: str) -> None:
    """Store one credential.

    Unlike `get`, this cannot degrade: there is nowhere else to put the value, so
    an unreachable store is a real failure. It is reported as one rather than as
    a `NoKeyringError` traceback, because storing a credential is the step
    immediately after installing and a bare backend exception at that moment
    reads as "this software is broken" rather than "this host needs a backend".
    """
    try:
        _keyring().set_password(SERVICE, name, value)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - backend errors vary wildly by platform
        raise ConfigError(
            f"credential store unavailable while storing {name!r}: {exc}",
            fix=store_fix(),
        ) from exc


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
    return {n: present(n) for n in ALL}


def _env_fallback_allowed() -> bool:
    return os.environ.get("GRAD_ALLOW_ENV_CREDENTIALS") == "1"


NOT_AUTHENTICATED = (
    "there are no subscription credentials for the Agent SDK, so the model was never "
    "reached. The agent runs these CLIs over Bash, and that hop strips "
    "CLAUDE_CODE_OAUTH_TOKEN from the environment -- so the token has to come from the "
    "credential store, not from the environment."
)
AUTH_FIX = (
    "claude setup-token   # mint a token, then store it where the hop cannot strip it:\n"
    "python -m tools.jobs credential set claude_oauth_token"
)


def sdk_env() -> dict[str, str]:
    """Subscription credentials for a CLI that spawns its own Agent SDK client.

    Two callers with the same problem: `core/haiku.py`'s funnel stages and
    `core/mutate.py`'s mutation operator. Both are SDK clients running inside a
    tool CLI, and both are reached the way every capability here is reached --
    the agent runs the CLI over Bash, and `scrub_environment` has already taken
    `CLAUDE_CODE_OAUTH_TOKEN` out of the environment that hop inherits.

    The ambient variable comes first, so running one of these by hand in a
    terminal keeps working with no setup at all. The credential store is the
    fallback that makes the same command work when the *agent* is the one running
    it.

    `ClaudeAgentOptions.env` merges over the inherited environment rather than
    replacing it, so this adds one variable and takes nothing away.
    """
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        token = get(CLAUDE_TOKEN, required=False)
    if not token:
        raise ConfigError(NOT_AUTHENTICATED, fix=AUTH_FIX)
    return {"CLAUDE_CODE_OAUTH_TOKEN": token}


def hydrate_environment() -> str | None:
    """Put the stored subscription token into *this* process's environment.

    The complement of `scrub_environment`, and the fix for a gap between two
    things that were each individually right. The main loop authenticates from
    the ambient `CLAUDE_CODE_OAUTH_TOKEN` (`agent.preflight_environment`), while
    `sdk_env` reads the credential store because the Bash hop strips that
    variable from child processes. Nothing joined them up -- so a token stored
    through the app's credentials panel authenticated the funnel and the
    mutation operator, and left the agent's own loop with no credentials at all.
    That is the worst shape for this to fail in: the panel says STORED, and the
    first turn fails anyway.

    It bit the installed app rather than the terminal, which is why it survived.
    `claude setup-token` is run in a shell, and a shell that exported the result
    has it -- but the desktop shortcut launches `pythonw.exe` from Explorer,
    which inherits whatever the user made persistent and usually that is
    nothing.

    Three constraints, all of them load-bearing:

    **Ambient wins.** A token exported in a terminal is a deliberate choice --
    a second account, a token being tested -- and this must not quietly outrank
    it. Same precedence `sdk_env` uses, for the same reason.

    **After the scrub, never before.** `ANTHROPIC_API_KEY` outranks this token
    in the SDK's credential chain, so hydrating first would set a variable the
    scrub had not yet cleared the way for, and the session would bill the
    Developer Platform while reporting a subscription token present.

    **It never raises.** A missing `keyring`, a Credential Manager that will not
    open for this user -- neither is a reason to refuse to start. The caller
    reports what happened; the SDK still gives its own auth error if the token
    was the only thing that could have helped.
    """
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return None
    try:
        token = get(CLAUDE_TOKEN, required=False)
    except ConfigError:
        # The store is unreachable. `status()` already reports that condition to
        # the credentials panel, and there is nothing useful to do about it on
        # the startup path.
        return None
    if not token:
        return None
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return "CLAUDE_CODE_OAUTH_TOKEN"


def scrub_environment() -> list[str]:
    """Remove credential-shaped variables from the agent's own environment.

    Called by `agent.py` at startup. ANTHROPIC_API_KEY is the important one:
    it outranks CLAUDE_CODE_OAUTH_TOKEN in the credential chain, so a stray
    export silently bills the API instead of the subscription (HANDOFF §2).
    """
    removed = []
    # The GRAD_* fallbacks are derived from `ALL` rather than listed, and that
    # is the fix for a real gap: the list used to be written out by hand, so
    # `claude_oauth_token` and `asta_api_key` were added to `get()`'s lookup and
    # not to this one. The agent inherits its environment, so each omission
    # handed it exactly the environment-resident credential §9 argues must not
    # exist -- and `GRAD_CLAUDE_OAUTH_TOKEN` is the worst of them, because
    # `core/haiku.py` passes that token to subprocesses on purpose and the
    # scrub is what bounds who else can read it. Deriving it means adding a
    # credential cannot silently widen the boundary again.
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENROUTER_API_KEY",
        "VOYAGE_API_KEY",
        "CONTEXT7_API_KEY",
        # The pair the `kaggle` CLI reads straight out of the environment. Both
        # are listed, not just the secret one: the CLI authenticates only when it
        # has *both*, so removing the key alone is what makes the remaining
        # username harmless -- and leaving the pair intact would hand the agent
        # exactly the general remote-execution capability §9 denies it, through a
        # CLI `hooks.py` can only ask it not to type.
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        # Not a credential, but it points at a `kaggle.json` that holds the pair.
        # Scrubbing the two variables and leaving this one behind moves the
        # capability rather than removing it.
        "KAGGLE_CONFIG_DIR",
        *(f"GRAD_{name.upper()}" for name in ALL),
    ):
        if os.environ.pop(var, None) is not None:
            removed.append(var)
    return removed
