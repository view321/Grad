"""Exit codes and the error type behind the CLI envelope (HANDOFF §8).

    "a usage error, a gate refusal, and an upstream failure are three different
     things and the model should not have to read prose to tell them apart"

So every failure carries a machine-readable `code`, a distinct exit status, and
where one exists, `fix` -- a literal next command.
"""

from __future__ import annotations

from typing import Any

# Exit codes. Stable; documented in README.md and in every --help epilog.
EXIT_OK = 0
EXIT_INTERNAL = 1          # unhandled - a bug in the CLI itself
EXIT_USAGE = 2             # bad/unknown flags, missing arguments
EXIT_NOT_FOUND = 3         # named entity does not exist
EXIT_PREFLIGHT = 4         # gate: no passing preflight for this submission hash
EXIT_EXPECTATION = 5       # gate: no open expectation bound to this submission
EXIT_SPEND = 6             # gate: per-job or rolling spend ceiling exceeded
EXIT_STALE_RUN = 7         # gate: an uncollected run is past its grace window
EXIT_UPSTREAM = 8          # a remote service failed
EXIT_CHECK_FAILED = 9      # a check ran and reported failure (preflight, nb verify)
EXIT_RUNNING = 10          # not an error: the job is still in flight
EXIT_CONFIG = 11           # missing credential, unknown host, malformed config

GATE_CODES = {EXIT_PREFLIGHT, EXIT_EXPECTATION, EXIT_SPEND, EXIT_STALE_RUN}

EXIT_MEANINGS = {
    EXIT_OK: "ok",
    EXIT_INTERNAL: "internal error",
    EXIT_USAGE: "usage error",
    EXIT_NOT_FOUND: "not found",
    EXIT_PREFLIGHT: "gate refusal: preflight missing or failing",
    EXIT_EXPECTATION: "gate refusal: no open expectation",
    EXIT_SPEND: "gate refusal: spend ceiling exceeded",
    EXIT_STALE_RUN: "gate refusal: stale uncollected run",
    EXIT_UPSTREAM: "upstream failure",
    EXIT_CHECK_FAILED: "a check failed",
    EXIT_RUNNING: "job still running",
    EXIT_CONFIG: "configuration or credential problem",
}


class GradError(Exception):
    """An error the agent is expected to act on.

    `fix` should be a command the caller can literally run. A bare traceback is
    the failure mode this class exists to prevent.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int = EXIT_INTERNAL,
        fix: str | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.fix = fix
        self.detail = detail

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.fix:
            payload["fix"] = self.fix
        if self.detail is not None:
            payload["detail"] = self.detail
        payload["exit_code"] = self.exit_code
        return payload


class UsageError(GradError):
    def __init__(self, message: str, *, fix: str | None = None) -> None:
        super().__init__("usage", message, exit_code=EXIT_USAGE, fix=fix)


class NotFound(GradError):
    def __init__(self, message: str, *, fix: str | None = None) -> None:
        super().__init__("not_found", message, exit_code=EXIT_NOT_FOUND, fix=fix)


class ConfigError(GradError):
    def __init__(self, message: str, *, fix: str | None = None) -> None:
        super().__init__("config", message, exit_code=EXIT_CONFIG, fix=fix)


class UpstreamError(GradError):
    def __init__(self, message: str, *, fix: str | None = None, detail: Any = None) -> None:
        super().__init__(
            "upstream", message, exit_code=EXIT_UPSTREAM, fix=fix, detail=detail
        )


class GateRefusal(GradError):
    """A submitter refused. The four gates of HANDOFF §6 all raise this."""

    def __init__(
        self,
        code: str,
        message: str,
        exit_code: int,
        *,
        fix: str | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(code, message, exit_code=exit_code, fix=fix, detail=detail)
