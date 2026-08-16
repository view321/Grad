"""The CLI contract from HANDOFF §8, implemented once.

    "A failed tool call returns a structured error the model can act on. A failed
     CLI returns a stack trace on stderr and an exit code of 1, and the
     characteristic model response to that is to retry with guessed flags."

Four obligations, all enforced here so no individual tool can forget one:

  * ``--json`` on every subcommand, emitting ``{"ok", "data", "error"}``.
  * distinct, documented exit codes (see `core.errors`).
  * errors state the fix, not just the fault.
  * unknown flags fail fast, naming the closest valid flag.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from typing import Any

from core.errors import (
    EXIT_INTERNAL,
    EXIT_MEANINGS,
    EXIT_OK,
    EXIT_USAGE,
    GradError,
    UsageError,
)

Handler = Callable[[argparse.Namespace], Any]
Setup = Callable[[argparse.ArgumentParser], None]


def envelope_ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def envelope_err(err: GradError) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": err.to_payload()}


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of calling sys.exit, so errors go through
    the same envelope as everything else."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message, fix=f"{self.prog} --help")

    def exit(self, status: int = 0, message: str | None = None):  # type: ignore[override]
        # --help / --version land here; let them through untouched.
        if message:
            sys.stderr.write(message)
        raise SystemExit(status)


def _suggest(unknown: Sequence[str], parser: argparse.ArgumentParser) -> str | None:
    """Name the closest valid flag for the first unrecognised one."""
    valid: list[str] = []
    for action in parser._actions:  # noqa: SLF001 - argparse offers no public view
        valid.extend(action.option_strings)
    for token in unknown:
        if not token.startswith("-"):
            continue
        near = difflib.get_close_matches(token, valid, n=1, cutoff=0.5)
        if near:
            return f"unknown flag {token!r}; did you mean {near[0]!r}?"
        return f"unknown flag {token!r}; valid flags: {' '.join(sorted(set(valid)))}"
    return None


class Cli:
    """A tool CLI. One instance per file in ``tools/``."""

    def __init__(self, prog: str, description: str, *, epilog: str = "") -> None:
        exit_docs = "\n".join(
            f"  {code:>2}  {meaning}" for code, meaning in sorted(EXIT_MEANINGS.items())
        )
        self.parser = _Parser(
            prog=prog,
            description=description,
            epilog=(epilog + "\n\nexit codes:\n" + exit_docs),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.parser.add_argument(
            "--json",
            action="store_true",
            help="emit the stable JSON envelope on stdout (always use this from the agent)",
        )
        self.sub = self.parser.add_subparsers(dest="_command", metavar="COMMAND")
        self._handlers: dict[str, Handler] = {}

    def command(
        self,
        name: str,
        help: str | None,
        *,
        setup: Setup | None = None,
        description: str | None = None,
    ) -> Callable[[Handler], Handler]:
        """Register a subcommand. `help=None` hides it from the command list.

        Hidden rather than absent, because some commands exist for the tool to
        call and not for a person to type: `tools/task.py`'s supervisor is the
        first, and it has to be reachable by name because it is spawned as
        `python -m tools.task _supervise`. `argparse.SUPPRESS` does not do this
        for a subparser -- it renders as the literal string `==SUPPRESS==` in the
        listing -- and the only mechanism that works is not passing `help` at all.
        """

        def decorate(fn: Handler) -> Handler:
            listing = {} if help is None else {"help": help}
            p = self.sub.add_parser(
                name,
                **listing,
                description=description or fn.__doc__ or help or name,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            p.set_defaults(_parser=p)
            # --json is accepted before *or* after the subcommand; models write both.
            p.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
            if setup:
                setup(p)
            self._handlers[name] = fn
            return fn

        return decorate

    def run(self, argv: Sequence[str] | None = None) -> int:
        argv = list(sys.argv[1:] if argv is None else argv)
        as_json = "--json" in argv
        try:
            args, unknown = self.parser.parse_known_args(argv)
            if unknown:
                target = getattr(args, "_parser", self.parser)
                raise UsageError(
                    _suggest(unknown, target) or f"unrecognised arguments: {' '.join(unknown)}",
                    fix=f"{target.prog} --help",
                )
            as_json = as_json or bool(getattr(args, "json", False))
            command = getattr(args, "_command", None)
            if not command:
                raise UsageError(
                    "no command given",
                    fix=f"{self.parser.prog} --help",
                )
            data = self._handlers[command](args)
        except SystemExit as exc:  # --help and --version
            return int(exc.code or 0)
        except GradError as exc:
            self._emit_error(exc, as_json)
            return exc.exit_code
        except KeyboardInterrupt:
            self._emit_error(
                GradError("interrupted", "interrupted", exit_code=EXIT_USAGE), as_json
            )
            return EXIT_USAGE
        except Exception as exc:  # noqa: BLE001 - last resort; never a bare traceback on stdout
            err = GradError(
                "internal",
                f"{type(exc).__name__}: {exc}",
                exit_code=EXIT_INTERNAL,
                fix="this is a bug in the CLI; the traceback is on stderr",
                detail={"traceback": traceback.format_exc().splitlines()[-6:]},
            )
            self._emit_error(err, as_json)
            traceback.print_exc(file=sys.stderr)
            return EXIT_INTERNAL

        self._emit_ok(data, as_json)
        return EXIT_OK

    # -- output ------------------------------------------------------------
    @staticmethod
    def _emit_ok(data: Any, as_json: bool) -> None:
        if as_json:
            print(json.dumps(envelope_ok(data), ensure_ascii=False, default=str))
        elif isinstance(data, str):
            print(data)
        elif data is not None:
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    @staticmethod
    def _emit_error(err: GradError, as_json: bool) -> None:
        if as_json:
            print(json.dumps(envelope_err(err), ensure_ascii=False, default=str))
        else:
            print(f"error [{err.code}]: {err.message}", file=sys.stderr)
            if err.fix:
                print(f"fix: {err.fix}", file=sys.stderr)


def main(cli: Cli) -> None:
    """Standard ``if __name__ == '__main__'`` body for a tool."""
    sys.exit(cli.run())
