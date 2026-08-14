"""grad-docs -- is this library call current? (HANDOFF-2 §18)

    "A checker relying on Context7 alone will confidently describe an API
     version that is not installed."

So there are **two oracles, and the order matters**:

1. **Introspection -- what actually exists on this machine.**
   `importlib.metadata.version()`, `inspect.signature()`, `dir()`. Offline and
   definitive. This is how §17's `namespace` parameter was found, in about ten
   seconds.
2. **Context7 -- what is current.** Deprecations, changed idioms, migration
   paths. Answers what introspection cannot see.

Introspect first. Always.

## `check` imports, and importing runs code

Introspection is not static analysis, and the difference matters. To read a
signature this has to *import* the module, and importing executes that module's
top level. `check <file>` therefore imports every module the file names, and
name resolution goes through `sys.path` -- which includes the working directory.
A file containing `import helper` runs a sibling `helper.py`.

**So `check` is safe on code you trust and unsafe on code you do not.** Run it
on your own pipeline, not on a freshly downloaded repository. There is no way to
have the introspection oracle without this: a checker that does not import can
only guess at what is installed, which is the failure mode this command exists
to remove. `signature` has the same property for the one module it is given.

If that boundary ever needs to be tightened, the fix is to run the introspection
half in a subprocess -- the parse and the reporting stay as they are.

**Why a CLI and not a subagent.** The original plan was a Haiku "reality
checker" with Context7 and Pyright. It was rejected after the brief narrowed,
and the reasoning is worth keeping because it will be tempting to revisit: a QA
layer staffed by a weaker model than the one it checks is only sound when every
claim it makes is checkable against an oracle. Narrowing to library currency
achieved that -- but once achieved, the agency was doing no work. "Point it at a
file, get a verdict" is a tool, not an agent. This form keeps `Task` denied,
keeps Context7's schemas out of the main loop's context entirely, and inherits
`--json`, exit codes, and `fix` fields from `core/cli.py` for free.

Revisit only if the iterate-and-recheck loop (introspect -> hypothesise -> run a
counter-example -> recheck) proves necessary. That is the one thing this form
cannot do.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import sys
from pathlib import Path
from typing import Any

from core import config as config_mod, credentials, http
from core.cli import Cli, main
from core.errors import EXIT_CHECK_FAILED, GradError, NotFound, UsageError

cli = Cli(
    "grad-docs",
    "Check library calls against what is installed, and against what is current.",
    epilog=(
        "Two oracles, in this order:\n"
        "  1. introspection -- importlib.metadata + inspect.signature. offline, definitive.\n"
        "  2. Context7      -- deprecations and changed idioms. what introspection cannot see.\n\n"
        "WARNING: `check` and `signature` IMPORT the modules they inspect, and importing\n"
        "runs that module's top-level code. Module names resolve through sys.path, which\n"
        "includes the working directory. Run these on code you trust -- your own pipeline,\n"
        "not a repository you just downloaded.\n\n"
        "`check` exits 9 when it finds something, so it composes with preflight's\n"
        "declared-check mechanism if a pipeline wants it as a gate.\n\n"
        "If a Context7 request 404s the API has moved: read context7.com/docs/api-guide\n"
        "and fix [docs] base / resolve_path / docs_path in config/grad.toml."
    ),
)


# ---------------------------------------------------------------------------
# oracle 1: introspection
# ---------------------------------------------------------------------------
# Modules that ship with Python. Reporting "no distribution provides `json`" for
# every stdlib import would bury the findings that matter in noise.
_STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def installed_version(module: str) -> str | None:
    """The distribution version providing a module, if any."""
    top = module.split(".")[0]
    try:
        packages = importlib.metadata.packages_distributions()
    except Exception:  # noqa: BLE001 - older/odd environments
        packages = {}
    for dist in packages.get(top, []) or [top]:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a module that fails to import is a finding, not a crash
        return None


def signature_of(module: str, attribute: str) -> dict[str, Any]:
    """What this machine says about `module.attribute`.

    Returns a report rather than raising: "this attribute does not exist" is the
    single most valuable thing this command says, and it is not an error in the
    CLI.
    """
    mod = _import(module)
    if mod is None:
        return {"module": module, "importable": False}
    if not hasattr(mod, attribute):
        near = _close(attribute, dir(mod))
        return {
            "module": module,
            "importable": True,
            "attribute": attribute,
            "exists": False,
            "did_you_mean": near,
        }
    obj = getattr(mod, attribute)
    report: dict[str, Any] = {"module": module, "importable": True, "attribute": attribute, "exists": True}
    try:
        sig = inspect.signature(obj)
        report["signature"] = f"{attribute}{sig}"
        report["parameters"] = list(sig.parameters)
        report["keyword_only"] = [
            n for n, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        report["accepts_kwargs"] = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        # Builtins and C extensions often have no introspectable signature.
        # Saying so is honest; guessing would not be.
        report["signature"] = None
        report["note"] = "no introspectable signature (C extension or builtin)"
    return report


def _close(name: str, options: list[str], n: int = 3) -> list[str]:
    import difflib  # noqa: PLC0415

    return difflib.get_close_matches(name, [o for o in options if not o.startswith("_")], n=n, cutoff=0.6)


# ---------------------------------------------------------------------------
# static analysis of a file
# ---------------------------------------------------------------------------
class _Calls(ast.NodeVisitor):
    """Collect `alias.attr(...)` calls and the keyword names they pass.

    Deliberately shallow. It resolves `import x` / `import x as y` /
    `from a import b` aliases and nothing more -- no type inference, no
    cross-file resolution. Everything it reports is then *checked* against the
    installed object, so a shallow parse produces false negatives (calls it does
    not look at) rather than false positives (findings that are not real).
    """

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}   # local name -> dotted module
        self.from_imports: dict[str, tuple[str, str]] = {}  # local name -> (module, attr)
        self.calls: list[dict[str, Any]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                # `import a.b as x` binds `x` to the submodule itself.
                self.aliases[alias.asname] = alias.name
            else:
                # `import a.b` binds the *top-level package* `a`, not `a.b`, so
                # a later `a.f()` is `a.f` and not `a.b.f`. Recording the full
                # dotted path here made `import os.path` turn every `os.<attr>`
                # call into a false "does not exist on os.path" finding.
                top = alias.name.split(".")[0]
                self.aliases[top] = top
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and not node.level:
            for alias in node.names:
                if alias.name == "*":
                    continue
                self.from_imports[alias.asname or alias.name] = (node.module, alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = self._target(node.func)
        if target:
            module, attribute = target
            self.calls.append(
                {
                    "module": module,
                    "attribute": attribute,
                    "line": node.lineno,
                    "keywords": [kw.arg for kw in node.keywords if kw.arg],
                    "has_star_kwargs": any(kw.arg is None for kw in node.keywords),
                    "positional": len(node.args),
                }
            )
        self.generic_visit(node)

    def _target(self, func: ast.AST) -> tuple[str, str] | None:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self.aliases.get(func.value.id)
            if module:
                return module, func.attr
            return None
        if isinstance(func, ast.Name):
            found = self.from_imports.get(func.id)
            if found:
                return found
        return None


def analyse(path: Path) -> dict[str, Any]:
    """Introspect every resolvable library call in a file.

    A finding is one of three things, and all three are checkable against an
    oracle rather than being an opinion:

      * the module does not import here at all;
      * the attribute does not exist on the installed version;
      * a keyword argument the call passes is not in the installed signature.

    That third one is the §17 case in reverse: it is what would have caught
    `run_job(..., namespace=...)` on a `huggingface_hub` too old to take it.
    """
    if not path.is_file():
        raise NotFound(f"{path} does not exist", fix="give a path to a Python file")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        raise GradError(
            "unparseable",
            f"{path} is not valid Python: {exc}",
            exit_code=EXIT_CHECK_FAILED,
            fix="fix the syntax error first; nothing else can be checked until then",
        ) from exc

    walker = _Calls()
    walker.visit(tree)

    modules: dict[str, dict[str, Any]] = {}
    for module in sorted({*walker.aliases.values(), *(m for m, _ in walker.from_imports.values())}):
        top = module.split(".")[0]
        modules[module] = {
            "module": module,
            "stdlib": top in _STDLIB,
            "version": None if top in _STDLIB else installed_version(module),
            "importable": _import(module) is not None,
        }

    findings: list[dict[str, Any]] = []
    checked = 0
    # One finding per unimportable module, not one per call site: a missing
    # package produces the same finding on every line that uses it, and twenty
    # copies of "pip install x" buries the signature mismatches that matter.
    reported_modules: set[str] = set()
    for call in walker.calls:
        module, attribute = call["module"], call["attribute"]
        info = modules.get(module, {})
        # Stdlib calls are checked too. `_STDLIB` only suppresses the *version*
        # lookup -- "no distribution provides `json`" is noise, but
        # `json.dumpz()` is a real finding and introspection settles it just as
        # definitively as it does for a third-party package.
        if not info.get("importable", True):
            if module not in reported_modules:
                reported_modules.add(module)
                findings.append(
                    {
                        "severity": "error",
                        "kind": "module_not_importable",
                        "line": call["line"],
                        "message": f"`{module}` does not import in this environment",
                        "fix": f"pip install {module.split('.')[0]}",
                    }
                )
            continue
        checked += 1
        report = signature_of(module, attribute)
        if report.get("exists") is False:
            findings.append(
                {
                    "severity": "error",
                    "kind": "missing_attribute",
                    "line": call["line"],
                    "message": (
                        f"`{module}.{attribute}` does not exist on the installed "
                        f"{module.split('.')[0]} {info.get('version') or '(unknown version)'}"
                    ),
                    "fix": (
                        f"did you mean: {', '.join(report['did_you_mean'])}?"
                        if report.get("did_you_mean")
                        else f"python -m tools.docs query <libraryId> '{attribute}' --json"
                    ),
                }
            )
            continue
        params = report.get("parameters")
        if params is None or report.get("accepts_kwargs") or call["has_star_kwargs"]:
            continue
        unknown = [k for k in call["keywords"] if k not in params]
        if unknown:
            findings.append(
                {
                    "severity": "error",
                    "kind": "unknown_keyword",
                    "line": call["line"],
                    "message": (
                        f"`{module}.{attribute}()` does not take "
                        + ", ".join(f"`{k}`" for k in unknown)
                        + f" on the installed version; its signature is {report['signature']}"
                    ),
                    "fix": (
                        f"python -m tools.docs query <libraryId> '{attribute} parameters' --json  "
                        "# to see what replaced it"
                    ),
                }
            )

    return {
        "file": str(path),
        "modules": list(modules.values()),
        "calls_checked": checked,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
@cli.command(
    "resolve",
    "library name -> Context7 library id",
    setup=lambda p: p.add_argument("name"),
)
def cmd_resolve(args: argparse.Namespace) -> dict[str, Any]:
    client = http.Context7(config_mod.load())
    candidates = client.resolve(args.name)
    return {
        "query": args.name,
        "authenticated": client.authenticated,
        "candidates": candidates,
        "installed_version": installed_version(args.name),
        "next": (
            f"python -m tools.docs query {candidates[0]['library_id']} '<topic>' --json"
            if candidates
            else None
        ),
        "note": (
            None
            if client.authenticated
            else "no context7_key stored; rate limits are lower. "
            f"python -m tools.jobs credential set {credentials.CONTEXT7_KEY}"
        ),
    }


def _query_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("library_id", help="a Context7 library id, from `resolve`")
    p.add_argument("query", help="what you want to know, e.g. 'run_job namespace parameter'")
    p.add_argument("--tokens", type=int, default=5000, help="documentation budget for the answer")


@cli.command("query", "ask Context7 about a library", setup=_query_args)
def cmd_query(args: argparse.Namespace) -> dict[str, Any]:
    """Oracle 2. Run oracle 1 first: introspection knows what is installed, and
    this knows what is current, and confusing the two is the failure mode."""
    if args.tokens <= 0:
        raise UsageError("--tokens must be positive", fix="--tokens 5000")
    client = http.Context7(config_mod.load())
    return client.docs(args.library_id, args.query, tokens=args.tokens)


def _check_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "path",
        help="a Python file whose library calls should be checked. NOTE: this imports "
        "the modules the file names, which runs their top-level code -- use it on "
        "code you trust",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="introspection only; skip Context7 entirely",
    )
    p.add_argument(
        "--currency",
        action="store_true",
        help="also ask Context7 whether each imported library is current (one call per module)",
    )


@cli.command("check", "introspect a file's library calls; exit 9 on findings", setup=_check_args)
def cmd_check(args: argparse.Namespace) -> dict[str, Any]:
    """Introspection first, then Context7.

    Exits 9 -- "a check ran and reported failure" -- so it composes with
    preflight's declared-check mechanism if a pipeline later wants it as a gate.

    This imports the modules the file names, and importing executes them. See
    the module docstring: safe on code you trust, unsafe on code you do not.
    """
    report = analyse(Path(args.path).resolve())

    if args.currency and not args.offline:
        report["currency"] = _currency(report["modules"])

    if report["findings"]:
        first = report["findings"][0]
        raise GradError(
            "stale_calls",
            f"{len(report['findings'])} finding(s) in {args.path}: {first['message']}",
            exit_code=EXIT_CHECK_FAILED,
            fix=first.get("fix") or "check the call against the installed signature",
            detail=report,
        )
    return {**report, "ok": True}


def _currency(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-effort: a Context7 outage must not turn an offline-verifiable
    finding into an unusable command.

    Construction is inside the guard for the same reason the requests are --
    a missing credential backend must degrade this to "currency unknown", not
    discard the introspection results that were already computed.
    """
    try:
        client = http.Context7(config_mod.load())
    except GradError as exc:
        return [
            {"module": m["module"], "installed": m["version"], "error": exc.message}
            for m in modules
            if not m["stdlib"]
        ]
    out = []
    for info in modules:
        if info["stdlib"]:
            continue
        entry: dict[str, Any] = {"module": info["module"], "installed": info["version"]}
        try:
            candidates = client.resolve(info["module"].split(".")[0])
            entry["library_id"] = candidates[0]["library_id"] if candidates else None
            entry["versions"] = candidates[0].get("versions") if candidates else None
        except GradError as exc:
            entry["error"] = exc.message
        out.append(entry)
    return out


@cli.command(
    "signature",
    "what the installed library says about one call",
    setup=lambda p: (
        p.add_argument("module"),
        p.add_argument("attribute"),
    ),
)
def cmd_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Oracle 1, directly. Ten seconds, offline, definitive."""
    report = signature_of(args.module, args.attribute)
    report["installed_version"] = installed_version(args.module)
    if report.get("exists") is False:
        raise GradError(
            "missing_attribute",
            f"`{args.module}.{args.attribute}` does not exist on the installed version",
            exit_code=EXIT_CHECK_FAILED,
            fix=(
                f"did you mean: {', '.join(report['did_you_mean'])}?"
                if report.get("did_you_mean")
                else f"python -m tools.docs resolve {args.module} --json"
            ),
            detail=report,
        )
    if report.get("importable") is False:
        raise GradError(
            "not_importable",
            f"`{args.module}` does not import in this environment",
            exit_code=EXIT_CHECK_FAILED,
            fix=f"pip install {args.module.split('.')[0]}",
            detail=report,
        )
    return report


if __name__ == "__main__":
    main(cli)
