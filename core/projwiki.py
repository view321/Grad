"""The facts a research project's wiki is built on, extracted without a model.

This is the *retrieved* half of the project wiki. Everything here is read off
disk and out of the ledger and is true by construction: the file tree, the
spec and its comments, what imports what, which functions exist and where, which
predictions were registered, which runs tested them and what they returned.
`core/wikigen.py` is the generated half, and it is handed this and nothing else.

**Why the split is the whole design.** A wiki over research code has one failure
mode that matters: sounding right. Prose about an experiment that never ran, a
hyperparameter that was never set, a function that was renamed last week -- all
of it reads exactly like prose about the real thing, and the reader has no way
to tell without going to the source, which is what the wiki was supposed to save
them. So the model is never asked what the code *is*. It is asked to explain
facts it is given, and the page it writes carries those facts beside the prose:
every module page names the symbols and line numbers it is describing, every
claim about a result names the run id it came from. What the model adds is the
connective argument -- why these pieces are arranged this way, what the
experiment is actually testing -- which is the part no extraction can produce and
the part a person reacquiring a project actually needs.

**Scope is `pipelines/<name>/`, not the whole workspace.** That is where the
agent's generated research code lives: the entrypoint, the model, the data
loader, the probe, the tests. `projects/<id>/` supplies the intent (PLAN.md,
MEMORY.md) and `ledger/` supplies the outcomes. Nothing else is read, and in
particular `notes/` and `data/papers/` are not -- the same allowlist discipline
`tools/wiki.py` applies for the same reason, since the generated half ships what
it is given to a model.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from core import paths, projects
from core.errors import NotFound

#: Source files a pipeline is made of. Notebooks are deliberately absent: a
#: `.ipynb` is JSON around source, `tools/nb.py` owns them, and a wiki page
#: describing cell 7 of a notebook that has since been re-executed is the
#: staleness problem in its worst form.
SOURCE_GLOB = "*.py"

#: Everything else worth naming in the tree, by suffix. Read for their
#: *presence and shape* rather than their contents -- a metrics file is a
#: result, not documentation, and the ledger is where results are read from.
DATA_SUFFIXES: tuple[str, ...] = (".toml", ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml")

#: Directories inside a pipeline that are never part of it.
SKIP_DIRS: frozenset[str] = frozenset({"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"})

#: How much of one source file reaches the generated half, in characters. A
#: bound rather than the whole file, because these go into a context window and
#: a 200KB `train.py` would crowd out the ledger facts that make the page
#: trustworthy. Symbols and signatures are extracted in full regardless, so what
#: is truncated is function *bodies* -- and a page that describes bodies in
#: detail is a page that goes stale on the next edit.
SOURCE_MAX_CHARS = 12_000

#: How much of a spec file reaches it. Specs are small and almost entirely
#: comments, and those comments are the single highest-signal thing in the tree:
#: they are where the person who chose `accelerator = "NvidiaTeslaT4"` wrote down
#: that Kaggle silently handed back a P100 last time.
SPEC_MAX_CHARS = 8_000


# ---------------------------------------------------------------------------
# which pipelines belong to a project
# ---------------------------------------------------------------------------
def pipelines_for(project_id: str, state: dict[str, Any] | None = None) -> list[Path]:
    """The pipeline directories this project's code lives in.

    Two sources of evidence, because neither alone is right. The convention is
    that `projects/<id>/` and `pipelines/<id>/` share a name, and that covers the
    common case; but a project may run several pipelines, or one whose directory
    was named before the project was, so every run bound to the project also
    votes -- its `spec` path names the directory the run was submitted from.

    Sorted, and the same-name directory first when it exists: it is the one a
    reader means by "the code", and the overview page is written about whatever
    is first.
    """
    root = paths.root() / "pipelines"
    found: dict[str, Path] = {}
    same_name = root / project_id
    if same_name.is_dir():
        found[project_id] = same_name

    snapshot = state if state is not None else projects.state(project_id)
    for run in snapshot.get("runs") or []:
        spec = run.get("spec") or run.get("spec_path")
        if not spec:
            continue
        try:
            directory = Path(str(spec)).parent
        except (TypeError, ValueError):
            continue
        # Only inside `pipelines/`. A run submitted from somewhere else names a
        # directory this wiki has no allowlist for, and following it would be
        # the wiki reading arbitrary paths out of the ledger.
        if directory.is_dir() and directory.resolve().parent == root.resolve():
            found.setdefault(directory.name, directory)
    ordered = [found.pop(project_id)] if project_id in found else []
    return ordered + [found[name] for name in sorted(found)]


# ---------------------------------------------------------------------------
# one module
# ---------------------------------------------------------------------------
def symbols(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Top-level classes and functions, with signatures and line numbers.

    `ast`, not a regex and not an import: a research pipeline imports torch and
    downloads a checkpoint at module scope, so importing it to introspect it
    would run it. Line numbers are the point -- they are what lets a page say
    "`build_corpus` (data.py:41)" and be checkable.

    Methods are included one level deep, because a `nn.Module` subclass keeps the
    whole model in `__init__` and `forward` and a class listed without them says
    almost nothing about the architecture.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (SyntaxError, ValueError) as exc:
        return [], [f"{path.name}: could not parse ({exc})"]
    except OSError as exc:
        return [], [f"{path.name}: could not read ({exc})"]

    out: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_callable(node, path))
        elif isinstance(node, ast.ClassDef):
            methods = [
                _callable(child, path, qualifier=node.name)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not (child.name.startswith("_") and child.name != "__init__")
            ]
            out.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "signature": f"class {node.name}({', '.join(_base(b) for b in node.bases)})",
                    "line": node.lineno,
                    "doc": _first_line(ast.get_docstring(node)),
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.Assign):
            # Module-level constants are configuration in disguise -- the number
            # of layers, the vocabulary size, the eval interval -- and they are
            # what a reader looks for first.
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    out.append(
                        {
                            "kind": "constant",
                            "name": target.id,
                            "signature": f"{target.id} = {_literal(node.value)}",
                            "line": node.lineno,
                            "doc": "",
                        }
                    )
    return out, []


def _callable(node: Any, path: Path, *, qualifier: str = "") -> dict[str, Any]:
    """One function or method, with the signature it actually has.

    Positional-only arguments and the `/` and `*` markers are part of the
    contract, not decoration: `def f(a, /, b, *, c)` rendered as `def f(b, c)`
    -- `posonlyargs` is a separate list and was never read -- which is not a
    shortened signature but a *wrong* one, in a fact sheet whose whole claim is
    that everything in it was read off disk.
    """
    spec = node.args
    args = [a.arg for a in spec.posonlyargs]
    if spec.posonlyargs:
        args.append("/")
    args += [a.arg for a in spec.args]
    if spec.vararg:
        args.append("*" + spec.vararg.arg)
    elif spec.kwonlyargs:
        # The bare `*` is what *makes* the arguments after it keyword-only.
        # Without it they read as ordinary positional ones.
        args.append("*")
    args += [a.arg for a in spec.kwonlyargs]
    if spec.kwarg:
        args.append("**" + spec.kwarg.arg)
    name = f"{qualifier}.{node.name}" if qualifier else node.name
    return {
        "kind": "method" if qualifier else "function",
        "name": name,
        "signature": f"def {name}({', '.join(args)})",
        "line": node.lineno,
        "doc": _first_line(ast.get_docstring(node)),
    }


def _base(node: ast.expr) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - a base class we cannot name is not a failure
        return "?"


def _literal(node: ast.expr) -> str:
    """A constant's value, short enough to read. Never evaluated."""
    try:
        text = ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "…"
    return text if len(text) <= 90 else text[:87] + "…"


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    line = doc.strip().splitlines()[0].strip()
    return line if len(line) <= 200 else line[:197] + "…"


# ---------------------------------------------------------------------------
# one pipeline
# ---------------------------------------------------------------------------
def _tree(directory: Path) -> list[dict[str, Any]]:
    """Every file worth naming, with its size. Not the contents."""
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or SKIP_DIRS & set(path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix != ".py" and suffix not in DATA_SUFFIXES:
            continue
        try:
            stat = path.stat()
            lines = (
                sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
                if suffix in (".py", ".toml", ".md", ".txt")
                else None
            )
        except OSError:
            continue
        rows.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": stat.st_size,
                "lines": lines,
                "kind": "source" if suffix == ".py" else "data",
            }
        )
    return rows


def _specs(directory: Path) -> list[dict[str, Any]]:
    """Every `*.toml` in the pipeline, parsed *and* kept as text.

    Both halves, deliberately. The parsed form is what a page can be checked
    against -- `[target] accelerator` is a fact. The raw text is where the
    reasoning is: these files are more comment than value, and the comment
    explaining that Kaggle silently substituted a P100 for the requested L4 is
    worth more to a person reacquiring the project than every key in the file.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.toml")):
        parsed: dict[str, Any] = {}
        error = None
        try:
            import tomllib  # noqa: PLC0415 - stdlib, but only needed here

            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a spec mid-edit is not a failure
            error = f"{type(exc).__name__}: {exc}"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:SPEC_MAX_CHARS]
        except OSError:
            text = ""
        out.append(
            {
                "name": path.name,
                "parsed": parsed,
                "text": text,
                "error": error,
                "entrypoint": parsed.get("entrypoint"),
            }
        )
    return out


def _modules(
    directory: Path, specs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every Python file in the pipeline, with its symbols and its imports.

    The import edges come from `core/submission.py:import_graph` where there is
    an entrypoint to walk from -- the same resolver the submission hash uses, so
    the wiki's picture of what depends on what is the one the gates already act
    on. Files it does not reach are still listed: a module nothing imports is a
    fact about the pipeline, and usually an interesting one.
    """
    from core import submission  # noqa: PLC0415 - avoids an import cycle at module scope

    files = sorted(
        p for p in directory.rglob(SOURCE_GLOB) if p.is_file() and not SKIP_DIRS & set(p.parts)
    )
    root = directory.resolve()
    reached: set[Path] = set()
    warnings: list[str] = []
    for spec in specs:
        entry = spec.get("entrypoint")
        if not entry:
            continue
        path = (directory / str(entry)).resolve()
        # The entrypoint is a *value read from a file*, and it is the one thing
        # here that can point outside the pipeline: an absolute path, or one
        # that climbs. `import_graph` would then read and parse it, which is
        # this module's scope allowlist -- core/ and tools/ are in scope,
        # `notes/` and `data/papers/` are not -- being decided by a `.toml`
        # rather than by code. Checked after `resolve`, so a symlink is caught
        # too.
        if root not in path.parents:
            warnings.append(
                f"{spec['name']}: entrypoint {entry} resolves outside the pipeline directory "
                "and was not followed"
            )
            continue
        if not path.is_file():
            warnings.append(f"{spec['name']}: entrypoint {entry} does not exist")
            continue
        found, notes = submission.import_graph(path, [directory])
        reached |= {p.resolve() for p in found}
        warnings += notes

    out: list[dict[str, Any]] = []
    for path in files:
        found, notes = symbols(path)
        warnings += notes
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source = ""
        out.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "doc": _module_doc(source),
                "symbols": found,
                "imports": _first_party_imports(source, directory, path),
                "reachable": path.resolve() in reached,
                "is_test": path.name.startswith("test_") or path.name.endswith("_test.py"),
                "lines": source.count("\n") + 1 if source else 0,
                "source": source[:SOURCE_MAX_CHARS],
                "truncated": len(source) > SOURCE_MAX_CHARS,
            }
        )
    return out, warnings


def _module_doc(source: str) -> str:
    try:
        return (ast.get_docstring(ast.parse(source)) or "").strip()[:1200]
    except (SyntaxError, ValueError):
        return ""


def _local_modules(directory: Path) -> set[str]:
    """Every module in the pipeline, by the name an import would use.

    Dotted, not by file stem. `rglob` (matching `_modules`, so a pipeline with a
    subdirectory has those files listed *and* reachable as someone's import)
    means stems collide with the world: a `kernels/scan.py` made every
    `import scan` in the tree look first-party, including a third-party one --
    a wrong edge in a fact sheet whose entire claim is that it was read off
    disk. `kernels/__init__.py` names the package `kernels` rather than a module
    called `__init__`.
    """
    out: set[str] = set()
    for path in directory.rglob(SOURCE_GLOB):
        if not path.is_file() or SKIP_DIRS & set(path.parts):
            continue
        parts = path.relative_to(directory).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            out.add(".".join(parts))
    return out


def _package_of(path: Path, directory: Path) -> str:
    """The package a relative import inside `path` is relative *to*."""
    parts = path.relative_to(directory).with_suffix("").parts
    # A package's `__init__.py` is inside the package it names; every other
    # module is inside its parent.
    return ".".join(parts[:-1]) if parts[-1] != "__init__" else ".".join(parts[:-1])


def _first_party_imports(source: str, directory: Path, path: Path | None = None) -> list[str]:
    """Modules in this pipeline that this file imports.

    Third-party imports are deliberately absent: they are pinned by the image
    digest, and a map of the pipeline is about how its own pieces fit. Only the
    names that resolve to a file in the tree are kept, so an import that merely
    shares a word with one of them is not reported as an edge.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    local = _local_modules(directory)
    package = _package_of(path, directory) if path is not None else ""

    def resolve(name: str) -> str | None:
        """`a.b.c` if the tree has it, else the longest prefix it does have."""
        parts = name.split(".")
        for stop in range(len(parts), 0, -1):
            candidate = ".".join(parts[:stop])
            if candidate in local:
                return candidate
        return None

    def relative(level: int, module: str | None, leaf: str = "") -> str | None:
        """`from ..pkg import x`, against the importing file's own package."""
        base = package.split(".") if package else []
        # One dot is "this package"; each further dot climbs one level.
        if level > 1:
            base = base[: len(base) - (level - 1)]
        parts = [*base, *(module.split(".") if module else []), *([leaf] if leaf else [])]
        return resolve(".".join(parts)) if parts else None

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {r for a in node.names if (r := resolve(a.name))}
        elif isinstance(node, ast.ImportFrom):
            # `from pkg import mod` names a *module* as often as an attribute,
            # so each alias is tried as one and the bare module is tried too --
            # the same ambiguity `core/submission.py:import_graph` resolves for
            # the submission hash, and for the same reason.
            resolved = {
                r
                for alias in node.names
                if (
                    r := (
                        relative(node.level, node.module, alias.name)
                        if node.level
                        else resolve(f"{node.module}.{alias.name}" if node.module else alias.name)
                    )
                )
            }
            if resolved:
                names |= resolved
                continue
            # Only when no alias was a module: `from kernels import scan` is an
            # edge to `kernels.scan`, and reporting `kernels` beside it would be
            # a second edge for one import. `from kernels import CONST` has no
            # submodule to name, and there the package *is* the edge.
            bare = (
                relative(node.level, node.module)
                if node.level
                else (resolve(node.module) if node.module else None)
            )
            if bare:
                names.add(bare)
    return sorted(names)


# ---------------------------------------------------------------------------
# the ledger half
# ---------------------------------------------------------------------------
def _ledger(state: dict[str, Any]) -> dict[str, Any]:
    """What this project predicted, what it ran, and what came back.

    Flattened out of `core/projects.py:state` into plain data: `Run` objects
    carry behaviour the generated half must not depend on, and a page that
    quotes a result has to quote a value that was written down rather than one
    computed at render time.
    """
    falsified = set(state.get("falsified") or [])
    bound = state.get("bound_to") or {}
    expectations = []
    for exp in state.get("expectations") or []:
        eid = str(exp.get("id") or "")
        predicted = exp.get("predicted") or {}
        expectations.append(
            {
                "id": eid,
                "task": exp.get("task"),
                "quantity": exp.get("quantity"),
                "low": predicted.get("low"),
                "high": predicted.get("high"),
                "direction": predicted.get("direction"),
                "confidence": exp.get("confidence"),
                "comparability": exp.get("comparability"),
                "basis": [
                    {
                        "paper": b.get("paper"),
                        "locator": b.get("locator"),
                        "value": b.get("value"),
                        "conditions": b.get("conditions"),
                    }
                    for b in (exp.get("basis") or [])
                ],
                "falsified": eid in falsified,
                "run": bound.get(eid),
            }
        )

    runs = []
    for run in state.get("runs") or []:
        runs.append(
            {
                "id": run.id,
                "task": run.get("task"),
                "status": run.status,
                "collected": run.collected,
                "platform": run.get("platform"),
                "spec": run.get("spec"),
                "submitted_at": run.get("submitted_at"),
                "cost_usd": run.cost_for_ceiling(),
                "results": run.get("results") or {},
                "expectation_id": run.get("expectation_id"),
                "deviations": run.get("deviations") or [],
                "unjudged": [d.get("quantity") for d in run.unjudged_deviations()],
                "error": run.get("error"),
            }
        )
    return {
        "expectations": expectations,
        "runs": runs,
        "counts": {
            "expectations": len(expectations),
            "falsified": len([e for e in expectations if e["falsified"]]),
            "runs": len(runs),
            "collected": len([r for r in runs if r["collected"]]),
            "in_flight": len([r for r in runs if not r["collected"]]),
            "awaiting_verdict": len([r for r in runs if r["unjudged"]]),
        },
    }


def _papers(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """The papers this project's predictions were based on, with real titles.

    The `basis` field holds whatever string the agent cited, and the corpus holds
    the title it parsed at ingest. Joining them is what turns a references list
    from a column of arXiv ids into something a reader recognises -- and the
    normalisation is the same one `ui/models.py` needs, for the same reason: a
    basis says "arXiv:2001.08361" and the corpus says `arxiv_2001.08361`.
    """
    cited: dict[str, dict[str, Any]] = {}
    for exp in ledger["expectations"]:
        for basis in exp["basis"]:
            key = _paper_key(basis.get("paper"))
            if not key:
                continue
            node = cited.setdefault(
                key, {"key": key, "cited_as": basis.get("paper"), "title": "", "expectations": []}
            )
            node["expectations"].append(exp["id"])

    if not cited:
        return []
    try:
        from core import corpus  # noqa: PLC0415 - optional; a workspace may have no index

        con = corpus.connect(create=False)
        try:
            for row in con.execute("SELECT id, title FROM documents").fetchall():
                key = _paper_key(row["id"])
                if key in cited and row["title"]:
                    cited[key]["title"] = str(row["title"])
        finally:
            con.close()
    except Exception:  # noqa: BLE001 - no index is the normal state of a new workspace
        pass
    return [cited[k] for k in sorted(cited)]


def _paper_key(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^(arxiv[:_/]?)", "", text)
    return re.sub(r"[^a-z0-9.]", "", text)


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------
#: What a change to invalidates a wiki. **Source and spec, and nothing else** --
#: which is narrower than the file list in `_tree` on purpose.
#:
#: A pipeline directory also holds `metrics.jsonl`, `runs.jsonl` and whatever
#: the last kernel wrote there, and those change every time a run is collected.
#: Hashing them would mark every page stale the moment a *result* arrived, which
#: is the opposite of what staleness means here: a page is stale when the code
#: it describes has moved. It is also the honest line, because these are the
#: only files whose contents ever reach a page -- `_tree` lists the rest by name
#: and size, and a name does not change when the bytes behind it do.
HASHED_SUFFIXES: tuple[str, ...] = (".py", ".toml")


def source_hash(project_id: str, pipelines: list[Path] | None = None) -> dict[str, Any]:
    """A digest over exactly what the wiki was built from.

    The same idea as `core/submission.py`'s hash and `tools/wiki.py`'s: not a
    directory mtime and not a TTL, because "is this wiki still true" is a
    question about file *contents*. Covers the pipeline sources and specs and the
    project's authored documents -- and deliberately not the ledger, which grows
    every time a run is collected: a wiki is not stale because a new result
    arrived, it is stale because the code it describes changed.

    `pipelines` is passed by `collect`, which has already worked out which
    directories belong to this project. Recomputing it here read the ledger a
    second time for an answer it already had -- and left a window in which the
    hash and `facts["pipelines"]` could describe two different sets of
    directories, so a wiki could record a digest over files no page was written
    from.
    """
    digests: dict[str, str] = {}
    root = paths.root()
    directory = projects.resolve_dir(project_id)
    for name in projects.AUTHORED:
        path = directory / name
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = _digest(path)
    for pipeline in pipelines if pipelines is not None else pipelines_for(project_id):
        for path in sorted(pipeline.rglob("*")):
            if not path.is_file() or SKIP_DIRS & set(path.parts):
                continue
            if path.suffix.lower() not in HASHED_SUFFIXES:
                continue
            digests[path.relative_to(root).as_posix()] = _digest(path)
    canonical = "\n".join(f"{k}:{v}" for k, v in sorted(digests.items()))
    return {
        "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "files": digests,
        "project": project_id,
    }


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
    except OSError:
        return "unreadable"
    return h.hexdigest()[:16]


def collect(project_id: str) -> dict[str, Any]:
    """Everything the wiki is built from, extracted and true by construction.

    No model, no network, no imports of the code being described. Safe to call
    on every render if anything ever wants to -- though nothing does, because
    `tools/projwiki.py` writes the result to disk beside the prose it grounded.
    """
    if not projects.exists(project_id):
        raise NotFound(
            f"no project {project_id!r} in this workspace",
            # `list`, not `status`: status answers "how is *this* project doing"
            # and defaults to the selected one, which is no help to someone who
            # has just been told the id they gave does not exist.
            fix="python -m tools.budget list --json   # every project id in this workspace",
        )
    directory = projects.resolve_dir(project_id)
    state = projects.state(project_id)
    ledger = _ledger(state)

    docs: dict[str, str] = {}
    for name in projects.AUTHORED:
        path = directory / name
        if path.is_file():
            try:
                docs[name] = path.read_text(encoding="utf-8", errors="replace")[:SOURCE_MAX_CHARS]
            except OSError:
                continue

    warnings: list[str] = []
    pipelines: list[dict[str, Any]] = []
    # Worked out once and reused for the digest below: two calls could disagree
    # -- `pipelines_for` reads the ledger, and a run collected between them adds
    # a directory -- and a wiki whose hash covers files no page was written from
    # is a wiki that reports itself stale for a reason nobody can find.
    directories = pipelines_for(project_id, state)
    for pipeline in directories:
        specs = _specs(pipeline)
        modules, notes = _modules(pipeline, specs)
        warnings += notes
        pipelines.append(
            {
                "name": pipeline.name,
                "dir": str(pipeline),
                "tree": _tree(pipeline),
                "specs": specs,
                "modules": modules,
            }
        )

    return {
        "project": project_id,
        "dir": str(directory),
        "docs": docs,
        "pipelines": pipelines,
        "ledger": ledger,
        "papers": _papers(ledger),
        "warnings": warnings,
        "source": source_hash(project_id, directories),
    }
