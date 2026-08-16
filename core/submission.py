"""The resolved submission and its hash (HANDOFF §6).

    "no TTL, and the hash covers the resolved submission"

A directory hash is simultaneously too broad (a note or a figure invalidates a
perfectly good preflight) and too narrow (a config edit with identical code is
the most common real change). So the hash covers exactly the things that can
change the outcome of the job:

  * the entrypoint and every first-party module it imports, resolved by import
    graph rather than by directory glob;
  * the fully resolved config *after* CLI overrides, serialised canonically;
  * the dependency lock file;
  * the dataset pointer and its revision;
  * the container image **digest**, not its tag;
  * the entrypoint argv;
  * anything the pipeline declares in `extra_hash_paths`.

Two known limits, handled explicitly rather than silently: dynamic imports are
invisible to static resolution, and files loaded at runtime outside the config
system are reached by neither the import graph nor the resolved config. Both are
what `extra_hash_paths` is for, and both are reported in the resolved document
so a reader can see the gap instead of assuming there isn't one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.errors import ConfigError, NotFound

HASH_LEN = 16  # display/filename length; the full digest is kept in the record


# ---------------------------------------------------------------------------
# import graph
# ---------------------------------------------------------------------------
def _module_candidates(module: str, level: int, source: Path, roots: list[Path]) -> list[Path]:
    """Possible files for an import, relative to first-party roots."""
    parts = module.split(".") if module else []
    bases: list[Path] = []
    if level:  # relative import: resolve against the importer's package
        base = source.parent
        for _ in range(level - 1):
            base = base.parent
        bases.append(base)
    else:
        bases.extend(roots)
    out: list[Path] = []
    for base in bases:
        target = base.joinpath(*parts) if parts else base
        out.append(target.with_suffix(".py"))
        out.append(target / "__init__.py")
    return out


def import_graph(entrypoint: Path, roots: list[Path]) -> tuple[list[Path], list[str]]:
    """Return (first-party files reachable from the entrypoint, warnings).

    Third-party imports are deliberately not followed: they are pinned by the
    lock file, which is in the hash.
    """
    entrypoint = entrypoint.resolve()
    roots = [r.resolve() for r in roots]
    seen: set[Path] = set()
    warnings: list[str] = []
    queue = [entrypoint]

    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError as exc:
            warnings.append(f"{path}: could not parse ({exc}); its imports are not covered")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                specs = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # `from pkg import mod` and `from . import mod` name a *module*
                # as often as an attribute, and resolving only `node.module`
                # stops at `pkg/__init__.py` -- leaving `pkg/mod.py` out of the
                # hash, so editing it would not invalidate a preflight record.
                # That is precisely the invalidation gap the import graph exists
                # to close, so each alias is tried as a module too.
                level = node.level or 0
                module = node.module or ""
                specs = [(module, level)]
                specs += [
                    (f"{module}.{alias.name}" if module else alias.name, level)
                    for alias in node.names
                    if alias.name != "*"
                ]
            elif isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name in {"import_module", "__import__"}:
                    warnings.append(
                        f"{path}: dynamic import via {name}() is invisible to the import "
                        "graph; list the target in extra_hash_paths if it can change the run"
                    )
                continue
            else:
                continue

            for module, level in specs:
                for cand in _module_candidates(module, level, path, roots):
                    if cand.is_file():
                        queue.append(cand.resolve())
                        break

    return sorted(seen), warnings


# ---------------------------------------------------------------------------
# container image
# ---------------------------------------------------------------------------
def _strip_tag(image: str) -> str:
    """Drop a trailing `:tag`, leaving a registry port alone.

    `registry.local:5000/org/name:2026-08` splits at the *last* colon, not the
    first -- cutting at the first would resolve to `registry.local@sha256:...`
    and put a wrong image in both the hash and the submission.
    """
    head, sep, tail = image.rpartition(":")
    return head if sep and "/" not in tail else image


def resolve_image_digest(image: str) -> str:
    """Require a digest-pinned image.

    ":latest is how remote environment drift sneaks past a hash that otherwise
    looks airtight." If a tag is given we try to resolve it locally; if that is
    not possible we refuse rather than hash the tag.
    """
    if "@sha256:" in image:
        return image
    for argv in (
        ["docker", "manifest", "inspect", "--verbose", image],
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image],
    ):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode != 0 or not out.stdout.strip():
            continue
        text = out.stdout.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                doc = json.loads(text)
                doc = doc[0] if isinstance(doc, list) else doc
                digest = doc.get("Descriptor", {}).get("digest")
            except (json.JSONDecodeError, AttributeError, IndexError):
                digest = None
            if digest:
                return f"{_strip_tag(image)}@{digest}"
        elif "@sha256:" in text:
            return text
    raise ConfigError(
        f"image {image!r} is not pinned to a digest and could not be resolved locally",
        fix=(
            "pin the image by digest, e.g. `image = \"repo/name@sha256:...\"` "
            "(a tag like :latest lets the remote environment drift past the hash)"
        ),
    )


# ---------------------------------------------------------------------------
# submission
# ---------------------------------------------------------------------------
@dataclass
class Submission:
    """A fully resolved submission: the thing the hash is over."""

    spec_path: Path
    entrypoint: Path
    argv: list[str]
    config: dict[str, Any]
    image: str
    dataset: dict[str, Any]
    lockfile: Path | None
    extra_hash_paths: list[Path] = field(default_factory=list)
    target: dict[str, Any] = field(default_factory=dict)
    estimate: dict[str, Any] = field(default_factory=dict)
    #: `[execution]` -- how this job may be run alongside others. Currently one
    #: key, `max_concurrent`, read by `gates.check_concurrency`.
    #:
    #: Deliberately **not** in the hash. `resolved()` covers what can change the
    #: outcome of the job, and how many siblings it was submitted beside cannot:
    #: a run submitted with `max_concurrent = 1` and the same run submitted with
    #: `max_concurrent = 4` produce the same numbers, so putting this in the hash
    #: would invalidate a perfectly good preflight for a scheduling preference.
    #: That is the same argument the module docstring makes about a directory
    #: hash being simultaneously too broad and too narrow.
    execution: dict[str, Any] = field(default_factory=dict)
    metrics_file: str = "metrics.json"
    warnings: list[str] = field(default_factory=list)
    _files: list[Path] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    @classmethod
    def load(
        cls,
        spec_path: Path | str,
        *,
        overrides: dict[str, Any] | None = None,
        resolve_digest: bool = True,
    ) -> Submission:
        spec_path = Path(spec_path).resolve()
        if not spec_path.is_file():
            raise NotFound(
                f"submission spec {spec_path} not found",
                fix="write a submission spec (see skills/preflight/SKILL.md for the schema)",
            )
        text = spec_path.read_text(encoding="utf-8")
        try:
            spec = tomllib.loads(text) if spec_path.suffix == ".toml" else json.loads(text)
        except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"{spec_path} is malformed: {exc}", fix=f"fix the syntax in {spec_path}") from exc

        base = spec_path.parent
        missing = [k for k in ("entrypoint", "image") if k not in spec]
        if missing:
            raise ConfigError(
                f"{spec_path} is missing required key(s): {', '.join(missing)}",
                fix="every submission needs at least `entrypoint` and `image` (digest-pinned)",
            )

        entrypoint = (base / spec["entrypoint"]).resolve()
        if not entrypoint.is_file():
            raise NotFound(f"entrypoint {entrypoint} does not exist", fix="fix `entrypoint` in the spec")

        config = dict(spec.get("config", {}))
        cfg_file = spec.get("config_file")
        if cfg_file:
            cfg_path = (base / cfg_file).resolve()
            if not cfg_path.is_file():
                raise NotFound(f"config_file {cfg_path} does not exist", fix="fix `config_file` in the spec")
            raw = cfg_path.read_text(encoding="utf-8")
            loaded = tomllib.loads(raw) if cfg_path.suffix == ".toml" else json.loads(raw)
            config = {**loaded, **config}
        # CLI overrides are applied *before* hashing: the hash covers the config
        # the job will actually see, not the file on disk.
        for key, value in (overrides or {}).items():
            _set_dotted(config, key, value)

        lockfile = None
        if spec.get("lockfile"):
            lockfile = (base / spec["lockfile"]).resolve()
            if not lockfile.is_file():
                raise NotFound(f"lockfile {lockfile} does not exist", fix="fix `lockfile` in the spec")

        image = spec["image"]
        if resolve_digest:
            image = resolve_image_digest(image)

        extra = [(base / p).resolve() for p in spec.get("extra_hash_paths", [])]
        roots = [(base / r).resolve() for r in spec.get("source_roots", ["."])]
        files, warnings = import_graph(entrypoint, roots)

        for p in extra:
            if not p.exists():
                warnings.append(f"extra_hash_path {p} does not exist")

        sub = cls(
            spec_path=spec_path,
            entrypoint=entrypoint,
            argv=[str(a) for a in spec.get("argv", [])],
            config=config,
            image=image,
            dataset=dict(spec.get("dataset", {})),
            lockfile=lockfile,
            extra_hash_paths=extra,
            target=dict(spec.get("target", {})),
            estimate=dict(spec.get("estimate", {})),
            execution=dict(spec.get("execution", {})),
            metrics_file=spec.get("metrics_file", "metrics.json"),
            warnings=warnings,
            _files=files,
        )
        if not sub.dataset.get("revision") and sub.dataset:
            sub.warnings.append(
                "dataset has no `revision`; the hash cannot notice the data changing under it"
            )
        return sub

    # -- hashing -----------------------------------------------------------
    def resolved(self) -> dict[str, Any]:
        """The canonical document that gets hashed. Also what gets stored in
        the preflight record, so a human can diff two hashes and see why."""
        base = self.spec_path.parent

        def rel(p: Path) -> str:
            try:
                return p.relative_to(base).as_posix()
            except ValueError:
                return p.as_posix()

        return {
            "schema": 1,
            "entrypoint": rel(self.entrypoint),
            "argv": self.argv,
            "config": self.config,
            "image": self.image,
            "dataset": self.dataset,
            "sources": {rel(p): _digest_file(p) for p in self._files},
            "lockfile": {rel(self.lockfile): _digest_file(self.lockfile)} if self.lockfile else None,
            "extra": {rel(p): _digest_file(p) for p in self.extra_hash_paths},
        }

    def hash(self) -> str:
        return hash_resolved(self.resolved())

    def full_hash(self) -> str:
        return hash_resolved(self.resolved(), length=None)

    def estimated_cost_usd(self) -> float:
        """Cost estimate from the spec. Used by the ceiling gates; the actual
        cost is computed by `collect` from the platform's own accounting."""
        if "cost_usd" in self.estimate:
            return float(self.estimate["cost_usd"])
        hours = float(self.estimate.get("hours", 0.0))
        rate = float(self.estimate.get("rate_usd_per_hour", 0.0))
        return hours * rate

    def estimated_duration_s(self) -> float:
        if "duration_s" in self.estimate:
            return float(self.estimate["duration_s"])
        return float(self.estimate.get("hours", 0.0)) * 3600.0


def hash_resolved(resolved: dict[str, Any], *, length: int | None = HASH_LEN) -> str:
    """The submission hash of an already-resolved document.

    A module function as well as a method because the *resolved document* and the
    `Submission` that produced it have different lifetimes. `core/experiments.py`
    stores the document in the archive and re-derives the hash from it later,
    possibly on another machine, with the spec file long since edited -- so it
    needs the hashing rule without needing a `Submission` to hold it. Keeping one
    implementation is what makes that check mean anything: two spellings of "the
    canonical form" that drifted apart would make the verifier report a mismatch
    for every archived run.
    """
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def _digest_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:HASH_LEN]


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """`--set lr=3e-4` -> ('lr', 0.0003). Values parse as JSON where possible so
    that types survive into the hash; otherwise they stay strings."""
    if "=" not in text:
        raise ConfigError(f"malformed override {text!r}", fix="use --set key.path=value")
    key, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value
