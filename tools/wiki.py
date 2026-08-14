"""grad-wiki -- RepoWiki, the human's map (HANDOFF-2 §20).

**Scope: human-facing only.** Not in the agent's tool list, not in
`prompts/system.md`, no context cost. Its job is letting a person reacquire the
shape of a growing codebase quickly. `HANDOFF.md` remains the design record and
`README.md` the report; this targets the third thing -- the module-level "what
calls what, and where does this value come from" view nobody wants to maintain
by hand.

**Try `map` first.** `repowiki map` is LLM-free: `cli.py:40 repo_map()` never
touches `LLMClient` and `core/graph.py` has no LLM references, so it needs no
credential at all. It is free, and it may cover enough of the need to make the
rest unnecessary.

**Two rules this wrapper exists to enforce**, because getting either wrong is
expensive and neither is enforced by RepoWiki itself:

1. **Scope.** `core/` and `tools/` only. **Never** `ledger/`, `notes/`, or any
   papers directory -- `scan` ships content to a third party, and those hold
   research data. This is a mechanical allowlist, not a convention.
2. **Staleness.** A wiki behind the code is worse than none, because it is
   trusted. The source-tree hash is recorded in the output and `check` compares
   it, using the same pattern `core/submission.py` already implements.

Output is HTML under `data/wiki/`, never markdown committed to the repo, so it
cannot compete with the hand-written docs.

**On the API-key question**, recorded here so the day is not spent by accident:
`scan` (the LLM half) reads `ANTHROPIC_API_KEY` by default, which is exactly
what `credentials.scrub_environment()` deletes. That scrub cleans only the
*agent's* process, so a human running `repowiki scan` in their own shell
violates nothing technically -- but a key in the user profile is also in the
agent's environment and trips the scrub warning on every launch. Safe, noisy,
and it erodes the §2 discipline by habituation. Forking `repowiki/llm/client.py`
onto the Agent SDK is ~120 lines against a two-method async interface with four
call sites, and `core/haiku.py` is the model for the plumbing. It is a
preference, not a requirement, which is why this wrapper refuses `scan` rather
than quietly enabling it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from core import jsonl, paths
from core.cli import Cli, main
from core.errors import ConfigError, GradError, UsageError

cli = Cli(
    "grad-wiki",
    "Generate a human-facing map of core/ and tools/. Not an agent tool.",
    epilog=(
        "Scope is an allowlist, not a convention: core/ and tools/ only. ledger/,\n"
        "notes/, and data/papers/ are never passed to it -- `scan` ships content to a\n"
        "third party and those hold research data.\n\n"
        "`map` is LLM-free and needs no credential. `scan` is refused here on purpose;\n"
        "see this module's docstring for the reasoning and the ~120-line fork that\n"
        "would make it clean."
    ),
)

# The allowlist. Everything else in the workspace is research data or generated
# output, and neither belongs in a third party's context window.
SCOPE = ("core", "tools")

# Files whose content defines "the code has changed". Same idea as the
# submission hash: not a directory mtime, and not a TTL.
_SOURCE_GLOB = "*.py"


def output_dir() -> Path:
    return paths.data_dir() / "wiki"


def _manifest_path() -> Path:
    return output_dir() / "manifest.json"


def source_hash(root: Path | None = None) -> dict[str, Any]:
    """A digest over exactly the files the wiki was generated from.

    Returned with its inputs listed rather than as a bare string, so a staleness
    report can say *which* file moved instead of only that something did.
    """
    root = root or paths.root()
    digests: dict[str, str] = {}
    for name in SCOPE:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob(_SOURCE_GLOB)):
            if "__pycache__" in path.parts:
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            digests[path.relative_to(root).as_posix()] = h.hexdigest()[:16]
    canonical = "\n".join(f"{k}:{v}" for k, v in sorted(digests.items()))
    return {
        "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "files": digests,
        "scope": list(SCOPE),
    }


def _repowiki() -> str:
    found = shutil.which("repowiki")
    if found:
        return found
    raise ConfigError(
        "repowiki is not installed",
        fix="pip install -e '.[wiki]'   # pins repowiki==0.3.1",
    )


def _scope_paths(root: Path) -> list[str]:
    present = [str(root / name) for name in SCOPE if (root / name).is_dir()]
    if not present:
        raise ConfigError(
            f"none of {', '.join(SCOPE)} exist under {root}",
            fix="run this from the workspace root",
        )
    return present


# ---------------------------------------------------------------------------
def _map_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", default="html", choices=["html", "json"], help="output format")
    p.add_argument("--open", action="store_true", help="open the result in a browser")


@cli.command("map", "generate the module map (LLM-free, no credential)", setup=_map_args)
def cmd_map(args: argparse.Namespace) -> dict[str, Any]:
    """`repowiki map` over core/ and tools/ only.

    LLM-free by construction, so this costs nothing and ships nothing anywhere.
    Try it before deciding whether the rest of §20 is wanted at all.
    """
    executable = _repowiki()
    root = paths.root()
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    argv = [executable, "map", *_scope_paths(root), "--format", args.format, "--output", str(out)]
    if args.open:
        argv.append("--open")

    started = time.time()
    try:
        proc = subprocess.run(
            argv, cwd=str(root), capture_output=True, text=True, timeout=600, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise GradError(
            "wiki_timeout", "repowiki map did not finish within 10 minutes",
            exit_code=8, fix="run it by hand to see where it stalls",
        ) from exc

    log = out / "map.log"
    log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise GradError(
            "wiki_failed",
            f"repowiki map exited {proc.returncode}",
            exit_code=8,
            fix=f"read {log}",
            detail={"log": str(log), "tail": (proc.stdout or proc.stderr or "").splitlines()[-15:]},
        )

    # The staleness record. A wiki behind the code is worse than none, because
    # it is trusted -- so the inputs are recorded at generation time, not
    # reconstructed later.
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": argv[1:],
        "duration_s": round(time.time() - started, 1),
        "source": source_hash(root),
        "output_dir": str(out),
        "format": args.format,
    }
    jsonl.write_json(_manifest_path(), manifest)
    return {
        "output_dir": str(out),
        "source_hash": manifest["source"]["hash"],
        "files_covered": len(manifest["source"]["files"]),
        "log": str(log),
        "next": "python -m tools.wiki check --json",
    }


@cli.command("check", "is the generated wiki still current?")
def cmd_check(_: argparse.Namespace) -> dict[str, Any]:
    """The one-line staleness check §20 asks for.

    Exits 9 when stale, so it can be wired into anything that wants the wiki to
    be trustworthy before it is read.
    """
    manifest = jsonl.read_json(_manifest_path())
    if not manifest:
        raise GradError(
            "no_wiki", "no wiki has been generated yet", exit_code=3,
            fix="python -m tools.wiki map --json",
        )
    current = source_hash()
    recorded = manifest.get("source", {})
    if current["hash"] == recorded.get("hash"):
        return {
            "current": True,
            "source_hash": current["hash"],
            "generated_at": manifest.get("generated_at"),
            "output_dir": manifest.get("output_dir"),
        }

    before, after = recorded.get("files", {}), current["files"]
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    raise GradError(
        "wiki_stale",
        f"the wiki was generated from a different source tree: {len(changed)} file(s) differ",
        exit_code=9,
        fix="python -m tools.wiki map --json",
        detail={
            "generated_at": manifest.get("generated_at"),
            "recorded_hash": recorded.get("hash"),
            "current_hash": current["hash"],
            "changed": changed[:50],
        },
    )


@cli.command("scan", "refused on purpose; prints the reasoning")
def cmd_scan(_: argparse.Namespace) -> dict[str, Any]:
    """Not enabled here, and the reason is worth reading before enabling it.

    `repowiki scan` is the LLM half, and it reads `ANTHROPIC_API_KEY` by
    default -- the exact variable `credentials.scrub_environment()` deletes. A
    key in the user profile is also in the agent's environment and trips the
    scrub warning on every launch: safe, but noisy, and it erodes the §2
    discipline by habituation.
    """
    raise UsageError(
        "`scan` is not enabled: it reads ANTHROPIC_API_KEY, which is the credential "
        "§2 exists to keep out of this system. `map` is LLM-free and covers most of "
        "the need.",
        fix=(
            "python -m tools.wiki map --json   # free, no credential, no data leaves the machine\n"
            "     If you want the prose wiki, fork repowiki/llm/client.py onto the Agent SDK: "
            "two async methods, four call sites, response_format never used by any caller, "
            "and core/haiku.py:110 is the model for the plumbing."
        ),
    )


if __name__ == "__main__":
    main(cli)
