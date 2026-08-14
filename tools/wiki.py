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
import json
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
        "would make it clean.\n\n"
        "repowiki 0.3.1's `map` takes ONE path and only --format text|json, so each\n"
        "scope directory is a separate invocation and the HTML is rendered here."
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
    p.add_argument("--top", type=int, default=200, help="max entries per scope directory")
    p.add_argument("--open", action="store_true", help="open the generated HTML afterwards")


@cli.command("map", "generate the module map (LLM-free, no credential)", setup=_map_args)
def cmd_map(args: argparse.Namespace) -> dict[str, Any]:
    """`repowiki map` over core/ and tools/ only.

    LLM-free by construction, so this costs nothing and ships nothing anywhere.
    Try it before deciding whether the rest of §20 is wanted at all.

    **The invocation matches repowiki 0.3.1's actual contract**, which differs
    from what HANDOFF-2 §20 recorded: `map` takes exactly *one* `path` argument,
    `--format` accepts only `text` or `json`, and there is no `--output` and no
    `--open`. The handoff's "`--format html --open`" would fail immediately. So
    each scope directory is a separate invocation asking for JSON, and the HTML
    -- which §20 wants so the output never competes with the hand-written docs --
    is rendered here from that JSON.
    """
    executable = _repowiki()
    root = paths.root()
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    scopes: dict[str, Any] = {}
    commands: list[list[str]] = []
    for path in _scope_paths(root):
        name = Path(path).name
        argv = [executable, "map", path, "--format", "json", "--top", str(args.top)]
        commands.append(argv[1:])
        try:
            proc = subprocess.run(
                argv, cwd=str(root), capture_output=True, text=True, timeout=600, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise GradError(
                "wiki_timeout", f"repowiki map did not finish for {name} within 10 minutes",
                exit_code=8, fix="run it by hand to see where it stalls",
            ) from exc

        log = out / f"map.{name}.log"
        log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
        if proc.returncode != 0:
            raise GradError(
                "wiki_failed",
                f"repowiki map exited {proc.returncode} for {name}",
                exit_code=8,
                fix=f"read {log}",
                detail={
                    "command": argv[1:],
                    "log": str(log),
                    "tail": (proc.stdout or proc.stderr or "").splitlines()[-15:],
                },
            )
        try:
            scopes[name] = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            # `--format json` is what was asked for; anything else is a version
            # skew worth reporting rather than silently rendering as prose.
            scopes[name] = {"raw": (proc.stdout or "").splitlines()}

    (out / "map.json").write_text(
        json.dumps(scopes, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    html_path = _render_html(out, scopes)

    # The staleness record. A wiki behind the code is worse than none, because
    # it is trusted -- so the inputs are recorded at generation time, not
    # reconstructed later.
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commands": commands,
        "duration_s": round(time.time() - started, 1),
        "source": source_hash(root),
        "output_dir": str(out),
        "html": str(html_path),
    }
    jsonl.write_json(_manifest_path(), manifest)

    if args.open:
        import webbrowser  # noqa: PLC0415

        webbrowser.open(html_path.as_uri())

    return {
        "output_dir": str(out),
        "html": str(html_path),
        "scopes": sorted(scopes),
        "source_hash": manifest["source"]["hash"],
        "files_covered": len(manifest["source"]["files"]),
        "next": "python -m tools.wiki check --json",
    }


def _render_html(out: Path, scopes: dict[str, Any]) -> Path:
    """HTML, generated here rather than by repowiki (which cannot emit it).

    §20 wants HTML specifically so the output never competes with the
    hand-written docs the way a committed markdown file would.
    """
    rows = []
    for scope, payload in sorted(scopes.items()):
        files = payload.get("files") if isinstance(payload, dict) else None
        rows.append(f"<h2>{_esc(scope)}/</h2>")
        if not isinstance(files, list) or not files:
            rows.append(f"<pre>{_esc(json.dumps(payload, indent=2, default=str)[:20000])}</pre>")
            continue
        rows.append("<table><tr><th>file</th><th>rank</th><th>lang</th><th>lines</th></tr>")
        for entry in files:
            if not isinstance(entry, dict):
                continue
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    _esc(str(entry.get("path", entry.get("file", "")))),
                    _esc(str(round(entry["rank"], 5) if isinstance(entry.get("rank"), float) else entry.get("rank", ""))),
                    _esc(str(entry.get("language", ""))),
                    _esc(str(entry.get("lines", ""))),
                )
            )
        rows.append("</table>")

    html = (
        "<!doctype html><meta charset='utf-8'><title>Grad repo map</title>"
        "<style>body{font:14px/1.6 system-ui;margin:2rem auto;max-width:60rem;"
        "background:#0b0f14;color:#dbe4ee}table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #1e2732;padding:.3rem .5rem;text-align:left}"
        "h2{margin-top:2rem}code,pre{background:#121820;padding:.5rem;overflow:auto}</style>"
        "<h1>Grad repo map</h1>"
        "<p>Generated by <code>python -m tools.wiki map</code> over "
        f"{_esc(', '.join(SCOPE))} only. Human-facing; not an agent tool. "
        "Check freshness with <code>python -m tools.wiki check</code>.</p>"
        + "\n".join(rows)
    )
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _esc(text: str) -> str:
    import html as _html  # noqa: PLC0415

    return _html.escape(str(text), quote=True)


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
