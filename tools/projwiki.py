"""grad-projwiki -- a wiki for the research code the agent generates.

Not `tools/wiki.py`. That one maps *Grad's own* `core/` and `tools/` with
repowiki, for a person maintaining Grad. This one documents `pipelines/<name>/`
-- the training script, the model, the data loader, the probe, the tests the
agent wrote to answer a research question -- for a person reacquiring a project
they have not looked at in three weeks. Different subject, different reader,
different lifetime: the code here changes every time an experiment does.

**Half retrieved, half generated, and the halves stay visible.**
`core/projwiki.py` extracts the facts (tree, spec, imports, symbols with line
numbers, expectations, runs, cited papers). `core/wikigen.py` asks a model to
explain those facts and nothing else, one page per call, with citations that are
checked afterwards. The extracted half is written to `facts.json` beside the
prose, so what a page was grounded on is always inspectable -- and `--no-prose`
produces the retrieved wiki alone, free, offline, and useful on its own.

**Why not just ask the agent.** The main loop can read the repository and often
should; what it cannot do is leave behind a stable artifact. A page it writes in
a chat turn is gone when the session compacts, costs a research turn to produce,
and is written against whatever happened to be in context. This is a document
with a source hash, regenerated deliberately, and legible to someone who was
never in the conversation.

**The staleness rule is `tools/wiki.py`'s, because it is the right one.** A wiki
behind the code is worse than none, since it is trusted. The digest covers the
pipeline sources, the specs and the project's authored documents -- and not the
ledger, because a new run arriving does not make a page about the architecture
wrong.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from core import config as config_mod, jsonl, paths, projwiki, wikigen
from core.cli import Cli, main
from core.errors import GradError, UsageError

cli = Cli(
    "grad-projwiki",
    "Build and read the wiki for a research project's generated code.",
    epilog=(
        "Scope is `pipelines/<name>/` plus the project's own PLAN/MEMORY/TODO and its\n"
        "ledger entries. `notes/` and `data/papers/` are never read: the generated half\n"
        "ships what it is given to a model.\n\n"
        "  python -m tools.projwiki build --project minimamba --json\n"
        "  python -m tools.projwiki build --project minimamba --no-prose --json  # free, offline\n"
        "  python -m tools.projwiki check --project minimamba --json\n"
        "  python -m tools.projwiki show  --project minimamba --page overview --json\n"
    ),
)


def output_dir(project_id: str) -> Path:
    """`data/wiki/projects/<id>/`. Under `data/` rather than in the project
    folder because it is generated: `core/projects.py` keeps the authored files
    somewhere nothing overwrites them, and a wiki that appeared in the same
    directory as `PLAN.md` would blur exactly that line."""
    return paths.data_dir() / "wiki" / "projects" / project_id


def _manifest_path(project_id: str) -> Path:
    return output_dir(project_id) / "manifest.json"


def manifest(project_id: str) -> dict[str, Any] | None:
    record = jsonl.read_json(_manifest_path(project_id))
    return record if isinstance(record, dict) else None


def _resolve_project(given: str | None) -> str:
    from core import budget as budget_mod

    project = (given or "").strip() or budget_mod.current_project()
    if not project:
        raise UsageError(
            "no project selected and none given",
            fix="python -m tools.budget use <id> --json   # or pass --project <id>",
        )
    return project


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="the project id; the selected one by default")
    p.add_argument(
        "--no-prose",
        action="store_true",
        help="extract the facts and write the index, but call no model. Free and offline",
    )
    p.add_argument("--model", help="override the model for the wiki role")
    p.add_argument(
        "--page",
        action="append",
        default=[],
        metavar="ID",
        help="rebuild only these page ids, keeping the rest of the existing wiki",
    )


@cli.command("build", "extract the facts and write the pages", setup=_build_args)
def cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    """Extract, then write. The extraction always happens; the writing is what
    `--no-prose` skips.

    **A failed page is a missing page, not a failed build.** Each call is caught
    on its own and recorded with its error, because the alternative -- one
    refusal taking down a build that had already written six good pages -- would
    make the whole thing feel unreliable for a reason that has nothing to do
    with the pages that worked. The envelope names what is missing, so nothing
    is quietly absent.
    """
    project = _resolve_project(args.project)
    cfg = config_mod.load()
    started = time.time()

    facts = projwiki.collect(project)
    out = output_dir(project)
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.json").write_text(wikigen.as_json([facts]), encoding="utf-8")

    planned = wikigen.plan(facts)
    if not facts["pipelines"]:
        # An honest empty state rather than an overview page about nothing. A
        # project whose code has not been written yet is a normal thing to find.
        raise GradError(
            "no_pipeline",
            f"project {project!r} has no pipeline directory: nothing under pipelines/ shares "
            "its name and no run bound to it names one",
            exit_code=3,
            fix=f"mkdir {paths.root() / 'pipelines' / project}   # then write a spec.toml in it",
            detail={"project": project, "looked_in": str(paths.root() / "pipelines")},
        )

    existing = {p["id"]: p for p in (_read_pages(project) or [])}
    if args.page:
        unknown = sorted(set(args.page) - {p["id"] for p in planned})
        if unknown:
            raise UsageError(
                f"unknown page id(s): {', '.join(unknown)}",
                fix="python -m tools.projwiki check --project "
                f"{project} --json   # lists every page id",
            )
    wanted = {p["id"] for p in planned if not args.page or p["id"] in set(args.page)}

    model = (args.model or "").strip() or cfg.model_for("wiki")
    log_name = f"wiki-{project}"
    pages: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for page in planned:
        if page["id"] not in wanted:
            # Kept as it was written, so `--page overview` costs one call rather
            # than nine and leaves the other eight pages intact.
            pages.append(existing.get(page["id"], {**page, "sections": [], "summary": ""}))
            continue
        if args.no_prose:
            pages.append({**page, "summary": "", "sections": [], "prose": False})
            continue
        try:
            pages.append(wikigen.write_page(facts, page, model=model, log_name=log_name))
        except Exception as exc:  # noqa: BLE001 - one page's failure is one page's
            failed.append({"page": page["id"], "error": f"{type(exc).__name__}: {exc}"})
            pages.append({**page, "summary": "", "sections": [], "error": str(exc)})

    (out / "pages.json").write_text(wikigen.as_json(pages), encoding="utf-8")
    for page in pages:
        if page.get("sections"):
            (out / f"{_slug(page['id'])}.md").write_text(
                wikigen.as_markdown(page), encoding="utf-8"
            )

    record = {
        "project": project,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - started, 1),
        "model": None if args.no_prose else model,
        "prose": not args.no_prose,
        "source": facts["source"],
        "output_dir": str(out),
        "pages": [
            {
                "id": p["id"],
                "kind": p["kind"],
                "title": p["title"],
                "written": bool(p.get("sections")),
                "unverified_refs": p.get("unverified_refs") or [],
            }
            for p in pages
        ],
    }
    jsonl.write_json(_manifest_path(project), record)

    unverified = sorted({r for p in pages for r in (p.get("unverified_refs") or [])})
    return {
        "project": project,
        "output_dir": str(out),
        "pages_written": len([p for p in pages if p.get("sections")]),
        "pages_planned": len(planned),
        "failed": failed,
        # Never silent: a build that quietly stopped at the page cap would read
        # as a wiki that covers everything.
        "dropped_modules": planned[0].get("dropped") or [],
        "unverified_refs": unverified,
        "warnings": facts["warnings"],
        "source_hash": facts["source"]["hash"],
        "model": record["model"],
        "next": f"python -m tools.projwiki show --project {project} --page overview --json",
    }


def _slug(page_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in page_id)[:80]


def _read_pages(project_id: str) -> list[dict[str, Any]] | None:
    record = jsonl.read_json(output_dir(project_id) / "pages.json")
    return record if isinstance(record, list) else None


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
@cli.command(
    "check",
    "is this project's wiki still current?",
    setup=lambda p: p.add_argument("--project", help="the project id; the selected one by default"),
)
def cmd_check(args: argparse.Namespace) -> dict[str, Any]:
    """Exits 9 when stale, so anything that wants the wiki to be trustworthy
    before reading it can say so."""
    project = _resolve_project(args.project)
    record = manifest(project)
    if not record:
        raise GradError(
            "no_wiki",
            f"no wiki has been built for {project!r} yet",
            exit_code=3,
            fix=f"python -m tools.projwiki build --project {project} --json",
        )
    current = projwiki.source_hash(project)
    recorded = record.get("source") or {}
    pages = record.get("pages") or []
    if current["hash"] == recorded.get("hash"):
        return {
            "current": True,
            "project": project,
            "source_hash": current["hash"],
            "generated_at": record.get("generated_at"),
            "output_dir": record.get("output_dir"),
            "pages": pages,
        }
    before, after = recorded.get("files") or {}, current["files"]
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    raise GradError(
        "wiki_stale",
        f"the wiki for {project!r} was built from a different source tree: "
        f"{len(changed)} file(s) differ",
        exit_code=9,
        fix=f"python -m tools.projwiki build --project {project} --json",
        detail={
            "project": project,
            "generated_at": record.get("generated_at"),
            "recorded_hash": recorded.get("hash"),
            "current_hash": current["hash"],
            "changed": changed[:50],
            "pages": [p["id"] for p in pages],
        },
    )


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------
def _show_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="the project id; the selected one by default")
    p.add_argument("--page", help="a page id; the index of pages by default")
    p.add_argument("--facts", action="store_true", help="the extracted facts instead of the prose")


@cli.command("show", "read one page, the index, or the facts underneath", setup=_show_args)
def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    project = _resolve_project(args.project)
    if args.facts:
        record = jsonl.read_json(output_dir(project) / "facts.json")
        if not record:
            raise GradError(
                "no_wiki",
                f"no facts extracted for {project!r} yet",
                exit_code=3,
                fix=f"python -m tools.projwiki build --project {project} --no-prose --json",
            )
        return {"project": project, "facts": record[0] if isinstance(record, list) else record}

    pages = _read_pages(project)
    if pages is None:
        raise GradError(
            "no_wiki",
            f"no wiki has been built for {project!r} yet",
            exit_code=3,
            fix=f"python -m tools.projwiki build --project {project} --json",
        )
    if not args.page:
        return {
            "project": project,
            "pages": [
                {"id": p["id"], "kind": p["kind"], "title": p["title"], "summary": p.get("summary", "")}
                for p in pages
            ],
        }
    found = next((p for p in pages if p["id"] == args.page), None)
    if found is None:
        raise GradError(
            "no_page",
            f"{project!r} has no page {args.page!r}",
            exit_code=3,
            fix=f"python -m tools.projwiki show --project {project} --json   # lists the pages",
            detail={"pages": [p["id"] for p in pages]},
        )
    return {"project": project, "page": found, "markdown": wikigen.as_markdown(found)}


if __name__ == "__main__":
    main(cli)
