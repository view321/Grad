"""The written half of a project wiki: one grounded page at a time.

`core/projwiki.py` extracts what is true. This asks a model to explain it, and
almost every decision here is about making that a *narrow* job rather than an
open one.

**It is not an agent, and that is the point.** The main loop is a research agent
with Bash, a ledger and a budget; a second one that could also read the
repository would be strictly worse at everything it does and would cost tokens
to be worse. So each page is a single call with exactly one tool -- the one that
returns the page -- and every file-touching tool explicitly denied. There is no
loop, no planning step and nothing to steer: the facts go in, a page comes back
or the call fails.

**Every page is written against a fact sheet it did not choose.** The prompt
carries the extracted symbols, the spec text, the run records and the
expectations, and says so: the model's job is to explain the arrangement, not to
recall the codebase. This is what separates a wiki from a plausible essay about
a wiki. A page that wants to claim something not in its fact sheet has one
honest place to put it -- `open_questions` -- and the schema makes that cheaper
than smuggling it into prose.

**Citations are required and checkable.** Every section names the `refs` it
rests on: `train.py:41`, `run-2026…`, `exp-…`, `spec.toml`. `verify_refs` checks
them against the fact sheet after the fact and marks the ones that match nothing,
so a page that drifts is visibly a page that drifted rather than a page that
reads well. The model is told this happens, which is itself most of why it
mostly does not need to happen.

**Failure is a missing page, never a wrong one.** A call that produces nothing
after its retry leaves the page absent with its error recorded;
`tools/projwiki.py` writes the rest of the wiki anyway. Half a wiki whose gaps
are visible is worth more than a whole one with an invented page in it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core import haiku, quota_log

#: How many module pages one build will write. A bound on cost that a person
#: should be able to predict before pressing the button: pipelines here run to a
#: handful of modules, and a pipeline with forty is one where the overview page
#: is the useful artifact anyway. What is dropped is reported, never silent.
MAX_MODULE_PAGES = 12

#: Test modules get no page of their own. They are described *in* the module
#: page for the code they test, where "what is checked" belongs, and a wiki whose
#: back half is one page per `test_*.py` buries the three pages worth reading.
SKIP_TEST_PAGES = True


SYSTEM = """\
You write one page of a technical wiki for a machine-learning research project.

You are given a FACT SHEET extracted from the project: its source files, the \
symbols in them, the submission spec, and the ledger of predictions and runs. \
Everything in it was read off disk. Nothing else about this project is known to \
you, and you must not supply anything from memory -- not a library's behaviour, \
not a "typical" hyperparameter, not what a file with that name usually contains.

Write for someone who is competent at ML and has never seen this project. What \
they need is the argument: what this is trying to establish, why the pieces are \
arranged this way, what a number means when it comes back. What they do not \
need is a paraphrase of the code -- they can read it, and a page that restates \
`def train(...)` as "the train function trains" has cost them time.

Rules, in the order they matter:

1. Every section carries `refs`: the exact fact-sheet items it rests on, as \
   `file.py:LINE`, `spec.toml`, `run-<id>`, or `exp-<id>`. These are checked \
   against the fact sheet after you return them. A section with no ref is a \
   section with nothing behind it.
2. If the fact sheet does not settle something, it goes in `open_questions` as \
   a question. Never write around a gap; naming it is more useful than filling \
   it, and a reader who sees the gap can go and look.
3. Prefer the specific: the actual number, the actual file, the actual run id. \
   "Several experiments were run" is worse than "four runs, three collected".
4. Do not invent structure the project does not have. No roadmap, no \
   "future work", no praise for the design.

Call submit_page exactly once, then stop. Do not explain yourself."""


PAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string", "description": "1-2 sentences. What this page is about."},
        "sections": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "string", "description": "Markdown. Two to six sentences."},
                    "refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "body", "refs"],
            },
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What the fact sheet does not settle. Empty is a valid answer.",
        },
    },
    "required": ["title", "summary", "sections"],
}


def _validate_page(args: dict[str, Any]) -> str | None:
    if not str(args.get("title") or "").strip():
        return "title must be a non-empty string"
    if not str(args.get("summary") or "").strip():
        return "summary must be a non-empty string"
    sections = args.get("sections")
    if not isinstance(sections, list) or not sections:
        return "sections must be a non-empty list"
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            return f"sections[{index}] must be an object with heading, body, refs"
        if not str(section.get("heading") or "").strip():
            return f"sections[{index}].heading is required"
        if not str(section.get("body") or "").strip():
            return f"sections[{index}].body is required"
        refs = section.get("refs")
        if not isinstance(refs, list) or not any(str(r).strip() for r in refs):
            # Enforced here rather than left to the prompt: a returned error is
            # what makes the model try again, and a page of unsourced prose is
            # exactly the artifact this whole design exists to not produce.
            return (
                f"sections[{index}].refs must name at least one fact-sheet item "
                "(file.py:LINE, spec.toml, run-<id>, exp-<id>)"
            )
    return None


# ---------------------------------------------------------------------------
# fact sheets: what each page is allowed to know
# ---------------------------------------------------------------------------
def _symbol_lines(module: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for symbol in module["symbols"]:
        doc = f"  -- {symbol['doc']}" if symbol.get("doc") else ""
        out.append(f"  {module['path']}:{symbol['line']}  {symbol['signature']}{doc}")
        for method in symbol.get("methods") or []:
            out.append(f"    {module['path']}:{method['line']}  {method['signature']}")
    return out


def _ledger_lines(ledger: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for exp in ledger["expectations"]:
        band = f"[{exp['low']}, {exp['high']}]" if exp["low"] is not None else exp["direction"]
        mark = " FALSIFIED" if exp["falsified"] else ""
        out.append(
            f"  {exp['id']}  predicts {exp['quantity']} in {band} "
            f"(confidence {exp['confidence']}, task {exp['task']}){mark}"
        )
        for basis in exp["basis"]:
            out.append(f"      basis: {basis['paper']} -- {basis['locator']} = {basis['value']}")
    for run in ledger["runs"]:
        results = ", ".join(f"{k}={v}" for k, v in list(run["results"].items())[:8]) or "no results"
        out.append(
            f"  {run['id']}  {run['status']} on {run['platform']} "
            f"(task {run['task']}, expectation {run['expectation_id'] or 'none'}): {results}"
        )
        for deviation in run["deviations"][:6]:
            out.append(
                f"      deviation: {deviation.get('quantity')} observed "
                f"{deviation.get('observed')} vs predicted {deviation.get('predicted')} "
                f"(in range: {deviation.get('in_range')})"
            )
    return out


def _project_header(facts: dict[str, Any]) -> list[str]:
    lines = [f"PROJECT: {facts['project']}", ""]
    for name, text in (facts.get("docs") or {}).items():
        lines += [f"--- {name} (authored by the researcher) ---", text.strip()[:3000], ""]
    return lines


def overview_sheet(facts: dict[str, Any]) -> str:
    """Everything, thinly: the whole shape of the project in one prompt."""
    lines = _project_header(facts)
    for pipeline in facts["pipelines"]:
        lines.append(f"--- pipeline {pipeline['name']} ---")
        for spec in pipeline["specs"]:
            lines += [f"  spec {spec['name']} (verbatim, comments included):", "", spec["text"], ""]
        for module in pipeline["modules"]:
            flag = " [test]" if module["is_test"] else ("" if module["reachable"] else " [not imported by any entrypoint]")
            lines.append(f"  {module['path']} ({module['lines']} lines){flag}")
            if module["doc"]:
                lines.append(f"      module docstring: {module['doc'][:400]}")
            lines += _symbol_lines(module)[:24]
        lines.append("")
    lines += ["--- ledger ---", *(_ledger_lines(facts["ledger"]) or ["  nothing recorded yet"]), ""]
    if facts["papers"]:
        lines.append("--- papers the predictions cite ---")
        for paper in facts["papers"]:
            lines.append(f"  {paper['cited_as']}: {paper['title'] or '(title not in the local index)'}")
    return "\n".join(lines)


def module_sheet(facts: dict[str, Any], pipeline: dict[str, Any], module: dict[str, Any]) -> str:
    """One module in full, plus just enough of its surroundings to place it."""
    importers = [m["path"] for m in pipeline["modules"] if module["path"].removesuffix(".py") in m["imports"]]
    lines = [
        f"PROJECT: {facts['project']}    PIPELINE: {pipeline['name']}",
        f"MODULE: {module['path']} ({module['lines']} lines)",
        "",
        f"Imported by: {', '.join(importers) or 'nothing in this pipeline'}",
        f"Imports: {', '.join(module['imports']) or 'no sibling modules'}",
        f"Reached from an entrypoint: {'yes' if module['reachable'] else 'no'}",
        "",
        "--- symbols ---",
        *_symbol_lines(module),
        "",
        f"--- source{' (truncated)' if module['truncated'] else ''} ---",
        module["source"],
        "",
    ]
    tests = [m for m in pipeline["modules"] if m["is_test"]]
    if tests and not module["is_test"]:
        lines.append("--- what the tests in this pipeline check ---")
        for test in tests:
            lines += _symbol_lines(test)[:20]
        lines.append("")
    for spec in pipeline["specs"]:
        if spec.get("entrypoint") == module["path"]:
            lines += [f"--- {spec['name']}, which names this file as its entrypoint ---", spec["text"], ""]
    return "\n".join(lines)


def run_path_sheet(facts: dict[str, Any], pipeline: dict[str, Any]) -> str:
    """How a run happens: spec to entrypoint to metrics to ledger."""
    lines = [f"PROJECT: {facts['project']}    PIPELINE: {pipeline['name']}", ""]
    for spec in pipeline["specs"]:
        lines += [f"--- {spec['name']} (verbatim) ---", spec["text"], ""]
    lines.append("--- modules, and what each imports ---")
    for module in pipeline["modules"]:
        lines.append(
            f"  {module['path']}: imports {', '.join(module['imports']) or 'nothing local'}"
            f"{' [test]' if module['is_test'] else ''}"
        )
    lines += ["", "--- files in the pipeline directory ---"]
    for row in pipeline["tree"]:
        lines.append(f"  {row['path']} ({row['bytes']} bytes)")
    lines += ["", "--- what the ledger recorded for these runs ---", *(_ledger_lines(facts["ledger"]) or ["  nothing yet"])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the pages of a wiki
# ---------------------------------------------------------------------------
def plan(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Which pages this project gets, before any of them is written.

    Separate from writing them so a caller can price the build, show it, and
    -- in `--no-prose` mode -- produce the whole retrieved wiki without spending
    a token. The order is the order they appear in the wiki.
    """
    pages: list[dict[str, Any]] = [
        {
            "id": "overview",
            "kind": "overview",
            "title": f"{facts['project']} — overview",
            "instruction": (
                "Write the overview. What is this project trying to establish, what does the "
                "code do to establish it, and where does it stand? Use the ledger for the last "
                "part: predictions registered, runs collected, anything falsified or still "
                "awaiting a verdict."
            ),
        }
    ]
    dropped: list[str] = []
    for pipeline in facts["pipelines"]:
        if pipeline["specs"]:
            pages.append(
                {
                    "id": f"run-path-{pipeline['name']}",
                    "kind": "run-path",
                    "pipeline": pipeline["name"],
                    "title": f"{pipeline['name']} — how a run happens",
                    "instruction": (
                        "Trace one run end to end: what the spec asks for, which file is the "
                        "entrypoint, what it produces, and how that becomes a row in the ledger. "
                        "The spec's comments are the reasoning behind its values -- use them, and "
                        "say which choices were made on evidence and which are stated without one."
                    ),
                }
            )
        for module in pipeline["modules"]:
            if SKIP_TEST_PAGES and module["is_test"]:
                continue
            if len([p for p in pages if p["kind"] == "module"]) >= MAX_MODULE_PAGES:
                dropped.append(f"{pipeline['name']}/{module['path']}")
                continue
            pages.append(
                {
                    "id": f"module-{pipeline['name']}-{module['path']}",
                    "kind": "module",
                    "pipeline": pipeline["name"],
                    "module": module["path"],
                    "title": f"{pipeline['name']}/{module['path']}",
                    "instruction": (
                        "Explain this module: what it is responsible for, how its pieces fit "
                        "together, and what a reader has to understand before changing it. Name "
                        "symbols by file and line. Skip anything the signature already says."
                    ),
                }
            )
    if dropped:
        pages[0].setdefault("dropped", []).extend(dropped)
    return pages


def sheet_for(facts: dict[str, Any], page: dict[str, Any]) -> str:
    """The fact sheet one page is written against."""
    if page["kind"] == "overview":
        return overview_sheet(facts)
    pipeline = next(p for p in facts["pipelines"] if p["name"] == page["pipeline"])
    if page["kind"] == "run-path":
        return run_path_sheet(facts, pipeline)
    module = next(m for m in pipeline["modules"] if m["path"] == page["module"])
    return module_sheet(facts, pipeline, module)


def write_page(facts: dict[str, Any], page: dict[str, Any], *, model: str, log_name: str) -> dict[str, Any]:
    """One page. Raises nothing the caller has to catch page-by-page."""
    sheet = sheet_for(facts, page)
    written = haiku.structured(
        stage=quota_log.STAGE_WIKI,
        tool_name="submit_page",
        tool_description="Return the finished wiki page",
        tool_schema=PAGE_SCHEMA,
        validate=_validate_page,
        system_prompt=SYSTEM,
        user_prompt=f"{page['instruction']}\n\n=== FACT SHEET ===\n\n{sheet}",
        model=model,
        role="wiki",
        log_name=log_name,
    )
    return {
        **page,
        "title": str(written.get("title") or page["title"]),
        "summary": written.get("summary", ""),
        "sections": written.get("sections") or [],
        "open_questions": written.get("open_questions") or [],
        "unverified_refs": verify_refs(facts, written),
    }


# ---------------------------------------------------------------------------
# checking the citations
# ---------------------------------------------------------------------------
_REF_FILE = re.compile(r"^(?P<path>[\w./\\-]+\.\w+)(?::(?P<line>\d+))?$")


def known_refs(facts: dict[str, Any]) -> set[str]:
    """Everything a page may legitimately cite, as written."""
    out: set[str] = set()
    for pipeline in facts["pipelines"]:
        for row in pipeline["tree"]:
            out.add(row["path"])
            out.add(f"{pipeline['name']}/{row['path']}")
        for module in pipeline["modules"]:
            for symbol in module["symbols"]:
                out.add(f"{module['path']}:{symbol['line']}")
                for method in symbol.get("methods") or []:
                    out.add(f"{module['path']}:{method['line']}")
    for name in (facts.get("docs") or {}):
        out.add(name)
    for exp in facts["ledger"]["expectations"]:
        out.add(exp["id"])
    for run in facts["ledger"]["runs"]:
        out.add(run["id"])
    return out


def verify_refs(facts: dict[str, Any], written: dict[str, Any]) -> list[str]:
    """The refs a page gave that match nothing extracted.

    Not an error and not a rejection: a line number a few lines off, or a file
    named without its directory, is a page that is still substantially right,
    and throwing it away would trade a small inaccuracy for a missing page. It
    is *reported*, in the page and in the build's envelope, because a reader
    deciding how much to trust a section is entitled to know which of its
    citations could not be resolved.

    A file reference resolves if the file exists, whatever line was named: line
    numbers move under an edit that leaves the page true, and being strict about
    them would mark the whole wiki unverified after a one-line change.
    """
    known = known_refs(facts)
    files = {ref.split(":")[0] for ref in known if ":" in ref} | {
        ref for ref in known if "." in ref and ":" not in ref
    }
    unresolved: list[str] = []
    for section in written.get("sections") or []:
        for raw in section.get("refs") or []:
            ref = str(raw).strip().strip("`")
            if not ref or ref in known:
                continue
            match = _REF_FILE.match(ref)
            if match and (match.group("path") in files or match.group("path").split("/")[-1] in files):
                continue
            unresolved.append(ref)
    return sorted(set(unresolved))


def as_markdown(page: dict[str, Any]) -> str:
    """One page as markdown, refs and all. What the UI renders and what a
    person reads if they open the file directly."""
    out = [f"# {page['title']}", "", page.get("summary", ""), ""]
    for section in page.get("sections") or []:
        out += [f"## {section['heading']}", "", section["body"], ""]
        refs = ", ".join(f"`{r}`" for r in section.get("refs") or [])
        if refs:
            out += [f"*{refs}*", ""]
    if page.get("open_questions"):
        out += ["## Open questions", ""]
        out += [f"- {q}" for q in page["open_questions"]]
        out.append("")
    if page.get("unverified_refs"):
        out += [
            "---",
            "",
            "*Citations that matched nothing in the extracted facts: "
            + ", ".join(f"`{r}`" for r in page["unverified_refs"])
            + ". Treat the sentences resting on them with more suspicion than the rest.*",
            "",
        ]
    return "\n".join(out)


def as_json(pages: list[dict[str, Any]]) -> str:
    return json.dumps(pages, indent=2, ensure_ascii=False, default=str)
