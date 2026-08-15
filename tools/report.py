"""grad-report -- the scientific report (HANDOFF-2 §22).

    "`check` refuses; it does not warn. A report generator is where this
     system's epistemics either hold or collapse -- the whole design exists to
     stop the user believing results too easily, and a paper generator is a
     machine for asserting them confidently."

**Built, not adopted.** Every surveyed harness reconstructs provenance from
unstructured experiment logs. Grad's is already structured -- expectations with
`basis` and `comparability`, runs with results and `deviations`, verdicts with
notes, figures, corpus paper ids. Adopting AI Scientist v2, PaperOrchestra,
Denario, Camyla, Jr. AI Scientist, or CiteLLM means discarding that advantage
and conforming to its log format.

What *was* worth stealing, and from whom:

  * **Camyla** -- the two-pass citation flow. `write` emits `[CITE:keyword]`
    placeholders; `cite` extracts a context window around each and verifies the
    candidate's title and abstract against that context. Much better than citing
    inline, where a model invents a plausible reference in the moment.
  * **PaperOrchestra** -- the constraint set (keys match the bib exactly, no
    fabricated results, compile-clean LaTeX), encoded as validation rather than
    as prompt text.
  * **Denario** -- progressive versions. It emits four because unattended LaTeX
    does not reliably compile; `build` checkpoints for the same reason.
  * **AI Scientist v2** -- the role split, which maps onto §16: `report` writes
    the prose, `cite` resolves the citations.
  * **Jr. AI Scientist** -- draft -> reflect -> adjust inside a template
    directory.

The pipeline, and only `write` costs anything:

    draft  deterministic skeleton from the ledger. No model. Useful on its own.
    write  prose plus [CITE:...] placeholders.            <- the only paid step
    cite   resolve, verify, emit references.bib.
    check  the gate. Refuses on an unresolved claim, an unverified citation,
           an unjudged deviation, or LaTeX that will not compile.
    build  PDF, with progressive checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core import (
    budget,
    config as config_mod,
    corpus,
    http,
    quota_log,
    report as report_lib,
)
from core.cli import Cli, main
from core.errors import EXIT_CHECK_FAILED, ConfigError, GradError, UpstreamError

cli = Cli(
    "grad-report",
    "Generate a scientific report from the ledger, with every number and every "
    "citation mechanically traceable.",
    epilog=(
        "`check` enforces four rules, in order:\n"
        "  1. every \\gradnum{} key resolves to a (run_id, quantity) in the ledger,\n"
        "     with a matching value;\n"
        "  2. every \\cite{} key exists in references.bib, and every bib entry came\n"
        "     from the corpus or a verified S2 id;\n"
        "  3. no cited run has an unjudged deviation;\n"
        "  4. the LaTeX compiles clean.\n\n"
        "Rule 3 is the one most in the spirit of this system: you should not be able\n"
        "to write up a result you have not judged.\n\n"
        "`draft` costs nothing and needs no model. Run it first."
    ),
)


def _project(args: argparse.Namespace) -> str:
    return budget.resolve_or_fail(getattr(args, "project", None), what="a report")


def _project_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="the project to report on (defaults to the current one)")


# ---------------------------------------------------------------------------
# draft
# ---------------------------------------------------------------------------
_PREAMBLE = r"""\documentclass[%(classoptions)s]{%(documentclass)s}
%(style)s\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}

%% Every asserted number goes through this macro. It renders as the number and
%% it is what `report check` verifies against the ledger: the key indexes
%% claims.json, which maps it to a (run_id, quantity). A number typed directly
%% into the prose is a number nothing can check.
\newcommand{\gradnum}[1]{\csname gradval@#1\endcsname}
\input{claims.tex}

\title{%(title)s}
\author{Grad}
\date{\today}

\begin{document}
\maketitle
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "claim"


@cli.command("draft", "the skeleton, straight from the ledger (no model)", setup=_project_arg)
def cmd_draft(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic and free.

    "It is useful on its own and costs nothing." Every expectation, its runs,
    its deviations, its verdict, its figures -- including the runs that went
    badly, because a skeleton that quietly omits them is a skeleton that invites
    writing up only what worked.
    """
    project_id = _project(args)
    evidence = report_lib.project_evidence(project_id)
    files = report_lib.paths_for(project_id)
    files["dir"].mkdir(parents=True, exist_ok=True)

    claims: dict[str, Any] = {}
    body: list[str] = []
    body.append(r"\section{Results}")
    body.append("")

    if not evidence["expectations"]:
        body.append(
            "No expectation in this project has a bound run yet. "
            "The skeleton is empty on purpose rather than invented.\n"
        )

    for block in evidence["expectations"]:
        exp = block["expectation"]
        body.append(rf"\subsection{{{_tex_escape(exp.get('claim') or exp['quantity'])}}}")
        body.append("")
        body.append(rf"\label{{exp:{_slug(exp['id'])}}}")
        body.append("")
        predicted = exp.get("predicted") or {}
        body.append(
            "Pre-registered prediction: "
            + _tex_escape(_describe_prediction(exp["quantity"], predicted))
            + f" (confidence: {exp.get('confidence', 'unstated')})."
        )
        if exp.get("comparability"):
            body.append("")
            body.append("Comparability: " + _tex_escape(exp["comparability"]))
        for basis in exp.get("basis") or []:
            body.append("")
            body.append(
                "Basis: "
                + _tex_escape(
                    f"{basis.get('paper', '?')} ({basis.get('locator', '')}) reports "
                    f"{basis.get('value')} under {basis.get('conditions', 'unstated conditions')}"
                )
                + f" [CITE:{basis.get('paper', 'unknown')}]"
            )
        body.append("")

        for run in block["runs"]:
            for quantity, value in sorted((run["results"] or {}).items()):
                key = f"{_slug(run['id'])}-{_slug(quantity)}"
                claims[key] = {
                    "run_id": run["id"],
                    "quantity": quantity,
                    "value": value,
                    "task": run["task"],
                }
                body.append(
                    rf"Run \texttt{{{_tex_escape(run['id'])}}} measured "
                    rf"{_tex_escape(quantity)} = \gradnum{{{key}}}."
                )
            for dev in run["deviations"]:
                verdict = dev.get("verdict")
                body.append("")
                body.append(
                    _tex_escape(
                        f"Deviation on {dev.get('quantity')}: "
                        f"{'in range' if dev.get('in_range') is True else 'out of range or unsettled'}"
                        + (f", judged {verdict}: {dev.get('note') or ''}" if verdict
                           else ". NOT YET JUDGED -- `report check` will refuse while this stands.")
                    )
                )
            body.append("")

    if evidence["figures"]:
        body.append(r"\section{Figures}")
        for figure in evidence["figures"]:
            rel = Path(figure).as_posix()
            body.append(r"\begin{figure}[h]\centering")
            body.append(rf"\includegraphics[width=0.7\linewidth]{{{rel}}}")
            body.append(rf"\caption{{{_tex_escape(Path(figure).name)}}}")
            body.append(r"\end{figure}")

    if evidence["unbound_runs"]:
        body.append(r"\section{Runs with no bound expectation}")
        body.append(
            _tex_escape(
                "These ran without a pre-registered prediction and are listed so their "
                "absence from the results above is visible rather than silent: "
                + ", ".join(r["id"] for r in evidence["unbound_runs"])
            )
        )

    cfg = config_mod.load()
    style = str(cfg.get("report", "style", "") or "")
    tex = (
        _PREAMBLE
        % {
            "title": _tex_escape(f"Report: {project_id}"),
            "documentclass": cfg.get("report", "documentclass", "article"),
            "classoptions": cfg.get("report", "classoptions", "11pt"),
            # A conference style is a vendored .sty dropped into the report
            # directory, named here. Empty by default so the skeleton compiles
            # on a stock TeX installation with nothing vendored.
            "style": f"\\usepackage{{{style}}}\n" if style else "",
        }
    ) + "\n".join(body)
    tex += (
        f"\n\n\\bibliographystyle{{{cfg.get('report', 'bibstyle', 'plainnat')}}}\n"
        "\\bibliography{references}\n\\end{document}\n"
    )

    files["tex"].write_text(tex, encoding="utf-8")
    files["claims"].write_text(json.dumps(claims, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_claims_tex(project_id, claims)

    return {
        "project": project_id,
        "tex": str(files["tex"]),
        "claims": str(files["claims"]),
        "claim_count": len(claims),
        "expectations": len(evidence["expectations"]),
        "runs": evidence["run_count"],
        "unjudged": report_lib.unjudged_for({c["run_id"] for c in claims.values()}),
        "next": f"python -m tools.report write --project {project_id} --json",
    }


def _describe_prediction(quantity: str, predicted: dict[str, Any]) -> str:
    low, high, direction = predicted.get("low"), predicted.get("high"), predicted.get("direction")
    if low is not None and high is not None:
        return f"{quantity} between {low} and {high}"
    if direction:
        return f"{quantity} should {direction.replace('_', ' ')}"
    if low is not None:
        return f"{quantity} at least {low}"
    if high is not None:
        return f"{quantity} at most {high}"
    return quantity


def _render_claims_tex(claims: dict[str, Any]) -> str:
    """The macro definitions `\\gradnum` expands, as text.

    Split out from writing them so `check` can regenerate the file in memory and
    compare: the sidecar is the checkable artifact and this is its rendering, and
    for a while nothing verified that the two still agreed.
    """
    lines = [
        "% Generated by `python -m tools.report draft`. Do not edit.",
        "% Each value is the one recorded in the ledger for its (run_id, quantity).",
    ]
    for key, entry in sorted(claims.items()):
        value = entry.get("value")
        lines.append(rf"\expandafter\def\csname gradval@{key}\endcsname{{{_tex_escape(str(value))}}}")
    return "\n".join(lines) + "\n"


def _write_claims_tex(project_id: str, claims: dict[str, Any]) -> Path:
    """Materialise claims.json as the macro definitions `\\gradnum` expands.

    Generated, never hand-edited: the sidecar is the checkable artifact and this
    file is its rendering. Editing the rendering would let a number drift away
    from the run it claims to come from, which is precisely what §22 forbids --
    and `check_claims_tex` is what makes that sentence true rather than merely
    stated. The PDF prints *this* file's macros, so a gate that verified only
    `claims.json` verified the wrong artifact.
    """
    path = report_lib.paths_for(project_id)["dir"] / "claims.tex"
    path.write_text(_render_claims_tex(claims), encoding="utf-8")
    return path


def check_claims_tex(project_id: str, claims: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule 1b: the rendered macros still match the sidecar they came from."""
    path = report_lib.paths_for(project_id)["dir"] / "claims.tex"
    expected = _render_claims_tex(claims)
    if not path.exists():
        # Only a problem if something actually expands a macro; `check_claims`
        # reports the unresolved keys in that case, and a report with no numbers
        # legitimately has no claims.tex.
        if not claims:
            return []
        return [
            {
                "rule": "claims",
                "problem": "claims.tex is missing, so \\gradnum has nothing to expand",
                "fix": f"python -m tools.report draft --project {project_id} --json",
            }
        ]
    actual = path.read_text(encoding="utf-8")
    if actual.strip() == expected.strip():
        return []
    return [
        {
            "rule": "claims",
            "problem": (
                "claims.tex does not match claims.json -- the numbers the PDF prints are not "
                "the numbers the ledger recorded"
            ),
            "fix": (
                f"python -m tools.report draft --project {project_id} --json   # regenerates it; "
                "claims.tex is generated, never hand-edited"
            ),
        }
    ]


def _tex_escape(text: str) -> str:
    out = str(text)
    for char, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(char, replacement)
    return out


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
WRITE_PROMPT = """You write the prose of a scientific report from a structured evidence bundle.

Call submit_prose exactly once, then stop.

Rules that are enforced mechanically after you, so violating them wastes a run:

- **Never state a number directly.** Every measured value is referenced as
  \\gradnum{key}, using a key from the claims list you were given. A number typed
  into the prose fails the check.
- **Never write a \\cite{}.** Where a citation belongs, write [CITE:keyword]
  with a keyword describing what should be cited. A later pass resolves those
  against a real corpus; anything you invent here is deleted.
- Do not claim a result whose deviation is unjudged. Those are marked in the
  bundle. Describe them as open questions, not findings.
- A surprise is an alarm: where a result lands far outside its pre-registered
  range, say so and treat a bug as the first hypothesis.
- Prefer the relational framing the predictions use over absolute numbers.

Write: abstract, introduction, method, results, discussion, limitations. LaTeX
body only -- no preamble, no \\begin{document}.
"""


# The generated prose is fenced so a second `write` replaces it rather than
# stacking another copy on top. Without these, re-running produced two full
# bodies -- duplicate [CITE:] placeholders, duplicate claims, no warning -- and
# `write` is precisely the command someone re-runs after an unsatisfying draft.
#
# The fence is also what `check_prose_numbers` scans: it is the part of the
# document a model wrote, as opposed to the skeleton `draft` builds from the
# ledger, which legitimately contains numbers.
PROSE_START = report_lib.PROSE_START
PROSE_END = report_lib.PROSE_END


def _splice_prose(body: str, prose: str) -> str:
    """Replace the fenced prose block, or create it just after `\\maketitle`."""
    fenced = f"{PROSE_START}\n{prose}\n{PROSE_END}"
    start, end = body.find(PROSE_START), body.find(PROSE_END)
    if start != -1 and end != -1 and end > start:
        return body[:start] + fenced + body[end + len(PROSE_END) :]
    marker = "\\maketitle"
    head, found, tail = body.partition(marker)
    if not found:
        return body.rstrip() + "\n\n" + fenced + "\n"
    return head + marker + "\n\n" + fenced + "\n\n" + tail.lstrip("\n")


def _write_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    # There is deliberately no `--section`. It was accepted and never read, so
    # `write --section results` silently regenerated the whole report -- a flag
    # that lies about what it did is worse than one that does not exist.
    p.add_argument("--dry-run", action="store_true", help="show the bundle that would be sent, and stop")


@cli.command("write", "prose plus [CITE:...] placeholders (costs quota)", setup=_write_args)
def cmd_write(args: argparse.Namespace) -> dict[str, Any]:
    """The only step that spends anything.

    §23 item 5 leaves open whether this should run while a project is over
    budget -- the report is how you find out what the spend bought, but it is
    also a cost-bearing loop like any other. Specified as denied by the §15
    hook, and implemented that way here too so the CLI and the hook agree.
    """
    project_id = _project(args)
    files = report_lib.paths_for(project_id)
    if not files["tex"].exists():
        raise GradError(
            "no_draft",
            "there is no draft to write into",
            exit_code=3,
            fix=f"python -m tools.report draft --project {project_id} --json",
        )

    over = budget.over_budget(project_id)
    if over:
        from core.errors import EXIT_PROJECT_BUDGET, GateRefusal

        raise GateRefusal(
            "project_budget",
            f"project {project_id!r} is over budget on {', '.join(over)}; "
            "`write` is a cost-bearing loop like any other",
            EXIT_PROJECT_BUDGET,
            fix=(
                f"python -m tools.budget raise --project {project_id} "
                f"--{over[0].replace('_', '-')} <new ceiling> --json\n"
                f"     `python -m tools.report draft --project {project_id}` is free and "
                "shows what the spend bought"
            ),
        )

    evidence = report_lib.project_evidence(project_id)
    claims = report_lib.load_claims(project_id)
    bundle = _bundle(evidence, claims)
    if args.dry_run:
        return {"project": project_id, "bundle": bundle, "sent": False}

    cfg = config_mod.load()
    prose = _generate_prose(bundle, model=cfg.model_for("report"), project=project_id)

    body = files["tex"].read_text(encoding="utf-8")
    files["tex"].write_text(_splice_prose(body, prose), encoding="utf-8")

    return {
        "project": project_id,
        "tex": str(files["tex"]),
        "model": cfg.model_for("report"),
        "placeholders": sorted(set(report_lib.PLACEHOLDER_RE.findall(prose))),
        "next": f"python -m tools.report cite --project {project_id} --json",
    }


def _bundle(evidence: dict[str, Any], claims: dict[str, Any]) -> dict[str, Any]:
    """What the model is allowed to see: structured evidence and claim keys.

    Not the raw ledger -- the model does not need run ids it cannot cite, and
    handing it more numbers than it has keys for is how a number ends up in the
    prose without a `\\gradnum` around it.
    """
    return {
        "project": evidence["project"],
        "claims": {
            key: {"quantity": c["quantity"], "task": c.get("task")}
            for key, c in claims.items()
        },
        "expectations": [
            {
                "claim": b["expectation"].get("claim"),
                "quantity": b["expectation"].get("quantity"),
                "predicted": b["expectation"].get("predicted"),
                "comparability": b["expectation"].get("comparability"),
                "confidence": b["expectation"].get("confidence"),
                "basis": b["expectation"].get("basis"),
                "runs": [
                    {
                        "task": r["task"],
                        "status": r["status"],
                        "deviations": [
                            {k: v for k, v in d.items() if k != "expectation_id"}
                            for d in r["deviations"]
                        ],
                        "unjudged": bool(r["unjudged"]),
                    }
                    for r in b["runs"]
                ],
            }
            for b in evidence["expectations"]
        ],
        "figures": [Path(f).name for f in evidence["figures"]],
    }


def _generate_prose(bundle: dict[str, Any], *, model: str, project: str | None = None) -> str:
    """One forced-tool call, for the same reason §5 uses one.

    "prompting for JSON and parsing it fails silently on the tenth call."

    `project` is threaded through rather than left to default: `--project` can
    name a project other than the selected one, and charging this report's
    tokens to whichever happened to be current would attribute the spend to the
    wrong allocation.
    """
    try:
        import asyncio  # noqa: PLC0415

        from claude_agent_sdk import (  # noqa: PLC0415
            ClaudeAgentOptions,
            create_sdk_mcp_server,
            query,
            tool,
        )
    except ImportError as exc:
        raise ConfigError(
            "claude-agent-sdk is not installed, so prose cannot be generated",
            fix="pip install -e '.[agent]'   (`report draft` is free and needs no model)",
        ) from exc

    captured: list[str] = []

    @tool(
        "submit_prose",
        "Return the LaTeX body of the report",
        {
            "type": "object",
            "properties": {"latex": {"type": "string"}},
            "required": ["latex"],
        },
    )
    async def submit_prose(args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("latex")
        if not isinstance(text, str) or len(text) < 200:
            return {
                "content": [{"type": "text", "text": "latex must be the full report body"}],
                "is_error": True,
            }
        # Enforced at the tool boundary, where a returned error actually makes
        # the model retry -- not after the fact, where it would just be deleted.
        if re.search(r"\\cite[tp]?\*?\{", text):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "do not write \\cite{}; use [CITE:keyword] placeholders",
                    }
                ],
                "is_error": True,
            }
        captured.append(text)
        return {"content": [{"type": "text", "text": "recorded"}]}

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=WRITE_PROMPT,
        mcp_servers={"report": create_sdk_mcp_server("report", tools=[submit_prose])},
        allowed_tools=["mcp__report__submit_prose"],
        disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
    )

    async def run() -> None:
        # The result message carries a cumulative total, so it wins outright
        # when it arrives. Assistant-message usage is accumulated only as a
        # fallback for the turn that dies before a result -- adding both would
        # double-count.
        final: Any = None
        # Every field `from_sdk_usage` knows how to read. Accumulating only the
        # two uncached counters would silently drop cache traffic from the
        # fallback, so a failed turn would under-report exactly the tokens a
        # long prompt spends most of.
        partial = dict.fromkeys(
            ("input_tokens", "output_tokens",
             "cache_read_input_tokens", "cache_creation_input_tokens"),
            0,
        )
        try:
            async for message in query(
                prompt="Evidence bundle:\n\n" + json.dumps(bundle, indent=2, default=str),
                options=options,
            ):
                usage = getattr(message, "usage", None)
                if usage is None:
                    continue
                if type(message).__name__ == "ResultMessage":
                    final = usage
                else:
                    get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
                    for field in partial:
                        partial[field] += get(field, 0) or 0
        finally:
            # In a finally block because a failed turn still spent quota, and an
            # unrecorded spend is exactly what §15's ceilings cannot see.
            quota_log.from_sdk_usage(
                "report.write", final if final is not None else partial,
                model=model, role="report", project=project,
            )

    asyncio.run(run())
    if not captured:
        raise UpstreamError(
            "the model ended its turn without returning any prose",
            fix="re-run; `report draft` output is unchanged and still valid",
        )
    return captured[-1]


# ---------------------------------------------------------------------------
# cite
# ---------------------------------------------------------------------------
def _cite_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    p.add_argument("--context-chars", type=int, default=600, help="window around each placeholder")
    p.add_argument("--no-s2", action="store_true", help="resolve against the local corpus only")


@cli.command("cite", "resolve [CITE:...] against the corpus and S2; emit the bib", setup=_cite_args)
def cmd_cite(args: argparse.Namespace) -> dict[str, Any]:
    """The two-pass flow, stolen from Camyla.

    A placeholder is resolved by extracting the context around it and verifying
    a candidate's title and abstract against that context -- much better than
    citing inline, where the model invents a plausible reference in the moment.
    Resolution is against the local corpus and verified S2 ids **only**; an
    unresolvable placeholder is left in place and `check` refuses on it, rather
    than being silently dropped or filled with a guess.
    """
    project_id = _project(args)
    files = report_lib.paths_for(project_id)
    if not files["tex"].exists():
        raise GradError(
            "no_draft", "there is nothing to cite yet", exit_code=3,
            fix=f"python -m tools.report draft --project {project_id} --json",
        )

    tex = files["tex"].read_text(encoding="utf-8")
    # Seeded with what is already there, not started empty. `cite` is naturally
    # re-run -- after adding a section, after ingesting a paper that previously
    # failed to resolve -- and by then the earlier placeholders are already
    # `\cite{}` keys, so a second pass finds nothing to resolve. Rewriting the
    # file from an empty dict would delete every entry the first pass earned and
    # leave `check` refusing on citations that were fine a moment ago.
    entries: dict[str, dict[str, Any]] = (
        report_lib.parse_bib(files["bib"].read_text(encoding="utf-8"))
        if files["bib"].exists()
        else {}
    )
    preexisting = set(entries)
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for match in list(report_lib.PLACEHOLDER_RE.finditer(tex)):
        keyword = match.group(1).strip()
        start = max(0, match.start() - args.context_chars)
        context = tex[start : match.end() + args.context_chars]
        entry = _resolve_citation(keyword, context, use_s2=not args.no_s2)
        if entry is None:
            unresolved.append({"keyword": keyword, "context": context[:200]})
            continue
        entries[entry["key"]] = entry
        resolved.append({"keyword": keyword, "key": entry["key"], "source": entry["gradsource"]})

    # Replace only what resolved. An unresolved placeholder stays visible and
    # `check` refuses on it: a citation quietly deleted is worse than one that
    # fails loudly, because the sentence it supported survives without support.
    def substitute(match: re.Match[str]) -> str:
        keyword = match.group(1).strip()
        for row in resolved:
            if row["keyword"] == keyword:
                return f"\\cite{{{row['key']}}}"
        return match.group(0)

    files["tex"].write_text(report_lib.PLACEHOLDER_RE.sub(substitute, tex), encoding="utf-8")
    files["bib"].write_text(_render_bib(entries), encoding="utf-8")

    payload = {
        "project": project_id,
        "bib": str(files["bib"]),
        "resolved": resolved,
        "unresolved": unresolved,
        "entries": len(entries),
        "kept_from_previous_run": sorted(preexisting),
        "next": f"python -m tools.report check --project {project_id} --json",
    }
    if unresolved:
        raise GradError(
            "citations_unresolved",
            f"{len(unresolved)} placeholder(s) did not resolve against the corpus or S2: "
            + ", ".join(sorted({u['keyword'] for u in unresolved})[:5]),
            exit_code=EXIT_CHECK_FAILED,
            fix=(
                "ingest the paper so it is in the corpus: "
                "python -m tools.paper_ingest arxiv <id> --json\n"
                "     A \\cite{} key with no resolved entry is a hard error, not a warning."
            ),
            detail=payload,
        )
    return payload


def _resolve_citation(keyword: str, context: str, *, use_s2: bool) -> dict[str, Any] | None:
    """Local corpus first, then S2. Never the model's memory."""
    found = _from_corpus(keyword)
    if found:
        return found
    if not use_s2:
        return None
    return _from_s2(keyword, context)


def _from_corpus(keyword: str) -> dict[str, Any] | None:
    try:
        con = corpus.connect(create=False)
    except GradError:
        return None
    try:
        row = con.execute(
            "SELECT id, title, authors, year FROM documents WHERE id = ? OR title LIKE ? LIMIT 1",
            (keyword, f"%{keyword}%"),
        ).fetchone()
        if row is None:
            hits = corpus.fts_search(con, keyword, limit=1)
            if not hits:
                return None
            row = con.execute(
                "SELECT id, title, authors, year FROM documents WHERE id = ?",
                (hits[0].get("doc_id"),),
            ).fetchone()
        if row is None:
            return None
        doc_id, title, authors, year = row
        return {
            "key": _bib_key(doc_id, authors, year),
            "type": "article",
            "title": title or doc_id,
            "author": authors or "Unknown",
            "year": str(year or ""),
            "note": doc_id,
            # The provenance `check` requires. Without it the entry is refused.
            "gradsource": "corpus",
        }
    finally:
        con.close()


# Two independent conditions, both required. Overlap against the abstract alone
# is easy to clear on shared jargon -- any two ML papers share "training",
# "model", "results" -- so a candidate must also connect to the *title*, which is
# where a paper's actual subject lives. The numbers are deliberately strict: a
# citation this refuses is one the author adds by hand after reading it, while a
# citation it wrongly accepts is a claim silently attributed to a paper that does
# not support it. Recorded on the entry as `gradmatch` / `gradtitlematch` so a
# borderline resolution is auditable rather than invisible -- and re-checked by
# `check_citations`, which is why the thresholds live in `core/report.py` where
# both the writer and the gate read the same two numbers.
S2_MIN_CONTEXT_OVERLAP = report_lib.S2_MIN_CONTEXT_OVERLAP
S2_MIN_TITLE_OVERLAP = report_lib.S2_MIN_TITLE_OVERLAP


def _from_s2(keyword: str, context: str) -> dict[str, Any] | None:
    """Verify the candidate's title and abstract against the surrounding text.

    A search hit is not a citation. The candidate has to actually be about what
    the sentence claims, and the overlap test below is deliberately crude and
    conservative: it rejects rather than accepts when unsure.
    """
    try:
        client = http.SemanticScholar(config_mod.load())
        hits = client.paper_search(keyword, limit=5)
    except GradError:
        return None
    if not hits:
        return None

    words = _content_words(context)
    if not words:
        return None

    # Both gates are applied *before* ranking, not to the winner afterwards.
    # Ranking first and then testing meant a loosely-related paper with a
    # keyword-stuffed abstract could win on context overlap, fail the title
    # gate, and take the genuinely correct paper down with it -- rejecting a
    # citation that was right there in the candidate list.
    qualifying = []
    for hit in hits:
        body = _content_words(f"{hit.get('title', '')} {hit.get('abstract', '')}")
        title = _content_words(hit.get("title", ""))
        if not body:
            continue
        score = len(words & body) / len(words)
        title_score = (len(words & title) / len(title)) if title else 0.0
        if score >= S2_MIN_CONTEXT_OVERLAP and title_score >= S2_MIN_TITLE_OVERLAP:
            qualifying.append((score, title_score, hit))

    if not qualifying:
        return None
    best_score, best_title, best = max(qualifying, key=lambda row: (row[0], row[1]))
    return {
        "key": _bib_key(best.get("paper_id", keyword), None, best.get("year")),
        "type": "article",
        "title": best.get("title") or keyword,
        "author": "Unknown",
        "year": str(best.get("year") or ""),
        "note": f"S2:{best.get('paper_id')}",
        "gradsource": "s2",
        "gradmatch": round(best_score, 3),
        "gradtitlematch": round(best_title, 3),
    }


# Words short enough to be grammar rather than subject matter carry no signal,
# and a handful of ubiquitous research words clear any threshold on their own.
_STOPWORDS = frozenset(
    """about above after again against because before being below between during
    further having their there these those through under until where which while
    model models method methods result results using training train paper approach
    work works show shows shown study propose proposed""".split()
)


def _content_words(text: str) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[a-zA-Z]{5,}", text or "")
        if w.lower() not in _STOPWORDS
    }


def _bib_key(doc_id: str, authors: Any, year: Any) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "", str(doc_id))[-12:] or "ref"
    lead = re.sub(r"[^A-Za-z]+", "", str(authors or "").split(",")[0])[:12].lower()
    return f"{lead or 'ref'}{year or ''}{stem}"


def _render_bib(entries: dict[str, dict[str, Any]]) -> str:
    out = [
        "% Generated by `python -m tools.report cite`. Do not hand-edit.",
        "% Every entry carries `gradsource`, which is what `report check` verifies:",
        "% only `corpus` (the local index) and `s2` (a verified Semantic Scholar id)",
        "% are accepted. A hand-written entry is exactly the hallucinated citation",
        "% this rule exists to make impossible.",
        "",
    ]
    for key, entry in sorted(entries.items()):
        out.append(f"@{entry['type']}{{{key},")
        for field in ("title", "author", "year", "note", "gradsource", "gradmatch"):
            if entry.get(field) not in (None, ""):
                out.append(f"  {field} = {{{entry[field]}}},")
        out.append("}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# check -- the gate
# ---------------------------------------------------------------------------
@cli.command("check", "the gate: refuses on an unresolved claim or an unjudged run", setup=_project_arg)
def cmd_check(args: argparse.Namespace) -> dict[str, Any]:
    """Four rules, in order. It refuses; it does not warn."""
    project_id = _project(args)
    files = report_lib.paths_for(project_id)
    if not files["tex"].exists():
        raise GradError(
            "no_report", f"no report exists for project {project_id}", exit_code=3,
            fix=f"python -m tools.report draft --project {project_id} --json",
        )

    tex = files["tex"].read_text(encoding="utf-8")
    claims = report_lib.load_claims(project_id)
    bib = report_lib.parse_bib(files["bib"].read_text(encoding="utf-8")) if files["bib"].exists() else {}

    findings: list[dict[str, Any]] = []
    findings += report_lib.check_claims(tex, claims)
    # The rendered macros, not just the sidecar: claims.tex is what the PDF
    # prints, and verifying only claims.json verified the wrong artifact.
    findings += check_claims_tex(project_id, claims)
    # And numbers that never went through \gradnum at all, which rule 1 cannot
    # see by construction.
    findings += report_lib.check_prose_numbers(tex)
    findings += report_lib.check_citations(tex, bib)

    # Rule 3. The one most in the spirit of this system.
    unjudged = report_lib.unjudged_for(report_lib.cited_run_ids(tex, claims))
    for row in unjudged:
        findings.append(
            {
                "rule": "unjudged",
                "run_id": row["run_id"],
                "problem": (
                    f"run {row['run_id']} has an unjudged deviation on {row['quantity']}, "
                    "and this report cites it"
                ),
                "fix": (
                    f"python -m tools.ledger verdict {row['run_id']} --quantity {row['quantity']} "
                    "--verdict bug|real|inconclusive --note '...' --json"
                ),
            }
        )

    findings += report_lib.check_latex(tex)

    payload = {
        "project": project_id,
        "tex": str(files["tex"]),
        "claims_checked": len(set(report_lib.GRADNUM_RE.findall(tex))),
        "citations_checked": len(bib),
        "cited_runs": sorted(report_lib.cited_run_ids(tex, claims)),
        "findings": findings,
        "by_rule": {
            rule: sum(1 for f in findings if f.get("rule") == rule)
            for rule in ("claims", "citations", "unjudged", "latex")
        },
    }
    if findings:
        first = findings[0]
        raise GradError(
            "report_check_failed",
            f"{len(findings)} finding(s); first: {first.get('problem')}",
            exit_code=EXIT_CHECK_FAILED,
            fix=first.get("fix") or "resolve the finding and re-run check",
            detail=payload,
        )
    return {
        **payload,
        "ok": True,
        "note": (
            "every number traces to a run record, every citation to the corpus or a "
            "verified S2 id, and every cited run has been judged"
        ),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _build_args(p: argparse.ArgumentParser) -> None:
    _project_arg(p)
    # There is deliberately no `--skip-check`. There was one, hidden with
    # `argparse.SUPPRESS` -- but SUPPRESS hides a flag from `--help`, and the
    # agent reads the source. A gate with an undocumented bypass is a gate that
    # is bypassed exactly when it matters, and "not skippable from the agent's
    # side" has to be true rather than merely written in the docstring below.
    p.add_argument("--passes", type=int, default=3, help="LaTeX passes (bibtex needs at least 2)")


@cli.command("build", "compile the PDF, checkpointing each version", setup=_build_args)
def cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    """Progressive versions, copied from Denario.

    "It emits four because unattended LaTeX does not reliably compile." Each
    successful pass is checkpointed, so a run that fails on pass 3 still leaves
    the pass-2 PDF rather than nothing at all.

    `check` runs first and is not skippable from the agent's side: building a
    PDF is the act of asserting the result, and that is exactly where the gate
    belongs.
    """
    project_id = _project(args)
    cmd_check(argparse.Namespace(project=project_id))

    files = report_lib.paths_for(project_id)
    engine = shutil.which("latexmk") or shutil.which("pdflatex")
    if not engine:
        raise ConfigError(
            "neither latexmk nor pdflatex is on PATH, so no PDF can be produced",
            fix=(
                "install a TeX distribution (MiKTeX or TeX Live), or read the checked "
                f"source at {files['tex']}"
            ),
        )

    checkpoints: list[dict[str, Any]] = []
    for attempt in range(1, max(1, args.passes) + 1):
        argv = (
            [engine, "-pdf", "-interaction=nonstopmode", "-halt-on-error", files["tex"].name]
            if engine.endswith("latexmk") or "latexmk" in Path(engine).stem
            else [engine, "-interaction=nonstopmode", "-halt-on-error", files["tex"].name]
        )
        proc = subprocess.run(
            argv, cwd=str(files["dir"]), capture_output=True, text=True, timeout=300, check=False
        )
        ok = proc.returncode == 0 and files["pdf"].exists()
        if ok:
            checkpoint = files["dir"] / f"main.v{attempt}.pdf"
            shutil.copyfile(files["pdf"], checkpoint)
            checkpoints.append({"pass": attempt, "pdf": str(checkpoint)})
        else:
            log = files["dir"] / f"build.pass{attempt}.log"
            log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            if not checkpoints:
                raise GradError(
                    "build_failed",
                    f"LaTeX failed on pass {attempt}",
                    exit_code=EXIT_CHECK_FAILED,
                    fix=f"read {log}",
                    detail={"log": str(log), "tail": (proc.stdout or "").splitlines()[-25:]},
                )
            break

    return {
        "project": project_id,
        "pdf": str(files["pdf"]),
        "checkpoints": checkpoints,
        "engine": engine,
        # Always true now: `check` runs above with no way to skip it. Kept in the
        # envelope because it is what a reader of a build record wants to know.
        "checked": True,
    }


if __name__ == "__main__":
    main(cli)
