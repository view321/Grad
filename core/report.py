"""The two structural guarantees behind `tools/report.py` (HANDOFF-2 §22).

    "The hard problem in machine-written papers is not prose. It is hallucinated
     citations and unsupported claims. Both can be made structurally impossible
     here."

Both, because Grad's provenance is *already structured*. Every surveyed harness
-- AI Scientist v2, PaperOrchestra, Jr. AI Scientist, Denario, Camyla, CiteLLM --
reconstructs provenance from unstructured experiment logs. Ours has expectations
with `basis` and `comparability`, runs with results and `deviations`, verdicts
with notes, figures, and corpus paper ids. That is strictly better input than
any of those systems receive, and adopting one means discarding the advantage.

**Claims.** Every asserted number carries `\\gradnum{<key>}`, backed by a
`claims.json` sidecar mapping each key to `(run_id, quantity)`. Each is verified
against the ledger. A number that does not resolve fails the check.

**Citations.** Every `\\cite{}` key resolves only against the local corpus and
S2-verified ids. A key with no resolved entry is a hard error, not a warning.

And the rule most in the spirit of this system: **no cited run may have an
unjudged deviation.** You should not be able to write up a result you have not
judged. `collect` already computes `needs_verdict`; this reads the same field.

`check` refuses; it does not warn. A report generator is where this system's
epistemics either hold or collapse -- the whole design exists to stop the user
believing results too easily, and a paper generator is a machine for asserting
them confidently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core import ledger_store as ls, paths, version

# `\gradnum{key}` and `\cite{a,b}` as they appear in the LaTeX source.
GRADNUM_RE = re.compile(r"\\gradnum\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
# The two-pass placeholder stolen from Camyla: write with these, resolve after.
PLACEHOLDER_RE = re.compile(r"\[CITE:([^\]]+)\]")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")

# LaTeX specials that must be escaped in text mode. `\` and `$` and `%` are
# excluded deliberately: they are legitimate in a document full of maths.
UNESCAPED_RE = re.compile(r"(?<!\\)([&#_])")

# Two independent conditions, both required, applied by `report cite` when it
# accepts an S2 match and re-checked by `check_citations` when it verifies the
# entry that resolution wrote. They live here rather than in `tools/report.py`
# because the writer and the gate have to agree on them, and a threshold defined
# next to only one of the two is a threshold that drifts.
S2_MIN_CONTEXT_OVERLAP = 0.25
S2_MIN_TITLE_OVERLAP = 0.20


def report_dir(project_id: str) -> Path:
    return paths.root() / "reports" / project_id


def paths_for(project_id: str) -> dict[str, Path]:
    base = report_dir(project_id)
    return {
        "dir": base,
        "tex": base / "main.tex",
        "claims": base / "claims.json",
        "bib": base / "references.bib",
        "pdf": base / "main.pdf",
    }


# ---------------------------------------------------------------------------
# the ledger view a report is built from
# ---------------------------------------------------------------------------
def project_evidence(project_id: str) -> dict[str, Any]:
    """Every expectation, its runs, its deviations, its verdict, its figures.

    This is what `draft` renders with no model in the loop, and what `check`
    verifies against. It is deliberately the *whole* picture rather than the
    successful part: a report skeleton that omits the runs that failed is a
    skeleton that invites writing up only the ones that worked.
    """
    runs = [r for r in ls.runs() if r.project == project_id and not r.is_smoke]
    by_expectation: dict[str, list[Any]] = {}
    for run in runs:
        by_expectation.setdefault(run.get("expectation_id") or "", []).append(run)

    expectations = []
    for exp in ls.expectations():
        bound = by_expectation.get(exp["id"], [])
        if not bound:
            continue
        expectations.append(
            {
                "expectation": exp,
                "runs": [
                    {
                        "id": r.id,
                        "task": r.get("task"),
                        "status": r.status,
                        "results": r.get("results") or {},
                        "deviations": r.get("deviations") or [],
                        "unjudged": r.unjudged_deviations(),
                        "cost_usd": r.get("cost_usd_actual"),
                        "collected": r.collected,
                    }
                    for r in bound
                ],
            }
        )

    orphans = [
        {"id": r.id, "task": r.get("task"), "results": r.get("results") or {}}
        for r in by_expectation.get("", [])
    ]
    return {
        "project": project_id,
        "expectations": expectations,
        "unbound_runs": orphans,
        "figures": sorted(str(p) for p in paths.figures_dir().glob("*.png")),
        "run_count": len(runs),
    }


def quantity_value(run_id: str, quantity: str) -> tuple[bool, Any, str | None]:
    """(resolved, value, problem). The oracle behind `\\gradnum`."""
    try:
        run = ls.run(run_id)
    except Exception:  # noqa: BLE001 - a missing run is a finding, not a crash
        return False, None, f"run {run_id!r} is not in the ledger"
    results = run.get("results") or {}
    if quantity not in results:
        known = ", ".join(sorted(results)) or "(none)"
        return False, None, f"run {run_id} reports no quantity {quantity!r}; it has: {known}"
    return True, results[quantity], None


def unjudged_for(run_ids: set[str]) -> list[dict[str, Any]]:
    """Cited runs whose deviations have not been judged.

    "You should not be able to write up a result you have not judged."
    """
    out = []
    for run_id in sorted(run_ids):
        try:
            run = ls.run(run_id)
        except Exception:  # noqa: BLE001
            continue
        for dev in run.unjudged_deviations():
            out.append({"run_id": run_id, "quantity": dev.get("quantity"), "reason": dev.get("reason")})
    return out


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
def load_claims(project_id: str) -> dict[str, Any]:
    path = paths_for(project_id)["claims"]
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return doc if isinstance(doc, dict) else {}


def check_claims(tex: str, claims: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule 1: every `\\gradnum{}` key resolves to a `(run_id, quantity)` present
    in the ledger, **with a matching value**.

    The value comparison is what makes this a guarantee rather than a gesture:
    a key that points at a real run and a real quantity but prints a different
    number is exactly the failure a citation-checker would miss.
    """
    findings: list[dict[str, Any]] = []
    for key in sorted(set(GRADNUM_RE.findall(tex))):
        entry = claims.get(key)
        if not isinstance(entry, dict):
            findings.append(
                {
                    "rule": "claims",
                    "key": key,
                    "problem": f"\\gradnum{{{key}}} has no entry in claims.json",
                    "fix": f'add "{key}": {{"run_id": "...", "quantity": "...", "value": ...}}',
                }
            )
            continue
        run_id, quantity = entry.get("run_id"), entry.get("quantity")
        if not run_id or not quantity:
            findings.append(
                {
                    "rule": "claims",
                    "key": key,
                    "problem": f"claim {key!r} needs both run_id and quantity",
                    "fix": "every asserted number traces to one run and one quantity",
                }
            )
            continue
        resolved, actual, problem = quantity_value(str(run_id), str(quantity))
        if not resolved:
            findings.append({"rule": "claims", "key": key, "problem": problem, "fix": problem})
            continue
        stated = entry.get("value")
        if stated is not None and not _values_match(stated, actual):
            findings.append(
                {
                    "rule": "claims",
                    "key": key,
                    "problem": (
                        f"claim {key!r} states {stated!r} but run {run_id} recorded {actual!r} "
                        f"for {quantity}"
                    ),
                    "fix": f'set "value" to {actual!r}, or point the claim at the right run',
                }
            )
    return findings


def _values_match(stated: Any, actual: Any) -> bool:
    """Numbers compare with tolerance for the rounding a paper does; anything
    else compares exactly."""
    if isinstance(stated, (int, float)) and isinstance(actual, (int, float)):
        if actual == 0:
            return abs(stated) < 1e-9
        return abs(stated - actual) / abs(actual) < 1e-3
    return str(stated).strip() == str(actual).strip()


def check_code_versions(run_ids: set[str]) -> list[dict[str, Any]]:
    """Rule 4: the runs this report cites all came from the same Grad.

    The stamp is written at submit time by `core/submit.py`; this is what makes
    it worth writing. Two failures, and they are different findings:

    *Straddling a version.* A report that cites a run from before an update and
    one from after is comparing two pieces of code and presenting the difference
    as a result about the research. That is not always wrong -- a release that
    changed only the UI cannot have moved a number -- so it is a finding to
    answer, with the versions named, rather than a refusal to override.

    *Modified code.* A run submitted from a checkout with uncommitted edits has
    no identifier anyone else can resolve. "Every number traces to a run record"
    fails at the last step: the record is there, the code it names is not.

    Runs with no stamp at all are silently allowed. They predate this field, and
    refusing a report because its evidence is *old* would make the rule a reason
    to avoid updating -- the opposite of what it is for.
    """
    stamps: dict[str, dict[str, Any]] = {}
    for run_id in sorted(run_ids):
        try:
            run = ls.run(run_id)
        except Exception:  # noqa: BLE001 - a missing run is rule 1's finding, not this one
            continue
        stamp = run.get("code_version")
        if isinstance(stamp, dict) and any(stamp.get(k) for k in ("commit", "tag", "version")):
            stamps[run_id] = stamp

    findings: list[dict[str, Any]] = []
    dirty = sorted(run_id for run_id, stamp in stamps.items() if stamp.get("dirty"))
    if dirty:
        findings.append(
            {
                "rule": "version",
                "run_id": dirty[0],
                "problem": (
                    f"{len(dirty)} cited run(s) were submitted from a modified installation "
                    f"({', '.join(dirty[:3])}); the code that produced them is not identified "
                    "by any commit"
                ),
                "fix": (
                    "commit the changes in the installation folder and re-run those runs, "
                    "or say in the report that the code was modified"
                ),
            }
        )

    distinct: list[dict[str, Any]] = []
    for stamp in stamps.values():
        if not any(version.same_version(stamp, seen) for seen in distinct):
            distinct.append(stamp)
    if len(distinct) > 1:
        named = ", ".join(sorted({version.label(s) for s in distinct}))
        findings.append(
            {
                "rule": "version",
                "problem": (
                    f"this report cites runs from {len(distinct)} different versions of Grad "
                    f"({named}); a number from one is not comparable to a number from another"
                ),
                "fix": (
                    "re-run the older results on the current version, or state the versions "
                    "in the report and say why the comparison holds"
                ),
            }
        )
    return findings


def cited_run_ids(tex: str, claims: dict[str, Any]) -> set[str]:
    keys = set(GRADNUM_RE.findall(tex))
    out = set()
    for key in keys:
        entry = claims.get(key)
        if isinstance(entry, dict) and entry.get("run_id"):
            out.add(str(entry["run_id"]))
    return out


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------
def parse_bib(text: str) -> dict[str, dict[str, Any]]:
    """A small BibTeX reader: enough to know what keys exist and where each came
    from. Full BibTeX parsing is not needed and would be a dependency."""
    entries: dict[str, dict[str, Any]] = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.DOTALL):
        kind, key, body = match.group(1), match.group(2).strip(), match.group(3)
        fields = {
            m.group(1).lower(): m.group(2).strip()
            for m in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", body, re.MULTILINE)
        }
        entries[key] = {"type": kind, "key": key, **fields}
    return entries


def corpus_doc_ids() -> set[str] | None:
    """Every document id in the local index, or None if there is no index.

    None and "empty" are different answers and the caller must not confuse them:
    an absent corpus cannot refute a `gradsource = {corpus}` claim, while an
    empty one refutes every such claim.
    """
    try:
        from core import corpus  # noqa: PLC0415 - optional dependency at the point of use

        con = corpus.connect(create=False)
    except Exception:  # noqa: BLE001 - no corpus, or sqlite unavailable
        return None
    try:
        return {str(row[0]) for row in con.execute("SELECT id FROM documents")}
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


def check_citations(tex: str, bib: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule 2: every `\\cite{}` key exists in references.bib, and every bib entry
    resolves against something real.

    The second half is the one that matters, and it used to be the easiest half
    to fake: the only provenance check was that the entry *carried the string*
    `gradsource = {corpus}`, which is one line of BibTeX to type. The claim in
    this docstring -- "what it cannot pass is 'this entry has a gradsource'" --
    was exactly backwards.

    So the id is re-resolved instead of the label being trusted:

      * `corpus` entries must name a document id that is in the local index now;
      * `s2` entries must carry the `S2:<id>` note and the two overlap scores
        `cite` recorded when it accepted the match, at or above the thresholds
        it applied.

    The S2 half is weaker than the corpus half and honestly so: re-querying the
    live service inside a gate would make `check` need the network. What it
    costs a forger is no longer one line but a consistent set of fields --
    and `report cite` writes all of them from a resolution that actually
    happened.
    """
    findings: list[dict[str, Any]] = []
    used: set[str] = set()
    for group in CITE_RE.findall(tex):
        used.update(k.strip() for k in group.split(",") if k.strip())

    for key in sorted(used):
        if key not in bib:
            findings.append(
                {
                    "rule": "citations",
                    "key": key,
                    "problem": f"\\cite{{{key}}} has no entry in references.bib",
                    "fix": "python -m tools.report cite --project <id> --json",
                }
            )

    doc_ids = corpus_doc_ids()
    for key, entry in sorted(bib.items()):
        source = entry.get("gradsource")
        if source not in ("corpus", "s2"):
            findings.append(
                {
                    "rule": "citations",
                    "key": key,
                    "problem": (
                        f"bib entry {key!r} has no verified provenance "
                        "(gradsource must be `corpus` or `s2`)"
                    ),
                    "fix": (
                        "entries are written by `report cite`, which resolves only against "
                        "the local corpus and verified S2 ids. A hand-written entry is "
                        "exactly the hallucinated citation this rule exists to stop."
                    ),
                }
            )
            continue

        note = str(entry.get("note") or "").strip()
        if source == "corpus":
            if not note:
                findings.append(
                    {
                        "rule": "citations",
                        "key": key,
                        "problem": f"bib entry {key!r} claims the corpus but names no document id",
                        "fix": "python -m tools.report cite --project <id> --json",
                    }
                )
            elif doc_ids is None:
                findings.append(
                    {
                        "rule": "citations",
                        "key": key,
                        "problem": (
                            f"bib entry {key!r} claims the corpus, but there is no local index "
                            "to resolve it against"
                        ),
                        "fix": (
                            "python -m tools.paper_ingest arxiv <id> --json   # build the corpus "
                            "this citation claims to come from"
                        ),
                    }
                )
            elif note not in doc_ids:
                findings.append(
                    {
                        "rule": "citations",
                        "key": key,
                        "problem": (
                            f"bib entry {key!r} claims corpus document {note!r}, which is not in "
                            "the local index -- the entry was not written by `report cite`, or "
                            "the document has since been removed"
                        ),
                        "fix": "python -m tools.report cite --project <id> --json",
                    }
                )
        else:  # s2
            if not re.fullmatch(r"S2:[A-Za-z0-9]+", note):
                findings.append(
                    {
                        "rule": "citations",
                        "key": key,
                        "problem": (
                            f"bib entry {key!r} claims Semantic Scholar but its note is {note!r}, "
                            "not the `S2:<paper_id>` a resolution records"
                        ),
                        "fix": "python -m tools.report cite --project <id> --json",
                    }
                )
                continue
            match, title_match = entry.get("gradmatch"), entry.get("gradtitlematch")
            try:
                ok = (
                    match is not None
                    and title_match is not None
                    and float(match) >= S2_MIN_CONTEXT_OVERLAP
                    and float(title_match) >= S2_MIN_TITLE_OVERLAP
                )
            except (TypeError, ValueError):
                ok = False
            if not ok:
                findings.append(
                    {
                        "rule": "citations",
                        "key": key,
                        "problem": (
                            f"bib entry {key!r} claims Semantic Scholar but carries no passing "
                            "overlap evidence (gradmatch >= "
                            f"{S2_MIN_CONTEXT_OVERLAP}, gradtitlematch >= {S2_MIN_TITLE_OVERLAP})"
                        ),
                        "fix": "python -m tools.report cite --project <id> --json",
                    }
                )
    # Unused entries are noted, not refused: over-collecting is not a lie.
    return findings


# ---------------------------------------------------------------------------
# bare numbers in prose
# ---------------------------------------------------------------------------
# What a *measured* value looks like when it is typed rather than referenced: a
# decimal, a percentage, or scientific notation. Bare small integers are not
# flagged -- "Figure 1", "the three seeds", "Section 2" are structure, and a
# check that fires on those is a check that gets argued around (§6). This is
# therefore a floor, not a proof: it catches the shape a result takes, and the
# `\gradnum` discipline is what covers the rest.
# The trailing guard is `(?!\w)(?!\.\d)` rather than `(?![\w.])`: a full stop
# after a number ends a sentence far more often than it continues a version, and
# excluding every following dot meant "the loss was 2.71." -- the most natural
# way anyone writes the thing this rule exists to catch -- matched nothing at
# all. `(?!\.\d)` still refuses to stop half-way through `1.2.3`.
BARE_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:\.\d+)?\s*\\?%|\d+[eE][-+]?\d+)"
    r"(?!\w)(?!\.\d)"
)

# A version, not a measurement. `GPT-3.5`, `Python 3.11`, `CUDA 12.1`, `v2.0`
# all have the shape this rule looks for, and an ML report that names a model or
# a library by version is writing honest prose -- refusing it would make the
# gate something to be switched off rather than satisfied.
#
# The discriminator is what comes *before* the number. A version follows a
# proper noun or attaches to one with a hyphen; a measured value follows a verb
# or a preposition -- "loss of 2.71", "reached 3.05", "improved to 0.94". So a
# number preceded by a capitalised word, or glued to letters, is left alone.
#
# The cost is a real false negative: "Loss 2.71" at the start of a sentence is
# missed. That is the right way round for a check whose failure mode is
# refusing a correct report, and `\gradnum` remains the discipline that actually
# guarantees traceability -- this only catches the lapse.
_VERSION_CONTEXT_RE = re.compile(
    r"(?:[A-Za-z][\w.+]*[-_]|\b[A-Z][\w.+]*\s+|\bv)\s*$"
)

# Structural commands whose arguments are not prose.
_STRUCTURAL_RE = re.compile(
    r"\\(?:gradnum|label|ref|eqref|cite[tp]?\*?|includegraphics|input|include|"
    r"documentclass|usepackage|bibliography|bibliographystyle|url|href)\s*"
    r"(?:\[[^\]]*\])*\s*\{[^}]*\}"
)

# Maths and generated tabular material, where numbers are legitimate.
_MATH_RE = re.compile(
    r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|"
    r"\\begin\{(equation|align|gather|multline|eqnarray|tabular|table|figure|verbatim|lstlisting)\*?\}"
    r".*?\\end\{\1\*?\}",
    re.DOTALL,
)


# The fence `report write` puts around model-written prose. Defined here because
# both the writer and the gate need it: the writer to replace its own output
# instead of stacking a second copy, and the gate to know which part of the
# document a model wrote.
PROSE_START = "% GRAD-PROSE-START -- generated by `report write`; edits inside are replaced"
PROSE_END = "% GRAD-PROSE-END"


def written_prose(tex: str) -> str | None:
    """The model-written region, or None if `write` has not run.

    The scan below is deliberately scoped to this. `report draft` puts real
    numbers into the skeleton -- a prediction's band, a run's cost -- straight
    from the ledger, and those are not claims a model typed: they are the
    evidence the model is being asked to write *about*. Flagging them would
    make the rule fire on the tool's own honest output, which is how a check
    ends up switched off.
    """
    start = tex.find(PROSE_START)
    end = tex.find(PROSE_END)
    if start == -1 or end == -1 or end < start:
        return None
    return tex[start + len(PROSE_START) : end]


def prose_of(tex: str) -> str:
    """The body text, with comments, maths, tables, and command arguments gone."""
    body = tex.partition(r"\begin{document}")[2] or tex
    body = re.sub(r"(?<!\\)%.*$", "", body, flags=re.MULTILINE)
    body = _MATH_RE.sub(" ", body)
    body = _STRUCTURAL_RE.sub(" ", body)
    return body


def check_prose_numbers(tex: str) -> list[dict[str, Any]]:
    """Rule 1c: a measured number typed into the prose, rather than referenced.

    `WRITE_PROMPT` has always told the model "a number typed into the prose
    fails the check". Nothing enforced it, so the sentence was a request --
    and the one number a model is most tempted to type is the headline result.
    A claim that never goes through `\\gradnum` is a claim rule 1 never sees,
    which also takes it out of `cited_run_ids` and therefore out of rule 3.

    Version strings are exempt -- see `_VERSION_CONTEXT_RE`. "GPT-3.5" and
    "Python 3.11" have exactly the shape of a measured value, and a report is
    entitled to name the model it compared against.
    """
    region = written_prose(tex)
    if region is None:
        return []
    findings: list[dict[str, Any]] = []
    body = prose_of(region)
    for line_no, line in enumerate(body.splitlines(), start=1):
        for match in BARE_NUMBER_RE.finditer(line):
            if _VERSION_CONTEXT_RE.search(line[: match.start()]):
                continue
            findings.append(
                {
                    "rule": "claims",
                    "line": line_no,
                    "problem": (
                        f"the prose states {match.group(0).strip()!r} directly; a measured "
                        "value has to be referenced as \\gradnum{<key>} so it traces to a run"
                    ),
                    "fix": (
                        "add the value to claims.json with its run_id and quantity, then write "
                        "\\gradnum{<key>} -- or, if it is not a measurement, put it in maths"
                    ),
                }
            )
            break  # one finding per line is enough to send the author to it
    return findings


# ---------------------------------------------------------------------------
# LaTeX hygiene
# ---------------------------------------------------------------------------
def check_latex(tex: str) -> list[dict[str, Any]]:
    """Rule 4: no unmatched braces, duplicate labels, or unescaped specials.

    Stolen from PaperOrchestra's constraint set, and encoded as validation
    rather than as prompt text -- which is the whole point: a constraint a model
    is asked to respect is a constraint that gets violated on the tenth run.
    """
    findings: list[dict[str, Any]] = []

    depth = 0
    for line_no, line in enumerate(tex.splitlines(), start=1):
        stripped = re.sub(r"(?<!\\)%.*$", "", line)
        stripped = stripped.replace(r"\{", "").replace(r"\}", "")
        depth += stripped.count("{") - stripped.count("}")
        if depth < 0:
            findings.append(
                {
                    "rule": "latex",
                    "line": line_no,
                    "problem": "a closing brace with no opener",
                    "fix": f"balance the braces at line {line_no}",
                }
            )
            depth = 0
    if depth:
        findings.append(
            {
                "rule": "latex",
                "problem": f"{depth} unmatched opening brace(s)",
                "fix": "balance the braces; LaTeX will not compile until you do",
            }
        )

    seen: dict[str, int] = {}
    for match in LABEL_RE.finditer(tex):
        label = match.group(1)
        line_no = tex[: match.start()].count("\n") + 1
        if label in seen:
            findings.append(
                {
                    "rule": "latex",
                    "line": line_no,
                    "problem": f"duplicate \\label{{{label}}} (first at line {seen[label]})",
                    "fix": "labels must be unique or every reference to them is ambiguous",
                }
            )
        else:
            seen[label] = line_no

    leftovers = PLACEHOLDER_RE.findall(tex)
    if leftovers:
        findings.append(
            {
                "rule": "latex",
                "problem": f"{len(leftovers)} unresolved [CITE:...] placeholder(s): "
                + ", ".join(sorted(set(leftovers))[:5]),
                "fix": "python -m tools.report cite --project <id> --json",
            }
        )

    in_tabular = 0
    for line_no, line in enumerate(tex.splitlines(), start=1):
        body = re.sub(r"(?<!\\)%.*$", "", line)
        # `&` is legitimate in a table, so table rows are skipped -- but the
        # skip used to be `"&" in body`, which discarded every line containing
        # the character *before* the rule that looks for it could fire. The one
        # thing it was meant to catch was the one thing it could never report.
        if re.search(r"\\begin\{(tabular|tabularx|array)", body):
            in_tabular += 1
        if re.search(r"\\end\{(tabular|tabularx|array)", body):
            in_tabular = max(0, in_tabular - 1)
            continue
        if in_tabular or "$" in body or body.lstrip().startswith("\\"):
            continue
        for match in UNESCAPED_RE.finditer(body):
            findings.append(
                {
                    "rule": "latex",
                    "line": line_no,
                    "problem": f"unescaped {match.group(1)!r} in text",
                    "fix": f"write \\{match.group(1)}",
                }
            )
            break
    return findings
