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

from core import ledger_store as ls, paths

# `\gradnum{key}` and `\cite{a,b}` as they appear in the LaTeX source.
GRADNUM_RE = re.compile(r"\\gradnum\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
# The two-pass placeholder stolen from Camyla: write with these, resolve after.
PLACEHOLDER_RE = re.compile(r"\[CITE:([^\]]+)\]")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")

# LaTeX specials that must be escaped in text mode. `\` and `$` and `%` are
# excluded deliberately: they are legitimate in a document full of maths.
UNESCAPED_RE = re.compile(r"(?<!\\)([&#_])")


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


def check_citations(tex: str, bib: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule 2: every `\\cite{}` key exists in references.bib, and every bib entry
    came from the corpus or a verified S2 id.

    The second half is the one that matters. A `.bib` a model wrote from memory
    passes "the key exists" trivially; what it cannot pass is "this entry has a
    `gradsource` naming where it was resolved from".
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
    # Unused entries are noted, not refused: over-collecting is not a lie.
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

    for line_no, line in enumerate(tex.splitlines(), start=1):
        body = re.sub(r"(?<!\\)%.*$", "", line)
        # Only outside maths: `_` and `&` are legitimate in equations and tables.
        if "$" in body or body.lstrip().startswith("\\") or "&" in body:
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
