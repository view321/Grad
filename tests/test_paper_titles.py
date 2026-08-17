r"""Getting a paper's title out of `\title{...}`, and out of the index once wrong.

The papers window is a list of titles, and nothing ever re-reads one: a title
parsed badly at ingest is what that paper is called for as long as the index
lives. So the parser is worth more care than its four lines suggest, and the
repair command is worth having at all.

Every case below is a real shape from the corpus in front of me or one line away
from it: a title wrapped in `\textsc`, a preamble that comments out an older
title, a paper that reports a percentage, a class file that redefines `\title`.
"""

from __future__ import annotations

import argparse

import pytest

from core import corpus, paths
from tools import paper_ingest


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------
def test_a_title_with_markup_in_it_is_not_cut_at_the_first_brace(workspace):
    r"""Non-greedy `\{(.+?)\}` stops at the wrong brace the moment a title
    contains any markup, and the corpus recorded `\textsc{Samba` as a paper's
    name because of it."""
    assert paper_ingest._title_from(
        r"\title{\textsc{Samba}: Simple Hybrid State Space Models}"
    ) == "Samba: Simple Hybrid State Space Models"


def test_font_and_break_macros_are_dropped_but_the_words_are_not(workspace):
    assert paper_ingest._title_from(
        r"\title{\LARGE \bf Bandwidth-Efficient\\ Multi-Agent Communication}"
    ) == "Bandwidth-Efficient Multi-Agent Communication"
    assert paper_ingest._title_from(
        r"\title{Beyond Tokens: A Unified Framework for\\Latent Communication}"
    ) == "Beyond Tokens: A Unified Framework for Latent Communication"


def test_a_commented_out_title_is_not_the_title(workspace):
    """Both shapes: a whole line commented, and a comment after real markup on
    the same line. The second is the one a line filter misses."""
    assert paper_ingest._title_from(
        "% \\title{An Older Draft}\n\\title{The Real One}"
    ) == "The Real One"
    assert paper_ingest._title_from(
        "\\documentclass{article} % \\title{Not The Title}\n\\title{The Real One}"
    ) == "The Real One"


def test_an_escaped_percent_does_not_truncate_the_title(workspace):
    r"""`\%` is a literal percent sign, and it appears in every paper that
    reports one. Treating it as a comment would silently cut the title in half
    at the first number with a unit."""
    assert paper_ingest._title_from(
        r"\title{Gains of 5\% on the Benchmark}"
    ) == r"Gains of 5\% on the Benchmark"


def test_a_command_that_redefines_title_is_not_a_title(workspace):
    r"""`\newcommand\title[1]{...}` is a *definition* of the command; its body
    is a formatting rule, not this paper's name."""
    assert paper_ingest._title_from(
        "\\newcommand\\title[1]{\\Large #1}\n\\title{Actual Title}"
    ) == "Actual Title"
    assert paper_ingest._title_from(
        "\\renewcommand{\\title}[1]{#1}\n\\title{Actual Title}"
    ) == "Actual Title"


def test_no_title_is_none_rather_than_a_guess(workspace):
    """`cmd_arxiv` falls back to the arXiv id, which is honest. Returning the
    first braced thing in the file would not be."""
    assert paper_ingest._title_from("no title anywhere in this source") is None
    assert paper_ingest._title_from(r"\title{}" "\n" r"\title{The Real One}") == "The Real One"


# ---------------------------------------------------------------------------
# repairing what a worse parser already stored
# ---------------------------------------------------------------------------
def _ingested(doc_id: str, title: str, tex: str) -> None:
    directory = paths.papers_dir() / doc_id.removeprefix("arXiv:")
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.tex"
    source.write_text(tex, encoding="utf-8")
    con = corpus.connect()
    try:
        corpus.upsert_document(
            con,
            {"id": doc_id, "title": title, "source": "arxiv-latex", "path": str(source),
             "ingested_at": "2026-08-16T00:00:00Z"},
        )
        con.commit()
    finally:
        con.close()


def _titles() -> dict[str, str]:
    con = corpus.connect()
    try:
        return {r["id"]: r["title"] for r in con.execute("SELECT id, title FROM documents")}
    finally:
        con.close()


def test_retitle_reparses_from_the_source_already_on_disk(workspace):
    """Re-ingesting to fix a string would re-download the tarball and re-spend
    the embeddings. This touches one column."""
    _ingested("arXiv:2406.07522", r"\textsc{Samba", r"\title{\textsc{Samba}: Simple Hybrid Models}")
    out = paper_ingest.cmd_retitle(argparse.Namespace(id=None, dry_run=False, json=True))

    assert [c["id"] for c in out["changed"]] == ["arXiv:2406.07522"]
    assert _titles()["arXiv:2406.07522"] == "Samba: Simple Hybrid Models"


def test_a_dry_run_reports_and_changes_nothing(workspace):
    _ingested("arXiv:1", r"\textsc{Bad", r"\title{\textsc{Good}: A Paper}")
    out = paper_ingest.cmd_retitle(argparse.Namespace(id=None, dry_run=True, json=True))

    assert out["dry_run"] is True
    assert out["changed"][0]["now"] == "Good: A Paper"
    assert _titles()["arXiv:1"] == r"\textsc{Bad"


def test_a_source_with_no_title_leaves_the_stored_one_alone(workspace):
    """A file that has been moved or emptied must not turn a correct title into
    a blank."""
    _ingested("arXiv:2", "A Perfectly Good Title", "no title macro in here")
    out = paper_ingest.cmd_retitle(argparse.Namespace(id=None, dry_run=False, json=True))

    assert out["changed"] == []
    assert _titles()["arXiv:2"] == "A Perfectly Good Title"


def test_an_unreadable_source_is_reported_not_swallowed(workspace):
    _ingested("arXiv:3", "Stored Title", r"\title{New}")
    (paths.papers_dir() / "3" / "source.tex").unlink()
    out = paper_ingest.cmd_retitle(argparse.Namespace(id=None, dry_run=False, json=True))

    assert out["unreadable"] == ["arXiv:3"]
    assert _titles()["arXiv:3"] == "Stored Title"


def test_one_document_can_be_repaired_on_its_own(workspace):
    _ingested("arXiv:4", r"\textsc{A", r"\title{\textsc{A}: One}")
    _ingested("arXiv:5", r"\textsc{B", r"\title{\textsc{B}: Two}")
    paper_ingest.cmd_retitle(argparse.Namespace(id="arXiv:4", dry_run=False, json=True))

    titles = _titles()
    assert titles["arXiv:4"] == "A: One"
    assert titles["arXiv:5"] == r"\textsc{B"


def test_it_says_so_rather_than_raising_when_there_is_no_index(workspace):
    out = paper_ingest.cmd_retitle(argparse.Namespace(id=None, dry_run=False, json=True))
    assert out["documents"] == []


# ---------------------------------------------------------------------------
# what the window shows
# ---------------------------------------------------------------------------
def test_the_window_reads_the_title_from_the_corpus(workspace):
    """`paper_ingest` records the title in the index and writes nothing beside
    the source, so a window reading only the directory shows the arXiv id it
    already knew."""
    from ui import models

    _ingested("arXiv:2405.21060", "Transformers are SSMs", r"\title{Transformers are SSMs}")
    row = next(r for r in models.papers_model(filter_name="read")["rows"] if r["id"] == "2405.21060")
    assert row["title"] == "Transformers are SSMs"
    assert row["authors"] == "arXiv:2405.21060"


def test_a_title_the_index_never_got_falls_back_to_the_id(workspace):
    from ui import models

    directory = paths.papers_dir() / "2501.14082"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.tex").write_text("no title macro", encoding="utf-8")
    row = next(r for r in models.papers_model(filter_name="read")["rows"] if r["id"] == "2501.14082")
    assert row["title"] == "2501.14082"


@pytest.mark.parametrize(
    "stored,shown",
    [
        (r"\textsc{Samba", "Samba"),
        ("\\LARGE \\bf\nBandwidth-Efficient Communication", "Bandwidth-Efficient Communication"),
        (r"Generalized Models \\ Through Duality", "Generalized Models Through Duality"),
    ],
)
def test_latex_residue_in_a_stored_title_is_cleaned_for_display(workspace, stored, shown):
    """The display side making the best of what an older parse already stored.
    `retitle` is the real repair; this is what the window does meanwhile."""
    from ui import models

    assert models._clean_latex(stored) == shown
