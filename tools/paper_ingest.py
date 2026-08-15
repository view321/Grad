"""grad-paper-ingest -- arXiv LaTeX source -> chunks -> the local index (HANDOFF §5).

    "Ingest from arXiv LaTeX source, not PDF -- this is the single largest
     quality lever in the retrieval stack, because it preserves equations,
     theorem environments, and section structure that PDF extraction destroys."

So the chunker is section- and environment-aware rather than a fixed-width
window: a theorem stays with its statement, an aligned block stays with the
sentence that introduces it. That is most of the difference between a local
index that can answer "where did I see that lemma" and one that cannot.
"""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from pathlib import Path
from typing import Any

from core import config as config_mod, corpus, http, paths
from core.cli import Cli, main
from core.errors import ConfigError, GradError, NotFound, UpstreamError, UsageError
from core.ledger_store import now_iso

cli = Cli(
    "grad-paper-ingest",
    "Ingest papers (LaTeX source) and personal notes into the local index.",
    epilog=(
        "The index records which embedding model built it and refuses vectors from any\n"
        "other, so changing models is a deliberate re-embed rather than a silent mix of\n"
        "incompatible vector spaces."
    ),
)

ARXIV_SRC = "https://arxiv.org/e-print/{id}"

SECTION_RE = re.compile(r"\\(sub)*section\*?\{([^}]*)\}")
ENV_RE = re.compile(
    r"\\begin\{(theorem|lemma|proposition|corollary|definition|proof|align\*?|equation\*?|gather\*?)\}"
    r"(.*?)\\end\{\1\}",
    re.DOTALL,
)
COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
def _fetch(url: str, *, timeout: float) -> bytes:
    """arXiv's e-print endpoint, through `core/http.py`'s accessor.

    Not a bare `import httpx`. Every other outbound request in this project goes
    through `http._httpx()`, and the suite's "no network" guarantee is
    implemented by replacing exactly that function -- so a second import site is
    a hole in it, and a test that reached this one would hang rather than fail.
    It also gets the same ImportError message for free.
    """
    httpx = http._httpx()  # noqa: SLF001 - the module's own accessor, see above
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(f"fetch failed: {exc}", fix="check connectivity and the arXiv id") from exc
    if resp.status_code >= 400:
        raise UpstreamError(
            f"arXiv returned {resp.status_code} for {url}",
            fix="check the id; some papers have no LaTeX source (then use --pdf-text)",
        )
    return resp.content


def _extract_tex(blob: bytes) -> str:
    """arXiv e-print payloads are usually a gzipped tar of .tex files."""
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            parts = []
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith((".tex", ".ltx")):
                    fh = tar.extractfile(member)
                    if fh:
                        parts.append(fh.read().decode("utf-8", errors="replace"))
            if parts:
                # Longest first: the main document usually dominates.
                parts.sort(key=len, reverse=True)
                return "\n\n".join(parts)
    except tarfile.TarError:
        pass
    text = blob.decode("utf-8", errors="replace")
    if "\\documentclass" in text or "\\begin{document}" in text:
        return text
    raise UpstreamError(
        "the e-print payload contained no LaTeX source",
        fix="this paper may be PDF-only; ingest the notes you took on it instead",
    )


def chunk_latex(tex: str, *, target_chars: int = 1800) -> list[dict[str, Any]]:
    """Section-aware chunking that keeps math environments intact.

    Environments are lifted out first and kept whole -- a theorem split across
    two chunks retrieves as neither -- then the remaining prose is packed to
    roughly `target_chars` on paragraph boundaries.
    """
    tex = COMMENT_RE.sub("", tex)
    body = tex.split("\\begin{document}", 1)[-1].split("\\end{document}", 1)[0]

    chunks: list[dict[str, Any]] = []
    section = "preamble"
    cursor = 0
    for match in SECTION_RE.finditer(body):
        segment = body[cursor : match.start()]
        chunks.extend(_chunk_segment(segment, section, target_chars))
        section = match.group(2).strip() or section
        cursor = match.end()
    chunks.extend(_chunk_segment(body[cursor:], section, target_chars))
    return [c for c in chunks if len(c["text"].strip()) > 60]


def _chunk_segment(segment: str, section: str, target_chars: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    remainder = segment
    for env in ENV_RE.finditer(segment):
        kind = "theorem" if env.group(1) in (
            "theorem", "lemma", "proposition", "corollary", "definition", "proof"
        ) else "equation"
        out.append({"section": section, "kind": kind, "text": _clean(env.group(0))})
        remainder = remainder.replace(env.group(0), " ")

    buffer = ""
    for para in re.split(r"\n\s*\n", remainder):
        para = _clean(para)
        if not para:
            continue
        if len(buffer) + len(para) > target_chars and buffer:
            out.append({"section": section, "kind": "text", "text": buffer.strip()})
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer.strip():
        out.append({"section": section, "kind": "text", "text": buffer.strip()})
    return out


def _clean(text: str) -> str:
    text = re.sub(r"\\(label|cite[a-z]*|ref|eqref|footnote)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(textbf|textit|emph|mathrm|mathbf)\{([^}]*)\}", r"\2", text)
    return re.sub(r"[ \t]+", " ", text).strip()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _arxiv_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("arxiv_id", help="e.g. 2001.08361 (with or without version suffix)")
    p.add_argument("--title", help="override the recorded title")
    p.add_argument("--no-vectors", action="store_true", help="FTS5 only; skip hosted embeddings")
    p.add_argument("--force", action="store_true", help="re-ingest a paper already in the index")


@cli.command("arxiv", "ingest an arXiv paper from its LaTeX source", setup=_arxiv_args)
def cmd_arxiv(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_mod.load()
    paths.ensure_workspace()
    arxiv_id = args.arxiv_id.strip().removeprefix("arXiv:").removeprefix("arxiv:")
    doc_id = f"arXiv:{arxiv_id}"

    con = corpus.connect()
    try:
        exists = con.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone()
        if exists and not args.force:
            raise GradError(
                "already_ingested",
                f"{doc_id} is already in the index",
                exit_code=2,
                fix=f"python -m tools.paper_ingest arxiv {arxiv_id} --force --json",
            )

        source_dir = paths.papers_dir() / arxiv_id.replace("/", "_")
        source_dir.mkdir(parents=True, exist_ok=True)
        tex_path = source_dir / "source.tex"
        if tex_path.exists() and not args.force:
            tex = tex_path.read_text(encoding="utf-8", errors="replace")
        else:
            blob = _fetch(ARXIV_SRC.format(id=arxiv_id), timeout=float(cfg.get("retrieval", "request_timeout_s", 60)))
            tex = _extract_tex(blob)
            tex_path.write_text(tex, encoding="utf-8")

        chunks = chunk_latex(tex)
        if not chunks:
            raise UpstreamError(
                "the LaTeX source produced no usable chunks",
                fix="inspect " + str(tex_path),
            )
        title = args.title or _title_from(tex) or doc_id
        corpus.upsert_document(
            con,
            {
                "id": doc_id, "title": title, "authors": None, "year": None,
                "source": "arxiv-latex", "path": str(tex_path), "ingested_at": now_iso(),
                "meta": {"arxiv_id": arxiv_id, "chunks": len(chunks)},
            },
        )
        chunk_ids = corpus.replace_chunks(con, doc_id, chunks)
        vectors = 0
        if not args.no_vectors:
            vectors = _embed_chunks(con, cfg, chunk_ids, [c["text"] for c in chunks])
        return {
            "document": doc_id,
            "title": title,
            "chunks": len(chunks),
            "vectors": vectors,
            "source": str(tex_path),
            "sections": sorted({c["section"] for c in chunks})[:20],
        }
    finally:
        con.close()


def _title_from(tex: str) -> str | None:
    m = re.search(r"\\title\{(.+?)\}", tex, re.DOTALL)
    return _clean(m.group(1)) if m else None


def _compute_vectors(cfg: Any, texts: list[str], *, model: str, dim: int) -> list[list[float]]:
    """Embed and validate, without touching the index.

    Kept separate from the write so `reembed` can compute the replacements
    *before* destroying what it is replacing.
    """
    vectors: list[list[float]] = []
    batch = 64
    for i in range(0, len(texts), batch):
        vectors.extend(http.embed(texts[i : i + batch], cfg=cfg, input_type="document"))
    if len(vectors) != len(texts):
        raise UpstreamError(
            f"embedded {len(vectors)} of {len(texts)} chunks",
            fix="retry; a partial batch cannot be aligned to its chunks and is not written",
        )
    if vectors and len(vectors[0]) != dim:
        raise ConfigError(
            f"{model} returned dimension {len(vectors[0])} but the index expects {dim}",
            fix=f"set retrieval.embed_dim = {len(vectors[0])} in config/grad.toml and re-embed",
        )
    return vectors


def _embed_chunks(con: Any, cfg: Any, chunk_ids: list[int], texts: list[str]) -> int:
    model = str(cfg.get("retrieval", "embed_model"))
    dim = int(cfg.get("retrieval", "embed_dim", 1024))
    corpus.bind_embedding_model(con, model, dim)
    vectors = _compute_vectors(cfg, texts, model=model, dim=dim)
    # Exact length, never a truncating slice: a short batch that quietly wrote
    # its first N vectors would pair chunk k with the embedding of some other
    # chunk, and no later check would catch it.
    corpus.store_vectors(con, chunk_ids, vectors)
    return len(vectors)


def _notes_id(path: Path) -> str:
    """A stable document id for a notes file: its path relative to the root.

    One file has to have one id however it was named on the command line. The
    id used to be `path.relative_to(root)` for an absolute path inside the
    workspace and `path.name` otherwise, so `paper_ingest notes notes/foo.md`
    and the same file by absolute path produced `notes:foo.md` and
    `notes:notes\\foo.md` -- two documents, same content, both citable -- while
    the bare-name form collided between same-named files in different folders.

    Separators are normalised because a corpus id that differs by slash
    direction is two ids on Windows and one everywhere else.
    """
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(paths.root().resolve())
    except (ValueError, OSError):
        # Outside the workspace: keep the absolute path, which is still one
        # stable id for one file.
        relative = path if path.is_absolute() else Path(path).absolute()
    return str(relative).replace("\\", "/")


def _notes_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="a markdown file or a directory of them")
    p.add_argument("--no-vectors", action="store_true")


@cli.command("notes", "ingest your own notes and derivations", setup=_notes_args)
def cmd_notes(args: argparse.Namespace) -> dict[str, Any]:
    """The half of tier 2 that no external index can ever hold."""
    cfg = config_mod.load()
    root = Path(args.path)
    if not root.exists():
        raise NotFound(f"{root} does not exist", fix="check the path")
    files = sorted(root.rglob("*.md")) if root.is_dir() else [root]
    con = corpus.connect()
    ingested = []
    try:
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks = [
                {"section": _heading_before(text, part), "kind": "note", "text": part.strip()}
                for part in re.split(r"\n\s*\n", text)
                if len(part.strip()) > 60
            ]
            if not chunks:
                continue
            doc_id = f"notes:{_notes_id(path)}"
            corpus.upsert_document(
                con,
                {"id": doc_id, "title": path.stem, "source": "notes", "path": str(path),
                 "ingested_at": now_iso(), "meta": {"chunks": len(chunks)}},
            )
            chunk_ids = corpus.replace_chunks(con, doc_id, chunks)
            if not args.no_vectors:
                _embed_chunks(con, cfg, chunk_ids, [c["text"] for c in chunks])
            ingested.append({"document": doc_id, "chunks": len(chunks)})
        return {"ingested": ingested, "files": len(files)}
    finally:
        con.close()


def _heading_before(text: str, part: str) -> str:
    index = text.find(part)
    headings = [m.group(1).strip() for m in re.finditer(r"^#+\s*(.+)$", text[:index], re.MULTILINE)]
    return headings[-1] if headings else ""


@cli.command(
    "reembed",
    "re-embed the whole corpus with a different model (deliberate, not incidental)",
    setup=lambda p: (
        p.add_argument("--model", required=True),
        p.add_argument("--dim", type=int),
        p.add_argument("--yes", action="store_true", help="confirm: this rewrites every vector"),
    ),
)
def cmd_reembed(args: argparse.Namespace) -> dict[str, Any]:
    """A model change means re-embedding the corpus, never a silent mix."""
    if not args.yes:
        raise UsageError(
            "re-embedding rewrites every vector in the index",
            fix=f"python -m tools.paper_ingest reembed --model {args.model} --yes --json",
        )
    cfg = config_mod.load()
    dim = args.dim or int(cfg.get("retrieval", "embed_dim", 1024))
    con = corpus.connect()
    try:
        rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        ids = [r["id"] for r in rows]
        texts = [r["text"] for r in rows]

        # Compute first, swap second. Deleting the old vectors up front would
        # mean a dimension mismatch, an upstream failure, or a Ctrl-C leaves the
        # index holding chunks and no vectors at all -- dense retrieval silently
        # degrades to lexical, and recovery costs another full paid re-embed.
        original = cfg.raw["retrieval"]["embed_model"]
        cfg.raw["retrieval"]["embed_model"] = args.model
        try:
            vectors = _compute_vectors(cfg, texts, model=args.model, dim=dim)
        finally:
            cfg.raw["retrieval"]["embed_model"] = original

        with con:  # one transaction: the old index survives until the new one lands
            con.execute("DELETE FROM chunk_vectors")
            con.execute("DELETE FROM meta WHERE key='embedding_model'")
            corpus.bind_embedding_model(con, args.model, dim)
            corpus.store_vectors(con, ids, vectors)
        return {"model": args.model, "dim": dim, "vectors": len(vectors)}
    finally:
        con.close()


@cli.command("list", "what is in the local index")
def cmd_list(_: argparse.Namespace) -> dict[str, Any]:
    path = paths.corpus_sqlite()
    if not path.exists():
        return {"documents": [], "note": "the index does not exist yet"}
    con = corpus.connect(path)
    try:
        rows = con.execute(
            "SELECT d.id, d.title, d.source, d.ingested_at, COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.ingested_at DESC"
        ).fetchall()
        return {"documents": [dict(r) for r in rows], **corpus.stats(con)}
    finally:
        con.close()


if __name__ == "__main__":
    main(cli)
