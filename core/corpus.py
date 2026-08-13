"""Tier-2 recall: the local index over papers actually read (HANDOFF §5).

    "This is for 'where did I see that lemma,' which no external index can answer."

SQLite FTS5 plus `sqlite-vec` in a single file. Two decisions are load-bearing:

  * ingest from arXiv **LaTeX source, not PDF** -- the largest single quality
    lever in the retrieval stack, because it preserves equations, theorem
    environments, and section structure that PDF extraction destroys. That part
    lives in `tools/paper_ingest.py`; this module stores what it produces.
  * the embedding model and version are recorded in the index, and adding
    vectors from a different model is refused. A model change is a deliberate
    re-embed, never a silent mix of incompatible vector spaces.

The two rankings are fused with reciprocal rank fusion rather than a weighted
score blend: BM25 scores and cosine similarities are on incomparable scales and
calibrating them per-corpus is exactly the tuning work this design avoids.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from core import paths
from core.errors import ConfigError

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,          -- arXiv id, doi, or notes/<path>
    title TEXT,
    authors TEXT,
    year INTEGER,
    source TEXT,                  -- 'arxiv-latex' | 'notes' | 'pdf'
    path TEXT,
    ingested_at TEXT,
    meta_json TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER,
    section TEXT,
    kind TEXT,                    -- 'text' | 'theorem' | 'equation' | 'note'
    text TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, doc_id UNINDEXED, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, section, doc_id) VALUES (new.id, new.text, new.section, new.doc_id);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section, doc_id)
    VALUES ('delete', old.id, old.text, old.section, old.doc_id);
END;
"""

# Fallback vector storage, used when sqlite-vec is unavailable. Brute force over
# a few thousand chunks is milliseconds; LanceDB is premature at this scale and
# so is anything else. Revisit past ~100k chunks.
FALLBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
"""


def connect(path: Path | None = None, *, create: bool = True) -> sqlite3.Connection:
    path = path or paths.corpus_sqlite()
    if not create and not path.exists():
        raise ConfigError(
            f"no local index at {path}",
            fix="python -m tools.paper_ingest arxiv <id> --json   # builds it on first use",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    _load_vec(con)
    return con


_VEC_AVAILABLE: bool | None = None


def _load_vec(con: sqlite3.Connection) -> bool:
    """Load sqlite-vec if present; otherwise install the fallback table."""
    global _VEC_AVAILABLE
    try:
        import sqlite_vec  # noqa: PLC0415

        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        _VEC_AVAILABLE = True
    except Exception:  # noqa: BLE001 - extension loading fails in many distinct ways
        _VEC_AVAILABLE = False
    con.executescript(FALLBACK_SCHEMA)
    return bool(_VEC_AVAILABLE)


def has_vec() -> bool:
    return bool(_VEC_AVAILABLE)


# ---------------------------------------------------------------------------
# embedding-model identity
# ---------------------------------------------------------------------------
def embedding_model(con: sqlite3.Connection) -> dict[str, Any] | None:
    row = con.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
    return json.loads(row["value"]) if row else None


def bind_embedding_model(con: sqlite3.Connection, model: str, dim: int) -> dict[str, Any]:
    """Record, or verify, the model this index's vectors come from.

    "paper_ingest.py refuses to add vectors from a model other than the one the
     index was built with, so a model change means a deliberate re-embed of the
     corpus, never a silent mix of incompatible vector spaces."
    """
    current = embedding_model(con)
    if current is None:
        record = {"model": model, "dim": dim}
        con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_model', ?)",
            (json.dumps(record),),
        )
        con.commit()
        return record
    if current["model"] != model or int(current["dim"]) != int(dim):
        raise ConfigError(
            f"this index was built with {current['model']} (dim {current['dim']}), "
            f"but {model} (dim {dim}) was requested; mixing embedding spaces makes the "
            "vector ranking noise",
            fix=(
                "python -m tools.paper_ingest reembed --model "
                f"{model} --json   # deliberate, re-embeds the whole corpus"
            ),
        )
    return current


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def upsert_document(con: sqlite3.Connection, doc: dict[str, Any]) -> None:
    con.execute(
        "INSERT OR REPLACE INTO documents(id,title,authors,year,source,path,ingested_at,meta_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            doc["id"], doc.get("title"), doc.get("authors"), doc.get("year"),
            doc.get("source"), doc.get("path"), doc.get("ingested_at"),
            json.dumps(doc.get("meta", {}), ensure_ascii=False),
        ),
    )


def replace_chunks(con: sqlite3.Connection, doc_id: str, chunks: Sequence[dict[str, Any]]) -> list[int]:
    con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    ids: list[int] = []
    for ordinal, chunk in enumerate(chunks):
        cur = con.execute(
            "INSERT INTO chunks(doc_id, ordinal, section, kind, text) VALUES (?,?,?,?,?)",
            (doc_id, ordinal, chunk.get("section", ""), chunk.get("kind", "text"), chunk["text"]),
        )
        ids.append(int(cur.lastrowid))
    con.commit()
    return ids


def store_vectors(con: sqlite3.Connection, chunk_ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
    if len(chunk_ids) != len(vectors):
        raise ValueError("chunk_ids and vectors differ in length")
    for chunk_id, vec in zip(chunk_ids, vectors):
        con.execute(
            "INSERT OR REPLACE INTO chunk_vectors(chunk_id, dim, vec) VALUES (?,?,?)",
            (chunk_id, len(vec), _pack(vec)),
        )
    con.commit()


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *[float(v) for v in vec])


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def fts_search(con: sqlite3.Connection, query: str, limit: int = 100) -> list[dict[str, Any]]:
    """BM25 over chunk text. FTS5 raises on some raw user input, so the query is
    quoted into a phrase-plus-terms form rather than passed through."""
    match = _fts_query(query)
    if not match:
        return []
    rows = con.execute(
        """
        SELECT c.id, c.doc_id, c.section, c.kind, c.text, d.title, d.year,
               bm25(chunks_fts) AS score
        FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN documents d ON d.id = c.doc_id
        WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _fts_query(query: str) -> str:
    terms = [t for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in query).split() if len(t) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


def vector_search(con: sqlite3.Connection, vector: Sequence[float], limit: int = 100) -> list[dict[str, Any]]:
    """Cosine similarity over stored chunk vectors."""
    rows = con.execute(
        "SELECT v.chunk_id, v.vec, c.doc_id, c.section, c.kind, c.text, d.title, d.year "
        "FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id JOIN documents d ON d.id = c.doc_id"
    ).fetchall()
    qnorm = math.sqrt(sum(v * v for v in vector)) or 1.0
    scored: list[dict[str, Any]] = []
    for row in rows:
        vec = _unpack(row["vec"])
        if len(vec) != len(vector):
            continue
        dot = sum(a * b for a, b in zip(vec, vector))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        scored.append(
            {
                "id": row["chunk_id"], "doc_id": row["doc_id"], "section": row["section"],
                "kind": row["kind"], "text": row["text"], "title": row["title"],
                "year": row["year"], "score": dot / (norm * qnorm),
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


def rrf(rankings: Iterable[Sequence[dict[str, Any]]], *, k: int = 60, key: str = "id") -> list[dict[str, Any]]:
    """Reciprocal rank fusion: sum of 1/(k + rank) across rankings.

    Score-free by construction, which is the point -- it needs no calibration
    between two incomparable scales.
    """
    fused: dict[Any, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            ident = item[key]
            node = fused.setdefault(ident, {**item, "rrf": 0.0, "ranks": []})
            node["rrf"] += 1.0 / (k + rank)
            node["ranks"].append(rank)
    out = sorted(fused.values(), key=lambda r: r["rrf"], reverse=True)
    return out


def stats(con: sqlite3.Connection) -> dict[str, Any]:
    docs = con.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    chunks = con.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    vectors = con.execute("SELECT COUNT(*) AS n FROM chunk_vectors").fetchone()["n"]
    return {
        "documents": docs,
        "chunks": chunks,
        "vectors": vectors,
        "embedding_model": embedding_model(con),
        "sqlite_vec": has_vec(),
        "path": str(paths.corpus_sqlite()),
    }
