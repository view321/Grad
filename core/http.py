"""HTTP clients for the hosted parts of retrieval (HANDOFF §5).

Semantic Scholar is free and unauthenticated at roughly one request per second,
so responses are cached on disk aggressively -- a funnel that re-runs the same
snippet query five times during one debugging session should cost one request.

The reranker and the embedding model are the two places retrieval spends
*credits* rather than *quota*, and they are tagged that way in the usage log so
the §5 stage decision is made on numbers that were never conflated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from core import credentials, paths, quota_log
from core.config import Config
from core.errors import ConfigError, UpstreamError

_last_request: dict[str, float] = {}


def _httpx() -> Any:
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError("httpx is not installed", fix="pip install httpx") from exc
    return httpx


def _cache_path(key: str) -> Path:
    d = paths.cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def _cached(key: str, ttl: float) -> Any | None:
    path = _cache_path(key)
    if not path.exists() or (time.time() - path.stat().st_mtime) > ttl:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _store(key: str, value: Any) -> None:
    _cache_path(key).write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")


def _throttle(host: str, min_interval: float) -> None:
    last = _last_request.get(host, 0.0)
    wait = min_interval - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.time()


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------
class SemanticScholar:
    """Tier 1: discovery over ~108M abstracts and ~12M full texts.

    `/snippet/search` is the high-value endpoint and the reason to prefer S2
    over arXiv's API: it returns ~500-word excerpts from full text, which is
    what makes triage possible without downloading anything.
    """

    def __init__(self, cfg: Config) -> None:
        self.base = str(cfg.get("retrieval", "s2_base"))
        self.timeout = float(cfg.get("retrieval", "request_timeout_s", 60))
        self.ttl = float(cfg.get("retrieval", "cache_ttl_s", 604800))
        self.interval = float(cfg.get("retrieval", "min_request_interval_s", 1.1))
        self.key = credentials.get(credentials.S2_KEY, required=False)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        key = f"s2:{path}:{json.dumps(params, sort_keys=True)}"
        hit = _cached(key, self.ttl)
        if hit is not None:
            return hit
        _throttle("s2", self.interval)
        httpx = _httpx()
        headers = {"x-api-key": self.key} if self.key else {}
        try:
            resp = httpx.get(f"{self.base}{path}", params=params, headers=headers, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(f"Semantic Scholar request failed: {exc}", fix="retry; the API is rate limited") from exc
        if resp.status_code == 429:
            raise UpstreamError(
                "Semantic Scholar rate-limited the request",
                fix="wait a few seconds and retry, or store an S2 API key: "
                "python -m tools.jobs credential set s2_api_key",
            )
        if resp.status_code >= 400:
            raise UpstreamError(
                f"Semantic Scholar returned {resp.status_code}: {resp.text[:200]}",
                fix="check the query and the endpoint",
            )
        data = resp.json()
        _store(key, data)
        return data

    def snippet_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        data = self._get("/snippet/search", {"query": query, "limit": limit})
        out = []
        for item in data.get("data", []):
            snippet = item.get("snippet", {})
            paper = item.get("paper", {})
            out.append(
                {
                    "id": f"s2:{paper.get('corpusId') or paper.get('paperId')}",
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "year": (paper.get("publicationDate") or "")[:4] or None,
                    "snippet": snippet.get("text", ""),
                    "section": (snippet.get("snippetKind") or ""),
                    "source": "s2.snippet",
                    "external": paper.get("externalIds", {}),
                }
            )
        return out

    def paper_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        fields = "title,abstract,year,externalIds,citationCount,authors"
        data = self._get("/paper/search", {"query": query, "limit": limit, "fields": fields})
        return [
            {
                "id": f"s2:{p.get('paperId')}",
                "paper_id": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "abstract": p.get("abstract") or "",
                "citations": p.get("citationCount"),
                "source": "s2.paper",
                "external": p.get("externalIds", {}),
            }
            for p in data.get("data", [])
        ]

    def neighbours(self, paper_id: str, *, direction: str = "citations", limit: int = 20) -> list[dict[str, Any]]:
        """Citation-graph expansion.

        §5: this is "worth more for recall than any reranker upgrade", because
        the retriever sets the ceiling and the graph reaches papers no query
        string does.
        """
        fields = "title,abstract,year,externalIds"
        data = self._get(f"/paper/{paper_id}/{direction}", {"fields": fields, "limit": limit})
        key = "citingPaper" if direction == "citations" else "citedPaper"
        out = []
        for item in data.get("data", []):
            p = item.get(key, {})
            if not p.get("paperId"):
                continue
            out.append(
                {
                    "id": f"s2:{p['paperId']}",
                    "paper_id": p["paperId"],
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "abstract": p.get("abstract") or "",
                    "source": f"s2.{direction}",
                    "external": p.get("externalIds", {}),
                }
            )
        return out


# ---------------------------------------------------------------------------
# OpenRouter rerank (credits, not quota)
# ---------------------------------------------------------------------------
def rerank(query: str, documents: Sequence[str], *, cfg: Config, top_n: int) -> list[dict[str, Any]]:
    """`voyageai/rerank-2.5` through OpenRouter's dedicated rerank endpoint.

    Hosted on purpose: local reranking competes for the same VRAM as the
    experiments this agent exists to run. It costs credits rather than quota,
    which is why it sits between the two Haiku stages -- the quota-consuming
    stage never sees the 350 candidates that were obviously wrong.
    """
    if not documents:
        return []
    key = credentials.get(credentials.OPENROUTER_KEY)
    base = str(cfg.get("retrieval", "openrouter_base"))
    model = str(cfg.get("retrieval", "rerank_model"))
    httpx = _httpx()
    try:
        resp = httpx.post(
            f"{base}/rerank",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "query": query, "documents": list(documents), "top_n": top_n},
            timeout=float(cfg.get("retrieval", "request_timeout_s", 60)),
        )
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(f"rerank request failed: {exc}", fix="retry, or run with --no-rerank") from exc
    if resp.status_code >= 400:
        raise UpstreamError(
            f"rerank returned {resp.status_code}: {resp.text[:200]}",
            fix="check the OpenRouter key and that the model id is still served",
        )
    data = resp.json()
    usage = data.get("usage", {}) or {}
    quota_log.record(
        quota_log.STAGE_RERANK,
        model=model,
        unit="credits",
        credits_usd=float(usage.get("cost", 0.0) or 0.0),
        detail={"documents": len(documents), "top_n": top_n},
    )
    return [
        {"index": r.get("index"), "score": r.get("relevance_score", r.get("score"))}
        for r in data.get("results", [])
    ]


# ---------------------------------------------------------------------------
# Voyage embeddings (credits, not quota)
# ---------------------------------------------------------------------------
def embed(texts: Sequence[str], *, cfg: Config, input_type: str = "document") -> list[list[float]]:
    """Hosted Voyage embeddings.

    Same no-VRAM-contention logic as the reranker. The model name and dimension
    are recorded in the index and enforced there, so this function never has to
    know whether the corpus it is embedding for was built with something else.
    """
    if not texts:
        return []
    key = credentials.get(credentials.VOYAGE_KEY)
    model = str(cfg.get("retrieval", "embed_model"))
    httpx = _httpx()
    try:
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "input": list(texts), "input_type": input_type},
            timeout=float(cfg.get("retrieval", "request_timeout_s", 60)),
        )
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(f"embedding request failed: {exc}", fix="retry, or ingest with --no-vectors") from exc
    if resp.status_code >= 400:
        raise UpstreamError(
            f"embeddings returned {resp.status_code}: {resp.text[:200]}",
            fix="check the Voyage key: python -m tools.jobs credential set voyage_key",
        )
    data = resp.json()
    quota_log.record(
        quota_log.STAGE_EMBED,
        model=model,
        unit="credits",
        detail={"texts": len(texts), "total_tokens": (data.get("usage") or {}).get("total_tokens")},
    )

    # The caller zips these against chunk ids, so position *is* identity here.
    # A short or reordered batch would attach wrong vectors to chunks and
    # nothing downstream would notice: `vector_search` silently skips rows whose
    # dimension differs, and a wrong-but-same-dimension vector is undetectable.
    rows = data.get("data", [])
    if len(rows) != len(texts):
        raise UpstreamError(
            f"asked for {len(texts)} embeddings and got {len(rows)}",
            fix="retry; a partial batch cannot be aligned to its chunks and is not written",
        )
    rows = sorted(rows, key=lambda r: r.get("index", 0))
    return [row["embedding"] for row in rows]
