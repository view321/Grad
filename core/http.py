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


#: Where a paper's identity is read from, **in this order, everywhere**.
#:
#: The order is the point. `/snippet/search` returns `corpusId` and `paperId`;
#: `/paper/search` is asked for fields that include neither corpus id nor
#: anything but `paperId`; Asta returns whichever its own shape carries. Reading
#: them in different orders in different methods -- corpus id first in one, SHA
#: only in another -- gave the *same paper* two ids, and `corpus.rrf` fuses by
#: id, so it ranked twice and took a slot from something else. `cmd_search`
#: calls both endpoints for every expanded query, so that was the ordinary path
#: and not a corner of it.
#:
#: `paperId` leads because it is the one field every endpoint returns, and
#: because `paper_id` -- the seed `neighbours` expands from -- is read from it
#: too. One field decides both, so a candidate and its citation expansion cannot
#: disagree about which paper they are.
IDENTITY_KEYS = ("paperId", "paper_id", "corpusId", "corpus_id", "id")


def identifier_of(paper: dict[str, Any]) -> Any:
    """A paper's identity, by `IDENTITY_KEYS`. None when it carries none."""
    for key in IDENTITY_KEYS:
        value = paper.get(key)
        if value not in (None, ""):
            return value
    return None


def candidate_id(
    identifier: Any, title: Any = "", text: Any = "", *, namespace: str = "s2"
) -> str | None:
    """The key the funnel fuses candidates on, or None when there is not one.

    Both tier-1 clients mint ids in one `s2:` namespace, deliberately: it is one
    corpus, so a paper found through both has to fuse to a single candidate
    rather than rank twice under two names.

    That sharing is also why an id-less hit cannot be given a shared literal.
    Formatting a missing identifier produced `"s2:None"`, and since `corpus.rrf`
    fuses by id, *every* hit without one -- from either client, across every
    query in the run -- collapsed into a single phantom candidate. Distinct
    papers vanished into each other with nothing on screen to say so, which is
    the same class of failure as an empty result that reads as "the literature
    has nothing on this".

    So: the real id when there is one; a digest of what the reranker would read
    when there is not, which fuses genuine duplicates and separates genuine
    distinctions; and None when there is neither, because a hit with no id, no
    title and no text has nothing to rank and nothing to cite.

    `namespace` is what keeps that sharing honest once there is more than one
    corpus. Asta and S2 share `s2:` because they are the same index and the same
    ids; Papers with Code is a different catalogue with its own numbering, and
    giving it the same prefix would fuse two unrelated papers whose ids happened
    to collide. The digest fallback is namespaced too, so a title-only hit from
    one corpus does not silently absorb a title-only hit from another.
    """
    if identifier not in (None, ""):
        return f"{namespace}:{identifier}"
    material = f"{title or ''}\n{str(text or '')[:400]}".strip()
    if not material:
        return None
    return f"{namespace}:t-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


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
            key = candidate_id(
                identifier_of(paper), paper.get("title"), snippet.get("text", "")
            )
            if key is None:
                continue
            out.append(
                {
                    "id": key,
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
        out = []
        for p in data.get("data", []):
            key = candidate_id(identifier_of(p), p.get("title"), p.get("abstract"))
            if key is None:
                continue
            out.append(
                {
                    "id": key,
                    "paper_id": p.get("paperId"),
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "abstract": p.get("abstract") or "",
                    "citations": p.get("citationCount"),
                    "source": "s2.paper",
                    "external": p.get("externalIds", {}),
                }
            )
        return out

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
# Papers with Code -- the corpus that answers in seconds
# ---------------------------------------------------------------------------
class PapersWithCode:
    """Tier 1, and the default: the ML/AI catalogue behind `paperswithcode.co`.

    **Why this replaced Asta as the default.** Asta serves the right corpus and
    is the only one of these with genuine full-text snippets, but measured
    against the live service it answers a `search_papers_by_relevance` in ~121
    seconds and takes ~283 seconds to report that its own backend refused a
    connection. Stage 0 turns one question into six queries and each goes to two
    endpoints, so that is twenty minutes of discovery before anything is ranked,
    and every caller -- the agent's Bash tool at 120s, a shell `timeout`, a
    person -- gives up first. This answers in one to two seconds. A retriever
    that returns is worth more than a better one that does not.

    **Two search modes, and they are genuinely different rankings.** `keyword`
    is lexical and `semantic` is dense, which is exactly the pair `corpus.rrf`
    exists to fuse -- so the two verbs the funnel already calls per query map
    onto them without the funnel knowing anything changed.

    **What is given up, stated plainly.** Search rows carry no abstract, so §5's
    "triage on ~500 words of the paper itself" becomes "triage on the abstract",
    and the abstracts are fetched separately -- see `arxiv_abstracts`, which
    fills the whole pool in one request because nearly every row here is an
    arXiv paper. `related` is a *dense neighbour* rather than a citation edge,
    and `neighbours` says so rather than presenting it as the citation graph
    §5 asks for.

    Anonymous and read-only: no account, no key, nothing to store. The
    endpoints and their parameters are read off `huggingface/pwc-cli`, which is
    the reference client for this API.
    """

    #: The largest page this API will return, checked against the live service.
    MAX_PAGE = 100

    #: `funnel verb -> the API's search mode`. The funnel calls both per query
    #: and fuses them, which is the whole point of there being two.
    MODES = {"snippet_search": "semantic", "paper_search": "keyword"}

    def __init__(self, cfg: Config) -> None:
        self.base = str(cfg.get("retrieval", "pwc_base")).rstrip("/")
        self.timeout = float(cfg.get("retrieval", "request_timeout_s", 60))
        self.ttl = float(cfg.get("retrieval", "cache_ttl_s", 604800))
        self.interval = float(cfg.get("retrieval", "min_request_interval_s", 1.1))

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        key = f"pwc:{self.base}:{path}:{json.dumps(params or {}, sort_keys=True)}"
        hit = _cached(key, self.ttl)
        if hit is not None:
            return hit
        _throttle("pwc", self.interval)
        httpx = _httpx()
        try:
            resp = httpx.get(
                f"{self.base}/{path.lstrip('/')}",
                params=params,
                headers={"Accept": "application/json", "User-Agent": "grad/1"},
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Papers with Code request failed: {exc}",
                fix="retry, or run with --local-only to search papers already ingested",
            ) from exc
        if resp.status_code == 429:
            raise UpstreamError(
                "Papers with Code rate-limited the request",
                fix="wait and retry; this API is anonymous, so there is no key to raise it",
            )
        if resp.status_code >= 400:
            raise UpstreamError(
                f"Papers with Code returned {resp.status_code}: {resp.text[:200]}",
                fix=(
                    "the endpoint may have moved: check github.com/huggingface/pwc-cli and "
                    "set [retrieval] pwc_base in config/grad.toml"
                ),
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Papers with Code returned a body that is not JSON: {resp.text[:200]}",
                fix="retry; if it persists the API contract has changed",
            ) from exc
        _store(key, data)
        return data

    def _search(self, query: str, mode: str, limit: int) -> list[dict[str, Any]]:
        data = self._get(
            "papers/search",
            {
                "q": query,
                "page": 1,
                "page_size": max(1, min(self.MAX_PAGE, int(limit))),
                "mode": mode,
            },
        )
        return _normalise_pwc(_pwc_rows(data, f"papers/search ({mode})"), f"pwc.{mode}")

    def snippet_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """The dense side. Named for the verb the funnel calls, not for what it
        returns: there are no snippets here, and `arxiv_abstracts` is what makes
        the candidates readable enough to rerank."""
        return self._search(query, self.MODES["snippet_search"], limit)

    def paper_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """The lexical side."""
        return self._search(query, self.MODES["paper_search"], limit)

    def neighbours(
        self, paper_id: str, *, direction: str = "citations", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Related work, and **not** the citation graph.

        The rows this returns carry `provenance: "dense"` and a similarity
        score: they are nearest neighbours in an embedding space, not papers
        that cite or are cited by the seed. That still buys recall -- §5's point
        is that expansion reaches papers no query string does -- but calling it
        a citation edge would put a claim in the trace that the data does not
        support, and a ledger entry's basis is the thing that must not be
        overstated. The backward direction is refused for the same reason it is
        on Asta: there is no endpoint for it, and answering it with the forward
        one would double-count a single direction under two names.
        """
        if direction != "citations":
            return []
        data = self._get(
            f"papers/{_quote(paper_id)}/related",
            {"limit": max(1, min(self.MAX_PAGE, int(limit)))},
        )
        return _normalise_pwc(_pwc_rows(data, "related"), "pwc.related")


def _quote(value: Any) -> str:
    """A path segment. Ids here are arXiv ids and integers, but they reach a URL
    from a search result rather than from a constant."""
    from urllib.parse import quote  # noqa: PLC0415

    return quote(str(value), safe="._-")


def _pwc_rows(data: Any, what: str) -> list[dict[str, Any]]:
    """The hits inside a response: `{"results": [...]}`, or a bare list.

    `related` answers with a list and `search` with an envelope, so both are
    read. An unrecognised shape raises for the reason `_rows` does: `ok: true`
    with no results reads as "the literature has nothing on this".
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "items", "data"):
            if isinstance(data.get(key), list):
                return [item for item in data[key] if isinstance(item, dict)]
    raise UpstreamError(
        f"Papers with Code's {what} returned no recognisable list of hits: "
        f"{sorted(data)[:8] if isinstance(data, dict) else type(data).__name__}",
        fix=(
            "the API contract has changed -- compare it against "
            "github.com/huggingface/pwc-cli and update core/http.py:_pwc_rows"
        ),
    )


def _normalise_pwc(items: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = []
    for item in items:
        row = _pwc_row(item, source=source)
        if row is not None:
            out.append(row)
    return out


def _pwc_row(item: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    """One hit, in the vocabulary the funnel already fuses and reranks.

    The arXiv id leads, and that is deliberate: it is the field that survives
    into `external`, that `paper_ingest` takes, and that `arxiv_abstracts`
    fetches on -- so a candidate, its abstract and the ingest of its PDF all
    name the same paper. The catalogue's own id is the fallback for the rows
    that are not arXiv preprints.
    """
    arxiv = str(item.get("arxiv_id") or "").strip()
    identifier = arxiv or item.get("id") or item.get("route_identifier")
    key = candidate_id(identifier, item.get("title"), "", namespace="pwc")
    if key is None:
        return None
    published = str(item.get("published") or "")
    return {
        "id": key,
        # What `neighbours` expands from: the API takes either, and the arXiv id
        # is the one that means something outside this catalogue.
        "paper_id": arxiv or str(item.get("id") or ""),
        "title": item.get("title"),
        "year": published[:4] or None,
        # No full text here -- see the class docstring. `abstract` is filled in
        # afterwards, and the reranker reads whichever of the two is present.
        "snippet": "",
        "abstract": str(item.get("abstract") or ""),
        "section": "",
        "source": source,
        "citations": item.get("citation_count"),
        "external": {"ArXiv": arxiv} if arxiv else {},
        "url": item.get("url_abs") or "",
    }


# ---------------------------------------------------------------------------
# arXiv -- one request, every abstract
# ---------------------------------------------------------------------------
#: How many ids arXiv will take in one `id_list`. The whole reason to use this
#: rather than a per-paper lookup: a pool of a hundred candidates costs one
#: request instead of a hundred.
ARXIV_BATCH = 100


def arxiv_abstracts(
    arxiv_ids: Sequence[str], *, cfg: Config, timeout: float | None = None
) -> dict[str, str]:
    """Abstracts for a batch of arXiv ids, as `{id: abstract}`.

    This exists because the fast corpus does not carry abstracts in its search
    results and the reranker and stage-3 triage both read them: a candidate pool
    of titles alone is a measurably worse funnel. Fetching them one at a time
    would cost a request per candidate and undo the reason the corpus was
    changed, so this uses `id_list`, which takes a hundred at once.

    Never raises. A missing abstract is a candidate that ranks on its title,
    which is what would have happened anyway -- so a failure here degrades the
    ranking rather than the run, and the caller records it as a warning.
    """
    ids = [str(i).strip() for i in arxiv_ids if str(i or "").strip()][:ARXIV_BATCH]
    if not ids:
        return {}
    key = f"arxiv:abstracts:{json.dumps(sorted(ids))}"
    hit = _cached(key, float(cfg.get("retrieval", "cache_ttl_s", 604800)))
    if isinstance(hit, dict):
        return hit
    _throttle("arxiv", max(3.0, float(cfg.get("retrieval", "min_request_interval_s", 1.1))))
    httpx = _httpx()
    try:
        resp = httpx.get(
            str(cfg.get("retrieval", "arxiv_base")),
            params={"id_list": ",".join(ids), "max_results": len(ids)},
            headers={"User-Agent": "grad/1"},
            timeout=timeout or float(cfg.get("retrieval", "request_timeout_s", 60)),
        )
        resp.raise_for_status()
        out = _parse_arxiv_atom(resp.text)
    except Exception:  # noqa: BLE001 - see the docstring: this degrades, it does not fail
        return {}
    _store(key, out)
    return out


def _parse_arxiv_atom(xml: str) -> dict[str, str]:
    """`{bare arxiv id: abstract}` out of an Atom feed.

    The id in the feed is a URL with a version suffix (`.../abs/1706.03762v7`)
    and the id asked for has neither, so it is reduced to the bare form -- the
    caller looks its candidates up by what Papers with Code gave it.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    out: dict[str, str] = {}
    for entry in root.findall("atom:entry", namespace):
        raw = (entry.findtext("atom:id", "", namespace) or "").rsplit("/", 1)[-1]
        bare = raw.split("v")[0] if raw else ""
        summary = " ".join((entry.findtext("atom:summary", "", namespace) or "").split())
        if bare and summary:
            out[bare] = summary
    return out


# ---------------------------------------------------------------------------
# Asta -- the same corpus, through a door that opens
# ---------------------------------------------------------------------------
#: The MCP protocol version this client speaks. Sent on `initialize` and echoed
#: on every request after it; a server that wants another version says so in its
#: `initialize` result and this follows it.
MCP_PROTOCOL_VERSION = "2025-06-18"

#: The largest `limit` Asta's tools accept, from the service: *"The limit
#: parameter must be between 1 and 100 inclusive."*
#:
#: Clamped here rather than left to callers because it is the service's
#: constraint, not the funnel's. `paper_search.py` divides its candidate ceiling
#: across the expanded queries, so skipping stage 0 -- one query instead of six
#: -- asks for six times as many per call, and the funnel's own `--no-expand`
#: path therefore refused *every* tier-1 call with a validation error. Asking
#: for the most the service will give is the honest reading of "as many as you
#: can"; raising `--candidates` should not be a way to get zero results.
MAX_LIMIT = 100


class Asta:
    """Ai2's scientific corpus over MCP, at `asta-tools.allen.ai`.

    **Why this exists.** `SemanticScholar` above is the better-documented client
    and it is not reachable: Ai2 stopped accepting API key requests from
    free-domain email addresses, so a personal account cannot get one, and the
    anonymous pool is shared with every other unauthenticated caller and is
    near-permanently rate limited. Asta is the *same index* -- Ai2 describe the
    MCP tool as an extension of the Semantic Scholar API -- and it exposes
    `snippet_search`, which is the endpoint §5's funnel is actually built around:
    ~500-word excerpts from full text are what make triage possible without
    downloading anything. A key is optional here and raises limits rather than
    unlocking anything.

    **Why it is not an MCP integration.** §5 already settles this: Asta's
    endpoint is reached "over streamable HTTP without adopting MCP as an
    architecture". Streamable HTTP is a POST with a JSON-RPC body; the parts of
    MCP that would be an architecture -- a client runtime, a tool registry, a
    server lifecycle -- buy nothing when the whole surface is three calls. So
    this is `httpx` and the same disk cache, rate limiter and usage log as
    everything else in this module.

    **What is unverified.** The endpoint, the transport and the tool names are
    from Ai2's published documentation. The *shape of each tool's result* is
    not, because that needs a live call. So `_rows` reads both the shape S2's
    REST API uses (`{"data": [{"snippet": …, "paper": …}]}`) and a flattened
    one, and an unrecognised payload becomes an `UpstreamError` naming what came
    back rather than an empty list that reads as "the literature has nothing".
    """

    def __init__(self, cfg: Config) -> None:
        self.base = str(cfg.get("retrieval", "asta_base")).rstrip("/")
        self.timeout = float(cfg.get("retrieval", "request_timeout_s", 60))
        #: Total wall clock for one request, enforced here rather than by httpx.
        #: See `_post`: the per-read timeout above cannot bound a stream that is
        #: being kept alive.
        self.deadline = float(cfg.get("retrieval", "request_deadline_s", 300))
        self.ttl = float(cfg.get("retrieval", "cache_ttl_s", 604800))
        self.interval = float(cfg.get("retrieval", "min_request_interval_s", 1.1))
        # Optional, and an unreachable store counts as absent -- see
        # `credentials.get`, which answers that for every optional read now.
        self.key = credentials.get(credentials.ASTA_KEY, required=False)
        self._session: str | None = None
        self._protocol = MCP_PROTOCOL_VERSION
        self._id = 0
        #: Set only once `initialize` *and* the notification after it have
        #: succeeded. See `_handshake` for why this is not inferred.
        self._ready = False

    @property
    def authenticated(self) -> bool:
        return bool(self.key)

    # -- transport ----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            # Both, because streamable HTTP lets the server answer either way
            # for the same request and does not tell you which in advance.
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self._protocol,
        }
        if self.key:
            headers["x-api-key"] = self.key
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        return headers

    def _post(self, body: dict[str, Any]) -> Any:
        """One JSON-RPC message out, one back -- under a deadline this enforces.

        **The response is streamed rather than buffered, and that is the whole
        fix.** Asta answers `tools/call` with an event stream and holds it open
        while it works, sending `: ping` comments every 15 seconds. `httpx`'s
        `timeout` is per socket read, so every ping reset it: a buffered
        `httpx.post` waited for a close that the server had no obligation to
        send, the read timeout could never fire no matter how long it took, and
        the funnel's first tier-1 call simply never returned. Discovery was
        unreachable, and the failure had no error to go with it.

        Two things bound it now. `_mcp_payload` stops at the reply to *this*
        request rather than at the end of the stream -- the answer is what was
        asked for, and reading past it is waiting for a close nobody promised --
        and `deadline` caps the whole exchange in case the reply never comes.
        """
        _throttle("asta", self.interval)
        httpx = _httpx()
        deadline = time.monotonic() + self.deadline
        try:
            with httpx.stream(
                "POST", self.base, json=body, headers=self._headers(), timeout=self.timeout
            ) as resp:
                # Assigned on `initialize` and echoed from then on. A server that
                # does not use sessions simply never sends it.
                session = resp.headers.get("mcp-session-id")
                if session:
                    self._session = session
                if resp.status_code >= 400:
                    self._refuse(resp)
                # A notification gets 202 Accepted and an empty body; there is
                # nothing to parse and nothing to wait for.
                if resp.status_code == 202:
                    return None
                return _mcp_payload(resp, request_id=body.get("id"), deadline=deadline)
        except UpstreamError:
            # Already the shaped refusal, with the fix that belongs to it. Left
            # alone rather than re-wrapped as a transport failure -- "Asta rate
            # limited the request" and "Asta request failed" send you to two
            # different places.
            raise
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Asta request failed: {exc}",
                fix="retry, or run with --local-only to search papers already ingested",
            ) from exc

    def _refuse(self, resp: Any) -> None:
        """Turn a 4xx/5xx into the refusal that says what to do about it."""
        resp.read()
        if resp.status_code in (401, 403):
            raise UpstreamError(
                f"Asta rejected the request ({resp.status_code})",
                fix=(
                    "the corpus tool is usable anonymously; if a key is stored it may be "
                    f"wrong: python -m tools.jobs credential set {credentials.ASTA_KEY}"
                ),
            )
        if resp.status_code == 429:
            raise UpstreamError(
                "Asta rate-limited the request",
                fix=(
                    "wait, or store a key to raise the limit -- it is requested from a form "
                    "rather than reviewed, so a personal address is fine: "
                    f"python -m tools.jobs credential set {credentials.ASTA_KEY}"
                ),
            )
        raise UpstreamError(
            f"Asta returned {resp.status_code}: {resp.text[:200]}",
            fix=(
                "the endpoint may have moved: check allenai.org/asta/resources/mcp and "
                "set [retrieval] asta_base in config/grad.toml"
            ),
        )

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._id += 1
        payload = self._post(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        )
        if payload is None:
            raise UpstreamError(
                f"Asta returned no body for {method}",
                fix="retry; if it persists the endpoint may have changed transport",
            )
        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            raise UpstreamError(
                f"Asta refused {method}: {error.get('message') or error}",
                fix="check the tool name and its arguments against allenai.org/asta/resources/mcp",
            )
        return (payload or {}).get("result")

    def _handshake(self) -> None:
        """`initialize`, then the notification that says the client is ready.

        Once per client, and `_ready` is an explicit flag rather than something
        inferred from `_session` or `_id`. Inferring it from the request counter
        was wrong in the case that matters: `_call` increments the counter before
        it sends, so an `initialize` that *failed* -- a timeout, a 429 on the
        very first call -- left the counter non-zero and every later request
        skipped the handshake and went straight to `tools/call` on a connection
        that was never initialised. One transient failure poisoned the client for
        the life of the process.
        """
        if self._ready:
            return
        result = self._call(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "grad", "version": "1"},
            },
        )
        negotiated = (result or {}).get("protocolVersion")
        if isinstance(negotiated, str) and negotiated:
            self._protocol = negotiated
        # A notification: no id, so no reply is expected and none is waited for.
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self._ready = True

    def tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one MCP tool, through the disk cache.

        The cache key deliberately excludes the session: a session is a
        transport detail and two of them asking the same question of the same
        corpus should not cost two requests.
        """
        key = f"asta:{self.base}:{name}:{json.dumps(arguments, sort_keys=True)}"
        hit = _cached(key, self.ttl)
        if hit is not None:
            return hit
        self._handshake()
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        if isinstance(result, dict) and result.get("isError"):
            raise UpstreamError(
                f"Asta's {name} failed: {_mcp_text(result)[:200]}",
                fix="check the arguments; the tool ran and reported an error",
            )
        data = _mcp_result(result)
        _store(key, data)
        return data

    # -- the three calls the funnel makes ------------------------------------
    def snippet_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text excerpts. The reason to prefer this over a metadata search:
        ~500 words of the paper itself is what stage 3 triages on."""
        data = self.tool("snippet_search", {"query": query, "limit": _limit(limit)})
        return _normalise(_rows(data, "snippet_search"), "asta.snippet")

    def paper_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        # `keyword`, not `query`. The two tools disagree: `snippet_search` takes
        # `query` and this one takes `keyword`, and sending the wrong one is not
        # a soft failure -- the server answers in ~1s with a pydantic validation
        # error ("keyword Field required"), which `tool` raises as an
        # UpstreamError, so tier 1 lost this endpoint entirely on every search
        # while snippet_search's slowness hid it. Checked against tools/list.
        data = self.tool(
            "search_papers_by_relevance", {"keyword": query, "limit": _limit(limit)}
        )
        return _normalise(_rows(data, "search_papers_by_relevance"), "asta.paper")

    def neighbours(
        self, paper_id: str, *, direction: str = "citations", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Citation-graph expansion.

        §5: worth more for recall than any reranker upgrade, because the
        retriever sets the ceiling and the graph reaches papers no query string
        does. Asta publishes `get_citations` and no references counterpart, so
        the backward direction is refused here rather than silently answered
        with the forward one -- which would quietly double-count one direction.
        """
        if direction != "citations":
            return []
        data = self.tool("get_citations", {"paper_id": paper_id, "limit": _limit(limit)})
        return _normalise(_rows(data, "get_citations"), "asta.citations")


def _limit(value: Any) -> int:
    """A `limit` the service will accept. See `MAX_LIMIT`."""
    try:
        wanted = int(value)
    except (TypeError, ValueError):
        return 20
    return max(1, min(MAX_LIMIT, wanted))


def _mcp_payload(resp: Any, *, request_id: Any = None, deadline: float | None = None) -> Any:
    """One JSON-RPC message out of a streamable-HTTP response.

    The same request may be answered with `application/json` or with an SSE
    stream, at the server's discretion, so both are handled.

    For a stream this **stops at the reply to `request_id`** rather than reading
    to the end. That is not an optimisation: the server may keep the stream open
    after answering, pinging every 15 seconds, and a reader that waits for the
    close waits forever -- which is exactly how a 60-second read timeout failed
    to bound a call that never returned. The reply is the answer; there is
    nothing after it worth waiting for.

    Progress notifications share the channel, so frames are filtered to those
    carrying a `result` or an `error`, and the last one seen is the fallback for
    a server that answers without echoing the id.
    """
    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/event-stream" not in content_type:
        resp.read()
        if not (resp.content or b"").strip():
            return None
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Asta returned a body that is not JSON: {resp.text[:200]}",
                fix="retry; if it persists the endpoint may no longer speak streamable HTTP",
            ) from exc

    answer: Any = None
    #: `data:` lines seen since the last event boundary. An SSE event may carry
    #: its payload across several of them, joined with newlines -- which is not
    #: an exotic corner of the spec but the ordinary way a server emits a JSON
    #: body large enough to wrap. Parsing each line on its own throws away every
    #: such message as unparseable JSON, so a long enough answer looked exactly
    #: like a stream that carried no result at all.
    chunks: list[str] = []

    def _frame() -> Any:
        """The completed event's payload, or None if it is not one of ours."""
        if not chunks:
            return None
        try:
            value = json.loads("\n".join(chunks))
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict) or not ("result" in value or "error" in value):
            return None
        return value

    for line in resp.iter_lines():
        if deadline is not None and time.monotonic() > deadline:
            raise UpstreamError(
                "Asta held the stream open past its deadline without answering",
                fix=(
                    "retry; raise [retrieval] request_deadline_s in config/grad.toml if the "
                    "corpus is genuinely this slow, or use --local-only meanwhile"
                ),
            )
        line = str(line).rstrip("\r")
        if line.startswith("data:"):
            # Exactly one leading space is part of the framing, not the data.
            chunks.append(line[6:] if line.startswith("data: ") else line[5:])
            continue
        if line:
            continue  # `event:`, `id:`, `retry:`, or a `:` comment
        frame = _frame()
        chunks.clear()
        if frame is None:
            continue
        answer = frame
        if request_id is None or frame.get("id") == request_id:
            return answer
    # A stream that ends without a trailing blank line still delivered its last
    # event; dropping it would fail on precisely the well-formed response that
    # arrived in one piece and closed.
    frame = _frame()
    if frame is not None:
        answer = frame
        if request_id is None or frame.get("id") == request_id:
            return answer
    if answer is None:
        raise UpstreamError(
            "Asta's event stream carried no result",
            fix="retry; the stream held only notifications",
        )
    return answer


def _mcp_text(result: Any) -> str:
    """The text content blocks of a `tools/call` result, joined."""
    if not isinstance(result, dict):
        return ""
    parts = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _mcp_result(result: Any) -> Any:
    """The data a tool returned, whichever way it chose to return it.

    `structuredContent` is the typed channel and is preferred. Failing that the
    text block is usually JSON; failing *that* it is prose, and it is handed back
    as-is rather than discarded -- `_rows` is where an unusable shape becomes an
    error that says what arrived.
    """
    if isinstance(result, dict) and result.get("structuredContent") is not None:
        return result["structuredContent"]
    text = _mcp_text(result)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


#: The keys a tool's payload may wrap its hits under, in the order they are
#: tried. `result` is first because it is the one a live call actually returns
#: -- `search_papers_by_relevance` answers `{"result": [{"paperId": …, "title":
#: …}]}` -- and its absence is why tier 1 found nothing even once the argument
#: name was right: every hit was thrown away as an unrecognised envelope, and
#: the funnel raised rather than returning them.
#:
#: Singular, and *not* the JSON-RPC `result` field. By the time this runs, `tool`
#: has already unwrapped the RPC envelope and the tool's own content block, so
#: this is the payload's own key -- they are only spelled the same.
ROW_KEYS = ("result", "data", "results", "snippets", "papers", "citations", "items")


def _rows(data: Any, tool_name: str) -> list[dict[str, Any]]:
    """The list of hits inside a tool's payload, whatever it is wrapped in.

    Partly verified against the live service now: `search_papers_by_relevance`
    uses `result`. The rest are the S2 REST shape and the obvious flattenings of
    it, still unverified. An unrecognised payload raises: a search that quietly
    returns nothing reads as "the literature has nothing on this", and that is a
    conclusion nobody should draw from a schema change.
    """
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ROW_KEYS:
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
        else:
            raise UpstreamError(
                f"Asta's {tool_name} returned no recognisable list of hits: "
                f"keys were {sorted(data)[:8]}",
                fix=(
                    "the tool's result shape has changed -- compare it against "
                    "allenai.org/asta/resources/mcp and update core/http.py:_rows"
                ),
            )
    else:
        raise UpstreamError(
            f"Asta's {tool_name} returned {type(data).__name__}, not a result set: "
            f"{str(data)[:200]}",
            fix="retry; if it persists the tool name or its arguments have changed",
        )
    return [c for c in candidates if isinstance(c, dict)]


def _normalise(items: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Hits in the funnel's vocabulary, dropping any that cannot be fused.

    A hit with no identifier, no title and no text has nothing for the reranker
    to read and nothing to cite. Dropping it costs no recall; keeping it under a
    shared placeholder id cost real recall, because fusion is by id -- see
    `candidate_id`.
    """
    out = []
    for item in items:
        row = _row(item, source=source)
        if row is not None:
            out.append(row)
    return out


def _row(item: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    """One hit, in the shape the funnel already fuses and reranks.

    `paper_search.py` does not know which tier a candidate came from, and it
    must not have to: RRF fuses rankings by id, and the reranker reads title and
    snippet. So the two clients in this module answer in one vocabulary.
    """
    # The `s2:` prefix is shared with `SemanticScholar` on purpose, not by
    # accident of copying: it is the same corpus and the same corpus ids, so a
    # paper found through both tiers has to fuse to one candidate rather than
    # rank twice under two names.
    #
    # The S2 shape nests the paper under the snippet; the flat one does not.
    paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
    snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else item

    text = snippet.get("text") or item.get("text") or ""
    key = candidate_id(identifier_of(paper), paper.get("title"), text)
    if key is None:
        return None
    year = paper.get("year")
    if year is None:
        year = (str(paper.get("publicationDate") or "")[:4]) or None
    external = paper.get("externalIds") or paper.get("external_ids") or {}
    return {
        "id": key,
        # The same field the id came from, by the same order -- so the seed
        # `neighbours` expands from cannot name a different paper than the
        # candidate it was taken from.
        "paper_id": paper.get("paperId") or paper.get("paper_id") or paper.get("id"),
        "title": paper.get("title"),
        "year": year,
        "snippet": text,
        "abstract": paper.get("abstract") or "",
        "section": snippet.get("snippetKind") or snippet.get("section") or "",
        "source": source,
        "external": external if isinstance(external, dict) else {},
    }


# ---------------------------------------------------------------------------
# Context7 (HANDOFF-2 §18) -- what is *current*, as opposed to what is installed
# ---------------------------------------------------------------------------
class Context7:
    """Library documentation over plain HTTP.

    This is the second instance of an existing pattern, not a new one: §5
    already reaches Asta's MCP endpoint over streamable HTTP "without adopting
    MCP as an architecture", and the same applies here. `tools/docs.py` wraps it
    rather than allowlisting the official `ctx7` CLI, because the `--json` /
    exit-code / `fix`-field contract is what makes a tool legible to the model
    (§8) and `ctx7` does not have it -- and because this is where the credential
    fetch and the cache live. Documentation lookups repeat heavily, and
    `core/http.py` already has the TTL cache and the rate limiter.

    The endpoint paths and the response keys below are verified against the live
    API. They remain configurable because a third-party API can move, and a 404
    should be a one-line config edit rather than a code change -- the error says
    which path it tried, so the mismatch names itself.
    """

    def __init__(self, cfg: Config) -> None:
        self.base = str(cfg.get("docs", "base", "https://context7.com")).rstrip("/")
        self.resolve_path = str(cfg.get("docs", "resolve_path", "/api/v2/libs/search"))
        self.docs_path = str(cfg.get("docs", "docs_path", "/api/v2/context"))
        self.timeout = float(cfg.get("docs", "request_timeout_s", 30))
        self.ttl = float(cfg.get("docs", "cache_ttl_s", 86400))
        self.interval = float(cfg.get("docs", "min_request_interval_s", 0.5))
        # Free from their dashboard, and it raises rate limits rather than
        # unlocking anything -- so its absence is a note, not an error. That
        # includes the case where the credential *store* is unreachable, which
        # `credentials.get` now answers with `None` for any optional read rather
        # than leaving each caller to catch it; this site used to do the catching
        # and the two beside it did not, which is how a headless Linux host lost
        # anonymous retrieval for want of keys it does not need.
        self.key = credentials.get(credentials.CONTEXT7_KEY, required=False)

    @property
    def authenticated(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base}{path}"
        key = f"ctx7:{url}:{json.dumps(params, sort_keys=True)}:{bool(self.key)}"
        hit = _cached(key, self.ttl)
        if hit is not None:
            return hit
        _throttle("context7", self.interval)
        httpx = _httpx()
        headers = {"Accept": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Context7 request failed: {exc}",
                fix="retry, or run `python -m tools.docs check <file>` which works offline",
            ) from exc
        if resp.status_code == 401:
            raise UpstreamError(
                "Context7 rejected the credential",
                fix=f"python -m tools.jobs credential set {credentials.CONTEXT7_KEY}",
            )
        if resp.status_code == 429:
            raise UpstreamError(
                "Context7 rate-limited the request",
                fix=(
                    "wait, or store a free key to raise the limit: "
                    f"python -m tools.jobs credential set {credentials.CONTEXT7_KEY}"
                ),
            )
        if resp.status_code == 404:
            raise UpstreamError(
                f"Context7 returned 404 for {url}",
                fix=(
                    "the API may have moved: read context7.com/docs/api-guide and set "
                    "[docs] base / resolve_path / docs_path in config/grad.toml"
                ),
            )
        if resp.status_code >= 400:
            raise UpstreamError(
                f"Context7 returned {resp.status_code}: {resp.text[:200]}",
                fix="check the library id and the query",
            )
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001 - the docs endpoint may serve text
            data = {"text": resp.text}
        _store(key, data)
        return data

    def resolve(self, name: str) -> list[dict[str, Any]]:
        """Library name -> candidate Context7 library ids.

        The MCP tool this mirrors is `resolve-library-id`; note that the docs
        tool is `query-docs`, and `get-library-docs` in older material is stale.
        """
        data = self._get(self.resolve_path, {"libraryName": name, "query": name})
        results = data.get("results") if isinstance(data, dict) else data
        out = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "library_id": item.get("id") or item.get("libraryId") or item.get("settings", {}).get("project"),
                    "title": item.get("title") or item.get("name"),
                    "description": item.get("description", ""),
                    "trust_score": item.get("trustScore") or item.get("trust_score"),
                    "snippets": item.get("totalSnippets") or item.get("snippets"),
                    "versions": item.get("versions") or [],
                }
            )
        return [o for o in out if o["library_id"]]

    def docs(self, library_id: str, query: str, *, tokens: int = 5000) -> dict[str, Any]:
        """Documentation for a library, narrowed by a topic query.

        `type=json` matters: without it the endpoint returns markdown prose,
        which is fine for a human and useless as a `--json` payload.
        """
        params = {"libraryId": library_id, "query": query, "tokens": tokens, "type": "json"}
        # A path template is still honoured, so an older `/{library_id}` style
        # endpoint keeps working from config alone.
        if "{library_id}" in self.docs_path:
            path = self.docs_path.format(library_id=library_id.lstrip("/"))
            params.pop("libraryId")
            params["topic"] = params.pop("query")
        else:
            path = self.docs_path
        data = self._get(path, params)

        if isinstance(data, dict) and "text" in data and len(data) == 1:
            return {"library_id": library_id, "query": query, "text": data["text"]}
        # The key differs by API version -- `codeSnippets` on v2, `snippets` on
        # v1 -- and reading only one of them turns a working response into an
        # empty result that looks like "this library has no docs".
        snippets = None
        if isinstance(data, dict):
            snippets = data.get("codeSnippets") or data.get("snippets")
        return {
            "library_id": library_id,
            "query": query,
            "snippets": snippets or [],
            "raw": None if snippets else data,
        }


# ---------------------------------------------------------------------------
# Rerank (credits, not quota) -- Voyage directly, or OpenRouter as a proxy
# ---------------------------------------------------------------------------
#: `[retrieval] rerank_provider`. `auto` is the default and resolves against
#: what is actually stored, so neither key is mandatory and having only one is
#: never a configuration step.
RERANK_PROVIDERS = ("auto", "voyage", "openrouter")


def rerank_provider(cfg: Config) -> str:
    """Which rail stage 2 rides, resolved against the credential store.

    Voyage serve `rerank-2.5` themselves, so routing it through OpenRouter was a
    second account and a second key for the same weights -- and `embed()`
    already requires the Voyage one. So Voyage wins whenever its key is present:
    one credential covers both of retrieval's credit-spending stages.

    OpenRouter is not dropped, because a key that is already stored and already
    billing is a working setup and upgrading should not break it. It is used
    when it is the only key present, and `rerank_provider = "openrouter"` pins
    it for anyone who wants that rail even with a Voyage key available.

    With neither stored this answers `voyage` rather than raising, so the
    caller's own missing-credential error names the key worth storing -- and
    `cmd_search` turns that into a warning and an unreranked pool, which is the
    established behaviour for a stage 2 that cannot run.
    """
    chosen = str(cfg.get("retrieval", "rerank_provider", "auto")).lower()
    if chosen not in RERANK_PROVIDERS:
        raise ConfigError(
            f"unknown rerank provider {chosen!r}",
            fix=f'[retrieval] rerank_provider must be one of: {", ".join(RERANK_PROVIDERS)}',
        )
    if chosen != "auto":
        return chosen
    if credentials.present(credentials.VOYAGE_KEY):
        return "voyage"
    return "openrouter" if credentials.present(credentials.OPENROUTER_KEY) else "voyage"


def rerank_model(cfg: Config, provider: str) -> str:
    """One `rerank_model` setting that means the same thing on either rail.

    OpenRouter namespaces its catalogue (`voyageai/rerank-2.5`); Voyage's own
    API names the model alone (`rerank-2.5`) and rejects the namespaced form.
    Rewriting it here is what keeps switching provider a one-key change instead
    of two, and what lets a config written for OpenRouter keep working when the
    default rail moves under it.

    A model id that already names some *other* provider is left alone in both
    directions: OpenRouter serves rerankers from several vendors, and only the
    bare Voyage form is ambiguous enough to need qualifying.
    """
    model = str(cfg.get("retrieval", "rerank_model")).strip()
    if provider == "voyage":
        return model.split("/", 1)[1] if model.lower().startswith("voyageai/") else model
    return model if "/" in model else f"voyageai/{model}"


def rerank(query: str, documents: Sequence[str], *, cfg: Config, top_n: int) -> list[dict[str, Any]]:
    """`rerank-2.5`, over whichever rail has a key. See `rerank_provider`.

    Hosted on purpose: local reranking competes for the same VRAM as the
    experiments this agent exists to run. It costs credits rather than quota,
    which is why it sits between the two Haiku stages -- the quota-consuming
    stage never sees the 350 candidates that were obviously wrong.

    Both rails book their spend against the same `STAGE_RERANK` ledger line and
    both say in `detail` how the number was arrived at, because they do not
    arrive at it the same way: OpenRouter prices the call and reports it, Voyage
    returns a token count and leaves the arithmetic here.
    """
    if not documents:
        return []
    provider = rerank_provider(cfg)
    model = rerank_model(cfg, provider)
    call = _rerank_voyage if provider == "voyage" else _rerank_openrouter
    rows, billing = call(query, documents, cfg=cfg, top_n=top_n, model=model)
    quota_log.record(
        quota_log.STAGE_RERANK,
        model=model,
        unit="credits",
        credits_usd=billing.pop("credits_usd"),
        detail={
            "documents": len(documents),
            "top_n": top_n,
            # Which rail, in the ledger. Two providers spending against one line
            # is fine; not being able to tell which one spent it is not.
            "provider": provider,
            **billing,
        },
    )
    return rows


def _rerank_voyage(
    query: str, documents: Sequence[str], *, cfg: Config, top_n: int, model: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Voyage's own rerank endpoint, which is not shaped like OpenRouter's.

    Three differences, all of them silent failures rather than errors if missed:
    the result count is `top_k`, the hits come back under `data` rather than
    `results`, and `usage` carries a token count instead of a price.
    """
    key = credentials.get(credentials.VOYAGE_KEY)
    httpx = _httpx()
    try:
        resp = httpx.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "query": query, "documents": list(documents), "top_k": top_n},
            timeout=float(cfg.get("retrieval", "request_timeout_s", 60)),
        )
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(f"rerank request failed: {exc}", fix="retry, or run with --no-rerank") from exc
    if resp.status_code >= 400:
        raise UpstreamError(
            f"rerank returned {resp.status_code}: {resp.text[:200]}",
            fix=(
                "check the Voyage key: python -m tools.jobs credential set voyage_key   "
                '(or set [retrieval] rerank_provider = "openrouter" to bill through OpenRouter)'
            ),
        )
    data = resp.json()
    # Priced from config for the same reason `embed` is: Voyage bills per token
    # and returns a token count but no price, and a `credits_usd` left at its
    # 0.0 default makes stage 2 free to every ceiling that sums that field.
    total_tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
    rate_per_1m = float(cfg.get("retrieval", "rerank_usd_per_1m_tokens", 0.0) or 0.0)
    rows = [
        {"index": r.get("index"), "score": r.get("relevance_score", r.get("score"))}
        for r in data.get("data", [])
    ]
    return rows, {
        "credits_usd": total_tokens / 1_000_000.0 * rate_per_1m,
        "total_tokens": total_tokens,
        "usd_per_1m_tokens": rate_per_1m,
        # Says which of the two it is wherever the number is read: a rate from
        # config priced this, not the provider's own accounting.
        "cost_basis": "configured_rate",
    }


def _rerank_openrouter(
    query: str, documents: Sequence[str], *, cfg: Config, top_n: int, model: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The same model through OpenRouter's dedicated rerank endpoint."""
    key = credentials.get(credentials.OPENROUTER_KEY)
    base = str(cfg.get("retrieval", "openrouter_base"))
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
            fix=(
                "check the OpenRouter key and that the model id is still served, or drop the "
                'key entirely: [retrieval] rerank_provider = "voyage" reranks on the Voyage '
                "credential the local index already uses"
            ),
        )
    data = resp.json()
    usage = data.get("usage", {}) or {}
    # A response that omits `usage.cost` used to book the call at $0.00, which is
    # indistinguishable from a call that was genuinely free. The shape of this
    # response is documented but unverified against the live service, so the
    # absence is recorded rather than rounded to zero: `cost_basis` is what tells
    # a later reader whether the credits ceiling actually saw this spend.
    reported = usage.get("cost")
    try:
        cost = float(reported) if reported is not None else 0.0
    except (TypeError, ValueError):
        reported, cost = None, 0.0
    rows = [
        {"index": r.get("index"), "score": r.get("relevance_score", r.get("score"))}
        for r in data.get("results", [])
    ]
    return rows, {
        "credits_usd": cost,
        "cost_basis": "reported" if reported is not None else "unreported",
    }


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
    # Voyage returns a token count and no price, so the cost is computed from
    # the configured rate. Recording the call with `credits_usd` left at its
    # 0.0 default made every embedding free to `budget.spend`, which sums that
    # field -- so ingesting a corpus spent real dollars no ceiling and no meter
    # ever saw.
    total_tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
    rate_per_1m = float(cfg.get("retrieval", "embed_usd_per_1m_tokens", 0.0) or 0.0)
    quota_log.record(
        quota_log.STAGE_EMBED,
        model=model,
        unit="credits",
        credits_usd=total_tokens / 1_000_000.0 * rate_per_1m,
        detail={
            "texts": len(texts),
            "total_tokens": total_tokens,
            "usd_per_1m_tokens": rate_per_1m,
            # Says which of the two it is wherever the number is read: a rate
            # from config priced this, not the provider's own accounting.
            "cost_basis": "configured_rate",
        },
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
