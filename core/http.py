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
# Asta -- the same corpus, through a door that opens
# ---------------------------------------------------------------------------
#: The MCP protocol version this client speaks. Sent on `initialize` and echoed
#: on every request after it; a server that wants another version says so in its
#: `initialize` result and this follows it.
MCP_PROTOCOL_VERSION = "2025-06-18"


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
        self.ttl = float(cfg.get("retrieval", "cache_ttl_s", 604800))
        self.interval = float(cfg.get("retrieval", "min_request_interval_s", 1.1))
        try:
            self.key = credentials.get(credentials.ASTA_KEY, required=False)
        except ConfigError:
            # Same reasoning as Context7: an optional credential whose *store* is
            # unreachable must not make an anonymous call impossible.
            self.key = None
        self._session: str | None = None
        self._protocol = MCP_PROTOCOL_VERSION
        self._id = 0

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
        _throttle("asta", self.interval)
        httpx = _httpx()
        try:
            resp = httpx.post(self.base, json=body, headers=self._headers(), timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Asta request failed: {exc}",
                fix="retry, or run with --local-only to search papers already ingested",
            ) from exc

        # Assigned on `initialize` and echoed from then on. A server that does
        # not use sessions simply never sends it.
        session = resp.headers.get("mcp-session-id")
        if session:
            self._session = session

        if resp.status_code == 401 or resp.status_code == 403:
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
        if resp.status_code >= 400:
            raise UpstreamError(
                f"Asta returned {resp.status_code}: {resp.text[:200]}",
                fix=(
                    "the endpoint may have moved: check allenai.org/asta/resources/mcp and "
                    "set [retrieval] asta_base in config/grad.toml"
                ),
            )
        # A notification gets 202 Accepted and an empty body; there is nothing
        # to parse and nothing to wait for.
        if resp.status_code == 202 or not (resp.content or b"").strip():
            return None
        return _mcp_payload(resp)

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

        Once per client. The session id and the negotiated protocol version both
        come out of this, and every request after it carries them.
        """
        if self._session is not None or self._id:
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
        data = self.tool("snippet_search", {"query": query, "limit": limit})
        return [_row(item, source="asta.snippet") for item in _rows(data, "snippet_search")]

    def paper_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        data = self.tool("search_papers_by_relevance", {"query": query, "limit": limit})
        return [_row(item, source="asta.paper") for item in _rows(data, "search_papers_by_relevance")]

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
        data = self.tool("get_citations", {"paper_id": paper_id, "limit": limit})
        return [_row(item, source="asta.citations") for item in _rows(data, "get_citations")]


def _mcp_payload(resp: Any) -> Any:
    """One JSON-RPC message out of a streamable-HTTP response.

    The same request may be answered with `application/json` or with an SSE
    stream, at the server's discretion, so both are handled. For a stream the
    *last* `data:` frame carrying a result is taken: progress notifications
    share the channel with the answer.
    """
    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/event-stream" not in content_type:
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                f"Asta returned a body that is not JSON: {resp.text[:200]}",
                fix="retry; if it persists the endpoint may no longer speak streamable HTTP",
            ) from exc

    answer: Any = None
    for line in resp.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            frame = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(frame, dict) and ("result" in frame or "error" in frame):
            answer = frame
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


def _rows(data: Any, tool_name: str) -> list[dict[str, Any]]:
    """The list of hits inside a tool's payload, whatever it is wrapped in.

    Unverified against the live service, so this reads the S2 REST shape and the
    obvious flattenings of it. An unrecognised payload raises: a search that
    quietly returns nothing reads as "the literature has nothing on this", and
    that is a conclusion nobody should draw from a schema change.
    """
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ("data", "results", "snippets", "papers", "citations", "items"):
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


def _row(item: dict[str, Any], *, source: str) -> dict[str, Any]:
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

    identifier = (
        paper.get("corpusId")
        or paper.get("corpus_id")
        or paper.get("paperId")
        or paper.get("paper_id")
        or paper.get("id")
    )
    text = snippet.get("text") or item.get("text") or ""
    year = paper.get("year")
    if year is None:
        year = (str(paper.get("publicationDate") or "")[:4]) or None
    external = paper.get("externalIds") or paper.get("external_ids") or {}
    return {
        "id": f"s2:{identifier}",
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
        # includes the case where the credential *store* is unreachable: on a
        # machine with no keyring installed, `credentials.get` raises even for
        # an optional credential, and letting that propagate would make an
        # anonymous lookup impossible for want of a key it does not need.
        try:
            self.key = credentials.get(credentials.CONTEXT7_KEY, required=False)
        except ConfigError:
            self.key = None

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
