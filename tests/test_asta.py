"""Asta over streamable HTTP: the transport, and the shapes it may answer in.

The endpoint, the transport and the tool names come from Ai2's published
documentation. **The shape of each tool's result does not** -- that needs a live
call with a real corpus behind it, and this suite has neither. So the tests here
are about what `core/http.py` promises regardless of the shape:

  * one JSON-RPC message is recovered whether the server answers with JSON or
    with an event stream, because streamable HTTP lets it pick either;
  * the handshake happens once and its session id is carried afterwards;
  * every plausible envelope around the hits is read;
  * an envelope that is *not* recognised raises, rather than returning an empty
    list -- a search that quietly finds nothing reads as "the literature has
    nothing on this", and that is the one conclusion a schema change must not be
    able to manufacture;
  * both tier-1 clients answer in one vocabulary, so the funnel cannot tell them
    apart and a paper found by both fuses to one candidate.
"""

from __future__ import annotations

import json

import pytest

from core import config as config_mod, http
from core.errors import UpstreamError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", content_type="application/json",
                 headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or (json.dumps(self._payload) if payload is not None else "")
        self.headers = {"content-type": content_type, **(headers or {})}
        self.content = self.text.encode()

    def json(self):
        return self._payload


def rpc(result) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def tool_result(payload) -> dict:
    """A `tools/call` result the way MCP wraps one: text content blocks."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


@pytest.fixture
def transport(monkeypatch):
    """A fake streamable-HTTP endpoint. `queue` is the reply to each POST in
    order; anything left unqueued replies with an empty handshake result."""
    posts: list[dict] = []
    queue: list[FakeResponse] = []

    class FakeHttpx:
        @staticmethod
        def post(url, json=None, headers=None, timeout=None):
            posts.append({"url": url, "body": json, "headers": headers})
            if queue:
                return queue.pop(0)
            return FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"}))

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    monkeypatch.setattr(http.credentials, "get", lambda name, required=True: None)
    return posts, queue


def client(**_):
    return http.Asta(config_mod.load(reload=True))


def methods(posts) -> list[str]:
    return [p["body"]["method"] for p in posts]


# ---------------------------------------------------------------------------
# the handshake
# ---------------------------------------------------------------------------
def test_the_first_call_initialises_then_notifies_then_calls_the_tool(workspace, transport):
    posts, queue = transport
    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(FakeResponse(payload=rpc(tool_result({"data": []}))))

    client().snippet_search("attention")
    assert methods(posts) == ["initialize", "notifications/initialized", "tools/call"]
    assert posts[-1]["body"]["params"]["name"] == "snippet_search"


def test_the_session_id_is_carried_after_the_handshake_assigns_it(workspace, transport):
    posts, queue = transport
    queue.append(
        FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"}),
                     headers={"mcp-session-id": "sess-42"})
    )
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(FakeResponse(payload=rpc(tool_result({"data": []}))))

    client().snippet_search("attention")
    assert posts[0]["headers"].get("Mcp-Session-Id") is None, "nothing to send yet"
    assert posts[-1]["headers"]["Mcp-Session-Id"] == "sess-42"


def test_a_negotiated_protocol_version_is_the_one_sent_afterwards(workspace, transport):
    """The server picks the version; a client that keeps announcing its own
    preference after being told otherwise is not speaking the protocol."""
    posts, queue = transport
    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2099-01-01"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(FakeResponse(payload=rpc(tool_result({"data": []}))))

    client().snippet_search("attention")
    assert posts[-1]["headers"]["MCP-Protocol-Version"] == "2099-01-01"


def test_a_stored_key_is_sent_and_its_absence_is_not_an_error(workspace, transport, monkeypatch):
    posts, queue = transport
    assert client().authenticated is False

    monkeypatch.setattr(
        http.credentials, "get",
        lambda name, required=True: "k-1" if name == http.credentials.ASTA_KEY else None,
    )
    keyed = client()
    assert keyed.authenticated is True

    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(FakeResponse(payload=rpc(tool_result({"data": []}))))
    keyed.snippet_search("attention")
    assert posts[-1]["headers"]["x-api-key"] == "k-1"


def test_an_unreachable_credential_store_still_allows_an_anonymous_call(workspace, transport):
    """Same reasoning as Context7: an optional credential whose *store* is
    missing must not make an anonymous call impossible."""
    from core.errors import ConfigError

    def explode(name, required=True):
        raise ConfigError("keyring is not installed")

    _, queue = transport
    http.credentials.get = explode  # restored by the fixture's monkeypatch teardown
    assert client().authenticated is False


# ---------------------------------------------------------------------------
# the transport
# ---------------------------------------------------------------------------
def test_an_event_stream_answer_is_read_as_a_json_rpc_message(workspace, transport):
    """The same request may be answered with JSON or with SSE, at the server's
    discretion and without saying which in advance."""
    posts, queue = transport
    body = tool_result({"data": [{"paper": {"paperId": "p1", "title": "Attention"}}]})
    stream = (
        "event: message\n"
        'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n'
        "\n"
        f"event: message\ndata: {json.dumps(rpc(body))}\n\n"
    )
    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(FakeResponse(text=stream, content_type="text/event-stream"))

    rows = client().snippet_search("attention")
    assert [r["title"] for r in rows] == ["Attention"]


def test_a_stream_carrying_only_notifications_is_an_error(workspace, transport):
    _, queue = transport
    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(
        FakeResponse(text='data: {"jsonrpc":"2.0","method":"x","params":{}}\n\n',
                     content_type="text/event-stream")
    )
    with pytest.raises(UpstreamError, match="no result"):
        client().snippet_search("attention")


def test_a_json_rpc_error_names_the_method_that_was_refused(workspace, transport):
    _, queue = transport
    queue.append(
        FakeResponse(payload={"jsonrpc": "2.0", "id": 1,
                              "error": {"code": -32601, "message": "no such tool"}})
    )
    with pytest.raises(UpstreamError) as exc:
        client().snippet_search("attention")
    assert "initialize" in str(exc.value)
    assert "no such tool" in str(exc.value)


def test_a_tool_that_reports_its_own_failure_is_not_an_empty_result(workspace, transport):
    _, queue = transport
    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(
        FakeResponse(payload=rpc({"content": [{"type": "text", "text": "limit must be > 0"}],
                                  "isError": True}))
    )
    with pytest.raises(UpstreamError, match="limit must be"):
        client().snippet_search("attention", limit=0)


def test_rate_limiting_points_at_the_key_that_can_actually_be_obtained(workspace, transport):
    """The S2 advice this replaced -- "store an API key, it is free" -- has no
    ending for a personal account. Asta's key comes from a form."""
    _, queue = transport
    queue.append(FakeResponse(status_code=429, text="slow down"))
    with pytest.raises(UpstreamError) as exc:
        client().snippet_search("attention")
    assert http.credentials.ASTA_KEY in (exc.value.fix or "")


def test_a_moved_endpoint_names_the_config_key_that_moves_with_it(workspace, transport):
    _, queue = transport
    queue.append(FakeResponse(status_code=404, text="gone"))
    with pytest.raises(UpstreamError) as exc:
        client().snippet_search("attention")
    assert "asta_base" in (exc.value.fix or "")


# ---------------------------------------------------------------------------
# the shapes -- the part that is not verified against the live service
# ---------------------------------------------------------------------------
def call_with(workspace, transport, payload, *, structured=False):
    _, queue = transport
    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    result = (
        {"structuredContent": payload, "content": []} if structured else tool_result(payload)
    )
    queue.append(FakeResponse(payload=rpc(result)))
    return client().snippet_search("attention")


NESTED = {"data": [{"snippet": {"text": "a 500-word excerpt", "snippetKind": "body"},
                    "paper": {"corpusId": 991, "paperId": "p1", "title": "Attention",
                              "publicationDate": "2017-06-12",
                              "externalIds": {"ArXiv": "1706.03762"}}}]}
FLAT = {"results": [{"corpusId": 991, "paperId": "p1", "title": "Attention", "year": 2017,
                     "text": "a 500-word excerpt", "externalIds": {"ArXiv": "1706.03762"}}]}


@pytest.mark.parametrize("payload", [NESTED, FLAT], ids=["nested", "flat"])
def test_both_plausible_result_shapes_produce_the_same_candidate(workspace, transport, payload):
    row = call_with(workspace, transport, payload)[0]
    # The SHA, not the corpus id: `IDENTITY_KEYS` leads with `paperId` because
    # it is the field every endpoint returns. Both fixtures carry both.
    assert row["id"] == "s2:p1"
    assert row["title"] == "Attention"
    assert row["year"] in (2017, "2017")
    assert row["snippet"] == "a 500-word excerpt"
    assert row["external"]["ArXiv"] == "1706.03762"


def test_the_typed_channel_is_preferred_over_the_text_one(workspace, transport):
    rows = call_with(workspace, transport, NESTED, structured=True)
    assert rows[0]["title"] == "Attention"


def test_a_bare_list_is_a_result_set_too(workspace, transport):
    rows = call_with(workspace, transport, [{"paperId": "p1", "title": "Attention"}])
    assert rows[0]["title"] == "Attention"


def test_an_unrecognised_envelope_raises_rather_than_returning_nothing(workspace, transport):
    """The failure this exists to prevent: `ok: true` with no results reads as
    "the literature has nothing on this"."""
    with pytest.raises(UpstreamError) as exc:
        call_with(workspace, transport, {"unexpected": {"shape": 1}})
    assert "snippet_search" in str(exc.value)
    assert "_rows" in (exc.value.fix or "")


def test_prose_where_a_result_set_was_expected_says_what_arrived(workspace, transport):
    with pytest.raises(UpstreamError, match="not a result set"):
        call_with(workspace, transport, "I could not find anything about that.")


def test_one_paper_gets_one_id_down_every_path_that_feeds_the_pool(workspace, transport, monkeypatch):
    """`cmd_search` calls snippet search *and* paper search for every expanded
    query, and fuses the results by id. `/snippet/search` returns `corpusId` and
    `paperId` while `/paper/search` is only asked for the latter -- so reading
    them in different orders gave the same paper two ids, and it ranked twice
    and took a slot from something else. This is the ordinary path, not a corner
    of it, which is why the order lives in one constant."""
    paper = {"paperId": "sha-1", "corpusId": 991, "title": "Attention",
             "abstract": "an abstract", "externalIds": {"ArXiv": "1706.03762"}}

    asta_id = call_with(workspace, transport, {"data": [dict(paper)]})[0]["id"]

    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            if "snippet" in url:
                return FakeResponse(payload={"data": [
                    {"snippet": {"text": "an excerpt"}, "paper": dict(paper)},
                ]})
            # What `/paper/search` actually returns: the SHA, no corpus id.
            flat = {k: v for k, v in paper.items() if k != "corpusId"}
            return FakeResponse(payload={"data": [flat]})

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    s2 = http.SemanticScholar(config_mod.load(reload=True))
    snippet_id = s2.snippet_search("attention")[0]["id"]
    paper_id = s2.paper_search("attention")[0]["id"]

    assert snippet_id == paper_id == asta_id == "s2:sha-1"


def test_the_id_and_the_citation_seed_name_the_same_paper(workspace, transport):
    """`paper_id` is what `neighbours` expands from. If it were read by a
    different rule than the id, a candidate and its citation expansion could
    disagree about which paper they were."""
    row = call_with(workspace, transport, {"data": [
        {"paperId": "sha-1", "corpusId": 991, "title": "Attention"},
    ]})[0]
    assert row["id"] == f"s2:{row['paper_id']}"


def test_a_paper_with_only_a_corpus_id_still_gets_one(workspace, transport):
    row = call_with(workspace, transport, {"data": [
        {"corpusId": 991, "title": "Attention"},
    ]})[0]
    assert row["id"] == "s2:991"


def test_hits_with_no_id_stay_distinct_instead_of_fusing_into_one(workspace, transport):
    """`corpus.rrf` fuses by id, so a shared placeholder is not a cosmetic
    problem: every id-less hit -- from either client, across every query in the
    run -- collapsed into one phantom candidate and the rest vanished."""
    rows = call_with(workspace, transport, {"data": [
        {"title": "One paper", "text": "the first excerpt"},
        {"title": "Another paper", "text": "the second excerpt"},
    ]})
    assert len({row["id"] for row in rows}) == 2
    assert not any(row["id"].endswith("None") for row in rows)


def test_the_same_id_less_paper_from_both_clients_still_fuses(workspace, transport, monkeypatch):
    """The other half: the fallback has to be *stable*, or the deduplication the
    shared namespace exists for stops working for exactly these hits."""
    asta_row = call_with(workspace, transport, {"data": [
        {"title": "Attention Is All You Need", "text": "an excerpt"},
    ]})[0]

    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            return FakeResponse(payload={"data": [
                {"snippet": {"text": "an excerpt"},
                 "paper": {"title": "Attention Is All You Need"}},
            ]})

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    s2_row = http.SemanticScholar(config_mod.load(reload=True)).snippet_search("attention")[0]
    assert asta_row["id"] == s2_row["id"]


def test_a_hit_with_nothing_to_rank_is_dropped(workspace, transport):
    """No id, no title, no text: nothing for the reranker to read and nothing to
    cite. Dropping it costs no recall; keeping it under a placeholder did."""
    rows = call_with(workspace, transport, {"data": [
        {"year": 2017},
        {"paperId": "p1", "title": "A real one"},
    ]})
    assert [row["title"] for row in rows] == ["A real one"]


def test_a_failed_handshake_can_be_retried(workspace, transport):
    """The guard used to be `if self._id:`, and `_call` increments that counter
    *before* it sends -- so an `initialize` that failed left the counter
    non-zero and every later request skipped straight to `tools/call` on a
    connection that was never initialised. One 429 poisoned the client."""
    posts, queue = transport
    client_ = client()

    queue.append(FakeResponse(status_code=429, text="slow down"))
    with pytest.raises(UpstreamError):
        client_.snippet_search("attention")
    assert client_._ready is False  # noqa: SLF001

    queue.append(FakeResponse(payload=rpc({"protocolVersion": "2025-06-18"})))
    queue.append(FakeResponse(status_code=202, text=""))
    queue.append(FakeResponse(payload=rpc(tool_result({"data": [
        {"paperId": "p1", "title": "Attention"},
    ]}))))
    rows = client_.snippet_search("attention")
    assert [row["title"] for row in rows] == ["Attention"]
    assert methods(posts)[-3:] == ["initialize", "notifications/initialized", "tools/call"]


def test_the_backward_citation_direction_is_refused_rather_than_faked(workspace, transport):
    """Asta publishes `get_citations` and no references counterpart. Answering
    the backward direction with the forward one would double-count it into RRF
    under a second name."""
    posts, queue = transport
    assert client().neighbours("p1", direction="references") == []
    assert posts == [], "no request should have been made at all"


# ---------------------------------------------------------------------------
# one vocabulary
# ---------------------------------------------------------------------------
def test_both_tier_one_clients_answer_in_the_same_shape(workspace, transport, monkeypatch):
    """`paper_search.py` does not know which tier a candidate came from and must
    not have to: RRF fuses by id, and the reranker reads title and snippet. So
    the two clients are asked for the same paper and their answers compared."""
    asta_row = call_with(workspace, transport, NESTED)[0]

    # The same hit as S2's REST API returns it.
    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            return FakeResponse(payload=NESTED)

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    s2_row = http.SemanticScholar(config_mod.load(reload=True)).snippet_search("attention")[0]

    shared = {"id", "paper_id", "title", "year", "snippet", "section", "source", "external"}
    assert shared <= set(asta_row)
    assert shared <= set(s2_row)
    # The same corpus and the same corpus ids, so a paper found through both
    # tiers fuses to one candidate instead of ranking twice under two names.
    assert asta_row["id"] == s2_row["id"] == "s2:p1"
    assert asta_row["title"] == s2_row["title"]
    assert asta_row["snippet"] == s2_row["snippet"]
    # `source` is the one field that must differ: a trace has to say who spoke.
    assert asta_row["source"] != s2_row["source"]


def test_the_tier_one_selector_builds_the_clients_the_config_names(workspace):
    from tools import paper_search

    cfg = config_mod.load(reload=True)
    assert [n for n, _ in paper_search.tier1_clients(cfg, "asta")] == ["asta"]
    assert [n for n, _ in paper_search.tier1_clients(cfg, "s2")] == ["s2"]
    assert [n for n, _ in paper_search.tier1_clients(cfg, "both")] == ["asta", "s2"]
    assert paper_search.tier1_clients(cfg, "none") == []
    # The default, when nothing overrides it.
    assert [n for n, _ in paper_search.tier1_clients(cfg)] == ["asta"]


def test_an_unknown_tier_one_source_lists_the_real_ones(workspace):
    from core.errors import UsageError
    from tools import paper_search

    with pytest.raises(UsageError) as exc:
        paper_search.tier1_clients(config_mod.load(reload=True), "scholar")
    assert "asta" in (exc.value.fix or "")
