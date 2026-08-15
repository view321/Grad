"""Papers with Code, and the arXiv batch that makes its rows readable.

This replaced Asta as the funnel's default tier 1 for one reason, and it is a
measured one rather than a preference: Asta answers a search in ~121 seconds and
takes ~283 seconds to report that its own backend refused a connection, and
stage 0 multiplies that by six queries and two endpoints. Every caller gives up
first. This answers in one to two seconds.

What it costs is the thing to keep honest, and most of what is asserted below is
about that: search rows carry **no abstract**, so `arxiv_abstracts` fills the
pool in one request, and `related` is a *dense neighbour* rather than a citation
edge and must not be presented as one.

No network: the transport is faked, the same way `tests/test_asta.py` fakes it.
The shapes here are not guesses -- they are what the live API returned.
"""

from __future__ import annotations

import json

import pytest

from core import config as config_mod, http
from core.errors import UpstreamError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


#: One row exactly as `papers/search` returned it live. Note what is *not* here.
LIVE_ROW = {
    "id": "93318",
    "title": "Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation",
    "arxiv_id": "2502.07752",
    "source": "arxiv",
    "authors": ["Wenbo Gong", "Meyer Scetbon"],
    "published": "2025-02-11",
    "url_abs": "https://arxiv.org/abs/2502.07752",
    "citation_count": 6,
}


@pytest.fixture
def transport(monkeypatch):
    """A fake catalogue. `queue` answers each GET in order."""
    gets: list[dict] = []
    queue: list[FakeResponse] = []

    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            gets.append({"url": url, "params": params or {}, "headers": headers})
            return queue.pop(0) if queue else FakeResponse(payload={"results": []})

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    return gets, queue


def client():
    return http.PapersWithCode(config_mod.load(reload=True))


# ---------------------------------------------------------------------------
# the two rankings
# ---------------------------------------------------------------------------
def test_the_two_verbs_are_two_genuinely_different_rankings(workspace, transport):
    """Lexical and dense, which is the pair `corpus.rrf` exists to fuse -- so the
    two calls the funnel already makes per query map onto them without the
    funnel knowing anything changed."""
    gets, queue = transport
    queue.append(FakeResponse(payload={"results": [LIVE_ROW]}))
    queue.append(FakeResponse(payload={"results": [LIVE_ROW]}))

    client().snippet_search("efficient optimizers")
    client().paper_search("efficient optimizers")
    assert gets[0]["params"]["mode"] == "semantic"
    assert gets[1]["params"]["mode"] == "keyword"


def test_a_row_arrives_in_the_vocabulary_the_funnel_already_fuses(workspace, transport):
    _, queue = transport
    queue.append(FakeResponse(payload={"results": [LIVE_ROW]}))
    row = client().paper_search("efficient optimizers")[0]

    shared = {"id", "paper_id", "title", "year", "snippet", "abstract", "source", "external"}
    assert shared <= set(row)
    assert row["id"] == "pwc:2502.07752"
    assert row["paper_id"] == "2502.07752"
    assert row["year"] == "2025"
    assert row["external"]["ArXiv"] == "2502.07752"
    # The honest part: the catalogue's search does not return text.
    assert row["snippet"] == ""
    assert row["abstract"] == ""


def test_this_catalogue_does_not_share_the_semantic_scholar_namespace(workspace, transport):
    """Asta and S2 share `s2:` because they are one index with one set of ids.
    This is a different catalogue with its own numbering, and giving it the same
    prefix would fuse two unrelated papers whose ids happened to collide."""
    _, queue = transport
    queue.append(FakeResponse(payload={"results": [{"id": "991", "title": "Attention"}]}))
    assert client().paper_search("attention")[0]["id"] == "pwc:991"


def test_a_page_larger_than_the_api_allows_is_clamped(workspace, transport):
    gets, queue = transport
    queue.append(FakeResponse(payload={"results": []}))
    client().paper_search("attention", limit=500)
    assert gets[0]["params"]["page_size"] == http.PapersWithCode.MAX_PAGE


# ---------------------------------------------------------------------------
# expansion, and what it is not
# ---------------------------------------------------------------------------
def test_related_work_is_reported_as_a_dense_neighbour_not_a_citation(workspace, transport):
    """The rows carry `provenance: "dense"` and a similarity score: they are
    nearest neighbours in an embedding space, not papers that cite the seed.
    A ledger entry's basis is the thing that must not be overstated."""
    _, queue = transport
    queue.append(FakeResponse(payload=[{**LIVE_ROW, "provenance": "dense", "similarity": 0.86}]))
    row = client().neighbours("1706.03762")[0]
    assert row["source"] == "pwc.related"
    assert "citation" not in row["source"]


def test_the_backward_direction_is_refused_rather_than_faked(workspace, transport):
    gets, _ = transport
    assert client().neighbours("1706.03762", direction="references") == []
    assert gets == [], "no request should have been made at all"


def test_related_answers_with_a_bare_list_and_search_with_an_envelope(workspace, transport):
    """Both are what the live API returns, and both have to be read."""
    _, queue = transport
    queue.append(FakeResponse(payload=[LIVE_ROW]))
    assert len(client().neighbours("1706.03762")) == 1


def test_an_unrecognised_shape_raises_rather_than_returning_nothing(workspace, transport):
    """`ok: true` with no results reads as "the literature has nothing on this",
    which is the one conclusion a schema change must not manufacture."""
    _, queue = transport
    queue.append(FakeResponse(payload={"papers_maybe": []}))
    with pytest.raises(UpstreamError) as exc:
        client().paper_search("attention")
    assert "pwc-cli" in (exc.value.fix or "")


def test_rate_limiting_says_there_is_no_key_to_add(workspace, transport):
    _, queue = transport
    queue.append(FakeResponse(status_code=429, text="slow down"))
    with pytest.raises(UpstreamError) as exc:
        client().paper_search("attention")
    assert "anonymous" in (exc.value.fix or "")


def test_a_moved_endpoint_names_the_config_key_that_moves_with_it(workspace, transport):
    _, queue = transport
    queue.append(FakeResponse(status_code=404, text="gone"))
    with pytest.raises(UpstreamError) as exc:
        client().paper_search("attention")
    assert "pwc_base" in (exc.value.fix or "")


# ---------------------------------------------------------------------------
# the abstracts
# ---------------------------------------------------------------------------
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2211.09760v1</id>
    <title>VeLO</title>
    <summary>While deep learning models have replaced
    hand-designed features, these models are still trained.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models.</summary>
  </entry>
</feed>"""


def test_a_hundred_abstracts_cost_one_request(workspace, transport):
    """The whole reason to use `id_list`: fetching them one at a time would cost
    a request per candidate and undo the reason the corpus was changed."""
    gets, queue = transport
    queue.append(FakeResponse(text=ATOM))
    out = http.arxiv_abstracts(
        ["2211.09760", "1706.03762"], cfg=config_mod.load(reload=True)
    )
    assert len(gets) == 1
    assert gets[0]["params"]["id_list"] == "2211.09760,1706.03762"
    assert out["2211.09760"].startswith("While deep learning models")
    # Flattened: the feed wraps them, and a reranker reading newlines as
    # structure would be reading the feed's formatting.
    assert "\n" not in out["2211.09760"]


def test_the_version_suffix_is_stripped_so_the_lookup_matches(workspace, transport):
    """The feed answers `.../abs/1706.03762v7` and the caller asked for
    `1706.03762`, which is what its candidates are keyed by."""
    _, queue = transport
    queue.append(FakeResponse(text=ATOM))
    out = http.arxiv_abstracts(["1706.03762"], cfg=config_mod.load(reload=True))
    assert "1706.03762" in out


def test_a_failed_fetch_degrades_the_ranking_rather_than_the_run(workspace, transport):
    """A candidate with no abstract ranks on its title, which is what would have
    happened without this. What it must not do is raise."""
    _, queue = transport
    queue.append(FakeResponse(status_code=503, text="down"))
    assert http.arxiv_abstracts(["1706.03762"], cfg=config_mod.load(reload=True)) == {}


def test_nothing_to_fetch_is_not_a_request(workspace, transport):
    gets, _ = transport
    assert http.arxiv_abstracts([], cfg=config_mod.load(reload=True)) == {}
    assert http.arxiv_abstracts(["", None], cfg=config_mod.load(reload=True)) == {}
    assert gets == []


# ---------------------------------------------------------------------------
# the funnel, over the whole of it
# ---------------------------------------------------------------------------
def test_the_pool_is_enriched_before_it_is_reranked(workspace, capsys, monkeypatch):
    """Stage 2 reranks on `title + snippet-or-abstract` and stage 3 triages on
    the same, so a pool of bare titles is a measurably worse funnel -- and a
    funnel silently reranking titles looks exactly like one reranking
    abstracts."""
    from tools import paper_search

    class Catalogue:
        def snippet_search(self, query, limit=20):
            return []

        def paper_search(self, query, limit=20):
            return [{
                "id": "pwc:2502.07752", "paper_id": "2502.07752", "title": "A paper",
                "year": "2025", "snippet": "", "abstract": "", "section": "",
                "source": "pwc.keyword", "external": {"ArXiv": "2502.07752"},
            }]

        def neighbours(self, *_a, **_k):
            return []

    monkeypatch.setattr(http, "PapersWithCode", lambda cfg: Catalogue())
    monkeypatch.setattr(
        http, "arxiv_abstracts", lambda ids, **_k: {"2502.07752": "the real abstract"}
    )
    paper_search.cli.run(
        ["search", "optimizers", "--no-expand", "--no-rerank", "--no-triage",
         "--no-local", "--no-citations", "--full", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["results"][0]["text"] == "the real abstract"
    assert payload["data"]["trace"]["stages"]["1_abstracts"] == {"wanted": 1, "found": 1}


def test_candidates_left_without_an_abstract_are_reported(workspace, capsys, monkeypatch):
    from tools import paper_search

    class Catalogue:
        def snippet_search(self, query, limit=20):
            return []

        def paper_search(self, query, limit=20):
            return [{
                "id": "pwc:1", "paper_id": "9999.99999", "title": "A paper", "year": "2025",
                "snippet": "", "abstract": "", "section": "", "source": "pwc.keyword",
                "external": {"ArXiv": "9999.99999"},
            }]

        def neighbours(self, *_a, **_k):
            return []

    monkeypatch.setattr(http, "PapersWithCode", lambda cfg: Catalogue())
    monkeypatch.setattr(http, "arxiv_abstracts", lambda ids, **_k: {})
    paper_search.cli.run(
        ["search", "optimizers", "--no-expand", "--no-rerank", "--no-triage",
         "--no-local", "--no-citations", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert any(
        "title alone" in w for w in payload["data"]["trace"]["warnings"]
    ), "a funnel reranking titles must not look like one reranking abstracts"
