"""Stage 2 on either rail (HANDOFF §5).

The reranker used to be reachable only through OpenRouter, which meant a second
account and a second key for weights the Voyage credential already reached --
`embed()` needs that key for the local index regardless. So Voyage is the
default now and OpenRouter is kept as an alternative rather than dropped.

Two providers behind one function is exactly the shape that rots quietly: the
request bodies differ (`top_k` against `top_n`), the responses differ (`data`
against `results`), and the billing differs (a token count against a reported
price). None of those is an error if it is wrong -- a `top_n` sent to Voyage is
ignored and reranks the whole pool, and reading `results` from Voyage's response
returns zero hits, which `cmd_search` treats as "reranker unavailable" and
degrades past. So each half is pinned to the contract it was written against.
"""

from __future__ import annotations

import pytest

from core import http, quota_log
from core.errors import ConfigError

from tests.test_models import write_config


class _Response:
    """One reranker reply, in whichever provider's shape."""

    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    def post(self, url, *, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        return _Response(self._payload)


VOYAGE_REPLY = {
    "object": "list",
    "data": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.22}],
    "model": "rerank-2.5",
    "usage": {"total_tokens": 1_000_000},
}

OPENROUTER_REPLY = {
    "results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.22}],
    "usage": {"cost": 0.004},
}


@pytest.fixture
def stored(monkeypatch):
    """Which credentials the store holds, per test."""

    def _stored(*names: str):
        held = set(names)
        monkeypatch.setattr(http.credentials, "present", lambda n: n in held)
        monkeypatch.setattr(http.credentials, "get", lambda n, **_k: f"{n}-value")

    return _stored


def _client(monkeypatch, payload: dict) -> _Client:
    client = _Client(payload)
    monkeypatch.setattr(http, "_httpx", lambda: client)
    return client


# ---------------------------------------------------------------------------
# the point of the change: a Voyage key alone is a working reranker
# ---------------------------------------------------------------------------
def test_a_voyage_key_alone_reranks(workspace, monkeypatch, stored):
    cfg = write_config(workspace, "")
    stored(http.credentials.VOYAGE_KEY)
    client = _client(monkeypatch, VOYAGE_REPLY)

    rows = http.rerank("efficient optimizers", ["a", "b"], cfg=cfg, top_n=2)

    assert len(client.calls) == 1, "no OpenRouter hop"
    call = client.calls[0]
    assert call["url"] == "https://api.voyageai.com/v1/rerank"
    assert call["headers"]["Authorization"] == "Bearer voyage_key-value"
    # Voyage's own API names the model alone and rejects the namespaced form,
    # and calls the count `top_k`. A `top_n` here is ignored, not refused, so
    # the whole pool comes back and nothing looks broken.
    assert call["json"]["model"] == "rerank-2.5"
    assert call["json"]["top_k"] == 2
    assert "top_n" not in call["json"]
    # Read out of `data`, which is where Voyage puts them; `results` is empty
    # here and would degrade to an unreranked pool without saying why.
    assert rows == [{"index": 1, "score": 0.91}, {"index": 0, "score": 0.22}]


def test_the_voyage_rail_reaches_the_credits_ceiling(workspace, monkeypatch, stored):
    """Voyage returns a token count and no price, so the cost is computed from
    the configured rate -- the same trap `embed` fell into, one stage earlier: a
    `credits_usd` of 0.0 is free to every ceiling that sums that field."""
    cfg = write_config(workspace, "")
    stored(http.credentials.VOYAGE_KEY)
    _client(monkeypatch, VOYAGE_REPLY)

    http.rerank("q", ["a", "b"], cfg=cfg, top_n=2)

    row = next(r for r in quota_log.entries() if r["stage"] == quota_log.STAGE_RERANK)
    assert row["unit"] == "credits"
    assert row["credits_usd"] == pytest.approx(0.05), "a million tokens at the list rate"
    assert row["detail"]["provider"] == "voyage"
    assert row["detail"]["cost_basis"] == "configured_rate"


# ---------------------------------------------------------------------------
# and the rail it replaced still works
# ---------------------------------------------------------------------------
def test_an_openrouter_key_alone_still_reranks(workspace, monkeypatch, stored):
    """The whole reason this is a default and not a removal: a key that is
    already stored and already billing is a working setup, and upgrading should
    not break it."""
    cfg = write_config(workspace, "")
    stored(http.credentials.OPENROUTER_KEY)
    client = _client(monkeypatch, OPENROUTER_REPLY)

    rows = http.rerank("q", ["a", "b"], cfg=cfg, top_n=2)

    call = client.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/rerank"
    assert call["headers"]["Authorization"] == "Bearer openrouter_key-value"
    assert call["json"]["model"] == "voyageai/rerank-2.5", "namespaced for the catalogue"
    assert call["json"]["top_n"] == 2
    assert rows == [{"index": 1, "score": 0.91}, {"index": 0, "score": 0.22}]

    row = next(r for r in quota_log.entries() if r["stage"] == quota_log.STAGE_RERANK)
    assert row["detail"]["provider"] == "openrouter"
    assert row["credits_usd"] == pytest.approx(0.004), "priced by the provider, not by us"
    assert row["detail"]["cost_basis"] == "reported"


def test_voyage_wins_when_both_keys_are_stored(workspace, monkeypatch, stored):
    """One credential covers both credit-spending stages, so the direct rail is
    preferred over the proxy for the same weights."""
    cfg = write_config(workspace, "")
    stored(http.credentials.VOYAGE_KEY, http.credentials.OPENROUTER_KEY)
    client = _client(monkeypatch, VOYAGE_REPLY)

    http.rerank("q", ["a"], cfg=cfg, top_n=1)
    assert client.calls[0]["url"].startswith("https://api.voyageai.com")


def test_the_provider_can_be_pinned_against_the_credentials(workspace, monkeypatch, stored):
    """`auto` reads the store; a name overrides it. Anyone who wants their spend
    on one bill gets to say so."""
    cfg = write_config(workspace, '[retrieval]\nrerank_provider = "openrouter"\n')
    stored(http.credentials.VOYAGE_KEY, http.credentials.OPENROUTER_KEY)
    client = _client(monkeypatch, OPENROUTER_REPLY)

    http.rerank("q", ["a"], cfg=cfg, top_n=1)
    assert client.calls[0]["url"].startswith("https://openrouter.ai")


def test_with_neither_key_the_error_names_voyage(workspace, monkeypatch, stored):
    """A funnel run with no reranker credential should be told which one to
    store -- and it is the one the local index needs anyway. `cmd_search`
    catches this and returns an unreranked pool, so it is a warning rather than
    a failed run."""
    cfg = write_config(workspace, "")
    stored()
    monkeypatch.setattr(
        http.credentials,
        "get",
        lambda n, **_k: (_ for _ in ()).throw(ConfigError(f"credential {n!r} is not in the store")),
    )

    with pytest.raises(ConfigError) as exc:
        http.rerank("q", ["a"], cfg=cfg, top_n=1)
    assert "voyage_key" in str(exc.value)


# ---------------------------------------------------------------------------
# one model setting, two catalogues
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("configured", "voyage", "openrouter"),
    [
        # The shipped default, and what an existing OpenRouter config holds.
        ("voyageai/rerank-2.5", "rerank-2.5", "voyageai/rerank-2.5"),
        # Written the way Voyage's own docs name it.
        ("rerank-2.5-lite", "rerank-2.5-lite", "voyageai/rerank-2.5-lite"),
        # Some other vendor's reranker on OpenRouter: already namespaced, and
        # prefixing it again would ask the catalogue for a model nobody serves.
        ("cohere/rerank-v3.5", "cohere/rerank-v3.5", "cohere/rerank-v3.5"),
    ],
)
def test_the_model_id_suits_whichever_rail_is_chosen(workspace, configured, voyage, openrouter):
    cfg = write_config(workspace, f'[retrieval]\nrerank_model = "{configured}"\n')
    assert http.rerank_model(cfg, "voyage") == voyage
    assert http.rerank_model(cfg, "openrouter") == openrouter


def test_an_unknown_provider_is_refused(workspace):
    """A typo that silently fell through to a default would bill the wrong
    account, which is the one outcome nobody would notice."""
    cfg = write_config(workspace, '[retrieval]\nrerank_provider = "voyagai"\n')
    with pytest.raises(ConfigError) as exc:
        http.rerank_provider(cfg)
    assert "voyagai" in str(exc.value)


def test_an_empty_pool_costs_nothing(workspace, monkeypatch, stored):
    """Guarded before the provider is resolved, so it does not read the
    credential store to decide not to make a call."""
    cfg = write_config(workspace, "")
    stored(http.credentials.VOYAGE_KEY)
    assert http.rerank("q", [], cfg=cfg, top_n=5) == []
    assert not [r for r in quota_log.entries() if r["stage"] == quota_log.STAGE_RERANK]
