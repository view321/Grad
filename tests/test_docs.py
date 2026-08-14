"""Library currency (HANDOFF-2 §18).

§24 says item 4 "needs a faked HTTP layer", and the introspection half needs no
faking at all -- which is the point of the ordering rule these tests encode:

    "Order matters: introspect first. A checker relying on Context7 alone will
     confidently describe an API version that is not installed."

So the introspection tests run against this interpreter's real packages, and
only the Context7 half is faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import http
from core.errors import GradError, UpstreamError
from tools import docs


def write(workspace, source: str, name: str = "sample.py") -> Path:
    path = workspace / name
    path.write_text(source, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# oracle 1: introspection
# ---------------------------------------------------------------------------
def test_a_missing_attribute_is_found(workspace):
    """The §17 case in miniature: this is how `namespace` was found in ten
    seconds."""
    path = write(workspace, "import json\n\njson.no_such_function(1)\n")
    report = docs.analyse(path)
    kinds = [f["kind"] for f in report["findings"]]
    assert "missing_attribute" in kinds


def test_a_close_match_is_suggested(workspace):
    path = write(workspace, "import json\n\njson.dumpz({})\n")
    finding = docs.analyse(path)["findings"][0]
    assert "dumps" in finding["fix"]


def test_an_unknown_keyword_argument_is_found(workspace):
    """What would have caught `run_job(..., namespace=...)` on a
    huggingface_hub too old to take it."""
    # `ast.parse` takes no **kwargs, so an unknown keyword is genuinely a
    # TypeError waiting to happen. (`json.dumps` absorbs anything into **kw,
    # which is why the test below asserts silence there.)
    path = write(workspace, "import ast\n\nast.parse('x', no_such_kwarg=1)\n")
    report = docs.analyse(path)
    assert [f["kind"] for f in report["findings"]] == ["unknown_keyword"]
    assert "no_such_kwarg" in report["findings"][0]["message"]


def test_a_valid_call_produces_nothing(workspace):
    path = write(workspace, "import ast\n\nast.parse('x', mode='eval')\n")
    assert docs.analyse(path)["findings"] == []


def test_a_function_absorbing_kwargs_is_never_flagged(workspace):
    """`json.dumps` ends in **kw, so any keyword is legal and reporting one
    would be a false positive."""
    path = write(workspace, "import json\n\njson.dumps({}, whatever=1)\n")
    assert docs.analyse(path)["findings"] == []


def test_kwargs_absorbing_signatures_are_not_false_positives(workspace):
    """A function taking **kwargs accepts anything, and reporting otherwise
    would make the tool noisy enough to be ignored."""
    path = write(
        workspace,
        "import nbformat\n\nnbformat.reads('{}', as_version=4, anything_at_all=1)\n",
    )
    pytest.importorskip("nbformat")
    assert docs.analyse(path)["findings"] == []


def test_star_kwargs_at_the_call_site_is_not_checked(workspace):
    path = write(workspace, "import json\n\nopts = {}\njson.dumps({}, **opts)\n")
    assert docs.analyse(path)["findings"] == []


def test_from_imports_resolve(workspace):
    path = write(workspace, "from ast import parse\n\nparse('x', bogus=1)\n")
    report = docs.analyse(path)
    assert [f["kind"] for f in report["findings"]] == ["unknown_keyword"]


def test_aliased_imports_resolve(workspace):
    path = write(workspace, "import json as j\n\nj.no_such_function()\n")
    assert docs.analyse(path)["findings"][0]["kind"] == "missing_attribute"


def test_an_unimportable_module_is_reported_once(workspace):
    """Twenty copies of `pip install x` buries the findings that matter."""
    path = write(
        workspace,
        "import definitely_not_a_real_package as p\n\np.a()\np.b()\np.c()\n",
    )
    findings = docs.analyse(path)["findings"]
    assert len(findings) == 1
    assert findings[0]["kind"] == "module_not_importable"


def test_stdlib_is_not_reported_as_missing_a_distribution(workspace):
    path = write(workspace, "import json\nimport ast\n\nast.parse('x')\njson.dumps({})\n")
    report = docs.analyse(path)
    assert all(m["stdlib"] for m in report["modules"])
    assert report["findings"] == []


def test_unparseable_files_fail_with_a_useful_message(workspace):
    path = write(workspace, "def broken(:\n")
    with pytest.raises(GradError) as exc:
        docs.analyse(path)
    assert exc.value.exit_code == 9
    assert "syntax" in (exc.value.fix or "").lower()


def test_signature_lookup_reports_keyword_only_parameters(workspace):
    report = docs.signature_of("json", "dumps")
    assert report["exists"] is True
    assert "indent" in report["parameters"]


# ---------------------------------------------------------------------------
# the CLI contract
# ---------------------------------------------------------------------------
def test_check_exits_9_so_it_composes_with_preflight(workspace, capsys):
    """"Exit 9 (`a check ran and failed`) on findings, so it composes with
    preflight's declared-check mechanism if a pipeline wants it as a gate." """
    path = write(workspace, "import json\n\njson.no_such_function()\n")
    code = docs.cli.run(["check", str(path), "--json"])
    assert code == 9
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["error"]["detail"]["findings"]


def test_check_exits_0_on_a_clean_file(workspace, capsys):
    path = write(workspace, "import json\n\njson.dumps({})\n")
    assert docs.cli.run(["check", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["data"]["ok"] is True


# ---------------------------------------------------------------------------
# oracle 2: Context7, faked
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def fake_httpx(monkeypatch):
    calls: list[dict] = []
    response = FakeResponse(
        payload={
            "results": [
                {
                    "id": "/huggingface/huggingface_hub",
                    "title": "huggingface_hub",
                    "description": "the hub client",
                    "trustScore": 9.4,
                    "totalSnippets": 812,
                }
            ]
        }
    )

    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            calls.append({"url": url, "params": params, "headers": headers})
            return response

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    monkeypatch.setattr(http.credentials, "get", lambda name, required=True: None)
    return calls, response


def test_resolve_returns_library_ids(workspace, fake_httpx):
    from core import config as config_mod

    calls, _ = fake_httpx
    client = http.Context7(config_mod.load(reload=True))
    candidates = client.resolve("huggingface_hub")
    assert candidates[0]["library_id"] == "/huggingface/huggingface_hub"
    assert calls[0]["params"] == {"query": "huggingface_hub"}


def test_no_key_is_a_note_not_an_error(workspace, fake_httpx):
    """The key is free and raises rate limits rather than unlocking anything."""
    from core import config as config_mod

    client = http.Context7(config_mod.load(reload=True))
    assert client.authenticated is False
    assert "Authorization" not in (fake_httpx[0][0]["headers"] if fake_httpx[0] else {})


def test_a_404_names_the_unverified_path(workspace, monkeypatch):
    """"Not verified in session: the exact REST endpoint paths." A 404 must say
    so rather than looking like a missing library."""
    from core import config as config_mod

    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            return FakeResponse(status_code=404, text="nope")

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    monkeypatch.setattr(http.credentials, "get", lambda name, required=True: None)

    client = http.Context7(config_mod.load(reload=True))
    with pytest.raises(UpstreamError) as exc:
        client.resolve("anything")
    assert "api-guide" in (exc.value.fix or "")
    assert "[docs]" in (exc.value.fix or "")


def test_rate_limiting_suggests_the_free_key(workspace, monkeypatch):
    from core import config as config_mod

    class FakeHttpx:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            return FakeResponse(status_code=429)

    monkeypatch.setattr(http, "_httpx", lambda: FakeHttpx)
    monkeypatch.setattr(http.credentials, "get", lambda name, required=True: None)
    client = http.Context7(config_mod.load(reload=True))
    with pytest.raises(UpstreamError) as exc:
        client.resolve("anything")
    assert "context7_key" in (exc.value.fix or "")


def test_responses_are_cached(workspace, fake_httpx):
    """"caching matters more than it sounds: documentation lookups repeat
    heavily." """
    from core import config as config_mod

    calls, _ = fake_httpx
    client = http.Context7(config_mod.load(reload=True))
    client.resolve("huggingface_hub")
    client.resolve("huggingface_hub")
    assert len(calls) == 1


def test_offline_check_never_touches_the_network(workspace, monkeypatch):
    """Introspection is the half that works with no network at all."""
    def explode():
        raise AssertionError("check --offline must not reach the network")

    monkeypatch.setattr(http, "_httpx", explode)
    path = write(workspace, "import json\n\njson.dumps({})\n")
    assert docs.cli.run(["check", str(path), "--offline", "--json"]) == 0


# ---------------------------------------------------------------------------
# the credential
# ---------------------------------------------------------------------------
def test_context7_is_the_fifth_credential(workspace):
    from core import credentials
    from tools import jobs

    assert credentials.CONTEXT7_KEY in jobs.CREDENTIAL_NAMES
    assert credentials.CONTEXT7_KEY in credentials.status()


def test_context7_env_vars_are_scrubbed(workspace, monkeypatch):
    from core import credentials

    monkeypatch.setenv("CONTEXT7_API_KEY", "secret")
    monkeypatch.setenv("GRAD_CONTEXT7_KEY", "secret")
    removed = credentials.scrub_environment()
    assert "CONTEXT7_API_KEY" in removed
    assert "GRAD_CONTEXT7_KEY" in removed
