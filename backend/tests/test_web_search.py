import json
from urllib.parse import parse_qs, urlparse

import pytest

from backend.services import web_search


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return json.dumps(self.payload).encode("utf-8")


def test_search_web_calls_serpapi_and_returns_safe_result_fields(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response(
            {
                "organic_results": [
                    {
                        "title": "Research result",
                        "link": "https://example.org/paper",
                        "snippet": "A source excerpt.",
                        "ignored": "unneeded field",
                    },
                    {
                        "title": "Unsafe link",
                        "link": "javascript:alert(1)",
                        "snippet": "Should be discarded.",
                    },
                ]
            }
        )

    monkeypatch.setattr(web_search, "urlopen", fake_urlopen)

    results = web_search.search_web("medical research")

    assert results == [
        {
            "title": "Research result",
            "link": "https://example.org/paper",
            "snippet": "A source excerpt.",
        }
    ]
    query = parse_qs(urlparse(captured["url"]).query)
    assert query["engine"] == ["google"]
    assert query["q"] == ["medical research"]
    assert query["api_key"] == ["test-key"]
    assert captured["timeout"] == web_search.SERPAPI_TIMEOUT_SECONDS


def test_search_web_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="set SERPAPI_API_KEY"):
        web_search.search_web("question")
