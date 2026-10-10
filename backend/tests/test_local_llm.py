import json
from urllib.error import URLError

import pytest

from backend.services import local_llm


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit):
        return self.body


def test_local_generate_uses_openai_compatible_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            b'{"choices":[{"message":{"content":"  Local answer  "}}]}'
        )

    monkeypatch.setenv("LOCAL_LLM_URL", "http://127.0.0.1:8081/v1/chat/completions")
    monkeypatch.setattr(local_llm, "urlopen", fake_urlopen)

    answer = local_llm.local_generate("User prompt", system="System prompt")

    assert answer == "Local answer"
    assert captured["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert captured["body"] == {
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "repeat_penalty": 1.1,
    }
    assert captured["timeout"] == 300


def test_local_generate_accepts_output_token_limit(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return FakeResponse(
            b'{"choices":[{"message":{"content":"Concise answer."}}]}'
        )

    monkeypatch.setattr(local_llm, "urlopen", fake_urlopen)

    assert local_llm.local_generate("User prompt", max_tokens=240) == "Concise answer."
    assert captured["body"]["max_tokens"] == 240
    assert captured["body"]["repeat_penalty"] == 1.1


def test_local_generate_reports_connection_failure(monkeypatch):
    def fail_urlopen(_request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(local_llm, "urlopen", fail_urlopen)

    with pytest.raises(RuntimeError, match="Local model failed \\(is llama-server running\\?\\)"):
        local_llm.local_generate("User prompt")


def test_local_generate_explains_server_context_size_failure(monkeypatch):
    from urllib.error import HTTPError
    from io import BytesIO

    def fail_urlopen(_request, timeout):
        raise HTTPError(
            "http://localhost:8080/v1/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(
                b'{"error":{"code":400,"message":"too large",'
                b'"type":"exceed_context_size_error","n_prompt_tokens":2080,"n_ctx":2048}}'
            ),
        )

    monkeypatch.setattr(local_llm, "urlopen", fail_urlopen)

    with pytest.raises(RuntimeError, match="2080 prompt tokens requested; 2048 available"):
        local_llm.local_generate("User prompt")
