import json
import sys
import types
from urllib.error import URLError

import pytest

from backend.services import own_llm


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return json.dumps(
            {"choices": [{"message": {"content": self.payload}}]}
        ).encode("utf-8")


def test_own_generate_http_sends_user_only_when_no_system_is_given(monkeypatch):
    monkeypatch.setenv("OWN_MODEL_PATH", "")
    monkeypatch.setenv("OWN_LLM_URL", "http://localhost:8081/v1/chat/completions")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response("Own-model response")

    monkeypatch.setattr(own_llm, "urlopen", fake_urlopen)

    result = own_llm.own_generate("Question", 1500)

    assert result == "Own-model response"
    assert captured["body"] == {
        "messages": [{"role": "user", "content": "Question"}],
        "temperature": 0.1,
        "max_tokens": 1500,
    }
    assert captured["timeout"] == 600


def test_own_generate_http_includes_system_message_when_given(monkeypatch):
    monkeypatch.setenv("OWN_MODEL_PATH", "")
    monkeypatch.setenv("OWN_LLM_URL", "http://localhost:8081/v1/chat/completions")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response("Answer")

    monkeypatch.setattr(own_llm, "urlopen", fake_urlopen)

    assert own_llm.own_generate("Question", 500, system="System") == "Answer"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
    ]


def test_own_generate_http_requests_json_mode(monkeypatch):
    monkeypatch.setenv("OWN_MODEL_PATH", "")
    monkeypatch.setenv("OWN_LLM_URL", "http://localhost:8081/v1/chat/completions")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response('{"valid": true}')

    monkeypatch.setattr(own_llm, "urlopen", fake_urlopen)

    assert own_llm.own_generate("Question", 1500, json_mode=True) == (
        '{"valid": true}'
    )
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_own_generate_in_process_loads_once_and_uses_physical_cores(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "fine-tuned.gguf"
    model_path.write_bytes(b"test model file")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    monkeypatch.setenv("OWN_LLM_URL", "http://localhost:8081/v1/chat/completions")
    monkeypatch.setattr(own_llm, "_MODEL", None)
    captured = {"instances": 0, "calls": []}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured["instances"] += 1
            captured["load_kwargs"] = kwargs

        def create_chat_completion(self, **kwargs):
            captured["calls"].append(kwargs)
            return {"choices": [{"message": {"content": "In-process response"}}]}

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        types.SimpleNamespace(Llama=FakeLlama),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(cpu_count=lambda logical: 8 if not logical else 16),
    )

    assert own_llm.own_generate("Question", 600, system="System") == "In-process response"
    assert own_llm.own_generate("Second question", 500) == "In-process response"
    assert captured["instances"] == 1
    assert captured["load_kwargs"] == {
        "model_path": str(model_path),
        "n_ctx": 16384,
        "n_threads": 8,
        "verbose": False,
    }
    assert captured["calls"][0]["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
    ]
    assert captured["calls"][1]["messages"] == [
        {"role": "user", "content": "Second question"}
    ]


def test_own_generate_in_process_requests_json_mode(monkeypatch, tmp_path):
    model_path = tmp_path / "fine-tuned.gguf"
    model_path.write_bytes(b"test model file")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    monkeypatch.setattr(own_llm, "_MODEL", None)
    captured = {}

    class FakeLlama:
        def __init__(self, **_kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": '{"valid": true}'}}]}

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        types.SimpleNamespace(Llama=FakeLlama),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(cpu_count=lambda logical: 8 if not logical else 16),
    )

    assert own_llm.own_generate("Question", 1500, json_mode=True) == (
        '{"valid": true}'
    )
    assert captured["response_format"] == {"type": "json_object"}


def test_own_generate_requires_a_model_configuration(monkeypatch):
    monkeypatch.setenv("OWN_MODEL_PATH", "")
    monkeypatch.setenv("OWN_LLM_URL", "")

    with pytest.raises(
        RuntimeError,
        match="Own model is not configured \\(set OWN_MODEL_PATH or OWN_LLM_URL\\)",
    ):
        own_llm.own_generate("Question", 500)


def test_own_generate_reports_http_connection_failure(monkeypatch):
    monkeypatch.setenv("OWN_MODEL_PATH", "")
    monkeypatch.setenv("OWN_LLM_URL", "http://localhost:8081/v1/chat/completions")
    monkeypatch.setattr(
        own_llm,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("refused")),
    )

    with pytest.raises(RuntimeError, match="Own model generation failed: refused"):
        own_llm.own_generate("Question", 500)
