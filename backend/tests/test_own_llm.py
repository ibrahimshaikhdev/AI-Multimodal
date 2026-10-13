import re
import sys
import types

import pytest

from backend.services import own_llm


class _FormattedPrompt:
    def __init__(self, prompt, added_special=False):
        self.prompt = prompt
        self.added_special = added_special


class _ChatFormatter:
    def __init__(self, prompt_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append(messages)
        return _FormattedPrompt(
            "rendered chat prompt",
            added_special=True,
        )


def _completion_handler(formatter):
    def handler(*, llama, messages, **_kwargs):
        return formatter(messages=messages)

    return handler


class _FakeLlama:
    def __init__(self, *, n_ctx=4096, prompt_tokens=5):
        self.context_size = n_ctx
        self.token_count = prompt_tokens
        self.chat_format = "qwen"
        self.formatter = _ChatFormatter(prompt_tokens)
        self._chat_handlers = {"qwen": _completion_handler(self.formatter)}
        self.calls = []

    def tokenize(self, _text, *, add_bos, special):
        assert add_bos is False
        assert special is True
        return list(range(self.token_count))

    def n_ctx(self):
        return self.context_size

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "Fake in-process response"}}]}


def _fake_model_import(monkeypatch, captured, *, cores=8, llama_cls=None):
    class FakeLlama:
        def __init__(self, **kwargs):
            captured.setdefault("instances", []).append(kwargs)
            captured.setdefault("models", []).append(
                _FakeLlama(n_ctx=kwargs["n_ctx"])
            )

        def _instance(self):
            return captured["models"][-1]

        def __getattr__(self, name):
            return getattr(self._instance(), name)

    selected_class = llama_cls or FakeLlama
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        types.SimpleNamespace(Llama=selected_class),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(cpu_count=lambda logical: cores if not logical else cores * 2),
    )


@pytest.fixture(autouse=True)
def reset_model_cache(monkeypatch):
    monkeypatch.setattr(own_llm, "_MODEL", None)


def test_own_generate_in_process_loads_once_with_config_and_physical_cores(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "fine-tuned-v3.gguf"
    model_path.write_bytes(b"test model file")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    monkeypatch.setenv("OWN_N_CTX", "8192")
    captured = {}
    _fake_model_import(monkeypatch, captured, cores=8)

    assert own_llm._MODEL is None
    assert (
        own_llm.own_generate("Question", 600, system="System")
        == "Fake in-process response"
    )
    assert own_llm.own_generate("Second question", 500) == "Fake in-process response"
    assert captured["instances"] == [
        {
            "model_path": str(model_path),
            "n_ctx": 8192,
            "n_threads": 8,
            "n_threads_batch": 8,
            "use_mmap": True,
            "verbose": False,
        }
    ]
    model = captured["models"][0]
    assert model.calls[0] == {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
        ],
        "temperature": 0.1,
        "max_tokens": 600,
    }
    assert model.calls[1]["messages"] == [
        {"role": "user", "content": "Second question"}
    ]
    assert len(model.formatter.messages) == 2


def test_own_generate_uses_default_context_and_at_least_one_thread(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    captured = {}
    _fake_model_import(monkeypatch, captured, cores=None)

    own_llm.own_generate("Question", 12)

    assert captured["instances"][0]["n_ctx"] == 4096
    assert captured["instances"][0]["n_threads"] == 1
    assert captured["instances"][0]["n_threads_batch"] == 1


def test_own_generate_keeps_json_response_mode_for_existing_route(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    captured = {}
    _fake_model_import(monkeypatch, captured)

    own_llm.own_generate("Question", 20, json_mode=True)

    assert captured["models"][0].calls[0]["response_format"] == {
        "type": "json_object"
    }


def test_own_generate_rejects_prompt_that_exceeds_context(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    captured = {}
    _fake_model_import(monkeypatch, captured, cores=1)
    fake_model = _FakeLlama(n_ctx=30, prompt_tokens=20)

    class FixedLlama:
        def __init__(self, **_kwargs):
            captured["instances"] = captured.get("instances", 0) + 1

        def __getattr__(self, name):
            return getattr(fake_model, name)

    _fake_model_import(monkeypatch, captured, llama_cls=FixedLlama)

    with pytest.raises(
        RuntimeError,
        match=r"Prompt too long for the own model \(40 > 30 tokens\)",
    ):
        own_llm.own_generate("Question", 20)
    assert fake_model.calls == []


def test_own_generate_requires_model_path(monkeypatch):
    monkeypatch.setenv("OWN_MODEL_PATH", "")

    with pytest.raises(
        RuntimeError,
        match=r"Own model is not configured\. Set OWN_MODEL_PATH in \.env\.",
    ):
        own_llm.own_generate("Question", 10)


def test_own_generate_reports_missing_model_file(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.gguf"
    monkeypatch.setenv("OWN_MODEL_PATH", str(missing_path))

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"Own model file not found at {missing_path}. "
            "Check OWN_MODEL_PATH in .env."
        ),
    ):
        own_llm.own_generate("Question", 10)


def test_own_generate_reports_missing_llama_cpp(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "llama-cpp-python is not installed. "
            "Run: python -m pip install llama-cpp-python"
        ),
    ):
        own_llm.own_generate("Question", 10)


def test_own_generate_rejects_invalid_context_size(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    monkeypatch.setenv("OWN_N_CTX", "not-a-number")
    captured = {}
    _fake_model_import(monkeypatch, captured)

    with pytest.raises(RuntimeError, match="OWN_N_CTX must be a positive integer"):
        own_llm.own_generate("Question", 10)


def test_own_generate_wraps_generation_failures(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("OWN_MODEL_PATH", str(model_path))
    captured = {}
    _fake_model_import(monkeypatch, captured)
    own_llm._load_model()
    fake_model = captured["models"][0]

    def fail_generation(**_kwargs):
        raise ValueError("backend failure")

    fake_model.create_chat_completion = fail_generation

    with pytest.raises(RuntimeError, match="Own model generation failed: backend failure"):
        own_llm.own_generate("Question", 10)
