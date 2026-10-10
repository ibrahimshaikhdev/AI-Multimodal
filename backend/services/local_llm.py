from __future__ import annotations

import json
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_LOCAL_LLM_URL = "http://localhost:8080/v1/chat/completions"
LOCAL_LLM_TIMEOUT_SECONDS = 300
MAX_LOCAL_LLM_RESPONSE_BYTES = 2 * 1024 * 1024


def _server_endpoint(path: str) -> str:
    configured_url = os.environ.get("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL)
    parsed_url = urlsplit(configured_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("LOCAL_LLM_URL must be an absolute HTTP URL")
    return f"{parsed_url.scheme}://{parsed_url.netloc}{path}"


def local_context_size() -> int:
    try:
        request = Request(
            _server_endpoint("/props"),
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=LOCAL_LLM_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read(64 * 1024).decode("utf-8"))
        context_size = result["default_generation_settings"]["n_ctx"]
        if isinstance(context_size, bool) or not isinstance(context_size, int):
            raise ValueError("server returned an invalid context size")
        return context_size
    except HTTPError as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): HTTP {exc.code} while reading /props"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): {exc.reason}"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "Local model failed (is llama-server running?): request timed out "
            "after 300 seconds"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): could not read server context size: {exc}"
        ) from exc


def local_token_count(content: str) -> int:
    try:
        request = Request(
            _server_endpoint("/tokenize"),
            data=json.dumps({"content": content}).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=LOCAL_LLM_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read(256 * 1024).decode("utf-8"))
        tokens = result["tokens"]
        if not isinstance(tokens, list):
            raise ValueError("server returned an invalid token list")
        return len(tokens)
    except HTTPError as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): HTTP {exc.code} while tokenizing the prompt"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): {exc.reason}"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "Local model failed (is llama-server running?): request timed out "
            "after 300 seconds"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): could not count prompt tokens: {exc}"
        ) from exc


def local_generate(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 900,
) -> str:
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be a string or None")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")

        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "repeat_penalty": 1.1,
        }
        request = Request(
            os.environ.get("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=LOCAL_LLM_TIMEOUT_SECONDS) as response:
            response_body = response.read(MAX_LOCAL_LLM_RESPONSE_BYTES + 1)
        if len(response_body) > MAX_LOCAL_LLM_RESPONSE_BYTES:
            raise ValueError("response exceeded the 2 MiB limit")
        result = json.loads(response_body.decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("server returned an empty response")
        return content.strip()
    except HTTPError as exc:
        reason = f"HTTP {exc.code}"
        try:
            detail = exc.read(1024).decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        if "exceed_context_size_error" in detail:
            try:
                error_data = json.loads(detail)["error"]
                requested = error_data.get("n_prompt_tokens")
                available = error_data.get("n_ctx")
                detail = (
                    f"server context is too small ({requested} prompt tokens requested; "
                    f"{available} available). Restart llama-server with "
                    "`-c 8192` and verify `/props` reports `n_ctx: 8192`."
                )
            except (KeyError, TypeError, json.JSONDecodeError):
                detail = (
                    "server context is too small. Restart llama-server with "
                    "`-c 8192` and verify `/props` reports `n_ctx: 8192`."
                )
        if detail:
            reason = f"{reason}: {detail}"
        raise RuntimeError(
            f"Local model failed (is llama-server running?): {reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): {exc.reason}"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "Local model failed (is llama-server running?): request timed out "
            "after 300 seconds"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Local model failed (is llama-server running?): {exc}"
        ) from exc
