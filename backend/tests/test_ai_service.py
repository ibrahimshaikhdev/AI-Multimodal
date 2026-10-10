import json
from io import BytesIO
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import User
from backend.services.ai_service import (
    AIService,
    AIServiceConfigurationError,
    AIServiceTimeout,
    AIServiceUnavailable,
    AIServiceUpstreamError,
)


@pytest.fixture
def ai_client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        user = User(
            first_name="AI",
            last_name="Tester",
            email="ai-tester@example.com",
            password_hash=generate_password_hash("AITestPass123!"),
            role="patient",
        )
        db.session.add(user)
        db.session.commit()
        with app.test_client() as client:
            yield client, app
        db.session.remove()
        db.drop_all()


def _token(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "ai-tester@example.com", "password": "AITestPass123!"},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def _configured_ai_service(**overrides):
    return AIService(
        gemini_api_key="test-gemini-key",
        openrouter_api_key="test-openrouter-key",
        **overrides,
    )


def test_ai_service_uses_gemini_generate_content_and_context():
    response_body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from Gemini."}],
                    },
                }
            ]
        }
    ).encode()
    service = _configured_ai_service(
        gemini_model="configured-model",
        timeout_seconds=12,
    )

    with patch(
        "backend.services.ai_service.urlopen",
        return_value=BytesIO(response_body),
    ) as open_url:
        answer = service.generate("Say hello.", context={"source": "test"})

    assert answer == "Hello from Gemini."
    request = open_url.call_args.args[0]
    request_payload = json.loads(request.data)
    assert request.full_url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "configured-model:generateContent"
    )
    assert request.get_header("X-goog-api-key") == "test-gemini-key"
    assert request_payload["contents"][0]["parts"][0]["text"] == (
        'Context:\n{\n  "source": "test"\n}\n\nRequest:\nSay hello.'
    )
    assert open_url.call_args.kwargs["timeout"] == 12


def test_legacy_local_setting_uses_gemini_without_contacting_local_server():
    response_body = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": "Cloud answer."}]}}]}
    ).encode()
    service = AIService(
        provider="local",
        fallback_provider="",
        gemini_api_key="test-gemini-key",
    )
    with patch(
        "backend.services.ai_service.urlopen",
        return_value=BytesIO(response_body),
    ) as open_url:
        assert service.generate("Prompt") == "Cloud answer."

    assert open_url.call_args.args[0].full_url.startswith(
        "https://generativelanguage.googleapis.com/"
    )


def test_ai_service_uses_openrouter_chat_completions_content_parts():
    service = _configured_ai_service(provider="openrouter", fallback_provider="")
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Part one. "},
                            {"type": "text", "text": "Part two."},
                        ]
                    }
                }
            ]
        }
    ).encode()

    with patch(
        "backend.services.ai_service.urlopen",
        return_value=BytesIO(response_body),
    ) as open_url:
        assert service.generate("Prompt") == "Part one. Part two."

    assert open_url.call_args.args[0].full_url == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert open_url.call_args.args[0].get_header("Authorization") == (
        "Bearer test-openrouter-key"
    )


def test_ai_service_summary_prompt_preserves_source_and_safety_instructions():
    source_text = "Date: 2025-01-02. Finding: Test value 7 units."
    response_body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Reported findings: Test value 7 units."}
                        ]
                    }
                }
            ]
        }
    ).encode()

    with patch(
        "backend.services.ai_service.urlopen",
        return_value=BytesIO(response_body),
    ) as open_url:
        assert _configured_ai_service().summarize(source_text) == (
            "Reported findings: Test value 7 units."
        )

    content = json.loads(open_url.call_args.args[0].data)["contents"][0]["parts"][0]["text"]
    for required_instruction in (
        "provided medical report only",
        "Do not invent",
        "measurements, dates",
        "Clearly distinguish reported findings from recommendations",
        "repeat each explicit important date and measurement exactly as written",
        "not a diagnosis",
        "If the report text is insufficient",
    ):
        assert required_instruction in content
    assert source_text in content


def test_ai_service_comparison_prompt_requires_named_sources_and_evidence():
    report_1 = "Report 1 date: 2024-01-15. Hemoglobin: 12.1 g/dL."
    report_2 = "Report 2 date: 2025-01-15. Hemoglobin: 13.4 g/dL."
    response_body = json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": "Evidence-based comparison."}]}}
            ]
        }
    ).encode()

    with patch(
        "backend.services.ai_service.urlopen",
        return_value=BytesIO(response_body),
    ) as open_url:
        assert _configured_ai_service().compare(
            report_1,
            report_2,
            "Morgan Reed — Lab report — 2024-01-15 (#12)",
            "Morgan Reed — Lab report — 2025-01-15 (#18)",
        ) == "Evidence-based comparison."

    content = json.loads(open_url.call_args.args[0].data)["contents"][0]["parts"][0]["text"]
    for required_instruction in (
        "Refer to each source by its exact context label",
        "never call them Report 1, Report 2, Paper 1, or Paper 2",
        "Keep the two sources separate",
        "exact source wording from BOTH reports",
        "same measurement and compatible units",
        "not a diagnosis",
    ):
        assert required_instruction in content
    assert report_1 in content
    assert report_2 in content
    assert "Morgan Reed — Lab report — 2024-01-15 (#12)" in content
    assert "Morgan Reed — Lab report — 2025-01-15 (#18)" in content


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (URLError(ConnectionRefusedError()), AIServiceUnavailable),
        (TimeoutError(), AIServiceTimeout),
        (HTTPError("https://generativelanguage.googleapis.com", 500, "failure", {}, None), AIServiceUpstreamError),
    ],
)
def test_ai_service_maps_transport_errors(response, error_type):
    service = _configured_ai_service(fallback_provider="")
    with patch("backend.services.ai_service.urlopen", side_effect=response):
        with pytest.raises(error_type):
            service.generate("Prompt")


def test_ai_service_falls_back_to_openrouter_after_gemini_quota_error():
    failure = HTTPError(
        "https://generativelanguage.googleapis.com",
        429,
        "quota",
        {},
        None,
    )
    fallback_body = json.dumps(
        {"choices": [{"message": {"content": "OpenRouter fallback answer."}}]}
    ).encode()
    with patch(
        "backend.services.ai_service.urlopen",
        side_effect=[failure, BytesIO(fallback_body)],
    ) as open_url:
        answer = _configured_ai_service().generate("Use OCR text only.")

    assert answer == "OpenRouter fallback answer."
    assert open_url.call_count == 2
    assert "generativelanguage.googleapis.com" in open_url.call_args_list[0].args[0].full_url
    fallback_request = open_url.call_args_list[1].args[0]
    assert fallback_request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert fallback_request.get_header("Authorization") == "Bearer test-openrouter-key"
    assert "localhost" not in fallback_request.full_url


def test_ai_service_requires_configured_cloud_credentials_without_network_request():
    service = AIService()
    with patch("backend.services.ai_service.urlopen") as open_url:
        with pytest.raises(AIServiceConfigurationError):
            service.generate("Prompt")
    open_url.assert_not_called()


def test_ai_service_uses_fallback_when_primary_credentials_are_missing():
    fallback_body = json.dumps(
        {"choices": [{"message": {"content": "Fallback response."}}]}
    ).encode()
    service = AIService(
        provider="gemini",
        fallback_provider="openrouter",
        openrouter_api_key="test-openrouter-key",
        openrouter_model="test-model",
    )
    with patch(
        "backend.services.ai_service.urlopen",
        return_value=BytesIO(fallback_body),
    ) as open_url:
        assert service.generate("Text only.") == "Fallback response."

    assert open_url.call_count == 1
    request = open_url.call_args.args[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-openrouter-key"


def test_ai_service_rejects_unsupported_provider_without_network_request():
    service = AIService(provider="cloud", fallback_provider="")
    with patch("backend.services.ai_service.urlopen") as open_url:
        with pytest.raises(AIServiceConfigurationError):
            service.generate("Prompt")
    open_url.assert_not_called()


@pytest.mark.parametrize(
    "prompt",
    ["", "  ", None, 1, "p" * (AIService.MAX_INPUT_CHARACTERS + 1)],
)
def test_ai_service_validates_prompts(prompt):
    with pytest.raises(ValueError):
        _configured_ai_service().generate(prompt)


def test_ai_test_endpoint_requires_authentication(ai_client):
    client, _ = ai_client
    response = client.post("/api/ai/test", json={"prompt": "Say hello."})
    assert response.status_code == 401
    assert response.get_json()["error"] == "Authentication required"


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (None, "A JSON object is required."),
        ({}, "prompt must be a non-empty string."),
        ({"prompt": " "}, "prompt must be a non-empty string."),
        ({"prompt": 3}, "prompt must be a non-empty string."),
    ],
)
def test_ai_test_endpoint_validates_request(ai_client, payload, expected_error):
    client, _ = ai_client
    token = _token(client)
    response = client.post(
        "/api/ai/test",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": expected_error}


def test_ai_test_endpoint_returns_generated_answer_and_safe_error(ai_client):
    client, app = ai_client
    token = _token(client)
    headers = {"Authorization": f"Bearer {token}"}

    class FakeAIService:
        MAX_INPUT_CHARACTERS = 20000

        def generate(self, prompt):
            assert prompt == "Say hello in one sentence."
            return "Hello!"

    app.extensions["ai_service"] = FakeAIService()
    response = client.post(
        "/api/ai/test",
        headers=headers,
        json={"prompt": " Say hello in one sentence. "},
    )
    assert response.status_code == 200
    assert response.get_json() == {"success": True, "answer": "Hello!"}

    app.extensions["ai_service"] = _configured_ai_service(fallback_provider="")
    with patch(
        "backend.services.ai_service.urlopen",
        side_effect=URLError(ConnectionRefusedError()),
    ):
        unavailable = client.post(
            "/api/ai/test",
            headers=headers,
            json={"prompt": "Say hello."},
        )
    assert unavailable.status_code == 503
    assert unavailable.get_json() == {
        "success": False,
        "error": "Configured AI providers are unavailable.",
    }
