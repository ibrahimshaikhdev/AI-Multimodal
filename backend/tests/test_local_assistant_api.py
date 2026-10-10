import json
from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import LocalAssistantConversation, MedicalReport, Patient, User
from backend.services.own_llm import OWN_ASSISTANT_SYSTEM, build_own_assistant_prompt


@pytest.fixture
def local_assistant_workspace():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        owner = User(
            first_name="Local",
            last_name="Owner",
            email="local-assistant@example.com",
            password_hash=generate_password_hash("LocalPass123!"),
            role="patient",
        )
        other = User(
            first_name="Other",
            last_name="Account",
            email="other-local-assistant@example.com",
            password_hash=generate_password_hash("LocalPass123!"),
            role="patient",
        )
        db.session.add_all([owner, other])
        db.session.flush()
        patient = Patient(first_name="Jamie", last_name="Reed", created_by=owner)
        other_patient = Patient(first_name="Private", last_name="Patient", created_by=other)
        db.session.add_all([patient, other_patient])
        db.session.flush()
        report = MedicalReport(
            patient_id=patient.id,
            report_type="lab_results",
            report_date=date(2025, 5, 1),
            extracted_text="Hemoglobin: 13.2 g/dL",
            processing_status="processed",
        )
        other_report = MedicalReport(
            patient_id=other_patient.id,
            report_type="lab_results",
            extracted_text="Private account data.",
            processing_status="processed",
        )
        db.session.add_all([report, other_report])
        db.session.commit()
        owner_id, other_id = owner.id, other.id
        patient_id, report_id = patient.id, report.id
        other_report_id = other_report.id

        with app.test_client() as client:
            yield client, app, owner_id, other_id, patient_id, report_id, other_report_id

        db.session.remove()
        db.drop_all()


def _token(client, email):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "LocalPass123!"},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def test_local_assistant_persists_account_scoped_turns_and_reuses_them(
    local_assistant_workspace,
    monkeypatch,
):
    client, app, owner_id, other_id, _patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    other_token = _token(client, "other-local-assistant@example.com")
    captured = []

    def generate(prompt, max_tokens, system=None):
        captured.append((prompt, system))
        assert max_tokens == 500
        return "The report lists hemoglobin as 13.2 g/dL."

    monkeypatch.setattr("backend.routes.local_assistant.own_generate", generate)
    monkeypatch.setattr(
        "backend.routes.local_assistant.build_own_assistant_prompt",
        lambda question, _user_id: (
            "PROJECT KNOWLEDGE:\n(nothing relevant found)\n\nPATIENT DATA:\n"
            "(none)\n\n"
            f"QUESTION: {question}"
        ),
    )

    first = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What hemoglobin value is reported?",
            "context_type": "report",
            "report_id": report_id,
        },
    )
    assert first.status_code == 200
    first_body = first.get_json()
    assert first_body["answer"] == "The report lists hemoglobin as 13.2 g/dL."
    assert first_body["sources"][0]["report_id"] == report_id
    assert len(first_body["messages"]) == 2
    assert captured[0][1]
    assert "QUESTION: What hemoglobin value is reported?" in captured[0][0]

    with app.app_context():
        saved = db.session.query(LocalAssistantConversation).filter_by(user_id=owner_id).one()
        assert len(json.loads(saved.messages_json)) == 2

    second = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "And what unit?",
            "context_type": "report",
            "report_id": report_id,
        },
    )
    assert second.status_code == 200
    assert "And what unit?" in captured[1][0]
    assert "What hemoglobin value is reported?" not in captured[1][0]

    other_history = client.get(
        "/api/ai/local-conversation",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert other_id != owner_id
    assert other_history.status_code == 200
    assert other_history.get_json()["messages"] == []

    cleared = client.delete(
        "/api/ai/local-conversation",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert cleared.status_code == 200
    assert cleared.get_json()["messages"] == []
    history = client.get(
        "/api/ai/local-conversation",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert history.get_json()["messages"] == []


def test_local_assistant_rejects_unowned_report(local_assistant_workspace):
    client, _app, _owner_id, _other_id, _patient_id, _report_id, other_report_id = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Read this report",
            "context_type": "report",
            "report_id": other_report_id,
        },
    )

    assert response.status_code == 404
    assert response.get_json() == {"success": False, "error": "Report not found."}


def test_local_assistant_failure_is_reported_without_fallback(
    local_assistant_workspace,
    monkeypatch,
):
    client, app, owner_id, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")

    def fail_local(_prompt, _max_tokens, system=None):
        raise RuntimeError("Own model generation failed: refused")

    monkeypatch.setattr("backend.routes.local_assistant.own_generate", fail_local)
    monkeypatch.setattr(
        "backend.routes.local_assistant.build_own_assistant_prompt",
        lambda _question, _user_id: "Own-model prompt",
    )
    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What does it say?",
            "context_type": "report",
            "report_id": local_assistant_workspace[5],
        },
    )

    assert response.status_code == 502
    assert response.get_json() == {
        "success": False,
        "error": "Own model generation failed: refused",
    }
    with app.app_context():
        saved = db.session.query(LocalAssistantConversation).filter_by(user_id=owner_id).first()
        assert saved is None


def test_own_assistant_uses_the_project_prompt_and_500_token_limit(
    local_assistant_workspace,
    monkeypatch,
):
    client, app, owner_id, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    with app.app_context():
        report = db.session.get(MedicalReport, local_assistant_workspace[5])
        report.extracted_text = "Recent report details. " * 1200
        db.session.commit()

    captured = {}

    def generate(prompt, max_tokens, system=None):
        captured["prompt"] = prompt
        captured["system"] = system
        captured["max_tokens"] = max_tokens
        return "A recent lab report with saved extracted results."

    monkeypatch.setattr("backend.routes.local_assistant.own_generate", generate)
    monkeypatch.setattr(
        "backend.routes.local_assistant.build_own_assistant_prompt",
        lambda _question, _user_id: (
            "PROJECT KNOWLEDGE:\nRelevant project details\n\n"
            "PATIENT DATA:\nRecent report details.\n\n"
            "QUESTION: What's the recent report about?"
        ),
    )
    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What's the recent report about?",
            "context_type": "report",
            "report_id": local_assistant_workspace[5],
        },
    )

    assert response.status_code == 200, response.get_json()
    assert "PROJECT KNOWLEDGE:" in captured["prompt"]
    assert captured["max_tokens"] == 500
    assert response.get_json()["success"] is True


def test_own_assistant_prompt_retrieves_project_kb_and_only_owned_named_patients(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, *_rest = local_assistant_workspace
    with app.app_context():
        prompt = build_own_assistant_prompt(
            "What does OCR use, and what does Jamie Reed's latest report say?",
            owner_id,
        )
        unowned_prompt = build_own_assistant_prompt(
            "What does Private Patient's latest report say?",
            owner_id,
        )

    assert OWN_ASSISTANT_SYSTEM == (
        "You are the built-in assistant of the project 'Multimodal AI Research & "
        "Medical Report Intelligence'. Answer questions about the project using ONLY "
        "the PROJECT KNOWLEDGE provided. For questions about a patient, use ONLY the "
        "PATIENT DATA provided. If the answer is not in the provided context, say you "
        "don't have that information. Never diagnose or prescribe; for personal "
        "medical decisions tell the user to consult a doctor. Be concise."
    )
    assert prompt.startswith("PROJECT KNOWLEDGE:\n")
    assert "OCR" in prompt
    assert "Jamie Reed, latest report (2025-05-01)" in prompt
    assert "Hemoglobin: 13.2 g/dL" in prompt
    assert "Private account data." not in prompt
    assert "PATIENT DATA:\n(none)" in unowned_prompt


def test_own_assistant_includes_account_report_count_for_count_questions(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, *_rest = local_assistant_workspace
    with app.app_context():
        prompt = build_own_assistant_prompt(
            "How many reports we uploaded?",
            owner_id,
        )

    assert "your account has 1 uploaded medical report" in prompt
    assert "Private account data." not in prompt


def test_own_assistant_project_question_does_not_require_uploaded_research(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    monkeypatch.setattr(
        "backend.routes.local_assistant.build_own_assistant_prompt",
        lambda question, _user_id: (
            "PROJECT KNOWLEDGE:\nProject overview\n\n"
            "PATIENT DATA:\n(none)\n\n"
            f"QUESTION: {question}"
        ),
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.own_generate",
        lambda _prompt, _max_tokens, system=None: "The project uses OCR.",
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What does the project use for OCR?", "context_type": "research"},
    )

    assert response.status_code == 200
    assert response.get_json()["context_type"] == "research"
    assert response.get_json()["sources"] == []


def test_own_assistant_web_search_returns_and_persists_separate_results(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    search_results = [
        {
            "title": "Research source",
            "link": "https://example.org/source",
            "snippet": "A relevant web excerpt.",
        }
    ]
    captured = {}

    monkeypatch.setattr(
        "backend.routes.local_assistant.search_web",
        lambda query: (captured.setdefault("query", query), search_results)[1],
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.build_own_assistant_prompt",
        lambda question, _user_id, web_search_results=None: (
            captured.update({"web_search_results": web_search_results})
            or f"PROJECT KNOWLEDGE:\n{question}"
        ),
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.own_generate",
        lambda *_args, **_kwargs: "The own model answer.",
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Summarize current research",
            "context_type": "research",
            "web_search": True,
        },
    )

    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    assert payload["answer"] == "The own model answer."
    assert payload["web_search_enabled"] is True
    assert payload["web_search_results"] == search_results
    assert captured["query"] == "Summarize current research"
    assert captured["web_search_results"] == search_results
    assert payload["messages"][-1]["web_search_results"] == search_results


def test_own_assistant_reports_web_search_error_without_skipping_own_model(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    monkeypatch.setattr(
        "backend.routes.local_assistant.search_web",
        lambda _query: (_ for _ in ()).throw(
            RuntimeError("Web search is not configured (set SERPAPI_API_KEY).")
        ),
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.build_own_assistant_prompt",
        lambda question, _user_id: f"PROJECT KNOWLEDGE:\n{question}",
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.own_generate",
        lambda *_args, **_kwargs: "The answer without web results.",
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Question",
            "context_type": "research",
            "web_search": True,
        },
    )

    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    assert payload["answer"] == "The answer without web results."
    assert payload["web_search_results"] == []
    assert "set SERPAPI_API_KEY" in payload["web_search_error"]
    assert payload["messages"][-1]["web_search_error"] == payload["web_search_error"]


def test_own_assistant_rejects_non_boolean_web_search(local_assistant_workspace):
    client, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "Question", "web_search": "true"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "web_search must be a boolean."
