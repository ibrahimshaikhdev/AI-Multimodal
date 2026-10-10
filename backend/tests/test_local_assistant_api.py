import json
from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import LocalAssistantConversation, MedicalReport, Patient, User


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

    def generate(prompt, system=None, max_tokens=900):
        captured.append((prompt, system))
        assert max_tokens == 500
        return "The report lists hemoglobin as 13.2 g/dL."

    monkeypatch.setattr("backend.routes.local_assistant.local_generate", generate)
    monkeypatch.setattr(
        "backend.routes.local_assistant.local_context_size",
        lambda: 2048,
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.local_token_count",
        lambda _text: 100,
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
    assert "Hemoglobin: 13.2 g/dL" in captured[0][0]

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
    assert "What hemoglobin value is reported?" in captured[1][0]
    assert "And what unit?" in captured[1][0]

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

    def fail_local(_prompt, system=None):
        raise RuntimeError("Local model failed (is llama-server running?): refused")

    monkeypatch.setattr("backend.routes.local_assistant.local_generate", fail_local)
    monkeypatch.setattr(
        "backend.routes.local_assistant.local_context_size",
        lambda: 2048,
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.local_token_count",
        lambda _text: 100,
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
        "error": "Local model failed (is llama-server running?): refused",
    }
    with app.app_context():
        saved = db.session.query(LocalAssistantConversation).filter_by(user_id=owner_id).first()
        assert saved is None


def test_local_assistant_shortens_source_context_to_fit_server_window(
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

    def count_tokens(text):
        return len(text) // 3

    def generate(prompt, system=None, max_tokens=900):
        captured["prompt"] = prompt
        captured["system"] = system
        captured["max_tokens"] = max_tokens
        return "A recent lab report with saved extracted results."

    monkeypatch.setattr(
        "backend.routes.local_assistant.local_context_size",
        lambda: 2048,
    )
    monkeypatch.setattr(
        "backend.routes.local_assistant.local_token_count",
        count_tokens,
    )
    monkeypatch.setattr("backend.routes.local_assistant.local_generate", generate)
    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"******"},
        json={
            "question": "What's the recent report about?",
            "context_type": "report",
            "report_id": local_assistant_workspace[5],
        },
    )

    assert response.status_code == 200
    assert len(captured["prompt"]) < len("Recent report details. " * 1200) + 1000
    assert captured["max_tokens"] == 500
    assert response.get_json()["success"] is True
