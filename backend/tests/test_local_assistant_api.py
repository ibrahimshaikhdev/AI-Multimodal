import json
from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import (
    DentalXrayAnalysisRecord,
    KneeScanAnalysis,
    LocalAssistantConversation,
    MedicalReport,
    Patient,
    ReportParameter,
    ResearchPaper,
    ScanAsset,
    User,
)
from backend.services import own_llm
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
            extracted_text=(
                "Hemoglobin: 13.2 g/dL\n"
                "WBC: 6.0 10^3/uL\n"
                "Platelets: 250 10^3/uL\n"
                "HbA1c: 5.4 %"
            ),
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


def _use_bearer_token(client, monkeypatch, token):
    original_post = client.post

    def authenticated_post(*args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        headers["Authorization"] = f"Bearer {token}"
        kwargs["headers"] = headers
        return original_post(*args, **kwargs)

    monkeypatch.setattr(client, "post", authenticated_post)


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
        assert max_tokens == 350
        return "The report lists hemoglobin as 13.2 g/dL."

    monkeypatch.setattr("backend.services.assistant_desks.own_generate_assistant", generate)
    monkeypatch.setattr(
        "backend.services.assistant_desks.build_own_assistant_prompt",
        lambda question, _user_id, **_options: (
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
    assert first_body["answer"].startswith(
        "The report lists hemoglobin as 13.2 g/dL."
    )
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
    assert "QUESTION: And what unit?" in captured[1][0]
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


def test_own_assistant_uses_selected_report_without_patient_name_in_question(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    captured = {}

    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda prompt, _max_tokens, system=None: (
            captured.update({"prompt": prompt, "system": system})
            or "This report lists the patient's lab results."
        ),
    )
    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What is this report about?",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )

    assert response.status_code == 200, response.get_json()
    assert "Jamie Reed, selected report (2025-05-01):" in captured["prompt"]
    assert "HbA1c: 5.4 %" in captured["prompt"]
    assert "PROJECT KNOWLEDGE:\n(none)" in captured["prompt"]
    assert "QUESTION: What is this report about?" in captured["prompt"]
    assert captured["system"] == OWN_ASSISTANT_SYSTEM


def test_own_assistant_retries_repeated_report_answers_then_fails_clearly(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    generated = []

    def build_prompt(question, _user_id, **options):
        return f"PATIENT DATA:\n{options.get('patient_data')}\nQUESTION: {question}"

    def generate(prompt, _max_tokens, system=None):
        generated.append(prompt)
        return "The report contains several lab values."

    monkeypatch.setattr(
        "backend.services.assistant_desks.build_own_assistant_prompt",
        build_prompt,
    )
    monkeypatch.setattr("backend.services.assistant_desks.own_generate_assistant", generate)

    first = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Any problem in the report?",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )
    second = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Is there anything good in the report?",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(generated) == 3
    assert "The prior assistant response was:" in generated[2]
    assert "Do not repeat or paraphrase that response." in generated[2]
    assert second.get_json()["answer"] != first.get_json()["answer"]
    assert (
        "couldn't produce a distinct selected-record answer"
        in second.get_json()["answer"]
    )


def test_own_assistant_handles_thanks_without_repeating_report_summary(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: pytest.fail("Thanks should not trigger generation."),
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "thank you",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )

    assert response.status_code == 200
    assert response.get_json()["answer"].startswith("You're welcome.")


def test_own_assistant_suppresses_unsubstantiated_report_recommendations(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: (
            "The report shows mild abnormalities. Recommend dietary modification "
            "and repeat testing in 3 months."
        ),
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Any problem in the report?",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )

    assert response.status_code == 200
    answer = response.get_json()["answer"]
    assert "does not substantiate that recommendation" in answer
    assert "dietary modification" not in answer
    assert "repeat testing" not in answer


def test_assistant_formats_selected_patient_timeline_chronologically(
    local_assistant_workspace,
    monkeypatch,
):
    client, app, _owner_id, _other_id, patient_id, _report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    with app.app_context():
        db.session.add_all(
            [
                MedicalReport(
                    patient_id=patient_id,
                    report_type="Follow-up",
                    report_date=date(2025, 6, 1),
                    extracted_text="Impression: Stable compared with prior report.",
                ),
                MedicalReport(
                    patient_id=patient_id,
                    report_type="diagnosis",
                    report_date=date(2025, 8, 1),
                    extracted_text="HbA1c: 8.0 %\nFasting glucose: 130 mg/dL",
                ),
                MedicalReport(
                    patient_id=patient_id,
                    report_type="general",
                    extracted_text=(
                        "IBRAHIM SHAIKH Full Stack Developer · React.js Developer "
                        "Pune, India · ibrahimshaikh.eng@gmail.com"
                    ),
                ),
                ScanAsset(
                    patient_id=patient_id,
                    modality="MRI",
                    body_region="knee",
                    study_date=date(2025, 7, 1),
                    original_filename="knee.png",
                    file_reference="uploads/knee.png",
                    processing_status="processed",
                ),
            ]
        )
        dental_scan = ScanAsset(
            patient_id=patient_id,
            modality="X_RAY",
            body_region="Dental",
            study_date=date(2025, 8, 3),
            original_filename="dental.png",
            file_reference="uploads/dental.png",
            processing_status="analysis_complete",
        )
        db.session.add(dental_scan)
        db.session.flush()
        db.session.add(
            DentalXrayAnalysisRecord(
                scan_id=dental_scan.id,
                model_name="experimental-detector",
                findings=[
                    {
                        "name": "Impacted tooth",
                        "score": 0.9,
                        "box": [10, 20, 30, 40],
                    }
                    for _ in range(4)
                ],
                image_shape=[512, 512],
            )
        )
        db.session.commit()
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: pytest.fail(
            "Timeline responses should use the deterministic record formatter."
        ),
    )
    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Tell me the timeline details.",
            "context_type": "patient_timeline",
            "patient_id": patient_id,
        },
    )

    assert response.status_code == 200, response.get_json()
    answer = response.get_json()["answer"]
    assert "Jamie Reed · Timeline overview (4 reports, 2 scans)" in answer
    assert "Aug 3, 2025 · X_RAY (Dental)" in answer
    assert "Experimental model marked 4 Impacted tooth" in answer
    assert "Aug 1, 2025 · diagnosis: 2 flagged:" in answer
    assert "HbA1c 8.0 % (high)" in answer
    assert "Jun 1, 2025 · Follow-up: Impression: Stable" in answer
    assert answer.index("Aug 3, 2025") < answer.index("Aug 1, 2025")
    assert answer.index("Aug 1, 2025") < answer.index("Jun 1, 2025")
    assert "box" not in answer
    assert "score" not in answer
    assert "email" not in answer
    assert "IBRAHIM SHAIKH" not in answer
    assert len(answer) < 1000
    assert "not diagnoses" in answer
    assert "not real medical advice" in answer


def test_own_assistant_extracts_selected_report_text_in_background(
    local_assistant_workspace,
    monkeypatch,
    tmp_path,
):
    client, app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    report_file = tmp_path / "report.pdf"
    report_file.write_bytes(b"pdf fixture")
    with app.app_context():
        report = db.session.get(MedicalReport, report_id)
        report.extracted_text = None
        report.file_reference = "uploads/report.pdf"
        db.session.commit()

    monkeypatch.setattr(
        "backend.routes.ai.get_stored_file_location",
        lambda _reference, _patient_id: (tmp_path, "report.pdf"),
    )
    monkeypatch.setattr(
        "backend.routes.ai.extract_document_text",
        lambda _content, _filename: "Report OCR text: vitamin D is 24 ng/mL.",
    )
    captured = {}
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda prompt, _max_tokens, system=None: (
            captured.update({"prompt": prompt})
            or "The report contains a vitamin D result."
        ),
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What is this report about?",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )

    assert response.status_code == 200, response.get_json()
    assert "Vitamin D: 24 ng/mL" in captured["prompt"]
    with app.app_context():
        report = db.session.get(MedicalReport, report_id)
        assert report.extracted_text == "Report OCR text: vitamin D is 24 ng/mL."
        assert report.processing_status == "processed"


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


def test_local_assistant_rejects_report_not_matching_selected_patient(
    local_assistant_workspace,
):
    client, _app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    with client.application.app_context():
        another_patient = Patient(first_name="Alex", last_name="Stone")
        another_patient.created_by_id = _owner_id
        db.session.add(another_patient)
        db.session.commit()
        other_patient_id = another_patient.id

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What is this report about?",
            "context_type": "report",
            "patient_id": other_patient_id,
            "report_id": report_id,
        },
    )

    assert other_patient_id != patient_id
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

    monkeypatch.setattr("backend.services.assistant_desks.own_generate_assistant", fail_local)
    monkeypatch.setattr(
        "backend.services.assistant_desks.build_own_assistant_prompt",
        lambda _question, _user_id, **_options: "Own-model prompt",
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

    monkeypatch.setattr("backend.services.assistant_desks.own_generate_assistant", generate)
    monkeypatch.setattr(
        "backend.services.assistant_desks.build_own_assistant_prompt",
        lambda _question, _user_id, **_options: (
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
    assert captured["max_tokens"] == 350
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
    assert "PATIENT DATA:\nJamie Reed, latest report (2025-05-01):" in prompt
    assert "Hemoglobin: 13.2 g/dL (normal)" in prompt
    assert "WBC: 6.0 10^3/uL (normal)" in prompt
    assert "HbA1c: 5.4 % (normal)" in prompt
    assert "Private account data." not in prompt
    assert "PATIENT DATA:\n(none)" in unowned_prompt


def test_own_assistant_retrieves_new_reports_and_scan_outputs_live(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, patient_id, *_rest = (
        local_assistant_workspace
    )
    with app.app_context():
        scan = ScanAsset(
            patient_id=patient_id,
            modality="MRI",
            body_region="knee",
            original_filename="knee-scan.nii.gz",
            file_reference="private/knee-scan.nii.gz",
            processing_status="processed",
        )
        db.session.add_all(
            [
                MedicalReport(
                    patient_id=patient_id,
                    report_type="Follow-up",
                    report_date=date(2025, 6, 1),
                    extracted_text=(
                        "Hemoglobin: 14 g/dL\n"
                        "WBC: 7 10^3/uL\n"
                        "HbA1c: 6.2 %"
                    ),
                    processing_status="processed",
                ),
                scan,
            ]
        )
        db.session.flush()
        db.session.add(
            KneeScanAnalysis(
                scan_id=scan.id,
                model_name="Experimental knee model",
                slice_index=4,
                predicted_class_index=1,
                predicted_class="Example class",
                confidence=0.81,
                probabilities={"example": 0.81},
                localization_status="localized",
                roi_box={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            )
        )
        db.session.commit()

        prompt = build_own_assistant_prompt(
            "What are Jamie Reed's latest report and scan?",
            owner_id,
        )

    assert "Jamie Reed, latest report (2025-06-01):" in prompt
    assert "Hemoglobin: 14 g/dL (normal)" in prompt
    assert "Scan (date unavailable): MRI knee, processed" in prompt
    assert "predicted class: Example class" in prompt
    patient_data = prompt.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    assert "experimental" not in patient_data.casefold()
    assert "Private account data." not in prompt


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


def test_own_assistant_lists_owned_patients_papers_and_latest_flagged_results(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    with app.app_context():
        latest_report = MedicalReport(
            patient_id=patient_id,
            report_type="lab_results",
            report_date=date(2026, 1, 15),
            extracted_text=(
                "HbAlc: 9.1 %\n"
                "Hemoglobin: 14 g/dL\n"
                "WEC: 12 10^3/uL"
            ),
            processing_status="processed",
        )
        second_patient = Patient(
            first_name="Asha",
            last_name="Patel",
            created_by_id=owner_id,
        )
        private_patient = (
            db.session.query(Patient)
            .filter(Patient.created_by_id != owner_id)
            .one()
        )
        private_latest = MedicalReport(
            patient_id=private_patient.id,
            report_type="lab_results",
            report_date=date(2026, 2, 1),
            extracted_text="Private latest result.",
            processing_status="processed",
        )
        db.session.add_all([latest_report, second_patient, private_latest])
        db.session.flush()
        db.session.add_all(
            [
                ReportParameter(
                    report_id=latest_report.id,
                    parameter="HbA1c",
                    value="9.1",
                    unit="%",
                    source="HbA1c 9.1 % (4.0-5.6) H",
                    fingerprint="latest-hba1c",
                ),
                ReportParameter(
                    report_id=report_id,
                    parameter="HbA1c",
                    value="12.0",
                    unit="%",
                    source="HbA1c 12.0 % H",
                    fingerprint="stale-hba1c",
                ),
                ReportParameter(
                    report_id=private_latest.id,
                    parameter="HbA1c",
                    value="11.0",
                    unit="%",
                    source="HbA1c 11.0 % H",
                    fingerprint="private-hba1c",
                ),
                ResearchPaper(
                    user_id=owner_id,
                    title="Owned paper",
                    topic="research",
                    original_filename="owned.pdf",
                    stored_filename="owned-paper.pdf",
                ),
                ResearchPaper(
                    user_id=private_patient.created_by_id,
                    title="Private paper",
                    topic="research",
                    original_filename="private.pdf",
                    stored_filename="private-paper.pdf",
                ),
            ]
        )
        db.session.commit()

        high_prompt = build_own_assistant_prompt(
            "Which patients have high HbA1c?", owner_id
        )
        patients_prompt = build_own_assistant_prompt("List patients", owner_id)
        papers_prompt = build_own_assistant_prompt(
            "Which research papers have I uploaded?", owner_id
        )

    high_data = high_prompt.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    assert high_data == "Jamie Reed: HbA1c 9.1 % (high)"
    assert "12.0" not in high_data
    assert "11.0" not in high_data
    assert "Patient count: 2." in patients_prompt
    assert "Patients: Jamie Reed, Asha Patel." in patients_prompt
    assert "Private Patient" not in patients_prompt
    assert "Uploaded research papers (1): Owned paper." in papers_prompt
    assert "Private paper" not in papers_prompt


def test_own_assistant_named_patient_data_keeps_latest_first_and_caps_length(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, patient_id, _report_id, _ = (
        local_assistant_workspace
    )
    with app.app_context():
        db.session.add_all(
            [
                MedicalReport(
                    patient_id=patient_id,
                    report_type="older",
                    report_date=date(2024, 1, 1),
                    extracted_text="Older report. " * 50,
                ),
                MedicalReport(
                    patient_id=patient_id,
                    report_type="latest",
                    report_date=date(2026, 1, 1),
                    extracted_text=(
                        "HbA1c 6.1%\n"
                        "Hemoglobin 14 g/dL\n"
                        "WBC 7 10^3/uL"
                    ),
                ),
            ]
        )
        db.session.commit()
        prompt = build_own_assistant_prompt(
            "What does Jamie Reed's latest report say?", owner_id
        )

    patient_data = prompt.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    assert patient_data.startswith("Jamie Reed, latest report (2026-01-01):")
    assert len(patient_data) <= own_llm.PATIENT_DATA_MAX_CHARACTERS


def test_own_assistant_retrieves_new_patient_records_on_the_next_request(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, patient_id, _report_id, _ = (
        local_assistant_workspace
    )
    with app.app_context():
        before_new_records = build_own_assistant_prompt(
            "What does Jamie's latest report and scan show?", owner_id
        )
        db.session.add_all(
            [
                MedicalReport(
                    patient_id=patient_id,
                    report_type="new_lab_results",
                    report_date=date(2026, 2, 1),
                    extracted_text=(
                        "HbA1c: 6.2 %\n"
                        "Hemoglobin: 14.0 g/dL\n"
                        "WBC: 7.0 10^3/uL"
                    ),
                    processing_status="processed",
                ),
                ScanAsset(
                    patient_id=patient_id,
                    modality="MRI",
                    body_region="knee",
                    study_date=date(2026, 2, 2),
                    original_filename="new-knee-scan.png",
                    file_reference="uploads/new-knee-scan.png",
                    processing_status="processed",
                ),
            ]
        )
        db.session.commit()

        after_new_records = build_own_assistant_prompt(
            "What does Jamie's latest report and scan show?", owner_id
        )

    before_patient_data = before_new_records.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    after_patient_data = after_new_records.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    assert "2026-02-01" not in before_patient_data
    assert "2026-02-01" in after_patient_data
    assert "HbA1c: 6.2 %" in after_patient_data
    assert "Scan (2026-02-02): MRI knee, processed" in after_patient_data


def test_own_assistant_project_retrieval_always_starts_with_overview_and_caps_chunks(
    local_assistant_workspace,
    monkeypatch,
    tmp_path,
):
    _client, app, owner_id, *_rest = local_assistant_workspace
    (tmp_path / "project_overview.md").write_text(
        "# Project overview\n\nAlways present overview details.", encoding="utf-8"
    )
    (tmp_path / "features.md").write_text(
        "# Features\n\nalpha feature details " * 4, encoding="utf-8"
    )
    (tmp_path / "api.md").write_text(
        "# API\n\nalpha endpoint details " * 4, encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text(
        "# Project notes\n\nAdd stable project facts here...\n",
        encoding="utf-8",
    )
    (tmp_path / "how_to_use.md").write_text(
        "# Setup\n\nInstall dependencies with pip install and git clone first.\n\n"
        "# Assistant\n\nThe assistant retrieves current patient reports.",
        encoding="utf-8",
    )
    monkeypatch.setattr(own_llm, "PROJECT_KB_DIRECTORY", tmp_path)

    with app.app_context():
        knowledge = own_llm._retrieve_project_knowledge("alpha", owner_id)

    chunks = knowledge.split("\n---\n")
    assert chunks[0].startswith("# Project overview")
    assert "Always present overview details." in chunks[0]
    assert len(chunks) <= 4
    assert "Add stable project facts here" not in knowledge
    assert "git clone" not in knowledge


def test_own_assistant_debug_logs_final_prompt(
    local_assistant_workspace,
    monkeypatch,
    caplog,
):
    _client, app, owner_id, *_rest = local_assistant_workspace
    monkeypatch.setenv("ASSISTANT_DEBUG", "1")
    with app.app_context(), caplog.at_level("INFO", logger=own_llm.__name__):
        project_prompt = build_own_assistant_prompt(
            "What does this project do?", owner_id
        )
        named_patient_prompt = build_own_assistant_prompt(
            "What is Jamie's latest HbA1c?", owner_id
        )
        high_hba1c_prompt = build_own_assistant_prompt(
            "Which patients have high HbA1c?", owner_id
        )

    assert "Own-model Assistant final prompt:" in caplog.text
    assert project_prompt in caplog.text
    assert named_patient_prompt in caplog.text
    assert high_hba1c_prompt in caplog.text
    assert caplog.text.count("Own-model Assistant final prompt:") == 3


def test_own_assistant_classifies_common_labs_using_ranges_not_ocr_flags():
    classify = own_llm._classify_lab_value

    assert 25 <= len(own_llm.STANDARD_LAB_RANGES) <= 27
    assert own_llm._canonical_test_name("HbAlc") == "HbA1c"
    assert own_llm._canonical_test_name("W.E.C") == "wbc"
    assert classify("HbA1c", "6.1", "%", "HbAlc 6.1% L")["status"] == "high"
    assert (
        classify("HbA1c", "5.5", "%", "HbAlc 5.5% H; reference 4.0-5.6")[
            "status"
        ]
        == "normal"
    )
    assert (
        classify("HbA1c", "6.1", "%", "HbA1c 6.1%; reference 0-999")["status"]
        == "high"
    )
    assert classify("fasting glucose", "6.0", "mmol/L", "")["status"] == "high"
    assert classify("wbc", "12", "10^9/L", "")["status"] == "high"


def test_own_assistant_uses_latest_report_with_three_recognized_labs(
    local_assistant_workspace,
):
    _client, app, owner_id, _other_id, patient_id, *_rest = (
        local_assistant_workspace
    )
    with app.app_context():
        db.session.add(
            MedicalReport(
                patient_id=patient_id,
                report_type="newer scan note",
                report_date=date(2026, 1, 1),
                extracted_text="No laboratory results in this document.",
            )
        )
        db.session.commit()
        prompt = build_own_assistant_prompt(
            "What is Jamie's latest HbA1c?", owner_id
        )

    patient_data = prompt.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    assert patient_data.startswith("Jamie Reed, latest report (2025-05-01):")
    assert "HbA1c: 5.4 % (normal)" in patient_data
    assert "WBC: 6.0 10^3/uL (normal)" in patient_data


def test_own_assistant_reports_when_named_patient_has_no_qualifying_lab_report(
    local_assistant_workspace,
):
    _client, app, owner_id, *_rest = local_assistant_workspace
    with app.app_context():
        patient = Patient(
            first_name="No",
            last_name="Labs",
            created_by_id=owner_id,
        )
        db.session.add(patient)
        db.session.flush()
        db.session.add(
            MedicalReport(
                patient_id=patient.id,
                report_type="discharge summary",
                report_date=date(2026, 2, 3),
                extracted_text="Follow-up guidance, no lab panel.",
            )
        )
        db.session.commit()
        prompt = build_own_assistant_prompt("What is No's HbA1c?", owner_id)

    patient_data = prompt.split("PATIENT DATA:\n", 1)[1].split(
        "\n\nQUESTION:", 1
    )[0]
    assert patient_data.startswith(
        "(no lab values found in this patient's reports; "
        "latest document is discharge summary, 2026-02-03)"
    )


def test_own_assistant_project_question_does_not_require_uploaded_research(
    local_assistant_workspace,
    monkeypatch,
):
    client, _app, _owner_id, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    monkeypatch.setattr(
        "backend.services.assistant_desks.build_own_assistant_prompt",
        lambda question, _user_id, **_options: (
            "PROJECT KNOWLEDGE:\nProject overview\n\n"
            "PATIENT DATA:\n(none)\n\n"
            f"QUESTION: {question}"
        ),
    )
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
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
        "backend.services.assistant_desks.build_own_assistant_prompt",
        lambda question, _user_id, web_search_results=None, **_options: (
            captured.update({"web_search_results": web_search_results})
            or f"PROJECT KNOWLEDGE:\n{question}"
        ),
    )
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
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
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: pytest.fail(
            "A failed web search must not trigger an ungrounded model answer."
        ),
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
    assert "questions about this project" in payload["answer"]
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


def test_fixed_patient_count_is_account_scoped_and_skips_model(
    local_assistant_workspace,
    monkeypatch,
):
    client, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    _use_bearer_token(client, monkeypatch, token)
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: pytest.fail(
            "Deterministic count intents must not invoke the model."
        ),
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "How many patients do we have?",
            "context_type": "research",
        },
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["intent"] == "counts_lists"
    assert response.get_json()["answer"] == "You have 1 patient(s)."


def test_missing_patient_selection_is_clear_and_skips_model(
    local_assistant_workspace,
    monkeypatch,
):
    client, *_rest = local_assistant_workspace
    token = _token(client, "local-assistant@example.com")
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: pytest.fail(
            "Missing-selection replies must not invoke the model."
        ),
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Tell me his timeline.",
            "context_type": "patient_timeline",
        },
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["intent"] == "timeline"
    assert response.get_json()["answer"] == "Please select a patient (and a report) first."


def test_lab_unit_conversions_cover_common_uploaded_formats():
    assert own_llm._convert_lab_value("wbc", 5831, "/uL") == pytest.approx(5.831)
    assert own_llm._convert_lab_value("platelets", 1.68, "lakh/uL") == pytest.approx(168)


def test_selected_report_without_readable_text_gets_clear_fixed_response(
    local_assistant_workspace,
    monkeypatch,
    tmp_path,
):
    client, app, _owner_id, _other_id, patient_id, report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    report_file = tmp_path / "unreadable.pdf"
    report_file.write_bytes(b"pdf fixture")
    with app.app_context():
        report = db.session.get(MedicalReport, report_id)
        report.extracted_text = None
        report.file_reference = "uploads/unreadable.pdf"
        db.session.commit()

    monkeypatch.setattr(
        "backend.routes.ai.get_stored_file_location",
        lambda _reference, _patient_id: (tmp_path, "unreadable.pdf"),
    )
    monkeypatch.setattr(
        "backend.routes.ai.extract_document_text",
        lambda _content, _filename: "",
    )
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: pytest.fail(
            "An unreadable selected report must not invoke the model."
        ),
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What is this report about?",
            "context_type": "report",
            "patient_id": patient_id,
            "report_id": report_id,
        },
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["answer"] == "This report has no readable text."


def test_scan_question_does_not_extract_irrelevant_report_text(
    local_assistant_workspace,
    monkeypatch,
):
    client, app, _owner_id, _other_id, patient_id, _report_id, _ = (
        local_assistant_workspace
    )
    token = _token(client, "local-assistant@example.com")
    with app.app_context():
        db.session.add(
            ScanAsset(
                patient_id=patient_id,
                modality="MRI",
                body_region="knee",
                original_filename="knee.png",
                file_reference="uploads/knee.png",
                processing_status="processed",
            )
        )
        db.session.commit()

    monkeypatch.setattr(
        "backend.routes.ai.extract_document_text",
        lambda *_args: pytest.fail(
            "Scan-only questions must not OCR patient reports."
        ),
    )
    monkeypatch.setattr(
        "backend.services.assistant_desks.own_generate_assistant",
        lambda *_args, **_kwargs: "The saved scan is an experimental model record.",
    )

    response = client.post(
        "/api/ai/query-local",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What do the scans say?",
            "context_type": "patient_timeline",
            "patient_id": patient_id,
        },
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["intent"] == "scans"
