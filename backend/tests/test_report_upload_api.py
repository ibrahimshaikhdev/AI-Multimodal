from datetime import date
from io import BytesIO

import fitz
import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models.medical_report import MedicalReport
from backend.models.patient import Patient
from backend.models.report_parameter import ReportParameter
from backend.models.user import User
from backend.models import AuditLog
from backend.services.parameter_persistence import persist_report_parameters
from backend.services.file_storage import get_stored_file_location
from backend.services.ai_service import AIServiceTimeout, AIServiceUnavailable


@pytest.fixture
def upload_client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        owner = User(
            first_name="Owner",
            last_name="Account",
            email="owner@example.com",
            password_hash=generate_password_hash("OwnerPass123!"),
            role="patient",
        )
        other_user = User(
            first_name="Other",
            last_name="Account",
            email="other@example.com",
            password_hash=generate_password_hash("OtherPass123!"),
            role="patient",
        )
        db.session.add_all([owner, other_user])
        db.session.flush()

        owner_patient = Patient(
            first_name="Morgan",
            last_name="Reed",
            date_of_birth=date(1985, 2, 3),
            created_by=owner,
        )
        other_patient = Patient(
            first_name="Taylor",
            last_name="Quinn",
            created_by=other_user,
        )
        db.session.add_all([owner_patient, other_patient])
        db.session.commit()

        with app.test_client() as client:
            yield client, owner_patient.id, other_patient.id

        db.session.remove()
        db.drop_all()


def login(client, email="owner@example.com", password="OwnerPass123!"):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    return response.get_json()["access_token"]


def make_text_pdf(text):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_report_upload_requires_auth_and_valid_patient(upload_client):
    client, owner_patient_id, other_patient_id = upload_client
    token = login(client)

    upload_response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "report_type": "lab_results",
            "report_date": "2024-01-15",
            "file": (BytesIO(make_text_pdf("Hemoglobin  :  14.2 g/dL")), "lab-report.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert upload_response.status_code == 201
    payload = upload_response.get_json()["report"]
    assert payload["patient_id"] == owner_patient_id
    assert payload["report_type"] == "lab_results"
    assert payload["processing_status"] == "processed"
    assert payload["extracted_text"] == "Hemoglobin: 14.2 g/dL"
    assert payload["file_reference"].startswith(f"uploads/patients/{owner_patient_id}/")

    unauthorized_response = client.post("/api/reports/upload", data={"patient_id": owner_patient_id})
    assert unauthorized_response.status_code == 401

    forbidden_response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": other_patient_id,
            "file": (BytesIO(b"%PDF-1.4\n% test pdf"), "other.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert forbidden_response.status_code == 404
    assert forbidden_response.get_json()["error"] == "Patient not found"


def test_report_list_returns_saved_extracted_text_for_owned_patient(upload_client):
    client, owner_patient_id, other_patient_id = upload_client
    token = login(client)

    upload_response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "file": (BytesIO(make_text_pdf("Patient: Morgan Reed")), "report.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 201

    response = client.get(
        f"/api/reports?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    reports = response.get_json()["reports"]
    assert len(reports) == 1
    assert reports[0]["extracted_text"] == "Patient: Morgan Reed"
    assert reports[0]["processing_status"] == "processed"

    forbidden_response = client.get(
        f"/api/reports?patient_id={other_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert forbidden_response.status_code == 404


def _create_report_with_text(patient_id, text):
    report = MedicalReport(
        patient_id=patient_id,
        report_type="lab_results",
        extracted_text=text,
        processing_status="processed" if text else "no_text_found",
    )
    db.session.add(report)
    db.session.commit()
    return report.id


def _create_comparison_report(patient_id, text, report_date=None):
    report = MedicalReport(
        patient_id=patient_id,
        report_type="lab_results",
        report_date=report_date,
        extracted_text=text,
        processing_status="processed" if text else "no_text_found",
    )
    db.session.add(report)
    db.session.commit()
    return report.id


def test_report_text_extraction_reuses_existing_text(upload_client, monkeypatch):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    source_text = "Saved extracted report text."
    with client.application.app_context():
        report_id = _create_comparison_report(owner_patient_id, source_text)

    def unexpected_ocr(*_args):
        raise AssertionError("Existing extracted text must be reused without OCR")

    monkeypatch.setattr("backend.routes.reports.extract_document_text", unexpected_ocr)
    unauthenticated = client.post(f"/api/reports/{report_id}/extract-text")
    response = client.post(
        f"/api/reports/{report_id}/extract-text",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert response.get_json()["extracted_text"] == source_text
    assert response.get_json()["reused"] is True


def test_report_text_extraction_reuses_existing_ocr_service_for_missing_text(
    upload_client,
    monkeypatch,
    tmp_path,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    stored_file = tmp_path / "stored-report.pdf"
    stored_file.write_bytes(b"stored report bytes")
    with client.application.app_context():
        report = MedicalReport(
            patient_id=owner_patient_id,
            report_type="lab_results",
            file_reference=f"uploads/patients/{owner_patient_id}/stored-report.pdf",
            extracted_text=None,
            processing_status="extraction_failed",
        )
        db.session.add(report)
        db.session.commit()
        report_id = report.id

    monkeypatch.setattr(
        "backend.routes.reports.get_stored_file_location",
        lambda _reference, _patient_id: (tmp_path, "stored-report.pdf"),
    )
    ocr_calls = []

    def existing_ocr_service(document_bytes, filename):
        ocr_calls.append((document_bytes, filename))
        return "Hemoglobin: 14.2 g/dL"

    monkeypatch.setattr(
        "backend.routes.reports.extract_document_text",
        existing_ocr_service,
    )
    response = client.post(
        f"/api/reports/{report_id}/extract-text",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.get_json()["extracted_text"] == "Hemoglobin: 14.2 g/dL"
    assert response.get_json()["reused"] is False
    assert ocr_calls == [(b"stored report bytes", "stored-report.pdf")]
    with client.application.app_context():
        saved_report = db.session.get(MedicalReport, report_id)
        assert saved_report.extracted_text == "Hemoglobin: 14.2 g/dL"
        assert saved_report.processing_status == "processed"


def test_report_comparison_uses_saved_text_and_returns_report_details(
    upload_client,
    monkeypatch,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    text_1 = "Study date: 2024-01-15\nHemoglobin: 12.1 g/dL"
    text_2 = "Study date: 2025-01-15\nHemoglobin: 13.4 g/dL"
    with client.application.app_context():
        report_id_1 = _create_comparison_report(
            owner_patient_id, text_1, date(2024, 1, 15)
        )
        report_id_2 = _create_comparison_report(
            owner_patient_id, text_2, date(2025, 1, 15)
        )

    class UnusedAIService:
        def compare(self, *_args):
            raise AssertionError("Report comparison must not call a generative AI service")

    monkeypatch.setitem(client.application.extensions, "ai_service", UnusedAIService())
    monkeypatch.setattr(
        client.application.extensions["research_rag_service"],
        "similarity_matrix",
        lambda first, second: [[0.1 for _ in second] for _ in first],
    )
    response = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id_1, "report_id_2": report_id_2},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert [report["id"] for report in payload["reports"]] == [
        report_id_1,
        report_id_2,
    ]
    assert [report["report_date"] for report in payload["reports"]] == [
        "2024-01-15",
        "2025-01-15",
    ]
    assert payload["comparison"]["parameters"][0]["label"] == "Hemoglobin"
    assert payload["comparison"]["parameters"][0]["status"] == "different_value"
    assert payload["disclaimer"].startswith("Local, rule-based")


def test_report_comparison_requires_authentication_and_two_valid_ids(upload_client):
    client, owner_patient_id, _ = upload_client
    with client.application.app_context():
        report_id_1 = _create_comparison_report(owner_patient_id, "Report one.")
        report_id_2 = _create_comparison_report(owner_patient_id, "Report two.")

    unauthenticated = client.post(
        "/api/reports/compare",
        json={"report_id_1": report_id_1, "report_id_2": report_id_2},
    )
    assert unauthenticated.status_code == 401

    token = login(client)
    invalid = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id_1},
    )
    duplicate = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id_1, "report_id_2": report_id_1},
    )
    assert invalid.status_code == 400
    assert duplicate.status_code == 400


def test_report_comparison_hides_missing_or_unowned_reports(upload_client):
    client, owner_patient_id, other_patient_id = upload_client
    token = login(client)
    other_token = login(client, "other@example.com", "OtherPass123!")
    with client.application.app_context():
        report_id = _create_comparison_report(owner_patient_id, "Owned report.")
        other_report_id = _create_comparison_report(other_patient_id, "Other report.")

    missing = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id, "report_id_2": 999999},
    )
    unowned = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id, "report_id_2": other_report_id},
    )
    assert missing.status_code == 404
    assert unowned.status_code == 404


def test_report_comparison_allows_reports_from_different_owned_patients(
    upload_client,
    monkeypatch,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    text_1 = "First patient's saved OCR text."
    text_2 = "Second patient's saved OCR text."
    with client.application.app_context():
        owner = db.session.query(User).filter_by(email="owner@example.com").one()
        second_patient = Patient(
            first_name="Second",
            last_name="Patient",
            created_by=owner,
        )
        db.session.add(second_patient)
        db.session.flush()
        second_patient_id = second_patient.id
        report_id_1 = _create_comparison_report(owner_patient_id, text_1)
        report_id_2 = _create_comparison_report(second_patient_id, text_2)

    class UnusedAIService:
        def compare(self, *_args):
            raise AssertionError("Report comparison must not call a generative AI service")

    monkeypatch.setitem(client.application.extensions, "ai_service", UnusedAIService())
    monkeypatch.setattr(
        client.application.extensions["research_rag_service"],
        "similarity_matrix",
        lambda first, second: [[0.1 for _ in second] for _ in first],
    )

    response = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id_1, "report_id_2": report_id_2},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["comparison"]["method"] == "local_rules_and_embeddings"
    assert [report["patient_id"] for report in payload["reports"]] == [
        owner_patient_id,
        second_patient_id,
    ]
    assert [report["patient_name"] for report in payload["reports"]] == [
        "Morgan Reed",
        "Second Patient",
    ]


def test_report_comparison_rejects_missing_saved_extracted_text(
    upload_client,
    monkeypatch,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    with client.application.app_context():
        report_id_1 = _create_comparison_report(owner_patient_id, "Has text.")
        report_id_2 = _create_comparison_report(owner_patient_id, None)

    response = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id_1, "report_id_2": report_id_2},
    )
    assert response.status_code == 422
    assert "usable extracted text" in response.get_json()["error"]


def test_report_comparison_does_not_depend_on_ai_provider(upload_client, monkeypatch):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    with client.application.app_context():
        report_id_1 = _create_comparison_report(owner_patient_id, "Hemoglobin: 12.0 g/dL")
        report_id_2 = _create_comparison_report(owner_patient_id, "Hemoglobin: 12.4 g/dL")

    class UnavailableAIService:
        def compare(self, *_args):
            raise AssertionError("Report comparison must work without the AI provider")

    monkeypatch.setitem(client.application.extensions, "ai_service", UnavailableAIService())
    response = client.post(
        "/api/reports/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"report_id_1": report_id_1, "report_id_2": report_id_2},
    )
    assert response.status_code == 200
    assert response.get_json()["comparison"]["parameters"][0]["status"] == "different_value"


def test_report_summary_uses_saved_extracted_text_and_returns_disclaimer(
    upload_client,
    monkeypatch,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    source_text = (
        "Study date: 2025-05-14\n"
        "Findings: Hemoglobin 14.2 g/dL.\n"
        "Recommendation: Repeat testing in six months."
    )
    with client.application.app_context():
        report_id = _create_report_with_text(owner_patient_id, source_text)

    class FakeAIService:
        MAX_INPUT_CHARACTERS = 20000

        def summarize(self, text):
            assert text == source_text
            return (
                "Reported findings: Hemoglobin 14.2 g/dL.\n"
                "Measurements and dates: 14.2 g/dL; 2025-05-14.\n"
                "Recommendations: Repeat testing in six months.\n"
                "Limitations: Based only on the supplied report."
            )

    monkeypatch.setitem(
        client.application.extensions,
        "ai_service",
        FakeAIService(),
    )
    response = client.post(
        f"/api/reports/{report_id}/summary",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["report_id"] == report_id
    assert "Hemoglobin 14.2 g/dL" in payload["summary"]
    assert "Recommendations:" in payload["summary"]
    assert payload["disclaimer"] == (
        "AI-generated summary for informational purposes; requires human review. "
        "This is not a diagnosis."
    )

    with client.application.app_context():
        audit_event = (
            db.session.query(AuditLog)
            .filter_by(
                action="ai.report_summary",
                resource_id=str(report_id),
                status="success",
            )
            .one()
        )
        assert audit_event.metadata_json == {}


def test_report_summary_requires_authentication(upload_client):
    client, owner_patient_id, _ = upload_client
    with client.application.app_context():
        report_id = _create_report_with_text(owner_patient_id, "Actual report text.")

    response = client.post(f"/api/reports/{report_id}/summary")
    assert response.status_code == 401
    assert response.get_json()["error"] == "Authentication required"


def test_report_summary_returns_not_found_for_missing_or_unowned_report(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    other_token = login(client, "other@example.com", "OtherPass123!")
    with client.application.app_context():
        report_id = _create_report_with_text(owner_patient_id, "Actual report text.")

    missing = client.post(
        "/api/reports/999999/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    unowned = client.post(
        f"/api/reports/{report_id}/summary",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert missing.status_code == 404
    assert missing.get_json()["error"] == "Report not found"
    assert unowned.status_code == 404
    assert unowned.get_json()["error"] == "Report not found"


def test_report_summary_rejects_missing_extracted_text(upload_client, monkeypatch):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    with client.application.app_context():
        report_id = _create_report_with_text(owner_patient_id, None)

    class UnexpectedAIService:
        MAX_INPUT_CHARACTERS = 20000

        def summarize(self, _text):
            raise AssertionError("AI must not be called without extracted text")

    monkeypatch.setitem(
        client.application.extensions,
        "ai_service",
        UnexpectedAIService(),
    )
    response = client.post(
        f"/api/reports/{report_id}/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
    assert "no extracted text" in response.get_json()["error"].lower()


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [(AIServiceUnavailable(), 503), (AIServiceTimeout(), 504)],
)
def test_report_summary_handles_local_ai_service_failures(
    upload_client,
    monkeypatch,
    failure,
    expected_status,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    with client.application.app_context():
        report_id = _create_report_with_text(owner_patient_id, "Actual report text.")

    class FailedAIService:
        MAX_INPUT_CHARACTERS = 20000

        def summarize(self, _text):
            raise failure

    monkeypatch.setitem(
        client.application.extensions,
        "ai_service",
        FailedAIService(),
    )
    response = client.post(
        f"/api/reports/{report_id}/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == expected_status
    assert response.get_json()["success"] is False
    assert response.get_json()["error"] == str(failure)


def test_report_summary_rejects_request_body_instead_of_accepting_client_text(
    upload_client,
):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    with client.application.app_context():
        report_id = _create_report_with_text(owner_patient_id, "Saved report text.")

    response = client.post(
        f"/api/reports/{report_id}/summary",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "Client-supplied report text"},
    )
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_report_upload_returns_source_grounded_metadata(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    pdf_text = (
        "Study Date: 2025-05-14\n"
        "Findings:\nMild joint effusion.\nNo acute fracture.\n"
        "Impression:\nMild joint effusion; no acute fracture."
    )

    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "file": (BytesIO(make_text_pdf(pdf_text)), "knee-mri.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    report = response.get_json()["report"]
    assert report["metadata"]["document_date"] == "2025-05-14"
    assert report["metadata"]["sections"] == [
        {"name": "findings", "text": "Mild joint effusion.\nNo acute fracture."},
        {"name": "impression", "text": "Mild joint effusion; no acute fracture."},
    ]

    history_response = client.get(
        f"/api/reports?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert history_response.status_code == 200
    assert history_response.get_json()["reports"][0]["metadata"] == report["metadata"]


def test_report_original_file_is_authenticated_and_owner_scoped(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    pdf_bytes = make_text_pdf("Lab report source")
    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "file": (BytesIO(pdf_bytes), "lab-report.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    report = response.get_json()["report"]
    assert report["original_url"] == f"/api/reports/{report['id']}/original"

    original = client.get(
        report["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert original.status_code == 200
    assert original.data == pdf_bytes
    assert original.mimetype == "application/pdf"

    unauthorized = client.get(report["original_url"])
    assert unauthorized.status_code == 401

    other_token = login(client, "other@example.com", "OtherPass123!")
    cross_owner = client.get(
        report["original_url"],
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert cross_owner.status_code == 404


def test_report_delete_is_owner_scoped_and_removes_uploaded_file(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "report_type": "lab_results",
            "file": (BytesIO(make_text_pdf("Hemoglobin 13 g/dL")), "lab-report.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    report = response.get_json()["report"]
    location = get_stored_file_location(report["file_reference"], owner_patient_id)
    assert location is not None
    upload_root, relative_path = location
    stored_file = upload_root / relative_path
    assert stored_file.is_file()

    other_token = login(client, "other@example.com", "OtherPass123!")
    forbidden = client.delete(
        f"/api/reports/{report['id']}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert forbidden.status_code == 404
    assert stored_file.is_file()

    deleted = client.delete(
        f"/api/reports/{report['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert deleted.status_code == 200
    assert deleted.get_json()["deleted"] is True
    assert not stored_file.exists()

    listing = client.get(
        f"/api/reports?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listing.get_json()["reports"] == []


def test_report_upload_returns_extracted_parameter_values(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    pdf_text = "Study Date: 2025-04-16\nHemoglobin: 14.2 g/dL\nWBC 7.1 x10^9/L"

    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "file": (BytesIO(make_text_pdf(pdf_text)), "lab-report.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    report = response.get_json()["report"]
    assert report["parameters"] == [
        {
            "parameter": "Hemoglobin",
            "value": "14.2",
            "unit": "g/dL",
            "date": "2025-04-16",
            "confidence": None,
            "source": "Hemoglobin: 14.2 g/dL",
        },
        {
            "parameter": "WBC",
            "value": "7.1",
            "unit": "x10^9/L",
            "date": "2025-04-16",
            "confidence": None,
            "source": "WBC: 7.1 x10^9/L",
        },
    ]


def test_report_upload_persists_and_deduplicates_parameter_rows(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    pdf_text = "Hemoglobin: 14.2 g/dL\nHemoglobin: 14.2 g/dL"

    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "file": (BytesIO(make_text_pdf(pdf_text)), "duplicate-labs.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    report = response.get_json()["report"]
    stored_rows = db.session.query(ReportParameter).filter_by(report_id=report["id"]).all()
    assert len(stored_rows) == 1
    assert stored_rows[0].parameter == "Hemoglobin"
    assert stored_rows[0].value == "14.2"
    assert stored_rows[0].unit == "g/dL"
    assert stored_rows[0].source == "Hemoglobin: 14.2 g/dL"
    assert len(report["parameters"]) == 1
    assert persist_report_parameters(report["id"], report["parameters"]) == 0

    history_response = client.get(
        f"/api/reports?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert history_response.status_code == 200
    assert history_response.get_json()["reports"][0]["parameters"] == report["parameters"]


def test_report_upload_saves_clear_status_when_no_text_is_found(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "file": (BytesIO(pdf_bytes), "blank.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    report = response.get_json()["report"]
    assert report["processing_status"] == "no_text_found"
    assert report["extracted_text"] is None
    assert "no readable text" in report["processing_message"].lower()


def test_report_upload_rejects_unsupported_file_types(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)

    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "report_type": "summary",
            "file": (BytesIO(b"not a real file"), "bad.exe"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "unsupported" in response.get_json()["error"].lower()


def test_report_upload_rejects_future_report_date(upload_client):
    client, owner_patient_id, _ = upload_client
    token = login(client)

    future_date = "2099-12-31"
    response = client.post(
        "/api/reports/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "report_type": "lab_results",
            "report_date": future_date,
            "file": (BytesIO(b"%PDF-1.4\n% test pdf"), "future-report.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "future" in response.get_json()["error"].lower()
