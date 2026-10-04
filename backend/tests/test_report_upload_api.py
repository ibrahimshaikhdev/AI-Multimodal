from datetime import date
from io import BytesIO

import fitz
import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models.patient import Patient
from backend.models.report_parameter import ReportParameter
from backend.models.user import User
from backend.services.parameter_persistence import persist_report_parameters


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
