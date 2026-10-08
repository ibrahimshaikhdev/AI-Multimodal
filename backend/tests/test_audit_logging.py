from io import BytesIO

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import AuditLog, User
from backend.services.audit_logging import record_audit_event


@pytest.fixture
def audit_client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        patient = User(
            first_name="Audit",
            last_name="Patient",
            email="audit-patient@example.com",
            password_hash=generate_password_hash("PatientPass123!"),
            role="patient",
        )
        admin = User(
            first_name="Audit",
            last_name="Admin",
            email="audit-admin@example.com",
            password_hash=generate_password_hash("AdminPass123!"),
            role="admin",
        )
        db.session.add_all([patient, admin])
        db.session.commit()
        patient_id = patient.id
        admin_id = admin.id

        with app.test_client() as client:
            yield client, patient_id, admin_id

        db.session.remove()
        db.drop_all()


def _login(client, email, password):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def test_audit_endpoint_requires_auth_and_scopes_non_admin_users(audit_client):
    client, patient_id, _ = audit_client
    patient_token = _login(client, "audit-patient@example.com", "PatientPass123!")
    admin_token = _login(client, "audit-admin@example.com", "AdminPass123!")

    patient_response = client.get(
        "/api/audit-logs",
        headers={"Authorization": f"Bearer {patient_token}"},
    )
    admin_response = client.get(
        "/api/audit-logs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert patient_response.status_code == 200
    patient_logs = patient_response.get_json()["logs"]
    assert len(patient_logs) == 1
    assert patient_logs[0]["actor"]["id"] == patient_id
    assert patient_logs[0]["action"] == "auth.login"
    assert patient_logs[0]["status"] == "success"
    assert patient_logs[0]["timestamp"]

    assert admin_response.status_code == 200
    assert len(admin_response.get_json()["logs"]) == 2
    assert client.get("/api/audit-logs").status_code == 401


def test_audit_records_failed_login_and_patient_creation(audit_client):
    client, patient_id, _ = audit_client
    patient_token = _login(client, "audit-patient@example.com", "PatientPass123!")

    failed_login = client.post(
        "/api/auth/login",
        json={"email": "audit-patient@example.com", "password": "WrongPassword"},
    )
    assert failed_login.status_code == 401

    created = client.post(
        "/api/patients",
        headers={"Authorization": f"Bearer {patient_token}"},
        json={"first_name": "New", "last_name": "Patient"},
    )
    assert created.status_code == 201

    with client.application.app_context():
        failure_event = (
            db.session.query(AuditLog)
            .filter_by(action="auth.login", status="failure")
            .one()
        )
        assert failure_event.actor_id == patient_id
        assert failure_event.metadata_json == {}
        assert (
            db.session.query(AuditLog)
            .filter_by(
                actor_id=patient_id,
                action="patient.create",
                resource_id=str(created.get_json()["patient"]["id"]),
            )
            .one()
        )

    listed = client.get(
        "/api/audit-logs?limit=1&offset=0",
        headers={"Authorization": f"Bearer {patient_token}"},
    )
    assert listed.status_code == 200
    assert len(listed.get_json()["logs"]) == 1


def test_audit_service_stores_resource_status_and_metadata(audit_client):
    client, patient_id, _ = audit_client
    with client.application.app_context():
        record_audit_event(
            actor_id=patient_id,
            action="ai.scan_analysis",
            resource_type="scan",
            resource_id=42,
            status="failure",
            metadata={"processing_status": "model_unavailable"},
        )
        event = db.session.query(AuditLog).filter_by(action="ai.scan_analysis").one()
        assert event.resource_id == "42"
        assert event.status == "failure"
        assert event.metadata_json == {"processing_status": "model_unavailable"}


def test_report_and_scan_views_and_uploads_are_audited(audit_client, monkeypatch):
    client, _, _ = audit_client
    token = _login(client, "audit-patient@example.com", "PatientPass123!")
    headers = {"Authorization": f"Bearer {token}"}
    patient_response = client.post(
        "/api/patients",
        headers=headers,
        json={"first_name": "Record", "last_name": "Owner"},
    )
    patient_id = patient_response.get_json()["patient"]["id"]

    monkeypatch.setattr(
        "backend.routes.reports.save_uploaded_file",
        lambda _file, patient_id: f"uploads/patients/{patient_id}/audit-report.pdf",
    )
    monkeypatch.setattr(
        "backend.routes.reports.extract_document_text",
        lambda *_args: "Report text is not audit metadata.",
    )
    report_response = client.post(
        "/api/reports/upload",
        headers=headers,
        data={
            "patient_id": str(patient_id),
            "file": (BytesIO(b"%PDF-1.4\nreport bytes"), "audit-report.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert report_response.status_code == 201
    report_id = report_response.get_json()["report"]["id"]

    monkeypatch.setattr(
        "backend.routes.scans.save_uploaded_file",
        lambda _file, patient_id: f"uploads/patients/{patient_id}/audit-scan.png",
    )
    scan_response = client.post(
        "/api/scans/upload",
        headers=headers,
        data={
            "patient_id": str(patient_id),
            "modality": "OTHER",
            "body_region": "Other",
            "file": (BytesIO(b"\x89PNG\r\n\x1a\nscan"), "audit-scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert scan_response.status_code == 201
    scan_id = scan_response.get_json()["scan"]["id"]

    assert client.get(
        f"/api/reports?patient_id={patient_id}",
        headers=headers,
    ).status_code == 200
    assert client.get(
        f"/api/scans?patient_id={patient_id}",
        headers=headers,
    ).status_code == 200

    with client.application.app_context():
        report_event = (
            db.session.query(AuditLog)
            .filter_by(action="report.upload", resource_id=str(report_id))
            .one()
        )
        scan_event = (
            db.session.query(AuditLog)
            .filter_by(action="scan.upload", resource_id=str(scan_id))
            .one()
        )
        assert report_event.metadata_json == {
            "patient_id": patient_id,
            "report_type": "general",
            "processing_status": "processed",
        }
        assert scan_event.metadata_json["modality"] == "OTHER"
        assert "Report text" not in str(report_event.metadata_json)
        assert (
            db.session.query(AuditLog)
            .filter_by(action="report.list", resource_id=str(patient_id))
            .one()
        )
        assert (
            db.session.query(AuditLog)
            .filter_by(action="scan.list", resource_id=str(patient_id))
            .one()
        )


@pytest.mark.parametrize(
    "query",
    ["/api/audit-logs?limit=0", "/api/audit-logs?limit=abc", "/api/audit-logs?offset=-1"],
)
def test_audit_endpoint_validates_pagination(audit_client, query):
    client, _, _ = audit_client
    token = _login(client, "audit-patient@example.com", "PatientPass123!")
    response = client.get(query, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 400
