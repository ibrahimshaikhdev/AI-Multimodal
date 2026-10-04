from datetime import date

import pytest

from backend.app import create_app
from backend.extensions import db
from backend.models.medical_report import MedicalReport
from backend.models.patient import Patient
from backend.models.user import User


@pytest.fixture
def app():
    app = create_app(testing=True)

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def test_medical_report_can_be_saved_with_patient_relationship(app):
    user = User(
        first_name="Dr.",
        last_name="Smith",
        email="doctor@example.com",
        password_hash="hashed_password_value",
        role="clinician",
    )
    patient = Patient(
        first_name="Jordan",
        last_name="Lee",
        date_of_birth=date(1990, 4, 12),
        created_by=user,
    )
    report = MedicalReport(
        patient=patient,
        report_type="lab_results",
        report_date=date(2024, 1, 15),
        file_reference="uploads/jordan-lab-results.pdf",
        extracted_text="Blood pressure normal. Vitamin D low.",
        processing_status="uploaded",
    )

    db.session.add(report)
    db.session.commit()

    saved_report = db.session.get(MedicalReport, report.id)
    assert saved_report is not None
    assert saved_report.patient_id == patient.id
    assert saved_report.report_type == "lab_results"
    assert saved_report.report_date == date(2024, 1, 15)
    assert saved_report.file_reference == "uploads/jordan-lab-results.pdf"
    assert saved_report.processing_status == "uploaded"
    assert saved_report.extracted_text == "Blood pressure normal. Vitamin D low."
    assert saved_report in patient.reports
