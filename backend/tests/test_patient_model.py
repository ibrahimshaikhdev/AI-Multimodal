from datetime import date

import pytest

from backend.app import create_app
from backend.extensions import db
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


def test_patient_can_be_saved_with_creator_relationship(app):
    user = User(
        first_name="Casey",
        last_name="Clinician",
        email="casey@example.com",
        password_hash="hashed_password_value",
        role="clinician",
    )
    patient = Patient(
        first_name="Jordan",
        last_name="Lee",
        date_of_birth=date(1990, 4, 12),
        created_by=user,
    )

    db.session.add(patient)
    db.session.commit()

    saved_patient = db.session.get(Patient, patient.id)
    assert saved_patient is not None
    assert saved_patient.first_name == "Jordan"
    assert saved_patient.date_of_birth == date(1990, 4, 12)
    assert saved_patient.created_by.email == "casey@example.com"
    assert saved_patient in user.patients
