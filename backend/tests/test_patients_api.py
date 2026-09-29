from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models.patient import Patient
from backend.models.user import User


@pytest.fixture
def patient_client():
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


def test_patient_create_list_and_detail_are_owner_scoped(patient_client):
    client, owner_patient_id, other_patient_id = patient_client
    token = login(client)

    create_response = client.post(
        "/api/patients",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "first_name": "Jamie",
            "last_name": "Park",
            "date_of_birth": "1994-06-17",
        },
    )

    assert create_response.status_code == 201
    created = create_response.get_json()["patient"]
    assert created["first_name"] == "Jamie"
    assert created["date_of_birth"] == "1994-06-17"
    assert "created_by_id" not in created

    list_response = client.get(
        "/api/patients",
        headers={"Authorization": f"Bearer {token}"},
    )
    listed_ids = {patient["id"] for patient in list_response.get_json()["patients"]}
    assert owner_patient_id in listed_ids
    assert created["id"] in listed_ids
    assert other_patient_id not in listed_ids

    detail_response = client.get(
        f"/api/patients/{created['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail_response.status_code == 200
    assert detail_response.get_json()["patient"]["last_name"] == "Park"


def test_patient_update_requires_both_names(patient_client):
    client, owner_patient_id, _ = patient_client
    token = login(client)

    response = client.put(
        f"/api/patients/{owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "first_name": "Morgan",
            "last_name": "Reed-Smith",
            "date_of_birth": None,
        },
    )

    assert response.status_code == 200
    patient = response.get_json()["patient"]
    assert patient["last_name"] == "Reed-Smith"
    assert patient["date_of_birth"] is None

    incomplete_response = client.put(
        f"/api/patients/{owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={"first_name": "Morgan"},
    )
    assert incomplete_response.status_code == 400


def test_patient_endpoint_rejects_invalid_date_and_missing_auth(patient_client):
    client, _, _ = patient_client
    token = login(client)

    invalid_date_response = client.post(
        "/api/patients",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "first_name": "Jamie",
            "last_name": "Park",
            "date_of_birth": "17-06-1994",
        },
    )
    assert invalid_date_response.status_code == 400

    unauthorized_response = client.get("/api/patients")
    assert unauthorized_response.status_code == 401


def test_user_cannot_read_or_update_another_users_patient(patient_client):
    client, _, other_patient_id = patient_client
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}

    get_response = client.get(f"/api/patients/{other_patient_id}", headers=headers)
    update_response = client.put(
        f"/api/patients/{other_patient_id}",
        headers=headers,
        json={"first_name": "Changed", "last_name": "Name"},
    )

    assert get_response.status_code == 404
    assert update_response.status_code == 404
    assert get_response.get_json()["error"] == "Patient not found"
