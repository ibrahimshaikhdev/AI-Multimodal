import pytest
from flask import jsonify
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.auth import require_roles
from backend.extensions import db
from backend.models.user import User


@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        db.session.add_all(
            [
                User(
                    first_name="Pat",
                    last_name="User",
                    email="patient@example.com",
                    password_hash=generate_password_hash("PatientPass123!"),
                    role="patient",
                ),
                User(
                    first_name="Ada",
                    last_name="Admin",
                    email="admin@example.com",
                    password_hash=generate_password_hash("AdminPass123!"),
                    role="admin",
                ),
            ]
        )
        db.session.commit()

        def admin_area():
            return jsonify({"message": "admin access granted"})

        app.add_url_rule(
            "/api/test/admin",
            view_func=require_roles("admin")(admin_area),
        )

        with app.test_client() as test_client:
            yield test_client

        db.session.remove()
        db.drop_all()


def get_access_token(client, email, password):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    return response.get_json()["access_token"]


def test_admin_role_can_access_restricted_route(client):
    token = get_access_token(client, "admin@example.com", "AdminPass123!")

    response = client.get(
        "/api/test/admin",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.get_json()["message"] == "admin access granted"


def test_patient_role_is_forbidden_from_admin_route(client):
    token = get_access_token(client, "patient@example.com", "PatientPass123!")

    response = client.get(
        "/api/test/admin",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "Insufficient permissions"


def test_admin_route_rejects_unauthenticated_request(client):
    response = client.get("/api/test/admin")

    assert response.status_code == 401
    assert response.get_json()["error"] == "Authentication required"


def test_registration_cannot_assign_an_admin_role(client):
    response = client.post(
        "/api/auth/register",
        json={
            "first_name": "New",
            "last_name": "User",
            "email": "new@example.com",
            "password": "NewUserPass123!",
            "role": "admin",
        },
    )

    assert response.status_code == 201
    assert response.get_json()["user"]["role"] == "patient"
