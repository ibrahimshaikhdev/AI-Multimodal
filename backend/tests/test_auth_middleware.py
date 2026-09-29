import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models.user import User


@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        user = User(
            first_name="Sam",
            last_name="Taylor",
            email="sam@example.com",
            password_hash=generate_password_hash("CorrectPass123!"),
            role="patient",
            is_active=True,
        )
        db.session.add(user)
        db.session.commit()

        with app.test_client() as test_client:
            yield test_client

        db.session.remove()
        db.drop_all()


def login(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "sam@example.com", "password": "CorrectPass123!"},
    )
    return response.get_json()["access_token"]


def test_protected_route_rejects_missing_token(client):
    response = client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.get_json()["error"] == "Authentication required"


def test_protected_route_accepts_valid_token(client):
    token = login(client)

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.get_json()["user"]["email"] == "sam@example.com"


def test_protected_route_rejects_revoked_token(client):
    token = login(client)
    logout_response = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert logout_response.status_code == 200

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "Invalid or expired token"
