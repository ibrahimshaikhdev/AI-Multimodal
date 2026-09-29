import pytest
import jwt

from backend.app import create_app
from backend.extensions import db
from backend.models.revoked_token import RevokedToken
from backend.models.user import User
from werkzeug.security import generate_password_hash


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


def test_logout_revokes_authenticated_token(client):
    token = login(client)

    response = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.get_json()["message"] == "Logged out successfully"

    claims = jwt.decode(
        token,
        client.application.config["SECRET_KEY"],
        algorithms=["HS256"],
    )
    with client.application.app_context():
        assert db.session.get(RevokedToken, claims["jti"]) is not None

    repeated_response = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert repeated_response.status_code == 401


def test_logout_requires_bearer_token(client):
    response = client.post("/api/auth/logout")

    assert response.status_code == 401
    assert response.get_json()["error"] == "Authentication required"


def test_logout_rejects_invalid_token(client):
    response = client.post(
        "/api/auth/logout",
        headers={"Authorization": "Bearer not-a-valid-token"},
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "Invalid or expired token"
