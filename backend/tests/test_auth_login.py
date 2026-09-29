import pytest
import jwt

from backend.app import create_app
from backend.extensions import db
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


def test_login_returns_access_token_for_valid_credentials(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "SAM@example.com", "password": "CorrectPass123!"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["access_token"]
    assert data["token_type"] == "Bearer"
    assert data["user"]["email"] == "sam@example.com"
    assert "password_hash" not in data["user"]

    claims = jwt.decode(
        data["access_token"],
        client.application.config["SECRET_KEY"],
        algorithms=["HS256"],
    )
    assert claims["sub"] == str(data["user"]["id"])
    assert claims["jti"]
    assert claims["email"] == "sam@example.com"
    assert claims["role"] == "patient"


def test_login_rejects_wrong_password(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "sam@example.com", "password": "WrongPass123!"},
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "Invalid email or password"


def test_login_rejects_unknown_email_with_same_error(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "unknown@example.com", "password": "WrongPass123!"},
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "Invalid email or password"


def test_login_requires_email_and_password(client):
    response = client.post("/api/auth/login", json={"email": "sam@example.com"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Email and password are required"
