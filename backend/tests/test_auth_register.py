import pytest

from backend.app import create_app
from backend.extensions import db
from backend.models.user import User


@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        with app.test_client() as test_client:
            yield test_client
        db.session.remove()
        db.drop_all()


def test_register_user_success(client):
    payload = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
        "password": "StrongPass123!",
    }

    response = client.post("/api/auth/register", json=payload)

    assert response.status_code == 201
    data = response.get_json()
    assert data["message"] == "User registered successfully"
    assert data["user"]["email"] == "john@example.com"

    saved_user = db.session.query(User).filter_by(email="john@example.com").first()
    assert saved_user is not None
    assert saved_user.password_hash != payload["password"]


def test_register_user_rejects_duplicate_email(client):
    payload = {
        "first_name": "Jane",
        "last_name": "Smith",
        "email": "jane@example.com",
        "password": "StrongPass123!",
    }

    client.post("/api/auth/register", json=payload)
    response = client.post("/api/auth/register", json=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Email already registered"
