import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import AuditLog, User


@pytest.fixture
def profile_client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        user = User(
            first_name="Profile",
            last_name="User",
            email="profile@example.com",
            password_hash=generate_password_hash("ProfilePass123!"),
            role="patient",
        )
        db.session.add(user)
        db.session.commit()
        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()


def _token(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "profile@example.com", "password": "ProfilePass123!"},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def test_profile_update_changes_only_name_and_records_audit(profile_client):
    token = _token(profile_client)
    response = profile_client.put(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "first_name": "  New ",
            "last_name": " Name ",
            "email": "changed@example.com",
            "role": "admin",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["user"] == {
        "id": 1,
        "first_name": "New",
        "last_name": "Name",
        "email": "profile@example.com",
    }
    with profile_client.application.app_context():
        updated_user = db.session.get(User, 1)
        assert updated_user.email == "profile@example.com"
        assert updated_user.role == "patient"
        assert (
            db.session.query(AuditLog)
            .filter_by(action="profile.update", actor_id=1, resource_id="1")
            .one()
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"first_name": "Only"},
        {"first_name": " ", "last_name": "User"},
        {"first_name": "A" * 81, "last_name": "User"},
    ],
)
def test_profile_update_validates_names(profile_client, payload):
    token = _token(profile_client)
    response = profile_client.put(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert response.status_code == 400


def test_profile_update_requires_authentication(profile_client):
    response = profile_client.put(
        "/api/auth/me",
        json={"first_name": "New", "last_name": "Name"},
    )
    assert response.status_code == 401
