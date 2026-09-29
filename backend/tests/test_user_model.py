import pytest

from backend.app import create_app
from backend.extensions import db
from backend.models.user import User


@pytest.fixture
def app():
    app = create_app(testing=True)

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def test_user_model_can_be_created(app):
    user = User(
        first_name="Alice",
        last_name="Brown",
        email="alice@example.com",
        password_hash="hashed_password_value",
        role="admin",
        is_active=True,
    )

    db.session.add(user)
    db.session.commit()

    saved_user = db.session.get(User, user.id)
    assert saved_user is not None
    assert saved_user.email == "alice@example.com"
    assert saved_user.role == "admin"
    assert saved_user.is_active is True
