from sqlalchemy import text

from backend.app import create_app
from backend.extensions import db


def test_create_app_initializes_default_sqlite_tables(monkeypatch, tmp_path):
    sqlite_path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{sqlite_path}")

    app = create_app()
    with app.app_context():
        table_names = {
            row[0]
            for row in db.session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).all()
        }

    assert "medical_reports" in table_names
    assert "patients" in table_names
    assert "users" in table_names
