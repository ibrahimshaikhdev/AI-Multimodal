from pathlib import Path

from sqlalchemy.engine import make_url

from backend.config.database import resolve_database_url


def test_default_database_uses_stable_instance_database(tmp_path):
    project_root = tmp_path
    database_path = project_root / "instance" / "medical_ai.db"

    resolved = make_url(
        resolve_database_url(
            None,
            project_root=project_root,
            default_sqlite_path=database_path,
        )
    )

    assert resolved.drivername == "sqlite"
    assert Path(resolved.database) == database_path.resolve()


def test_missing_legacy_relative_database_reuses_existing_instance_database(
    tmp_path,
):
    project_root = tmp_path
    database_path = project_root / "instance" / "medical_ai.db"
    database_path.parent.mkdir()
    database_path.write_bytes(b"existing database")

    resolved = make_url(
        resolve_database_url(
            "sqlite:///medical_ai.db",
            project_root=project_root,
            default_sqlite_path=database_path,
        )
    )

    assert Path(resolved.database) == database_path.resolve()


def test_explicit_absolute_database_url_is_preserved(tmp_path):
    database_path = tmp_path / "custom.db"

    resolved = make_url(
        resolve_database_url(
            f"sqlite:///{database_path.as_posix()}",
            project_root=tmp_path,
            default_sqlite_path=tmp_path / "instance" / "medical_ai.db",
        )
    )

    assert Path(resolved.database) == database_path
