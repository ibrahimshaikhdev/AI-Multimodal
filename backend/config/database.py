from pathlib import Path

from sqlalchemy.engine import URL, make_url


def resolve_database_url(
    configured_url: str | None,
    *,
    project_root: Path,
    default_sqlite_path: Path,
) -> str:
    if configured_url:
        url = make_url(configured_url)
    else:
        url = URL.create("sqlite", database=str(default_sqlite_path.resolve()))

    if not url.drivername.startswith("sqlite") or not url.database:
        return url.render_as_string(hide_password=False)
    if url.database == ":memory:":
        return url.render_as_string(hide_password=False)

    database_path = Path(url.database)
    if not database_path.is_absolute():
        database_path = (project_root / database_path).resolve()
        persistent_path = default_sqlite_path.resolve()
        if (
            not database_path.exists()
            and persistent_path.is_file()
            and Path(url.database).name in {"app.db", "medical_ai.db"}
        ):
            database_path = persistent_path

    return url.set(database=str(database_path)).render_as_string(
        hide_password=False
    )
