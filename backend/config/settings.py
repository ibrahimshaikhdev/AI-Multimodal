import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "instance" / "medical_ai.db"


@dataclass
class Settings:
    APP_NAME: str = "Multimodal AI Research & Medical Report Intelligence"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}",
    )
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() in {"1", "true", "yes", "on"}
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 15 * 1024 * 1024))
    MAX_CT_UPLOAD_SIZE: int = int(
        os.getenv("MAX_CT_UPLOAD_SIZE", 512 * 1024 * 1024)
    )


settings = Settings()


def get_settings() -> Settings:
    return settings
