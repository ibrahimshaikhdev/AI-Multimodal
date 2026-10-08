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
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini")
    AI_FALLBACK_PROVIDER: str = os.getenv("AI_FALLBACK_PROVIDER", "openrouter")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "google/gemini-2.5-flash:free",
    )
    AI_TIMEOUT_SECONDS: float = float(os.getenv("AI_TIMEOUT_SECONDS", "60"))
    RESEARCH_EMBEDDING_MODEL: str = os.getenv(
        "RESEARCH_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


settings = Settings()


def get_settings() -> Settings:
    return settings
