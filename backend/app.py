import os
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from backend.config.settings import settings
from backend.extensions import db
from backend.routes import register_blueprints
from backend.services.spine_mri import (
    SpineMRIPredictionService,
    SpineModelContractError,
    SpineModelUnavailableError,
)
from backend.services.brain_mri import (
    BrainMRIPredictionService,
    BrainModelContractError,
    BrainModelUnavailableError,
)


def create_app(testing: bool = False):
    app = Flask(__name__)
    app.config["TESTING"] = testing
    app.config["SECRET_KEY"] = settings.SECRET_KEY

    if testing:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    else:
        default_db_path = Path(app.root_path).resolve().parent / "instance" / "app.db"
        app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
            "DATABASE_URL",
            f"sqlite:///{default_db_path}",
        )

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "..", "instance", "uploads")

    CORS(app)
    db.init_app(app)
    register_blueprints(app)

    spine_mri_service = SpineMRIPredictionService()
    app.extensions["spine_mri_service"] = spine_mri_service
    if not testing:
        try:
            spine_mri_service.load()
        except (SpineModelContractError, SpineModelUnavailableError) as exc:
            app.logger.warning("Spine MRI model unavailable at startup: %s", exc)

    brain_mri_service = BrainMRIPredictionService()
    app.extensions["brain_mri_service"] = brain_mri_service
    if not testing:
        try:
            brain_mri_service.load()
        except (BrainModelContractError, BrainModelUnavailableError) as exc:
            app.logger.warning("Brain MRI model unavailable at startup: %s", exc)

    with app.app_context():
        db.create_all()

    @app.get("/")
    def index():
        return {
            "app": settings.APP_NAME,
            "status": "running",
            "message": "Welcome to the medical AI platform backend.",
        }

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=settings.DEBUG)
