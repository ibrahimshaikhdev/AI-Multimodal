import os
from pathlib import Path

from flask import Flask
from flask import send_from_directory
from flask_cors import CORS
from sqlalchemy import inspect, text

from backend.config.database import resolve_database_url
from backend.config.settings import settings
from backend.config.settings import DEFAULT_DATABASE_PATH, PROJECT_ROOT
from backend.extensions import db
from backend.routes import register_blueprints
from backend.services.ai_service import AIService
from backend.services.research_rag_service import ResearchRAGService
from backend.services.brain_mri import (
    BrainMRIPredictionService,
    BrainModelContractError,
    BrainModelUnavailableError,
)
from backend.services.spine_mri import (
    SpineMRIPredictionService,
    SpineModelContractError,
    SpineModelUnavailableError,
)
from backend.services.chest_xray import (
    ChestXrayAnalysisService,
    ChestXrayError,
)
from backend.services.bone_xray import (
    BoneXrayAnalysisService,
    BoneXrayError,
)
from backend.services.dental_xray import (
    DentalXrayAnalysisService,
    DentalXrayError,
)
from backend.services.ct_head_hemorrhage import (
    CTHeadHemorrhageService,
    CTHeadModelUnavailableError,
)
from backend.services.chest_ct_segmentation import ChestCTSegmentationService
from backend.services.obstetric_ultrasound import ObstetricUltrasoundAnalysisService
from backend.services.abdominal_aorta_ultrasound import (
    AbdominalAortaUltrasoundAnalysisService,
    AbdominalAortaUltrasoundModelContractError,
    AbdominalAortaUltrasoundModelUnavailableError,
)
from backend.services.vascular_ultrasound import (
    VascularUltrasoundAnalysisService,
    VascularUltrasoundModelContractError,
    VascularUltrasoundModelUnavailableError,
)
from backend.services.echoview47 import EchoView47AnalysisService


def create_app(testing: bool = False):
    app = Flask(__name__)
    app.config["TESTING"] = testing
    warm_models_at_startup = (
        not testing
        and os.getenv("SKIP_MODEL_WARMUP", "").strip().lower()
        not in {"1", "true", "yes", "on"}
    )
    app.config["SECRET_KEY"] = settings.SECRET_KEY
    app.config["AI_PROVIDER"] = settings.AI_PROVIDER
    app.config["AI_FALLBACK_PROVIDER"] = settings.AI_FALLBACK_PROVIDER
    app.config["AI_TIMEOUT_SECONDS"] = settings.AI_TIMEOUT_SECONDS
    app.config["RESEARCH_EMBEDDING_MODEL"] = settings.RESEARCH_EMBEDDING_MODEL

    if testing:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = resolve_database_url(
            os.getenv("DATABASE_URL"),
            project_root=PROJECT_ROOT,
            default_sqlite_path=DEFAULT_DATABASE_PATH,
        )

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = max(
        settings.MAX_CONTENT_LENGTH,
        settings.MAX_CT_UPLOAD_SIZE + 1024 * 1024,
    )
    app.config["MAX_CT_UPLOAD_SIZE"] = settings.MAX_CT_UPLOAD_SIZE
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "..", "instance", "uploads")

    CORS(app)
    db.init_app(app)
    register_blueprints(app)
    app.extensions["ai_service"] = AIService(
        provider=app.config["AI_PROVIDER"],
        fallback_provider=app.config["AI_FALLBACK_PROVIDER"],
        gemini_api_key=settings.GEMINI_API_KEY,
        gemini_model=settings.GEMINI_MODEL,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        openrouter_model=settings.OPENROUTER_MODEL,
        timeout_seconds=app.config["AI_TIMEOUT_SECONDS"],
    )
    app.extensions["research_rag_service"] = ResearchRAGService(
        index_root=Path(app.instance_path) / "research_indexes",
        embedding_model=app.config["RESEARCH_EMBEDDING_MODEL"],
        upload_root=app.config["UPLOAD_FOLDER"],
    )

    spine_mri_service = SpineMRIPredictionService()
    app.extensions["spine_mri_service"] = spine_mri_service
    if warm_models_at_startup:
        try:
            spine_mri_service.load()
        except (SpineModelContractError, SpineModelUnavailableError) as exc:
            app.logger.warning("Spine MRI model unavailable at startup: %s", exc)

    brain_mri_service = BrainMRIPredictionService()
    app.extensions["brain_mri_service"] = brain_mri_service
    if warm_models_at_startup:
        try:
            brain_mri_service.load()
        except (BrainModelContractError, BrainModelUnavailableError) as exc:
            app.logger.warning("Brain MRI model unavailable at startup: %s", exc)

    chest_xray_service = ChestXrayAnalysisService()
    app.extensions["chest_xray_service"] = chest_xray_service
    if warm_models_at_startup:
        try:
            chest_xray_service.load()
        except ChestXrayError as exc:
            app.logger.warning("Chest X-ray model unavailable at startup: %s", exc)

    bone_xray_service = BoneXrayAnalysisService()
    app.extensions["bone_xray_service"] = bone_xray_service
    if warm_models_at_startup:
        try:
            bone_xray_service.load()
        except BoneXrayError as exc:
            app.logger.warning("Bone X-ray model unavailable at startup: %s", exc)

    dental_xray_service = DentalXrayAnalysisService()
    app.extensions["dental_xray_service"] = dental_xray_service
    if warm_models_at_startup:
        try:
            dental_xray_service.load()
        except DentalXrayError as exc:
            app.logger.warning("Dental X-ray model unavailable at startup: %s", exc)

    ct_head_service = CTHeadHemorrhageService()
    app.extensions["ct_head_hemorrhage_service"] = ct_head_service
    if warm_models_at_startup:
        try:
            ct_head_service.load()
        except CTHeadModelUnavailableError as exc:
            app.logger.warning("Head CT model unavailable at startup: %s", exc)

    app.extensions["chest_ct_segmentation_service"] = ChestCTSegmentationService()

    app.extensions["obstetric_ultrasound_service"] = (
        ObstetricUltrasoundAnalysisService()
    )
    abdominal_aorta_service = AbdominalAortaUltrasoundAnalysisService()
    app.extensions["abdominal_aorta_ultrasound_service"] = abdominal_aorta_service
    app.extensions["echoview47_service"] = EchoView47AnalysisService()
    vascular_ultrasound_service = VascularUltrasoundAnalysisService()
    app.extensions["vascular_ultrasound_service"] = vascular_ultrasound_service
    if warm_models_at_startup:
        try:
            abdominal_aorta_service.load()
        except (
            AbdominalAortaUltrasoundModelContractError,
            AbdominalAortaUltrasoundModelUnavailableError,
        ) as exc:
            app.logger.warning("Abdominal aorta model unavailable at startup: %s", exc)
        try:
            vascular_ultrasound_service.load()
        except (
            VascularUltrasoundModelContractError,
            VascularUltrasoundModelUnavailableError,
        ) as exc:
            app.logger.warning("Carotid ultrasound model unavailable at startup: %s", exc)

    with app.app_context():
        db.create_all()
        columns = {
            column["name"]
            for column in inspect(db.engine).get_columns("spine_scan_analyses")
        }
        if "severity_scores" not in columns:
            with db.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE spine_scan_analyses "
                        "ADD COLUMN severity_scores JSON"
                    )
                )

    @app.get("/")
    @app.get("/medicalreports.html")
    @app.get("/scan.html")
    @app.get("/healthtimeline.html")
    @app.get("/researchpapers.html")
    def index():
        return send_from_directory(PROJECT_ROOT / "frontend", "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=settings.DEBUG)
