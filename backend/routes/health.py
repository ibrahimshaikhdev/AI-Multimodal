from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check():
    """Simple health endpoint for confirming the API is running."""
    return jsonify(
        {
            "status": "ok",
            "service": "multimodal-ai-medical-platform",
            "message": "Backend is running and ready for the next module.",
        }
    )
