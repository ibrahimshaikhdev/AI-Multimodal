from .auth import auth_bp
from .health import health_bp
from .patients import patients_bp
from .reports import reports_bp
from .scans import scans_bp
from .research_papers import research_papers_bp
from .audit_logs import audit_logs_bp
from .ai import ai_bp


def register_blueprints(app):
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(patients_bp, url_prefix="/api")
    app.register_blueprint(reports_bp, url_prefix="/api")
    app.register_blueprint(scans_bp, url_prefix="/api")
    app.register_blueprint(research_papers_bp, url_prefix="/api")
    app.register_blueprint(audit_logs_bp, url_prefix="/api")
    app.register_blueprint(ai_bp, url_prefix="/api")
