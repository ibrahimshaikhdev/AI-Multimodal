from .auth import auth_bp
from .health import health_bp
from .patients import patients_bp
from .reports import reports_bp
from .scans import scans_bp


def register_blueprints(app):
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(patients_bp, url_prefix="/api")
    app.register_blueprint(reports_bp, url_prefix="/api")
    app.register_blueprint(scans_bp, url_prefix="/api")
