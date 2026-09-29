from .auth import auth_bp
from .health import health_bp
from .patients import patients_bp


def register_blueprints(app):
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(patients_bp, url_prefix="/api")
