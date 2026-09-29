import os

from flask import Flask
from flask_cors import CORS

from backend.config.settings import settings
from backend.extensions import db
from backend.routes import register_blueprints


def create_app(testing: bool = False):
    app = Flask(__name__)
    app.config["TESTING"] = testing
    app.config["SECRET_KEY"] = settings.SECRET_KEY

    if testing:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", settings.DATABASE_URL)

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_CONTENT_LENGTH

    CORS(app)
    db.init_app(app)
    register_blueprints(app)

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
