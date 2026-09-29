from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash

from backend.extensions import db
from backend.models.user import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/auth/register")
def register_user():
    data = request.get_json(silent=True) or {}

    required_fields = ["first_name", "last_name", "email", "password"]
    missing_fields = [field for field in required_fields if not data.get(field)]

    if missing_fields:
        return jsonify({"error": "Missing required fields", "missing_fields": missing_fields}), 400

    first_name = str(data["first_name"]).strip()
    last_name = str(data["last_name"]).strip()
    email = str(data["email"]).strip().lower()
    password = str(data["password"]).strip()

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters long"}), 400

    if not email or "@" not in email:
        return jsonify({"error": "Valid email is required"}), 400

    existing_user = db.session.query(User).filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "Email already registered"}), 400

    user = User(
        first_name=first_name,
        last_name=last_name,
        email=email,
        password_hash=generate_password_hash(password),
        role="patient",
        is_active=True,
    )

    db.session.add(user)
    db.session.commit()

    return jsonify(
        {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role": user.role,
                "is_active": user.is_active,
            },
        }
    ), 201
