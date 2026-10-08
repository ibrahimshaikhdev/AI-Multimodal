from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from flask import Blueprint, current_app, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth import require_auth
from backend.extensions import db
from backend.models import RevokedToken, User
from backend.services.audit_logging import record_audit_event

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/auth/login")
def login_user():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    if not isinstance(email, str) or not isinstance(password, str) or not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = db.session.query(User).filter_by(email=email.strip().lower()).first()
    if user is None or not user.is_active or not check_password_hash(user.password_hash, password):
        record_audit_event(
            actor_id=user.id if user is not None else None,
            action="auth.login",
            resource_type="authentication",
            status="failure",
        )
        return jsonify({"error": "Invalid email or password"}), 401

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": str(user.id),
            "jti": str(uuid4()),
            "email": user.email,
            "role": user.role,
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        current_app.config["SECRET_KEY"],
        algorithm="HS256",
    )
    record_audit_event(
        actor_id=user.id,
        action="auth.login",
        resource_type="authentication",
    )

    return jsonify(
        {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role": user.role,
            },
        }
    ), 200


@auth_bp.post("/auth/logout")
@require_auth
def logout_user():
    claims = g.token_claims
    revoked_token = RevokedToken(
        jti=claims["jti"],
        expires_at=datetime.fromtimestamp(claims["exp"], timezone.utc),
    )
    db.session.add(revoked_token)
    db.session.commit()
    record_audit_event(
        actor_id=g.current_user.id,
        action="auth.logout",
        resource_type="authentication",
    )

    return jsonify({"message": "Logged out successfully"}), 200


@auth_bp.get("/auth/me")
@require_auth
def current_user():
    user = g.current_user
    return jsonify(
        {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role": user.role,
            }
        }
    ), 200


@auth_bp.put("/auth/me")
@require_auth
def update_current_user():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    name_values = {}
    for field in ("first_name", "last_name"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            return jsonify({"error": f"{field} is required"}), 400
        value = value.strip()
        if len(value) > 80:
            return jsonify({"error": f"{field} must be 80 characters or fewer"}), 400
        name_values[field] = value

    user = g.current_user
    user.first_name = name_values["first_name"]
    user.last_name = name_values["last_name"]
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Could not update profile for user %s", user.id)
        return jsonify({"error": "Could not update profile"}), 500

    record_audit_event(
        actor_id=user.id,
        action="profile.update",
        resource_type="user",
        resource_id=user.id,
    )
    return jsonify(
        {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
            }
        }
    ), 200


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
    record_audit_event(
        actor_id=user.id,
        action="auth.register",
        resource_type="user",
        resource_id=user.id,
    )

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
