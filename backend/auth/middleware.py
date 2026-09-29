from functools import wraps

import jwt
from flask import current_app, g, jsonify, request
from jwt import InvalidTokenError

from backend.extensions import db
from backend.models import RevokedToken, User


def require_auth(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        authorization = request.headers.get("Authorization", "")
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"error": "Authentication required"}), 401

        try:
            claims = jwt.decode(
                parts[1],
                current_app.config["SECRET_KEY"],
                algorithms=["HS256"],
                options={"require": ["exp", "sub", "jti"]},
            )
        except InvalidTokenError:
            return jsonify({"error": "Invalid or expired token"}), 401

        subject = claims.get("sub")
        token_id = claims.get("jti")
        if not isinstance(subject, str) or not isinstance(token_id, str):
            return jsonify({"error": "Invalid or expired token"}), 401

        try:
            user_id = int(subject)
        except ValueError:
            return jsonify({"error": "Invalid or expired token"}), 401

        if db.session.get(RevokedToken, token_id) is not None:
            return jsonify({"error": "Invalid or expired token"}), 401

        user = db.session.get(User, user_id)
        if user is None or not user.is_active:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.current_user = user
        g.token_claims = claims
        return view_function(*args, **kwargs)

    return wrapped_view


def require_roles(*allowed_roles):
    roles = {
        role.strip().lower()
        for role in allowed_roles
        if isinstance(role, str) and role.strip()
    }
    if not roles:
        raise ValueError("At least one non-empty role is required")

    def decorator(view_function):
        @wraps(view_function)
        def check_role(*args, **kwargs):
            user_role = g.current_user.role.strip().lower()
            if user_role not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return view_function(*args, **kwargs)

        return require_auth(check_role)

    return decorator