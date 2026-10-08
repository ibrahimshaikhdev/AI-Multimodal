from flask import Blueprint, g, jsonify, request

from backend.auth import require_auth
from backend.extensions import db
from backend.models import AuditLog


audit_logs_bp = Blueprint("audit_logs", __name__)


def _serialize_audit_log(event):
    actor = event.actor
    return {
        "id": event.id,
        "actor": (
            {
                "id": actor.id,
                "name": f"{actor.first_name} {actor.last_name}",
                "email": actor.email,
            }
            if actor is not None
            else None
        ),
        "action": event.action,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "timestamp": event.timestamp.isoformat(),
        "status": event.status,
        "metadata": event.metadata_json,
    }


@audit_logs_bp.get("/audit-logs")
@require_auth
def list_audit_logs():
    try:
        limit = int(request.args.get("limit", "100"))
        offset = int(request.args.get("offset", "0"))
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400
    if not 1 <= limit <= 500 or offset < 0:
        return jsonify({"error": "limit must be 1-500 and offset cannot be negative"}), 400

    query = db.session.query(AuditLog)
    if g.current_user.role.strip().lower() != "admin":
        query = query.filter_by(actor_id=g.current_user.id)

    events = (
        query.order_by(AuditLog.timestamp.desc(), AuditLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return jsonify({"logs": [_serialize_audit_log(event) for event in events]}), 200
