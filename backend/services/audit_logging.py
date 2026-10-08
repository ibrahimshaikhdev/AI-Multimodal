from sqlalchemy.exc import SQLAlchemyError

from flask import current_app

from backend.extensions import db
from backend.models.audit_log import AuditLog


def record_audit_event(
    *,
    actor_id,
    action,
    resource_type,
    resource_id=None,
    status="success",
    metadata=None,
):
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Audit metadata must be a dictionary")

    event = AuditLog(
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        status=status,
        metadata_json=metadata or {},
    )
    db.session.add(event)
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception(
            "Could not persist audit event %s for %s %s",
            action,
            resource_type,
            resource_id,
        )
