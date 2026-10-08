from datetime import datetime, timezone

from backend.extensions import db


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    actor_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action = db.Column(db.String(100), nullable=False, index=True)
    resource_type = db.Column(db.String(80), nullable=False)
    resource_id = db.Column(db.String(100), nullable=True)
    timestamp = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    status = db.Column(db.String(32), nullable=False)
    metadata_json = db.Column("metadata", db.JSON, nullable=False, default=dict)

    actor = db.relationship("User")

    def __repr__(self):
        return f"<AuditLog {self.action} ({self.id})>"
