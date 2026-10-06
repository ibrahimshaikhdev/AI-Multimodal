from datetime import datetime, timezone

from backend.extensions import db


class ChestXrayAnalysisRecord(db.Model):
    __tablename__ = "chest_xray_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    findings = db.Column(db.JSON, nullable=False)
    input_shape = db.Column(db.JSON, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="chest_xray_analysis")

    def __repr__(self):
        return f"<ChestXrayAnalysisRecord scan={self.scan_id} model={self.model_name}>"


__all__ = ["ChestXrayAnalysisRecord"]