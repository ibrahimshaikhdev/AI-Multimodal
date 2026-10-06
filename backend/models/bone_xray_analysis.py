from datetime import datetime, timezone

from backend.extensions import db


class BoneXrayAnalysisRecord(db.Model):
    __tablename__ = "bone_xray_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(120), nullable=False)
    predicted_label = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="bone_xray_analysis")

    def __repr__(self):
        return f"<BoneXrayAnalysisRecord scan={self.scan_id} label={self.predicted_label}>"


__all__ = ["BoneXrayAnalysisRecord"]
