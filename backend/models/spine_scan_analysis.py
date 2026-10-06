from datetime import datetime, timezone

from backend.extensions import db


class SpineScanAnalysis(db.Model):
    __tablename__ = "spine_scan_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    model_score = db.Column(db.Float, nullable=False)
    severity_scores = db.Column(db.JSON, nullable=True)
    frame_count = db.Column(db.Integer, nullable=False)
    frame_source = db.Column(db.String(80), nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="spine_analysis")

    def __repr__(self):
        return f"<SpineScanAnalysis scan={self.scan_id} score={self.model_score:.4f}>"


__all__ = ["SpineScanAnalysis"]