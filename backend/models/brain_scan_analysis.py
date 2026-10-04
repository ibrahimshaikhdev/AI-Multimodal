from datetime import datetime, timezone

from backend.extensions import db


class BrainScanAnalysis(db.Model):
    __tablename__ = "brain_scan_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    predicted_class_index = db.Column(db.Integer, nullable=False)
    model_score = db.Column(db.Float, nullable=False)
    class_scores = db.Column(db.JSON, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="brain_analysis")

    def __repr__(self):
        return f"<BrainScanAnalysis scan={self.scan_id} class={self.predicted_class_index}>"


__all__ = ["BrainScanAnalysis"]