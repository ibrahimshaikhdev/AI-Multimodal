from datetime import datetime, timezone

from backend.extensions import db


class KneeScanAnalysis(db.Model):
    __tablename__ = "knee_scan_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    slice_index = db.Column(db.Integer, nullable=False, default=0)
    predicted_class_index = db.Column(db.Integer, nullable=False)
    predicted_class = db.Column(db.String(80), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.JSON, nullable=False)
    localization_status = db.Column(db.String(40), nullable=False)
    roi_box = db.Column(db.JSON, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="knee_analysis")

    def __repr__(self):
        return f"<KneeScanAnalysis scan={self.scan_id} class={self.predicted_class}>"


__all__ = ["KneeScanAnalysis"]