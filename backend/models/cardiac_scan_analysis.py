from datetime import datetime, timezone

from backend.extensions import db


class CardiacScanAnalysis(db.Model):
    __tablename__ = "cardiac_scan_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    pixel_counts = db.Column(db.JSON, nullable=False)
    overlay_file_reference = db.Column(db.String(500), nullable=False, unique=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="cardiac_analysis")

    def __repr__(self):
        return f"<CardiacScanAnalysis scan={self.scan_id} model={self.model_name}>"


__all__ = ["CardiacScanAnalysis"]