from datetime import datetime, timezone

from backend.extensions import db


class AbdominalAortaUltrasoundAnalysisRecord(db.Model):
    __tablename__ = "abdominal_aorta_ultrasound_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    mask_area_pixels = db.Column(db.Integer, nullable=False)
    image_shape = db.Column(db.JSON, nullable=False)
    overlay_file_reference = db.Column(db.String(500), nullable=True, unique=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship(
        "ScanAsset",
        back_populates="abdominal_aorta_ultrasound_analysis",
    )

    def __repr__(self):
        return (
            f"<AbdominalAortaUltrasoundAnalysisRecord scan={self.scan_id} "
            f"mask_area={self.mask_area_pixels}>"
        )


__all__ = ["AbdominalAortaUltrasoundAnalysisRecord"]
