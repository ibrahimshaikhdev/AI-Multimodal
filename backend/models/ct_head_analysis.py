from datetime import datetime, timezone

from backend.extensions import db


class CTHeadAnalysisRecord(db.Model):
    __tablename__ = "ct_head_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    series_classification = db.Column(db.JSON, nullable=False)
    slice_classification = db.Column(db.JSON, nullable=False)
    slice_count = db.Column(db.Integer, nullable=False)
    highest_any_slice_index = db.Column(db.Integer, nullable=False)
    input_format = db.Column(db.String(40), nullable=False)
    localization_file_reference = db.Column(db.String(500), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship("ScanAsset", back_populates="ct_head_analysis")


__all__ = ["CTHeadAnalysisRecord"]
