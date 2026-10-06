from datetime import datetime, timezone

from backend.extensions import db


class EchoView47AnalysisRecord(db.Model):
    __tablename__ = "echoview47_analyses"

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(
        db.Integer,
        db.ForeignKey("scan_assets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model_name = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(80), nullable=False)
    model_score = db.Column(db.Float, nullable=False)
    class_scores = db.Column(db.JSON, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan = db.relationship(
        "ScanAsset",
        back_populates="echoview47_analysis",
    )

    def __repr__(self):
        return (
            f"<EchoView47AnalysisRecord scan={self.scan_id} "
            f"class={self.predicted_class}>"
        )


__all__ = ["EchoView47AnalysisRecord"]
