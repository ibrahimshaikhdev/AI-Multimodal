from datetime import datetime, timezone

from backend.extensions import db


class ScanAsset(db.Model):
    __tablename__ = "scan_assets"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.Integer,
        db.ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    modality = db.Column(db.String(30), nullable=False)
    body_region = db.Column(db.String(80), nullable=True)
    study_date = db.Column(db.Date, nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    file_reference = db.Column(db.String(500), nullable=False, unique=True)
    processing_status = db.Column(db.String(50), nullable=False, default="awaiting_model")
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    patient = db.relationship("Patient", back_populates="scans")
    cardiac_analysis = db.relationship(
        "CardiacScanAnalysis",
        back_populates="scan",
        cascade="all, delete-orphan",
        uselist=False,
    )
    knee_analysis = db.relationship(
        "KneeScanAnalysis",
        back_populates="scan",
        cascade="all, delete-orphan",
        uselist=False,
    )
    spine_analysis = db.relationship(
        "SpineScanAnalysis",
        back_populates="scan",
        cascade="all, delete-orphan",
        uselist=False,
    )
    brain_analysis = db.relationship(
        "BrainScanAnalysis",
        back_populates="scan",
        cascade="all, delete-orphan",
        uselist=False,
    )

    def __repr__(self):
        return f"<ScanAsset {self.id}: {self.modality} {self.body_region or ''}>"


__all__ = ["ScanAsset"]