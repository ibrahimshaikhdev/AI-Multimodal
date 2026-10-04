from datetime import datetime, timezone

from backend.extensions import db


class MedicalReport(db.Model):
    __tablename__ = "medical_reports"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.Integer,
        db.ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    report_type = db.Column(db.String(80), nullable=False, default="general")
    report_date = db.Column(db.Date, nullable=True)
    file_reference = db.Column(db.String(500), nullable=True)
    extracted_text = db.Column(db.Text, nullable=True)
    processing_status = db.Column(db.String(50), nullable=False, default="uploaded")
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    patient = db.relationship("Patient", back_populates="reports")
    parameters = db.relationship(
        "ReportParameter",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="ReportParameter.id",
    )

    def __repr__(self):
        return f"<MedicalReport {self.id}: {self.report_type}>"


Report = MedicalReport

__all__ = ["MedicalReport", "Report"]
