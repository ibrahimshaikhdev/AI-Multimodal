from backend.extensions import db


class ReportParameter(db.Model):
    __tablename__ = "report_parameters"
    __table_args__ = (
        db.UniqueConstraint(
            "report_id",
            "fingerprint",
            name="uq_report_parameter_fingerprint",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(
        db.Integer,
        db.ForeignKey("medical_reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parameter = db.Column(db.String(150), nullable=False)
    value = db.Column(db.String(100), nullable=False)
    unit = db.Column(db.String(60), nullable=False)
    result_date = db.Column(db.Date, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    source = db.Column(db.Text, nullable=False)
    fingerprint = db.Column(db.String(64), nullable=False)

    report = db.relationship("MedicalReport", back_populates="parameters")

    def __repr__(self):
        return f"<ReportParameter {self.parameter}={self.value} {self.unit}>"


__all__ = ["ReportParameter"]