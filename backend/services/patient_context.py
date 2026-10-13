from typing import Sequence, TypedDict

from backend.extensions import db
from backend.models import MedicalReport, Patient, ScanAsset


class PatientRecords(TypedDict):
    patient: Patient
    reports: list[MedicalReport]
    scans: list[ScanAsset]


def list_owned_patients(user_id: int) -> list[Patient]:
    return (
        db.session.query(Patient)
        .filter_by(created_by_id=user_id)
        .order_by(Patient.id.asc())
        .all()
    )


def get_owned_patient_records(
    user_id: int,
    patient_id: int,
    *,
    report_limit: int | None = None,
    scan_limit: int | None = None,
) -> PatientRecords | None:
    patient = (
        db.session.query(Patient)
        .filter_by(id=patient_id, created_by_id=user_id)
        .first()
    )
    if patient is None:
        return None

    reports_query = (
        db.session.query(MedicalReport)
        .join(Patient, MedicalReport.patient_id == Patient.id)
        .filter(
            MedicalReport.patient_id == patient.id,
            Patient.created_by_id == user_id,
        )
        .order_by(
            MedicalReport.report_date.desc().nullslast(),
            MedicalReport.created_at.desc(),
            MedicalReport.id.desc(),
        )
    )
    scans_query = (
        db.session.query(ScanAsset)
        .join(Patient, ScanAsset.patient_id == Patient.id)
        .filter(
            ScanAsset.patient_id == patient.id,
            Patient.created_by_id == user_id,
        )
        .order_by(
            ScanAsset.study_date.desc().nullslast(),
            ScanAsset.created_at.desc(),
            ScanAsset.id.desc(),
        )
    )
    if report_limit is not None:
        reports_query = reports_query.limit(report_limit)
    if scan_limit is not None:
        scans_query = scans_query.limit(scan_limit)

    return {
        "patient": patient,
        "reports": reports_query.all(),
        "scans": scans_query.all(),
    }


def get_owned_reports(
    user_id: int,
    report_ids: Sequence[int],
) -> list[MedicalReport] | None:
    reports = (
        db.session.query(MedicalReport)
        .join(Patient, MedicalReport.patient_id == Patient.id)
        .filter(
            MedicalReport.id.in_(report_ids),
            Patient.created_by_id == user_id,
        )
        .all()
    )
    reports_by_id = {report.id: report for report in reports}
    if len(reports_by_id) != len(report_ids):
        return None
    return [reports_by_id[report_id] for report_id in report_ids]
