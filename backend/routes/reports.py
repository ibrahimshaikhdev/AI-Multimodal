from datetime import date, datetime
from pathlib import PurePosixPath

from flask import Blueprint, abort, g, jsonify, request, send_from_directory

from backend.auth import require_auth
from backend.extensions import db
from backend.models import MedicalReport, Patient
from backend.services.file_storage import (
    get_stored_file_location,
    save_uploaded_file,
    validate_uploaded_file,
)
from backend.services.ocr_service import extract_document_text
from backend.services.parameter_persistence import persist_report_parameters
from backend.services.report_parameters import extract_report_parameters
from backend.services.report_metadata import extract_report_metadata
from backend.services.text_cleaning import clean_extracted_text

reports_bp = Blueprint("reports", __name__)


def _serialize_parameter(parameter):
    return {
        "parameter": parameter.parameter,
        "value": parameter.value,
        "unit": parameter.unit,
        "date": parameter.result_date.isoformat() if parameter.result_date else None,
        "confidence": parameter.confidence,
        "source": parameter.source,
    }


def _serialize_report(report):
    processing_messages = {
        "processed": "Text extracted successfully.",
        "no_text_found": "The report was saved, but no readable text was found.",
        "extraction_failed": "The report was saved, but text extraction could not be completed.",
        "uploaded": "The report has not been processed yet.",
    }
    metadata = extract_report_metadata(report.extracted_text) if report.extracted_text else None
    document_date = (
        report.report_date.isoformat()
        if report.report_date
        else metadata["document_date"] if metadata else None
    )
    parameters = [_serialize_parameter(parameter) for parameter in report.parameters]
    if not parameters and report.extracted_text:
        parameters = extract_report_parameters(report.extracted_text, document_date=document_date)

    return {
        "id": report.id,
        "patient_id": report.patient_id,
        "report_type": report.report_type,
        "report_date": report.report_date.isoformat() if report.report_date else None,
        "file_reference": report.file_reference,
        "original_url": f"/api/reports/{report.id}/original" if report.file_reference else None,
        "extracted_text": report.extracted_text,
        "metadata": metadata,
        "parameters": parameters,
        "processing_status": report.processing_status,
        "processing_message": processing_messages.get(report.processing_status, "Report processing status is unknown."),
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "updated_at": report.updated_at.isoformat() if report.updated_at else None,
    }


@reports_bp.get("/reports")
@require_auth
def list_reports():
    patient_id_value = request.args.get("patient_id")
    try:
        patient_id = int(patient_id_value)
    except (TypeError, ValueError):
        return jsonify({"error": "A valid patient_id is required"}), 400

    patient = (
        db.session.query(Patient)
        .filter_by(id=patient_id, created_by_id=g.current_user.id)
        .first()
    )
    if patient is None:
        return jsonify({"error": "Patient not found"}), 404

    reports = (
        db.session.query(MedicalReport)
        .filter_by(patient_id=patient.id)
        .order_by(MedicalReport.created_at.desc(), MedicalReport.id.desc())
        .all()
    )
    return jsonify({"reports": [_serialize_report(report) for report in reports]}), 200


@reports_bp.get("/reports/<int:report_id>/original")
@require_auth
def get_original_report(report_id):
    report = (
        db.session.query(MedicalReport)
        .join(Patient)
        .filter(
            MedicalReport.id == report_id,
            Patient.created_by_id == g.current_user.id,
        )
        .first()
    )
    if report is None:
        return jsonify({"error": "Report not found"}), 404

    location = get_stored_file_location(report.file_reference, report.patient_id)
    if location is None:
        abort(404)

    upload_root, relative_path = location
    extension = PurePosixPath(relative_path).suffix
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"report-{report.id}{extension}",
    )


@reports_bp.post("/reports/upload")
@require_auth
def upload_report():
    data = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
    patient_id_value = data.get("patient_id")
    if patient_id_value is None:
        return jsonify({"error": "patient_id is required"}), 400

    try:
        patient_id = int(patient_id_value)
    except (TypeError, ValueError):
        return jsonify({"error": "patient_id must be an integer"}), 400

    patient = (
        db.session.query(Patient)
        .filter_by(id=patient_id, created_by_id=g.current_user.id)
        .first()
    )
    if patient is None:
        return jsonify({"error": "Patient not found"}), 404

    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "A report file is required"}), 400

    try:
        validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    report_type = data.get("report_type") or "general"
    if not isinstance(report_type, str) or not report_type.strip():
        return jsonify({"error": "report_type is required"}), 400
    report_type = report_type.strip()
    if len(report_type) > 80:
        return jsonify({"error": "report_type must be 80 characters or fewer"}), 400

    report_date_value = data.get("report_date")
    if report_date_value in (None, "", "null"):
        report_date = None
    elif isinstance(report_date_value, str):
        try:
            report_date = datetime.strptime(report_date_value, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "report_date must use YYYY-MM-DD format"}), 400
        if report_date > date.today():
            return jsonify({"error": "report_date cannot be in the future"}), 400
    else:
        return jsonify({"error": "report_date must use YYYY-MM-DD format or be null"}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    uploaded_file.stream.seek(0)
    try:
        extracted_text = clean_extracted_text(
            extract_document_text(uploaded_file.stream.read(), uploaded_file.filename)
        )
        processing_status = "processed"
    except ValueError as exc:
        extracted_text = None
        processing_status = (
            "no_text_found"
            if "no readable text found" in str(exc).lower()
            else "extraction_failed"
        )
    except RuntimeError:
        extracted_text = None
        processing_status = "extraction_failed"

    report = MedicalReport(
        patient_id=patient.id,
        report_type=report_type,
        report_date=report_date,
        file_reference=file_reference,
        extracted_text=extracted_text,
        processing_status=processing_status,
    )
    db.session.add(report)
    db.session.flush()

    if extracted_text:
        metadata = extract_report_metadata(extracted_text)
        document_date = report_date.isoformat() if report_date else metadata["document_date"]
        parameters = extract_report_parameters(extracted_text, document_date=document_date)
        persist_report_parameters(report.id, parameters)

    db.session.commit()

    return jsonify({"report": _serialize_report(report)}), 201
