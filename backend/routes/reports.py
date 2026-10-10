import json
from datetime import date, datetime
from pathlib import Path
from pathlib import PurePosixPath
import re

from flask import Blueprint, abort, current_app, g, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest

from backend.auth import require_auth
from backend.extensions import db
from backend.models import MedicalReport, Patient
from backend.services.file_storage import (
    delete_stored_file,
    get_stored_file_location,
    save_uploaded_file,
    validate_uploaded_file,
)
from backend.services.ocr_service import extract_document_text
from backend.services.parameter_persistence import persist_report_parameters
from backend.services.report_parameters import extract_report_parameters
from backend.services.report_metadata import extract_report_metadata
from backend.services.text_cleaning import clean_extracted_text
from backend.services.audit_logging import record_audit_event
from backend.services.ai_service import AIServiceError
from backend.services.document_comparison import compare_report_texts
from backend.services.local_llm import local_generate

reports_bp = Blueprint("reports", __name__)
SUMMARY_DISCLAIMER = (
    "AI-generated summary for informational purposes; requires human review. "
    "This is not a diagnosis."
)
COMPARISON_DISCLAIMER = (
    "Structured values and wording are compared locally; the AI narrative summary uses only the two saved "
    "extracted texts. Similarity scores are not probabilities and do not establish clinical equivalence or a "
    "diagnosis; review the original reports."
)


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
    record_audit_event(
        actor_id=g.current_user.id,
        action="report.list",
        resource_type="patient",
        resource_id=patient.id,
        metadata={"count": len(reports)},
    )
    return jsonify({"reports": [_serialize_report(report) for report in reports]}), 200


@reports_bp.post("/reports/<int:report_id>/extract-text")
@require_auth
def extract_report_text(report_id):
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
        return jsonify({"success": False, "error": "Report not found"}), 404

    if isinstance(report.extracted_text, str) and report.extracted_text.strip():
        return jsonify(
            {
                "success": True,
                "report_id": report.id,
                "extracted_text": report.extracted_text,
                "processing_status": report.processing_status,
                "reused": True,
            }
        ), 200

    location = get_stored_file_location(report.file_reference, report.patient_id)
    if location is None:
        return jsonify(
            {
                "success": False,
                "error": "The original report file is unavailable for text extraction.",
            }
        ), 422

    upload_root, relative_path = location
    try:
        document_bytes = (Path(upload_root) / relative_path).read_bytes()
    except FileNotFoundError:
        return jsonify(
            {
                "success": False,
                "error": "The original report file could not be found.",
            }
        ), 404
    except OSError:
        current_app.logger.exception(
            "Could not read source file for report %s text extraction",
            report.id,
        )
        return jsonify(
            {"success": False, "error": "Could not read the original report file."}
        ), 500

    filename = PurePosixPath(report.file_reference).name
    try:
        extracted_text = clean_extracted_text(
            extract_document_text(document_bytes, filename)
        )
    except ValueError as exc:
        report.processing_status = (
            "no_text_found"
            if "no readable text found" in str(exc).lower()
            else "extraction_failed"
        )
        db.session.commit()
        return jsonify(
            {
                "success": False,
                "error": "The existing text extraction service could not extract usable text.",
                "processing_status": report.processing_status,
            }
        ), 422
    except RuntimeError:
        current_app.logger.exception(
            "Text extraction failed for report %s",
            report.id,
        )
        report.processing_status = "extraction_failed"
        db.session.commit()
        return jsonify(
            {
                "success": False,
                "error": "The existing text extraction service could not process this report.",
                "processing_status": report.processing_status,
            }
        ), 422

    if not isinstance(extracted_text, str) or not extracted_text.strip():
        report.processing_status = "no_text_found"
        db.session.commit()
        return jsonify(
            {
                "success": False,
                "error": "The existing text extraction service found no readable text.",
                "processing_status": report.processing_status,
            }
        ), 422

    report.extracted_text = extracted_text
    report.processing_status = "processed"
    metadata = extract_report_metadata(extracted_text)
    document_date = (
        report.report_date.isoformat()
        if report.report_date
        else metadata["document_date"]
    )
    parameters = extract_report_parameters(
        extracted_text,
        document_date=document_date,
    )
    persist_report_parameters(report.id, parameters)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception(
            "Could not save extracted text for report %s",
            report.id,
        )
        return jsonify(
            {"success": False, "error": "Could not save extracted report text."}
        ), 500

    return jsonify(
        {
            "success": True,
            "report_id": report.id,
            "extracted_text": report.extracted_text,
            "processing_status": report.processing_status,
            "reused": False,
        }
    ), 200


@reports_bp.post("/reports/compare")
@require_auth
def compare_reports():
    if not request.is_json:
        return jsonify({"success": False, "error": "A JSON request body is required."}), 400
    try:
        data = request.get_json()
    except BadRequest:
        return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400

    report_ids = (data.get("report_id_1"), data.get("report_id_2"))
    if any(
        isinstance(report_id, bool) or not isinstance(report_id, int) or report_id <= 0
        for report_id in report_ids
    ):
        return jsonify(
            {
                "success": False,
                "error": "Exactly two positive integer report IDs are required.",
            }
        ), 400
    if report_ids[0] == report_ids[1]:
        return jsonify(
            {"success": False, "error": "Select two different reports to compare."}
        ), 400

    reports = (
        db.session.query(MedicalReport)
        .join(Patient)
        .filter(
            MedicalReport.id.in_(report_ids),
            Patient.created_by_id == g.current_user.id,
        )
        .all()
    )
    if len(reports) != 2:
        return jsonify({"success": False, "error": "One or both reports were not found."}), 404

    reports_by_id = {report.id: report for report in reports}
    report_1, report_2 = (reports_by_id[report_id] for report_id in report_ids)
    report_texts = (report_1.extracted_text, report_2.extracted_text)
    if any(
        not isinstance(text, str) or not text.strip()
        for text in report_texts
    ):
        return jsonify(
            {
                "success": False,
                "error": "Both reports must have usable extracted text before comparison.",
            }
        ), 422

    ai_summary = None
    ai_error = None
    comparison = None
    fallback_error = None
    def comparison_source_name(report):
        report_date = (
            report.report_date.isoformat()
            if report.report_date
            else "date unavailable"
        )
        report_type = (report.report_type or "Medical report").replace(
            "_",
            " ",
        ).title()
        return (
            f"{report.patient.first_name} {report.patient.last_name} — "
            f"{report_type} — {report_date} "
            f"(report #{report.id})"
        )

    source_names = (
        comparison_source_name(report_1),
        comparison_source_name(report_2),
    )
    try:
        ai_summary = current_app.extensions["ai_service"].compare(
            report_texts[0],
            report_texts[1],
            *source_names,
        )
        if not isinstance(ai_summary, str) or not ai_summary.strip():
            ai_summary = None
            ai_error = "AI provider returned an empty comparison."
        else:
            ai_summary = re.sub(
                r"\b(?:Report|Paper)\s*([12])\b",
                lambda match: source_names[int(match.group(1)) - 1],
                ai_summary,
                flags=re.IGNORECASE,
            )
    except (ValueError, AIServiceError) as exc:
        ai_error = str(exc)
    except Exception:
        current_app.logger.exception(
            "AI report comparison summary failed for reports %s and %s",
            report_1.id,
            report_2.id,
        )
        ai_error = "AI-generated report comparison is temporarily unavailable."

    if ai_summary is None:
        try:
            comparison = compare_report_texts(
                *report_texts,
                similarity_matrix=current_app.extensions[
                    "research_rag_service"
                ].similarity_matrix,
            )
        except Exception:
            current_app.logger.exception(
                "Local fallback comparison failed for reports %s and %s",
                report_1.id,
                report_2.id,
            )
            fallback_error = (
                "The local fallback comparison is temporarily unavailable."
            )

    def report_details(report):
        return {
            "id": report.id,
            "patient_id": report.patient_id,
            "patient_name": f"{report.patient.first_name} {report.patient.last_name}",
            "report_type": report.report_type,
            "report_date": report.report_date.isoformat() if report.report_date else None,
            "metadata": extract_report_metadata(report.extracted_text),
            "processing_status": report.processing_status,
        }

    return jsonify(
        {
            "success": True,
            "reports": [report_details(report_1), report_details(report_2)],
            "comparison": comparison,
            "ai_summary": ai_summary,
            "ai_error": ai_error,
            "fallback_error": fallback_error,
            "disclaimer": COMPARISON_DISCLAIMER,
        }
    ), 200


@reports_bp.post("/reports/compare-local")
@require_auth
def compare_reports_local():
    if not request.is_json:
        return jsonify({"success": False, "error": "A JSON request body is required."}), 400
    try:
        data = request.get_json()
    except BadRequest:
        return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400

    report_ids = (data.get("report_id_1"), data.get("report_id_2"))
    if any(
        isinstance(report_id, bool) or not isinstance(report_id, int) or report_id <= 0
        for report_id in report_ids
    ):
        return jsonify(
            {
                "success": False,
                "error": "Exactly two positive integer report IDs are required.",
            }
        ), 400
    if report_ids[0] == report_ids[1]:
        return jsonify(
            {"success": False, "error": "Select two different reports to compare."}
        ), 400

    reports = (
        db.session.query(MedicalReport)
        .join(Patient)
        .filter(
            MedicalReport.id.in_(report_ids),
            Patient.created_by_id == g.current_user.id,
        )
        .all()
    )
    if len(reports) != 2:
        return jsonify({"success": False, "error": "One or both reports were not found."}), 404

    reports_by_id = {report.id: report for report in reports}
    report_1, report_2 = (reports_by_id[report_id] for report_id in report_ids)
    report_texts = (report_1.extracted_text, report_2.extracted_text)
    if any(not isinstance(text, str) or not text.strip() for text in report_texts):
        return jsonify(
            {
                "success": False,
                "error": "Both reports must have usable extracted text before comparison.",
            }
        ), 422

    def comparison_source_name(report):
        report_date = (
            report.report_date.isoformat()
            if report.report_date
            else "date unavailable"
        )
        report_type = (report.report_type or "Medical report").replace(
            "_",
            " ",
        ).title()
        return (
            f"{report.patient.first_name} {report.patient.last_name} — "
            f"{report_type} — {report_date} "
            f"(report #{report.id})"
        )

    source_names = (
        comparison_source_name(report_1),
        comparison_source_name(report_2),
    )
    prompt = (
        "Compare only the two sources in the supplied context. Refer to each "
        "source by its exact context label; never call them Report 1, Report 2, "
        "Paper 1, or Paper 2. "
        "Keep the two sources separate; "
        "never transfer, repeat, or attribute a finding, measurement, or date from one report to the other. "
        "Do not invent medical information or fabricate trends. For every stated change, identify the "
        "exact source wording from BOTH reports that supports it. If either source does not explicitly "
        "support a comparison, say there is not enough information instead of inferring a change. "
        "Only say a measurement increased or decreased when the same measurement and compatible units "
        "are explicitly present in both reports; preserve each exact value and unit. Only compare dates "
        "that are explicitly present. Compare recommendations and distinguish reported facts from "
        "interpretation. Do not claim to diagnose the patient. Support possible improvement or worsening "
        "only when the exact report text directly supports it. "
        "Use clear section headings such as: Overall changes; New findings; Findings no longer reported; "
        "Changed measurements; Possible improvement / worsening; Recommendations; Important observations. "
        "State that this is an AI-generated comparison for informational purposes; requires human review. "
        "This is not a diagnosis."
    )
    prompt = (
        f"Context:\n{json.dumps(dict(zip(source_names, report_texts)), ensure_ascii=False, indent=2)}"
        f"\n\nRequest:\n{prompt}"
        "\n\nKeep the response to five concise bullets and at most 150 words. "
        "Never repeat a sentence or finding; stop when the summary is complete."
    )

    try:
        comparison = compare_report_texts(
            *report_texts,
            similarity_matrix=current_app.extensions[
                "research_rag_service"
            ].similarity_matrix,
        )
        ai_summary = local_generate(prompt, max_tokens=240)
        summary_sentences = re.split(r"(?<=[.!?])\s+", ai_summary)
        unique_sentences = []
        seen_sentences = set()
        for sentence in summary_sentences:
            normalized = re.sub(r"[\W_]+", " ", sentence).strip().casefold()
            if normalized and normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
        ai_summary = " ".join(unique_sentences)
        ai_summary = re.sub(
            r"\b(?:Report|Paper)\s*([12])\b",
            lambda match: source_names[int(match.group(1)) - 1],
            ai_summary,
            flags=re.IGNORECASE,
        )
    except Exception as exc:
        current_app.logger.exception(
            "Local report comparison failed for reports %s and %s",
            report_1.id,
            report_2.id,
        )
        return jsonify({"success": False, "error": str(exc)}), 502

    def report_details(report):
        return {
            "id": report.id,
            "patient_id": report.patient_id,
            "patient_name": f"{report.patient.first_name} {report.patient.last_name}",
            "report_type": report.report_type,
            "report_date": report.report_date.isoformat() if report.report_date else None,
            "metadata": extract_report_metadata(report.extracted_text),
            "processing_status": report.processing_status,
        }

    return jsonify(
        {
            "success": True,
            "reports": [report_details(report_1), report_details(report_2)],
            "comparison": comparison,
            "ai_summary": ai_summary,
            "ai_error": None,
            "fallback_error": None,
            "disclaimer": COMPARISON_DISCLAIMER,
        }
    ), 200


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

    record_audit_event(
        actor_id=g.current_user.id,
        action="report.view",
        resource_type="report",
        resource_id=report.id,
    )
    upload_root, relative_path = location
    extension = PurePosixPath(relative_path).suffix
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"report-{report.id}{extension}",
    )


@reports_bp.post("/reports/<int:report_id>/summary")
@require_auth
def summarize_report(report_id):
    if request.get_data(cache=True):
        if not request.is_json:
            return jsonify({"success": False, "error": "This endpoint does not accept a request body."}), 400
        try:
            data = request.get_json()
        except BadRequest:
            return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400
        if not isinstance(data, dict) or data:
            return jsonify(
                {
                    "success": False,
                    "error": "This endpoint summarizes the saved report and accepts no input fields.",
                }
            ), 400

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
        return jsonify({"success": False, "error": "Report not found"}), 404
    if not isinstance(report.extracted_text, str) or not report.extracted_text.strip():
        return jsonify(
            {
                "success": False,
                "error": "This report has no extracted text available to summarize.",
            }
        ), 422

    ai_service = current_app.extensions["ai_service"]
    if len(report.extracted_text) > ai_service.MAX_INPUT_CHARACTERS:
        return jsonify(
            {
                "success": False,
                "error": "The extracted report text is too long to summarize in one request.",
            }
        ), 413

    try:
        summary = ai_service.summarize(report.extracted_text)
    except ValueError as exc:
        current_app.logger.info(
            "Report %s summary request exceeded or failed input validation: %s",
            report.id,
            str(exc),
        )
        return jsonify(
            {
                "success": False,
                "error": "The extracted report text could not be accepted by the AI service.",
            }
        ), 413
    except AIServiceError as exc:
        record_audit_event(
            actor_id=g.current_user.id,
            action="ai.report_summary",
            resource_type="report",
            resource_id=report.id,
            status="failure",
        )
        return jsonify({"success": False, "error": str(exc)}), exc.status_code
    except Exception:
        current_app.logger.exception(
            "Unexpected error generating summary for report %s",
            report.id,
        )
        record_audit_event(
            actor_id=g.current_user.id,
            action="ai.report_summary",
            resource_type="report",
            resource_id=report.id,
            status="failure",
        )
        return jsonify(
            {"success": False, "error": "Could not generate a summary for this report."}
        ), 500

    record_audit_event(
        actor_id=g.current_user.id,
        action="ai.report_summary",
        resource_type="report",
        resource_id=report.id,
    )
    return jsonify(
        {
            "success": True,
            "report_id": report.id,
            "summary": summary,
            "disclaimer": SUMMARY_DISCLAIMER,
        }
    ), 200


@reports_bp.delete("/reports/<int:report_id>")
@require_auth
def delete_report(report_id):
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

    file_reference = report.file_reference
    patient_id = report.patient_id
    db.session.delete(report)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Could not delete report %s", report_id)
        return jsonify({"error": "Could not delete report"}), 500

    record_audit_event(
        actor_id=g.current_user.id,
        action="report.delete",
        resource_type="report",
        resource_id=report_id,
    )
    if file_reference:
        try:
            delete_stored_file(file_reference, patient_id)
        except (OSError, ValueError):
            current_app.logger.exception(
                "Report %s was deleted but its uploaded file could not be removed",
                report_id,
            )
            return jsonify(
                {
                    "deleted": True,
                    "warning": "Report was deleted, but its uploaded file could not be removed.",
                }
            ), 200

    return jsonify({"deleted": True}), 200


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
    record_audit_event(
        actor_id=g.current_user.id,
        action="report.upload",
        resource_type="report",
        resource_id=report.id,
        metadata={
            "patient_id": patient.id,
            "report_type": report.report_type,
            "processing_status": report.processing_status,
        },
    )

    return jsonify({"report": _serialize_report(report)}), 201
