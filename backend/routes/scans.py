from datetime import date, datetime
from io import BytesIO

from flask import Blueprint, abort, current_app, g, jsonify, request, send_from_directory
from werkzeug.datastructures import FileStorage

from backend.auth import require_auth
from backend.extensions import db
from backend.models import (
    BrainScanAnalysis,
    CardiacScanAnalysis,
    KneeScanAnalysis,
    Patient,
    ScanAsset,
    SpineScanAnalysis,
)
from backend.services.cardiac_segmentation import (
    CARDIAC_CLASS_COLORS,
    CARDIAC_CLASS_NAMES,
    CardiacModelContractError,
    CardiacModelUnavailableError,
    CardiacSegmentationService,
)
from backend.services.brain_mri import (
    BrainImageValidationError,
    BrainModelContractError,
    BrainModelUnavailableError,
    BrainMRIPredictionService,
)
from backend.services.knee_mri import (
    ACLROILocalizerUnavailableError,
    ACLROINotFoundError,
    KneeImageValidationError,
    KneeModelContractError,
    KneeModelUnavailableError,
    KneeMRIPredictionService,
)
from backend.services.file_storage import (
    get_stored_file_location,
    save_uploaded_file,
    validate_uploaded_file,
)
from backend.services.spine_mri import (
    SpineImageValidationError,
    SpineModelContractError,
    SpineModelUnavailableError,
    SpineMRIPredictionService,
)

scans_bp = Blueprint("scans", __name__)

SUPPORTED_SCAN_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
SUPPORTED_MODALITIES = {"MRI", "CT", "X_RAY", "ULTRASOUND", "OTHER"}
SCAN_SUBTYPES = {
    "MRI": {"Brain", "Knee", "Spine", "Cardiac", "Other"},
    "CT": {"Brain/head", "Chest", "Abdomen/pelvis", "Spine", "Cardiac", "Other"},
    "X_RAY": {"Chest", "Bone/joint", "Spine", "Dental", "Other"},
    "ULTRASOUND": {
        "Abdomen",
        "Obstetric",
        "Cardiac/echocardiogram",
        "Vascular",
        "Thyroid",
        "Other",
    },
    "OTHER": {"Other"},
}


def _serialize_scan(scan):
    is_other = scan.body_region == "Other" or scan.modality == "OTHER"
    analysis = scan.cardiac_analysis
    knee_analysis = scan.knee_analysis
    spine_analysis = scan.spine_analysis
    brain_analysis = scan.brain_analysis
    is_knee_mri = scan.modality == "MRI" and scan.body_region == "Knee"
    is_spine_mri = scan.modality == "MRI" and scan.body_region == "Spine"
    is_brain_mri = scan.modality == "MRI" and scan.body_region == "Brain"
    if knee_analysis:
        preprocessing_status = "complete"
        localization_status = knee_analysis.localization_status
    elif is_knee_mri:
        localization_status = {
            "localizer_unavailable": "unavailable",
            "localization_failed": "failed",
            "roi_localization_unavailable": "unavailable",
            "roi_not_found": "failed",
            "model_unavailable": "localized",
            "analysis_failed": "localized",
        }.get(scan.processing_status, "pending")
        preprocessing_status = (
            "complete"
            if scan.processing_status in {"model_unavailable", "analysis_failed"}
            else "failed"
            if scan.processing_status in {
                "localizer_unavailable",
                "localization_failed",
                "roi_not_found",
                "roi_localization_unavailable",
            }
            else "not_started"
        )
    else:
        preprocessing_status = None
        localization_status = None

    if analysis:
        processing_message = (
            "Experimental AI cardiac MRI segmentation; not a confirmed diagnosis."
        )
    elif knee_analysis:
        processing_message = (
            "Experimental AI knee MRI prediction; not a medical diagnosis."
        )
    elif spine_analysis:
        processing_message = (
            "Experimental Spine MRI model score only; positive-class meaning is not specified."
        )
    elif brain_analysis:
        processing_message = (
            "Experimental Brain MRI model scores only; class label meanings are unspecified."
        )
    elif is_other:
        processing_message = "Stored as Other; it is not eligible for prediction."
    elif scan.processing_status in {
        "localizer_unavailable",
        "roi_localization_unavailable",
    }:
        processing_message = (
            "Scan stored, but the ACL ROI localizer is unavailable; no prediction was made."
        )
    elif scan.processing_status in {"localization_failed", "roi_not_found"}:
        processing_message = (
            "ACL localization failed: no valid ROI component was found; "
            "no classifier prediction was made."
        )
    elif scan.processing_status == "model_unavailable":
        if scan.modality == "MRI" and scan.body_region == "Knee":
            processing_message = "Scan stored, but the ACL ResNet-14 classifier is unavailable."
        elif is_spine_mri:
            processing_message = "Scan stored, but the Spine MRI model is unavailable."
        elif is_brain_mri:
            processing_message = "Scan stored, but the Brain MRI model is unavailable."
        else:
            processing_message = (
                "Scan stored, but the cardiac model or TensorFlow runtime is unavailable."
            )
    elif scan.processing_status == "analysis_failed":
        if is_knee_mri:
            processing_message = "Scan stored, but ACL preprocessing or inference failed; no result is available."
        elif is_spine_mri:
            processing_message = "Scan stored, but Spine MRI preprocessing or inference failed; no score is available."
        elif is_brain_mri:
            processing_message = "Scan stored, but Brain MRI preprocessing or inference failed; no scores are available."
        else:
            processing_message = "Scan stored, but segmentation failed; no result is available."
    else:
        processing_message = "Original scan is stored; awaiting a compatible image-analysis model."

    return {
        "id": scan.id,
        "patient_id": scan.patient_id,
        "modality": scan.modality,
        "body_region": scan.body_region,
        "study_date": scan.study_date.isoformat() if scan.study_date else None,
        "original_filename": scan.original_filename,
        "processing_status": scan.processing_status,
        "processing_message": processing_message,
        "preprocessing_status": preprocessing_status,
        "localization_status": localization_status,
        "original_url": f"/api/scans/{scan.id}/original",
        "analysis": (
            {
                "model_name": analysis.model_name,
                "class_names": list(CARDIAC_CLASS_NAMES),
                "class_colors": CARDIAC_CLASS_COLORS,
                "pixel_counts": analysis.pixel_counts,
                "overlay_url": f"/api/scans/{scan.id}/analysis/overlay",
                "disclaimer": "Experimental AI segmentation; not a confirmed diagnosis.",
            }
            if analysis
            else None
        ),
        "knee_analysis": (
            {
                "model_name": knee_analysis.model_name,
                "slice_index": knee_analysis.slice_index,
                "class_index": knee_analysis.predicted_class_index,
                "class_name": knee_analysis.predicted_class,
                "confidence": knee_analysis.confidence,
                "probabilities": knee_analysis.probabilities,
                "localization_status": knee_analysis.localization_status,
                "roi_box": knee_analysis.roi_box,
                "disclaimer": (
                    "Experimental AI prediction only; not a medical diagnosis. "
                    "Professional medical review is required."
                ),
            }
            if knee_analysis
            else None
        ),
        "spine_analysis": (
            {
                "model_name": spine_analysis.model_name,
                "model_score": spine_analysis.model_score,
                "frame_count": spine_analysis.frame_count,
                "frame_source": spine_analysis.frame_source,
                "disclaimer": (
                    "Experimental model score only; its positive-class meaning is unknown. "
                    "This is not a medical diagnosis."
                ),
            }
            if spine_analysis
            else None
        ),
        "brain_analysis": (
            {
                "model_name": brain_analysis.model_name,
                "predicted_class_index": brain_analysis.predicted_class_index,
                "model_score": brain_analysis.model_score,
                "class_scores": brain_analysis.class_scores,
                "disclaimer": (
                    "Experimental model scores only; output class labels are unspecified. "
                    "This is not a medical diagnosis."
                ),
            }
            if brain_analysis
            else None
        ),
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
    }


def _upload_knee_mri(patient, study_date):
    uploaded_files = request.files.getlist("file")
    if not uploaded_files:
        return jsonify({"error": "Upload at least one knee MRI JPG or PNG slice."}), 400

    validated_files = []
    for uploaded_file in uploaded_files:
        try:
            validated = validate_uploaded_file(uploaded_file)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
            return jsonify({"error": "Knee MRI uploads support JPG and PNG images only."}), 400
        validated_files.append(validated)

    source_images = []
    for uploaded_file in uploaded_files:
        uploaded_file.stream.seek(0)
        source_images.append(uploaded_file.stream.read())
        uploaded_file.stream.seek(0)

    service = KneeMRIPredictionService()
    try:
        image_slices = service.decode_slices(source_images)
    except KneeImageValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    scans = []
    for uploaded_file, validated in zip(uploaded_files, validated_files):
        file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
        scan = ScanAsset(
            patient_id=patient.id,
            modality="MRI",
            body_region="Knee",
            study_date=study_date,
            original_filename=validated["safe_name"],
            file_reference=file_reference,
            processing_status="awaiting_model",
        )
        db.session.add(scan)
        scans.append(scan)
    db.session.flush()

    try:
        predictions = service.predict_image_slices(image_slices)
    except ACLROILocalizerUnavailableError:
        for scan in scans:
            scan.processing_status = "localizer_unavailable"
    except ACLROINotFoundError:
        for scan in scans:
            scan.processing_status = "localization_failed"
    except KneeModelUnavailableError:
        for scan in scans:
            scan.processing_status = "model_unavailable"
    except KneeModelContractError:
        for scan in scans:
            scan.processing_status = "analysis_failed"
    else:
        for scan, prediction in zip(scans, predictions):
            db.session.add(
                KneeScanAnalysis(
                    scan_id=scan.id,
                    model_name="acl_resnet14_best.keras",
                    slice_index=prediction.slice_index,
                    predicted_class_index=prediction.class_index,
                    predicted_class=prediction.class_name,
                    confidence=prediction.confidence,
                    probabilities=prediction.probabilities,
                    localization_status=prediction.localization_status,
                    roi_box=prediction.roi_box,
                )
            )
            scan.processing_status = "analysis_complete"

    db.session.commit()
    serialized_scans = [_serialize_scan(scan) for scan in scans]
    response = {"scans": serialized_scans}
    if len(serialized_scans) == 1:
        response["scan"] = serialized_scans[0]
    return jsonify(response), 201


def _upload_spine_mri(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Spine MRI uploads support JPG and PNG images only."}), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("spine_mri_service")
    if service is None:
        service = SpineMRIPredictionService()
    try:
        model_input = service.preprocess_image(image_bytes)
    except SpineImageValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="MRI",
        body_region="Spine",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status="awaiting_model",
    )
    db.session.add(scan)
    db.session.flush()

    try:
        score = service.predict_preprocessed(model_input)
    except SpineModelUnavailableError:
        scan.processing_status = "model_unavailable"
    except SpineModelContractError:
        scan.processing_status = "analysis_failed"
    else:
        db.session.add(
            SpineScanAnalysis(
                scan_id=scan.id,
                model_name="best_mrnet_fast.keras",
                model_score=score.score,
                frame_count=score.frame_count,
                frame_source=score.frame_source,
            )
        )
        scan.processing_status = "analysis_complete"

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_brain_mri(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Brain MRI uploads support JPG and PNG images only."}), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("brain_mri_service")
    if service is None:
        service = BrainMRIPredictionService()
    try:
        model_input = service.preprocess_image(image_bytes)
    except BrainImageValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="MRI",
        body_region="Brain",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status="awaiting_model",
    )
    db.session.add(scan)
    db.session.flush()

    try:
        prediction = service.predict_preprocessed(model_input)
    except BrainModelUnavailableError:
        scan.processing_status = "model_unavailable"
    except BrainModelContractError:
        scan.processing_status = "analysis_failed"
    else:
        db.session.add(
            BrainScanAnalysis(
                scan_id=scan.id,
                model_name="mri_brain_tumor_efficientnetb0_final.keras",
                predicted_class_index=prediction.class_index,
                model_score=prediction.class_score,
                class_scores=prediction.class_scores,
            )
        )
        scan.processing_status = "analysis_complete"

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _owned_scan(scan_id):
    return (
        db.session.query(ScanAsset)
        .join(Patient)
        .filter(
            ScanAsset.id == scan_id,
            Patient.created_by_id == g.current_user.id,
        )
        .first()
    )


@scans_bp.get("/scans")
@require_auth
def list_scans():
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

    scans = (
        db.session.query(ScanAsset)
        .filter_by(patient_id=patient.id)
        .order_by(ScanAsset.created_at.desc(), ScanAsset.id.desc())
        .all()
    )
    return jsonify({"scans": [_serialize_scan(scan) for scan in scans]}), 200


@scans_bp.post("/scans/upload")
@require_auth
def upload_scan():
    data = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
    try:
        patient_id = int(data.get("patient_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "A valid patient_id is required"}), 400

    patient = (
        db.session.query(Patient)
        .filter_by(id=patient_id, created_by_id=g.current_user.id)
        .first()
    )
    if patient is None:
        return jsonify({"error": "Patient not found"}), 404

    modality_value = data.get("modality", "")
    if not isinstance(modality_value, str):
        return jsonify({"error": "modality must be a supported scan type"}), 400
    modality = modality_value.strip().upper()
    if modality not in SUPPORTED_MODALITIES:
        return jsonify({"error": "modality must be MRI, CT, X_RAY, ULTRASOUND, or OTHER"}), 400

    body_region_value = data.get("body_region", "")
    if not isinstance(body_region_value, str) or len(body_region_value.strip()) > 80:
        return jsonify({"error": "body_region must be 80 characters or fewer"}), 400
    body_region = body_region_value.strip()
    if body_region not in SCAN_SUBTYPES[modality]:
        return jsonify({"error": f"Choose a supported body region for {modality}"}), 400

    is_knee_mri = modality == "MRI" and body_region == "Knee"
    is_spine_mri = modality == "MRI" and body_region == "Spine"
    is_brain_mri = modality == "MRI" and body_region == "Brain"

    study_date_value = data.get("study_date")
    if study_date_value in (None, "", "null"):
        study_date = None
    elif isinstance(study_date_value, str):
        try:
            study_date = datetime.strptime(study_date_value, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "study_date must use YYYY-MM-DD format"}), 400
        if study_date.isoformat() != study_date_value:
            return jsonify({"error": "study_date must use YYYY-MM-DD format"}), 400
        if study_date > date.today():
            return jsonify({"error": "study_date cannot be in the future"}), 400
    else:
        return jsonify({"error": "study_date must use YYYY-MM-DD format or be null"}), 400

    if is_knee_mri:
        return _upload_knee_mri(patient, study_date)
    if is_spine_mri:
        return _upload_spine_mri(patient, study_date)
    if is_brain_mri:
        return _upload_brain_mri(patient, study_date)

    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if validated["extension"] not in SUPPORTED_SCAN_EXTENSIONS:
        return jsonify({"error": "Scan uploads support PDF, JPG, and PNG files"}), 400

    is_cardiac_mri = modality == "MRI" and body_region == "Cardiac"
    is_other = body_region == "Other" or modality == "OTHER"
    if is_cardiac_mri and validated["extension"] not in {".png", ".jpg", ".jpeg"}:
        return jsonify({"error": "Please upload a cardiac MRI image in PNG or JPG format."}), 400

    uploaded_file.stream.seek(0)
    source_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality=modality,
        body_region=body_region,
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status=("unsupported_region" if is_other else "awaiting_model"),
    )
    db.session.add(scan)
    db.session.flush()

    if is_cardiac_mri:
        try:
            result = CardiacSegmentationService().segment(source_bytes)
            overlay_upload = FileStorage(
                stream=BytesIO(result.overlay_png),
                filename="cardiac-segmentation-overlay.png",
                content_type="image/png",
            )
            overlay_reference = save_uploaded_file(
                overlay_upload,
                patient_id=patient.id,
            )
            db.session.add(
                CardiacScanAnalysis(
                    scan_id=scan.id,
                    model_name="best_cardiac_model.keras",
                    pixel_counts=result.pixel_counts,
                    overlay_file_reference=overlay_reference,
                )
            )
            scan.processing_status = "analysis_complete"
        except CardiacModelUnavailableError:
            scan.processing_status = "model_unavailable"
        except (CardiacModelContractError, ValueError):
            scan.processing_status = "analysis_failed"

    db.session.commit()

    return jsonify({"scan": _serialize_scan(scan)}), 201


@scans_bp.get("/scans/<int:scan_id>/analysis/overlay")
@require_auth
def get_cardiac_overlay(scan_id):
    scan = _owned_scan(scan_id)
    if scan is None or scan.cardiac_analysis is None:
        return jsonify({"error": "Cardiac segmentation overlay not found"}), 404

    location = get_stored_file_location(
        scan.cardiac_analysis.overlay_file_reference,
        scan.patient_id,
    )
    if location is None:
        abort(404)

    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"cardiac-scan-{scan.id}-overlay.png",
    )


@scans_bp.get("/scans/<int:scan_id>/original")
@require_auth
def get_original_scan(scan_id):
    scan = _owned_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    location = get_stored_file_location(scan.file_reference, scan.patient_id)
    if location is None:
        abort(404)

    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=scan.original_filename,
    )


__all__ = ["scans_bp"]