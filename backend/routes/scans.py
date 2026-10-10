from datetime import date, datetime
from io import BytesIO

from flask import Blueprint, abort, current_app, g, jsonify, request, send_from_directory
from werkzeug.datastructures import FileStorage

from backend.auth import require_auth
from backend.extensions import db
from backend.models import (
    AbdominalAortaUltrasoundAnalysisRecord,
    BrainScanAnalysis,
    BoneXrayAnalysisRecord,
    CardiacScanAnalysis,
    ChestCTAnalysisRecord,
    ChestXrayAnalysisRecord,
    CTHeadAnalysisRecord,
    DentalXrayAnalysisRecord,
    EchoView47AnalysisRecord,
    KneeScanAnalysis,
    ObstetricUltrasoundAnalysisRecord,
    Patient,
    ScanAsset,
    SpineScanAnalysis,
    VascularUltrasoundAnalysisRecord,
)
from backend.services.cardiac_segmentation import (
    CARDIAC_CLASS_COLORS,
    CARDIAC_CLASS_NAMES,
    CardiacModelContractError,
    CardiacModelUnavailableError,
    CardiacSegmentationService,
)
from backend.services.brain_mri import (
    BRAIN_OUTPUT_NAMES,
    BrainImageValidationError,
    BrainModelContractError,
    BrainModelUnavailableError,
    BrainMRIPredictionService,
)
from backend.services.chest_xray import (
    ChestXrayAnalysisService,
    ChestXrayImageError,
    ChestXrayModelContractError,
    ChestXrayModelUnavailableError,
)
from backend.services.bone_xray import (
    BoneXrayAnalysisService,
    BoneXrayImageError,
    BoneXrayModelContractError,
    BoneXrayModelUnavailableError,
)
from backend.services.dental_xray import (
    DentalXrayAnalysisService,
    DentalXrayImageError,
    DentalXrayModelContractError,
    DentalXrayModelUnavailableError,
)
from backend.services.ct_head_hemorrhage import (
    CTHeadHemorrhageService,
    CTHeadInputError,
    CTHeadModelContractError,
    CTHeadModelUnavailableError,
)
from backend.services.chest_ct_segmentation import (
    ChestCTImageError,
    ChestCTModelContractError,
    ChestCTModelUnavailableError,
    ChestCTSegmentationService,
)
from backend.services.knee_mri import (
    ACLROILocalizerUnavailableError,
    ACLROINotFoundError,
    KneeImageValidationError,
    KneeModelContractError,
    KneeModelUnavailableError,
    KneeMRIPredictionService,
)
from backend.services.spine_mri import (
    SPINE_CONDITIONS,
    SPINE_MODEL_ID,
    SPINE_SEVERITIES,
    SpineImageValidationError,
    SpineModelContractError,
    SpineModelUnavailableError,
    SpineMRIPredictionService,
)
from backend.services.obstetric_ultrasound import (
    MODEL_ID as OBSTETRIC_ULTRASOUND_MODEL_ID,
    ObstetricUltrasoundAnalysisService,
    ObstetricUltrasoundImageError,
    ObstetricUltrasoundModelContractError,
    ObstetricUltrasoundModelUnavailableError,
)
from backend.services.abdominal_aorta_ultrasound import (
    CLASS_NAME as AORTA_CLASS_NAME,
    MODEL_ID as ABDOMINAL_AORTA_ULTRASOUND_MODEL_ID,
    AbdominalAortaUltrasoundAnalysisService,
    AbdominalAortaUltrasoundImageError,
    AbdominalAortaUltrasoundModelContractError,
    AbdominalAortaUltrasoundModelUnavailableError,
)
from backend.services.echoview47 import (
    MODEL_ID as ECHOVIEW47_MODEL_ID,
    EchoView47AnalysisService,
    EchoView47ImageError,
    EchoView47ModelContractError,
    EchoView47ModelUnavailableError,
    describe_view_class,
)
from backend.services.vascular_ultrasound import (
    MODEL_ID as VASCULAR_ULTRASOUND_MODEL_ID,
    VascularUltrasoundAnalysisService,
    VascularUltrasoundImageError,
    VascularUltrasoundModelContractError,
    VascularUltrasoundModelUnavailableError,
)
from backend.services.file_storage import (
    delete_stored_file,
    get_stored_file_location,
    save_uploaded_file,
    validate_uploaded_file,
)
from backend.services.audit_logging import record_audit_event

scans_bp = Blueprint("scans", __name__)

SUPPORTED_SCAN_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
SUPPORTED_MODALITIES = {"MRI", "CT", "X_RAY", "ULTRASOUND", "OTHER"}
SCAN_SUBTYPES = {
    "MRI": {"Brain", "Knee", "Spine", "Cardiac", "Other"},
    "CT": {"Brain/head", "Chest", "Other"},
    "X_RAY": {"Chest", "Bone/joint", "Dental", "Other"},
    "ULTRASOUND": {
        "Abdomen",
        "Obstetric",
        "Cardiac/echocardiogram",
        "Vascular",
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
    chest_xray_analysis = scan.chest_xray_analysis
    bone_xray_analysis = scan.bone_xray_analysis
    dental_xray_analysis = scan.dental_xray_analysis
    ct_head_analysis = scan.ct_head_analysis
    obstetric_ultrasound_analysis = scan.obstetric_ultrasound_analysis
    abdominal_aorta_ultrasound_analysis = (
        scan.abdominal_aorta_ultrasound_analysis
    )
    echoview47_analysis = scan.echoview47_analysis
    vascular_ultrasound_analysis = scan.vascular_ultrasound_analysis
    chest_ct_analysis = scan.chest_ct_analysis
    is_knee_mri = scan.modality == "MRI" and scan.body_region == "Knee"
    is_spine_mri = scan.modality == "MRI" and scan.body_region == "Spine"
    is_brain_mri = scan.modality == "MRI" and scan.body_region == "Brain"
    is_echoview47_ultrasound = (
        scan.modality == "ULTRASOUND"
        and scan.body_region == "Cardiac/echocardiogram"
    )
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

    if is_spine_mri and spine_analysis and spine_analysis.severity_scores:
        processing_message = (
            "Experimental Spine MRI model scores for a single uploaded 2D image. The model "
            "does not analyze a complete MRI series; output is not a confirmed diagnosis."
        )
    elif is_spine_mri:
        processing_message = (
            "Spine MRI scan is stored, but no model output is available for this scan."
        )
    elif scan.modality == "ULTRASOUND" and scan.body_region == "Vascular":
        if vascular_ultrasound_analysis and vascular_ultrasound_analysis.mask_pixel_count:
            processing_message = (
                "Experimental predicted carotid-region mask from one 2D ultrasound "
                "image. It may be inaccurate and does not assess stenosis, plaque, "
                "blockage, DVT, blood flow, or vascular disease."
            )
        elif vascular_ultrasound_analysis:
            processing_message = (
                "The model returned no carotid-region mask. This does not establish "
                "that the artery is absent or abnormal."
            )
        elif scan.processing_status == "model_unavailable":
            processing_message = (
                "Vascular ultrasound image is stored, but the carotid segmentation "
                "model is unavailable."
            )
        elif scan.processing_status == "analysis_failed":
            processing_message = (
                "Vascular ultrasound image is stored, but carotid segmentation failed; "
                "no result is available."
            )
        else:
            processing_message = (
                "Experimental carotid-region segmentation is available for suitable "
                "vascular ultrasound images only."
            )
    elif (
        scan.modality == "ULTRASOUND"
        and scan.body_region == "Abdomen"
        and abdominal_aorta_ultrasound_analysis
    ):
        if abdominal_aorta_ultrasound_analysis.mask_area_pixels:
            processing_message = (
                "Experimental aortic POCUS segmentation from one 2D image. This is not "
                "general abdominal analysis, aneurysm detection, or a diagnosis."
            )
        else:
            processing_message = (
                "The model did not return a segmentation mask. This does not establish "
                "that the aorta is absent or abnormal."
            )
    elif scan.modality == "ULTRASOUND" and scan.body_region == "Abdomen":
        if scan.processing_status == "model_unavailable":
            processing_message = (
                "Abdominal ultrasound image is stored, but the aorta segmentation model "
                "is unavailable."
            )
        elif scan.processing_status == "analysis_failed":
            processing_message = (
                "Abdominal ultrasound image is stored, but aorta segmentation failed; "
                "no result is available."
            )
        else:
            processing_message = (
                "Experimental aortic POCUS segmentation is available for suitable "
                "aorta-view images only; it is not general abdominal analysis."
            )
    elif scan.modality == "ULTRASOUND" and scan.body_region == "Obstetric":
        if obstetric_ultrasound_analysis:
            processing_message = (
                "Experimental fetal ultrasound view classification from one 2D image. "
                "It does not assess fetal health or diagnose a condition."
            )
        elif scan.processing_status == "model_unavailable":
            processing_message = (
                "Obstetric ultrasound image is stored, but the local view-classification "
                "model is unavailable."
            )
        elif scan.processing_status == "analysis_failed":
            processing_message = (
                "Obstetric ultrasound image is stored, but model analysis failed; "
                "no result is available."
            )
        else:
            processing_message = (
                "Obstetric ultrasound image is stored, but no model output is available."
            )
    elif is_echoview47_ultrasound and echoview47_analysis:
        display_label, meaning = describe_view_class(
            echoview47_analysis.predicted_class
        )
        processing_message = (
            f"Experimental still-image view classification. Leading view label: "
            f"{display_label}. {meaning} The result is not a diagnosis or an "
            "assessment of heart function."
        )
    elif is_echoview47_ultrasound:
        if scan.processing_status == "model_unavailable":
            processing_message = (
                "Echocardiogram image is stored, but the local EchoView47 model "
                "is unavailable."
            )
        elif scan.processing_status == "analysis_failed":
            processing_message = (
                "Echocardiogram image is stored, but view classification failed; "
                "no model output is available."
            )
        else:
            processing_message = (
                "Echocardiogram image is stored, but no view-classification "
                "output is available."
            )
    elif analysis:
        processing_message = (
            "Experimental AI cardiac MRI segmentation; not a confirmed diagnosis."
        )
    elif knee_analysis:
        processing_message = (
            "Experimental AI knee MRI prediction; not a medical diagnosis."
        )
    elif brain_analysis:
        processing_message = (
            "Experimental Brain MRI model output with classes mapped to Glioma, Meningioma, "
            "No tumor, and Pituitary tumor in the confirmed training order. Scores are model "
            "outputs, not necessarily calibrated probabilities or confirmed diagnoses."
        )
    elif chest_xray_analysis:
        processing_message = (
            "Experimental Chest X-ray AI/model scores; not confirmed diagnoses."
        )
    elif dental_xray_analysis:
        processing_message = (
            "Experimental Dental X-ray AI detections; not confirmed diagnoses."
        )
    elif ct_head_analysis:
        processing_message = (
            "Experimental Head CT AI/model scores; not confirmed diagnoses."
        )
    elif scan.modality == "CT" and scan.body_region == "Chest":
        if chest_ct_analysis:
            processing_message = (
                "Experimental mask from one chest CT slice; it may be inaccurate "
                "and is not a diagnosis."
            )
        elif scan.processing_status == "model_unavailable":
            processing_message = (
                "Chest CT slice is stored, but the local segmentation model is unavailable."
            )
        elif scan.processing_status == "analysis_failed":
            processing_message = (
                "Chest CT slice is stored, but segmentation failed; no model output is available."
            )
        else:
            processing_message = (
                "Chest CT model analyzes one 2D slice only; no result is available."
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
        elif is_brain_mri:
            processing_message = "Scan stored, but the Brain MRI model is unavailable."
        elif scan.modality == "X_RAY" and scan.body_region == "Chest":
            processing_message = "Chest X-ray stored, but the local pretrained model is unavailable."
        elif scan.modality == "X_RAY" and scan.body_region == "Bone/joint":
            processing_message = "Bone/Joint X-ray stored, but the local pretrained model is unavailable."
        elif scan.modality == "X_RAY" and scan.body_region == "Dental":
            processing_message = "Dental X-ray stored, but the OralGuard detector is unavailable."
        elif scan.modality == "CT" and scan.body_region == "Brain/head":
            processing_message = "Head CT stored, but the local pretrained model is unavailable."
        else:
            processing_message = (
                "Scan stored, but the cardiac model or TensorFlow runtime is unavailable."
            )
    elif scan.processing_status == "analysis_failed":
        if is_knee_mri:
            processing_message = "Scan stored, but ACL preprocessing or inference failed; no result is available."
        elif is_brain_mri:
            processing_message = "Scan stored, but Brain MRI preprocessing or inference failed; no scores are available."
        elif scan.modality == "X_RAY" and scan.body_region == "Chest":
            processing_message = "Chest X-ray stored, but model inference failed; no scores are available."
        elif scan.modality == "X_RAY" and scan.body_region == "Dental":
            processing_message = "Dental X-ray stored, but OralGuard inference failed; no findings are available."
        elif scan.modality == "CT" and scan.body_region == "Brain/head":
            processing_message = "Head CT stored, but model inference failed; no analysis is available."
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
                "condition_scores": spine_analysis.severity_scores,
                "conditions": list(SPINE_CONDITIONS),
                "severities": list(SPINE_SEVERITIES),
                "input_format": "single 2D image",
                "disclaimer": (
                    "Experimental AI/model output from a single uploaded 2D image, not a "
                    "complete MRI series. Scores may not be calibrated probabilities and "
                    "are independent model scores, not normalized category probabilities. "
                    "This is not a confirmed medical diagnosis. Professional radiology review "
                    "is required."
                ),
            }
            if spine_analysis and spine_analysis.severity_scores
            else None
        ),
        "brain_analysis": (
            {
                "model_name": brain_analysis.model_name,
                "predicted_class_index": brain_analysis.predicted_class_index,
                "predicted_class": BRAIN_OUTPUT_NAMES[
                    brain_analysis.predicted_class_index
                ],
                "model_score": brain_analysis.model_score,
                "class_scores": {
                    name: brain_analysis.class_scores.get(
                        name,
                        brain_analysis.class_scores.get(f"Class {index}"),
                    )
                    for index, name in enumerate(BRAIN_OUTPUT_NAMES)
                    if name in brain_analysis.class_scores
                    or f"Class {index}" in brain_analysis.class_scores
                },
                "class_mapping_available": True,
                "disclaimer": (
                    "Experimental AI/model output. Class names follow the confirmed training "
                    "order: Glioma, Meningioma, No tumor, and Pituitary tumor. Scores are "
                    "not necessarily calibrated probabilities. This is not a medical diagnosis."
                ),
            }
            if brain_analysis
            else None
        ),
        "chest_xray_analysis": (
            {
                "model_name": chest_xray_analysis.model_name,
                "findings": chest_xray_analysis.findings,
                "disclaimer": (
                    "AI/model scores from a research model, not confirmed diagnoses. "
                    "Professional medical review is required."
                ),
            }
            if chest_xray_analysis
            else None
        ),
        "bone_xray_analysis": (
            {
                "model_name": bone_xray_analysis.model_name,
                "predicted_label": bone_xray_analysis.predicted_label,
                "confidence_score": bone_xray_analysis.confidence_score,
                "disclaimer": (
                    "AI prediction from a research model, not a confirmed diagnosis. "
                    "Professional medical review is required."
                ),
            }
            if bone_xray_analysis
            else None
        ),
        "dental_xray_analysis": (
            {
                "model_name": dental_xray_analysis.model_name,
                "findings": dental_xray_analysis.findings,
                "image_shape": dental_xray_analysis.image_shape,
                "disclaimer": (
                    "AI detections from a research model (OralGuard), not confirmed diagnoses. "
                    "Professional dental review is required."
                ),
            }
            if dental_xray_analysis
            else None
        ),
        "ct_head_analysis": (
            {
                "model_name": ct_head_analysis.model_name,
                "series_classification": ct_head_analysis.series_classification,
                "slice_classification": ct_head_analysis.slice_classification,
                "slice_count": ct_head_analysis.slice_count,
                "highest_any_slice_index": ct_head_analysis.highest_any_slice_index,
                "input_format": ct_head_analysis.input_format,
                "localization_url": (
                    f"/api/scans/{scan.id}/analysis/ct-localization"
                    if ct_head_analysis.localization_file_reference
                    else None
                ),
                "disclaimer": (
                    "AI/model output from an experimental research model, NOT a confirmed "
                    "medical diagnosis. The localization map is qualitative only; its green "
                    "outline encloses pixels at or above a 0.5 model-output threshold and "
                    "is not a verified lesion boundary."
                ),
            }
            if ct_head_analysis
            else None
        ),
        "chest_ct_analysis": (
            {
                "model_name": chest_ct_analysis.model_name,
                "pixel_counts": chest_ct_analysis.pixel_counts,
                "image_shape": chest_ct_analysis.image_shape,
                "threshold": chest_ct_analysis.threshold,
                "overlay_url": (
                    f"/api/scans/{scan.id}/analysis/chest-ct-overlay"
                    if chest_ct_analysis.overlay_file_reference
                    else None
                ),
                "finding_meanings": {
                    "Ground-glass opacity": (
                        "A hazy-appearing region pattern. The model mask does not "
                        "identify its cause or establish a disease."
                    ),
                    "Consolidation": (
                        "A denser-appearing region pattern. The model mask does not "
                        "identify its cause or establish a disease."
                    ),
                },
                "disclaimer": (
                    "This is an experimental mask from one 2D CT slice and may be "
                    "inaccurate. Pixel counts are not probabilities, physical "
                    "measurements, severity, COVID status, or a diagnosis."
                ),
            }
            if chest_ct_analysis
            else None
        ),
        "obstetric_ultrasound_analysis": (
            {
                "model_name": obstetric_ultrasound_analysis.model_name,
                "predicted_class": obstetric_ultrasound_analysis.predicted_class,
                "model_score": obstetric_ultrasound_analysis.model_score,
                "class_scores": obstetric_ultrasound_analysis.class_scores,
                "disclaimer": (
                    "Experimental classification of the uploaded 2D ultrasound image's "
                    "view only. Scores are not calibrated confidence estimates. This does "
                    "not assess fetal health or diagnose a condition."
                ),
            }
            if obstetric_ultrasound_analysis
            else None
        ),
        "abdominal_aorta_ultrasound_analysis": (
            {
                "model_name": abdominal_aorta_ultrasound_analysis.model_name,
                "class_name": AORTA_CLASS_NAME,
                "confidence_score": (
                    abdominal_aorta_ultrasound_analysis.confidence_score
                ),
                "mask_area_pixels": (
                    abdominal_aorta_ultrasound_analysis.mask_area_pixels
                ),
                "image_shape": abdominal_aorta_ultrasound_analysis.image_shape,
                "has_mask": bool(
                    abdominal_aorta_ultrasound_analysis.mask_area_pixels
                    and abdominal_aorta_ultrasound_analysis.overlay_file_reference
                ),
                "overlay_url": (
                    f"/api/scans/{scan.id}/analysis/aorta-overlay"
                    if abdominal_aorta_ultrasound_analysis.overlay_file_reference
                    else None
                ),
                "disclaimer": (
                    "Experimental segmentation of aorta-like anatomy in suitable POCUS "
                    "images only. This is not aneurysm detection, general abdominal "
                    "analysis, or a confirmed diagnosis."
                ),
            }
            if abdominal_aorta_ultrasound_analysis
            else None
        ),
        "vascular_ultrasound_analysis": (
            {
                "model_name": vascular_ultrasound_analysis.model_name,
                "label": "Predicted carotid artery region",
                "meaning": (
                    "The highlighted area contains pixels labeled by the model as "
                    "carotid-like anatomy; it is not an assessment of vessel disease."
                ),
                "has_mask": bool(
                    vascular_ultrasound_analysis.mask_pixel_count
                    and vascular_ultrasound_analysis.overlay_file_reference
                ),
                "overlay_url": (
                    f"/api/scans/{scan.id}/analysis/vascular-ultrasound-overlay"
                    if vascular_ultrasound_analysis.overlay_file_reference
                    else None
                ),
                "disclaimer": (
                    "Experimental mask that may be inaccurate. It does not diagnose "
                    "stenosis, plaque, blockage, DVT, or vascular disease, and does "
                    "not assess blood flow."
                ),
            }
            if vascular_ultrasound_analysis
            else None
        ),
        "echoview47_analysis": (
            {
                "model_name": echoview47_analysis.model_name,
                "predicted_class": echoview47_analysis.predicted_class,
                "display_label": describe_view_class(
                    echoview47_analysis.predicted_class
                )[0],
                "meaning": describe_view_class(
                    echoview47_analysis.predicted_class
                )[1],
                "model_score": echoview47_analysis.model_score,
                "top_views": [
                    {
                        "predicted_class": class_name,
                        "display_label": describe_view_class(class_name)[0],
                        "meaning": describe_view_class(class_name)[1],
                        "model_score": score,
                    }
                    for class_name, score in sorted(
                        echoview47_analysis.class_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                ],
                "disclaimer": (
                    "Experimental classification of the echocardiogram view in "
                    "one still image only. The leading label and scores are "
                    "uncalibrated model outputs, not measures of correctness. "
                    "This does not assess heart function, identify disease, or "
                    "establish that the heart is normal."
                ),
            }
            if echoview47_analysis
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
        prediction = service.predict_preprocessed(model_input)
    except SpineModelUnavailableError:
        scan.processing_status = "model_unavailable"
    except SpineModelContractError:
        scan.processing_status = "analysis_failed"
    else:
        db.session.add(
            SpineScanAnalysis(
                scan_id=scan.id,
                model_name=SPINE_MODEL_ID,
                model_score=prediction.highest_score,
                severity_scores=prediction.condition_scores,
                frame_count=1,
                frame_source="single_uploaded_2d_image",
            )
        )
        scan.processing_status = "analysis_complete"

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_obstetric_ultrasound(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify(
            {"error": "Obstetric ultrasound uploads support JPG and PNG images only."}
        ), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("obstetric_ultrasound_service")
    if service is None:
        service = ObstetricUltrasoundAnalysisService()
    try:
        image = service.prepare_image(image_bytes)
    except ObstetricUltrasoundImageError as exc:
        return jsonify({"error": str(exc)}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="ULTRASOUND",
        body_region="Obstetric",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status="awaiting_model",
    )
    db.session.add(scan)
    db.session.flush()

    try:
        prediction = service.analyze_prepared_image(image)
    except ObstetricUltrasoundModelUnavailableError as exc:
        current_app.logger.warning("Obstetric ultrasound model unavailable: %s", exc)
        scan.processing_status = "model_unavailable"
    except ObstetricUltrasoundModelContractError as exc:
        current_app.logger.exception("Obstetric ultrasound model contract failure: %s", exc)
        scan.processing_status = "analysis_failed"
    else:
        db.session.add(
            ObstetricUltrasoundAnalysisRecord(
                scan_id=scan.id,
                model_name=OBSTETRIC_ULTRASOUND_MODEL_ID,
                predicted_class=prediction.predicted_class,
                model_score=prediction.model_score,
                class_scores=prediction.class_scores,
            )
        )
        scan.processing_status = "analysis_complete"

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_abdominal_aorta_ultrasound(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify(
            {"error": "Abdominal aorta ultrasound uploads support JPG and PNG images only."}
        ), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("abdominal_aorta_ultrasound_service")
    if service is None:
        service = AbdominalAortaUltrasoundAnalysisService()
    try:
        image = service.prepare_image(image_bytes)
    except AbdominalAortaUltrasoundImageError as exc:
        return jsonify({"error": str(exc)}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="ULTRASOUND",
        body_region="Abdomen",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status="awaiting_model",
    )
    db.session.add(scan)
    db.session.flush()

    try:
        prediction = service.analyze_prepared_image(image)
    except AbdominalAortaUltrasoundModelUnavailableError as exc:
        current_app.logger.warning("Abdominal aorta model unavailable: %s", exc)
        scan.processing_status = "model_unavailable"
    except AbdominalAortaUltrasoundModelContractError as exc:
        current_app.logger.exception("Abdominal aorta model contract failure: %s", exc)
        scan.processing_status = "analysis_failed"
    else:
        overlay_reference = None
        if prediction.overlay_png is not None:
            overlay_upload = FileStorage(
                stream=BytesIO(prediction.overlay_png),
                filename="abdominal-aorta-segmentation-overlay.png",
                content_type="image/png",
            )
            overlay_reference = save_uploaded_file(
                overlay_upload,
                patient_id=patient.id,
            )
        db.session.add(
            AbdominalAortaUltrasoundAnalysisRecord(
                scan_id=scan.id,
                model_name=ABDOMINAL_AORTA_ULTRASOUND_MODEL_ID,
                confidence_score=prediction.confidence_score,
                mask_area_pixels=prediction.mask_area_pixels,
                image_shape=list(prediction.image_shape),
                overlay_file_reference=overlay_reference,
            )
        )
        scan.processing_status = "analysis_complete"

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_vascular_ultrasound(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify(
            {"error": "Vascular ultrasound uploads support JPG and PNG images only."}
        ), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("vascular_ultrasound_service")
    if service is None:
        service = VascularUltrasoundAnalysisService()
    try:
        image = service.prepare_image(image_bytes)
    except VascularUltrasoundImageError as exc:
        return jsonify({"error": str(exc)}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="ULTRASOUND",
        body_region="Vascular",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status="awaiting_model",
    )
    db.session.add(scan)
    db.session.flush()

    try:
        prediction = service.analyze_prepared_image(image)
    except VascularUltrasoundModelUnavailableError as exc:
        current_app.logger.warning("Carotid ultrasound model unavailable: %s", exc)
        scan.processing_status = "model_unavailable"
    except VascularUltrasoundModelContractError as exc:
        current_app.logger.error("Carotid ultrasound model contract failure: %s", exc)
        scan.processing_status = "analysis_failed"
    else:
        overlay_reference = None
        if prediction.overlay_png is not None:
            overlay_upload = FileStorage(
                stream=BytesIO(prediction.overlay_png),
                filename="carotid-ultrasound-segmentation-overlay.png",
                content_type="image/png",
            )
            overlay_reference = save_uploaded_file(
                overlay_upload,
                patient_id=patient.id,
            )
        db.session.add(
            VascularUltrasoundAnalysisRecord(
                scan_id=scan.id,
                model_name=VASCULAR_ULTRASOUND_MODEL_ID,
                mask_pixel_count=prediction.mask_pixel_count,
                image_shape=list(prediction.image_shape),
                overlay_file_reference=overlay_reference,
            )
        )
        scan.processing_status = "analysis_complete"

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_echoview47_ultrasound(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify(
            {"error": "Echocardiogram view classification supports JPG and PNG images only."}
        ), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("echoview47_service")
    if service is None:
        service = EchoView47AnalysisService()
    try:
        image = service.prepare_image(image_bytes)
    except EchoView47ImageError as exc:
        return jsonify({"error": str(exc)}), 400

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="ULTRASOUND",
        body_region="Cardiac/echocardiogram",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status="awaiting_model",
    )
    db.session.add(scan)
    db.session.flush()

    try:
        prediction = service.analyze_prepared_image(image)
    except EchoView47ModelUnavailableError as exc:
        current_app.logger.warning("EchoView47 model unavailable: %s", exc)
        scan.processing_status = "model_unavailable"
    except EchoView47ModelContractError as exc:
        current_app.logger.exception("EchoView47 model contract failure: %s", exc)
        scan.processing_status = "analysis_failed"
    else:
        db.session.add(
            EchoView47AnalysisRecord(
                scan_id=scan.id,
                model_name=ECHOVIEW47_MODEL_ID,
                predicted_class=prediction.predicted_class,
                model_score=prediction.model_score,
                class_scores=prediction.class_scores,
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


def _upload_chest_xray(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Chest X-ray uploads support JPG and PNG images only."}), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("chest_xray_service")
    if service is None:
        service = ChestXrayAnalysisService()
    try:
        result = service.analyze_bytes(image_bytes)
    except ChestXrayImageError as exc:
        return jsonify({"error": str(exc)}), 400
    except ChestXrayModelUnavailableError:
        result = None
        processing_status = "model_unavailable"
    except ChestXrayModelContractError:
        result = None
        processing_status = "analysis_failed"
    else:
        processing_status = "analysis_complete"

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="X_RAY",
        body_region="Chest",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status=processing_status,
    )
    db.session.add(scan)
    db.session.flush()

    if result is not None:
        db.session.add(
            ChestXrayAnalysisRecord(
                scan_id=scan.id,
                model_name=result.model_name,
                findings=[
                    {"name": finding.name, "score": finding.score}
                    for finding in result.findings
                ],
                input_shape=list(result.input_shape),
            )
        )
    db.session.commit()

    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_bone_xray(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Bone/Joint X-ray uploads support JPG and PNG images only."}), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("bone_xray_service")
    if service is None:
        service = BoneXrayAnalysisService()
    try:
        result = service.analyze_bytes(image_bytes)
    except BoneXrayImageError as exc:
        return jsonify({"error": str(exc)}), 400
    except BoneXrayModelUnavailableError:
        result = None
        processing_status = "model_unavailable"
    except BoneXrayModelContractError:
        result = None
        processing_status = "analysis_failed"
    else:
        processing_status = "analysis_complete"

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="X_RAY",
        body_region="Bone/joint",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status=processing_status,
    )
    db.session.add(scan)
    db.session.flush()

    if result is not None:
        db.session.add(
            BoneXrayAnalysisRecord(
                scan_id=scan.id,
                model_name="yakshpanchal/bone-fracture-resnet50",
                predicted_label=result.label,
                confidence_score=result.confidence,
            )
        )
    db.session.commit()

    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_dental_xray(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if validated["extension"] not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Dental X-ray uploads support JPG and PNG images only."}), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("dental_xray_service")
    if service is None:
        service = DentalXrayAnalysisService()
    try:
        result = service.analyze_bytes(image_bytes)
    except DentalXrayImageError as exc:
        return jsonify({"error": str(exc)}), 400
    except DentalXrayModelUnavailableError:
        result = None
        processing_status = "model_unavailable"
    except DentalXrayModelContractError:
        result = None
        processing_status = "analysis_failed"
    else:
        processing_status = "analysis_complete"

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="X_RAY",
        body_region="Dental",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status=processing_status,
    )
    db.session.add(scan)
    db.session.flush()

    if result is not None:
        db.session.add(
            DentalXrayAnalysisRecord(
                scan_id=scan.id,
                model_name=result.model_name,
                findings=[
                    {
                        "name": finding.name,
                        "score": finding.score,
                        "box": list(finding.box),
                    }
                    for finding in result.findings
                ],
                image_shape=list(result.image_shape),
            )
        )
    db.session.commit()

    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_ct_head(patient, study_date):
    uploaded_file = request.files.get("file")
    max_ct_upload_size = current_app.config["MAX_CT_UPLOAD_SIZE"]
    try:
        validated = validate_uploaded_file(
            uploaded_file,
            max_size_bytes=max_ct_upload_size,
            allowed_extensions={".zip", ".nii", ".gz"},
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    filename = validated["safe_name"]
    if validated["extension"] not in {".zip", ".nii", ".gz"}:
        return jsonify(
            {
                "error": (
                    "Head CT uploads require a zipped DICOM series or a 3D .nii/.nii.gz volume."
                )
            }
        ), 400
    if validated["extension"] == ".gz" and not filename.lower().endswith(".nii.gz"):
        return jsonify({"error": "Gzip CT uploads must use the .nii.gz extension."}), 400

    uploaded_file.stream.seek(0)
    source_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("ct_head_hemorrhage_service")
    if service is None:
        service = CTHeadHemorrhageService()

    try:
        result = service.analyze_bytes(source_bytes, filename)
    except CTHeadInputError as exc:
        return jsonify({"error": str(exc)}), 400
    except CTHeadModelUnavailableError as exc:
        current_app.logger.warning("Head CT model unavailable: %s", exc)
        result = None
        processing_status = "model_unavailable"
    except CTHeadModelContractError as exc:
        current_app.logger.error("Head CT model contract failure: %s", exc)
        result = None
        processing_status = "analysis_failed"
    else:
        processing_status = "analysis_complete"

    file_reference = save_uploaded_file(
        uploaded_file,
        patient_id=patient.id,
        max_size_bytes=max_ct_upload_size,
        allowed_extensions={".zip", ".nii", ".gz"},
    )
    scan = ScanAsset(
        patient_id=patient.id,
        modality="CT",
        body_region="Brain/head",
        study_date=study_date,
        original_filename=filename,
        file_reference=file_reference,
        processing_status=processing_status,
    )
    db.session.add(scan)
    db.session.flush()

    if result is not None:
        localization_reference = None
        if result.localization_png is not None:
            localization_upload = FileStorage(
                stream=BytesIO(result.localization_png),
                filename="ct-head-localization.png",
                content_type="image/png",
            )
            localization_reference = save_uploaded_file(
                localization_upload,
                patient_id=patient.id,
            )
        db.session.add(
            CTHeadAnalysisRecord(
                scan_id=scan.id,
                model_name=result.model_name,
                series_classification=result.series_classification,
                slice_classification=result.slice_classification,
                slice_count=result.slice_count,
                highest_any_slice_index=result.highest_any_slice_index,
                input_format=result.input_format,
                localization_file_reference=localization_reference,
            )
        )

    db.session.commit()
    return jsonify({"scan": _serialize_scan(scan)}), 201


def _upload_ct_chest(patient, study_date):
    uploaded_file = request.files.get("file")
    try:
        validated = validate_uploaded_file(
            uploaded_file,
            allowed_extensions={".jpg", ".jpeg", ".png"},
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    uploaded_file.stream.seek(0)
    image_bytes = uploaded_file.stream.read()
    uploaded_file.stream.seek(0)
    service = current_app.extensions.get("chest_ct_segmentation_service")
    if service is None:
        service = ChestCTSegmentationService()

    try:
        result = service.analyze_bytes(image_bytes)
    except ChestCTImageError as exc:
        return jsonify({"error": str(exc)}), 400
    except ChestCTModelUnavailableError as exc:
        current_app.logger.warning("Chest CT model unavailable: %s", exc)
        result = None
        processing_status = "model_unavailable"
    except ChestCTModelContractError as exc:
        current_app.logger.error("Chest CT model contract failure: %s", exc)
        result = None
        processing_status = "analysis_failed"
    else:
        processing_status = "analysis_complete"

    file_reference = save_uploaded_file(uploaded_file, patient_id=patient.id)
    scan = ScanAsset(
        patient_id=patient.id,
        modality="CT",
        body_region="Chest",
        study_date=study_date,
        original_filename=validated["safe_name"],
        file_reference=file_reference,
        processing_status=processing_status,
    )
    db.session.add(scan)
    db.session.flush()

    if result is not None:
        overlay_upload = FileStorage(
            stream=BytesIO(result.overlay_png),
            filename="chest-ct-segmentation-overlay.png",
            content_type="image/png",
        )
        overlay_reference = save_uploaded_file(
            overlay_upload,
            patient_id=patient.id,
        )
        db.session.add(
            ChestCTAnalysisRecord(
                scan_id=scan.id,
                model_name=result.model_name,
                pixel_counts=result.pixel_counts,
                image_shape=list(result.image_shape),
                threshold=result.threshold,
                overlay_file_reference=overlay_reference,
            )
        )

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
    record_audit_event(
        actor_id=g.current_user.id,
        action="scan.list",
        resource_type="patient",
        resource_id=patient.id,
        metadata={"count": len(scans)},
    )
    return jsonify({"scans": [_serialize_scan(scan) for scan in scans]}), 200


def _record_scan_upload(response, patient, modality, body_region):
    response_body, response_status = response
    if response_status != 201:
        return response

    payload = response_body.get_json()
    scans = payload.get("scans")
    if scans is None:
        scan = payload.get("scan")
        scans = [scan] if scan is not None else []

    for scan in scans:
        record_audit_event(
            actor_id=g.current_user.id,
            action="scan.upload",
            resource_type="scan",
            resource_id=scan["id"],
            metadata={
                "patient_id": patient.id,
                "modality": modality,
                "body_region": body_region,
                "processing_status": scan["processing_status"],
            },
        )
        if scan["processing_status"] not in {"awaiting_model", "unsupported_region"}:
            record_audit_event(
                actor_id=g.current_user.id,
                action="ai.scan_analysis",
                resource_type="scan",
                resource_id=scan["id"],
                status=(
                    "success"
                    if scan["processing_status"] == "analysis_complete"
                    else "failure"
                ),
                metadata={"processing_status": scan["processing_status"]},
            )
    return response


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
    is_chest_xray = modality == "X_RAY" and body_region == "Chest"
    is_bone_xray = modality == "X_RAY" and body_region == "Bone/joint"
    is_dental_xray = modality == "X_RAY" and body_region == "Dental"
    is_ct_head = modality == "CT" and body_region == "Brain/head"
    is_ct_chest = modality == "CT" and body_region == "Chest"
    is_obstetric_ultrasound = (
        modality == "ULTRASOUND" and body_region == "Obstetric"
    )
    is_abdominal_aorta_ultrasound = (
        modality == "ULTRASOUND" and body_region == "Abdomen"
    )
    is_vascular_ultrasound = (
        modality == "ULTRASOUND" and body_region == "Vascular"
    )
    is_echoview47_ultrasound = (
        modality == "ULTRASOUND"
        and body_region == "Cardiac/echocardiogram"
    )

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

    upload_handler = None
    if is_knee_mri:
        upload_handler = _upload_knee_mri
    elif is_spine_mri:
        upload_handler = _upload_spine_mri
    elif is_brain_mri:
        upload_handler = _upload_brain_mri
    elif is_chest_xray:
        upload_handler = _upload_chest_xray
    elif is_bone_xray:
        upload_handler = _upload_bone_xray
    elif is_dental_xray:
        upload_handler = _upload_dental_xray
    elif is_ct_head:
        upload_handler = _upload_ct_head
    elif is_ct_chest:
        upload_handler = _upload_ct_chest
    elif is_obstetric_ultrasound:
        upload_handler = _upload_obstetric_ultrasound
    elif is_abdominal_aorta_ultrasound:
        upload_handler = _upload_abdominal_aorta_ultrasound
    elif is_vascular_ultrasound:
        upload_handler = _upload_vascular_ultrasound
    elif is_echoview47_ultrasound:
        upload_handler = _upload_echoview47_ultrasound
    if upload_handler is not None:
        return _record_scan_upload(
            upload_handler(patient, study_date),
            patient,
            modality,
            body_region,
        )

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

    return _record_scan_upload(
        (jsonify({"scan": _serialize_scan(scan)}), 201),
        patient,
        modality,
        body_region,
    )


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

    record_audit_event(
        actor_id=g.current_user.id,
        action="ai.scan_analysis.view",
        resource_type="scan",
        resource_id=scan.id,
        metadata={"output_type": "cardiac_overlay"},
    )
    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"cardiac-scan-{scan.id}-overlay.png",
    )


@scans_bp.get("/scans/<int:scan_id>/analysis/aorta-overlay")
@require_auth
def get_abdominal_aorta_overlay(scan_id):
    scan = _owned_scan(scan_id)
    analysis = (
        scan.abdominal_aorta_ultrasound_analysis
        if scan is not None
        else None
    )
    if analysis is None or not analysis.overlay_file_reference:
        return jsonify({"error": "Abdominal aorta segmentation overlay not found"}), 404

    location = get_stored_file_location(
        analysis.overlay_file_reference,
        scan.patient_id,
    )
    if location is None:
        abort(404)

    record_audit_event(
        actor_id=g.current_user.id,
        action="ai.scan_analysis.view",
        resource_type="scan",
        resource_id=scan.id,
        metadata={"output_type": "aorta_overlay"},
    )
    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"abdominal-aorta-scan-{scan.id}-overlay.png",
    )


@scans_bp.get("/scans/<int:scan_id>/analysis/vascular-ultrasound-overlay")
@require_auth
def get_vascular_ultrasound_overlay(scan_id):
    scan = _owned_scan(scan_id)
    analysis = (
        scan.vascular_ultrasound_analysis
        if scan is not None
        else None
    )
    if analysis is None or not analysis.overlay_file_reference:
        return jsonify({"error": "Carotid ultrasound overlay not found"}), 404

    location = get_stored_file_location(
        analysis.overlay_file_reference,
        scan.patient_id,
    )
    if location is None:
        abort(404)

    record_audit_event(
        actor_id=g.current_user.id,
        action="ai.scan_analysis.view",
        resource_type="scan",
        resource_id=scan.id,
        metadata={"output_type": "vascular_ultrasound_overlay"},
    )
    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"carotid-ultrasound-scan-{scan.id}-overlay.png",
    )


@scans_bp.get("/scans/<int:scan_id>/analysis/ct-localization")
@require_auth
def get_ct_localization(scan_id):
    scan = _owned_scan(scan_id)
    analysis = scan.ct_head_analysis if scan is not None else None
    if analysis is None or not analysis.localization_file_reference:
        return jsonify({"error": "Head CT localization map not found"}), 404

    location = get_stored_file_location(
        analysis.localization_file_reference,
        scan.patient_id,
    )
    if location is None:
        abort(404)

    record_audit_event(
        actor_id=g.current_user.id,
        action="ai.scan_analysis.view",
        resource_type="scan",
        resource_id=scan.id,
        metadata={"output_type": "ct_localization"},
    )
    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"ct-head-{scan.id}-localization.png",
    )


@scans_bp.get("/scans/<int:scan_id>/analysis/chest-ct-overlay")
@require_auth
def get_chest_ct_overlay(scan_id):
    scan = _owned_scan(scan_id)
    analysis = scan.chest_ct_analysis if scan is not None else None
    if analysis is None or not analysis.overlay_file_reference:
        return jsonify({"error": "Chest CT segmentation overlay not found"}), 404

    location = get_stored_file_location(
        analysis.overlay_file_reference,
        scan.patient_id,
    )
    if location is None:
        abort(404)

    record_audit_event(
        actor_id=g.current_user.id,
        action="ai.scan_analysis.view",
        resource_type="scan",
        resource_id=scan.id,
        metadata={"output_type": "chest_ct_overlay"},
    )
    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=f"chest-ct-{scan.id}-segmentation.png",
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

    record_audit_event(
        actor_id=g.current_user.id,
        action="scan.view",
        resource_type="scan",
        resource_id=scan.id,
    )
    upload_root, relative_path = location
    return send_from_directory(
        upload_root,
        relative_path,
        as_attachment=False,
        download_name=scan.original_filename,
    )


@scans_bp.delete("/scans/<int:scan_id>")
@require_auth
def delete_scan(scan_id):
    scan = _owned_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    patient_id = scan.patient_id
    file_references = {scan.file_reference}
    for relationship in scan.__mapper__.relationships:
        if relationship.key == "patient":
            continue
        analysis = getattr(scan, relationship.key)
        if analysis is None or not hasattr(analysis, "__table__"):
            continue
        file_references.update(
            getattr(analysis, column.name)
            for column in analysis.__table__.columns
            if column.name.endswith("_file_reference")
            and getattr(analysis, column.name)
        )

    db.session.delete(scan)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Could not delete scan %s", scan_id)
        return jsonify({"error": "Could not delete scan"}), 500

    record_audit_event(
        actor_id=g.current_user.id,
        action="scan.delete",
        resource_type="scan",
        resource_id=scan_id,
    )
    cleanup_failed = False
    for file_reference in file_references:
        try:
            delete_stored_file(file_reference, patient_id)
        except (OSError, ValueError):
            cleanup_failed = True
            current_app.logger.exception(
                "Scan %s was deleted but an uploaded file could not be removed",
                scan_id,
            )

    if cleanup_failed:
        return jsonify(
            {
                "deleted": True,
                "warning": "Scan was deleted, but one or more uploaded files could not be removed.",
            }
        ), 200
    return jsonify({"deleted": True}), 200


__all__ = ["scans_bp"]