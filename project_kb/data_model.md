# Data model

Database models are defined in `backend/models/`. These records are persisted by SQLAlchemy; patient information is scoped through the patient owner.

## AbdominalAortaUltrasoundAnalysisRecord

Database table: `abdominal_aorta_ultrasound_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `confidence_score` (db.Float), `mask_area_pixels` (db.Integer), `image_shape` (db.JSON), `overlay_file_reference` (db.String(500)), `created_at` (db.DateTime(timezone=True)).

## AuditLog

Database table: `audit_logs`.

Stored fields: `id` (db.Integer), `actor_id` (db.Integer), `action` (db.String(100)), `resource_type` (db.String(80)), `resource_id` (db.String(100)), `timestamp` (db.DateTime(timezone=True)), `status` (db.String(32)), `metadata_json` ('metadata').

## BoneXrayAnalysisRecord

Database table: `bone_xray_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(120)), `predicted_label` (db.String(50)), `confidence_score` (db.Float), `created_at` (db.DateTime(timezone=True)).

## BrainScanAnalysis

Database table: `brain_scan_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `predicted_class_index` (db.Integer), `model_score` (db.Float), `class_scores` (db.JSON), `created_at` (db.DateTime(timezone=True)).

## CardiacScanAnalysis

Database table: `cardiac_scan_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `pixel_counts` (db.JSON), `overlay_file_reference` (db.String(500)), `created_at` (db.DateTime(timezone=True)).

## ChestCTAnalysisRecord

Database table: `chest_ct_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `pixel_counts` (db.JSON), `image_shape` (db.JSON), `threshold` (db.Float), `overlay_file_reference` (db.String(500)), `created_at` (db.DateTime(timezone=True)).

## ChestXrayAnalysisRecord

Database table: `chest_xray_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `findings` (db.JSON), `input_shape` (db.JSON), `created_at` (db.DateTime(timezone=True)).

## CTHeadAnalysisRecord

Database table: `ct_head_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `series_classification` (db.JSON), `slice_classification` (db.JSON), `slice_count` (db.Integer), `highest_any_slice_index` (db.Integer), `input_format` (db.String(40)), `localization_file_reference` (db.String(500)), `created_at` (db.DateTime(timezone=True)).

## DentalXrayAnalysisRecord

Database table: `dental_xray_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `findings` (db.JSON), `image_shape` (db.JSON), `created_at` (db.DateTime(timezone=True)).

## EchoView47AnalysisRecord

Database table: `echoview47_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `predicted_class` (db.String(80)), `model_score` (db.Float), `class_scores` (db.JSON), `created_at` (db.DateTime(timezone=True)).

## KneeScanAnalysis

Database table: `knee_scan_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `slice_index` (db.Integer), `predicted_class_index` (db.Integer), `predicted_class` (db.String(80)), `confidence` (db.Float), `probabilities` (db.JSON), `localization_status` (db.String(40)), `roi_box` (db.JSON), `created_at` (db.DateTime(timezone=True)).

## LocalAssistantConversation

Database table: `local_assistant_conversations`.

Stored fields: `id` (db.Integer), `user_id` (db.Integer), `messages_json` (db.Text), `created_at` (db.DateTime), `updated_at` (db.DateTime).

## MedicalReport

Database table: `medical_reports`.

Stored fields: `id` (db.Integer), `patient_id` (db.Integer), `report_type` (db.String(80)), `report_date` (db.Date), `file_reference` (db.String(500)), `extracted_text` (db.Text), `processing_status` (db.String(50)), `created_at` (db.DateTime(timezone=True)), `updated_at` (db.DateTime(timezone=True)).

## ObstetricUltrasoundAnalysisRecord

Database table: `obstetric_ultrasound_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `predicted_class` (db.String(80)), `model_score` (db.Float), `class_scores` (db.JSON), `created_at` (db.DateTime(timezone=True)).

## Patient

Database table: `patients`.

Stored fields: `id` (db.Integer), `first_name` (db.String(80)), `last_name` (db.String(80)), `date_of_birth` (db.Date), `created_by_id` (db.Integer), `created_at` (db.DateTime(timezone=True)), `updated_at` (db.DateTime(timezone=True)).

## ReportParameter

Database table: `report_parameters`.

Stored fields: `id` (db.Integer), `report_id` (db.Integer), `parameter` (db.String(150)), `value` (db.String(100)), `unit` (db.String(60)), `result_date` (db.Date), `confidence` (db.Float), `source` (db.Text), `fingerprint` (db.String(64)).

## ResearchPaper

Database table: `research_papers`.

Stored fields: `id` (db.Integer), `user_id` (db.Integer), `title` (db.String(200)), `topic` (db.String(120)), `paper_date` (db.Date), `authors` (db.String(255)), `journal` (db.String(255)), `original_filename` (db.String(255)), `stored_filename` (db.String(80)), `created_at` (db.DateTime(timezone=True)).

## RevokedToken

Database table: `revoked_tokens`.

Stored fields: `jti` (db.String(36)), `expires_at` (db.DateTime(timezone=True)), `created_at` (db.DateTime(timezone=True)).

## ScanAsset

Database table: `scan_assets`.

Stored fields: `id` (db.Integer), `patient_id` (db.Integer), `modality` (db.String(30)), `body_region` (db.String(80)), `study_date` (db.Date), `original_filename` (db.String(255)), `file_reference` (db.String(500)), `processing_status` (db.String(50)), `created_at` (db.DateTime(timezone=True)).

## SpineScanAnalysis

Database table: `spine_scan_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `model_score` (db.Float), `severity_scores` (db.JSON), `frame_count` (db.Integer), `frame_source` (db.String(80)), `created_at` (db.DateTime(timezone=True)).

## User

Database table: `users`.

Stored fields: `id` (db.Integer), `first_name` (db.String(80)), `last_name` (db.String(80)), `email` (db.String(255)), `password_hash` (db.String(255)), `role` (db.String(50)), `is_active` (db.Boolean), `created_at` (db.DateTime), `updated_at` (db.DateTime).

## VascularUltrasoundAnalysisRecord

Database table: `vascular_ultrasound_analyses`.

Stored fields: `id` (db.Integer), `scan_id` (db.Integer), `model_name` (db.String(255)), `mask_pixel_count` (db.Integer), `image_shape` (db.JSON), `overlay_file_reference` (db.String(500)), `created_at` (db.DateTime(timezone=True)).
