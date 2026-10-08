from datetime import date
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image
from werkzeug.security import generate_password_hash
from werkzeug.datastructures import MultiDict

from backend.app import create_app
from backend.extensions import db
from backend.models.patient import Patient
from backend.models.user import User


@pytest.fixture
def scan_client():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        owner = User(
            first_name="Scan",
            last_name="Owner",
            email="scan-owner@example.com",
            password_hash=generate_password_hash("ScanPass123!"),
            role="patient",
        )
        other_user = User(
            first_name="Other",
            last_name="Owner",
            email="other-scan-owner@example.com",
            password_hash=generate_password_hash("OtherPass123!"),
            role="patient",
        )
        db.session.add_all([owner, other_user])
        db.session.flush()
        owner_patient = Patient(first_name="Riley", last_name="Shaw", created_by=owner)
        other_patient = Patient(first_name="Casey", last_name="Lane", created_by=other_user)
        db.session.add_all([owner_patient, other_patient])
        db.session.commit()

        with app.test_client() as client:
            yield client, owner_patient.id, other_patient.id

        db.session.remove()
        db.drop_all()


def login(client, email="scan-owner@example.com", password="ScanPass123!"):
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    return response.get_json()["access_token"]


def png_bytes():
    return b"\x89PNG\r\n\x1a\nscan image bytes"


def valid_png_bytes(size=(64, 48), color=(120, 90, 160)):
    image = Image.new("RGB", size, color=color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeChestXrayService:
    def analyze_bytes(self, image_bytes):
        from backend.services.chest_xray import (
            ChestXrayAnalysis,
            ChestXrayFindingScore,
        )

        return ChestXrayAnalysis(
            model_name="densenet121-res224-all",
            findings=[
                ChestXrayFindingScore("Cardiomegaly", 0.91),
                ChestXrayFindingScore("Effusion", 0.78),
                ChestXrayFindingScore("Pneumothorax", 0.08),
            ],
            input_shape=(1, 1, 224, 224),
        )


class FakeBoneXrayService:
    def analyze_bytes(self, image_bytes):
        from backend.services.bone_xray import BoneXrayResult

        return BoneXrayResult(
            label="Fractured",
            confidence=87.0,
        )


class FakeSpineService:
    def preprocess_image(self, image_bytes):
        return image_bytes

    def predict_preprocessed(self, model_input):
        assert model_input
        return SimpleNamespace(
            condition_scores={
                "Spinal canal stenosis": {
                    "Normal/Mild": 0.7,
                    "Moderate": 0.2,
                    "Severe": 0.1,
                },
                "Neural foraminal narrowing": {
                    "Normal/Mild": 0.6,
                    "Moderate": 0.3,
                    "Severe": 0.1,
                },
                "Subarticular stenosis": {
                    "Normal/Mild": 0.5,
                    "Moderate": 0.3,
                    "Severe": 0.2,
                },
            },
            highest_score=0.7,
        )


class FakeDentalXrayService:
    def analyze_bytes(self, image_bytes):
        from backend.services.dental_xray import (
            DentalXrayAnalysis,
            DentalXrayFinding,
            MODEL_ID,
        )

        return DentalXrayAnalysis(
            model_name=MODEL_ID,
            findings=[
                DentalXrayFinding(
                    name="Impacted tooth",
                    score=0.901,
                    box=(264.0, 316.0, 380.0, 418.0),
                )
            ],
            image_shape=(640, 1280),
        )


class FakeCTHeadService:
    def analyze_bytes(self, file_bytes, filename):
        from backend.services.ct_head_hemorrhage import CLASS_NAMES, MODEL_ID

        scores = {name: 0.25 for name in CLASS_NAMES}
        return SimpleNamespace(
            model_name=MODEL_ID,
            series_classification=scores,
            slice_classification=[scores],
            slice_count=1,
            highest_any_slice_index=0,
            input_format="DICOM series",
            localization_png=None,
        )


class FakeChestCTService:
    def analyze_bytes(self, image_bytes):
        assert image_bytes
        return SimpleNamespace(
            model_name="best_chest_ct_model.keras",
            pixel_counts={
                "Ground-glass opacity": 125,
                "Consolidation": 42,
            },
            image_shape=(256, 256),
            threshold=0.5,
            overlay_png=valid_png_bytes(),
        )


class FakeBrainService:
    def preprocess_image(self, image_bytes):
        return image_bytes

    def predict_preprocessed(self, _):
        from backend.services.brain_mri import BrainMRIPrediction

        return BrainMRIPrediction(
            class_index=2,
            class_score=0.7,
            class_scores={
                "Glioma": 0.1,
                "Meningioma": 0.1,
                "No tumor": 0.7,
                "Pituitary tumor": 0.1,
            },
        )


class FakeObstetricUltrasoundService:
    def prepare_image(self, image_bytes):
        assert image_bytes
        return image_bytes

    def analyze_prepared_image(self, _):
        return SimpleNamespace(
            predicted_class="Fetal brain",
            model_score=0.82,
            class_scores={
                "Placenta": 0.01,
                "Fetal brain": 0.82,
                "Fetal femur": 0.01,
                "Maternal Cervix": 0.01,
                "Fetal thorax": 0.02,
                "Fetal abdomen": 0.08,
                "Other": 0.02,
                "Fetal spine": 0.02,
                "Fetal heart rate": 0.01,
            },
        )


class FakeAbdominalAortaUltrasoundService:
    def __init__(self, *, include_mask=True):
        self.include_mask = include_mask

    def prepare_image(self, image_bytes):
        assert image_bytes
        return image_bytes

    def analyze_prepared_image(self, _):
        return SimpleNamespace(
            overlay_png=valid_png_bytes(size=(64, 48)) if self.include_mask else None,
            confidence_score=0.84 if self.include_mask else None,
            mask_area_pixels=240 if self.include_mask else 0,
            image_shape=(48, 64),
        )


class FakeVascularUltrasoundService:
    def __init__(self, *, include_mask=True):
        self.include_mask = include_mask

    def prepare_image(self, image_bytes):
        assert image_bytes
        return image_bytes

    def analyze_prepared_image(self, _):
        return SimpleNamespace(
            model_name="best_carotid_ultrasound_model.keras",
            mask_pixel_count=24 if self.include_mask else 0,
            image_shape=(48, 64),
            overlay_png=valid_png_bytes(size=(64, 48)) if self.include_mask else None,
        )


class FakeEchoView47Service:
    def prepare_image(self, image_bytes):
        assert image_bytes
        return image_bytes

    def analyze_prepared_image(self, _):
        from backend.services.echoview47 import CLASS_NAMES

        scores = {class_name: 0.0 for class_name in CLASS_NAMES}
        scores["a4ch-full"] = 0.6
        scores["a4ch-rv"] = 0.3
        scores["subcostal-heart"] = 0.1
        return SimpleNamespace(
            predicted_class="a4ch-full",
            model_score=0.6,
            class_scores=scores,
        )


def test_scan_upload_list_and_original_download_are_owner_scoped(scan_client, monkeypatch):
    client, owner_patient_id, other_patient_id = scan_client
    token = login(client)
    other_token = login(client, "other-scan-owner@example.com", "OtherPass123!")
    monkeypatch.setitem(
        client.application.extensions,
        "brain_mri_service",
        FakeBrainService(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "chest_xray_service",
        FakeChestXrayService(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "bone_xray_service",
        FakeBoneXrayService(),
    )
    uploaded_image = valid_png_bytes()

    upload = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Brain",
            "study_date": "2025-06-18",
            "file": (BytesIO(uploaded_image), "brain-scan.png"),
        },
        content_type="multipart/form-data",
    )

    assert upload.status_code == 201
    scan = upload.get_json()["scan"]
    assert scan["modality"] == "MRI"
    assert scan["body_region"] == "Brain"
    assert scan["study_date"] == "2025-06-18"
    assert scan["processing_status"] == "analysis_complete"
    assert scan["brain_analysis"]["class_mapping_available"] is True
    assert scan["brain_analysis"]["predicted_class_index"] == 2
    assert scan["brain_analysis"]["predicted_class"] == "No tumor"

    listing = client.get(
        f"/api/scans?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listing.status_code == 200
    assert listing.get_json()["scans"][0]["id"] == scan["id"]

    original = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert original.status_code == 200
    assert original.data == uploaded_image

    unauthorized = client.get(scan["original_url"])
    assert unauthorized.status_code == 401

    cross_owner = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert cross_owner.status_code == 404

    hidden_patient = client.get(
        f"/api/scans?patient_id={other_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert hidden_patient.status_code == 404


def test_scan_delete_is_owner_scoped_and_removes_uploaded_file(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "OTHER",
            "body_region": "Other",
            "file": (BytesIO(valid_png_bytes()), "sample-scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    scan = response.get_json()["scan"]
    original = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert original.status_code == 200
    assert original.data

    other_token = login(
        client,
        email="other-scan-owner@example.com",
        password="OtherPass123!",
    )
    forbidden = client.delete(
        f"/api/scans/{scan['id']}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert forbidden.status_code == 404
    still_available = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert still_available.status_code == 200

    deleted = client.delete(
        f"/api/scans/{scan['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert deleted.status_code == 200
    assert deleted.get_json()["deleted"] is True
    removed = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert removed.status_code == 404

    listing = client.get(
        f"/api/scans?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert all(item["id"] != scan["id"] for item in listing.get_json()["scans"])


def test_scan_upload_rejects_invalid_modality_and_future_date(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}

    invalid_modality = client.post(
        "/api/scans/upload",
        headers=headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI-CT-MAGIC",
            "file": (BytesIO(png_bytes()), "scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert invalid_modality.status_code == 400

    future_scan = client.post(
        "/api/scans/upload",
        headers=headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "CT",
            "study_date": "2099-01-01",
            "file": (BytesIO(png_bytes()), "scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert future_scan.status_code == 400


def test_scan_subtype_options_are_validated_and_other_is_never_prediction_eligible(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}

    mismatched_region = client.post(
        "/api/scans/upload",
        headers=headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "CT",
            "body_region": "Knee",
            "file": (BytesIO(png_bytes()), "ct-scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert mismatched_region.status_code == 400

    for removed_region in ("Abdomen/pelvis", "Spine"):
        removed_ct_region = client.post(
            "/api/scans/upload",
            headers=headers,
            data={
                "patient_id": owner_patient_id,
                "modality": "CT",
                "body_region": removed_region,
                "file": (BytesIO(png_bytes()), "ct-scan.png"),
            },
            content_type="multipart/form-data",
        )
        assert removed_ct_region.status_code == 400

    other_scan = client.post(
        "/api/scans/upload",
        headers=headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Other",
            "file": (BytesIO(png_bytes()), "other-mri.png"),
        },
        content_type="multipart/form-data",
    )

    assert other_scan.status_code == 201
    payload = other_scan.get_json()["scan"]
    assert payload["processing_status"] == "unsupported_region"
    assert "not eligible for prediction" in payload["processing_message"].lower()

    removed_thyroid_region = client.post(
        "/api/scans/upload",
        headers=headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Thyroid",
            "file": (BytesIO(png_bytes()), "thyroid-ultrasound.png"),
        },
        content_type="multipart/form-data",
    )
    assert removed_thyroid_region.status_code == 400


def test_popular_scan_subtypes_are_accepted(scan_client, monkeypatch):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}
    monkeypatch.setitem(
        client.application.extensions,
        "brain_mri_service",
        FakeBrainService(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "ct_head_hemorrhage_service",
        FakeCTHeadService(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "chest_ct_segmentation_service",
        FakeChestCTService(),
    )
    supported_pairs = [
        ("MRI", "Brain"),
        ("MRI", "Knee"),
        ("MRI", "Spine"),
        ("MRI", "Cardiac"),
        ("CT", "Brain/head"),
        ("CT", "Chest"),
        ("X_RAY", "Chest"),
        ("X_RAY", "Bone/joint"),
        ("X_RAY", "Dental"),
        ("ULTRASOUND", "Abdomen"),
        ("ULTRASOUND", "Obstetric"),
        ("ULTRASOUND", "Cardiac/echocardiogram"),
        ("ULTRASOUND", "Vascular"),
        ("OTHER", "Other"),
    ]
    monkeypatch.setitem(
        client.application.extensions,
        "obstetric_ultrasound_service",
        FakeObstetricUltrasoundService(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "abdominal_aorta_ultrasound_service",
        FakeAbdominalAortaUltrasoundService(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "echoview47_service",
        FakeEchoView47Service(),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "vascular_ultrasound_service",
        FakeVascularUltrasoundService(),
    )

    for modality, body_region in supported_pairs:
        if modality == "CT" and body_region == "Brain/head":
            archive = BytesIO()
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("slice.dcm", b"test DICOM payload")
            image = archive.getvalue()
            filename = "head-ct-series.zip"
        else:
            image = (
                valid_png_bytes()
                if (modality == "MRI" and body_region in {"Brain", "Knee", "Spine"})
                or (modality == "X_RAY" and body_region in {"Chest", "Bone/joint", "Dental"})
                or (modality == "CT" and body_region == "Chest")
                or (
                    modality == "ULTRASOUND"
                    and body_region
                    in {"Abdomen", "Obstetric", "Cardiac/echocardiogram"}
                )
                else png_bytes()
            )
            filename = f"{modality.lower()}-scan.png"
        response = client.post(
            "/api/scans/upload",
            headers=headers,
            data={
                "patient_id": owner_patient_id,
                "modality": modality,
                "body_region": body_region,
                "file": (BytesIO(image), filename),
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 201, response.get_json()
        if modality == "MRI" and body_region == "Spine":
            spine_scan = response.get_json()["scan"]
            assert spine_scan["processing_status"] == "analysis_complete"
            assert spine_scan["spine_analysis"]["model_name"] == (
                "mrimperium/Lumbar-Spine-Degenerative-Classification"
            )
        elif modality == "CT" and body_region == "Chest":
            assert response.get_json()["scan"]["chest_ct_analysis"]["pixel_counts"] == {
                "Ground-glass opacity": 125,
                "Consolidation": 42,
            }
            assert len(spine_scan["spine_analysis"]["condition_scores"]) == 3
        if modality == "ULTRASOUND" and body_region == "Obstetric":
            obstetric_scan = response.get_json()["scan"]
            assert obstetric_scan["processing_status"] == "analysis_complete"
            assert (
                obstetric_scan["obstetric_ultrasound_analysis"]["predicted_class"]
                == "Fetal brain"
            )
            assert len(
                obstetric_scan["obstetric_ultrasound_analysis"]["class_scores"]
            ) == 9
            assert "does not assess fetal health" in (
                obstetric_scan["obstetric_ultrasound_analysis"]["disclaimer"].lower()
            )
        if (
            modality == "ULTRASOUND"
            and body_region == "Cardiac/echocardiogram"
        ):
            echo_scan = response.get_json()["scan"]
            analysis = echo_scan["echoview47_analysis"]
            assert echo_scan["processing_status"] == "analysis_complete"
            assert analysis["display_label"].startswith("Apical 4-chamber")
            assert "all four heart chambers" in analysis["meaning"]
            assert "uncalibrated" in analysis["disclaimer"].lower()
            assert len(analysis["top_views"]) == 3
            assert "not a diagnosis" in echo_scan["processing_message"].lower()


def test_echoview47_ultrasound_requires_jpg_or_png(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Cardiac/echocardiogram",
            "file": (BytesIO(b"%PDF-1.4\nscan"), "echo.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "JPG and PNG" in response.get_json()["error"]


def test_obstetric_ultrasound_requires_jpg_or_png(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Obstetric",
            "file": (BytesIO(b"%PDF-1.4\nscan"), "obstetric-scan.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "JPG and PNG" in response.get_json()["error"]


def test_abdominal_aorta_ultrasound_upload_persists_mask_and_protected_overlay(
    scan_client,
    monkeypatch,
):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    monkeypatch.setitem(
        client.application.extensions,
        "abdominal_aorta_ultrasound_service",
        FakeAbdominalAortaUltrasoundService(),
    )
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Abdomen",
            "file": (BytesIO(valid_png_bytes()), "aorta-view.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    scan = response.get_json()["scan"]
    analysis = scan["abdominal_aorta_ultrasound_analysis"]
    assert scan["processing_status"] == "analysis_complete"
    assert analysis["class_name"] == "Aorta"
    assert analysis["has_mask"] is True
    assert analysis["mask_area_pixels"] == 240
    assert "not aneurysm detection" in analysis["disclaimer"].lower()

    overlay = client.get(
        analysis["overlay_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert overlay.status_code == 200
    assert overlay.mimetype == "image/png"


def test_abdominal_aorta_no_mask_is_not_reported_as_absent(
    scan_client,
    monkeypatch,
):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    monkeypatch.setitem(
        client.application.extensions,
        "abdominal_aorta_ultrasound_service",
        FakeAbdominalAortaUltrasoundService(include_mask=False),
    )
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Abdomen",
            "file": (BytesIO(valid_png_bytes()), "aorta-view.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "analysis_complete"
    assert scan["abdominal_aorta_ultrasound_analysis"]["has_mask"] is False
    assert "does not establish" in scan["processing_message"]


def test_abdominal_aorta_ultrasound_requires_jpg_or_png(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Abdomen",
            "file": (BytesIO(b"%PDF-1.4\nscan"), "abdominal-ultrasound.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "JPG and PNG" in response.get_json()["error"]


def test_vascular_ultrasound_upload_persists_overlay_without_scores(
    scan_client,
    monkeypatch,
):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    monkeypatch.setitem(
        client.application.extensions,
        "vascular_ultrasound_service",
        FakeVascularUltrasoundService(),
    )
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Vascular",
            "file": (BytesIO(valid_png_bytes()), "carotid-view.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    scan = response.get_json()["scan"]
    analysis = scan["vascular_ultrasound_analysis"]
    assert scan["processing_status"] == "analysis_complete"
    assert analysis["label"] == "Predicted carotid artery region"
    assert analysis["has_mask"] is True
    assert "may be inaccurate" in analysis["disclaimer"]
    assert "does not diagnose" in analysis["disclaimer"]
    assert "score" not in str(analysis).lower()
    assert "dice" not in str(analysis).lower()
    assert "iou" not in str(analysis).lower()

    overlay = client.get(
        analysis["overlay_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert overlay.status_code == 200
    assert overlay.mimetype == "image/png"

    other_token = login(
        client,
        email="other-scan-owner@example.com",
        password="OtherPass123!",
    )
    unauthorized_overlay = client.get(
        analysis["overlay_url"],
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert unauthorized_overlay.status_code == 404


def test_vascular_ultrasound_no_mask_is_not_reported_as_absent(
    scan_client,
    monkeypatch,
):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    monkeypatch.setitem(
        client.application.extensions,
        "vascular_ultrasound_service",
        FakeVascularUltrasoundService(include_mask=False),
    )
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Vascular",
            "file": (BytesIO(valid_png_bytes()), "carotid-view.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "analysis_complete"
    assert scan["vascular_ultrasound_analysis"]["has_mask"] is False
    assert "does not establish" in scan["processing_message"]


def test_vascular_ultrasound_upload_requires_jpg_or_png(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "ULTRASOUND",
            "body_region": "Vascular",
            "file": (BytesIO(b"%PDF-1.4\nscan"), "vascular-ultrasound.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "JPG and PNG" in response.get_json()["error"]


def test_dental_xray_upload_persists_documented_candidate_and_pixel_box(
    scan_client,
    monkeypatch,
):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    auth_headers = {"Authorization": "Bearer " + token}
    monkeypatch.setitem(
        client.application.extensions,
        "dental_xray_service",
        FakeDentalXrayService(),
    )

    response = client.post(
        "/api/scans/upload",
        headers=auth_headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "X_RAY",
            "body_region": "Dental",
            "file": (BytesIO(valid_png_bytes()), "panoramic.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "analysis_complete"
    assert scan["dental_xray_analysis"]["findings"] == [
        {
            "name": "Impacted tooth",
            "score": 0.901,
            "box": [264.0, 316.0, 380.0, 418.0],
        }
    ]
    assert "not confirmed diagnoses" in scan["dental_xray_analysis"]["disclaimer"].lower()


def test_head_ct_route_serializes_six_model_outputs(scan_client, monkeypatch):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    auth_headers = {"Authorization": "Bearer " + token}
    monkeypatch.setitem(
        client.application.extensions,
        "ct_head_hemorrhage_service",
        FakeCTHeadService(),
    )
    archive = BytesIO()
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("slice.dcm", b"test DICOM payload")

    response = client.post(
        "/api/scans/upload",
        headers=auth_headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "CT",
            "body_region": "Brain/head",
            "file": (BytesIO(archive.getvalue()), "head-ct-series.zip"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    analysis = response.get_json()["scan"]["ct_head_analysis"]
    assert set(analysis["series_classification"]) == {
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "any",
    }
    assert "NOT a confirmed medical diagnosis" in analysis["disclaimer"]


def test_chest_ct_upload_persists_results_and_protects_overlay(
    scan_client,
    monkeypatch,
    tmp_path,
):
    client, owner_patient_id, _ = scan_client
    owner_headers = {"Authorization": "Bearer " + login(client)}
    monkeypatch.setitem(
        client.application.extensions,
        "chest_ct_segmentation_service",
        FakeChestCTService(),
    )

    def save_to_temp(file_storage, patient_id, **_):
        name = Path(file_storage.filename).name
        destination = tmp_path / name
        file_storage.stream.seek(0)
        destination.write_bytes(file_storage.stream.read())
        return f"uploads/patients/{patient_id}/{name}"

    monkeypatch.setattr("backend.routes.scans.save_uploaded_file", save_to_temp)
    monkeypatch.setattr(
        "backend.routes.scans.get_stored_file_location",
        lambda reference, _patient_id: (tmp_path, Path(reference).name),
    )

    response = client.post(
        "/api/scans/upload",
        headers=owner_headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "CT",
            "body_region": "Chest",
            "file": (BytesIO(valid_png_bytes()), "chest-ct-slice.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201, response.get_json()
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "analysis_complete"
    analysis = scan["chest_ct_analysis"]
    assert analysis["pixel_counts"]["Ground-glass opacity"] == 125
    assert analysis["pixel_counts"]["Consolidation"] == 42
    assert "reported_test_dice" not in analysis
    assert "experimental mask" in analysis["disclaimer"].lower()
    assert "may be inaccurate" in analysis["disclaimer"].lower()
    assert "COVID status" in analysis["disclaimer"]

    overlay = client.get(analysis["overlay_url"], headers=owner_headers)
    assert overlay.status_code == 200
    assert overlay.mimetype == "image/png"

    other_headers = {
        "Authorization": "Bearer "
        + login(
            client,
            email="other-scan-owner@example.com",
            password="OtherPass123!",
        )
    }
    unauthorized_overlay = client.get(analysis["overlay_url"], headers=other_headers)
    assert unauthorized_overlay.status_code == 404


def test_knee_upload_runs_real_localizer_and_fails_closed_on_invalid_mask(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    image_path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "sample images"
        / "kneetest.jpg"
    )
    image_bytes = image_path.read_bytes()

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Knee",
            "file": (BytesIO(image_bytes), "kneetest.jpg"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "localization_failed"
    assert scan["preprocessing_status"] == "failed"
    assert scan["localization_status"] == "failed"
    assert scan["knee_analysis"] is None
    assert "no classifier prediction was made" in scan["processing_message"].lower()

    original = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert original.status_code == 200
    assert original.data == image_bytes


def test_knee_upload_returns_and_persists_each_localized_slice(scan_client, monkeypatch):
    from backend.services.knee_mri import KneeSlicePrediction

    client, owner_patient_id, _ = scan_client
    token = login(client)

    class FakeKneeService:
        def decode_slices(self, image_bytes):
            return [image_bytes for _ in image_bytes]

        def predict_image_slices(self, image_slices):
            return [
                KneeSlicePrediction(
                    slice_index=index,
                    class_index=index,
                    class_name=("Healthy", "Partial ACL tear")[index],
                    confidence=(0.8, 0.7)[index],
                    probabilities=(
                        {"Healthy": 0.8, "Partial ACL tear": 0.1, "Complete ACL tear": 0.1},
                        {"Healthy": 0.2, "Partial ACL tear": 0.7, "Complete ACL tear": 0.1},
                    )[index],
                    localization_status="localized",
                    roi_box={"x": 8, "y": 9, "width": 20, "height": 21},
                )
                for index in range(len(image_slices))
            ]

    monkeypatch.setattr(
        "backend.routes.scans.KneeMRIPredictionService",
        FakeKneeService,
    )
    first_image = valid_png_bytes()
    second_image = valid_png_bytes()
    upload_data = MultiDict(
        [
            ("patient_id", str(owner_patient_id)),
            ("modality", "MRI"),
            ("body_region", "Knee"),
            ("file", (BytesIO(first_image), "sagittal-1.png")),
            ("file", (BytesIO(second_image), "sagittal-2.png")),
        ]
    )

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data=upload_data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scans = response.get_json()["scans"]
    assert [scan["knee_analysis"]["slice_index"] for scan in scans] == [0, 1]
    assert [scan["knee_analysis"]["class_name"] for scan in scans] == [
        "Healthy",
        "Partial ACL tear",
    ]
    assert scans[1]["knee_analysis"]["probabilities"]["Partial ACL tear"] == 0.7
    assert all(scan["processing_status"] == "analysis_complete" for scan in scans)

    listing = client.get(
        f"/api/scans?patient_id={owner_patient_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listing.status_code == 200
    assert sum(scan["knee_analysis"] is not None for scan in listing.get_json()["scans"]) == 2


def test_knee_route_rejects_corrupt_image(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Knee",
            "file": (BytesIO(b"\x89PNG\r\n\x1a\nnot an image"), "knee.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "corrupt or unreadable" in response.get_json()["error"].lower()


def test_non_knee_mri_does_not_invoke_knee_service(scan_client, monkeypatch):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    monkeypatch.setattr(
        "backend.routes.scans.KneeMRIPredictionService",
        lambda: pytest.fail("Knee service must not run for brain MRI"),
    )
    monkeypatch.setitem(
        client.application.extensions,
        "brain_mri_service",
        FakeBrainService(),
    )

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Brain",
            "file": (BytesIO(valid_png_bytes()), "brain.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert response.get_json()["scan"]["processing_status"] == "analysis_complete"


def test_brain_mri_upload_returns_named_class_scores(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    image_bytes = valid_png_bytes(size=(180, 140), color=(90, 120, 160))

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Brain",
            "file": (BytesIO(image_bytes), "brain-mri.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    analysis = scan["brain_analysis"]
    assert scan["processing_status"] == "analysis_complete"
    assert analysis["model_name"] == "mri_brain_tumor_efficientnetb0_final.keras"
    assert analysis["predicted_class_index"] in range(4)
    assert set(analysis["class_scores"]) == {
        "Glioma",
        "Meningioma",
        "No tumor",
        "Pituitary tumor",
    }
    assert analysis["class_mapping_available"] is True
    expected_classes = ("Glioma", "Meningioma", "No tumor", "Pituitary tumor")
    assert analysis["predicted_class"] == expected_classes[
        analysis["predicted_class_index"]
    ]
    assert sum(analysis["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert analysis["model_score"] == pytest.approx(
        analysis["class_scores"][analysis["predicted_class"]]
    )
    assert "glioma" in scan["processing_message"].lower()
    assert "not a medical diagnosis" in analysis["disclaimer"].lower()


def test_brain_mri_upload_endpoint_runs_actual_model(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    image_bytes = valid_png_bytes(size=(180, 140), color=(75, 115, 155))

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Brain",
            "file": (BytesIO(image_bytes), "brain-model-smoke-test.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    analysis = scan["brain_analysis"]
    assert scan["processing_status"] == "analysis_complete"
    assert analysis["predicted_class_index"] in range(4)
    assert len(analysis["class_scores"]) == 4
    assert sum(analysis["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)


def test_chest_xray_upload_runs_real_pretrained_model(scan_client):
    import tempfile

    client, owner_patient_id, _ = scan_client
    token = login(client)
    auth_headers = {"Authorization": "Bearer " + token}
    sample = Path(tempfile.gettempdir()) / "torchxrayvision_00000001_000.png"
    if not sample.is_file():
        pytest.skip("Real CXR smoke sample not present in temp folder.")
    image_bytes = sample.read_bytes()

    response = client.post(
        "/api/scans/upload",
        headers=auth_headers,
        data={
            "patient_id": owner_patient_id,
            "modality": "X_RAY",
            "body_region": "Chest",
            "file": (BytesIO(image_bytes), "chest-xray-smoke.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    analysis = scan["chest_xray_analysis"]
    assert scan["processing_status"] == "analysis_complete"
    assert analysis["model_name"] == "densenet121-res224-all"
    assert len(analysis["findings"]) == 18
    assert all(0.0 <= f["score"] <= 1.0 for f in analysis["findings"])
    assert any(f["name"] == "Cardiomegaly" for f in analysis["findings"])
    assert "not confirmed diagnoses" in scan["processing_message"].lower()
    assert "not confirmed diagnoses" in analysis["disclaimer"].lower()


def test_spine_mri_upload_runs_model_and_persists_severity_scores(
    scan_client,
    monkeypatch,
):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    monkeypatch.setitem(
        client.application.extensions,
        "spine_mri_service",
        FakeSpineService(),
    )
    image_path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "sample images"
        / "spinetest.jpg"
    )
    image_bytes = image_path.read_bytes()

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Spine",
            "file": (BytesIO(image_bytes), "spinetest.jpg"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "analysis_complete"
    assert scan["spine_analysis"]["model_name"] == (
        "mrimperium/Lumbar-Spine-Degenerative-Classification"
    )
    scores = scan["spine_analysis"]["condition_scores"]
    assert set(scores) == {
        "Spinal canal stenosis",
        "Neural foraminal narrowing",
        "Subarticular stenosis",
    }
    assert set(scores["Spinal canal stenosis"]) == {
        "Normal/Mild",
        "Moderate",
        "Severe",
    }
    assert "not a confirmed medical diagnosis" in scan["spine_analysis"]["disclaimer"].lower()
    assert "single uploaded 2d image" in scan["processing_message"].lower()

    original = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert original.status_code == 200
    assert original.data == image_bytes


def test_cardiac_mri_image_runs_and_persists_segmentation_overlay(scan_client, monkeypatch):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    overlay = png_bytes()
    pixel_counts = {"Background": 10, "RV": 20, "Myocardium": 30, "LV": 40}

    class FakeCardiacService:
        def segment(self, image_bytes):
            assert image_bytes == valid_image
            return SimpleNamespace(overlay_png=overlay, pixel_counts=pixel_counts)

    valid_image = valid_png_bytes()
    monkeypatch.setattr(
        "backend.routes.scans.CardiacSegmentationService",
        lambda: FakeCardiacService(),
        raising=False,
    )

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Cardiac",
            "file": (BytesIO(valid_image), "cardiac-mri.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "analysis_complete"
    assert scan["analysis"]["pixel_counts"] == pixel_counts
    assert scan["analysis"]["class_names"] == ["Background", "RV", "Myocardium", "LV"]

    overlay_response = client.get(
        scan["analysis"]["overlay_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert overlay_response.status_code == 200
    assert overlay_response.data == overlay
    assert overlay_response.mimetype == "image/png"


def test_cardiac_mri_accepts_only_png_or_jpeg(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)

    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Cardiac",
            "file": (BytesIO(b"%PDF-1.4\ncardiac scan"), "cardiac-mri.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == (
        "Please upload a cardiac MRI image in PNG or JPG format."
    )


def test_cardiac_scan_is_stored_without_prediction_when_model_runtime_is_unavailable(
    scan_client,
    monkeypatch,
):
    from backend.services.cardiac_segmentation import CardiacModelUnavailableError

    client, owner_patient_id, _ = scan_client
    token = login(client)

    class UnavailableCardiacService:
        def segment(self, image_bytes):
            raise CardiacModelUnavailableError("TensorFlow is not installed")

    monkeypatch.setattr(
        "backend.routes.scans.CardiacSegmentationService",
        lambda: UnavailableCardiacService(),
    )
    image_bytes = valid_png_bytes()
    response = client.post(
        "/api/scans/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "body_region": "Cardiac",
            "file": (BytesIO(image_bytes), "cardiac-mri.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    scan = response.get_json()["scan"]
    assert scan["processing_status"] == "model_unavailable"
    assert scan["analysis"] is None
    assert "unavailable" in scan["processing_message"].lower()

    original = client.get(
        scan["original_url"],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert original.status_code == 200
    assert original.data == image_bytes


def test_scan_upload_requires_authentication(scan_client):
    client, owner_patient_id, _ = scan_client

    response = client.post(
        "/api/scans/upload",
        data={
            "patient_id": owner_patient_id,
            "modality": "MRI",
            "file": (BytesIO(png_bytes()), "scan.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 401