from datetime import date
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

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


class FakeBrainService:
    def preprocess_image(self, image_bytes):
        return image_bytes

    def predict_preprocessed(self, _):
        from backend.services.brain_mri import BrainMRIPrediction

        return BrainMRIPrediction(
            class_index=2,
            class_score=0.7,
            class_scores={
                "Class 0": 0.1,
                "Class 1": 0.1,
                "Class 2": 0.7,
                "Class 3": 0.1,
            },
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
    assert scan["brain_analysis"]["predicted_class_index"] == 2

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


def test_popular_scan_subtypes_are_accepted(scan_client, monkeypatch):
    client, owner_patient_id, _ = scan_client
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}
    monkeypatch.setitem(
        client.application.extensions,
        "brain_mri_service",
        FakeBrainService(),
    )
    supported_pairs = [
        ("MRI", "Brain"),
        ("MRI", "Knee"),
        ("MRI", "Spine"),
        ("MRI", "Cardiac"),
        ("CT", "Brain/head"),
        ("CT", "Chest"),
        ("CT", "Abdomen/pelvis"),
        ("CT", "Spine"),
        ("CT", "Cardiac"),
        ("X_RAY", "Chest"),
        ("X_RAY", "Bone/joint"),
        ("X_RAY", "Spine"),
        ("X_RAY", "Dental"),
        ("ULTRASOUND", "Abdomen"),
        ("ULTRASOUND", "Obstetric"),
        ("ULTRASOUND", "Cardiac/echocardiogram"),
        ("ULTRASOUND", "Vascular"),
        ("ULTRASOUND", "Thyroid"),
        ("OTHER", "Other"),
    ]

    for modality, body_region in supported_pairs:
        image = (
            valid_png_bytes()
            if modality == "MRI" and body_region in {"Brain", "Knee", "Spine"}
            else png_bytes()
        )
        response = client.post(
            "/api/scans/upload",
            headers=headers,
            data={
                "patient_id": owner_patient_id,
                "modality": modality,
                "body_region": body_region,
                "file": (BytesIO(image), f"{modality.lower()}-scan.png"),
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 201, response.get_json()


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


def test_brain_mri_upload_runs_real_model_and_returns_unlabeled_scores(scan_client):
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
        "Class 0",
        "Class 1",
        "Class 2",
        "Class 3",
    }
    assert sum(analysis["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert analysis["model_score"] == pytest.approx(
        analysis["class_scores"][f"Class {analysis['predicted_class_index']}"]
    )
    assert "label meanings are unspecified" in scan["processing_message"].lower()
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


def test_spine_mri_upload_runs_real_model_and_returns_raw_score(scan_client):
    client, owner_patient_id, _ = scan_client
    token = login(client)
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
    assert scan["spine_analysis"]["model_name"] == "best_mrnet_fast.keras"
    assert 0.0 <= scan["spine_analysis"]["model_score"] <= 1.0
    assert scan["spine_analysis"]["frame_count"] == 12
    assert scan["spine_analysis"]["frame_source"] == "single_uploaded_image_repeated"
    assert "meaning is not specified" in scan["processing_message"].lower()
    assert "diagnosis" in scan["spine_analysis"]["disclaimer"].lower()

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