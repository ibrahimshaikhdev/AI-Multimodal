from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from backend.services.spine_mri import (
    SPINE_CONDITIONS,
    SPINE_IMAGE_SIZE,
    SPINE_MODEL_ID,
    SPINE_MODEL_PATH,
    SPINE_SEVERITIES,
    SpineImageValidationError,
    SpineModelContractError,
    SpineModelUnavailableError,
    SpineMRIPredictionService,
)


def image_bytes(size=(240, 180), color=(120, 80, 40), image_format="PNG"):
    image = Image.new("RGB", size, color=color)
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class FakeSpineModel(torch.nn.Module):
    def __init__(self, logits=None):
        super().__init__()
        self.logits = torch.tensor(
            logits or [[-2.0, 0.0, 2.0, -1.0, 1.0, 3.0, -3.0, 0.5, 1.5]],
            dtype=torch.float32,
        )
        self.received = None

    def forward(self, batch):
        self.received = batch.detach().clone()
        return self.logits


def test_preprocess_resizes_rgb_and_applies_imagenet_normalization():
    content = image_bytes(size=(237, 191), color=(200, 100, 50))
    source = np.asarray(Image.open(BytesIO(content)).convert("RGB"))
    resized = cv2.resize(source, SPINE_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    expected = (resized.astype(np.float32) / 255.0 - np.array(
        [0.485, 0.456, 0.406], dtype=np.float32
    )) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

    batch = SpineMRIPredictionService.preprocess_image(content)

    assert tuple(batch.shape) == (1, 3, 224, 224)
    assert batch.dtype == torch.float32
    assert np.allclose(batch[0].numpy(), np.transpose(expected, (2, 0, 1)))


def test_prediction_returns_nine_named_sigmoid_scores_and_loads_once():
    model = FakeSpineModel()
    loads = []
    service = SpineMRIPredictionService(
        model_loader=lambda path: loads.append(path) or model
    )

    first = service.predict_image(image_bytes())
    second = service.predict_image(image_bytes(color=(20, 40, 80)))

    assert tuple(first.condition_scores) == SPINE_CONDITIONS
    assert all(
        tuple(severity_scores) == SPINE_SEVERITIES
        for severity_scores in first.condition_scores.values()
    )
    assert first.condition_scores["Spinal canal stenosis"]["Normal/Mild"] == pytest.approx(
        torch.sigmoid(torch.tensor(-2.0)).item()
    )
    assert first.highest_score == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())
    assert second.condition_scores == first.condition_scores
    assert tuple(model.received.shape) == (1, 3, 224, 224)
    assert len(loads) == 1


@pytest.mark.parametrize("content", [b"", b"not an image", b"%PDF-1.4 fake"])
def test_invalid_image_is_rejected(content):
    with pytest.raises(SpineImageValidationError):
        SpineMRIPredictionService.preprocess_image(content)


def test_model_rejects_wrong_input_and_output_shapes():
    service = SpineMRIPredictionService(
        model_loader=lambda _: FakeSpineModel(logits=[[0.0, 1.0]])
    )
    with pytest.raises(SpineModelContractError, match="Expected nine"):
        service.predict_image(image_bytes())

    with pytest.raises(SpineModelContractError, match="Expected one Spine MRI tensor"):
        service.predict_preprocessed(torch.zeros((1, 3, 160, 160)))


def test_hugging_face_checkpoint_loads_and_runs_on_cpu():
    if not Path(SPINE_MODEL_PATH).is_file():
        pytest.skip("The local Hugging Face Spine MRI checkpoint is not present.")

    service = SpineMRIPredictionService()
    sample_image = Path(SPINE_MODEL_PATH).parent / "sample images" / "spinetest.jpg"
    prediction = service.predict_image(
        sample_image.read_bytes() if sample_image.is_file() else image_bytes()
    )

    assert service.load().training is False
    assert SPINE_MODEL_ID == "mrimperium/Lumbar-Spine-Degenerative-Classification"
    assert tuple(prediction.condition_scores) == SPINE_CONDITIONS
    assert all(
        0.0 <= score <= 1.0
        for severity_scores in prediction.condition_scores.values()
        for score in severity_scores.values()
    )


def test_flask_app_registers_spine_model_service():
    from backend.app import app

    assert isinstance(
        app.extensions["spine_mri_service"], SpineMRIPredictionService
    )


def test_missing_model_file_reports_clear_error(tmp_path):
    service = SpineMRIPredictionService(model_path=tmp_path / "missing.pth")

    with pytest.raises(SpineModelUnavailableError, match="missing.pth"):
        service.load()
