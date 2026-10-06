from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from backend.services.obstetric_ultrasound import (
    CLASS_NAMES,
    ObstetricUltrasoundAnalysisService,
    ObstetricUltrasoundImageError,
)


def png_bytes():
    image = Image.new("RGB", (48, 32), color=(80, 120, 160))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeProcessor:
    def __call__(self, *, images, return_tensors):
        assert images.mode == "RGB"
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


class FakeModel:
    config = SimpleNamespace(
        id2label={index: label for index, label in enumerate(CLASS_NAMES)}
    )

    def __call__(self, pixel_values):
        assert tuple(pixel_values.shape) == (1, 3, 224, 224)
        return SimpleNamespace(
            logits=torch.tensor(
                [[0.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
            )
        )


def fake_model_loader(_):
    return torch, FakeProcessor(), FakeModel()


def test_obstetric_classifier_preprocesses_jpg_png_and_returns_labeled_scores():
    service = ObstetricUltrasoundAnalysisService(model_loader=fake_model_loader)
    prepared = service.prepare_image(png_bytes())
    result = service.analyze_prepared_image(prepared)

    assert result.predicted_class == "Fetal brain"
    assert result.model_score == pytest.approx(max(result.class_scores.values()))
    assert tuple(result.class_scores) == CLASS_NAMES
    assert result.input_shape == (1, 3, 224, 224)
    assert np.isclose(sum(result.class_scores.values()), 1.0)


@pytest.mark.parametrize(
    "image_bytes",
    [b"", b"not an image"],
)
def test_obstetric_classifier_rejects_empty_or_corrupt_image(image_bytes):
    with pytest.raises(ObstetricUltrasoundImageError):
        ObstetricUltrasoundAnalysisService.prepare_image(image_bytes)


def test_obstetric_classifier_rejects_non_jpeg_png_image():
    image = Image.new("RGB", (16, 16))
    output = BytesIO()
    image.save(output, format="BMP")

    with pytest.raises(ObstetricUltrasoundImageError, match="JPG and PNG"):
        ObstetricUltrasoundAnalysisService.prepare_image(output.getvalue())
