from io import BytesIO

import numpy as np
import pytest
import cv2
from PIL import Image

from backend.services.vascular_ultrasound import (
    INPUT_SIZE,
    VascularUltrasoundAnalysisService,
    VascularUltrasoundImageError,
    VascularUltrasoundModelContractError,
)


class FakeSegmentationModel:
    input_shape = (None, INPUT_SIZE, INPUT_SIZE, 1)
    output_shape = (None, INPUT_SIZE, INPUT_SIZE, 1)

    def __init__(self, *, has_mask=True):
        self.has_mask = has_mask

    def predict(self, model_input, verbose=0):
        assert verbose == 0
        assert model_input.shape == (1, INPUT_SIZE, INPUT_SIZE, 1)
        assert model_input.dtype == np.float32
        assert np.isfinite(model_input).all()
        assert model_input.min() >= 0
        assert model_input.max() <= 1
        output = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 1), dtype=np.float32)
        if self.has_mask:
            output[0, 20:30, 40:50, 0] = 0.8
        return output


def png_bytes(size=(128, 96), color=128):
    output = BytesIO()
    Image.new("L", size, color=color).save(output, format="PNG")
    return output.getvalue()


def test_vascular_ultrasound_service_returns_mask_overlay_at_source_size():
    service = VascularUltrasoundAnalysisService(
        model_loader=lambda _: FakeSegmentationModel(),
    )

    result = service.analyze_bytes(png_bytes())

    assert result.mask_pixel_count == 100
    assert result.image_shape == (96, 128)
    assert result.overlay_png
    expected_mask = cv2.resize(
        np.pad(
            np.ones((10, 10), dtype=np.uint8),
            ((20, INPUT_SIZE - 30), (40, INPUT_SIZE - 50)),
        ),
        (128, 96),
        interpolation=cv2.INTER_NEAREST,
    )
    points = cv2.findNonZero(expected_mask)
    assert points is not None
    x, y, _, _ = cv2.boundingRect(points)
    with Image.open(BytesIO(result.overlay_png)) as overlay:
        assert overlay.size == (128, 96)
        assert overlay.mode == "RGB"
        assert overlay.getpixel((x, y)) == (255, 255, 0)


def test_vascular_ultrasound_service_returns_no_overlay_for_empty_mask():
    service = VascularUltrasoundAnalysisService(
        model_loader=lambda _: FakeSegmentationModel(has_mask=False),
    )

    result = service.analyze_bytes(png_bytes())

    assert result.mask_pixel_count == 0
    assert result.overlay_png is None


def test_vascular_ultrasound_service_rejects_non_image_input():
    service = VascularUltrasoundAnalysisService(
        model_loader=lambda _: FakeSegmentationModel(),
    )

    with pytest.raises(VascularUltrasoundImageError, match="corrupt or unreadable"):
        service.analyze_bytes(b"not an image")


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [
        ((None, 128, 128, 1), (None, INPUT_SIZE, INPUT_SIZE, 1)),
        (
            (None, INPUT_SIZE, INPUT_SIZE, 1),
            (None, INPUT_SIZE, INPUT_SIZE, 2),
        ),
    ],
)
def test_vascular_ultrasound_service_rejects_incompatible_model_shapes(
    input_shape,
    output_shape,
):
    model = FakeSegmentationModel()
    model.input_shape = input_shape
    model.output_shape = output_shape
    service = VascularUltrasoundAnalysisService(model_loader=lambda _: model)

    with pytest.raises(VascularUltrasoundModelContractError):
        service.load()


def test_vascular_ultrasound_service_rejects_invalid_mask_values():
    model = FakeSegmentationModel()

    def invalid_prediction(model_input, verbose=0):
        assert model_input.shape == (1, INPUT_SIZE, INPUT_SIZE, 1)
        assert verbose == 0
        return np.full(
            (1, INPUT_SIZE, INPUT_SIZE, 1),
            np.nan,
            dtype=np.float32,
        )

    model.predict = invalid_prediction
    service = VascularUltrasoundAnalysisService(model_loader=lambda _: model)

    with pytest.raises(VascularUltrasoundModelContractError, match="invalid mask"):
        service.analyze_bytes(png_bytes())
