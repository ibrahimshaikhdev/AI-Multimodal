from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import cv2
from PIL import Image

from backend.services.abdominal_aorta_ultrasound import (
    AbdominalAortaUltrasoundAnalysisService,
    AbdominalAortaUltrasoundImageError,
)


def png_bytes(size=(64, 48), color=(80, 90, 100)):
    output = BytesIO()
    Image.new("RGB", size, color=color).save(output, format="PNG")
    return output.getvalue()


class FakeSegmentationModel:
    names = {0: "Aorta"}
    task = "segment"

    def __init__(self, *, include_mask=True):
        self.include_mask = include_mask

    def predict(self, *, source, device, verbose, retina_masks):
        assert source.shape == (48, 64, 3)
        assert device == "cpu"
        assert verbose is False
        assert retina_masks is True
        if not self.include_mask:
            return [SimpleNamespace(masks=None, boxes=None)]

        mask = np.zeros((48, 64), dtype=np.float32)
        mask[10:30, 20:40] = 1.0
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=torch.tensor(mask[None, ...])),
                boxes=FakeBoxes(),
            )
        ]


class FakeBoxes:
    cls = torch.tensor([0])
    conf = torch.tensor([0.83])

    def __len__(self):
        return 1


def test_aorta_segmentation_returns_mask_overlay_and_original_dimensions():
    service = AbdominalAortaUltrasoundAnalysisService(
        model_loader=lambda _: FakeSegmentationModel()
    )
    result = service.analyze_bytes(png_bytes())

    assert result.overlay_png
    assert result.confidence_score == pytest.approx(0.83)
    assert result.mask_area_pixels == 400
    assert result.image_shape == (48, 64)
    with Image.open(BytesIO(result.overlay_png)) as overlay:
        assert overlay.size == (64, 48)
        assert overlay.getpixel((25, 15)) != (80, 90, 100)
        assert overlay.getpixel((20, 10)) == (255, 255, 0)
        assert overlay.getpixel((5, 5)) == (80, 90, 100)


def test_aorta_segmentation_no_mask_is_not_reported_as_absence():
    service = AbdominalAortaUltrasoundAnalysisService(
        model_loader=lambda _: FakeSegmentationModel(include_mask=False)
    )
    result = service.analyze_bytes(png_bytes())

    assert result.overlay_png is None
    assert result.confidence_score is None
    assert result.mask_area_pixels == 0


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_aorta_segmentation_rejects_empty_or_corrupt_image(image_bytes):
    with pytest.raises(AbdominalAortaUltrasoundImageError):
        AbdominalAortaUltrasoundAnalysisService.prepare_image(image_bytes)


def test_aorta_segmentation_rejects_non_jpeg_png_image():
    output = BytesIO()
    Image.new("RGB", (16, 16)).save(output, format="BMP")

    with pytest.raises(AbdominalAortaUltrasoundImageError, match="JPG and PNG"):
        AbdominalAortaUltrasoundAnalysisService.prepare_image(output.getvalue())
