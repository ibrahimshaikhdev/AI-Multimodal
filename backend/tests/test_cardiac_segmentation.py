from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.services.cardiac_segmentation import (
    CardiacModelContractError,
    CardiacSegmentationService,
)


def png_image(size=(512, 128)):
    image = Image.new("RGB", size, color=(255, 128, 64))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeCardiacModel:
    input_shape = (None, 256, 256, 1)
    output_shape = (None, 256, 256, 4)

    def __init__(self, class_index=2, regions=None):
        self.class_index = class_index
        self.regions = regions
        self.received = None

    def predict(self, batch, verbose=0):
        self.received = batch
        output = np.zeros((1, 256, 256, 4), dtype=np.float32)
        if self.regions is None:
            output[:, :, :, self.class_index] = 1.0
        else:
            output[:, :, :, 0] = 1.0
            for class_index, (top, left, bottom, right) in self.regions.items():
                output[:, top:bottom, left:right, class_index] = 2.0
        return output


def test_cardiac_overlay_draws_a_class_colored_box_around_each_foreground_mask():
    model = FakeCardiacModel(
        regions={
            1: (20, 10, 50, 45),
            2: (70, 60, 100, 95),
            3: (120, 110, 150, 145),
        }
    )
    service = CardiacSegmentationService(model_loader=lambda _: model)

    result = service.segment(png_image())

    with Image.open(BytesIO(result.overlay_png)) as overlay:
        assert overlay.getpixel((10, 20)) == (255, 77, 79)
        assert overlay.getpixel((60, 70)) == (255, 197, 61)
        assert overlay.getpixel((110, 120)) == (22, 119, 255)
        assert overlay.getpixel((5, 5)) != (0, 0, 0)


def test_cardiac_segmentation_preprocesses_argmaxes_and_builds_overlay():
    model = FakeCardiacModel(class_index=2)
    service = CardiacSegmentationService(model_loader=lambda _: model)

    result = service.segment(png_image())

    assert model.received.shape == (1, 256, 256, 1)
    assert model.received.dtype == np.float32
    assert 0 <= model.received.min() <= model.received.max() <= 1
    assert result.pixel_counts == {
        "Background": 0,
        "RV": 0,
        "Myocardium": 256 * 256,
        "LV": 0,
    }
    assert result.overlay_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.mask_png.startswith(b"\x89PNG\r\n\x1a\n")

    overlay = Image.open(BytesIO(result.overlay_png))
    assert overlay.size == (256, 256)
    assert overlay.mode == "RGB"


def test_cardiac_segmentation_rejects_non_image_and_non_four_class_outputs():
    service = CardiacSegmentationService(model_loader=lambda _: FakeCardiacModel())

    with pytest.raises(ValueError, match="PNG or JPG"):
        service.segment(b"not an image")

    class InvalidOutputModel(FakeCardiacModel):
        output_shape = (None, 256, 256, 3)

    invalid_service = CardiacSegmentationService(
        model_loader=lambda _: InvalidOutputModel()
    )
    with pytest.raises(CardiacModelContractError, match="four classes"):
        invalid_service.segment(png_image())