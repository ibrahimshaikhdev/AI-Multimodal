from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.services.chest_ct_segmentation import (
    ChestCTImageError,
    ChestCTModelContractError,
    ChestCTSegmentationService,
    FINDING_NAMES,
    INPUT_SIZE,
)


class FakeSegmentationModel:
    input_shape = (None, INPUT_SIZE, INPUT_SIZE, 1)
    output_shape = (None, INPUT_SIZE, INPUT_SIZE, len(FINDING_NAMES))

    def predict(self, model_input, verbose=0):
        assert verbose == 0
        assert model_input.shape == (1, *self.input_shape[1:])
        output = np.zeros(
            (1, INPUT_SIZE, INPUT_SIZE, len(FINDING_NAMES)),
            dtype=np.float32,
        )
        output[0, 10:20, 30:40, 0] = 0.8
        output[0, 40:50, 60:70, 1] = 0.9
        return output


def png_bytes(size=(128, 96), color=128):
    output = BytesIO()
    Image.new("L", size, color=color).save(output, format="PNG")
    return output.getvalue()


def test_chest_ct_service_returns_two_thresholded_masks_and_overlay():
    model = FakeSegmentationModel()
    service = ChestCTSegmentationService(model_loader=lambda _: model)

    result = service.analyze_bytes(png_bytes())

    assert result.pixel_counts == {
        "Ground-glass opacity": 100,
        "Consolidation": 100,
    }
    assert result.image_shape == (96, 128)
    assert result.threshold == 0.5
    with Image.open(BytesIO(result.overlay_png)) as overlay:
        assert overlay.size == (INPUT_SIZE, INPUT_SIZE)
        assert overlay.mode == "RGB"
        assert overlay.getpixel((30, 10)) == (36, 165, 255)
        assert overlay.getpixel((60, 40)) == (255, 145, 38)
        assert overlay.getpixel((100, 100)) == (128, 128, 128)


def test_chest_ct_service_rejects_non_image_input():
    service = ChestCTSegmentationService(model_loader=lambda _: FakeSegmentationModel())

    with pytest.raises(ChestCTImageError, match="Could not decode"):
        service.analyze_bytes(b"not an image")


def test_chest_ct_service_rejects_incompatible_model_shape():
    model = FakeSegmentationModel()
    model.input_shape = (None, 512, 512, 1)
    service = ChestCTSegmentationService(model_loader=lambda _: model)

    with pytest.raises(ChestCTModelContractError, match="256x256"):
        service.load()
