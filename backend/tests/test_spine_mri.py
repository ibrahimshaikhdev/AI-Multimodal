from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.services.spine_mri import (
    SPINE_MODEL_PATH,
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


class FakeSpineModel:
    input_shape = (None, 12, 160, 160, 3)
    output_shape = (None, 1)

    def __init__(self, score=0.63):
        self.score = score
        self.received = None

    def predict(self, batch, verbose=0):
        self.received = np.array(batch, copy=True)
        return np.array([[self.score]], dtype=np.float32)


def test_preprocess_resizes_rgb_and_repeats_to_twelve_frames_without_external_scaling():
    content = image_bytes(color=(200, 100, 50))

    batch = SpineMRIPredictionService.preprocess_image(content)

    assert batch.shape == (1, 12, 160, 160, 3)
    assert batch.dtype == np.float32
    assert batch.min() > 1.0
    assert np.array_equal(batch[0, 0], batch[0, 11])


def test_preprocess_uses_opencv_inter_linear():
    source = np.asarray(Image.open(BytesIO(image_bytes(size=(237, 191)))).convert("RGB"))

    batch = SpineMRIPredictionService.preprocess_image(image_bytes(size=(237, 191)))
    expected = cv2.resize(source, (160, 160), interpolation=cv2.INTER_LINEAR)

    assert np.array_equal(batch[0, 0], expected.astype(np.float32))


def test_prediction_returns_raw_sigmoid_score_and_loads_model_once():
    model = FakeSpineModel(score=0.63)
    loads = []
    service = SpineMRIPredictionService(model_loader=lambda path: loads.append(path) or model)

    first = service.predict_image(image_bytes())
    second = service.predict_image(image_bytes(color=(20, 40, 80)))

    assert first.score == pytest.approx(0.63)
    assert second.score == pytest.approx(0.63)
    assert model.received.shape == (1, 12, 160, 160, 3)
    assert len(loads) == 1


@pytest.mark.parametrize("content", [b"", b"not an image", b"%PDF-1.4 fake"])
def test_invalid_image_is_rejected(content):
    with pytest.raises(SpineImageValidationError):
        SpineMRIPredictionService.preprocess_image(content)


def test_bad_model_output_shape_is_rejected():
    class WrongOutputModel(FakeSpineModel):
        def predict(self, batch, verbose=0):
            return np.zeros((1, 2), dtype=np.float32)

    service = SpineMRIPredictionService(model_loader=lambda _: WrongOutputModel())
    with pytest.raises(SpineModelContractError, match=r"received \(1, 2\)"):
        service.predict_image(image_bytes())


def test_existing_spine_model_loads_with_expected_contract():
    if not Path(SPINE_MODEL_PATH).is_file():
        pytest.skip("MRNet spine model artifact is not included in this checkout.")

    model = SpineMRIPredictionService().load()

    assert tuple(model.input_shape) == (None, 12, 160, 160, 3)
    assert tuple(model.output_shape) == (None, 1)


def test_flask_app_loads_spine_model_at_startup():
    if not Path(SPINE_MODEL_PATH).is_file():
        pytest.skip("MRNet spine model artifact is not included in this checkout.")
    from backend.app import app

    service = app.extensions["spine_mri_service"]
    loaded_model = service.load()

    assert loaded_model is not None
    assert service.load() is loaded_model


def test_missing_model_file_reports_clear_error(tmp_path):
    service = SpineMRIPredictionService(model_path=tmp_path / "missing.keras")

    with pytest.raises(SpineModelUnavailableError, match="missing.keras"):
        service.load()