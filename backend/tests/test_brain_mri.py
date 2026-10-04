from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.services.brain_mri import (
    BRAIN_MODEL_PATH,
    BRAIN_OUTPUT_NAMES,
    BrainImageValidationError,
    BrainModelContractError,
    BrainModelUnavailableError,
    BrainMRIPredictionService,
)


def image_bytes(size=(300, 180), color=(180, 90, 35), image_format="PNG"):
    image = Image.new("RGB", size, color=color)
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class FakeBrainModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, 4)

    def __init__(self, scores=(0.1, 0.2, 0.6, 0.1)):
        self.scores = np.asarray(scores, dtype=np.float32)[None, :]
        self.received = None

    def predict(self, batch, verbose=0):
        self.received = np.array(batch, copy=True)
        return self.scores.copy()


def test_preprocess_converts_rgb_resizes_and_preserves_raw_pixel_scale():
    content = image_bytes(color=(200, 100, 50))

    batch = BrainMRIPredictionService.preprocess_image(content)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.min() > 1.0
    assert batch.max() <= 255.0


def test_preprocess_uses_opencv_inter_linear():
    source = np.asarray(Image.open(BytesIO(image_bytes(size=(291, 177)))).convert("RGB"))
    expected = cv2.resize(source, (224, 224), interpolation=cv2.INTER_LINEAR)

    batch = BrainMRIPredictionService.preprocess_image(image_bytes(size=(291, 177)))

    assert np.array_equal(batch[0], expected.astype(np.float32))


def test_prediction_returns_four_raw_class_index_scores():
    model = FakeBrainModel()
    loads = []
    service = BrainMRIPredictionService(model_loader=lambda path: loads.append(path) or model)

    result = service.predict_image(image_bytes())

    assert result.class_index == 2
    assert result.class_score == pytest.approx(0.6)
    assert result.class_scores == {
        "Class 0": pytest.approx(0.1),
        "Class 1": pytest.approx(0.2),
        "Class 2": pytest.approx(0.6),
        "Class 3": pytest.approx(0.1),
    }
    assert model.received.shape == (1, 224, 224, 3)
    assert len(loads) == 1


@pytest.mark.parametrize("content", [b"", b"not an image", b"%PDF-1.4 fake"])
def test_invalid_image_is_rejected(content):
    with pytest.raises(BrainImageValidationError):
        BrainMRIPredictionService.preprocess_image(content)


def test_model_contract_rejects_wrong_output_shape():
    class WrongOutputModel(FakeBrainModel):
        output_shape = (None, 2)

    service = BrainMRIPredictionService(model_loader=lambda _: WrongOutputModel())
    with pytest.raises(BrainModelContractError, match=r"received \(None, 2\)"):
        service.load()


def test_actual_brain_model_loads_with_expected_shapes():
    if not Path(BRAIN_MODEL_PATH).is_file():
        pytest.skip("Brain MRI model artifact is not included in this checkout.")

    model = BrainMRIPredictionService().load()

    assert tuple(model.input_shape) == (None, 224, 224, 3)
    assert tuple(model.output_shape) == (None, 4)
    assert BRAIN_OUTPUT_NAMES == ("Class 0", "Class 1", "Class 2", "Class 3")


def test_flask_startup_loads_and_caches_brain_model():
    if not Path(BRAIN_MODEL_PATH).is_file():
        pytest.skip("Brain MRI model artifact is not included in this checkout.")
    from backend.app import app

    service = app.extensions["brain_mri_service"]
    model = service.load()

    assert model is not None
    assert service.load() is model


def test_missing_brain_model_reports_clear_error(tmp_path):
    service = BrainMRIPredictionService(model_path=tmp_path / "missing.keras")

    with pytest.raises(BrainModelUnavailableError, match="missing.keras"):
        service.load()