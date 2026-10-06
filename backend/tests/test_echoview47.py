from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.services.echoview47 import (
    CLASS_NAMES,
    EchoView47AnalysisService,
    EchoView47ImageError,
    EchoView47ModelContractError,
    EchoView47ModelUnavailableError,
    describe_view_class,
)


def png_bytes():
    image = Image.new("RGB", (48, 32), color=(255, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, len(CLASS_NAMES))

    def __init__(self, scores=None):
        self.scores = (
            scores
            if scores is not None
            else np.full((1, len(CLASS_NAMES)), 1 / len(CLASS_NAMES))
        )
        self.last_input = None

    def predict(self, pixels, verbose):
        assert verbose == 0
        self.last_input = pixels
        return self.scores


def test_echoview47_prepares_rgb_and_returns_plain_language_view():
    scores = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
    scores[0, 7] = 0.6
    scores[0, 12] = 0.3
    scores[0, 44] = 0.1
    model = FakeModel(scores)
    service = EchoView47AnalysisService(model_loader=lambda _: model)

    result = service.analyze_bytes(png_bytes())

    assert result.predicted_class == "a4ch-full"
    assert result.display_label == "Apical 4-chamber view"
    assert "tip (apex) of the heart" in result.meaning
    assert "all four heart chambers" in result.meaning
    assert "not a heart condition" in result.meaning
    assert result.model_score == pytest.approx(0.6)
    assert tuple(result.class_scores) == CLASS_NAMES
    assert result.input_shape == (1, 224, 224, 3)
    assert model.last_input.dtype == np.float32
    assert model.last_input.shape == (1, 224, 224, 3)
    assert tuple(model.last_input[0, 0, 0]) == (255.0, 0.0, 0.0)
    assert np.isclose(sum(result.class_scores.values()), 1.0)


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_echoview47_rejects_empty_or_corrupt_image(image_bytes):
    with pytest.raises(EchoView47ImageError):
        EchoView47AnalysisService.prepare_image(image_bytes)


def test_echoview47_rejects_non_jpeg_png_image():
    image = Image.new("RGB", (16, 16))
    output = BytesIO()
    image.save(output, format="BMP")

    with pytest.raises(EchoView47ImageError, match="JPG and PNG"):
        EchoView47AnalysisService.prepare_image(output.getvalue())


def test_echoview47_rejects_model_with_wrong_output_shape():
    model = FakeModel()
    model.output_shape = (None, 46)
    service = EchoView47AnalysisService(model_loader=lambda _: model)

    with pytest.raises(EchoView47ModelContractError, match="47 view classes"):
        service.load()


def test_echoview47_rejects_invalid_softmax_output():
    scores = np.full((1, len(CLASS_NAMES)), 0.5, dtype=np.float32)
    model = FakeModel(scores)
    service = EchoView47AnalysisService(model_loader=lambda _: model)

    with pytest.raises(EchoView47ModelContractError, match="softmax scores"):
        service.analyze_bytes(png_bytes())


def test_echoview47_requires_the_expected_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "wrong.keras"
    checkpoint.write_bytes(b"not the EchoView47 model")

    with pytest.raises(EchoView47ModelUnavailableError, match="hash"):
        from backend.services.echoview47 import _load_model

        _load_model(checkpoint)


def test_all_echoview47_classes_have_plain_language_explanations():
    assert len(CLASS_NAMES) == 47
    assert len(set(CLASS_NAMES)) == 47
    for class_name in CLASS_NAMES:
        label, meaning = describe_view_class(class_name)
        assert label
        assert meaning
        assert "view" in meaning.lower()


def test_echoview47_rejects_unknown_explanation_label():
    with pytest.raises(ValueError, match="Unsupported EchoView47 class"):
        describe_view_class("unrecognized-view")
