from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import cv2
from PIL import Image

from backend.services.knee_mri import (
    ACLROIBox,
    ACLROINotFoundError,
    KerasACLROILocalizer,
    KNEE_CLASS_NAMES,
    KneeImageValidationError,
    KneeModelUnavailableError,
    KneeMRIPredictionService,
    LOCALIZER_MODEL_PATH,
    MODEL_PATH,
    _load_keras_model,
)


def image_bytes(size=(120, 90), color=(120, 60, 30), format="JPEG"):
    image = Image.new("RGB", size, color=color)
    output = BytesIO()
    image.save(output, format=format)
    return output.getvalue()


class FakeModel:
    input_shape = (None, 75, 75, 1)
    output_shape = (None, 3)

    def __init__(self):
        self.received = None

    def predict(self, batch, verbose=0):
        self.received = np.array(batch, copy=True)
        return np.tile(np.array([[0.1, 0.7, 0.2]], dtype=np.float32), (len(batch), 1))


class FakeMaskModel:
    input_shape = (None, 224, 224, 1)
    output_shape = (None, 224, 224, 1)

    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=np.float32).reshape((1, 224, 224, 1))
        self.received = None

    def predict(self, batch, verbose=0):
        self.received = np.array(batch, copy=True)
        return self.mask.copy()


class FixedLocalizer:
    def __init__(self, box=ACLROIBox(15, 10, 60, 60)):
        self.box = box
        self.calls = 0

    def locate(self, grayscale_slice):
        self.calls += 1
        return self.box


def test_localizer_roi_is_resized_and_formatted_without_normalization():
    model = FakeModel()
    service = KneeMRIPredictionService(
        roi_localizer=FixedLocalizer(),
        model_loader=lambda _: model,
    )

    result = service.predict_slices([image_bytes()])[0]

    assert model.received.shape == (1, 75, 75, 1)
    assert model.received.dtype == np.float32
    assert model.received.min() > 1.0
    assert result.class_index == 1
    assert result.class_name == "Partial ACL tear"
    assert result.confidence == pytest.approx(0.7)
    assert result.probabilities == {
        "Healthy": pytest.approx(0.1),
        "Partial ACL tear": pytest.approx(0.7),
        "Complete ACL tear": pytest.approx(0.2),
    }
    assert result.localization_status == "localized"
    assert result.roi_box == {"x": 15, "y": 10, "width": 60, "height": 60}


def test_roi_resize_matches_training_opencv_inter_linear_without_scaling():
    roi = np.arange(43 * 61, dtype=np.uint8).reshape((43, 61))

    prepared = KneeMRIPredictionService.preprocess_roi(roi)
    expected = cv2.resize(roi, (75, 75), interpolation=cv2.INTER_LINEAR)

    assert prepared.shape == (75, 75, 1)
    assert prepared.dtype == np.float32
    assert np.array_equal(prepared[..., 0], expected.astype(np.float32))
    assert prepared.max() > 1.0


def test_multiple_slices_return_individual_predictions_without_aggregation():
    model = FakeModel()
    localizer = FixedLocalizer()
    service = KneeMRIPredictionService(
        roi_localizer=localizer,
        model_loader=lambda _: model,
    )

    results = service.predict_slices(
        [image_bytes(color=(40, 50, 60)), image_bytes(color=(80, 90, 100))]
    )

    assert model.received.shape == (2, 75, 75, 1)
    assert localizer.calls == 2
    assert [result.slice_index for result in results] == [0, 1]
    assert [result.class_name for result in results] == [
        "Partial ACL tear",
        "Partial ACL tear",
    ]


def test_localization_failure_does_not_load_or_call_classifier():
    classifier_loads = []
    service = KneeMRIPredictionService(
        roi_localizer=FixedLocalizer(box=None),
        model_loader=lambda path: classifier_loads.append(path),
    )

    with pytest.raises(ACLROINotFoundError, match="slice 1"):
        service.predict_slices([image_bytes()])

    assert classifier_loads == []


def test_localizer_missing_or_invalid_roi_fails_clearly():
    service = KneeMRIPredictionService(
        roi_localizer=FixedLocalizer(box=None),
        model_loader=lambda _: FakeModel(),
    )

    with pytest.raises(ACLROINotFoundError, match="slice 1"):
        service.predict_slices([image_bytes()])


def test_localizer_model_loads_with_expected_input_and_mask_output():
    if not Path(LOCALIZER_MODEL_PATH).is_file():
        pytest.skip("ACL ROI localizer artifact is not included in this checkout.")

    model = _load_keras_model(str(LOCALIZER_MODEL_PATH))

    assert tuple(model.input_shape) == (None, 224, 224, 1)
    assert tuple(model.output_shape) == (None, 224, 224, 1)


def test_real_localizer_returns_a_224_pixel_probability_mask():
    if not Path(LOCALIZER_MODEL_PATH).is_file():
        pytest.skip("ACL ROI localizer artifact is not included in this checkout.")
    image = np.asarray(Image.open(BytesIO(image_bytes())).convert("RGB"))

    mask = KerasACLROILocalizer().predict_mask(image)

    assert mask.shape == (224, 224)
    assert np.isfinite(mask).all()
    assert mask.min() >= 0
    assert mask.max() <= 1


def test_mask_bbox_selects_largest_valid_connected_component():
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[10:13, 10:13] = 0.9
    mask[50:80, 40:75] = 0.9
    mask[150:160, 150:160] = 0.9

    box = KerasACLROILocalizer.mask_to_box(mask, (224, 224, 3))

    assert box == ACLROIBox(40, 50, 35, 30)


def test_mask_bbox_maps_224_coordinates_to_original_image():
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[56:112, 28:84] = 0.9

    box = KerasACLROILocalizer.mask_to_box(mask, (448, 896, 3))

    assert box == ACLROIBox(112, 112, 224, 112)


def test_localizer_to_classifier_crops_original_then_preprocesses_roi():
    original = np.zeros((448, 896, 3), dtype=np.uint8)
    original[:, :, 0] = np.arange(896, dtype=np.uint16)[None, :] % 256
    original[:, :, 1] = np.arange(448, dtype=np.uint16)[:, None] % 256
    png_buffer = BytesIO()
    Image.fromarray(original).save(png_buffer, format="PNG")

    predicted_mask = np.zeros((224, 224), dtype=np.float32)
    predicted_mask[56:112, 28:84] = 0.9
    localizer_model = FakeMaskModel(predicted_mask)
    localizer = KerasACLROILocalizer(
        model_loader=lambda _: localizer_model,
    )
    classifier = FakeModel()
    service = KneeMRIPredictionService(
        roi_localizer=localizer,
        model_loader=lambda _: classifier,
    )

    result = service.predict_slices([png_buffer.getvalue()])[0]

    expected_crop = original[112:224, 112:336]
    expected_gray = cv2.cvtColor(expected_crop, cv2.COLOR_RGB2GRAY)
    expected_classifier_input = KneeMRIPredictionService.preprocess_roi(expected_gray)
    expected_localizer_input = cv2.resize(
        cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    assert localizer_model.received.shape == (1, 224, 224, 1)
    assert np.array_equal(localizer_model.received[0, ..., 0], expected_localizer_input)
    assert classifier.received.shape == (1, 75, 75, 1)
    assert np.array_equal(classifier.received[0], expected_classifier_input)
    assert result.roi_box == {"x": 112, "y": 112, "width": 224, "height": 112}
    assert result.localization_status == "localized"


def test_localized_roi_runs_through_real_acl_classifier():
    if not Path(MODEL_PATH).is_file():
        pytest.skip("ACL ResNet-14 classifier artifact is not included in this checkout.")
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[65:145, 75:155] = 0.9
    localizer = KerasACLROILocalizer(
        model_loader=lambda _: FakeMaskModel(mask),
    )
    service = KneeMRIPredictionService(roi_localizer=localizer)
    test_image = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "sample images"
        / "kneetest.jpg"
    )

    result = service.predict_slices([test_image.read_bytes()])[0]

    assert result.localization_status == "localized"
    assert result.class_name in KNEE_CLASS_NAMES
    assert set(result.probabilities) == set(KNEE_CLASS_NAMES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert result.confidence == pytest.approx(result.probabilities[result.class_name])


def test_invalid_localizer_mask_never_loads_classifier():
    invalid_mask = np.full((224, 224), 0.95, dtype=np.float32)
    localizer = KerasACLROILocalizer(
        model_loader=lambda _: FakeMaskModel(invalid_mask),
    )
    classifier_loads = []
    service = KneeMRIPredictionService(
        roi_localizer=localizer,
        model_loader=lambda path: classifier_loads.append(path),
    )

    with pytest.raises(ACLROINotFoundError, match="no valid ROI component"):
        service.predict_slices([image_bytes()])

    assert classifier_loads == []


def test_missing_localizer_model_reports_clear_error(tmp_path):
    localizer = KerasACLROILocalizer(model_path=tmp_path / "missing-localizer.keras")

    with pytest.raises(KneeModelUnavailableError, match="missing-localizer.keras"):
        localizer.predict_mask(np.zeros((256, 256), dtype=np.uint8))


@pytest.mark.parametrize("content", [b"", b"not an image"])
def test_empty_or_corrupt_image_is_rejected(content):
    with pytest.raises(KneeImageValidationError):
        KneeMRIPredictionService.decode_slices([content])


def test_non_image_file_is_rejected():
    with pytest.raises(KneeImageValidationError, match="JPG or PNG"):
        KneeMRIPredictionService.decode_slices([b"%PDF-1.4 fake pdf"])


def test_saved_acl_classifier_loads_with_expected_contract():
    if not Path(MODEL_PATH).is_file():
        pytest.skip("ACL ResNet-14 model artifact is not included in this checkout.")

    model = _load_keras_model(str(MODEL_PATH))

    assert tuple(model.input_shape) == (None, 75, 75, 1)
    assert tuple(model.output_shape) == (None, 3)
    assert KNEE_CLASS_NAMES == (
        "Healthy",
        "Partial ACL tear",
        "Complete ACL tear",
    )


def test_missing_model_file_reports_unavailable(tmp_path):
    with pytest.raises(KneeModelUnavailableError, match="not found"):
        _load_keras_model(str(tmp_path / "missing.keras"))