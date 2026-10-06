from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import pytest
from PIL import Image

from backend.services.dental_xray import (
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    MODEL_ID,
    DentalXrayAnalysisService,
    DentalXrayImageError,
    DentalXrayModelContractError,
    DentalXrayModelUnavailableError,
)


def image_bytes(size=(640, 320), value=140, image_format="PNG"):
    image = Image.new("RGB", size, color=(value, value, value))
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _TensorLike(xyxy)
        self.conf = _TensorLike(conf)
        self.cls = _TensorLike(cls)

    def __len__(self):
        return len(self.xyxy.data)


class _TensorLike:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYoloModel:
    names = {
        0: "caries",
        1: "deep_caries",
        2: "periapical_lesion",
        3: "impacted_tooth",
    }

    def __init__(self, results):
        self._results = results
        self.received = None
        self.received_conf = None

    def predict(self, image, conf=None, verbose=False):
        self.received = image
        self.received_conf = conf
        return self._results


def make_service(results):
    model = FakeYoloModel(results)
    service = DentalXrayAnalysisService(model_loader=lambda: model)
    return service, model


def test_decode_image_returns_rgb_for_valid_png():
    image = DentalXrayAnalysisService.decode_image(image_bytes())

    assert image.mode == "RGB"
    assert image.size == (640, 320)


@pytest.mark.parametrize("content", [b"", b"not an image", b"%PDF-1.4 fake"])
def test_invalid_upload_is_rejected(content):
    with pytest.raises(DentalXrayImageError):
        DentalXrayAnalysisService.decode_image(content)


def test_unsupported_image_format_is_rejected():
    image = Image.new("RGB", (32, 32), color=(10, 10, 10))
    output = BytesIO()
    image.save(output, format="BMP")

    with pytest.raises(DentalXrayImageError, match="JPG and PNG"):
        DentalXrayAnalysisService.decode_image(output.getvalue())


def test_no_detections_returns_empty_findings():
    service, model = make_service([FakeResult(None)])

    result = service.analyze_bytes(image_bytes())

    assert result.findings == []
    assert result.model_name == MODEL_ID
    assert result.image_shape == (320, 640)
    assert model.received_conf == CONFIDENCE_THRESHOLD


def test_findings_are_labelled_and_sorted_by_score_descending():
    boxes = FakeBoxes(
        xyxy=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        conf=[0.30, 0.91, 0.55],
        cls=[0, 2, 3],
    )
    service, model = make_service([FakeResult(boxes)])

    result = service.analyze_bytes(image_bytes())

    assert [finding.name for finding in result.findings] == [
        CLASS_LABELS[2],
        CLASS_LABELS[3],
        CLASS_LABELS[0],
    ]
    assert [round(finding.score, 2) for finding in result.findings] == [0.91, 0.55, 0.30]
    assert result.findings[0].box == (5.0, 6.0, 7.0, 8.0)
    assert model.received.shape == (320, 640, 3)


def test_unknown_class_index_is_rejected():
    boxes = FakeBoxes(xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[0.8], cls=[9])
    service, _ = make_service([FakeResult(boxes)])

    with pytest.raises(DentalXrayModelContractError, match="unknown class index 9"):
        service.analyze_bytes(image_bytes())


def test_non_finite_score_is_rejected():
    boxes = FakeBoxes(xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[float("nan")], cls=[0])
    service, _ = make_service([FakeResult(boxes)])

    with pytest.raises(DentalXrayModelContractError, match="non-finite"):
        service.analyze_bytes(image_bytes())


def test_mismatched_box_and_score_counts_are_rejected():
    boxes = FakeBoxes(xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[0.5, 0.6], cls=[0, 1])
    service, _ = make_service([FakeResult(boxes)])

    with pytest.raises(DentalXrayModelContractError, match="mismatched"):
        service.analyze_bytes(image_bytes())


def test_multiple_detection_results_are_rejected():
    service, _ = make_service([FakeResult(None), FakeResult(None)])

    with pytest.raises(DentalXrayModelContractError, match="one detection result"):
        service.analyze_bytes(image_bytes())


def test_inference_failure_maps_to_model_unavailable():
    class ExplodingModel(FakeYoloModel):
        def predict(self, image, conf=None, verbose=False):
            raise RuntimeError("boom")

    service = DentalXrayAnalysisService(model_loader=lambda: ExplodingModel([]))

    with pytest.raises(DentalXrayModelUnavailableError, match="inference failed"):
        service.analyze_bytes(image_bytes())


def test_class_label_contract_matches_checkpoint_names():
    assert CLASS_LABELS == {
        0: "Caries",
        1: "Deep caries",
        2: "Periapical lesion",
        3: "Impacted tooth",
    }


def test_pretrained_model_loads_from_cache_if_available():
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cached = list(cache_root.glob("**/oralguard_det_best.pt")) if cache_root.is_dir() else []
    if not cached:
        pytest.skip("OralGuard checkpoint is not cached locally.")

    model = DentalXrayAnalysisService().load()

    assert {int(k): v for k, v in model.names.items()} == {
        0: "caries",
        1: "deep_caries",
        2: "periapical_lesion",
        3: "impacted_tooth",
    }


def test_real_panoramic_sample_runs_if_available():
    sample = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "sample images"
        / "dentaltest.jpg"
    )
    if not sample.is_file():
        pytest.skip("A real panoramic dental X-ray sample is not present in the workspace.")

    result = DentalXrayAnalysisService().analyze_bytes(sample.read_bytes())

    assert result.model_name == MODEL_ID
    assert result.findings, "The real abscessed-tooth radiograph should yield detections."
    assert all(0.0 <= finding.score <= 1.0 for finding in result.findings)
    assert all(len(finding.box) == 4 for finding in result.findings)
    assert all(finding.name in CLASS_LABELS.values() for finding in result.findings)
