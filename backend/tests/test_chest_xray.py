from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from backend.services.chest_xray import (
    MODEL_WEIGHTS,
    WEIGHTS_CACHE,
    ChestXrayAnalysisService,
    ChestXrayImageError,
    ChestXrayModelContractError,
    ChestXrayModelUnavailableError,
)


def image_bytes(size=(512, 384), value=120, image_format="PNG"):
    image = Image.new("L", size, color=value)
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class FakeXRVModel(torch.nn.Module):
    input_resolution = 224
    pathologies = [f"Finding {index}" for index in range(18)]

    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, tensor):
        self.received = tensor.detach().clone()
        return torch.linspace(0.01, 0.18, 18, dtype=torch.float32)[None, :]


def test_preprocess_center_crops_normalizes_and_resizes_using_xrv_pipeline():
    service = ChestXrayAnalysisService(model_loader=lambda *_: FakeXRVModel())

    tensor = service.preprocess_image(image_bytes(size=(512, 384)))

    assert tuple(tensor.shape) == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= -1024.0
    assert float(tensor.max()) <= 1024.0


def test_model_scores_include_multiple_findings_sorted_descending():
    model = FakeXRVModel()
    loads = []
    service = ChestXrayAnalysisService(
        model_loader=lambda weights, cache: loads.append((weights, cache)) or model,
    )

    result = service.analyze_bytes(image_bytes())

    assert len(result.findings) == 18
    assert result.findings[0].name == "Finding 17"
    assert result.findings[0].score == pytest.approx(0.18)
    assert result.findings[-1].name == "Finding 0"
    assert result.input_shape == (1, 1, 224, 224)
    assert len(loads) == 1
    assert loads[0] == (MODEL_WEIGHTS, str(WEIGHTS_CACHE))
    assert tuple(model.received.shape) == result.input_shape


@pytest.mark.parametrize("content", [b"", b"not an image", b"%PDF-1.4 fake"])
def test_invalid_upload_is_rejected(content):
    with pytest.raises(ChestXrayImageError):
        ChestXrayAnalysisService._decode_grayscale(content)


def test_unexpected_model_output_shape_is_rejected():
    class WrongOutputModel(FakeXRVModel):
        def forward(self, tensor):
            return torch.zeros((1, 3), dtype=torch.float32)

    service = ChestXrayAnalysisService(model_loader=lambda *_: WrongOutputModel())

    with pytest.raises(ChestXrayModelContractError, match=r"received \(3,\)"):
        service.analyze_bytes(image_bytes())


def test_xrv_pretrained_model_weights_are_cached_and_loadable():
    cached_models = list(Path(WEIGHTS_CACHE).glob("*densenet121-d121*.pt"))
    if not cached_models:
        pytest.skip("TorchXRayVision pretrained checkpoint is not cached locally.")

    service = ChestXrayAnalysisService()
    model = service.load()

    assert model.weights == MODEL_WEIGHTS
    assert len(model.pathologies) == 18


def test_real_cxr_sample_produces_multiple_findings_if_available():
    sample = Path(tempfile.gettempdir()) / "torchxrayvision_00000001_000.png"
    if not sample.is_file():
        pytest.skip("One-image XRV CXR smoke sample is not present in the temp folder.")

    result = ChestXrayAnalysisService().analyze_bytes(sample.read_bytes())

    assert len(result.findings) == 18
    assert all(0.0 <= finding.score <= 1.0 for finding in result.findings)
    assert result.input_shape == (1, 1, 224, 224)