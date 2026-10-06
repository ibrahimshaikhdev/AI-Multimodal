from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_WEIGHTS = "densenet121-res224-all"
WEIGHTS_CACHE = Path.home() / ".torchxrayvision"


class ChestXrayError(RuntimeError):
    pass


class ChestXrayImageError(ValueError):
    pass


class ChestXrayModelUnavailableError(ChestXrayError):
    pass


class ChestXrayModelContractError(ChestXrayError):
    pass


@dataclass(frozen=True)
class ChestXrayFindingScore:
    name: str
    score: float


@dataclass(frozen=True)
class ChestXrayAnalysis:
    model_name: str
    findings: list[ChestXrayFindingScore]
    input_shape: tuple[int, ...]


@lru_cache(maxsize=1)
def _load_xrv_model(weights: str, cache_dir: str):
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ChestXrayModelUnavailableError(
            "Chest X-ray inference requires torch and torchxrayvision."
        ) from exc

    try:
        model = xrv.models.DenseNet(
            weights=weights,
            cache_dir=cache_dir,
            apply_sigmoid=True,
        )
        model.eval()
        return model
    except Exception as exc:
        raise ChestXrayModelUnavailableError(
            f"TorchXRayVision weights could not be loaded from the local cache: {weights}"
        ) from exc


class ChestXrayAnalysisService:
    def __init__(self, model_loader=None):
        self.model_loader = model_loader or _load_xrv_model
        self._model = None
        self._torch = None
        self._xrv = None

    def load(self):
        if self._model is None:
            try:
                import torch
                import torchxrayvision as xrv
            except ImportError as exc:
                raise ChestXrayModelUnavailableError(
                    "Chest X-ray inference requires torch and torchxrayvision."
                ) from exc
            self._torch = torch
            self._xrv = xrv
            self._model = self.model_loader(MODEL_WEIGHTS, str(WEIGHTS_CACHE))
            pathologies = getattr(self._model, "pathologies", None)
            if not pathologies or len(pathologies) != 18:
                raise ChestXrayModelContractError(
                    "Expected densenet121-res224-all to expose 18 finding labels."
                )
            if getattr(self._model, "input_resolution", None) != 224:
                raise ChestXrayModelContractError(
                    "Expected densenet121-res224-all native input resolution 224."
                )
            self._pathologies = list(pathologies)
        return self._model

    def preprocess_image(self, image_bytes: bytes):
        grayscale = self._decode_grayscale(image_bytes)
        model = self.load()
        try:
            center_crop = self._xrv.datasets.XRayCenterCrop()(grayscale[np.newaxis, ...])
            normalized = self._xrv.utils.normalize(center_crop, maxval=255)
            tensor = self._torch.from_numpy(normalized).unsqueeze(0)
            tensor = self._torch.nn.functional.interpolate(
                tensor,
                size=(model.input_resolution, model.input_resolution),
                mode="bilinear",
                antialias=True,
            )
        except Exception as exc:
            raise ChestXrayImageError(
                "Could not crop, normalize, or resize the chest X-ray image."
            ) from exc
        return tensor

    @staticmethod
    def _decode_grayscale(image_bytes: bytes) -> np.ndarray:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ChestXrayImageError("Upload a non-empty chest X-ray JPG or PNG image.")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ChestXrayImageError(
                        "Chest X-ray uploads support JPG and PNG images only."
                    )
                image.load()
                grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        except ChestXrayImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ChestXrayImageError(
                "The chest X-ray image is corrupt or unreadable."
            ) from exc
        if grayscale.ndim != 2 or not grayscale.size:
            raise ChestXrayImageError("The chest X-ray image has invalid dimensions.")
        return grayscale

    def analyze_bytes(self, image_bytes: bytes) -> ChestXrayAnalysis:
        model = self.load()
        tensor = self.preprocess_image(image_bytes)
        try:
            with self._torch.inference_mode():
                output = model(tensor)
            scores = output.detach().cpu().numpy().reshape(-1)
        except Exception as exc:
            raise ChestXrayModelUnavailableError(
                "TorchXRayVision failed during chest X-ray inference."
            ) from exc

        if scores.shape != (18,):
            raise ChestXrayModelContractError(
                f"Expected 18 finding scores, received {scores.shape}."
            )
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ChestXrayModelContractError(
                "TorchXRayVision returned non-finite or out-of-range finding scores."
            )

        findings = [
            ChestXrayFindingScore(name=name, score=float(scores[index]))
            for index, name in enumerate(self._pathologies)
        ]
        findings.sort(key=lambda finding: finding.score, reverse=True)
        return ChestXrayAnalysis(
            model_name=MODEL_WEIGHTS,
            findings=findings,
            input_shape=tuple(tensor.shape),
        )


__all__ = [
    "ChestXrayAnalysis",
    "ChestXrayAnalysisService",
    "ChestXrayError",
    "ChestXrayFindingScore",
    "ChestXrayImageError",
    "ChestXrayModelContractError",
    "ChestXrayModelUnavailableError",
    "MODEL_WEIGHTS",
    "WEIGHTS_CACHE",
]