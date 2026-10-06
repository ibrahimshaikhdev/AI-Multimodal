from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn


SPINE_MODEL_PATH = Path(
    os.getenv(
        "SPINE_MRI_MODEL_PATH",
        Path(__file__).resolve().parents[2]
        / "models"
        / "lumbar_spine_mrimperium_best.pth",
    )
)
SPINE_IMAGE_SIZE = (224, 224)
SPINE_MODEL_ID = "mrimperium/Lumbar-Spine-Degenerative-Classification"
SPINE_CONDITIONS = (
    "Spinal canal stenosis",
    "Neural foraminal narrowing",
    "Subarticular stenosis",
)
SPINE_SEVERITIES = ("Normal/Mild", "Moderate", "Severe")
SPINE_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
SPINE_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class SpineMRIError(RuntimeError):
    pass


class SpineImageValidationError(ValueError):
    pass


class SpineModelUnavailableError(SpineMRIError):
    pass


class SpineModelContractError(SpineMRIError):
    pass


@dataclass(frozen=True)
class SpineMRIPrediction:
    condition_scores: dict[str, dict[str, float]]

    @property
    def highest_score(self) -> float:
        return max(
            score
            for severity_scores in self.condition_scores.values()
            for score in severity_scores.values()
        )


class _LumbarSpineConditionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans=3,
            features_only=False,
            num_classes=0,
        )
        self.num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(SPINE_CONDITIONS) * len(SPINE_SEVERITIES)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        return self.classifier(features)


def _load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise SpineModelUnavailableError(
            f"Spine MRI model file was not found: {path.name}"
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise SpineModelUnavailableError(
            f"Spine MRI checkpoint could not be read: {path.name}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise SpineModelContractError("Spine MRI checkpoint must contain a state dictionary.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SpineModelContractError(
            "Spine MRI checkpoint does not contain model_state_dict."
        )

    model = _LumbarSpineConditionModel()
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise SpineModelContractError(
            "Spine MRI checkpoint weights do not match the documented ResNet-18 model."
        ) from exc
    model.eval()
    return model


class SpineMRIPredictionService:
    def __init__(self, model_path: str | Path = SPINE_MODEL_PATH, model_loader=None):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_model
        self._model = None

    def load(self):
        if self._model is None:
            self._model = self.model_loader(self.model_path)
            if not isinstance(self._model, nn.Module):
                raise SpineModelContractError("Spine MRI loader did not return a PyTorch model.")
            self._model.eval()
        return self._model

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> torch.Tensor:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise SpineImageValidationError("Upload a non-empty Spine MRI JPG or PNG image.")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise SpineImageValidationError(
                        "Spine MRI uploads support JPG and PNG images only."
                    )
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except SpineImageValidationError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise SpineImageValidationError(
                "The Spine MRI image is corrupt or unreadable."
            ) from exc

        if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise SpineImageValidationError("The Spine MRI image has invalid dimensions.")

        resized = cv2.resize(
            rgb,
            SPINE_IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32) / 255.0
        normalized = (resized - SPINE_IMAGENET_MEAN) / SPINE_IMAGENET_STD
        chw = np.transpose(normalized, (2, 0, 1)).copy()
        return torch.from_numpy(chw).unsqueeze(0)

    def predict_preprocessed(self, batch: torch.Tensor) -> SpineMRIPrediction:
        if not isinstance(batch, torch.Tensor):
            raise SpineModelContractError("Spine MRI model input must be a PyTorch tensor.")
        if tuple(batch.shape) != (1, 3, *SPINE_IMAGE_SIZE):
            raise SpineModelContractError(
                "Expected one Spine MRI tensor shaped (1, 3, 224, 224), "
                f"received {tuple(batch.shape)}."
            )
        if not torch.isfinite(batch).all():
            raise SpineImageValidationError("Spine MRI input contains invalid pixel values.")

        model = self.load()
        try:
            with torch.inference_mode():
                logits = model(batch.to(device="cpu", dtype=torch.float32))
                scores = torch.sigmoid(logits).cpu().numpy()
        except Exception as exc:
            raise SpineModelUnavailableError(
                "The Spine MRI model failed during inference."
            ) from exc

        expected_shape = (1, len(SPINE_CONDITIONS) * len(SPINE_SEVERITIES))
        if tuple(scores.shape) != expected_shape:
            raise SpineModelContractError(
                f"Expected nine Spine MRI scores shaped {expected_shape}, "
                f"received {tuple(scores.shape)}."
            )
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise SpineModelContractError(
                "The Spine MRI model returned non-finite or out-of-range scores."
            )

        grouped_scores = scores[0].reshape(len(SPINE_CONDITIONS), len(SPINE_SEVERITIES))
        return SpineMRIPrediction(
            condition_scores={
                condition: {
                    severity: float(grouped_scores[condition_index, severity_index])
                    for severity_index, severity in enumerate(SPINE_SEVERITIES)
                }
                for condition_index, condition in enumerate(SPINE_CONDITIONS)
            }
        )

    def predict_image(self, image_bytes: bytes) -> SpineMRIPrediction:
        return self.predict_preprocessed(self.preprocess_image(image_bytes))


__all__ = [
    "SPINE_CONDITIONS",
    "SPINE_IMAGE_SIZE",
    "SPINE_MODEL_ID",
    "SPINE_MODEL_PATH",
    "SPINE_SEVERITIES",
    "SpineImageValidationError",
    "SpineModelContractError",
    "SpineModelUnavailableError",
    "SpineMRIPrediction",
    "SpineMRIPredictionService",
]
