from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "yakshpanchal/bone-fracture-resnet50"
LABELS = {0: "Fractured", 1: "Normal"}


class BoneXrayError(RuntimeError):
    pass


class BoneXrayImageError(ValueError):
    pass


class BoneXrayModelUnavailableError(BoneXrayError):
    pass


class BoneXrayModelContractError(BoneXrayError):
    pass


@dataclass(frozen=True)
class BoneXrayResult:
    label: str
    confidence: float


@lru_cache(maxsize=1)
def _load_bone_processor():
    try:
        return AutoImageProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:
        raise BoneXrayModelUnavailableError(
            f"Failed to load bone fracture image processor from Hugging Face: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def _load_bone_model():
    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID, config=config)
    except Exception as exc:
        raise BoneXrayModelUnavailableError(
            f"Failed to load bone fracture model from Hugging Face: {exc}"
        ) from exc

    labels = config.id2label or {}
    if set(labels) != set(LABELS) or any(labels[i] != name for i, name in LABELS.items()):
        raise BoneXrayModelContractError(
            "Unexpected class labels on the bone fracture model: "
            f"{labels} (expected {LABELS})."
        )

    model.eval()
    return model


class BoneXrayAnalysisService:
    def __init__(self, model_loader=None, processor_loader=None):
        self.model_loader = model_loader or _load_bone_model
        self.processor_loader = processor_loader or _load_bone_processor
        self._model = None
        self._processor = None

    def load(self):
        if self._model is None:
            self._model = self.model_loader()
        return self._model

    def load_processor(self):
        if self._processor is None:
            self._processor = self.processor_loader()
        return self._processor

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise BoneXrayImageError("Upload a non-empty Bone/Joint X-ray JPG or PNG image.")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise BoneXrayImageError(
                        "Bone/Joint X-ray uploads support JPG and PNG images only."
                    )
                image.load()
                rgb = image.convert("RGB")
        except BoneXrayImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise BoneXrayImageError(
                "The Bone/Joint X-ray image is corrupt or unreadable."
            ) from exc

        processor = self.load_processor()
        pixel_values = processor(images=rgb, return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze(0)

    def analyze_bytes(self, image_bytes: bytes) -> BoneXrayResult:
        model = self.load()
        tensor = self.preprocess_image(image_bytes)
        try:
            with torch.no_grad():
                outputs = model(tensor.unsqueeze(0))
                probs = torch.softmax(outputs.logits, dim=-1)
        except Exception as exc:
            raise BoneXrayModelUnavailableError(
                f"Bone X-ray inference failed: {exc}"
            ) from exc

        if probs.shape != (1, len(LABELS)):
            raise BoneXrayModelContractError(
                f"Expected {len(LABELS)} class scores, received {tuple(probs.shape)}."
            )
        if not torch.isfinite(probs).all():
            raise BoneXrayModelContractError(
                "Bone X-ray model returned non-finite class probabilities."
            )

        confidence, predicted_idx = torch.max(probs, dim=1)
        label = LABELS.get(predicted_idx.item())
        if label is None:
            raise BoneXrayModelContractError(
                f"Bone X-ray model predicted unknown class index {predicted_idx.item()}."
            )

        return BoneXrayResult(label=label, confidence=confidence.item() * 100)


__all__ = [
    "BoneXrayAnalysisService",
    "BoneXrayError",
    "BoneXrayImageError",
    "BoneXrayModelContractError",
    "BoneXrayModelUnavailableError",
    "BoneXrayResult",
]
