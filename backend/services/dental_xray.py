from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_REPO = "Enosh729/oralguard"
MODEL_FILENAME = "oralguard_det_best.pt"
MODEL_ID = f"{MODEL_REPO}/{MODEL_FILENAME}"
# Raw class names as serialized in the OralGuard checkpoint.
MODEL_CLASS_NAMES = {
    0: "caries",
    1: "deep_caries",
    2: "periapical_lesion",
    3: "impacted_tooth",
}
# Human-readable labels surfaced to the API and UI.
CLASS_LABELS = {
    0: "Caries",
    1: "Deep caries",
    2: "Periapical lesion",
    3: "Impacted tooth",
}
CONFIDENCE_THRESHOLD = 0.25


class DentalXrayError(RuntimeError):
    pass


class DentalXrayImageError(ValueError):
    pass


class DentalXrayModelUnavailableError(DentalXrayError):
    pass


class DentalXrayModelContractError(DentalXrayError):
    pass


@dataclass(frozen=True)
class DentalXrayFinding:
    name: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DentalXrayAnalysis:
    model_name: str
    findings: list[DentalXrayFinding]
    image_shape: tuple[int, int]


@lru_cache(maxsize=1)
def _load_dental_model():
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as exc:
        raise DentalXrayModelUnavailableError(
            "Dental X-ray inference requires ultralytics and huggingface_hub."
        ) from exc

    try:
        weights_path = hf_hub_download(MODEL_REPO, MODEL_FILENAME)
        model = YOLO(weights_path)
    except Exception as exc:
        raise DentalXrayModelUnavailableError(
            f"Failed to load the OralGuard dental detector: {exc}"
        ) from exc

    names = getattr(model, "names", None) or {}
    if {int(index): str(name) for index, name in names.items()} != MODEL_CLASS_NAMES:
        raise DentalXrayModelContractError(
            f"Unexpected OralGuard class labels: {names} (expected {MODEL_CLASS_NAMES})."
        )
    return model


class DentalXrayAnalysisService:
    def __init__(self, model_loader=None):
        self.model_loader = model_loader or _load_dental_model
        self._model = None

    def load(self):
        if self._model is None:
            self._model = self.model_loader()
        return self._model

    @staticmethod
    def decode_image(image_bytes: bytes) -> Image.Image:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise DentalXrayImageError(
                "Upload a non-empty panoramic dental X-ray JPG or PNG image."
            )
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise DentalXrayImageError(
                        "Dental X-ray uploads support JPG and PNG images only."
                    )
                image.load()
                return image.convert("RGB")
        except DentalXrayImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise DentalXrayImageError(
                "The dental X-ray image is corrupt or unreadable."
            ) from exc

    def analyze_bytes(self, image_bytes: bytes) -> DentalXrayAnalysis:
        image = self.decode_image(image_bytes)
        model = self.load()
        try:
            results = model.predict(
                np.asarray(image),
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            raise DentalXrayModelUnavailableError(
                f"Dental X-ray inference failed: {exc}"
            ) from exc

        if len(results) != 1:
            raise DentalXrayModelContractError(
                f"Expected one detection result, received {len(results)}."
            )

        boxes = getattr(results[0], "boxes", None)
        findings: list[DentalXrayFinding] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            if not (len(xyxy) == len(confidences) == len(classes)):
                raise DentalXrayModelContractError(
                    "Dental detector returned mismatched box, score, and class counts."
                )
            for coordinates, score, class_index in zip(xyxy, confidences, classes):
                name = CLASS_LABELS.get(int(class_index))
                if name is None:
                    raise DentalXrayModelContractError(
                        f"Dental detector returned unknown class index {class_index}."
                    )
                if not np.isfinite(score) or not 0.0 <= float(score) <= 1.0:
                    raise DentalXrayModelContractError(
                        "Dental detector returned a non-finite or out-of-range score."
                    )
                findings.append(
                    DentalXrayFinding(
                        name=name,
                        score=float(score),
                        box=tuple(float(value) for value in coordinates),
                    )
                )

        findings.sort(key=lambda finding: finding.score, reverse=True)
        return DentalXrayAnalysis(
            model_name=MODEL_ID,
            findings=findings,
            image_shape=(image.height, image.width),
        )


__all__ = [
    "CLASS_LABELS",
    "CONFIDENCE_THRESHOLD",
    "DentalXrayAnalysis",
    "DentalXrayAnalysisService",
    "DentalXrayError",
    "DentalXrayFinding",
    "DentalXrayImageError",
    "DentalXrayModelContractError",
    "DentalXrayModelUnavailableError",
    "MODEL_ID",
]
