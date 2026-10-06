from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


BRAIN_MODEL_PATH = Path(
    os.getenv(
        "BRAIN_MRI_MODEL_PATH",
        Path(__file__).resolve().parents[2]
        / "models"
        / "mri_brain_tumor_efficientnetb0_final.keras",
    )
)
BRAIN_IMAGE_SIZE = (224, 224)
BRAIN_OUTPUT_NAMES = (
    "Glioma",
    "Meningioma",
    "No tumor",
    "Pituitary tumor",
)


class BrainMRIError(RuntimeError):
    pass


class BrainImageValidationError(ValueError):
    pass


class BrainModelUnavailableError(BrainMRIError):
    pass


class BrainModelContractError(BrainMRIError):
    pass


@dataclass(frozen=True)
class BrainMRIPrediction:
    class_index: int
    class_score: float
    class_scores: dict[str, float]


@lru_cache(maxsize=1)
def _load_keras_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise BrainModelUnavailableError(
            f"Brain MRI model file was not found: {path.name}"
        )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise BrainModelUnavailableError(
            "Brain MRI inference requires tensorflow-cpu==2.20.0."
        ) from exc
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise BrainModelUnavailableError(
            f"Brain MRI model could not be loaded: {path.name}"
        ) from exc


class BrainMRIPredictionService:
    def __init__(self, model_path: str | Path = BRAIN_MODEL_PATH, model_loader=None):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_keras_model
        self._model = None

    def load(self):
        if self._model is None:
            self._model = self.model_loader(self.model_path)
            input_shape = tuple(self._model.input_shape)
            output_shape = tuple(self._model.output_shape)
            if input_shape != (None, 224, 224, 3):
                raise BrainModelContractError(
                    "Expected Brain MRI model input (None, 224, 224, 3), "
                    f"received {input_shape}."
                )
            if output_shape != (None, 4):
                raise BrainModelContractError(
                    f"Expected Brain MRI model output (None, 4), received {output_shape}."
                )
        return self._model

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> np.ndarray:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise BrainImageValidationError("Upload a non-empty Brain MRI JPG or PNG image.")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise BrainImageValidationError(
                        "Brain MRI uploads support JPG and PNG images only."
                    )
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except BrainImageValidationError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise BrainImageValidationError(
                "The Brain MRI image is corrupt or unreadable."
            ) from exc

        if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise BrainImageValidationError("The Brain MRI image has invalid dimensions.")

        resized = cv2.resize(
            rgb,
            BRAIN_IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )
        # The saved EfficientNet contains Rescaling and Normalization layers.
        return resized.astype(np.float32)[np.newaxis, ...]

    def predict_preprocessed(self, batch: np.ndarray) -> BrainMRIPrediction:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.shape != (1, 224, 224, 3):
            raise BrainModelContractError(
                "Expected one Brain MRI tensor shaped (1, 224, 224, 3), "
                f"received {batch.shape}."
            )
        if not np.isfinite(batch).all():
            raise BrainImageValidationError("Brain MRI input contains invalid pixel values.")

        model = self.load()
        try:
            scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        except Exception as exc:
            raise BrainModelUnavailableError(
                "The Brain MRI model failed during inference."
            ) from exc

        if scores.shape != (1, 4):
            raise BrainModelContractError(
                f"Expected four Brain MRI output scores, received {scores.shape}."
            )
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise BrainModelContractError(
                "The Brain MRI model returned non-finite or out-of-range scores."
            )
        if not np.isclose(scores[0].sum(), 1.0, atol=1e-3):
            raise BrainModelContractError(
                "The Brain MRI model output does not sum to a softmax distribution."
            )

        class_index = int(np.argmax(scores[0]))
        return BrainMRIPrediction(
            class_index=class_index,
            class_score=float(scores[0, class_index]),
            class_scores={
                name: float(scores[0, index])
                for index, name in enumerate(BRAIN_OUTPUT_NAMES)
            },
        )

    def predict_image(self, image_bytes: bytes) -> BrainMRIPrediction:
        return self.predict_preprocessed(self.preprocess_image(image_bytes))


__all__ = [
    "BRAIN_IMAGE_SIZE",
    "BRAIN_MODEL_PATH",
    "BRAIN_OUTPUT_NAMES",
    "BrainImageValidationError",
    "BrainModelContractError",
    "BrainModelUnavailableError",
    "BrainMRIPrediction",
    "BrainMRIPredictionService",
]