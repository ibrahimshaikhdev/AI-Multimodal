from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


SPINE_MODEL_PATH = Path(
    os.getenv(
        "SPINE_MRI_MODEL_PATH",
        Path(__file__).resolve().parents[2] / "models" / "best_mrnet_fast.keras",
    )
)
SPINE_FRAME_COUNT = 12
SPINE_FRAME_SIZE = (160, 160)


class SpineMRIError(RuntimeError):
    pass


class SpineImageValidationError(ValueError):
    pass


class SpineModelUnavailableError(SpineMRIError):
    pass


class SpineModelContractError(SpineMRIError):
    pass


@dataclass(frozen=True)
class SpineModelScore:
    score: float
    frame_count: int = SPINE_FRAME_COUNT
    frame_source: str = "single_uploaded_image_repeated"


@lru_cache(maxsize=1)
def _load_keras_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise SpineModelUnavailableError(
            f"Spine MRI model file was not found: {path.name}"
        )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SpineModelUnavailableError(
            "Spine MRI inference requires tensorflow-cpu==2.20.0."
        ) from exc

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise SpineModelUnavailableError(
            f"Spine MRI model could not be loaded: {path.name}"
        ) from exc


class SpineMRIPredictionService:
    def __init__(self, model_path: str | Path = SPINE_MODEL_PATH, model_loader=None):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_keras_model
        self._model = None

    def load(self):
        if self._model is None:
            self._model = self.model_loader(self.model_path)
            input_shape = tuple(self._model.input_shape)
            output_shape = tuple(self._model.output_shape)
            if input_shape != (None, 12, 160, 160, 3):
                raise SpineModelContractError(
                    "Expected Spine MRI model input (None, 12, 160, 160, 3), "
                    f"received {input_shape}."
                )
            if output_shape != (None, 1):
                raise SpineModelContractError(
                    f"Expected Spine MRI model output (None, 1), received {output_shape}."
                )
        return self._model

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> np.ndarray:
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
            SPINE_FRAME_SIZE,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        # The serialized EfficientNet encoder contains its own rescaling layers.
        repeated_frames = np.repeat(resized[np.newaxis, ...], SPINE_FRAME_COUNT, axis=0)
        return repeated_frames[np.newaxis, ...]

    def predict_preprocessed(self, batch: np.ndarray) -> SpineModelScore:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.shape != (1, 12, 160, 160, 3):
            raise SpineModelContractError(
                "Expected one Spine MRI input tensor shaped (1, 12, 160, 160, 3), "
                f"received {batch.shape}."
            )
        if not np.isfinite(batch).all():
            raise SpineImageValidationError("Spine MRI input contains invalid pixel values.")

        model = self.load()
        try:
            prediction = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        except Exception as exc:
            raise SpineModelUnavailableError(
                "The Spine MRI model failed during inference."
            ) from exc

        if prediction.shape != (1, 1):
            raise SpineModelContractError(
                f"Expected one sigmoid model score shaped (1, 1), received {prediction.shape}."
            )
        if not np.isfinite(prediction).all() or not 0.0 <= float(prediction[0, 0]) <= 1.0:
            raise SpineModelContractError(
                "The Spine MRI model returned a non-finite or out-of-range score."
            )
        return SpineModelScore(score=float(prediction[0, 0]))

    def predict_image(self, image_bytes: bytes) -> SpineModelScore:
        return self.predict_preprocessed(self.preprocess_image(image_bytes))


__all__ = [
    "SPINE_FRAME_COUNT",
    "SPINE_FRAME_SIZE",
    "SPINE_MODEL_PATH",
    "SpineImageValidationError",
    "SpineModelContractError",
    "SpineModelScore",
    "SpineModelUnavailableError",
    "SpineMRIPredictionService",
]