from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, UnidentifiedImageError

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until cardiac inference is requested
    np = None


CARDIAC_CLASS_NAMES = ("Background", "RV", "Myocardium", "LV")
CARDIAC_CLASS_COLORS = {
    "Background": "#000000",
    "RV": "#ff4d4f",
    "Myocardium": "#ffc53d",
    "LV": "#1677ff",
}
MODEL_PATH = Path(
    os.getenv(
        "CARDIAC_MRI_MODEL_PATH",
        Path(__file__).resolve().parents[2] / "models" / "best_cardiac_model.keras",
    )
)


class CardiacModelUnavailableError(RuntimeError):
    pass


class CardiacModelContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class CardiacSegmentationResult:
    mask_png: bytes
    overlay_png: bytes
    pixel_counts: dict[str, int]


@lru_cache(maxsize=2)
def _load_keras_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise CardiacModelUnavailableError(
            f"Cardiac model file was not found: {path.name}"
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise CardiacModelUnavailableError(
            "Cardiac MRI analysis requires tensorflow-cpu==2.20.0 in the active environment."
        ) from exc

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise CardiacModelUnavailableError(
            "The cardiac Keras model could not be loaded. Check its TensorFlow/Keras compatibility."
        ) from exc


class CardiacSegmentationService:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        model_loader: Callable | None = None,
    ):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_keras_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = self.model_loader(self.model_path)
            input_shape = tuple(self._model.input_shape)
            output_shape = tuple(self._model.output_shape)
            if input_shape != (None, 256, 256, 1):
                raise CardiacModelContractError(
                    f"Expected model input (None, 256, 256, 1), received {input_shape}"
                )
            if output_shape != (None, 256, 256, 4):
                raise CardiacModelContractError(
                    f"Expected output with four classes (None, 256, 256, 4), received {output_shape}"
                )
        return self._model

    @staticmethod
    def _preprocess(image_bytes: bytes):
        if np is None:
            raise CardiacModelUnavailableError(
                "Cardiac MRI analysis requires the TensorFlow CPU dependencies."
            )
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("Please upload a cardiac MRI image in PNG or JPG format.")

        try:
            image = Image.open(BytesIO(image_bytes))
            if image.format not in {"PNG", "JPEG"}:
                image.close()
                raise ValueError(
                    "Please upload a cardiac MRI image in PNG or JPG format."
                )
            grayscale = image.convert("L").resize(
                (256, 256),
                Image.Resampling.BILINEAR,
            )
            image.close()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                "Please upload a cardiac MRI image in PNG or JPG format."
            ) from exc

        normalized = np.asarray(grayscale, dtype=np.float32) / 255.0
        batch = normalized[np.newaxis, :, :, np.newaxis]
        return grayscale, batch

    def segment(self, image_bytes: bytes) -> CardiacSegmentationResult:
        grayscale, batch = self._preprocess(image_bytes)
        model = self._get_model()

        try:
            prediction = np.asarray(model.predict(batch, verbose=0))
        except Exception as exc:
            raise CardiacModelUnavailableError(
                "The cardiac model failed during inference."
            ) from exc

        if prediction.shape != (1, 256, 256, 4):
            raise CardiacModelContractError(
                f"Expected prediction shape (1, 256, 256, 4), received {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise CardiacModelContractError("The cardiac model returned non-finite values")

        mask = np.argmax(prediction, axis=-1)[0].astype(np.uint8)
        pixel_counts = {
            class_name: int(np.count_nonzero(mask == class_index))
            for class_index, class_name in enumerate(CARDIAC_CLASS_NAMES)
        }

        colors = np.array(
            [
                (0, 0, 0),
                (255, 77, 79),
                (255, 197, 61),
                (22, 119, 255),
            ],
            dtype=np.uint8,
        )
        base_image = np.asarray(grayscale.convert("RGB"), dtype=np.uint8)
        class_colors = colors[mask]
        overlay = base_image.copy()
        foreground = mask > 0
        overlay[foreground] = (
            base_image[foreground].astype(np.float32) * 0.45
            + class_colors[foreground].astype(np.float32) * 0.55
        ).clip(0, 255).astype(np.uint8)
        overlay_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_image)
        for class_index, class_name in enumerate(CARDIAC_CLASS_NAMES[1:], start=1):
            rows, columns = np.nonzero(mask == class_index)
            if not len(rows):
                continue
            bounds = (
                int(columns.min()),
                int(rows.min()),
                int(columns.max()),
                int(rows.max()),
            )
            draw.rectangle(bounds, outline=(0, 0, 0), width=5)
            draw.rectangle(
                bounds,
                outline=tuple(colors[class_index].tolist()),
                width=2,
            )

        mask_buffer = BytesIO()
        Image.fromarray(mask * 85).save(mask_buffer, format="PNG")
        overlay_buffer = BytesIO()
        overlay_image.save(overlay_buffer, format="PNG")

        return CardiacSegmentationResult(
            mask_png=mask_buffer.getvalue(),
            overlay_png=overlay_buffer.getvalue(),
            pixel_counts=pixel_counts,
        )


__all__ = [
    "CARDIAC_CLASS_NAMES",
    "CARDIAC_CLASS_COLORS",
    "CardiacModelUnavailableError",
    "CardiacModelContractError",
    "CardiacSegmentationResult",
    "CardiacSegmentationService",
]