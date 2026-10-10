from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_ID = "best_carotid_ultrasound_model.keras"
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "vascular carotid"
    / MODEL_ID
)
INPUT_SIZE = 256
MASK_THRESHOLD = 0.5
MASK_COLOR_RGB = (255, 40, 40)


class VascularUltrasoundError(RuntimeError):
    pass


class VascularUltrasoundImageError(ValueError):
    pass


class VascularUltrasoundModelUnavailableError(VascularUltrasoundError):
    pass


class VascularUltrasoundModelContractError(VascularUltrasoundError):
    pass


@dataclass(frozen=True)
class VascularUltrasoundPrediction:
    model_name: str
    mask_pixel_count: int
    image_shape: tuple[int, int]
    overlay_png: bytes | None


def _load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise VascularUltrasoundModelUnavailableError(
            f"Carotid ultrasound checkpoint was not found: {path}"
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise VascularUltrasoundModelUnavailableError(
            "Carotid ultrasound inference requires TensorFlow."
        ) from exc

    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise VascularUltrasoundModelUnavailableError(
            f"Carotid ultrasound checkpoint could not be loaded: {path.name}"
        ) from exc
    return model


class VascularUltrasoundAnalysisService:
    def __init__(self, model_path: str | Path = MODEL_PATH, model_loader=None):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_model
        self._model = None

    @staticmethod
    def prepare_image(image_bytes: bytes) -> Image.Image:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise VascularUltrasoundImageError(
                "Upload a non-empty vascular ultrasound JPG or PNG image."
            )
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise VascularUltrasoundImageError(
                        "Vascular ultrasound uploads support JPG and PNG images only."
                    )
                image.load()
                decoded = image.convert("L")
        except VascularUltrasoundImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise VascularUltrasoundImageError(
                "The vascular ultrasound image is corrupt or unreadable."
            ) from exc

        if decoded.width < 2 or decoded.height < 2:
            raise VascularUltrasoundImageError(
                "The vascular ultrasound image dimensions are invalid."
            )
        return decoded

    def load(self):
        if self._model is None:
            model = self.model_loader(self.model_path)
            expected_shape = (None, INPUT_SIZE, INPUT_SIZE, 1)
            if tuple(getattr(model, "input_shape", ())) != expected_shape:
                raise VascularUltrasoundModelContractError(
                    "The carotid model must accept grayscale 256x256 images."
                )
            if tuple(getattr(model, "output_shape", ())) != expected_shape:
                raise VascularUltrasoundModelContractError(
                    "The carotid model must return one 256x256 segmentation mask."
                )
            self._model = model
        return self._model

    def analyze_prepared_image(
        self,
        image: Image.Image,
    ) -> VascularUltrasoundPrediction:
        model = self.load()
        source_gray = np.asarray(image.convert("L"), dtype=np.uint8)
        height, width = source_gray.shape
        model_image = cv2.resize(
            source_gray,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        model_input = model_image[None, ..., None]

        try:
            output = np.asarray(model.predict(model_input, verbose=0))
        except Exception as exc:
            raise VascularUltrasoundModelUnavailableError(
                "The carotid ultrasound model failed during image analysis."
            ) from exc

        expected_output_shape = (1, INPUT_SIZE, INPUT_SIZE, 1)
        if output.shape != expected_output_shape:
            raise VascularUltrasoundModelContractError(
                "The carotid model must return one 256x256 segmentation mask."
            )
        if (
            not np.isfinite(output).all()
            or np.any(output < 0)
            or np.any(output > 1)
        ):
            raise VascularUltrasoundModelContractError(
                "The carotid model returned invalid mask values."
            )

        model_mask = output[0, ..., 0] > MASK_THRESHOLD
        mask_pixel_count = int(np.count_nonzero(model_mask))
        if mask_pixel_count == 0:
            overlay_png = None
        else:
            source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            source_mask = cv2.resize(
                model_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            overlay = source_rgb.copy()
            tint = np.asarray(MASK_COLOR_RGB, dtype=np.float32)
            overlay[source_mask] = (
                overlay[source_mask].astype(np.float32) * 0.55 + tint * 0.45
            ).astype(np.uint8)
            mask_points = cv2.findNonZero(source_mask.astype(np.uint8))
            if mask_points is None:
                raise VascularUltrasoundModelContractError(
                    "The carotid mask could not be mapped to the source image."
                )
            x, y, box_width, box_height = cv2.boundingRect(mask_points)
            cv2.rectangle(
                overlay,
                (x, y),
                (x + box_width - 1, y + box_height - 1),
                (0, 0, 0),
                thickness=5,
            )
            cv2.rectangle(
                overlay,
                (x, y),
                (x + box_width - 1, y + box_height - 1),
                (255, 255, 0),
                thickness=2,
            )
            output_buffer = BytesIO()
            Image.fromarray(overlay).save(output_buffer, format="PNG")
            overlay_png = output_buffer.getvalue()

        return VascularUltrasoundPrediction(
            model_name=MODEL_ID,
            mask_pixel_count=mask_pixel_count,
            image_shape=(height, width),
            overlay_png=overlay_png,
        )

    def analyze_bytes(self, image_bytes: bytes) -> VascularUltrasoundPrediction:
        return self.analyze_prepared_image(self.prepare_image(image_bytes))


__all__ = [
    "INPUT_SIZE",
    "MASK_THRESHOLD",
    "MODEL_ID",
    "MODEL_PATH",
    "VascularUltrasoundAnalysisService",
    "VascularUltrasoundImageError",
    "VascularUltrasoundModelContractError",
    "VascularUltrasoundModelUnavailableError",
    "VascularUltrasoundPrediction",
]
