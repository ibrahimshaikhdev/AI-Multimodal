from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError


MODEL_NAME = "best_chest_ct_model.keras"
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "chest ct scan model"
    / MODEL_NAME
)
INPUT_SIZE = 256
MASK_THRESHOLD = 0.5
FINDING_NAMES = ("Ground-glass opacity", "Consolidation")
REPORTED_TEST_DICE = 0.1711
MAX_IMAGE_PIXELS = 25_000_000


class ChestCTError(RuntimeError):
    pass


class ChestCTImageError(ValueError):
    pass


class ChestCTModelUnavailableError(ChestCTError):
    pass


class ChestCTModelContractError(ChestCTError):
    pass


@dataclass(frozen=True)
class ChestCTSegmentation:
    model_name: str
    pixel_counts: dict[str, int]
    image_shape: tuple[int, int]
    threshold: float
    overlay_png: bytes


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    if not Path(model_path).is_file():
        raise ChestCTModelUnavailableError(
            f"Chest CT model file was not found: {model_path}"
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ChestCTModelUnavailableError(
            "Chest CT segmentation requires TensorFlow."
        ) from exc

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise ChestCTModelUnavailableError(
            f"Could not load the chest CT model: {exc}"
        ) from exc


class ChestCTSegmentationService:
    def __init__(self, model_loader=None):
        self.model_loader = model_loader or _load_model
        self._model = None

    def load(self):
        if self._model is None:
            try:
                self._model = self.model_loader(str(MODEL_PATH))
            except ChestCTError:
                raise
            except Exception as exc:
                raise ChestCTModelUnavailableError(
                    f"Could not load the chest CT model: {exc}"
                ) from exc

            if getattr(self._model, "input_shape", None) != (
                None,
                INPUT_SIZE,
                INPUT_SIZE,
                1,
            ):
                raise ChestCTModelContractError(
                    "Expected a 256x256 single-channel chest CT model input."
                )
            if getattr(self._model, "output_shape", None) != (
                None,
                INPUT_SIZE,
                INPUT_SIZE,
                len(FINDING_NAMES),
            ):
                raise ChestCTModelContractError(
                    "Expected two 256x256 segmentation output channels."
                )
        return self._model

    @staticmethod
    def _preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ChestCTImageError("Upload a non-empty chest CT slice image.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ChestCTImageError(
                        "Chest CT slice image exceeds the supported pixel limit."
                    )
                source = np.asarray(image.convert("L"), dtype=np.uint8)
        except ChestCTImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ChestCTImageError(
                "Could not decode the chest CT slice as a JPG or PNG image."
            ) from exc

        if source.ndim != 2 or not source.size:
            raise ChestCTImageError("Chest CT slice must be a non-empty 2D image.")

        try:
            import tensorflow as tf

            resized = tf.image.resize(
                source[..., np.newaxis].astype(np.float32) / 255.0,
                (INPUT_SIZE, INPUT_SIZE),
            ).numpy()
        except Exception as exc:
            raise ChestCTImageError(
                "Could not normalize or resize the chest CT slice."
            ) from exc

        if resized.shape != (INPUT_SIZE, INPUT_SIZE, 1) or not np.isfinite(
            resized
        ).all():
            raise ChestCTImageError(
                "Chest CT image preprocessing returned invalid data."
            )
        return resized[np.newaxis, ...], source.shape

    @staticmethod
    def _create_overlay(image: np.ndarray, masks: np.ndarray) -> bytes:
        base = np.repeat(image[..., np.newaxis], 3, axis=-1).astype(np.float32)
        output = base.copy()
        colors = (
            (36, 165, 255),
            (255, 145, 38),
        )
        ground_glass = masks[..., 0]
        consolidation = masks[..., 1]
        output[ground_glass & ~consolidation] = np.asarray(colors[0], dtype=np.float32)
        output[consolidation & ~ground_glass] = np.asarray(colors[1], dtype=np.float32)
        output[ground_glass & consolidation] = np.array(
            [255.0, 235.0, 59.0],
            dtype=np.float32,
        )

        overlay = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(overlay)
        for class_index, color in enumerate(colors):
            rows, columns = np.nonzero(masks[..., class_index])
            if not len(rows):
                continue
            bounds = (
                int(columns.min()),
                int(rows.min()),
                int(columns.max()),
                int(rows.max()),
            )
            draw.rectangle(bounds, outline=(0, 0, 0), width=5)
            draw.rectangle(bounds, outline=color, width=2)

        encoded = BytesIO()
        overlay.save(encoded, format="PNG")
        return encoded.getvalue()

    def analyze_bytes(self, image_bytes: bytes) -> ChestCTSegmentation:
        model_input, source_shape = self._preprocess_image(image_bytes)
        model = self.load()

        try:
            output = np.asarray(model.predict(model_input, verbose=0), dtype=np.float32)
        except Exception as exc:
            raise ChestCTModelContractError(
                f"Chest CT model inference failed: {exc}"
            ) from exc

        expected_shape = (1, INPUT_SIZE, INPUT_SIZE, len(FINDING_NAMES))
        if (
            output.shape != expected_shape
            or not np.isfinite(output).all()
            or (output < 0).any()
            or (output > 1).any()
        ):
            raise ChestCTModelContractError(
                "Chest CT model returned invalid segmentation maps."
            )

        masks = output[0] >= MASK_THRESHOLD
        pixel_counts = {
            name: int(masks[..., index].sum())
            for index, name in enumerate(FINDING_NAMES)
        }
        overlay_png = self._create_overlay(
            (model_input[0, ..., 0] * 255.0).clip(0, 255),
            masks,
        )
        return ChestCTSegmentation(
            model_name=MODEL_NAME,
            pixel_counts=pixel_counts,
            image_shape=source_shape,
            threshold=MASK_THRESHOLD,
            overlay_png=overlay_png,
        )


__all__ = [
    "ChestCTError",
    "ChestCTImageError",
    "ChestCTModelContractError",
    "ChestCTModelUnavailableError",
    "ChestCTSegmentation",
    "ChestCTSegmentationService",
    "FINDING_NAMES",
    "MASK_THRESHOLD",
    "MODEL_NAME",
    "REPORTED_TEST_DICE",
]
