from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


KNEE_CLASS_NAMES = (
    "Healthy",
    "Partial ACL tear",
    "Complete ACL tear",
)
TARGET_ROI_SIZE = (75, 75)
LOCALIZER_INPUT_SIZE = (224, 224)
LOCALIZER_THRESHOLD = 0.5
MIN_ROI_COMPONENT_AREA = 8
MAX_ROI_COMPONENT_FRACTION = 0.8
MODEL_PATH = Path(
    os.getenv(
        "ACL_KNEE_MODEL_PATH",
        Path(__file__).resolve().parents[2] / "models" / "acl_resnet14_best.keras",
    )
)
LOCALIZER_MODEL_PATH = Path(
    os.getenv(
        "ACL_ROI_LOCALIZER_PATH",
        Path(__file__).resolve().parents[2] / "models" / "acl_roi_localizer.keras",
    )
)


class KneeMRIError(RuntimeError):
    pass


class KneeImageValidationError(ValueError):
    pass


class KneeModelUnavailableError(KneeMRIError):
    pass


class ACLROILocalizerUnavailableError(KneeModelUnavailableError):
    pass


class KneeModelContractError(KneeMRIError):
    pass


class ACLLocalizationUnavailableError(KneeMRIError):
    pass


class ACLROINotFoundError(KneeMRIError):
    pass


@dataclass(frozen=True)
class ACLROIBox:
    x: int
    y: int
    width: int
    height: int


class ACLROILocalizer(Protocol):
    def locate(self, grayscale_slice: np.ndarray) -> ACLROIBox | None:
        """Return ACL ROI coordinates in the supplied full-slice image."""


@dataclass(frozen=True)
class KneeSlicePrediction:
    slice_index: int
    class_index: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    localization_status: str
    roi_box: dict[str, int]


@lru_cache(maxsize=2)
def _load_keras_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise KneeModelUnavailableError(
            f"Required knee model file was not found: {path.name}"
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise KneeModelUnavailableError(
            "Knee ACL inference requires tensorflow-cpu==2.20.0."
        ) from exc

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise KneeModelUnavailableError(
            f"Knee model could not be loaded: {path.name}"
        ) from exc


class KerasACLROILocalizer:
    def __init__(
        self,
        model_path: str | Path = LOCALIZER_MODEL_PATH,
        model_loader=None,
    ):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_keras_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                self._model = self.model_loader(self.model_path)
            except KneeModelUnavailableError as exc:
                raise ACLROILocalizerUnavailableError(str(exc)) from exc
            input_shape = tuple(self._model.input_shape)
            output_shape = tuple(self._model.output_shape)
            if input_shape != (None, 224, 224, 1):
                raise KneeModelContractError(
                    "Expected ACL localizer input (None, 224, 224, 1), "
                    f"received {input_shape}."
                )
            if output_shape != (None, 224, 224, 1):
                raise KneeModelContractError(
                    "Expected ACL localizer output (None, 224, 224, 1), "
                    f"received {output_shape}."
                )
        return self._model

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise KneeImageValidationError(
                "ACL localization requires a grayscale or RGB MRI slice."
            )
        if grayscale.dtype != np.uint8:
            if not np.isfinite(grayscale).all():
                raise KneeImageValidationError("MRI slice contains invalid pixel values.")
            grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
        return grayscale

    @staticmethod
    def mask_to_box(
        probabilities: np.ndarray,
        original_shape: tuple[int, ...],
        threshold: float = LOCALIZER_THRESHOLD,
        min_component_area: int = MIN_ROI_COMPONENT_AREA,
        max_component_fraction: float = MAX_ROI_COMPONENT_FRACTION,
    ) -> ACLROIBox:
        mask = np.asarray(probabilities)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != LOCALIZER_INPUT_SIZE[::-1]:
            raise ACLROINotFoundError(
                f"ACL localizer returned an invalid mask shape: {mask.shape}."
            )
        if not np.isfinite(mask).all():
            raise ACLROINotFoundError("ACL localizer returned a non-finite mask.")

        binary_mask = (mask >= threshold).astype(np.uint8)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
        total_pixels = binary_mask.size
        max_area = int(total_pixels * max_component_fraction)
        valid_components = []
        for component_index in range(1, component_count):
            x, y, width, height, area = stats[component_index]
            if (
                area >= min_component_area
                and area <= max_area
                and width > 1
                and height > 1
            ):
                valid_components.append((int(area), int(x), int(y), int(width), int(height)))
        if not valid_components:
            raise ACLROINotFoundError(
                "ACL localization failed: no valid ROI component was found."
            )

        _, x, y, width, height = max(valid_components)
        original_height, original_width = original_shape[:2]
        scale_x = original_width / LOCALIZER_INPUT_SIZE[0]
        scale_y = original_height / LOCALIZER_INPUT_SIZE[1]
        left = max(0, min(original_width, floor(x * scale_x)))
        top = max(0, min(original_height, floor(y * scale_y)))
        right = max(0, min(original_width, ceil((x + width) * scale_x)))
        bottom = max(0, min(original_height, ceil((y + height) * scale_y)))
        if right <= left or bottom <= top:
            raise ACLROINotFoundError(
                "ACL localization failed: mapped ROI is empty."
            )
        return ACLROIBox(left, top, right - left, bottom - top)

    def predict_mask(self, original_image: np.ndarray) -> np.ndarray:
        grayscale = self._to_grayscale(original_image)
        resized = cv2.resize(
            grayscale,
            LOCALIZER_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = resized.astype(np.float32) / 255.0
        batch = normalized[np.newaxis, ..., np.newaxis]
        model = self._get_model()
        try:
            prediction = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        except Exception as exc:
            raise ACLROILocalizerUnavailableError(
                "ACL ROI localizer failed during inference."
            ) from exc
        if prediction.shape != (1, 224, 224, 1):
            raise KneeModelContractError(
                "Expected ACL localizer prediction (1, 224, 224, 1), "
                f"received {prediction.shape}."
            )
        if np.any(prediction < 0) or np.any(prediction > 1):
            raise KneeModelContractError(
                "ACL localizer output must be a sigmoid mask in [0, 1]."
            )
        return prediction[0, ..., 0]

    def locate(self, original_image: np.ndarray) -> ACLROIBox:
        probabilities = self.predict_mask(original_image)
        return self.mask_to_box(probabilities, original_image.shape)


class KneeMRIPredictionService:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        roi_localizer: ACLROILocalizer | None = None,
        model_loader=None,
        localizer_path: str | Path = LOCALIZER_MODEL_PATH,
    ):
        self.model_path = str(model_path)
        self.roi_localizer = (
            roi_localizer
            if roi_localizer is not None
            else KerasACLROILocalizer(localizer_path)
        )
        self.model_loader = model_loader or _load_keras_model
        self._model = None

    @staticmethod
    def decode_slices(image_bytes: Sequence[bytes]) -> list[np.ndarray]:
        if isinstance(image_bytes, (bytes, bytearray)):
            image_bytes = [image_bytes]
        if not image_bytes:
            raise KneeImageValidationError("Upload at least one knee MRI image.")

        grayscale_slices = []
        for index, content in enumerate(image_bytes):
            if not isinstance(content, (bytes, bytearray)) or not content:
                raise KneeImageValidationError(
                    f"Knee MRI slice {index + 1} is empty. Upload a valid JPG or PNG image."
                )
            if not (
                content.startswith(b"\xff\xd8\xff")
                or content.startswith(b"\x89PNG\r\n\x1a\n")
            ):
                raise KneeImageValidationError(
                    f"Knee MRI slice {index + 1} must be a valid JPG or PNG image."
                )
            try:
                with Image.open(BytesIO(content)) as image:
                    if image.format not in {"JPEG", "PNG"}:
                        raise KneeImageValidationError(
                            f"Knee MRI slice {index + 1} must be a JPG or PNG image."
                        )
                    image.load()
                    rgb = image.convert("RGB")
                    array = np.asarray(rgb, dtype=np.uint8)
            except KneeImageValidationError:
                raise
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise KneeImageValidationError(
                    f"Knee MRI slice {index + 1} is corrupt or unreadable."
                ) from exc

            if array.ndim not in {2, 3} or not array.size:
                raise KneeImageValidationError(
                    f"Knee MRI slice {index + 1} has invalid image dimensions."
                )
            grayscale_slices.append(array)
        return grayscale_slices

    @staticmethod
    def preprocess_roi(grayscale_roi: np.ndarray) -> np.ndarray:
        roi = np.asarray(grayscale_roi)
        if roi.ndim != 2 or not roi.size:
            raise ACLROINotFoundError("The ACL localizer returned an empty ROI.")
        if roi.dtype != np.uint8:
            if not np.isfinite(roi).all():
                raise ACLROINotFoundError("The ACL localizer returned invalid ROI pixels.")
            roi = np.clip(roi, 0, 255).astype(np.uint8)

        if (roi.shape[1], roi.shape[0]) != TARGET_ROI_SIZE:
            roi = cv2.resize(roi, TARGET_ROI_SIZE, interpolation=cv2.INTER_LINEAR)

        # Training used float32 pixel values without intensity normalization.
        return roi.astype(np.float32)[..., np.newaxis]

    def _get_model(self):
        if self._model is None:
            self._model = self.model_loader(self.model_path)
            input_shape = tuple(self._model.input_shape)
            output_shape = tuple(self._model.output_shape)
            if input_shape != (None, 75, 75, 1):
                raise KneeModelContractError(
                    f"Expected classifier input (None, 75, 75, 1), received {input_shape}."
                )
            if output_shape != (None, 3):
                raise KneeModelContractError(
                    f"Expected classifier output (None, 3), received {output_shape}."
                )
        return self._model

    @staticmethod
    def _validate_roi_box(box: ACLROIBox, image: np.ndarray, slice_index: int):
        if not isinstance(box, ACLROIBox):
            raise ACLROINotFoundError(
                f"ACL localization failed for slice {slice_index + 1}; no valid ROI was returned."
            )
        if box.x < 0 or box.y < 0 or box.width <= 0 or box.height <= 0:
            raise ACLROINotFoundError(
                f"ACL localization returned invalid ROI coordinates for slice {slice_index + 1}."
            )
        if box.x + box.width > image.shape[1] or box.y + box.height > image.shape[0]:
            raise ACLROINotFoundError(
                f"ACL localization returned an ROI outside slice {slice_index + 1}."
            )
        return image[box.y : box.y + box.height, box.x : box.x + box.width]

    def predict_slices(self, image_bytes: Sequence[bytes]) -> list[KneeSlicePrediction]:
        image_slices = self.decode_slices(image_bytes)
        return self.predict_image_slices(image_slices)

    def predict_grayscale_slices(
        self,
        grayscale_slices: Sequence[np.ndarray],
    ) -> list[KneeSlicePrediction]:
        return self.predict_image_slices(grayscale_slices)

    def predict_image_slices(
        self,
        image_slices: Sequence[np.ndarray],
    ) -> list[KneeSlicePrediction]:
        if not image_slices:
            raise KneeImageValidationError("Upload at least one knee MRI image.")

        roi_batches = []
        roi_boxes = []
        localization_statuses = []
        for index, original_image in enumerate(image_slices):
            original_image = np.asarray(original_image)
            if original_image.ndim not in {2, 3} or not original_image.size:
                raise KneeImageValidationError(
                    f"Knee MRI slice {index + 1} has invalid image dimensions."
                )
            box = self.roi_localizer.locate(original_image)
            roi_original = self._validate_roi_box(box, original_image, index)
            if roi_original.ndim == 3:
                roi_grayscale = cv2.cvtColor(roi_original, cv2.COLOR_RGB2GRAY)
            else:
                roi_grayscale = roi_original
            roi_batches.append(self.preprocess_roi(roi_grayscale))
            roi_boxes.append(box)
            localization_statuses.append("localized")

        batch = np.stack(roi_batches).astype(np.float32, copy=False)
        model = self._get_model()
        try:
            probabilities = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        except Exception as exc:
            raise KneeModelUnavailableError(
                "The ACL ResNet-14 failed during inference."
            ) from exc

        if probabilities.shape != (len(image_slices), 3):
            raise KneeModelContractError(
                "Expected one three-class prediction per slice; "
                f"received {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise KneeModelContractError("The ACL model returned non-finite probabilities.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise KneeModelContractError("The ACL model returned values outside [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
            raise KneeModelContractError("The ACL model output is not a probability distribution.")

        results = []
        for index, (row, box, localization_status) in enumerate(
            zip(probabilities, roi_boxes, localization_statuses)
        ):
            class_index = int(np.argmax(row))
            results.append(
                KneeSlicePrediction(
                    slice_index=index,
                    class_index=class_index,
                    class_name=KNEE_CLASS_NAMES[class_index],
                    confidence=float(row[class_index]),
                    probabilities={
                        class_name: float(row[class_id])
                        for class_id, class_name in enumerate(KNEE_CLASS_NAMES)
                    },
                    localization_status=localization_status,
                    roi_box={
                        "x": box.x,
                        "y": box.y,
                        "width": box.width,
                        "height": box.height,
                    },
                )
            )
        return results


__all__ = [
    "ACLLocalizationUnavailableError",
    "ACLROILocalizerUnavailableError",
    "ACLROIBox",
    "ACLROILocalizer",
    "ACLROINotFoundError",
    "KNEE_CLASS_NAMES",
    "KerasACLROILocalizer",
    "LOCALIZER_MODEL_PATH",
    "KneeImageValidationError",
    "KneeModelContractError",
    "KneeModelUnavailableError",
    "KneeMRIPredictionService",
    "KneeSlicePrediction",
    "MODEL_PATH",
]