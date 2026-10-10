from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_ID = "sumit-ai-ml/POCUS_Aorta_segmentation"
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "abdominal_aorta_ultrasound_best.pt"
)
MODEL_SHA256 = "ea582b9cc377088c20b336cc6047ffb6805d77283281ef7fdcfa641079639251"
CLASS_NAME = "Aorta"
MASK_COLOR_RGB = (0, 255, 128)


class AbdominalAortaUltrasoundError(RuntimeError):
    pass


class AbdominalAortaUltrasoundImageError(ValueError):
    pass


class AbdominalAortaUltrasoundModelUnavailableError(
    AbdominalAortaUltrasoundError
):
    pass


class AbdominalAortaUltrasoundModelContractError(
    AbdominalAortaUltrasoundError
):
    pass


@dataclass(frozen=True)
class AbdominalAortaUltrasoundPrediction:
    overlay_png: bytes | None
    confidence_score: float | None
    mask_area_pixels: int
    image_shape: tuple[int, int]


def _load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise AbdominalAortaUltrasoundModelUnavailableError(
            f"Abdominal aorta model checkpoint was not found: {path.name}"
        )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != MODEL_SHA256:
        raise AbdominalAortaUltrasoundModelUnavailableError(
            f"Abdominal aorta checkpoint hash does not match the expected model: "
            f"{path.name}"
        )

    try:
        from ultralytics import YOLO

        model = YOLO(str(path), task="segment")
    except ImportError as exc:
        raise AbdominalAortaUltrasoundModelUnavailableError(
            "Abdominal aorta segmentation requires the ultralytics package."
        ) from exc
    except Exception as exc:
        raise AbdominalAortaUltrasoundModelUnavailableError(
            f"Abdominal aorta checkpoint could not be loaded: {path.name}"
        ) from exc

    names = getattr(model, "names", None)
    if not isinstance(names, dict) or tuple(names.values()) != (CLASS_NAME,):
        raise AbdominalAortaUltrasoundModelContractError(
            "Expected a single segmentation class named Aorta."
        )
    if getattr(model, "task", None) != "segment":
        raise AbdominalAortaUltrasoundModelContractError(
            "Expected the abdominal aorta checkpoint to be a segmentation model."
        )
    return model


class AbdominalAortaUltrasoundAnalysisService:
    def __init__(self, model_path: str | Path = MODEL_PATH, model_loader=None):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_model
        self._model = None

    @staticmethod
    def prepare_image(image_bytes: bytes) -> Image.Image:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise AbdominalAortaUltrasoundImageError(
                "Upload a non-empty abdominal ultrasound JPG or PNG image."
            )
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise AbdominalAortaUltrasoundImageError(
                        "Abdominal ultrasound uploads support JPG and PNG images only."
                    )
                image.load()
                decoded = image.convert("RGB")
        except AbdominalAortaUltrasoundImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise AbdominalAortaUltrasoundImageError(
                "The abdominal ultrasound image is corrupt or unreadable."
            ) from exc

        if decoded.width < 2 or decoded.height < 2:
            raise AbdominalAortaUltrasoundImageError(
                "The abdominal ultrasound image dimensions are invalid."
            )
        return decoded

    def load(self):
        if self._model is None:
            self._model = self.model_loader(self.model_path)
        return self._model

    def analyze_prepared_image(
        self,
        image: Image.Image,
    ) -> AbdominalAortaUltrasoundPrediction:
        model = self.load()
        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = image_rgb.shape[:2]

        try:
            results = model.predict(
                source=image_rgb,
                device="cpu",
                verbose=False,
                retina_masks=True,
            )
        except Exception as exc:
            raise AbdominalAortaUltrasoundModelUnavailableError(
                "The abdominal aorta segmentation model failed during inference."
            ) from exc

        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise AbdominalAortaUltrasoundModelContractError(
                "Expected one segmentation result for the uploaded image."
            )
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return AbdominalAortaUltrasoundPrediction(
                overlay_png=None,
                confidence_score=None,
                mask_area_pixels=0,
                image_shape=(height, width),
            )

        try:
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            masks = result.masks.data.detach().cpu().numpy()
        except (AttributeError, TypeError, ValueError) as exc:
            raise AbdominalAortaUltrasoundModelContractError(
                "The segmentation model returned an unsupported result structure."
            ) from exc

        if masks.ndim != 3 or len(masks) != len(class_ids) or len(masks) != len(confidences):
            raise AbdominalAortaUltrasoundModelContractError(
                "The segmentation masks, labels, and scores have inconsistent shapes."
            )
        if np.any(class_ids != 0):
            raise AbdominalAortaUltrasoundModelContractError(
                "The segmentation model returned an undocumented class."
            )
        if not np.isfinite(confidences).all() or np.any(confidences < 0) or np.any(confidences > 1):
            raise AbdominalAortaUltrasoundModelContractError(
                "The segmentation model returned an invalid score."
            )

        mask = np.any(masks[class_ids == 0] > 0.5, axis=0)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask_area_pixels = int(np.count_nonzero(mask))
        if mask_area_pixels == 0:
            return AbdominalAortaUltrasoundPrediction(
                overlay_png=None,
                confidence_score=None,
                mask_area_pixels=0,
                image_shape=(height, width),
            )

        overlay = image_rgb.copy()
        tint = np.asarray(MASK_COLOR_RGB, dtype=np.uint8)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.55
            + tint.astype(np.float32) * 0.45
        ).astype(np.uint8)
        mask_points = cv2.findNonZero(mask.astype(np.uint8))
        if mask_points is None:
            raise AbdominalAortaUltrasoundModelContractError(
                "The aorta mask does not contain a drawable region."
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
        output = BytesIO()
        Image.fromarray(overlay).save(output, format="PNG")
        return AbdominalAortaUltrasoundPrediction(
            overlay_png=output.getvalue(),
            confidence_score=float(np.max(confidences)),
            mask_area_pixels=mask_area_pixels,
            image_shape=(height, width),
        )

    def analyze_bytes(self, image_bytes: bytes) -> AbdominalAortaUltrasoundPrediction:
        return self.analyze_prepared_image(self.prepare_image(image_bytes))


__all__ = [
    "AbdominalAortaUltrasoundAnalysisService",
    "AbdominalAortaUltrasoundError",
    "AbdominalAortaUltrasoundImageError",
    "AbdominalAortaUltrasoundModelContractError",
    "AbdominalAortaUltrasoundModelUnavailableError",
    "AbdominalAortaUltrasoundPrediction",
    "CLASS_NAME",
    "MODEL_ID",
    "MODEL_PATH",
    "MODEL_SHA256",
]
