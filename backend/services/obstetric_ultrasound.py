from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_ID = (
    "Beijuka/ultrasound_plane_classification-swin-all-planes-class-weight-v1"
)
MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "obstetric_ultrasound_swin"
)
WEIGHTS_PATH = MODEL_DIRECTORY / "model.safetensors"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = (
    "Placenta",
    "Fetal brain",
    "Fetal femur",
    "Maternal Cervix",
    "Fetal thorax",
    "Fetal abdomen",
    "Other",
    "Fetal spine",
    "Fetal heart rate",
)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class ObstetricUltrasoundError(RuntimeError):
    pass


class ObstetricUltrasoundImageError(ValueError):
    pass


class ObstetricUltrasoundModelUnavailableError(ObstetricUltrasoundError):
    pass


class ObstetricUltrasoundModelContractError(ObstetricUltrasoundError):
    pass


@dataclass(frozen=True)
class ObstetricUltrasoundPrediction:
    predicted_class: str
    model_score: float
    class_scores: dict[str, float]
    input_shape: tuple[int, ...]


def _load_model(model_directory: str):
    if not (Path(model_directory) / "model.safetensors").is_file():
        raise ObstetricUltrasoundModelUnavailableError(
            "Fetal ultrasound model weights are missing. Download the model "
            f"checkpoint to {Path(model_directory) / 'model.safetensors'}."
        )

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise ObstetricUltrasoundModelUnavailableError(
            "Fetal ultrasound inference requires torch, transformers, and safetensors."
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_directory,
            local_files_only=True,
            use_safetensors=True,
        )
    except Exception as exc:
        raise ObstetricUltrasoundModelUnavailableError(
            "The fetal ultrasound model or image processor could not be loaded "
            f"from {model_directory}."
        ) from exc
    model.to("cpu")
    model.eval()
    return torch, processor, model


class ObstetricUltrasoundAnalysisService:
    def __init__(self, model_directory: str | Path = MODEL_DIRECTORY, model_loader=None):
        self.model_directory = str(model_directory)
        self.model_loader = model_loader or _load_model
        self._torch = None
        self._processor = None
        self._model = None

    @staticmethod
    def prepare_image(image_bytes: bytes) -> Image.Image:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ObstetricUltrasoundImageError(
                "Upload a non-empty fetal ultrasound JPG or PNG image."
            )
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ObstetricUltrasoundImageError(
                        "Obstetric ultrasound uploads support JPG and PNG images only."
                    )
                image.load()
                return image.convert("RGB")
        except ObstetricUltrasoundImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ObstetricUltrasoundImageError(
                "The fetal ultrasound image is corrupt or unreadable."
            ) from exc

    def load(self):
        if self._model is None:
            self._torch, self._processor, self._model = self.model_loader(
                self.model_directory
            )
            config = getattr(self._model, "config", None)
            id2label = getattr(config, "id2label", None)
            if not isinstance(id2label, dict):
                raise ObstetricUltrasoundModelContractError(
                    "The fetal ultrasound model is missing its class labels."
                )
            try:
                labels = tuple(
                    str(id2label.get(index, id2label.get(str(index))))
                    for index in range(len(CLASS_NAMES))
                )
            except (TypeError, ValueError) as exc:
                raise ObstetricUltrasoundModelContractError(
                    "The fetal ultrasound model class labels are invalid."
                ) from exc
            if labels != CLASS_NAMES:
                raise ObstetricUltrasoundModelContractError(
                    "The fetal ultrasound model labels do not match the documented "
                    "nine fetal ultrasound view classes."
                )
        return self._model

    def analyze_prepared_image(
        self,
        image: Image.Image,
    ) -> ObstetricUltrasoundPrediction:
        self.load()
        try:
            processed = self._processor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            if tuple(pixel_values.shape) != (1, 3, *IMAGE_SIZE):
                raise ObstetricUltrasoundModelContractError(
                    "The fetal ultrasound processor must return one RGB image "
                    "shaped (1, 3, 224, 224)."
                )
            if not self._torch.isfinite(pixel_values).all():
                raise ObstetricUltrasoundImageError(
                    "The fetal ultrasound image contains invalid pixel values."
                )
            with self._torch.inference_mode():
                output = self._model(pixel_values.to(device="cpu", dtype=self._torch.float32))
                logits = output.logits
                probabilities = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()
        except (ObstetricUltrasoundImageError, ObstetricUltrasoundModelContractError):
            raise
        except Exception as exc:
            raise ObstetricUltrasoundModelUnavailableError(
                "The fetal ultrasound model failed during image analysis."
            ) from exc

        if probabilities.shape != (len(CLASS_NAMES),):
            raise ObstetricUltrasoundModelContractError(
                f"Expected nine fetal ultrasound view scores, received "
                f"{probabilities.shape}."
            )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise ObstetricUltrasoundModelContractError(
                "The fetal ultrasound model returned invalid class scores."
            )

        class_index = int(np.argmax(probabilities))
        return ObstetricUltrasoundPrediction(
            predicted_class=CLASS_NAMES[class_index],
            model_score=float(probabilities[class_index]),
            class_scores={
                name: float(probabilities[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            input_shape=tuple(pixel_values.shape),
        )

    def analyze_bytes(self, image_bytes: bytes) -> ObstetricUltrasoundPrediction:
        return self.analyze_prepared_image(self.prepare_image(image_bytes))


__all__ = [
    "CLASS_NAMES",
    "IMAGE_SIZE",
    "MODEL_DIRECTORY",
    "MODEL_ID",
    "ObstetricUltrasoundAnalysisService",
    "ObstetricUltrasoundError",
    "ObstetricUltrasoundImageError",
    "ObstetricUltrasoundModelContractError",
    "ObstetricUltrasoundModelUnavailableError",
    "ObstetricUltrasoundPrediction",
    "WEIGHTS_PATH",
]
