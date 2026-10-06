from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


MODEL_ID = "thrive-2025/EchoView47"
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "echoview47"
    / "EchoView47.keras"
)
MODEL_SHA256 = "5946e975664ace4de8b11385bb8d73bf4df9df2b9c1adfe51e243680cb673fe8"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = (
    "a2ch-full",
    "a2ch-la",
    "a2ch-lv",
    "a3ch-full",
    "a3ch-la",
    "a3ch-lv",
    "a3ch-outflow",
    "a4ch-full",
    "a4ch-ias",
    "a4ch-la",
    "a4ch-lv",
    "a4ch-ra",
    "a4ch-rv",
    "a5ch-full",
    "a5ch-outflow",
    "apex",
    "doppler-ao-descending",
    "doppler-av",
    "doppler-mv",
    "doppler-pv",
    "doppler-tissue-lateral",
    "doppler-tissue-rv",
    "doppler-tissue-septal",
    "doppler-tv",
    "mmode-a4ch-rv",
    "mmode-ivc",
    "mmode-plax-av",
    "mmode-plax-lv",
    "mmode-plax-mitral",
    "plax-full-la",
    "plax-full-lv",
    "plax-full-mv",
    "plax-full-out",
    "plax-full-rv-ao",
    "plax-tv",
    "plax-valves-av",
    "plax-valves-mv",
    "psax-all",
    "psax-av",
    "psax-lv-apex",
    "psax-lv-base",
    "psax-lv-mid",
    "psax-pv",
    "psax-tv",
    "subcostal-heart",
    "subcostal-ivc",
    "suprasternal",
)

_APICAL_FOCI = {
    "ias": "the wall between the left and right atria",
    "la": "the left atrium",
    "lv": "the left ventricle",
    "ra": "the right atrium",
    "rv": "the right ventricle",
    "outflow": "the left-ventricle outflow tract",
}
_DOPPLER_TARGETS = {
    "ao-descending": "blood-flow Doppler in the descending aorta",
    "av": "blood-flow Doppler across the aortic valve",
    "mv": "blood-flow Doppler across the mitral valve",
    "pv": "blood-flow Doppler across the pulmonary valve",
    "tv": "blood-flow Doppler across the tricuspid valve",
    "tissue-lateral": "tissue-motion Doppler at the lateral heart wall",
    "tissue-rv": "tissue-motion Doppler focused on the right ventricle",
    "tissue-septal": "tissue-motion Doppler at the wall between the ventricles",
}
_PLAX_FOCI = {
    "full-la": "the left atrium",
    "full-lv": "the left ventricle",
    "full-mv": "the mitral valve",
    "full-out": "the left-ventricle outflow tract",
    "full-rv-ao": "the right ventricle and aortic root",
    "tv": "the tricuspid valve",
    "valves-av": "the aortic valve",
    "valves-mv": "the mitral valve",
}
_PSAX_FOCI = {
    "all": "multiple short-axis levels",
    "av": "the aortic-valve level",
    "lv-apex": "the left ventricle near its tip",
    "lv-base": "the left ventricle near its base",
    "lv-mid": "the middle of the left ventricle",
    "pv": "the pulmonary-valve level",
    "tv": "the tricuspid-valve level",
}


class EchoView47Error(RuntimeError):
    pass


class EchoView47ImageError(ValueError):
    pass


class EchoView47ModelUnavailableError(EchoView47Error):
    pass


class EchoView47ModelContractError(EchoView47Error):
    pass


@dataclass(frozen=True)
class EchoView47Prediction:
    predicted_class: str
    model_score: float
    class_scores: dict[str, float]
    input_shape: tuple[int, ...]

    @property
    def display_label(self) -> str:
        return describe_view_class(self.predicted_class)[0]

    @property
    def meaning(self) -> str:
        return describe_view_class(self.predicted_class)[1]


def describe_view_class(class_name: str) -> tuple[str, str]:
    if class_name not in CLASS_NAMES:
        raise ValueError(f"Unsupported EchoView47 class: {class_name}")

    if class_name.startswith(("a2ch-", "a3ch-", "a4ch-", "a5ch-")):
        view_name, focus = class_name.split("-", maxsplit=1)
        chamber_count = view_name[1]
        view_label = f"Apical {chamber_count}-chamber view"
        if focus == "full":
            full_view_meanings = {
                "2": "a two-chamber plane, usually showing the left atrium and left ventricle",
                "3": "a three-chamber plane showing the left-heart chambers and outflow view",
                "4": "all four heart chambers together",
                "5": "a five-chamber plane that also includes the outflow tract",
            }
            return (
                view_label,
                f"The ultrasound probe looks up from the tip (apex) of the heart. "
                f"This view shows {full_view_meanings[chamber_count]}. It describes "
                "the image view, not a heart condition.",
            )
        focus_description = _APICAL_FOCI[focus]
        return (
            f"{view_label} — {focus_description}",
            f"An ultrasound view taken from the tip of the heart, showing a "
            f"{chamber_count}-chamber plane focused on {focus_description}. "
            "This describes the camera/view position, not a heart condition.",
        )

    if class_name.startswith("doppler-"):
        target = _DOPPLER_TARGETS[class_name.removeprefix("doppler-")]
        return (
            target.capitalize(),
            f"This is a Doppler-style view used to display {target}. "
            "It does not determine whether blood flow or a valve is normal.",
        )

    if class_name.startswith("mmode-"):
        target_name = class_name.removeprefix("mmode-")
        targets = {
            "a4ch-rv": "right-ventricle motion in an apical four-chamber view",
            "ivc": "inferior vena cava motion over time",
            "plax-av": "aortic-valve motion in a parasternal long-axis view",
            "plax-lv": "left-ventricle motion in a parasternal long-axis view",
            "plax-mitral": "mitral-valve motion in a parasternal long-axis view",
        }
        target = targets[target_name]
        return (
            f"M-mode view — {target}",
            f"This M-mode view displays motion along a selected line over time; this class "
            f"focuses on {target}. It is a view label, not a measurement or diagnosis.",
        )

    if class_name.startswith("plax-"):
        focus = _PLAX_FOCI[class_name.removeprefix("plax-")]
        return (
            f"Parasternal long-axis view — {focus}",
            f"An ultrasound view taken from the chest beside the breastbone, "
            f"showing the heart along its long axis with focus on {focus}. "
            "This describes the view, not a heart condition.",
        )

    if class_name.startswith("psax-"):
        focus = _PSAX_FOCI[class_name.removeprefix("psax-")]
        return (
            f"Parasternal short-axis view — {focus}",
            f"An ultrasound view taken from the chest beside the breastbone, "
            f"showing a cross-section of the heart focused on {focus}. "
            "This describes the view, not a heart condition.",
        )

    if class_name == "subcostal-heart":
        return (
            "Subcostal heart view",
            "An ultrasound view of the heart taken from below the rib cage. "
            "This describes the view, not a heart condition.",
        )
    if class_name == "subcostal-ivc":
        return (
            "Subcostal inferior vena cava view",
            "An ultrasound view of the large vein returning blood to the heart, "
            "taken from below the rib cage. This is a view label, not an assessment.",
        )
    if class_name == "suprasternal":
        return (
            "Suprasternal view",
            "An ultrasound view taken from the hollow above the breastbone, "
            "often used to image the aortic arch. This describes the view only.",
        )
    if class_name == "apex":
        return (
            "Apical view — specific chamber view not specified",
            "This image view appears to be taken from the tip of the heart, but this "
            "class does not specify a particular chamber plane.",
        )

    raise EchoView47ModelContractError(
        f"No plain-language explanation is defined for EchoView47 class {class_name}."
    )


def _load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.is_file():
        raise EchoView47ModelUnavailableError(
            f"EchoView47 checkpoint was not found: {path.name}"
        )

    digest = hashlib.sha256()
    try:
        with path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EchoView47ModelUnavailableError(
            f"EchoView47 checkpoint could not be read: {path.name}"
        ) from exc
    if digest.hexdigest() != MODEL_SHA256:
        raise EchoView47ModelUnavailableError(
            f"EchoView47 checkpoint hash does not match the expected model: "
            f"{path.name}"
        )

    try:
        from tensorflow import keras

        model = keras.models.load_model(
            path,
            compile=False,
            safe_mode=True,
        )
    except ImportError as exc:
        raise EchoView47ModelUnavailableError(
            "EchoView47 inference requires tensorflow-cpu."
        ) from exc
    except Exception as exc:
        raise EchoView47ModelUnavailableError(
            f"EchoView47 checkpoint could not be loaded: {path.name}"
        ) from exc
    return model


class EchoView47AnalysisService:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        model_loader=None,
    ):
        self.model_path = str(model_path)
        self.model_loader = model_loader or _load_model
        self._model = None

    @staticmethod
    def prepare_image(image_bytes: bytes) -> Image.Image:
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise EchoView47ImageError(
                "Upload a non-empty echocardiogram JPG or PNG image."
            )
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise EchoView47ImageError(
                        "Echocardiogram view classification supports JPG and PNG only."
                    )
                image.load()
                decoded = image.convert("RGB")
        except EchoView47ImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise EchoView47ImageError(
                "The echocardiogram image is corrupt or unreadable."
            ) from exc

        if decoded.width < 2 or decoded.height < 2:
            raise EchoView47ImageError(
                "The echocardiogram image dimensions are invalid."
            )
        return decoded

    def load(self):
        if self._model is None:
            model = self.model_loader(self.model_path)
            if tuple(model.input_shape) != (None, *IMAGE_SIZE, 3):
                raise EchoView47ModelContractError(
                    "EchoView47 must accept one RGB image shaped (1, 224, 224, 3)."
                )
            if tuple(model.output_shape) != (None, len(CLASS_NAMES)):
                raise EchoView47ModelContractError(
                    "EchoView47 must return scores for its documented 47 view classes."
                )
            self._model = model
        return self._model

    def analyze_prepared_image(
        self,
        image: Image.Image,
    ) -> EchoView47Prediction:
        model = self.load()
        resized = image.convert("RGB").resize(
            IMAGE_SIZE,
            resample=Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(resized, dtype=np.float32)[None, ...]
        if pixels.shape != (1, *IMAGE_SIZE, 3) or not np.isfinite(pixels).all():
            raise EchoView47ImageError(
                "The echocardiogram image could not be prepared as a valid RGB image."
            )

        try:
            outputs = np.asarray(model.predict(pixels, verbose=0), dtype=np.float32)
        except Exception as exc:
            raise EchoView47ModelUnavailableError(
                "EchoView47 failed while classifying the image view."
            ) from exc

        if outputs.shape != (1, len(CLASS_NAMES)):
            raise EchoView47ModelContractError(
                f"EchoView47 must return one set of 47 scores; received {outputs.shape}."
            )
        scores = outputs[0]
        if (
            not np.isfinite(scores).all()
            or np.any(scores < 0)
            or np.any(scores > 1)
            or not np.isclose(float(scores.sum()), 1.0, atol=1e-3)
        ):
            raise EchoView47ModelContractError(
                "EchoView47 returned invalid 47-class softmax scores."
            )

        class_index = int(np.argmax(scores))
        class_name = CLASS_NAMES[class_index]
        try:
            describe_view_class(class_name)
        except ValueError as exc:
            raise EchoView47ModelContractError(
                "EchoView47 returned an unknown class label."
            ) from exc
        return EchoView47Prediction(
            predicted_class=class_name,
            model_score=float(scores[class_index]),
            class_scores={
                name: float(scores[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            input_shape=tuple(pixels.shape),
        )

    def analyze_bytes(self, image_bytes: bytes) -> EchoView47Prediction:
        return self.analyze_prepared_image(self.prepare_image(image_bytes))


__all__ = [
    "CLASS_NAMES",
    "EchoView47AnalysisService",
    "EchoView47Error",
    "EchoView47ImageError",
    "EchoView47ModelContractError",
    "EchoView47ModelUnavailableError",
    "EchoView47Prediction",
    "IMAGE_SIZE",
    "MODEL_ID",
    "MODEL_PATH",
    "MODEL_SHA256",
    "describe_view_class",
]
