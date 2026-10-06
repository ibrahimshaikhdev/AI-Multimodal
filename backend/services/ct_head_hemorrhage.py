from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
from nibabel.filebasedimages import ImageFileError

from backend.services.ct_head_inference import (
    CLASS_NAMES,
    HemorrhageSliceModel,
    read_dicom_series,
    read_nifti_volume,
)


MODEL_ID = "ianpan/ct-head-hemorrhage-detection"
MODEL_REVISION = "e6d89c253fdcc2a597dd0fd0f5771821f4947b37"
MAX_CT_ARCHIVE_MEMBERS = 4096
MAX_CT_ARCHIVE_UNPACKED_BYTES = 1024 * 1024 * 1024
MAX_CT_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024


class CTHeadError(RuntimeError):
    pass


class CTHeadInputError(ValueError):
    pass


class CTHeadModelUnavailableError(CTHeadError):
    pass


class CTHeadModelContractError(CTHeadError):
    pass


@dataclass(frozen=True)
class CTHeadHemorrhageAnalysis:
    model_name: str
    series_classification: dict[str, float]
    slice_classification: list[dict[str, float]]
    slice_count: int
    highest_any_slice_index: int
    localization_png: bytes | None
    input_format: str


@lru_cache(maxsize=1)
def _load_model() -> HemorrhageSliceModel:
    try:
        return HemorrhageSliceModel(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
    except Exception as exc:
        raise CTHeadModelUnavailableError(
            f"Could not load the local head CT hemorrhage model: {exc}"
        ) from exc


class CTHeadHemorrhageService:
    def __init__(self, model_loader=None):
        self.model_loader = model_loader or _load_model
        self._model: Any | None = None

    def load(self):
        if self._model is None:
            self._model = self.model_loader()
        return self._model

    @staticmethod
    def _safe_extract_zip(archive: ZipFile, destination: Path) -> None:
        members = archive.infolist()
        if not members or len(members) > MAX_CT_ARCHIVE_MEMBERS:
            raise CTHeadInputError(
                f"DICOM archive must contain 1 to {MAX_CT_ARCHIVE_MEMBERS} files."
            )

        unpacked_size = 0
        root = destination.resolve()
        for member in members:
            if member.is_dir():
                continue
            path = Path(member.filename)
            if (
                path.is_absolute()
                or any(part in {"..", ""} for part in path.parts)
                or (path.parts and ":" in path.parts[0])
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise CTHeadInputError("DICOM archive contains an unsafe path.")
            if member.file_size > MAX_CT_ARCHIVE_MEMBER_BYTES:
                raise CTHeadInputError("DICOM archive contains an oversized file.")
            unpacked_size += member.file_size
            if unpacked_size > MAX_CT_ARCHIVE_UNPACKED_BYTES:
                raise CTHeadInputError("DICOM archive exceeds the extraction limit.")
            if member.compress_size and member.file_size / member.compress_size > 200:
                raise CTHeadInputError("DICOM archive has an unsafe compression ratio.")

            target = (root / path).resolve()
            if root not in target.parents:
                raise CTHeadInputError("DICOM archive contains an unsafe path.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)

    @staticmethod
    def _localization_png(image: np.ndarray, mask: np.ndarray) -> bytes:
        import cv2

        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise CTHeadModelContractError(
                "The model returned an invalid hemorrhage localization map."
            )
        from backend.services.ct_head_inference import window_hu_slice

        brain_window = (
            window_hu_slice(image)[0] * 255
        ).clip(0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(
            (mask.clip(0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        base = cv2.cvtColor(brain_window, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.55, heatmap, 0.45, 0)
        success, encoded = cv2.imencode(".png", overlay)
        if not success:
            raise CTHeadModelContractError("Could not encode the CT localization map.")
        return encoded.tobytes()

    def analyze_bytes(self, file_bytes: bytes, filename: str) -> CTHeadHemorrhageAnalysis:
        if not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes:
            raise CTHeadInputError("Upload a non-empty DICOM ZIP or 3D NIfTI CT volume.")
        if not isinstance(filename, str) or not filename:
            raise CTHeadInputError("A CT volume filename is required.")

        suffix = filename.lower()
        try:
            with TemporaryDirectory(prefix="ct-head-") as temporary:
                temp_dir = Path(temporary)
                if suffix.endswith(".zip"):
                    try:
                        with ZipFile(BytesIO(file_bytes)) as archive:
                            self._safe_extract_zip(archive, temp_dir)
                    except BadZipFile as exc:
                        raise CTHeadInputError("The uploaded DICOM ZIP is invalid.") from exc
                    try:
                        series = read_dicom_series(temp_dir)
                    except (
                        FileNotFoundError,
                        ValueError,
                        OSError,
                        IndexError,
                        KeyError,
                        RuntimeError,
                        NotImplementedError,
                    ) as exc:
                        raise CTHeadInputError(
                            f"Could not read a supported single-frame head CT DICOM series: {exc}"
                        ) from exc
                    volume = np.asarray(series["image"], dtype=np.float32)
                    input_format = "DICOM series"
                elif suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
                    path = temp_dir / Path(filename).name
                    path.write_bytes(file_bytes)
                    try:
                        volume = read_nifti_volume(path)
                    except (ImageFileError, ValueError, OSError, EOFError) as exc:
                        raise CTHeadInputError(
                            f"Could not read a supported 3D NIfTI CT volume: {exc}"
                        ) from exc
                    input_format = "NIfTI volume"
                else:
                    raise CTHeadInputError(
                        "Head CT analysis supports a zipped DICOM series or .nii/.nii.gz volume; JPG/PNG images are not CT volumes."
                    )

                if volume.ndim != 3 or len(volume) == 0 or not np.isfinite(volume).all():
                    raise CTHeadInputError(
                        f"Expected a non-empty finite CT volume (D, H, W), got {volume.shape}."
                    )

                try:
                    model = self.load()
                    result = model.predict_hu_volume(
                        volume,
                        batch_size=1,
                        return_segmentation=False,
                    )
                except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                    raise CTHeadModelContractError(
                        f"Head CT model inference failed: {exc}"
                    ) from exc

                series_scores = np.asarray(result["series_classification"])
                slice_scores = np.asarray(result["slice_classification"])
                if (
                    series_scores.shape != (len(CLASS_NAMES),)
                    or slice_scores.shape != (len(volume), len(CLASS_NAMES))
                    or not np.isfinite(series_scores).all()
                    or not np.isfinite(slice_scores).all()
                    or (series_scores < 0).any()
                    or (series_scores > 1).any()
                    or (slice_scores < 0).any()
                    or (slice_scores > 1).any()
                ):
                    raise CTHeadModelContractError(
                        "Head CT model returned invalid series or slice probabilities."
                    )

                any_scores = slice_scores[:, CLASS_NAMES.index("any")]
                top_slice = int(any_scores.argmax())
                try:
                    from backend.services.ct_head_inference import (
                        preprocess_hu_volume_slab,
                    )

                    localization = model.predict(
                        preprocess_hu_volume_slab(volume, top_slice)
                    )
                    segmentation = np.asarray(localization["segmentation"])
                    if segmentation.shape != (len(CLASS_NAMES), 512, 512):
                        raise CTHeadModelContractError(
                            "Head CT model returned an unexpected localization shape."
                        )
                    localization_png = self._localization_png(
                        volume[top_slice],
                        segmentation[CLASS_NAMES.index("any")],
                    )
                except (KeyError, ValueError, RuntimeError) as exc:
                    raise CTHeadModelContractError(
                        f"Head CT localization failed: {exc}"
                    ) from exc

                return CTHeadHemorrhageAnalysis(
                    model_name=MODEL_ID,
                    series_classification={
                        name: float(score)
                        for name, score in zip(CLASS_NAMES, series_scores)
                    },
                    slice_classification=[
                        {
                            name: float(score)
                            for name, score in zip(CLASS_NAMES, scores)
                        }
                        for scores in slice_scores
                    ],
                    slice_count=len(volume),
                    highest_any_slice_index=top_slice,
                    localization_png=localization_png,
                    input_format=input_format,
                )
        except CTHeadInputError:
            raise
        except CTHeadError:
            raise


__all__ = [
    "CLASS_NAMES",
    "CTHeadError",
    "CTHeadHemorrhageAnalysis",
    "CTHeadHemorrhageService",
    "CTHeadInputError",
    "CTHeadModelContractError",
    "CTHeadModelUnavailableError",
    "MODEL_ID",
    "MODEL_REVISION",
]
