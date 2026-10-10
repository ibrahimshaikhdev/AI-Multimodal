from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from backend.services.ct_head_hemorrhage import (
    CLASS_NAMES,
    CTHeadHemorrhageService,
    CTHeadInputError,
    LOCALIZATION_THRESHOLD,
    MODEL_ID,
)


class FakeHemorrhageModel:
    def predict_hu_volume(
        self,
        volume,
        *,
        batch_size,
        return_segmentation,
    ):
        assert batch_size == 1
        assert return_segmentation is False
        depth = len(volume)
        series_scores = np.linspace(0.1, 0.6, 6, dtype=np.float32)
        slice_scores = np.tile(series_scores, (depth, 1))
        slice_scores[1, 5] = 0.95
        output = {
            "series_classification": series_scores,
            "slice_classification": slice_scores,
        }
        return output

    def predict(self, slab):
        assert slab.shape == (9, 512, 512)
        return {
            "classification": np.full(6, 0.25, dtype=np.float32),
            "segmentation": np.full((6, 512, 512), 0.25, dtype=np.float32),
        }


def nifti_bytes(volume):
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "head.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), path)
        return path.read_bytes()


def test_nifti_series_returns_six_class_scores_and_localization():
    service = CTHeadHemorrhageService(
        model_loader=lambda: FakeHemorrhageModel(),
    )
    volume = np.zeros((40, 48, 3), dtype=np.float32)

    result = service.analyze_bytes(nifti_bytes(volume), "head.nii.gz")

    assert result.model_name == MODEL_ID
    assert list(result.series_classification) == CLASS_NAMES
    assert all(0 <= score <= 1 for score in result.series_classification.values())
    assert len(result.slice_classification) == 3
    assert result.highest_any_slice_index == 1
    assert result.slice_count == 3
    assert result.input_format == "NIfTI volume"
    assert result.localization_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_localization_overlay_draws_box_around_thresholded_mask():
    import cv2

    image = np.zeros((512, 512), dtype=np.float32)
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:120, 200:230] = LOCALIZATION_THRESHOLD

    encoded = CTHeadHemorrhageService._localization_png(image, mask)
    overlay = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert tuple(overlay[100, 210]) == (0, 255, 0)
    assert tuple(overlay[110, 200]) == (0, 255, 0)
    assert tuple(overlay[110, 210]) != (0, 255, 0)


def test_jpeg_screenshot_is_rejected_for_head_ct():
    service = CTHeadHemorrhageService(model_loader=lambda: FakeHemorrhageModel())

    with pytest.raises(CTHeadInputError, match="JPG/PNG images are not CT volumes"):
        service.analyze_bytes(b"\xff\xd8\xffnot a CT", "screenshot.jpg")


def test_invalid_nifti_volume_is_rejected():
    service = CTHeadHemorrhageService(model_loader=lambda: FakeHemorrhageModel())

    with pytest.raises(CTHeadInputError, match="3D NIfTI"):
        service.analyze_bytes(b"not a NIfTI volume", "head.nii.gz")


def test_zip_path_traversal_is_rejected():
    service = CTHeadHemorrhageService(model_loader=lambda: FakeHemorrhageModel())
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr("../outside.dcm", b"not a DICOM")

    with pytest.raises(CTHeadInputError, match="unsafe path"):
        service.analyze_bytes(archive_bytes.getvalue(), "series.zip")
