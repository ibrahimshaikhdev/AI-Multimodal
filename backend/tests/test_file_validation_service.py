from io import BytesIO

import pytest
from werkzeug.datastructures import FileStorage

from backend.services.file_storage import save_uploaded_file, validate_uploaded_file


def make_upload(filename, content_type, payload):
    stream = BytesIO(payload)
    return FileStorage(stream=stream, filename=filename, content_type=content_type)


def test_validate_uploaded_file_accepts_supported_report_types():
    upload = make_upload("lab-report.pdf", "application/pdf", b"%PDF-1.4\n% test pdf")

    result = validate_uploaded_file(upload)

    assert result["extension"] == ".pdf"
    assert result["mime_type"] == "application/pdf"
    assert result["safe_name"].endswith(".pdf")
    assert result["size"] == len(b"%PDF-1.4\n% test pdf")


def test_validate_uploaded_file_rejects_unsupported_or_dangerous_files():
    upload = make_upload("../evil.exe", "application/octet-stream", b"not a real report")

    with pytest.raises(ValueError, match="unsupported|unsafe|not allowed"):
        validate_uploaded_file(upload)

    upload = make_upload("scan.jpg", "image/jpeg", b"\xff\xd8\xff\x00" * 20000)
    with pytest.raises(ValueError, match="too large|size"):
        validate_uploaded_file(upload, max_size_bytes=10)


def test_save_uploaded_file_safely_persists_to_instance_uploads_dir(tmp_path):
    upload = make_upload("report.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    saved_path = save_uploaded_file(upload, patient_id=42, upload_root=tmp_path)

    assert saved_path.startswith("uploads/patients/42/")
    assert saved_path.endswith(".png")
    assert (tmp_path / saved_path.replace("uploads/", "")).exists()
