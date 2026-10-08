import os
import shutil
from pathlib import Path, PurePosixPath
from uuid import uuid4

from werkzeug.utils import secure_filename

SUPPORTED_FILE_TYPES = {
    ".pdf": {
        "mime_type": "application/pdf",
        "magic_bytes": (b"%PDF",),
    },
    ".docx": {
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "magic_bytes": (b"PK\x03\x04",),
    },
    ".jpg": {
        "mime_type": "image/jpeg",
        "magic_bytes": (b"\xff\xd8\xff",),
    },
    ".jpeg": {
        "mime_type": "image/jpeg",
        "magic_bytes": (b"\xff\xd8\xff",),
    },
    ".png": {
        "mime_type": "image/png",
        "magic_bytes": (b"\x89PNG\r\n\x1a\n",),
    },
    ".zip": {
        "mime_type": ("application/zip", "application/x-zip-compressed"),
        "magic_bytes": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    },
    ".nii": {
        "mime_type": ("application/octet-stream", "application/x-nifti"),
        "magic_bytes": (b"\x5c\x01\x00\x00", b"\x00\x00\x01\x5c"),
    },
    ".gz": {
        "mime_type": ("application/gzip", "application/x-gzip"),
        "magic_bytes": (b"\x1f\x8b",),
    },
}

DEFAULT_MAX_UPLOAD_SIZE = 10 * 1024 * 1024
DEFAULT_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}


def _safe_filename(filename, allowed_extensions=None):
    if not filename or not str(filename).strip():
        raise ValueError("no file selected")

    original_name = os.path.basename(str(filename))
    safe_name = secure_filename(original_name)
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("unsafe file name")

    suffix = Path(safe_name).suffix.lower()
    allowed_extensions = allowed_extensions or DEFAULT_SUPPORTED_EXTENSIONS
    if suffix not in allowed_extensions or suffix not in SUPPORTED_FILE_TYPES:
        raise ValueError("unsupported file type. Allowed: PDF, DOCX, JPG, PNG")

    return safe_name, suffix


def validate_uploaded_file(
    file_storage,
    max_size_bytes=DEFAULT_MAX_UPLOAD_SIZE,
    allowed_extensions=None,
):
    if file_storage is None:
        raise ValueError("No file selected")

    file_name = getattr(file_storage, "filename", None)
    if not file_name:
        raise ValueError("No file selected")

    safe_name, extension = _safe_filename(file_name, allowed_extensions)
    file_storage.stream.seek(0, os.SEEK_END)
    file_size = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if file_size <= 0:
        raise ValueError("Uploaded file is empty")
    if file_size > max_size_bytes:
        raise ValueError(f"File exceeds size limit of {max_size_bytes} bytes")

    signature = file_storage.stream.read(8)
    file_storage.stream.seek(0)
    permitted_signatures = SUPPORTED_FILE_TYPES[extension]["magic_bytes"]
    if not any(signature.startswith(expected) for expected in permitted_signatures):
        raise ValueError("File content does not match the supported document type")

    mime_type = (file_storage.content_type or "").lower()
    expected_mime = SUPPORTED_FILE_TYPES[extension]["mime_type"]
    permitted_mimes = (
        expected_mime if isinstance(expected_mime, tuple) else (expected_mime,)
    )
    if mime_type and mime_type not in permitted_mimes:
        raise ValueError("File MIME type does not match the file extension")

    return {
        "safe_name": safe_name,
        "extension": extension,
        "mime_type": permitted_mimes[0],
        "size": file_size,
    }


def save_uploaded_file(
    file_storage,
    patient_id,
    upload_root=None,
    max_size_bytes=DEFAULT_MAX_UPLOAD_SIZE,
    allowed_extensions=None,
):
    validated = validate_uploaded_file(
        file_storage,
        max_size_bytes=max_size_bytes,
        allowed_extensions=allowed_extensions,
    )
    patient_id = str(patient_id)

    if upload_root is None:
        base_dir = Path(__file__).resolve().parents[2] / "instance" / "uploads"
    else:
        base_dir = Path(upload_root)

    target_dir = base_dir / "patients" / patient_id
    target_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}{validated['extension']}"
    destination = target_dir / stored_name

    file_storage.stream.seek(0)
    with destination.open("wb") as output_file:
        shutil.copyfileobj(file_storage.stream, output_file)

    relative_path = Path("uploads") / "patients" / patient_id / stored_name
    return str(relative_path).replace("\\", "/")


def save_research_paper_file(file_storage, user_id, upload_root=None):
    validated = validate_uploaded_file(
        file_storage,
        allowed_extensions={".pdf", ".docx"},
    )
    base_dir = (
        Path(upload_root)
        if upload_root is not None
        else Path(__file__).resolve().parents[2] / "instance" / "uploads"
    )
    target_dir = base_dir / "research_papers" / str(user_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}{validated['extension']}"
    destination = target_dir / stored_name

    file_storage.stream.seek(0)
    with destination.open("wb") as output_file:
        shutil.copyfileobj(file_storage.stream, output_file)

    return validated["safe_name"], stored_name


def get_stored_file_location(file_reference, patient_id):
    if not isinstance(file_reference, str) or not file_reference:
        return None

    relative_reference = PurePosixPath(file_reference)
    parts = relative_reference.parts
    if (
        relative_reference.is_absolute()
        or len(parts) != 4
        or parts[0] != "uploads"
        or parts[1] != "patients"
        or parts[2] != str(patient_id)
        or any(part in {".", ".."} for part in parts)
    ):
        return None

    upload_root = Path(__file__).resolve().parents[2] / "instance" / "uploads"
    return upload_root, PurePosixPath(*parts[1:]).as_posix()


def delete_stored_file(file_reference, patient_id):
    location = get_stored_file_location(file_reference, patient_id)
    if location is None:
        raise ValueError("Stored file reference is invalid for this patient")

    upload_root, relative_path = location
    (upload_root / relative_path).unlink(missing_ok=True)
