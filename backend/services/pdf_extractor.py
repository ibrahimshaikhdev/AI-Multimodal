from __future__ import annotations

import fitz


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("PDF content is required")

    try:
        document = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
    except Exception as exc:  # pragma: no cover - fitz raises library-specific errors
        raise ValueError("Invalid PDF content") from exc

    try:
        extracted_pages = []
        for page in document:
            text = page.get_text("text", sort=True)
            if text and text.strip():
                extracted_pages.append(text.strip())

        return "\n\n".join(extracted_pages)
    finally:
        document.close()
