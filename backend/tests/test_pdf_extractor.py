import fitz

from backend.services.pdf_extractor import extract_pdf_text


def test_extract_pdf_text_returns_readable_text():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Patient: Jordan Lee\nHemoglobin: 14.2 g/dL\nDoctor: Dr. Smith")
    pdf_bytes = document.tobytes()
    document.close()

    extracted = extract_pdf_text(pdf_bytes)

    assert "Jordan Lee" in extracted
    assert "Hemoglobin" in extracted
    assert "Dr. Smith" in extracted


def test_extract_pdf_text_uses_visual_reading_order():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((350, 72), "Right column")
    page.insert_text((72, 72), "Left column")
    pdf_bytes = document.tobytes()
    document.close()

    extracted = extract_pdf_text(pdf_bytes)

    assert extracted.index("Left column") < extracted.index("Right column")


def test_extract_pdf_text_rejects_invalid_pdf_bytes():
    try:
        extract_pdf_text(b"not a valid pdf")
        assert False, "Expected ValueError for invalid PDF bytes"
    except ValueError:
        pass
