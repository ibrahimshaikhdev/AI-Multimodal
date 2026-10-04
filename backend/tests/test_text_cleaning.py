import re

from backend.services.text_cleaning import clean_extracted_text


def test_clean_extracted_text_normalizes_common_ocr_noise():
    raw_text = """
    Patient:  Jordan Lee\n\nDoctor:  Dr. Smith\n
    Hemoglobin  :  14.2  g/dL\n
    Blood Pressure   118/76 mmHg\n
    ---\n\n
    Notes:   a   lot   of   spaces   and\n\n\n\nweird linebreaks.
    """

    cleaned = clean_extracted_text(raw_text)

    assert "Patient: Jordan Lee" in cleaned
    assert "Doctor: Dr. Smith" in cleaned
    assert "Hemoglobin: 14.2 g/dL" in cleaned
    assert "Blood Pressure: 118/76 mmHg" in cleaned
    assert "Notes: a lot of spaces and" in cleaned
    assert re.search(r"\n\n\n", cleaned) is None


def test_clean_extracted_text_preserves_document_layout():
    raw_text = """PATIENT INFORMATION
Name:  Morgan Reed
Date of Birth:  1985-02-03

FINDINGS
- Mild joint effusion
- No acute fracture

IMPRESSION
No acute osseous abnormality.
"""

    cleaned = clean_extracted_text(raw_text)

    assert "PATIENT INFORMATION\nName: Morgan Reed\nDate of Birth: 1985-02-03" in cleaned
    assert "\n\nFINDINGS\n- Mild joint effusion\n- No acute fracture\n\nIMPRESSION\n" in cleaned


def test_clean_extracted_text_rejects_empty_input():
    try:
        clean_extracted_text("   \n\t  ")
        assert False, "Expected ValueError for empty text"
    except ValueError:
        pass
