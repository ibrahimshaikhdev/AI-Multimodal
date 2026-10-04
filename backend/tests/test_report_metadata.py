import pytest

from backend.services.report_metadata import extract_report_metadata


def test_extract_report_metadata_preserves_labeled_sections_and_findings():
    text = """MRI KNEE WITHOUT CONTRAST
Study Date: 2025-05-14
Patient DOB: 01/02/1980

Findings:
Mild joint effusion.
No acute fracture.

Impression: Mild joint effusion; no acute fracture.
"""

    metadata = extract_report_metadata(text)

    assert metadata["document_date"] == "2025-05-14"
    assert metadata["sections"] == [
        {"name": "findings", "text": "Mild joint effusion.\nNo acute fracture."},
        {"name": "impression", "text": "Mild joint effusion; no acute fracture."},
    ]
    assert metadata["observations"] == [
        {"section": "findings", "text": "Mild joint effusion."},
        {"section": "findings", "text": "No acute fracture."},
        {"section": "impression", "text": "Mild joint effusion; no acute fracture."},
    ]


def test_extract_report_metadata_does_not_guess_date_or_sections():
    metadata = extract_report_metadata("Patient fell last week and reports knee pain.")

    assert metadata == {
        "document_date": None,
        "sections": [],
        "observations": [],
    }


def test_extract_report_metadata_rejects_blank_text():
    with pytest.raises(ValueError, match="Text content is required"):
        extract_report_metadata(" \n\t ")