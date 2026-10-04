import pytest

from backend.services.report_parameters import extract_report_parameters


def test_extract_report_parameters_returns_explicit_lab_rows_with_source():
    text = """Study Date: 2025-04-16
Hemoglobin: 14.2 g/dL
WBC 7.1 x10^9/L
Blood Pressure | 118/76 mmHg
Patient DOB: 1985-02-03
"""

    parameters = extract_report_parameters(text, document_date="2025-04-16")

    assert parameters == [
        {
            "parameter": "Hemoglobin",
            "value": "14.2",
            "unit": "g/dL",
            "date": "2025-04-16",
            "confidence": None,
            "source": "Hemoglobin: 14.2 g/dL",
        },
        {
            "parameter": "WBC",
            "value": "7.1",
            "unit": "x10^9/L",
            "date": "2025-04-16",
            "confidence": None,
            "source": "WBC 7.1 x10^9/L",
        },
        {
            "parameter": "Blood Pressure",
            "value": "118/76",
            "unit": "mmHg",
            "date": "2025-04-16",
            "confidence": None,
            "source": "Blood Pressure | 118/76 mmHg",
        },
    ]


def test_extract_report_parameters_keeps_comparator_in_value_and_does_not_guess_date():
    parameters = extract_report_parameters("Vitamin D: < 20 ng/mL")

    assert parameters == [
        {
            "parameter": "Vitamin D",
            "value": "< 20",
            "unit": "ng/mL",
            "date": None,
            "confidence": None,
            "source": "Vitamin D: < 20 ng/mL",
        }
    ]


def test_extract_report_parameters_ignores_numbers_without_recognized_units():
    parameters = extract_report_parameters("Patient is 42 years old. History: pain for 3 weeks.")

    assert parameters == []


def test_extract_report_parameters_rejects_blank_text():
    with pytest.raises(ValueError, match="Text content is required"):
        extract_report_parameters("  \n\t ")