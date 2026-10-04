from __future__ import annotations

import re


_UNITS = (
    "mL/min/1.73m2",
    "copies/mL",
    "cells/µL",
    "cells/uL",
    "x10^9/L",
    "x10^3/µL",
    "x10^3/uL",
    "mmol/L",
    "µmol/L",
    "umol/L",
    "pmol/L",
    "nmol/L",
    "mEq/L",
    "mg/dL",
    "mg/L",
    "mg/mmol",
    "g/dL",
    "g/L",
    "ng/mL",
    "ng/dL",
    "pg/mL",
    "mIU/L",
    "uIU/mL",
    "IU/mL",
    "IU/L",
    "U/L",
    "mmHg",
    "mm/hr",
    "mm/h",
    "mL/min",
    "10^9/L",
    "10^6/µL",
    "10^6/uL",
    "µg/dL",
    "µg/L",
    "mcg/dL",
    "mcg/L",
    "mcg/mL",
    "cells/L",
    "kg/m2",
    "kg",
    "cm",
    "mm",
    "mL",
    "mg",
    "mcg",
    "µg",
    "ng",
    "g",
    "L",
    "fL",
    "pg",
    "bpm",
    "sec",
    "min",
    "%",
)
_UNIT_PATTERN = "(?:" + "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True)) + ")"
_VALUE_PATTERN = r"(?:[<>≤≥]\s*)?[+-]?\d+(?:[.,]\d+)?(?:\s*/\s*[+-]?\d+(?:[.,]\d+)?)?"
_PARAMETER_ROW = re.compile(
    rf"^\s*(?P<parameter>[A-Za-z][A-Za-z0-9 _()./%-]*?)\s*(?::|=|\|\s*|\s+)\s*"
    rf"(?P<value>{_VALUE_PATTERN})\s*(?P<unit>{_UNIT_PATTERN})"
    r"(?:\s+(?:H|L|HIGH|LOW|\*))?\s*$",
    re.IGNORECASE,
)


def extract_report_parameters(text: str, document_date: str | None = None) -> list[dict]:
    """Extract explicit value/unit rows without inferring missing results.

    Confidence is null because this rule-based matcher has not been calibrated.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text content is required")

    parameters = []
    for source_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match_line = re.sub(r"^\s*[-*]\s+", "", source_line)
        match = _PARAMETER_ROW.match(match_line)
        if not match:
            continue

        parameter = match.group("parameter").strip().rstrip(":=| ")
        if not parameter:
            continue

        parameters.append(
            {
                "parameter": parameter,
                "value": match.group("value").strip(),
                "unit": match.group("unit"),
                "date": document_date,
                "confidence": None,
                "source": source_line.strip(),
            }
        )

    return parameters


__all__ = ["extract_report_parameters"]