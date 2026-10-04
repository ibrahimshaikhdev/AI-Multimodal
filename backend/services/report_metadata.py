from __future__ import annotations

import re
from datetime import datetime


_SECTION_ALIASES = {
    "clinical history": "clinical_history",
    "history": "clinical_history",
    "indication": "indication",
    "technique": "technique",
    "comparison": "comparison",
    "findings": "findings",
    "impression": "impression",
    "conclusion": "conclusion",
    "assessment": "assessment",
    "recommendation": "recommendation",
}
_DATE_LABEL = re.compile(
    r"^\s*(?:study date|date of study|exam date|date of exam|examination date|report date|date of report)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)
_DATE_CANDIDATE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|"
    r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\b"
)
_HEADING = re.compile(r"^\s*(?P<heading>[A-Za-z ]+?)\s*:\s*(?P<content>.*?)\s*$")


def _parse_labeled_date(line: str) -> str | None:
    match = _DATE_LABEL.match(line)
    if not match:
        return None

    candidate = _DATE_CANDIDATE.search(match.group(1))
    if not candidate:
        return None

    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate.group(0), date_format).date().isoformat()
        except ValueError:
            continue
    return None


def _section_heading(line: str) -> tuple[str, str] | None:
    match = _HEADING.match(line)
    if match:
        section_name = _SECTION_ALIASES.get(match.group("heading").strip().casefold())
        if section_name:
            return section_name, match.group("content").strip()

    section_name = _SECTION_ALIASES.get(line.strip().rstrip(":").casefold())
    if section_name:
        return section_name, ""
    return None


def extract_report_metadata(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text content is required")

    document_date = None
    sections = []
    current_name = None
    current_lines = []

    def save_section():
        if current_name is None:
            return
        section_text = "\n".join(line for line in current_lines if line).strip()
        if section_text:
            sections.append({"name": current_name, "text": section_text})

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        date_value = _parse_labeled_date(line)
        if date_value and document_date is None:
            document_date = date_value

        heading = _section_heading(line)
        if heading:
            save_section()
            current_name, inline_content = heading
            current_lines = [inline_content] if inline_content else []
        elif current_name is not None:
            if line.strip():
                current_lines.append(line.strip())
            elif current_lines and current_lines[-1] != "":
                current_lines.append("")

    save_section()

    observations = []
    for section in sections:
        if section["name"] in {"findings", "impression", "conclusion", "assessment"}:
            for observation in section["text"].splitlines():
                observation = observation.strip()
                if observation:
                    observations.append({"section": section["name"], "text": observation})

    return {
        "document_date": document_date,
        "sections": sections,
        "observations": observations,
    }


__all__ = ["extract_report_metadata"]