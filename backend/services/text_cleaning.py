from __future__ import annotations

import re


def _normalize_whitespace(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "")
    cleaned = re.sub(r"[\t\f\v]+", " ", cleaned)

    lines = []
    blank_lines = 0
    for line in cleaned.split("\n"):
        line = re.sub(r" +", " ", line).strip()
        if line:
            lines.append(line)
            blank_lines = 0
        elif lines and blank_lines == 0:
            lines.append("")
            blank_lines = 1

    return "\n".join(lines).strip()


def _normalize_label_spacing(text: str) -> str:
    text = re.sub(r"\s*:\s*", ": ", text)

    match = re.match(r"^(?P<label>[A-Za-z][A-Za-z0-9 /()\-]+?)\s+(?P<value>\d+.*)$", text)
    if match and ":" not in text:
        label = match.group("label").strip()
        value = match.group("value").strip()
        if label and value:
            text = f"{label}: {value}"

    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_extracted_text(raw_text: str) -> str:
    if raw_text is None:
        raise ValueError("Text content is required")

    cleaned = str(raw_text).strip()
    if not cleaned:
        raise ValueError("Text content is required")

    cleaned = _normalize_whitespace(cleaned)
    lines = []
    for line in cleaned.split("\n"):
        if not line:
            lines.append("")
            continue
        normalized = _normalize_label_spacing(line)
        lines.append(normalized.strip())

    cleaned_lines = ["" if re.fullmatch(r"[-*_]{3,}", line) else line for line in lines]
    final_text = "\n".join(cleaned_lines)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)

    return final_text.strip()


__all__ = ["clean_extracted_text"]
