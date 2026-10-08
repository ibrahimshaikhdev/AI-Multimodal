from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from backend.services.report_parameters import extract_report_parameters


SIMILARITY_THRESHOLD = 0.72
_SECTION_HEADER = re.compile(
    r"^\s*(assessment|conclusions?|findings?|impressions?|observations?|"
    r"recommendations?)\s*[:\-]?\s*(.*)$",
    re.IGNORECASE,
)
_LABEL_VALUE = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 /()_-]{1,40})\s*:\s*(?P<value>.*)$")
_METADATA_LABELS = {
    "accession",
    "age",
    "birth date",
    "date of birth",
    "dob",
    "facility",
    "gender",
    "hospital",
    "mrn",
    "name",
    "patient",
    "patient id",
    "patient name",
    "performed",
    "referring physician",
    "report date",
    "sex",
    "study date",
}
_ALIASES = {
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "white blood cell count": "white blood cells",
    "white cell count": "white blood cells",
    "wbc": "white blood cells",
    "rbc": "red blood cells",
    "plt": "platelets",
    "platelet count": "platelets",
    "creat": "creatinine",
    "na": "sodium",
    "k": "potassium",
    "cl": "chloride",
}
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _label_key(label: str) -> str:
    normalized = _NON_ALPHANUMERIC.sub(" ", label.casefold()).strip()
    return _ALIASES.get(normalized, normalized)


def _normalized_value(value: str) -> str:
    if "," in value:
        return re.sub(r"\s+", "", value).casefold()
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return re.sub(r"\s+", "", value).casefold()


def _normalized_unit(unit: str) -> str:
    return re.sub(r"\s+", "", unit.casefold().replace("µ", "u").replace("μ", "u"))


def _parameter_rows(first_text: str, second_text: str) -> list[dict[str, Any]]:
    first_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    second_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target, text in ((first_by_key, first_text), (second_by_key, second_text)):
        for parameter in extract_report_parameters(text):
            target[_label_key(parameter["parameter"])].append(parameter)

    rows = []
    for key in sorted(first_by_key.keys() | second_by_key.keys()):
        first_values = first_by_key.get(key, [])
        second_values = second_by_key.get(key, [])
        for index in range(max(len(first_values), len(second_values))):
            first = first_values[index] if index < len(first_values) else None
            second = second_values[index] if index < len(second_values) else None
            if first is None:
                status = "only_in_report_2"
            elif second is None:
                status = "only_in_report_1"
            elif _normalized_unit(first["unit"]) != _normalized_unit(second["unit"]):
                status = "units_not_comparable"
            elif _normalized_value(first["value"]) == _normalized_value(second["value"]):
                status = "same_value"
            else:
                status = "different_value"
            rows.append(
                {
                    "label": (first or second)["parameter"],
                    "report_1": first,
                    "report_2": second,
                    "status": status,
                }
            )
    status_order = {
        "same_value": 0,
        "different_value": 1,
        "units_not_comparable": 1,
        "only_in_report_1": 2,
        "only_in_report_2": 2,
    }
    rows.sort(key=lambda row: (status_order[row["status"]], _label_key(row["label"])))
    return rows


def _report_observations(text: str) -> list[str]:
    observations = []
    parameter_source_lines = {
        re.sub(r"\W+", "", parameter["source"]).casefold()
        for parameter in extract_report_parameters(text)
    }
    for source_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = source_line.strip()
        if not line:
            continue

        line = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if not line or re.fullmatch(r"(?:page\s+)?\d+(?:\s+of\s+\d+)?", line, re.IGNORECASE):
            continue

        header = _SECTION_HEADER.match(line)
        if header:
            line = header.group(2).strip()
            if not line:
                continue

        label_value = _LABEL_VALUE.match(line)
        if label_value:
            label = _label_key(label_value.group("label"))
            if label in _METADATA_LABELS:
                continue
            value = label_value.group("value").strip()
            if label in {"clinical history", "indication", "technique"} and value:
                line = value
            elif label in {
                "assessment",
                "conclusion",
                "finding",
                "findings",
                "impression",
                "observation",
                "observations",
                "recommendation",
                "recommendations",
            } and value:
                line = value
            elif not value or label in {"report", "page"}:
                continue
            else:
                line = f"{label_value.group('label').strip()}: {value}"

        if re.sub(r"\W+", "", source_line).casefold() in parameter_source_lines:
            continue
        observations.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
            if sentence.strip()
        )
    return list(dict.fromkeys(observations))


def _align_observations(
    first: list[str],
    second: list[str],
    similarity_matrix: Callable[[list[str], list[str]], list[list[float]]],
) -> dict[str, Any]:
    first = first[:40]
    second = second[:40]
    used_first: set[int] = set()
    used_second: set[int] = set()
    related = []
    second_by_normalized: dict[str, list[int]] = defaultdict(list)
    for index, statement in enumerate(second):
        second_by_normalized[_label_key(statement)].append(index)
    for first_index, statement in enumerate(first):
        normalized = _label_key(statement)
        matching_indexes = second_by_normalized.get(normalized, [])
        second_index = next(
            (index for index in matching_indexes if index not in used_second),
            None,
        )
        if normalized and second_index is not None:
            used_first.add(first_index)
            used_second.add(second_index)
            related.append(
                {
                    "report_1": statement,
                    "report_2": second[second_index],
                    "similarity": 1.0,
                    "relationship": "same_wording",
                }
            )

    remaining_first = [index for index in range(len(first)) if index not in used_first]
    remaining_second = [index for index in range(len(second)) if index not in used_second]
    first_candidates = [first[index] for index in remaining_first]
    second_candidates = [second[index] for index in remaining_second]
    similarities = (
        similarity_matrix(first_candidates, second_candidates)
        if first_candidates and second_candidates
        else []
    )
    candidates = sorted(
        (
            (
                float(similarities[first_candidate_index][second_candidate_index]),
                remaining_first[first_candidate_index],
                remaining_second[second_candidate_index],
            )
            for first_candidate_index in range(len(remaining_first))
            for second_candidate_index in range(len(remaining_second))
        ),
        reverse=True,
    )
    for score, first_index, second_index in candidates:
        if (
            score < SIMILARITY_THRESHOLD
            or first_index in used_first
            or second_index in used_second
        ):
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        related.append(
            {
                "report_1": first[first_index],
                "report_2": second[second_index],
                "similarity": round(score, 3),
                "relationship": (
                    "same_wording" if score >= 0.999 else "related_wording"
                ),
            }
        )
    return {
        "related": related,
        "report_1_only": [
            statement for index, statement in enumerate(first) if index not in used_first
        ],
        "report_2_only": [
            statement for index, statement in enumerate(second) if index not in used_second
        ],
    }


def compare_report_texts(
    first_text: str,
    second_text: str,
    similarity_matrix: Callable[[list[str], list[str]], list[list[float]]],
) -> dict[str, Any]:
    parameters = _parameter_rows(first_text, second_text)
    first_observations = _report_observations(first_text)
    second_observations = _report_observations(second_text)
    narrative = _align_observations(
        first_observations,
        second_observations,
        similarity_matrix,
    )
    narrative["omitted_report_1_count"] = max(0, len(first_observations) - 40)
    narrative["omitted_report_2_count"] = max(0, len(second_observations) - 40)
    return {
        "method": "local_rules_and_embeddings",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "summary": {
            "same_value_count": sum(row["status"] == "same_value" for row in parameters),
            "different_value_count": sum(
                row["status"] in {"different_value", "units_not_comparable"}
                for row in parameters
            ),
            "report_1_only_count": sum(row["status"] == "only_in_report_1" for row in parameters)
            + len(narrative["report_1_only"]),
            "report_2_only_count": sum(row["status"] == "only_in_report_2" for row in parameters)
            + len(narrative["report_2_only"]),
            "related_observation_count": len(narrative["related"]),
        },
        "parameters": parameters,
        "observations": narrative,
    }


def compare_research_evidence(
    evidence: dict[str, dict[str, list[dict[str, str]]]],
    fields: tuple[str, ...],
    similarity_matrix: Callable[[list[str], list[str]], list[list[float]]],
) -> dict[str, dict[str, Any]]:
    comparison = {}
    for field in fields:
        first_passages = evidence["paper_1"].get(field, [])
        second_passages = evidence["paper_2"].get(field, [])
        first_text = first_passages[0]["excerpt"] if first_passages else None
        second_text = second_passages[0]["excerpt"] if second_passages else None
        similarity = None
        if first_text and second_text:
            similarity = round(similarity_matrix([first_text], [second_text])[0][0], 3)
            status = "related_evidence" if similarity >= SIMILARITY_THRESHOLD else "different_evidence"
            note = (
                f"Retrieved passages have local embedding similarity {similarity:.3f}. "
                "This score is not a finding of scientific equivalence."
            )
        elif first_text:
            status = "only_in_paper_1"
            note = "A relevant passage was retrieved only from Paper 1."
        elif second_text:
            status = "only_in_paper_2"
            note = "A relevant passage was retrieved only from Paper 2."
        else:
            status = "not_found"
            note = "No relevant passage was retrieved for this field in either paper."
        comparison[field] = {
            "paper_1": first_text or "No relevant passage retrieved.",
            "paper_2": second_text or "No relevant passage retrieved.",
            "paper_1_citation": first_passages[0]["citation"] if first_passages else None,
            "paper_2_citation": second_passages[0]["citation"] if second_passages else None,
            "similarity": similarity,
            "status": status,
            "comparison": note,
        }
    return comparison


__all__ = ["compare_report_texts", "compare_research_evidence"]
