from __future__ import annotations

from collections import Counter
import re
from difflib import get_close_matches
from pathlib import Path

from sqlalchemy.inspection import inspect as sqlalchemy_inspect

from backend.extensions import db
from backend.models import MedicalReport, Patient, ResearchPaper, ScanAsset
from backend.services.own_llm import (
    LAB_DISPLAY_NAMES,
    _LAB_ALIAS_PATTERNS,
    OWN_ASSISTANT_SYSTEM,
    STANDARD_LAB_RANGES,
    _canonical_test_name,
    _convert_lab_value,
    _latest_lab_report,
    _lab_results_for_report,
    _report_date,
    build_own_assistant_prompt,
    own_generate_assistant,
    retrieve_project_knowledge,
)
from backend.services.patient_context import (
    get_owned_patient_records,
    get_owned_reports,
    list_owned_patients,
)


MEDICAL_DISCLAIMER = (
    "Note: this is AI-generated general information, not real medical advice."
)
MISSING_SELECTION = "Please select a patient (and a report) first."
MAX_PATIENT_DATA_CHARACTERS = 2500


def _extract_test_name(question: str) -> str | None:
    for alias_pattern, canonical in _LAB_ALIAS_PATTERNS:
        if alias_pattern.search(question):
            return canonical
    tokens = re.findall(r"[a-z0-9]+", question.casefold())
    for token in tokens:
        if len(token) < 4:
            continue
        aliases = [
            alias
            for specification in STANDARD_LAB_RANGES.values()
            for alias in specification["aliases"]
        ]
        repaired = get_close_matches(token, aliases, n=1, cutoff=0.8)
        if repaired:
            canonical = _canonical_test_name(repaired[0])
            if canonical:
                return canonical
    return None


def count_or_list(question: str, user_id: int) -> str:
    question_folded = question.casefold()
    patients = list_owned_patients(user_id)
    patient_names = [f"{patient.first_name} {patient.last_name}" for patient in patients]
    reports = (
        db.session.query(MedicalReport.id)
        .join(Patient, MedicalReport.patient_id == Patient.id)
        .filter(Patient.created_by_id == user_id)
        .count()
    )
    scans = (
        db.session.query(ScanAsset.id)
        .join(Patient, ScanAsset.patient_id == Patient.id)
        .filter(Patient.created_by_id == user_id)
        .count()
    )
    papers = (
        db.session.query(ResearchPaper.id)
        .filter_by(user_id=user_id)
        .count()
    )
    answers = []
    if re.search(r"\bpatients?\b", question_folded):
        answers.append(f"You have {len(patient_names)} patient(s).")
        if re.search(r"\b(list|show|names?)\b", question_folded):
            names = ", ".join(patient_names[:20]) or "none"
            if len(patient_names) > 20:
                names += f", and {len(patient_names) - 20} more"
            answers.append(f"Patients: {names}.")
    if re.search(r"\breports?\b", question_folded):
        answers.append(f"You have {reports} medical report(s).")
    if re.search(r"\bscans?\b|\bx[- ]?rays?\b|\bimaging\b", question_folded):
        answers.append(f"You have {scans} medical scan(s).")
    if re.search(r"\b(?:research\s+)?papers?\b", question_folded):
        answers.append(f"You have {papers} uploaded research paper(s).")
    return " ".join(answers)


def find_patients_by_lab(question: str, user_id: int) -> str:
    canonical = _extract_test_name(question)
    if canonical is None:
        return "Please name a supported lab test, such as HbA1c, glucose, or hemoglobin."
    folded = question.casefold()
    if "lowest" in folded:
        direction = "lowest"
    elif "highest" in folded:
        direction = "highest"
    elif "low" in folded:
        direction = "low"
    elif "abnormal" in folded or "diabetic" in folded:
        direction = "abnormal"
    else:
        direction = "high"

    matches = []
    for patient in list_owned_patients(user_id):
        records = get_owned_patient_records(user_id, patient.id)
        if records is None:
            continue
        result = None
        for report in records["reports"]:
            result = next(
                (
                    value
                    for value in _lab_results_for_report(report)
                    if value["canonical"] == canonical
                ),
                None,
            )
            if result is not None:
                break
        if result is None:
            continue
        if direction in {"highest", "lowest"}:
            number = _convert_lab_value(
                canonical,
                float(result["value"].replace(",", "")),
                result["unit"],
            )
            if number is not None:
                matches.append((number, patient, result))
        elif direction == "abnormal" and result["status"] != "normal":
            matches.append((0.0, patient, result))
        elif result["status"] == direction:
            matches.append((0.0, patient, result))

    display_test = LAB_DISPLAY_NAMES.get(canonical, canonical)
    if not matches:
        return (
            f"No patients with {direction} {display_test} were found in the "
            "uploaded reports."
        )
    if direction == "highest":
        matches = [max(matches, key=lambda item: item[0])]
    elif direction == "lowest":
        matches = [min(matches, key=lambda item: item[0])]
    prefix = f"Patients with {direction} {display_test}: "
    entries = [
        f"{patient.first_name} {patient.last_name} "
        f"({result['value']} {result['unit']})"
        for _, patient, result in matches[:15]
    ]
    return prefix + ", ".join(entries) + "."


def research_titles(user_id: int) -> str:
    papers = (
        db.session.query(ResearchPaper.title)
        .filter_by(user_id=user_id)
        .order_by(ResearchPaper.created_at.desc(), ResearchPaper.id.desc())
        .all()
    )
    titles = [title for (title,) in papers]
    if not titles:
        return "There are no research papers uploaded to your account yet. Ask research questions on the Research RAG page."
    return (
        "Research questions are answered on the Research RAG page. "
        f"Your uploaded papers ({len(titles)}): " + "; ".join(titles[:20]) + "."
    )


def _remarks_line(text: str) -> str | None:
    for line in text.splitlines():
        if re.match(
            r"^\s*(?:remarks?|impression|conclusion|comments?)\s*[:\-]",
            line,
            re.IGNORECASE,
        ):
            return line.strip()
    return None


def _report_patient_data(report: MedicalReport, patient_name: str) -> str:
    date_text = _report_date(report)
    results = _lab_results_for_report(report)
    lines = [f"{patient_name}, selected report ({date_text}):"]
    if results:
        lines.extend(
            f"{result['test']}: {result['value']} {result['unit']} "
            f"({result['status']})"
            for result in results
        )
    elif report.extracted_text:
        lines.append(report.extracted_text.strip())
    remarks = _remarks_line(report.extracted_text or "")
    if remarks:
        lines.append(remarks)
    return "\n".join(lines)


def _scan_output(scan: ScanAsset) -> str:
    base = (
        f"Scan ({scan.study_date.isoformat() if scan.study_date else 'date unavailable'}): "
        f"{scan.modality}/{scan.body_region or 'body region not specified'}"
    )
    for relationship_name in (
        "cardiac_analysis", "knee_analysis", "spine_analysis", "brain_analysis",
        "chest_xray_analysis", "bone_xray_analysis", "dental_xray_analysis",
        "ct_head_analysis", "chest_ct_analysis", "obstetric_ultrasound_analysis",
        "abdominal_aorta_ultrasound_analysis", "echoview47_analysis",
        "vascular_ultrasound_analysis",
    ):
        analysis = getattr(scan, relationship_name, None)
        if analysis is None:
            continue
        values = {
            column.key: getattr(analysis, column.key)
            for column in sqlalchemy_inspect(type(analysis)).column_attrs
            if column.key not in {"id", "scan_id", "created_at", "model_name"}
            and getattr(analysis, column.key) is not None
        }
        label = next(
            (
                values[key]
                for key in (
                    "predicted_label", "predicted_class", "classification",
                    "label", "class_name", "findings",
                )
                if key in values
            ),
            None,
        )
        confidence = next(
            (
                values[key]
                for key in ("confidence_score", "confidence", "model_score")
                if key in values
            ),
            None,
        )
        if label is None:
            label = ", ".join(f"{key}: {value}" for key, value in values.items())
        confidence_text = (
            f", confidence {float(confidence) * 100:.1f}%"
            if isinstance(confidence, (int, float))
            else ""
        )
        return (
            f"{base}; model output (experimental, not a diagnosis): "
            f"{label}{confidence_text}"
        )
    return f"{base}; no saved model output ({scan.processing_status})"


def _trim_patient_data(lines: list[str]) -> str:
    text = "\n".join(lines)
    if len(text) <= MAX_PATIENT_DATA_CHARACTERS:
        return text
    return text[: MAX_PATIENT_DATA_CHARACTERS - 3].rstrip() + "..."


def _timeline_date(value) -> str:
    if value is None:
        return "Date unavailable"
    return value.strftime("%b %d, %Y").replace(" 0", " ")


def _timeline_report_summary(report: MedicalReport) -> str | None:
    results = _lab_results_for_report(report)
    if results:
        flagged = [result for result in results if result["status"] != "normal"]
        if not flagged:
            return f"{len(results)} results parsed; no values flagged out of range"
        highlights = [
            f"{result['test']} {result['value']} {result['unit']} "
            f"({result['status']})"
            for result in flagged[:3]
        ]
        remaining = len(flagged) - len(highlights)
        if remaining:
            highlights.append(f"+{remaining} more")
        return f"{len(flagged)} flagged: " + "; ".join(highlights)

    remarks = _remarks_line(report.extracted_text or "")
    if remarks:
        clean_remarks = re.sub(r"\s+", " ", remarks).strip()
        return clean_remarks[:160] + ("..." if len(clean_remarks) > 160 else "")
    return None


def _timeline_scan_summary(scan: ScanAsset) -> str:
    for relationship_name in (
        "cardiac_analysis", "knee_analysis", "spine_analysis", "brain_analysis",
        "chest_xray_analysis", "bone_xray_analysis", "dental_xray_analysis",
        "ct_head_analysis", "chest_ct_analysis", "obstetric_ultrasound_analysis",
        "abdominal_aorta_ultrasound_analysis", "echoview47_analysis",
        "vascular_ultrasound_analysis",
    ):
        analysis = getattr(scan, relationship_name, None)
        if analysis is None:
            continue
        for field_name in (
            "predicted_label", "predicted_class", "classification",
            "label", "class_name", "findings",
        ):
            value = getattr(analysis, field_name, None)
            if isinstance(value, str) and value.strip():
                return f"Experimental model label: {value.strip()}"
            if isinstance(value, list):
                labels = [
                    item.get("name") or item.get("label") or item.get("class_name")
                    for item in value
                    if isinstance(item, dict)
                ]
                counts = Counter(label for label in labels if isinstance(label, str) and label)
                if counts:
                    summary = ", ".join(
                        f"{count} {label}" for label, count in counts.most_common(2)
                    )
                    extra = len(counts) - 2
                    if extra:
                        summary += f", +{extra} other label(s)"
                    return f"Experimental model marked {summary}"
    if scan.processing_status in {"awaiting_model", "model_unavailable"}:
        return "Analysis not available"
    return "Experimental analysis saved; technical details omitted"


def format_timeline_answer(context: dict) -> str:
    patient_id = context["patient_id"]
    user_id = context["user_id"]
    patient_records = get_owned_patient_records(
        user_id,
        patient_id,
        report_limit=0,
        scan_limit=None,
    )
    if patient_records is None:
        return "The selected patient could not be found in your account."

    patient = patient_records["patient"]
    patient_name = f"{patient.first_name} {patient.last_name}"
    report_ids = [
        source["report_id"]
        for source in context.get("sources", [])
        if source.get("type") == "report"
    ]
    reports = get_owned_reports(user_id, report_ids) if report_ids else []
    events = []
    unbriefed_reports = 0
    for report in reports or []:
        details = _timeline_report_summary(report)
        if details is None:
            unbriefed_reports += 1
            continue
        report_type = report.report_type or "medical report"
        events.append(
            (
                report.report_date.isoformat() if report.report_date else None,
                f"- {_timeline_date(report.report_date)} · "
                f"{report_type}: {details}",
            )
        )

    scans = patient_records["scans"]
    for scan in scans:
        date_value = scan.study_date.isoformat() if scan.study_date else None
        if date_value is None:
            continue
        modality = scan.modality or "Imaging"
        region = scan.body_region or "body region not specified"
        events.append(
            (
                date_value,
                f"- {_timeline_date(scan.study_date)} · {modality} "
                f"({region}): {_timeline_scan_summary(scan)}",
            )
        )

    events.sort(key=lambda event: (event[0] is not None, event[0] or ""), reverse=True)
    dated_lines = [line for _, line in events[:4]]
    undated_scans = [scan for scan in scans if scan.study_date is None]
    if undated_scans:
        scan_groups = Counter(
            f"{scan.modality or 'Imaging'} ({scan.body_region or 'region unspecified'})"
            for scan in undated_scans
        )
        group_text = ", ".join(
            f"{count} {label}" if count > 1 else label
            for label, count in scan_groups.most_common(3)
        )
        remaining_groups = len(scan_groups) - 3
        if remaining_groups > 0:
            group_text += f", +{remaining_groups} other type(s)"
        dated_lines.append(
            f"- Undated scans: {len(undated_scans)} saved "
            f"({group_text}); technical model details omitted."
        )

    all_reports = len(report_ids)
    total_records = all_reports + len(scans)
    if not total_records:
        return f"No reports or scans are saved for {patient_name} yet."
    if not dated_lines:
        dated_lines.append("No dated report or scan events are available.")
    hidden_events = max(0, len(events) - min(4, len(events)))
    if hidden_events:
        dated_lines.append(
            f"- {hidden_events} additional dated event(s) not shown."
        )
    if unbriefed_reports:
        dated_lines.append(
            f"- {unbriefed_reports} report(s) omitted because no concise "
            "medical summary was available."
        )

    overview = (
        f"{patient_name} · Timeline overview "
        f"({all_reports} reports, {len(scans)} scans)\n"
        + "\n".join(dated_lines)
    )
    return (
        f"{overview}\n\n"
        "Scan model outputs are experimental, not diagnoses. "
        f"{MEDICAL_DISCLAIMER}"
    )


def _prepare_patient_data(intent: str, context: dict) -> str:
    sources = context.get("sources", [])
    report_sources = [
        source for source in sources if source.get("type") == "report"
    ]
    if context.get("context_type") == "report" and sources:
        report_sources = [
            {**sources[0], "type": "report"}
        ]
    patient_name = context.get("patient_name", "Patient")
    if context.get("context_type") == "report" and sources:
        patient_name = sources[0].get("patient_name", patient_name)
    records = []
    for source in report_sources:
        reports = get_owned_reports(
            context["user_id"],
            [source["report_id"]],
        )
        if reports:
            records.append(reports[0])
    scans = []
    patient_id = context.get("patient_id")
    if patient_id is not None:
        patient_records = get_owned_patient_records(
            context["user_id"],
            patient_id,
            scan_limit=None,
        )
        if patient_records is not None:
            scans = patient_records["scans"]

    if intent == "selected_report":
        if not records:
            return ""
        report = records[0]
        return _trim_patient_data(
            [_report_patient_data(report, patient_name)]
        )
    if intent == "patient_values":
        lab_report, results = _latest_lab_report(records)
        if lab_report is None:
            lines = ["(no lab values found in this patient's reports)"]
        else:
            lines = [
                f"{patient_name}, latest report ({_report_date(lab_report)}):",
                *[
                    f"{value['test']}: {value['value']} {value['unit']} ({value['status']})"
                    for value in results
                ],
            ]
        lines.extend(_scan_output(scan) for scan in scans[:2])
        return _trim_patient_data(lines)
    if intent == "scans":
        return _trim_patient_data(
            [_scan_output(scan) for scan in scans[:10]]
            or ["(no scan records found for this patient)"]
        )
    if intent == "timeline":
        events = []
        for report in records:
            event_date = report.report_date.isoformat() if report.report_date else _report_date(report)
            results = _lab_results_for_report(report)
            abnormal = [
                f"{result['test']} {result['value']} {result['unit']} ({result['status']})"
                for result in results
                if result["status"] != "normal"
            ]
            description = ", ".join(abnormal) if abnormal else "No recognized abnormal lab flags"
            events.append(
                (event_date, f"{event_date} — {report.report_type}: {description}")
            )
        for scan in scans:
            event_date = scan.study_date.isoformat() if scan.study_date else "date unavailable"
            events.append((event_date, f"{event_date} — {_scan_output(scan)}"))
        return _trim_patient_data(
            [line for _, line in sorted(events, key=lambda item: item[0])]
            or ["(no report or scan events found for this patient)"]
        )
    if intent == "compare_reports":
        lab_reports = [
            (report, _lab_results_for_report(report))
            for report in records
        ]
        lab_reports = [
            (report, values) for report, values in lab_reports if len(values) >= 3
        ][:2]
        if len(lab_reports) < 2:
            return "(not enough lab reports with recognized values to compare)"
        previous_report, previous_results = lab_reports[1]
        latest_report, latest_results = lab_reports[0]
        previous = {result["canonical"]: result for result in previous_results}
        latest = {result["canonical"]: result for result in latest_results}
        changes = []
        for canonical in sorted(previous.keys() & latest.keys()):
            old = previous[canonical]
            new = latest[canonical]
            changes.append(
                f"{LAB_DISPLAY_NAMES.get(canonical, canonical)}: "
                f"{old['value']} -> {new['value']} {new['unit']} "
                f"({old['status']} -> {new['status']})"
            )
        return _trim_patient_data(
            [
                f"Previous report ({_report_date(previous_report)}):",
                *[
                    f"{value['test']}: {value['value']} {value['unit']} ({value['status']})"
                    for value in previous_results
                ],
                f"Latest report ({_report_date(latest_report)}):",
                *[
                    f"{value['test']}: {value['value']} {value['unit']} ({value['status']})"
                    for value in latest_results
                ],
                "Changes (previous -> latest):",
                *changes,
            ]
        )
    return ""


def _remove_repeated_content(answer: str) -> str:
    disclaimer_pattern = re.compile(
        r"(?im)^\s*(?:note:\s*)?(?:this is an? )?ai[- ]generated[^.!?\n]*"
        r"(?:not real medical advice|not medical advice)[.!?]*\s*",
    )
    answer = disclaimer_pattern.sub("", answer).strip()
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    unique = []
    seen = set()
    for sentence in sentences:
        normalized = re.sub(r"[\W_]+", " ", sentence).strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(sentence)
    answer = " ".join(unique)
    words = answer.split()
    for start in range(len(words)):
        repeated_phrase = words[start : start + 8]
        if len(repeated_phrase) < 8:
            break
        for later in range(start + 8, len(words) - 7):
            if [word.casefold() for word in words[later : later + 8]] == [
                word.casefold() for word in repeated_phrase
            ]:
                return " ".join(words[:later]).rstrip(" ,;:-")
    return answer.strip()


def generate_grounded_answer(
    intent: str,
    question: str,
    user_id: int,
    *,
    patient_data: str = "(none)",
    project_knowledge: str = "(none)",
    web_search_results: list[dict[str, str]] | None = None,
    max_tokens: int,
    previous_answer: str | None = None,
) -> tuple[str, str]:
    prompt = build_own_assistant_prompt(
        question,
        user_id,
        patient_data=patient_data,
        project_knowledge=project_knowledge,
        web_search_results=web_search_results,
    )
    if previous_answer:
        prompt += (
            "\n\nThe prior assistant response was:\n"
            f"{previous_answer}\n"
            "Do not repeat or paraphrase that response. Answer the current "
            "question with different, directly relevant information."
        )
    answer = _remove_repeated_content(
        own_generate_assistant(
            prompt,
            max_tokens,
            system=OWN_ASSISTANT_SYSTEM,
        )
    )
    if intent in {"selected_report", "patient_values", "scans", "timeline", "compare_reports"}:
        if intent in {"scans", "timeline"} and "Scan (" in patient_data:
            answer = (
                "Scan model outputs are experimental and are not confirmed diagnoses.\n\n"
                + answer
            )
        answer = f"{answer}\n\n{MEDICAL_DISCLAIMER}".strip()
    return answer, prompt


def medical_term_answer(question: str) -> str:
    glossary_path = Path(__file__).resolve().parents[2] / "project_kb" / "glossary.md"
    if not glossary_path.is_file():
        return f"I don't have that term in my reference.\n\n{MEDICAL_DISCLAIMER}"
    text = glossary_path.read_text(encoding="utf-8")
    normalized_question = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    for line in text.splitlines():
        if line.strip().startswith("#") or "requires human review" in line.casefold():
            continue
        match = re.match(r"^\s*[-*]?\s*\*{0,2}([^:*]+)\*{0,2}\s*:\s*(.+)$", line)
        if match is None:
            continue
        term = re.sub(r"[^a-z0-9]+", " ", match.group(1).casefold()).strip()
        if term and re.search(rf"\b{re.escape(term)}\b", normalized_question):
            return f"{match.group(2).strip()}\n\n{MEDICAL_DISCLAIMER}"
    return f"I don't have that term in my reference.\n\n{MEDICAL_DISCLAIMER}"


def handle_fixed_intent(intent: str, question: str, user_id: int) -> str | None:
    if intent == "small_talk":
        normalized = question.casefold()
        if re.search(r"\b(?:thanks|thank you|thx|thanku|tnx)\b", normalized):
            return "You're welcome. What would you like to know?"
        if re.search(r"\b(?:bye|goodbye|see you)\b", normalized):
            return "Goodbye! Feel free to ask whenever you need help."
        if re.search(r"\b(?:how are you|how r you|whats up)\b", normalized):
            return "I'm here and ready to help. What would you like to explore?"
        return "Hi! I can help with your selected records, project information, or research papers."
    if intent == "about_assistant":
        return (
            "I can answer questions about this project, retrieve information from "
            "your selected patient reports and scans, summarize timelines, compare "
            "reports, and show uploaded research-paper titles. I can also explain "
            "common lab terms. I don't diagnose or prescribe."
        )
    if intent == "fallback":
        return (
            "I can help with patient reports, lab values, timelines, scans, "
            "research papers, medical terms, and questions about this project. "
            "What would you like to know?"
        )
    if intent == "counts_lists":
        return count_or_list(question, user_id)
    if intent == "find_patients_by_lab":
        return find_patients_by_lab(question, user_id)
    if intent == "research":
        return research_titles(user_id)
    if intent == "medical_term":
        return medical_term_answer(question)
    if intent == "project_info":
        return None
    return None
