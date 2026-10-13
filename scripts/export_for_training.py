from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import types
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from flask import Flask
from sqlalchemy import event
from sqlalchemy.engine import URL, make_url

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Import backend modules without running package initializers that create the
# web app, warm scan models, or initialize the application's database.
for package_name, package_path in (
    ("backend", SCRIPT_ROOT / "backend"),
    ("backend.services", SCRIPT_ROOT / "backend" / "services"),
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package

from backend.config.database import resolve_database_url
from backend.config.settings import DEFAULT_DATABASE_PATH, PROJECT_ROOT, settings
from backend.extensions import db
from backend.models import MedicalReport, Patient, ResearchPaper
from backend.services.ocr_service import extract_document_text
from backend.services.own_llm import OWN_ASSISTANT_SYSTEM, build_own_assistant_prompt


OUTPUT_DIRECTORY = PROJECT_ROOT / "export_for_training"
DATE_SHIFT_DAYS = 731
REPORTS_REQUIRED = 6
PAPERS_REQUIRED = 2
PAPER_SAMPLE_CHARACTERS = 3000

DATE_PATTERNS = (
    (
        re.compile(r"(?<!\d)(?P<year>19\d{2}|20\d{2})-(?P<month>\d{1,2})-(?P<day>\d{1,2})(?!\d)"),
        "iso",
    ),
    (
        re.compile(r"(?<!\d)(?P<first>\d{1,2})/(?P<second>\d{1,2})/(?P<year>(?:19|20)\d{2})(?!\d)"),
        "numeric",
    ),
    (
        re.compile(
            r"(?i)\b(?P<month_name>January|February|March|April|May|June|July|"
            r"August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|"
            r"Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(?P<day>\d{1,2}),?\s+"
            r"(?P<year>(?:19|20)\d{2})\b"
        ),
        "month_first",
    ),
    (
        re.compile(
            r"(?i)\b(?P<day>\d{1,2})\s+(?P<month_name>January|February|March|"
            r"April|May|June|July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec),?\s+"
            r"(?P<year>(?:19|20)\d{2})\b"
        ),
        "day_first",
    ),
)
MONTH_NUMBERS = {
    name.casefold(): number
    for number, names in enumerate(
        (
            ("January", "Jan"),
            ("February", "Feb"),
            ("March", "Mar"),
            ("April", "Apr"),
            ("May",),
            ("June", "Jun"),
            ("July", "Jul"),
            ("August", "Aug"),
            ("September", "Sep", "Sept"),
            ("October", "Oct"),
            ("November", "Nov"),
            ("December", "Dec"),
        ),
        start=1,
    )
    for name in names
}


@dataclass(frozen=True)
class ReportCandidate:
    report_id: int
    patient_id: int
    patient_first_name: str
    patient_last_name: str
    extracted_text: str
    report_type: str
    lab_names: frozenset[str]
    layout_signature: tuple[int, bool, bool]


def _sqlite_readonly_url(database_url: str) -> URL:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database:
        raise RuntimeError("The configured database must be a persistent SQLite database.")
    if url.database == ":memory:":
        raise RuntimeError("A persistent SQLite database is required for this export.")

    path = Path(url.database)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise RuntimeError("The configured SQLite database file was not found.")

    return URL.create(
        drivername=url.drivername,
        database=f"file:{path.as_posix()}",
        query={"mode": "ro", "uri": "true"},
    )


def _database_app() -> Flask:
    configured_url = resolve_database_url(
        settings.DATABASE_URL,
        project_root=PROJECT_ROOT,
        default_sqlite_path=DEFAULT_DATABASE_PATH,
    )
    readonly_url = _sqlite_readonly_url(configured_url)
    app = Flask("training_export_readonly")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=readonly_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    return app


def _patient_label(index: int) -> str:
    letters = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"Patient {letters}"


def _date_formats(match: re.Match[str], kind: str) -> tuple[date, str] | None:
    groups = match.groupdict()
    try:
        year = int(groups["year"])
        if kind == "iso":
            return date(year, int(groups["month"]), int(groups["day"])), "%Y-%m-%d"
        if kind == "numeric":
            first = int(groups["first"])
            second = int(groups["second"])
            if first > 12 and second <= 12:
                value = date(year, second, first)
                return value, "%d/%m/%Y"
            return date(year, first, second), "%m/%d/%Y"
        if groups.get("month_name"):
            month_name = groups["month_name"].casefold()
            month = MONTH_NUMBERS.get(month_name)
            if month is None:
                return None
            day = int(groups["day"])
            value = date(year, month, day)
            if kind == "month_first":
                fmt = "%B %d, %Y" if "," in match.group(0) else "%B %d %Y"
                if len(groups["month_name"]) <= 4:
                    fmt = "%b %d, %Y" if "," in match.group(0) else "%b %d %Y"
            else:
                fmt = "%d %B %Y" if len(groups["month_name"]) > 4 else "%d %b %Y"
                if "," in match.group(0):
                    fmt = fmt.replace("%Y", ", %Y")
            return value, fmt
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _shift_dates(text: str) -> str:
    matches = []
    for pattern, kind in DATE_PATTERNS:
        matches.extend((match.start(), match.end(), match, kind) for match in pattern.finditer(text))
    accepted = []
    occupied_until = -1
    for start, end, match, kind in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if start < occupied_until:
            continue
        accepted.append((start, end, match, kind))
        occupied_until = end

    shifted = text
    for start, end, match, kind in reversed(accepted):
        def replace_date(match: re.Match[str]) -> str:
            parsed = _date_formats(match, kind)
            if parsed is None:
                return match.group(0)
            parsed_date, output_format = parsed
            return (parsed_date + timedelta(days=DATE_SHIFT_DAYS)).strftime(output_format)

        shifted = shifted[:start] + replace_date(match) + shifted[end:]
    return shifted


def _dob_variants(value: date | None) -> set[str]:
    if value is None:
        return set()
    variants = {value.isoformat(), value.strftime("%m/%d/%Y"), value.strftime("%d/%m/%Y")}
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        variants.add(value.strftime(fmt))
    return variants


def _replace_known_names_and_dobs(
    text: str,
    patients: list[Patient],
    labels: dict[int, str],
) -> str:
    replacements: list[tuple[str, str]] = []
    for patient in patients:
        label = labels[patient.id]
        full_name = f"{patient.first_name} {patient.last_name}".strip()
        if full_name:
            replacements.append((full_name, label))
        if patient.first_name.strip():
            replacements.append((patient.first_name.strip(), label))
        if patient.last_name.strip():
            replacements.append((patient.last_name.strip(), label))
        for dob in _dob_variants(patient.date_of_birth):
            replacements.append((dob, "[REDACTED DOB]"))

    for original, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = re.sub(
            rf"(?<!\w){re.escape(original)}(?!\w)",
            lambda _match, value=replacement: value,
            text,
            flags=re.IGNORECASE,
        )
    return text


SENSITIVE_FIELD_PATTERNS = (
    (
        re.compile(
            r"(?i)\b(?:date\s+of\s+birth|birth\s+date|dob)\b\s*[:#=-]?\s*.*$"
        ),
        "[REDACTED DOB]",
    ),
    (
        re.compile(
            r"(?i)\b(?:patient\s*)?(?:mrn|medical\s+record\s*(?:number|no\.?)|"
            r"patient\s*(?:id|number|no\.?)|record\s*(?:id|number|no\.?)|"
            r"accession\s*(?:number|no\.?)|uhid|health\s*id|ssn|"
            r"social\s+security\s*(?:number|no\.?))\b\s*[:#=-]?\s*.*$"
        ),
        "[REDACTED ID]",
    ),
    (
        re.compile(r"(?i)\b(?:home|street|residential|mailing)?\s*address\b\s*[:#=-]?\s*.*$"),
        "[REDACTED ADDRESS]",
    ),
    (
        re.compile(r"(?i)\b(?:phone|telephone|mobile|cell)\b\s*(?:number)?\s*[:#=-]?\s*.*$"),
        "[REDACTED PHONE]",
    ),
    (
        re.compile(r"(?i)\b(?:e-?mail)\b\s*[:#=-]?\s*.*$"),
        "[REDACTED EMAIL]",
    ),
    (
        re.compile(
            r"(?i)\b(?:attending|referring|ordering|primary\s+care|consulting)?\s*"
            r"(?:doctor|physician|provider|consultant)\b\s*[:#=-]?\s*.*$"
        ),
        "[REDACTED PROVIDER]",
    ),
    (
        re.compile(
            r"(?i)\b(?:hospital|clinic|medical\s+center|health\s+system|"
            r"facility)\b"
        ),
        "[REDACTED FACILITY]",
    ),
    (
        re.compile(r"(?i)\b(?:id|identifier)\s*[:#=-]\s*.*$"),
        "[REDACTED ID]",
    ),
)
INLINE_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s().-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{10,12}(?!\d)"),
    re.compile(r"(?i)\bDr\.?\s+[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,3}"),
    re.compile(r"(?i)\b\d{1,6}\s+[\w.'-]+(?:\s+[\w.'-]+){0,4}\s+(?:street|st\.?|road|rd\.?|avenue|ave\.?|"
               r"boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|court|ct\.?|way)\b[^\n,;]*"),
)


def deidentify_text(
    text: str,
    patients: list[Patient],
    labels: dict[int, str],
) -> str:
    text = _replace_known_names_and_dobs(text, patients, labels)
    output_lines = []
    for line in text.splitlines(keepends=True):
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        content = line[: -len(ending)] if ending else line
        redacted = False
        for pattern, replacement in SENSITIVE_FIELD_PATTERNS:
            match = pattern.search(content)
            if match:
                if replacement == "[REDACTED FACILITY]":
                    content = replacement
                else:
                    content = content[: match.start()] + replacement
                redacted = True
                break
        if not redacted:
            for pattern in INLINE_SENSITIVE_PATTERNS:
                content = pattern.sub("[REDACTED]", content)
        content = _shift_dates(content)
        output_lines.append(content + ending)
    return "".join(output_lines)


def _layout_signature(text: str) -> tuple[int, bool, bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    line_count_band = min(len(lines) // 15, 5)
    has_table = any("|" in line or "\t" in line for line in lines)
    has_section_headings = any(
        re.match(r"^\s*(?:[A-Z][A-Z /-]{3,}|(?:section|panel)\s+\d+)", line)
        for line in lines
    )
    return line_count_band, has_table, has_section_headings


def _report_candidates() -> tuple[list[Patient], list[ReportCandidate]]:
    patients = db.session.query(Patient).order_by(Patient.id.asc()).all()
    candidates = []
    rows = (
        db.session.query(MedicalReport, Patient)
        .join(Patient, MedicalReport.patient_id == Patient.id)
        .filter(MedicalReport.extracted_text.isnot(None))
        .order_by(MedicalReport.id.asc())
        .all()
    )
    for report, patient in rows:
        text = report.extracted_text
        if not isinstance(text, str) or not text.strip():
            continue
        lab_names = frozenset(
            re.sub(r"\s+", " ", parameter.parameter.casefold()).strip()
            for parameter in report.parameters
            if parameter.parameter
        )
        candidates.append(
            ReportCandidate(
                report_id=report.id,
                patient_id=patient.id,
                patient_first_name=patient.first_name,
                patient_last_name=patient.last_name,
                extracted_text=text,
                report_type=report.report_type or "",
                lab_names=lab_names,
                layout_signature=_layout_signature(text),
            )
        )
    return patients, candidates


def _select_diverse_reports(candidates: list[ReportCandidate]) -> list[ReportCandidate]:
    remaining = list(candidates)
    selected: list[ReportCandidate] = []
    seen_labs: set[str] = set()
    seen_layouts: set[tuple[int, bool, bool]] = set()
    seen_types: set[str] = set()
    seen_patients: set[int] = set()
    while remaining and len(selected) < REPORTS_REQUIRED:
        best = max(
            remaining,
            key=lambda candidate: (
                3 * len(candidate.lab_names - seen_labs)
                + 2 * (candidate.layout_signature not in seen_layouts)
                + (candidate.report_type.casefold() not in seen_types)
                + (candidate.patient_id not in seen_patients),
                len(candidate.lab_names),
                -candidate.report_id,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        seen_labs.update(best.lab_names)
        seen_layouts.add(best.layout_signature)
        seen_types.add(best.report_type.casefold())
        seen_patients.add(best.patient_id)
    if len(selected) < REPORTS_REQUIRED:
        raise RuntimeError(
            f"Need {REPORTS_REQUIRED} reports with extracted text; found {len(selected)}."
        )
    return selected


def _first_patient_account() -> tuple[int, Patient]:
    first_owner_id = (
        db.session.query(Patient.created_by_id)
        .group_by(Patient.created_by_id)
        .order_by(Patient.created_by_id.asc())
        .first()
    )
    if first_owner_id is None:
        raise RuntimeError("No account with patient records exists in the database.")
    patient = (
        db.session.query(Patient)
        .filter_by(created_by_id=first_owner_id[0])
        .order_by(Patient.id.asc())
        .first()
    )
    if patient is None:
        raise RuntimeError("The first patient account could not be selected.")
    return first_owner_id[0], patient


def _research_paper_samples() -> list[str]:
    papers = (
        db.session.query(ResearchPaper)
        .order_by(ResearchPaper.id.asc())
        .limit(PAPERS_REQUIRED)
        .all()
    )
    if len(papers) < PAPERS_REQUIRED:
        raise RuntimeError(f"Need {PAPERS_REQUIRED} uploaded research papers; found {len(papers)}.")

    samples = []
    research_root = (PROJECT_ROOT / "instance" / "uploads" / "research_papers").resolve()
    for paper in papers:
        user_id = paper.user_id
        filename = Path(paper.stored_filename)
        if filename.name != paper.stored_filename or filename.suffix.lower() not in {".pdf", ".docx"}:
            raise RuntimeError("An uploaded research paper has an invalid stored file reference.")
        paper_path = (research_root / str(user_id) / filename).resolve()
        if paper_path.parent != (research_root / str(user_id)).resolve() or not paper_path.is_file():
            raise RuntimeError("An uploaded research paper file is unavailable.")
        try:
            extracted = extract_document_text(paper_path.read_bytes(), paper_path.name)
        except Exception as exc:
            raise RuntimeError("Could not extract text from an uploaded research paper.") from exc
        samples.append(extracted[:PAPER_SAMPLE_CHARACTERS])
    return samples


def _format_assistant_prompts(
    user_id: int,
    first_patient: Patient,
    patients: list[Patient],
    labels: dict[int, str],
) -> str:
    questions = (
        "What does this project do?",
        f"What is {first_patient.first_name}'s latest HbA1c?",
        "Which patients have high HbA1c?",
    )
    sections = []
    previous_debug = os.environ.pop("ASSISTANT_DEBUG", None)
    try:
        for number, question in enumerate(questions, start=1):
            user_message = build_own_assistant_prompt(question, user_id)
            user_message = deidentify_text(user_message, patients, labels)
            system_message = deidentify_text(OWN_ASSISTANT_SYSTEM, patients, labels)
            sections.append(
                f"===== QUESTION {number} =====\n"
                f"SYSTEM:\n{system_message}\n\n"
                f"USER:\n{user_message}"
            )
    finally:
        if previous_debug is not None:
            os.environ["ASSISTANT_DEBUG"] = previous_debug
    return "\n\n".join(sections) + "\n"


def _cpu_model_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(value).strip()
        except (ImportError, OSError):
            pass
    return "Unavailable"


def _gpu_names() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except (ImportError, RuntimeError):
        pass
    return []


def _system_information() -> str:
    import psutil

    memory_gb = psutil.virtual_memory().total / (1024**3)
    physical_cores = psutil.cpu_count(logical=False) or "Unavailable"
    logical_cores = psutil.cpu_count(logical=True) or "Unavailable"
    gpu_names = _gpu_names()
    return (
        f"RAM: {memory_gb:.2f} GB\n"
        f"CPU model: {_cpu_model_name()}\n"
        f"CPU cores: {physical_cores} physical, {logical_cores} logical\n"
        f"GPU: {', '.join(gpu_names) if gpu_names else 'None detected'}\n"
        f"Python: {platform.python_version()}\n"
        f"OS: {platform.platform()}\n"
    )


def _validate_no_known_identifiers(
    output_texts: list[str],
    patients: list[Patient],
) -> None:
    for patient in patients:
        full_name = f"{patient.first_name} {patient.last_name}".strip()
        if full_name and re.search(
            rf"(?<!\w){re.escape(full_name)}(?!\w)",
            "\n".join(output_texts),
            re.I,
        ):
            raise RuntimeError(
                "De-identification validation found a patient name in the export."
            )
        for dob in _dob_variants(patient.date_of_birth):
            if dob and dob in "\n".join(output_texts):
                raise RuntimeError(
                    "De-identification validation found a date of birth in the export."
                )


def _write_new_file(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        output.write(content)


def export_training_data() -> dict[str, int]:
    app = _database_app()
    with app.app_context():
        database_engine = db.engine

        @event.listens_for(database_engine, "connect")
        def _set_query_only(connection: Any, _record: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA query_only = ON")
            cursor.close()

        try:
            patients, candidates = _report_candidates()
            reports = _select_diverse_reports(candidates)
            user_id, first_patient = _first_patient_account()
            labels = {
                patient.id: _patient_label(index)
                for index, patient in enumerate(patients)
            }

            report_text = "\n\n".join(
                f"===== REPORT {index} =====\n"
                f"{deidentify_text(report.extracted_text, patients, labels).rstrip()}"
                for index, report in enumerate(reports, start=1)
            ) + "\n"
            paper_samples = _research_paper_samples()
            papers_text = "\n\n".join(
                f"===== PAPER {index} =====\n"
                f"{deidentify_text(sample, patients, labels).rstrip()}"
                for index, sample in enumerate(paper_samples, start=1)
            ) + "\n"
            prompts_text = _format_assistant_prompts(
                user_id,
                first_patient,
                patients,
                labels,
            )
            system_text = _system_information()
            exported_texts = [
                report_text,
                papers_text,
                prompts_text,
            ]
            _validate_no_known_identifiers(exported_texts, patients)
            knowledge_contents = [
                path.read_text(encoding="utf-8")
                for path in (PROJECT_ROOT / "project_kb").rglob("*")
                if path.is_file()
            ]
            _validate_no_known_identifiers(knowledge_contents, patients)

            if OUTPUT_DIRECTORY.exists():
                raise RuntimeError(
                    "export_for_training already exists; refusing to overwrite existing files."
                )
            OUTPUT_DIRECTORY.mkdir()
            try:
                _write_new_file(
                    OUTPUT_DIRECTORY / "reports_deidentified.txt",
                    report_text,
                )
                _write_new_file(
                    OUTPUT_DIRECTORY / "papers_sample.txt",
                    papers_text,
                )
                _write_new_file(
                    OUTPUT_DIRECTORY / "assistant_prompts.txt",
                    prompts_text,
                )
                _write_new_file(
                    OUTPUT_DIRECTORY / "system_info.txt",
                    system_text,
                )
                shutil.copytree(
                    PROJECT_ROOT / "project_kb",
                    OUTPUT_DIRECTORY / "project_kb",
                )
            except Exception:
                shutil.rmtree(OUTPUT_DIRECTORY)
                raise
            return {
                "reports": len(reports),
                "papers": len(paper_samples),
                "prompts": 3,
                "kb_files": sum(
                    1
                    for path in (OUTPUT_DIRECTORY / "project_kb").rglob("*")
                    if path.is_file()
                ),
            }
        finally:
            db.session.remove()
            database_engine.dispose()


def main() -> int:
    try:
        summary = export_training_data()
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Created {OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT)}")
    print(f"De-identified reports: {summary['reports']}")
    print(f"Research paper samples: {summary['papers']}")
    print(f"Assistant prompts: {summary['prompts']}")
    print(f"Copied project knowledge files: {summary['kb_files']}")
    print(f"System information: {(OUTPUT_DIRECTORY / 'system_info.txt').name}")
    print("Review every file by hand before sharing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
