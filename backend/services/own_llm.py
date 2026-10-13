from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.inspection import inspect as sqlalchemy_inspect

from backend.extensions import db
from backend.models import (
    MedicalReport,
    Patient,
    ResearchPaper,
    ScanAsset,
)
from backend.services.patient_context import (
    get_owned_patient_records,
    list_owned_patients,
)


OWN_ASSISTANT_SYSTEM = (
    "You are the built-in assistant of the project 'Multimodal AI Research & "
    "Medical Report Intelligence'. Answer questions about the project using ONLY "
    "the PROJECT KNOWLEDGE provided. For questions about a patient, use ONLY the "
    "PATIENT DATA provided. If the answer is not in the provided context, say you "
    "don't have that information. Never diagnose or prescribe; for personal "
    "medical decisions tell the user to consult a doctor. Be concise."
)
DEFAULT_OWN_N_CTX = 4096
PROJECT_KB_DIRECTORY = Path(__file__).resolve().parents[2] / "project_kb"
_LOGGER = logging.getLogger(__name__)
_MODEL = None
_MODEL_LOAD_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
PATIENT_DATA_MAX_CHARACTERS = 2500
STANDARD_LAB_RANGES = {
    "hemoglobin": {
        "range": (12.0, 17.0),
        "unit": "g/dL",
        "sanity": (4.0, 30.0),
        "aliases": ("hemoglobin", "haemoglobin", "hb", "hgb"),
    },
    "rbc": {
        "range": (3.8, 5.8),
        "unit": "10^6/uL",
        "sanity": (1.0, 10.0),
        "aliases": ("rbc", "r.b.c", "red blood cell count"),
    },
    "wbc": {
        "range": (4.0, 11.0),
        "unit": "10^3/uL",
        "sanity": (0.5, 100.0),
        "aliases": ("wbc", "w.b.c", "wec", "w.e.c", "white blood cell count"),
    },
    "platelets": {
        "range": (150.0, 450.0),
        "unit": "10^3/uL",
        "sanity": (10.0, 2000.0),
        "aliases": ("platelets", "platelet count", "plt"),
    },
    "hematocrit": {
        "range": (36.0, 52.0),
        "unit": "%",
        "sanity": (10.0, 80.0),
        "aliases": ("hematocrit", "haematocrit", "hct", "pcv", "packed cell volume"),
    },
    "fasting glucose": {
        "range": (70.0, 99.0),
        "unit": "mg/dL",
        "sanity": (10.0, 1000.0),
        "aliases": ("fasting glucose", "fasting blood sugar", "fasting blood glucose", "fbs"),
    },
    "post-prandial glucose": {
        "range": (70.0, 140.0),
        "unit": "mg/dL",
        "sanity": (10.0, 1000.0),
        "aliases": (
            "post prandial glucose", "postprandial glucose", "post prandial blood sugar",
            "postprandial blood sugar", "pp blood sugar", "ppbs", "pp glucose",
        ),
    },
    "HbA1c": {
        "range": (4.0, 5.6),
        "unit": "%",
        "sanity": (2.0, 20.0),
        "aliases": (
            "hba1c", "hbalc", "hbaic", "hba1c", "hb a1c", "hb a1c",
            "hemoglobin a1c", "haemoglobin a1c", "glycated hemoglobin",
            "glycosylated hemoglobin",
        ),
    },
    "total cholesterol": {
        "range": (0.0, 200.0),
        "unit": "mg/dL",
        "sanity": (0.0, 1000.0),
        "aliases": ("total cholesterol", "cholesterol total", "serum cholesterol", "cholesterol"),
    },
    "LDL cholesterol": {
        "range": (0.0, 100.0),
        "unit": "mg/dL",
        "sanity": (0.0, 500.0),
        "aliases": ("ldl cholesterol", "ldl-c", "ldl", "low density lipoprotein"),
    },
    "HDL cholesterol": {
        "range": (40.0, 60.0),
        "unit": "mg/dL",
        "sanity": (2.0, 200.0),
        "aliases": ("hdl cholesterol", "hdl-c", "hdl", "high density lipoprotein"),
    },
    "triglycerides": {
        "range": (0.0, 150.0),
        "unit": "mg/dL",
        "sanity": (0.0, 5000.0),
        "aliases": ("triglycerides", "triglyceride", "trigs", "tg"),
    },
    "ALT/SGPT": {
        "range": (7.0, 56.0),
        "unit": "U/L",
        "sanity": (0.0, 5000.0),
        "aliases": ("sgpt", "alt", "alanine aminotransferase", "alanine transaminase"),
    },
    "AST/SGOT": {
        "range": (10.0, 40.0),
        "unit": "U/L",
        "sanity": (0.0, 5000.0),
        "aliases": ("sgot", "ast", "aspartate aminotransferase", "aspartate transaminase"),
    },
    "creatinine": {
        "range": (0.6, 1.3),
        "unit": "mg/dL",
        "sanity": (0.05, 30.0),
        "aliases": ("creatinine", "serum creatinine", "creat"),
    },
    "urea": {
        "range": (7.0, 20.0),
        "unit": "mg/dL",
        "sanity": (1.0, 300.0),
        "aliases": ("urea", "serum urea"),
    },
    "vitamin D": {
        "range": (20.0, 50.0),
        "unit": "ng/mL",
        "sanity": (1.0, 250.0),
        "aliases": ("vitamin d", "25 hydroxy vitamin d", "25 oh vitamin d", "25-oh vitamin d", "25 oh d"),
    },
    "TSH": {
        "range": (0.4, 4.0),
        "unit": "uIU/mL",
        "sanity": (0.001, 200.0),
        "aliases": ("tsh", "thyroid stimulating hormone"),
    },
    "sodium": {
        "range": (135.0, 145.0),
        "unit": "mmol/L",
        "sanity": (80.0, 200.0),
        "aliases": ("sodium", "serum sodium", "na"),
    },
    "potassium": {
        "range": (3.5, 5.1),
        "unit": "mmol/L",
        "sanity": (1.0, 15.0),
        "aliases": ("potassium", "serum potassium", "k"),
    },
    "MCV": {
        "range": (80.0, 100.0),
        "unit": "fL",
        "sanity": (30.0, 200.0),
        "aliases": ("mcv", "mean corpuscular volume"),
    },
    "MCH": {
        "range": (27.0, 33.0),
        "unit": "pg",
        "sanity": (5.0, 100.0),
        "aliases": ("mch", "mean corpuscular hemoglobin", "mean corpuscular haemoglobin"),
    },
    "MCHC": {
        "range": (32.0, 36.0),
        "unit": "g/dL",
        "sanity": (10.0, 80.0),
        "aliases": ("mchc", "mean corpuscular hemoglobin concentration"),
    },
    "uric acid": {
        "range": (2.4, 7.0),
        "unit": "mg/dL",
        "sanity": (0.1, 40.0),
        "aliases": ("uric acid", "serum uric acid", "urate"),
    },
    "BUN": {
        "range": (7.0, 20.0),
        "unit": "mg/dL",
        "sanity": (1.0, 300.0),
        "aliases": ("bun", "blood urea nitrogen"),
    },
    "calcium": {
        "range": (8.5, 10.5),
        "unit": "mg/dL",
        "sanity": (2.0, 30.0),
        "aliases": ("calcium", "serum calcium", "ca"),
    },
}
LAB_DISPLAY_NAMES = {
    "hemoglobin": "Hemoglobin",
    "rbc": "RBC",
    "wbc": "WBC",
    "platelets": "Platelets",
    "hematocrit": "Hematocrit",
    "fasting glucose": "Fasting glucose",
    "post-prandial glucose": "Post-prandial glucose",
    "HbA1c": "HbA1c",
    "total cholesterol": "Total cholesterol",
    "LDL cholesterol": "LDL cholesterol",
    "HDL cholesterol": "HDL cholesterol",
    "triglycerides": "Triglycerides",
    "ALT/SGPT": "ALT/SGPT",
    "AST/SGOT": "AST/SGOT",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "vitamin D": "Vitamin D",
    "TSH": "TSH",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "MCV": "MCV",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "uric acid": "Uric acid",
    "BUN": "BUN",
    "calcium": "Calcium",
}
_SCAN_ANALYSIS_RELATIONSHIPS = (
    "cardiac_analysis",
    "knee_analysis",
    "spine_analysis",
    "brain_analysis",
    "chest_xray_analysis",
    "bone_xray_analysis",
    "dental_xray_analysis",
    "ct_head_analysis",
    "chest_ct_analysis",
    "obstetric_ultrasound_analysis",
    "abdominal_aorta_ultrasound_analysis",
    "echoview47_analysis",
    "vascular_ultrasound_analysis",
)


def _load_model():
    global _MODEL

    with _MODEL_LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL

        model_path = os.environ.get("OWN_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError("Own model is not configured. Set OWN_MODEL_PATH in .env.")
        if not Path(model_path).is_file():
            raise RuntimeError(
                f"Own model file not found at {model_path}. "
                "Check OWN_MODEL_PATH in .env."
            )

        try:
            from llama_cpp import Llama
            from psutil import cpu_count
        except ImportError as exc:
            if exc.name == "llama_cpp":
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Run: python -m pip install llama-cpp-python"
                ) from exc
            raise RuntimeError(f"Own model dependencies could not be imported: {exc}") from exc

        try:
            n_ctx = int(os.environ.get("OWN_N_CTX", str(DEFAULT_OWN_N_CTX)))
        except ValueError as exc:
            raise RuntimeError("OWN_N_CTX must be a positive integer.") from exc
        if n_ctx < 1:
            raise RuntimeError("OWN_N_CTX must be a positive integer.")

        try:
            physical_cores = max(1, cpu_count(logical=False) or 1)
            _MODEL = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=physical_cores,
                n_threads_batch=physical_cores,
                use_mmap=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Own model could not be loaded: {exc}") from exc
        return _MODEL


def _chat_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def own_generate(
    prompt: str,
    max_tokens: int,
    system: str | None = None,
    json_mode: bool = False,
) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if system is not None and not isinstance(system, str):
        raise ValueError("system must be a string or None")
    if not isinstance(json_mode, bool):
        raise ValueError("json_mode must be a boolean")

    return _generate(
        prompt,
        max_tokens,
        system=system,
        json_mode=json_mode,
    )


def own_generate_assistant(
    prompt: str,
    max_tokens: int,
    system: str | None = None,
) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    return _generate(
        prompt,
        max_tokens,
        system=system,
        repeat_penalty=1.1,
        stop=["<|im_end|>"],
    )


def _generate(
    prompt: str,
    max_tokens: int,
    *,
    system: str | None = None,
    json_mode: bool = False,
    repeat_penalty: float | None = None,
    stop: list[str] | None = None,
) -> str:
    messages = _chat_messages(prompt, system)
    with _GENERATION_LOCK:
        llm = _load_model()
        prompt_tokens = _chat_prompt_token_count(llm, messages)
        n_ctx = llm.n_ctx()
        if prompt_tokens + max_tokens > n_ctx:
            raise RuntimeError(
                "Prompt too long for the own model "
                f"({prompt_tokens + max_tokens} > {n_ctx} tokens)"
            )
        try:
            generation_options = {
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if repeat_penalty is not None:
                generation_options["repeat_penalty"] = repeat_penalty
            if stop is not None:
                generation_options["stop"] = stop
            if json_mode:
                generation_options["response_format"] = {"type": "json_object"}
            result = llm.create_chat_completion(**generation_options)
            content = result["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Own model generation failed: {exc}") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Own model returned an empty response")
    return content.strip()


def _chat_prompt_token_count(llm, messages: list[dict[str, str]]) -> int:
    handler = getattr(llm, "chat_handler", None)
    if handler is None:
        handlers = getattr(llm, "_chat_handlers", {})
        handler = handlers.get(getattr(llm, "chat_format", None))
    if handler is None:
        from llama_cpp import llama_chat_format

        handler = llama_chat_format.get_chat_completion_handler(llm.chat_format)

    formatter = next(
        (
            cell.cell_contents
            for cell in (getattr(handler, "__closure__", None) or ())
            if callable(cell.cell_contents)
            and not hasattr(cell.cell_contents, "__code__")
        ),
        None,
    )
    if formatter is not None:
        formatted = formatter(messages=messages)
    else:
        formatted = handler(llama=llm, messages=messages)
    rendered_prompt = getattr(formatted, "prompt", None)
    if not isinstance(rendered_prompt, str):
        raise RuntimeError("Could not count tokens for the own model prompt.")
    tokens = llm.tokenize(
        rendered_prompt.encode("utf-8"),
        add_bos=not getattr(formatted, "added_special", False),
        special=True,
    )
    return len(tokens)


def _split_knowledge_chunks(text: str, chunk_size: int = 700) -> list[str]:
    chunks = []
    heading = ""
    section_lines = []

    def flush_section() -> None:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            return
        paragraphs = re.split(r"\n\s*\n", section_text)
        content = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            prefix = f"{heading}\n" if heading else ""
            available = max(1, chunk_size - len(prefix))
            if len(paragraph) <= available:
                if content and len(content) + len(paragraph) + 2 <= available:
                    content = f"{content}\n\n{paragraph}"
                else:
                    if content:
                        chunks.append(f"{prefix}{content}".strip())
                    content = paragraph
                continue

            if content:
                chunks.append(f"{prefix}{content}".strip())
                content = ""
            start = 0
            while start < len(paragraph):
                end = min(start + available, len(paragraph))
                if end < len(paragraph):
                    boundary = paragraph.rfind(" ", start + available // 2, end)
                    if boundary > start:
                        end = boundary
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(f"{prefix}{piece}".strip())
                if end >= len(paragraph):
                    break
                start = max(end - 80, start + 1)
        if content:
            chunks.append(f"{heading}\n{content}".strip() if heading else content)

    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            flush_section()
            heading = line.strip()
            section_lines = []
        else:
            section_lines.append(line)
    flush_section()
    return chunks


def _retrieve_project_knowledge(
    question: str,
    user_id: int,
    *,
    top_k: int = 3,
    min_score: float = 0.0,
    overview_first: bool = True,
    append_dynamic_counts: bool = True,
) -> str:
    if not PROJECT_KB_DIRECTORY.is_dir():
        raise RuntimeError(f"Project knowledge directory was not found: {PROJECT_KB_DIRECTORY}")

    overview_path = PROJECT_KB_DIRECTORY / "project_overview.md"
    if not overview_path.is_file():
        raise RuntimeError(f"Project overview knowledge file was not found: {overview_path}")

    overview_chunks = []
    try:
        overview_chunks = _split_knowledge_chunks(
            overview_path.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not read project knowledge file {overview_path.name}: {exc}"
        ) from exc
    if not overview_chunks:
        raise RuntimeError("The project overview knowledge file is empty.")

    other_chunks = []
    for path in sorted(PROJECT_KB_DIRECTORY.iterdir()):
        if path == overview_path:
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            chunks = _split_knowledge_chunks(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Could not read project knowledge file {path.name}: {exc}") from exc
        if path.name.casefold() == "how_to_use.md":
            chunks = [
                chunk
                for chunk in chunks
                if not re.search(
                    r"(?i)\b(?:git\s+lfs\s+install|git\s+clone|git\s+lfs\s+pull|"
                    r"python\s+-m\s+venv|activate\.ps1|pip\s+install|"
                    r"virtual\s+environment|install\s+dependencies|"
                    r"copy-item\s+\.env|create\s+and\s+activate)\b",
                    chunk,
                )
            ]
        elif path.name.casefold() == "notes.md":
            chunks = [
                chunk
                for chunk in chunks
                if not _is_notes_template_chunk(chunk)
            ]
        other_chunks.extend(chunks)

    ranked_source_chunks = (
        overview_chunks + other_chunks if not overview_first else other_chunks
    )
    selected_chunks = overview_chunks[:1] if overview_first else []
    if ranked_source_chunks:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        try:
            document_vectors = vectorizer.fit_transform(ranked_source_chunks)
            question_vector = vectorizer.transform([question])
            scores = (document_vectors @ question_vector.T).toarray().ravel()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            scores = []
        if len(scores):
            ranked_indexes = sorted(
                range(len(scores)),
                key=lambda index: (-scores[index], index),
            )
            selected_chunks.extend(
                ranked_source_chunks[index]
                for index in ranked_indexes
                if scores[index] >= min_score and scores[index] > 0
                and len(selected_chunks) < top_k + (1 if overview_first else 0)
            )
    knowledge = "\n---\n".join(selected_chunks)
    question_folded = question.casefold()
    if (
        append_dynamic_counts
        and
        re.search(r"\b(how many|number of|count|total)\b", question_folded)
        and re.search(r"\breports?\b", question_folded)
    ):
        report_count = (
            db.session.query(func.count(MedicalReport.id))
            .join(Patient, MedicalReport.patient_id == Patient.id)
            .filter(Patient.created_by_id == user_id)
            .scalar()
        ) or 0
        report_label = "report" if report_count == 1 else "reports"
        knowledge = (
            f"{knowledge}\n---\n"
            f"Current account data: your account has {report_count} uploaded "
            f"medical {report_label} across patients you own."
        )
    return knowledge


def retrieve_project_knowledge(
    question: str,
    user_id: int,
    *,
    top_k: int = 2,
    min_score: float = 0.08,
) -> str:
    return _retrieve_project_knowledge(
        question,
        user_id,
        top_k=top_k,
        min_score=min_score,
        overview_first=False,
        append_dynamic_counts=False,
    )


def _is_notes_template_chunk(chunk: str) -> bool:
    lines = [
        line.strip()
        for line in chunk.splitlines()
        if line.strip() and not re.match(r"^#{1,6}\s+", line.strip())
    ]
    return bool(lines) and all(
        line.casefold().startswith("add stable project facts here")
        or line.casefold().startswith("this file is intentionally not generated")
        for line in lines
    )


def _patient_matches_question(patient: Patient, question: str) -> bool:
    question_folded = question.casefold()
    full_name = f"{patient.first_name} {patient.last_name}"
    return any(
        re.search(
            rf"(?<!\w){re.escape(name.casefold())}(?!\w)",
            question_folded,
        )
        for name in (full_name, patient.first_name)
        if name.strip()
    )


def _scan_summary_line(scan: ScanAsset) -> str:
    study_date = scan.study_date.isoformat() if scan.study_date else "date unavailable"
    region = f" {scan.body_region}" if scan.body_region else ""
    result_parts = []
    for relationship_name in _SCAN_ANALYSIS_RELATIONSHIPS:
        analysis = getattr(scan, relationship_name)
        if analysis is None:
            continue
        values = [
            f"{attribute.key.replace('_', ' ')}: {getattr(analysis, attribute.key)}"
            for attribute in sqlalchemy_inspect(type(analysis)).column_attrs
            if attribute.key not in {"id", "scan_id", "created_at", "model_name"}
            and not attribute.key.endswith("_file_reference")
            and getattr(analysis, attribute.key) is not None
        ]
        if values:
            result_parts.extend(values[:3])
        break
    summary = f"; {'; '.join(result_parts)}" if result_parts else ""
    return (
        f"Scan ({study_date}): {scan.modality}{region}, "
        f"{scan.processing_status}{summary}"
    )


def _normalize_test_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return {
        "wec": "wbc",
        "hbalc": "hba1c",
        "hbaic": "hba1c",
        "hb1ac": "hba1c",
        "ha1c": "hba1c",
    }.get(normalized, normalized)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    characters = [character for character in alias.casefold() if character.isalnum()]
    pattern = r"[\W_]*".join(re.escape(character) for character in characters)
    return re.compile(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", re.IGNORECASE)


_LAB_ALIAS_PATTERNS = sorted(
    (
        (_alias_pattern(alias), canonical)
        for canonical, specification in STANDARD_LAB_RANGES.items()
        for alias in specification["aliases"]
    ),
    key=lambda entry: len(entry[0].pattern),
    reverse=True,
)
_LAB_NUMBER_PATTERN = re.compile(r"(?<![a-z])[-+]?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
_LAB_RANGE_PATTERN = re.compile(
    r"(?<![a-z0-9])(?P<lower>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:-|–|—|\bto\b)\s*"
    r"(?P<upper>-?\d[\d,]*(?:\.\d+)?)(?![a-z0-9])",
    re.IGNORECASE,
)
_LAB_UNIT_PATTERN = re.compile(
    r"(?i)(?:mg\s*/\s*d[l1]|mmol\s*/\s*l|g\s*/\s*d[l1]|g\s*/\s*l|"
    r"ng\s*/\s*ml|pg|fl|u\s*/\s*l|iu\s*/\s*ml|uiu\s*/\s*ml|"
    r"10\s*\^?\s*[369]\s*/\s*(?:ul|µl|μl|l)|"
    r"(?:lakh|lac)s?\s*/\s*(?:ul|µl|μl|cumm)|"
    r"/\s*(?:ul|µl|μl)|%|umol\s*/\s*l)"
)


def _canonical_test_name(value: str) -> str | None:
    normalized_value = _normalize_test_name(value)
    for canonical, specification in STANDARD_LAB_RANGES.items():
        aliases = (canonical, *specification["aliases"])
        if normalized_value in {_normalize_test_name(alias) for alias in aliases}:
            return canonical
    return None


def _unit_key(unit: str | None) -> str:
    if not unit:
        return ""
    return (
        unit.casefold()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("l1", "l")
        .replace("litres", "l")
        .replace("liter", "l")
        .replace(" ", "")
        .replace("·", "")
        .replace("10^", "10")
        .replace("×", "x")
    )


def _convert_lab_value(canonical: str, value: float, unit: str | None) -> float | None:
    key = _unit_key(unit)
    if not key:
        return value
    conversions: dict[str, dict[str, tuple[str, float]]] = {
        "hemoglobin": {"g/l": ("g/dl", 0.1)},
        "rbc": {"1012/l": ("106/ul", 1.0), "106/ul": ("106/ul", 1.0)},
        "wbc": {
            "109/l": ("103/ul", 1.0),
            "103/ul": ("103/ul", 1.0),
            "/ul": ("103/ul", 0.001),
        },
        "platelets": {
            "109/l": ("103/ul", 1.0),
            "103/ul": ("103/ul", 1.0),
            "/ul": ("103/ul", 0.001),
            "lakh/ul": ("103/ul", 100.0),
            "lac/ul": ("103/ul", 100.0),
            "lakh/cumm": ("103/ul", 100.0),
            "lac/cumm": ("103/ul", 100.0),
        },
        "fasting glucose": {"mmol/l": ("mg/dl", 18.0182)},
        "post-prandial glucose": {"mmol/l": ("mg/dl", 18.0182)},
        "HbA1c": {"mmol/mol": ("%", 1.0)},
        "total cholesterol": {"mmol/l": ("mg/dl", 38.67)},
        "LDL cholesterol": {"mmol/l": ("mg/dl", 38.67)},
        "HDL cholesterol": {"mmol/l": ("mg/dl", 38.67)},
        "triglycerides": {"mmol/l": ("mg/dl", 88.57)},
        "ALT/SGPT": {"ukat/l": ("u/l", 60.0), "iu/l": ("u/l", 1.0)},
        "AST/SGOT": {"ukat/l": ("u/l", 60.0), "iu/l": ("u/l", 1.0)},
        "creatinine": {"umol/l": ("mg/dl", 1 / 88.4)},
        "urea": {"mmol/l": ("mg/dl", 6.006)},
        "vitamin D": {"nmol/l": ("ng/ml", 1 / 2.496)},
        "TSH": {"miu/l": ("uiu/ml", 1.0), "uiu/ml": ("uiu/ml", 1.0)},
        "sodium": {"meq/l": ("mmol/l", 1.0)},
        "potassium": {"meq/l": ("mmol/l", 1.0)},
        "calcium": {"mmol/l": ("mg/dl", 4.008)},
    }
    canonical_unit = _unit_key(STANDARD_LAB_RANGES[canonical]["unit"])
    if key == canonical_unit:
        return value
    conversion = conversions.get(canonical, {}).get(key)
    if conversion is None:
        return None
    destination, factor = conversion
    if destination != canonical_unit:
        return None
    return value * factor


def _printed_reference_range(source: str, canonical: str, unit: str | None) -> tuple[float, float] | None:
    specification = STANDARD_LAB_RANGES[canonical]
    sanity_low, sanity_high = specification["sanity"]
    for match in reversed(list(_LAB_RANGE_PATTERN.finditer(source or ""))):
        try:
            lower = float(match.group("lower").replace(",", ""))
            upper = float(match.group("upper").replace(",", ""))
        except ValueError:
            continue
        lower = _convert_lab_value(canonical, lower, unit)
        upper = _convert_lab_value(canonical, upper, unit)
        if (
            lower is not None
            and upper is not None
            and lower < upper
            and sanity_low <= lower <= sanity_high
            and sanity_low <= upper <= sanity_high
        ):
            return lower, upper
    return None


def _classify_lab_value(
    canonical: str,
    raw_value: str,
    unit: str | None,
    source: str,
) -> dict[str, str] | None:
    number_match = _LAB_NUMBER_PATTERN.search(raw_value or "")
    if number_match is None:
        return None
    try:
        numeric_value = float(number_match.group(0).replace(",", ""))
    except ValueError:
        return None
    converted_value = _convert_lab_value(canonical, numeric_value, unit)
    if converted_value is None:
        return None
    value_range = _printed_reference_range(source, canonical, unit)
    if value_range is None:
        value_range = STANDARD_LAB_RANGES[canonical]["range"]
    lower, upper = value_range
    status = "low" if converted_value < lower else "high" if converted_value > upper else "normal"
    return {
        "canonical": canonical,
        "test": LAB_DISPLAY_NAMES.get(canonical, canonical),
        "value": number_match.group(0),
        "unit": unit.strip() if unit and unit.strip() else STANDARD_LAB_RANGES[canonical]["unit"],
        "status": status,
    }


def _lab_results_for_report(report: MedicalReport) -> list[dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for parameter in report.parameters:
        canonical = _canonical_test_name(parameter.parameter or "")
        if canonical is None or canonical in results:
            continue
        result = _classify_lab_value(
            canonical,
            parameter.value or "",
            parameter.unit,
            parameter.source or "",
        )
        if result is not None:
            results[canonical] = result

    for line in (report.extracted_text or "").splitlines():
        matches = []
        for pattern, canonical in _LAB_ALIAS_PATTERNS:
            match = pattern.search(line)
            if match:
                matches.append((match.start(), -(match.end() - match.start()), match, canonical))
        if not matches:
            continue
        _, _, label_match, canonical = min(matches, key=lambda item: (item[0], item[1]))
        if canonical in results:
            continue

        suffix = line[label_match.end() :]
        numeric_matches = list(_LAB_NUMBER_PATTERN.finditer(suffix))
        range_matches = list(_LAB_RANGE_PATTERN.finditer(suffix))
        value_match = next(
            (
                match
                for match in numeric_matches
                if not any(
                    reference.start() <= match.start() < reference.end()
                    for reference in range_matches
                )
            ),
            None,
        )
        if value_match is None:
            continue
        range_start = min((match.start() for match in range_matches), default=len(suffix))
        between_value_and_range = suffix[value_match.end() : range_start]
        unit_match = _LAB_UNIT_PATTERN.search(between_value_and_range)
        if unit_match is None:
            unit_match = _LAB_UNIT_PATTERN.search(suffix)
        unit = unit_match.group(0) if unit_match else ""
        result = _classify_lab_value(
            canonical,
            value_match.group(0),
            unit,
            suffix,
        )
        if result is not None:
            results[canonical] = result
    return list(results.values())


def _latest_lab_report(
    reports: list[MedicalReport],
) -> tuple[MedicalReport | None, list[dict[str, str]]]:
    for report in reports:
        lab_results = _lab_results_for_report(report)
        if len(lab_results) >= 3:
            return report, lab_results
    return None, []


def _report_date(report: MedicalReport | None) -> str:
    return report.report_date.isoformat() if report and report.report_date else "date unavailable"


def _format_lab_lines(results: list[dict[str, str]]) -> list[str]:
    return [
        f"{result['test']}: {result['value']} {result['unit']} ({result['status']})"
        for result in results
    ]


def _requested_abnormal_test(question: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(high|low|abnormal)\b(?:\s+levels?\s+of)?\s+(.+?)(?:[?.!,]|$)",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    test_text = re.sub(
        r"\s+(?:levels?|values?|tests?|results?)\s*$",
        "",
        match.group(2),
        flags=re.IGNORECASE,
    ).strip()
    canonical = _canonical_test_name(test_text)
    return (match.group(1).casefold(), canonical) if canonical else None


def _patient_data_for_question(question: str, user_id: int) -> str:
    patients = list_owned_patients(user_id)
    question_folded = question.casefold()
    named_patients = [
        patient for patient in patients if _patient_matches_question(patient, question)
    ][:2]
    named_patient_ids = {patient.id for patient in named_patients}
    abnormal_query = _requested_abnormal_test(question)
    records_needed = named_patient_ids | (
        {patient.id for patient in patients} if abnormal_query else set()
    )
    records_by_patient_id = {}
    for patient in patients:
        if patient.id not in records_needed:
            continue
        records = get_owned_patient_records(
            user_id,
            patient.id,
            scan_limit=2 if patient.id in named_patient_ids else 0,
        )
        if records is not None:
            records_by_patient_id[patient.id] = records
    patient_lines = []
    optional_patient_lines = []

    for patient in named_patients:
        patient_records = records_by_patient_id.get(patient.id)
        if patient_records is None:
            continue
        reports = patient_records["reports"]
        lab_report, lab_results = _latest_lab_report(reports)
        if lab_report is not None:
            patient_lines.append(
                f"{patient.first_name} {patient.last_name}, latest report "
                f"({_report_date(lab_report)}):"
            )
            patient_lines.extend(_format_lab_lines(lab_results))
        else:
            latest_document = reports[0] if reports else None
            report_type = (
                latest_document.report_type or "unknown"
                if latest_document
                else "unknown"
            )
            patient_lines.append(
                "(no lab values found in this patient's reports; "
                f"latest document is {report_type}, {_report_date(latest_document)})"
            )

        scans = patient_records["scans"]
        optional_patient_lines.extend(_scan_summary_line(scan) for scan in scans)

    if re.search(r"\b(how many|number of)\s+patients?\b|\bcount\s+patients?\b", question_folded):
        names = ", ".join(f"{patient.first_name} {patient.last_name}" for patient in patients)
        patient_lines.append(f"Patient count: {len(patients)}.")
        patient_lines.append(f"Patients: {names or 'none'}.")
    elif re.search(r"\blist\s+(?:(?:all|my)\s+)?patients?\b", question_folded):
        names = ", ".join(f"{patient.first_name} {patient.last_name}" for patient in patients)
        patient_lines.append(f"Patient count: {len(patients)}.")
        patient_lines.append(f"Patients: {names or 'none'}.")

    if re.search(r"\b(?:research\s+)?papers?\b", question_folded) and re.search(
        r"\b(uploaded|upload|list|show|have|my|which|what)\b",
        question_folded,
    ):
        papers = (
            db.session.query(ResearchPaper.title)
            .filter_by(user_id=user_id)
            .order_by(ResearchPaper.created_at.desc(), ResearchPaper.id.desc())
            .limit(100)
            .all()
        )
        paper_titles = [title for (title,) in papers]
        patient_lines.append(
            f"Uploaded research papers ({len(paper_titles)}): "
            f"{'; '.join(paper_titles) if paper_titles else 'none'}."
        )

    if abnormal_query:
        requested_direction, canonical_test = abnormal_query
        abnormal_lines = []
        for patient in patients:
            patient_records = records_by_patient_id.get(patient.id)
            if patient_records is None:
                continue
            reports = patient_records["reports"]
            latest_report, report_results = _latest_lab_report(reports)
            if latest_report is None:
                continue
            for result in report_results:
                if result["canonical"] != canonical_test:
                    continue
                status = result["status"]
                if status == "normal":
                    continue
                if requested_direction != "abnormal" and status != requested_direction:
                    continue
                abnormal_lines.append(
                    f"{patient.first_name} {patient.last_name}: {result['test']} "
                    f"{result['value']} {result['unit']} ({status})"
                )
                if len(abnormal_lines) == 15:
                    break
            if len(abnormal_lines) == 15:
                break
        patient_lines.extend(
            abnormal_lines
            or [f"No high or low {canonical_test} results were found in latest lab reports."]
        )

    result_lines = patient_lines + optional_patient_lines[: len(named_patients) * 2]
    result = "\n".join(result_lines) if result_lines else "(none)"
    if len(result) > PATIENT_DATA_MAX_CHARACTERS:
        result = result[: PATIENT_DATA_MAX_CHARACTERS - 3].rstrip() + "..."
    return result


def format_own_assistant_patient_context(context: dict) -> str:
    context_type = context.get("context_type")
    sources = context.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Selected patient context must contain a source list.")

    if context_type == "report":
        if not sources:
            raise ValueError("Selected report context is empty.")
        source = sources[0]
        patient_name = source.get("patient_name", "Patient")
        report_date = source.get("report_date") or "date unavailable"
        report_text = source.get("extracted_text")
        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("Selected report has no extracted text.")
        patient_data = (
            f"{patient_name}, selected report ({report_date}):\n{report_text.strip()}"
        )
    elif context_type == "patient_timeline":
        patient_name = context.get("patient_name", "Patient")
        report_sources = [
            source
            for source in sources
            if source.get("type") == "report"
            and isinstance(source.get("extracted_text"), str)
            and source["extracted_text"].strip()
        ][:3]
        scan_sources = [
            source for source in sources if source.get("type") == "scan"
        ][:2]
        patient_lines = []
        if report_sources:
            latest = report_sources[0]
            report_date = latest.get("report_date") or "date unavailable"
            patient_lines.append(
                f"{patient_name}, latest report ({report_date}):\n"
                f"{latest['extracted_text'].strip()}"
            )
            for source in report_sources[1:]:
                report_date = source.get("report_date") or "date unavailable"
                report_type = source.get("report_type") or "Medical report"
                patient_lines.append(
                    f"Older report ({report_date}, {report_type}): "
                    f"{source['extracted_text'].strip()}"
                )
        else:
            patient_lines.append("(no extracted report text found)")

        for source in scan_sources:
            scan = db.session.get(ScanAsset, source.get("scan_id"))
            if scan is not None:
                patient_lines.append(_scan_summary_line(scan))

        patient_data = "\n".join(patient_lines)
    else:
        raise ValueError("Selected patient context must be a report or timeline.")

    if len(patient_data) > PATIENT_DATA_MAX_CHARACTERS:
        patient_data = (
            patient_data[: PATIENT_DATA_MAX_CHARACTERS - 3].rstrip() + "..."
        )
    return patient_data


def build_own_assistant_prompt(
    question: str,
    user_id: int,
    web_search_results: list[dict[str, str]] | None = None,
    patient_data: str | None = None,
    include_project_knowledge: bool = True,
    project_knowledge: str | None = None,
) -> str:
    if project_knowledge is None:
        project_knowledge = (
            _retrieve_project_knowledge(question, user_id)
            if include_project_knowledge
            else "(none)"
        )
    if web_search_results:
        formatted_results = "\n".join(
            f"[W{index}] {result['title']}\n"
            f"Snippet: {result['snippet']}\n"
            f"URL: {result['link']}"
            for index, result in enumerate(web_search_results, start=1)
        )
        project_knowledge = (
            f"{project_knowledge}\n\n"
            "WEB SEARCH RESULTS (use as external context when relevant):\n"
            f"{formatted_results}"
        )
    if patient_data is None:
        patient_data = _patient_data_for_question(question, user_id)
    final_prompt = (
        f"PROJECT KNOWLEDGE:\n{project_knowledge}\n\n"
        f"PATIENT DATA:\n{patient_data}\n\n"
        f"QUESTION: {question}"
    )
    if os.environ.get("ASSISTANT_DEBUG", "").strip() == "1":
        _LOGGER.info("Own-model Assistant final prompt:\n%s", final_prompt)
    return final_prompt
