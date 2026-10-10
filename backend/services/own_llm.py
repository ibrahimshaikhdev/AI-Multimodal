from __future__ import annotations

import json
import os
import re
import socket
import threading
from collections import Counter
from math import log
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.extensions import db
from backend.models import MedicalReport, Patient


OWN_ASSISTANT_SYSTEM = (
    "You are the built-in assistant of the project 'Multimodal AI Research & "
    "Medical Report Intelligence'. Answer questions about the project using ONLY "
    "the PROJECT KNOWLEDGE provided. For questions about a patient, use ONLY the "
    "PATIENT DATA provided. If the answer is not in the provided context, say you "
    "don't have that information. Never diagnose or prescribe; for personal "
    "medical decisions tell the user to consult a doctor. Be concise."
)
OWN_LLM_TIMEOUT_SECONDS = 600
MAX_OWN_LLM_RESPONSE_BYTES = 2 * 1024 * 1024
PROJECT_KB_DIRECTORY = Path(__file__).resolve().parents[2] / "project_kb"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_MODEL = None
_MODEL_LOAD_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()


def _load_model():
    global _MODEL

    with _MODEL_LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL

        model_path = os.environ.get("OWN_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError(
                "Own model is not configured (set OWN_MODEL_PATH or OWN_LLM_URL)"
            )
        if not Path(model_path).is_file():
            raise RuntimeError(f"Own model file was not found: {model_path}")

        try:
            from llama_cpp import Llama
            from psutil import cpu_count

            physical_cores = cpu_count(logical=False) or 1
            _MODEL = Llama(
                model_path=model_path,
                n_ctx=16384,
                n_threads=physical_cores,
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
) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if system is not None and not isinstance(system, str):
        raise ValueError("system must be a string or None")

    model_path = os.environ.get("OWN_MODEL_PATH", "").strip()
    model_url = os.environ.get("OWN_LLM_URL", "").strip()
    messages = _chat_messages(prompt, system)
    if model_path:
        with _GENERATION_LOCK:
            try:
                result = _load_model().create_chat_completion(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                content = result["choices"][0]["message"]["content"]
            except Exception as exc:
                raise RuntimeError(f"Own model generation failed: {exc}") from exc
    elif model_url:
        payload = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        request = Request(
            model_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with _GENERATION_LOCK:
            try:
                with urlopen(request, timeout=OWN_LLM_TIMEOUT_SECONDS) as response:
                    response_body = response.read(MAX_OWN_LLM_RESPONSE_BYTES + 1)
                if len(response_body) > MAX_OWN_LLM_RESPONSE_BYTES:
                    raise ValueError("response exceeded the 2 MiB limit")
                result = json.loads(response_body.decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
            except HTTPError as exc:
                detail = exc.read(1024).decode("utf-8", errors="replace").strip()
                reason = f"HTTP {exc.code}: {detail}" if detail else f"HTTP {exc.code}"
                raise RuntimeError(f"Own model generation failed: {reason}") from exc
            except URLError as exc:
                raise RuntimeError(
                    f"Own model generation failed: {exc.reason}"
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                raise RuntimeError(
                    "Own model generation failed: request timed out after 600 seconds"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Own model generation failed: {exc}") from exc
    else:
        raise RuntimeError(
            "Own model is not configured (set OWN_MODEL_PATH or OWN_LLM_URL)"
        )

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Own model returned an empty response")
    return content.strip()


def _split_knowledge_chunks(text: str, chunk_size: int = 700) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - 80, start + 1)
    return chunks


def _retrieve_project_knowledge(question: str) -> str:
    if not PROJECT_KB_DIRECTORY.is_dir():
        raise RuntimeError(f"Project knowledge directory was not found: {PROJECT_KB_DIRECTORY}")

    documents = []
    for path in sorted(PROJECT_KB_DIRECTORY.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            documents.extend(_split_knowledge_chunks(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise RuntimeError(f"Could not read project knowledge file {path.name}: {exc}") from exc
    if not documents:
        raise RuntimeError("No .md or .txt project knowledge files were found.")

    tokenized = [Counter(token.casefold() for token in _TOKEN_PATTERN.findall(doc)) for doc in documents]
    query_terms = set(token.casefold() for token in _TOKEN_PATTERN.findall(question))
    if not query_terms:
        return "(nothing relevant found)"

    document_frequency = Counter(
        term
        for counts in tokenized
        for term in counts
        if term in query_terms
    )
    scores = []
    for index, counts in enumerate(tokenized):
        total = sum(counts.values()) or 1
        score = sum(
            (counts[term] / total)
            * (log((len(tokenized) + 1) / (document_frequency[term] + 1)) + 1)
            for term in query_terms
            if counts[term]
        )
        scores.append((score, index))
    selected = [
        documents[index]
        for score, index in sorted(scores, key=lambda item: (-item[0], item[1]))[:4]
        if score > 0
    ]
    return "\n---\n".join(selected) if selected else "(nothing relevant found)"


def _patient_data_for_question(question: str, user_id: int) -> str:
    patients = (
        db.session.query(Patient)
        .filter_by(created_by_id=user_id)
        .order_by(Patient.id.asc())
        .all()
    )
    question_folded = question.casefold()
    matched_patients = []
    for patient in patients:
        full_name = f"{patient.first_name} {patient.last_name}"
        names_to_match = (full_name, patient.first_name)
        if any(
            re.search(
                rf"(?<!\w){re.escape(name.casefold())}(?!\w)",
                question_folded,
            )
            for name in names_to_match
            if name.strip()
        ):
            matched_patients.append(patient)
            if len(matched_patients) == 2:
                break

    patient_records = []
    for patient in matched_patients:
        report = (
            db.session.query(MedicalReport)
            .filter(
                MedicalReport.patient_id == patient.id,
                MedicalReport.extracted_text.isnot(None),
                MedicalReport.extracted_text != "",
            )
            .order_by(
                MedicalReport.report_date.desc().nullslast(),
                MedicalReport.created_at.desc(),
                MedicalReport.id.desc(),
            )
            .first()
        )
        if report is None:
            continue
        report_date = report.report_date.isoformat() if report.report_date else "date unavailable"
        patient_records.append(
            f"{patient.first_name} {patient.last_name}, latest report ({report_date}):\n"
            f"{report.extracted_text}"
        )
    return "\n---\n".join(patient_records) if patient_records else "(none)"


def build_own_assistant_prompt(question: str, user_id: int) -> str:
    project_knowledge = _retrieve_project_knowledge(question)
    patient_data = _patient_data_for_question(question, user_id)
    return (
        f"PROJECT KNOWLEDGE:\n{project_knowledge}\n\n"
        f"PATIENT DATA:\n{patient_data}\n\n"
        f"QUESTION: {question}"
    )
