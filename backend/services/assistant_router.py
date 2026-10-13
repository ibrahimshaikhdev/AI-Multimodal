from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from functools import lru_cache
from difflib import get_close_matches
from pathlib import Path
import threading

from sklearn.feature_extraction.text import TfidfVectorizer


MIN_SCORE = 0.40
MARGIN = 0.04

DOMAIN_VOCAB = {
    "patient", "patients", "report", "reports", "scan", "scans", "timeline",
    "history", "hemoglobin", "haemoglobin", "hba1c", "glucose", "sugar",
    "cholesterol", "ldl", "hdl", "triglycerides", "creatinine", "platelets",
    "wbc", "rbc", "vitamin", "thyroid", "tsh", "sodium", "potassium", "urea",
    "summary", "summarize", "compare", "comparison", "abnormal", "normal",
    "high", "low", "highest", "lowest", "latest", "previous", "research",
    "paper", "papers", "project", "technology", "stack", "database", "feature",
    "features", "assistant", "count", "many", "list", "show", "diagnosis",
    "medicine",
}

_SHORTHAND = {
    "thx": "thanks",
    "thnx": "thanks",
    "thanku": "thanks",
    "thankuu": "thanks",
    "tnx": "thanks",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "r": "are",
    "ur": "your",
    "wat": "what",
    "wht": "what",
    "hw": "how",
    "bcz": "because",
    "abt": "about",
    "pt": "patient",
    "pts": "patients",
    "rpt": "report",
    "tel": "tell",
    "wats": "whats",
}

_EXAMPLES = {
    "small_talk": [
        "hi", "hello", "hey", "hii", "helo", "good morning", "good evening",
        "thanks", "thank you", "thanku", "thx", "ok thanks", "bye", "see you",
        "how are you", "how r u", "whats up", "what are you doing", "nice",
        "great", "cool", "hiii there", "heloo", "thnks",
    ],
    "about_assistant": [
        "what can you do", "help", "who are you", "what are you",
        "how do i use this", "what can i ask you", "what questions can you answer",
        "your capabilities", "how can you help me", "what are your functions",
        "what can this assistant do", "how to use the assistant", "capabilites",
        "assitant help", "what can you asisst with",
    ],
    "counts_lists": [
        "how many patients", "how many patients do we have", "total number of patients",
        "number of reports", "how many scans", "how many research papers",
        "list all patients", "show patient names", "count of reports uploaded",
        "how many pateints are there", "how many patients are in the database",
        "list patients", "how many medical reports", "number of scans uploaded",
        "count my patients", "how many papers did i upload", "hw many pts",
        "pateints list", "how meny reports",
    ],
    "find_patients_by_lab": [
        "which patients have high hba1c", "who has low hemoglobin",
        "patients with high cholesterol", "which patient has the highest sugar",
        "lowest hemoglobin patient", "who is diabetic based on hba1c",
        "patients with abnormal creatinine", "show patients with low platelets",
        "list patients with high ldl", "who has high glucose",
        "which patient has low vitamin d", "patients with high triglycerides",
        "who has abnormal tsh", "highest wbc patient", "lowest sodium",
        "which patients have high haemoglobin", "pateints with high hba1c",
        "hemoglobn low patients", "which pateints have abnormal creatinine",
    ],
    "selected_report": [
        "whats in the report", "what is this report about", "summarize this report",
        "explain the report", "is anything abnormal in this report", "read the report",
        "report summary", "tell me about this report", "wat is in report",
        "what does this report contain", "what is in this report", "summarize it",
        "explain this", "is it ok", "wats in the rpt", "whats in teh report",
        "summarise this repot", "explan the report",
    ],
    "patient_values": [
        "what is his hba1c", "latest results", "show his lab values",
        "her hemoglobin value", "abnormal values for this patient",
        "what are the latest lab results", "is his sugar high",
        "cholesterol of this patient", "show this patient's latest results",
        "what are the patient's values", "latest blood test results",
        "show me patient lab results", "patient's hba1c result",
        "what is the latest hemoglobin", "latest lab values for patient",
        "what are the latest pateint values", "hemoglobn value for this patient",
    ],
    "timeline": [
        "tell me his timeline", "patient timeline", "health history",
        "show the history of this patient", "what happened over time",
        "timeline details", "sequence of reports", "his medical history over the months",
        "show this patient's timeline", "what changed over time",
        "chronological medical history", "events in patient history",
        "timeline of reports and scans", "show health events in date order",
        "timline details", "patient histry", "show the timline",
    ],
    "scans": [
        "show his scans", "what do the scans say", "latest scan result", "xray result",
        "any fracture detected", "scan analysis", "what is in the scan",
        "imaging results", "show this patient's scan summaries",
        "what did the mri show", "latest xray analysis", "ct scan result",
        "tell me about the patient's scans", "scan findings",
        "what does his scan show", "show the scan reslts", "xray reslt",
    ],
    "compare_reports": [
        "compare his last two reports", "what changed since the previous report",
        "difference between reports", "did his values improve",
        "improvement since last time", "compare latest and previous report",
        "has he got better", "compare this patient's reports",
        "compare latest two lab reports", "changes between earlier and later report",
        "what is different in the two reports", "compare reports over time",
        "compare the last two lab results", "are the latest results changed",
        "compare his last reports", "comapre the reports", "what chagned since last report",
    ],
    "project_info": [
        "what tech stack is used", "which database is used",
        "how does report comparison work", "what does this project do",
        "explain the features", "which ocr is used", "how is data stored",
        "what is rag in this project", "how does the assistant work",
        "is my data private", "which model does this app use",
        "what technologies are used", "what backend does the app use",
        "what programming language is used", "how are reports compared",
        "what tech stak is used", "what datbase does this use",
    ],
    "medical_term": [
        "what is hba1c", "explain hemoglobin", "what does creatinine mean",
        "meaning of ldl", "what is vitamin d", "what does high sgpt mean",
        "define triglycerides", "what is tsh", "wat is hba1c",
        "what does wbc measure", "explain platelets", "what is sodium",
        "what is fasting glucose", "define hematocrit", "what is rbc",
        "what is hemoglobn", "meaning of creatnine", "what does cholestrol mean",
    ],
    "research": [
        "summarize the research papers", "what do the papers say",
        "which papers are uploaded", "compare research papers",
        "ask about a paper", "main finding of the paper",
        "what research papers do i have", "show uploaded paper titles",
        "summarize my uploaded studies", "tell me about the research documents",
        "what is in the paper library", "list my research articles",
        "compare my uploaded articles", "research paper question",
        "summarise research papres", "what do my reserch papers say",
    ],
}

_CONTEXT_KEYWORDS = {
    "timeline": ("timeline", "history", "over time", "chronolog"),
    "scans": ("scan", "scans", "xray", "x-ray", "mri", "ct"),
    "compare_reports": ("compare", "comparison", "changed", "change", "improv"),
}
_PATIENT_INTENTS = {"patient_values", "timeline", "scans", "compare_reports"}
_GENERIC_SELECTED_REPORT = re.compile(
    r"\b(?:what(?: is|s)? (?:this|it)|what does (?:this|it) say|"
    r"what(?:s)? in (?:the )?(?:report|it)|and what unit|"
    r"summar(?:ize|ise) (?:this report|it)|explain (?:this|the report|it)|"
    r"is it ok)\b"
)
_REPORT_WORD = re.compile(r"\b(?:report|reports|reported|rpt)\b")
_SMALL_TALK_PATTERNS = (
    re.compile(r"^(?:hi|hello|hey|hii|helo|good morning|good evening|"
               r"thanks|thank you|thanku|thankuu|thx|thnx|tnx|ok thanks|"
               r"bye|see you|how are you|how r you|whats up|nice|great|cool)$"),
    re.compile(r"^what are you doing(?: today)?$"),
)
_PATIENT_REFERENCE = re.compile(
    r"\b(?:his|her|their|he|she|they|this patient|that patient)\b"
)
_UNMATCHED_WRITE_LOCK = threading.Lock()


def normalize(text: str) -> str:
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text.casefold())
    tokens = re.findall(r"[a-z0-9]+", text)
    normalized_tokens = []
    for token in tokens:
        token = _SHORTHAND.get(token, token)
        if len(token) >= 4 and token not in DOMAIN_VOCAB:
            match = get_close_matches(token, sorted(DOMAIN_VOCAB), n=1, cutoff=0.8)
            if match:
                token = match[0]
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


@lru_cache(maxsize=1)
def _vectorizer_and_matrix():
    examples = [
        (intent, normalize(example))
        for intent, values in _EXAMPLES.items()
        for example in values
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=False,
    )
    matrix = vectorizer.fit_transform([example for _, example in examples])
    return vectorizer, matrix, tuple(intent for intent, _ in examples)


def classify(question: str, ctx: dict | None = None) -> tuple[str, float]:
    normalized = normalize(question)
    context = ctx or {}
    has_report = bool(context.get("report_id"))
    has_patient = bool(context.get("patient_id"))

    if has_report and _GENERIC_SELECTED_REPORT.search(normalized):
        return "selected_report", 1.0
    if normalized in {
        "how do i use this",
        "how do i use the app",
        "how do i use the assistant",
    }:
        return "about_assistant", 1.0

    if has_patient and _PATIENT_REFERENCE.search(normalized):
        if "compare_reports" in _PATIENT_INTENTS and any(
            keyword in normalized for keyword in _CONTEXT_KEYWORDS["compare_reports"]
        ):
            return "compare_reports", 1.0
        for intent in ("timeline", "scans"):
            if any(keyword in normalized for keyword in _CONTEXT_KEYWORDS[intent]):
                return intent, 1.0
        return "patient_values", 1.0
    if any(pattern.fullmatch(normalized) for pattern in _SMALL_TALK_PATTERNS):
        return "small_talk", 1.0
    if normalized.startswith(("how do i ", "how can i ")) and not any(
        term in normalized
        for term in (
            "assistant", "app", "application", "patient", "report", "scan",
            "research", "paper", "medical", "lab", "project", "model", "ocr",
        )
    ):
        return "fallback", 1.0

    vectorizer, matrix, intents = _vectorizer_and_matrix()
    scores = (matrix @ vectorizer.transform([normalized]).T).toarray().ravel()
    best_by_intent = {}
    for index, score in enumerate(scores):
        intent = intents[index]
        best_by_intent[intent] = max(best_by_intent.get(intent, 0.0), float(score))
    ranked = sorted(best_by_intent.items(), key=lambda item: (-item[1], item[0]))
    if ranked and has_report and _REPORT_WORD.search(normalized):
        candidate_intent = ranked[0][0]
        if candidate_intent in {
            "fallback",
            "patient_values",
            "selected_report",
            "medical_term",
        }:
            return "selected_report", max(ranked[0][1], 1.0)
    if normalized.startswith("tell me a ") and not any(
        term in normalized
        for term in ("report", "patient", "scan", "timeline", "project", "paper")
    ):
        return "fallback", ranked[0][1] if ranked else 0.0
    if not ranked or ranked[0][1] < MIN_SCORE:
        return "fallback", ranked[0][1] if ranked else 0.0
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MARGIN:
        return "fallback", ranked[0][1]
    intent, score = ranked[0]
    required_terms = {
        "counts_lists": ("patient", "report", "scan", "paper"),
        "find_patients_by_lab": tuple(
            set(DOMAIN_VOCAB)
            & {
                "patient", "patients", "hemoglobin", "haemoglobin", "hba1c",
                "glucose", "sugar", "cholesterol", "ldl", "hdl",
                "triglycerides", "creatinine", "platelets", "wbc", "rbc",
                "vitamin", "thyroid", "tsh", "sodium", "potassium", "urea",
            }
        ),
        "selected_report": ("report", "rpt"),
        "patient_values": ("patient", "report", "hemoglobin", "hba1c", "glucose", "lab"),
        "timeline": ("timeline", "history", "over time", "chronolog", "report"),
        "scans": ("scan", "scans", "xray", "mri", "ct", "imaging"),
        "compare_reports": ("report", "compare", "changed", "change", "improv"),
        "project_info": (
            "project", "technology", "stack", "database", "feature", "ocr",
            "rag", "assistant", "model", "backend", "frontend", "private",
        ),
        "medical_term": (
            "hemoglobin", "haemoglobin", "hb", "hgb", "rbc", "wbc", "wec",
            "platelet", "pcv", "hematocrit", "glucose", "sugar", "hba1c",
            "cholesterol", "ldl", "hdl", "triglyceride", "sgpt", "alt",
            "sgot", "ast", "creatinine", "urea", "vitamin d", "tsh",
            "sodium", "potassium", "mcv", "mch", "mchc", "uric acid",
            "bun", "calcium",
        ),
        "research": ("research", "paper", "papers", "study", "studies", "article"),
    }
    if intent in required_terms and not any(
        term in normalized for term in required_terms[intent]
    ):
        return "fallback", score
    return intent, score


def log_unmatched_question(question: str, user_id: int, instance_path: str) -> None:
    destination = Path(instance_path) / "assistant_unmatched.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "question": question,
    }
    try:
        encoded = json.dumps(entry, ensure_ascii=False)
        with _UNMATCHED_WRITE_LOCK:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as unmatched_file:
                unmatched_file.write(encoded + "\n")
    except OSError:
        raise RuntimeError("Could not log unmatched Assistant question.")


_vectorizer_and_matrix()
