import pytest

from backend.services.assistant_router import classify, normalize


ROUTER_CASES = [
    ("hi", "small_talk"),
    ("thanku", "small_talk"),
    ("how are you", "small_talk"),
    ("what can you do", "about_assistant"),
    ("who are you", "about_assistant"),
    ("how can you help me", "about_assistant"),
    ("how many patients do we have", "counts_lists"),
    ("hw many pts", "counts_lists"),
    ("list all patients", "counts_lists"),
    ("count of reports uploaded", "counts_lists"),
    ("which patients have high hba1c", "find_patients_by_lab"),
    ("hemoglobn low patients", "find_patients_by_lab"),
    ("patients with abnormal creatinine", "find_patients_by_lab"),
    ("whats in the report", "selected_report"),
    ("wats in the rpt", "selected_report"),
    ("summarize this report", "selected_report"),
    ("what is his hba1c", "patient_values"),
    ("show her lab values", "patient_values"),
    ("latest lab results for this patient", "patient_values"),
    ("tell me his timeline", "timeline"),
    ("timline details", "timeline"),
    ("show the history of this patient", "timeline"),
    ("what do the scans say", "scans"),
    ("latest xray analysis", "scans"),
    ("what did the mri show", "scans"),
    ("compare his last two reports", "compare_reports"),
    ("what changed since the previous report", "compare_reports"),
    ("compare latest and previous report", "compare_reports"),
    ("what tech stack is used", "project_info"),
    ("which database is used", "project_info"),
    ("what is hba1c", "medical_term"),
    ("what does creatinine mean", "medical_term"),
    ("summarize the research papers", "research"),
    ("which papers are uploaded", "research"),
    ("tell me a joke", "fallback"),
    ("who won the match", "fallback"),
    ("capital of france", "fallback"),
    ("what is the weather tomorrow", "fallback"),
    ("write me a poem about the ocean", "fallback"),
    ("what time is it in tokyo", "fallback"),
    ("how do i bake sourdough bread", "fallback"),
    ("who is the current president", "fallback"),
    ("tell me a fun fact", "fallback"),
    ("what should i watch tonight", "fallback"),
    ("how far is the moon", "fallback"),
    ("what is the best football team", "fallback"),
    ("convert 10 miles to kilometers", "fallback"),
    ("give me a random number", "fallback"),
    ("what is the meaning of life", "fallback"),
    ("play some music", "fallback"),
    ("how do airplanes fly", "fallback"),
    ("what is a good vacation destination", "fallback"),
    ("recommend a movie", "fallback"),
    ("tell me a story about a dragon", "fallback"),
    ("what is two plus two", "fallback"),
    ("who won the world cup", "fallback"),
    ("what is the capital of japan", "fallback"),
    ("how to fix a bicycle tire", "fallback"),
    ("what are the latest sports results", "fallback"),
    ("can you explain quantum physics", "fallback"),
    ("how do i make coffee", "fallback"),
    ("how many planets are in the solar system", "fallback"),
    ("write a birthday message for my friend", "fallback"),
    ("what is the tallest mountain", "fallback"),
    ("tell me about the history of jazz", "fallback"),
    ("how can i learn to draw", "fallback"),
    ("what is the exchange rate today", "fallback"),
    ("give me a pasta recipe", "fallback"),
]


def test_router_accuracy_on_labelled_questions(capsys):
    assert len(ROUTER_CASES) >= 60
    assert sum(label == "fallback" for _, label in ROUTER_CASES) >= len(
        ROUTER_CASES
    ) * 0.5
    misses = [
        (question, expected, classify(question)[0])
        for question, expected in ROUTER_CASES
        if classify(question)[0] != expected
    ]
    accuracy = (len(ROUTER_CASES) - len(misses)) / len(ROUTER_CASES)
    print(f"Router accuracy: {accuracy:.1%} ({len(ROUTER_CASES) - len(misses)}/{len(ROUTER_CASES)})")
    for question, expected, actual in misses:
        print(f"MISS: {question!r}: expected {expected}, got {actual}")
    captured = capsys.readouterr()
    assert "Router accuracy:" in captured.out
    assert accuracy >= 0.90, captured.out


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("thankuuu!!!", "thanks"),
        ("WATS in the RPT??", "whats in the report"),
        ("hw many pateints", "how many patients"),
        ("hemoglobn", "hemoglobin"),
    ],
)
def test_normalize_chat_shorthand_and_domain_typos(question, expected):
    assert normalize(question) == expected


@pytest.mark.parametrize(
    ("question", "context", "expected"),
    [
        ("whats in the report", {"report_id": 4}, "selected_report"),
        ("what is this", {"report_id": 4}, "selected_report"),
        ("show his scans", {"patient_id": 3}, "scans"),
        ("compare his last reports", {"patient_id": 3}, "compare_reports"),
        ("tell me his timeline", {"patient_id": 3}, "timeline"),
        ("what is his hba1c", {"patient_id": 3}, "patient_values"),
    ],
)
def test_context_overrides(question, context, expected):
    assert classify(question, context)[0] == expected
