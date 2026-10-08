import numpy as np
import pytest

from backend.services.document_comparison import (
    compare_report_texts,
    compare_research_evidence,
)
from backend.services.research_rag_service import ResearchRAGService


def _exact_text_similarity(first_texts, second_texts):
    return [
        [1.0 if first == second else 0.1 for second in second_texts]
        for first in first_texts
    ]


def test_report_comparison_aligns_explicit_values_and_keeps_unique_findings_last():
    comparison = compare_report_texts(
        (
            "Hemoglobin: 12.0 g/dL\n"
            "Findings:\n"
            "No pleural effusion. Mild linear opacity."
        ),
        (
            "Hgb: 12.00 g/dL\n"
            "Findings:\n"
            "No pleural effusion. New opacity at the left base."
        ),
        _exact_text_similarity,
    )

    assert comparison["parameters"][0]["status"] == "same_value"
    assert comparison["parameters"][0]["report_1"]["value"] == "12.0"
    assert comparison["parameters"][0]["report_2"]["value"] == "12.00"
    assert comparison["summary"]["same_value_count"] == 1
    assert len(comparison["observations"]["related"]) == 1
    assert comparison["observations"]["related"][0]["relationship"] == "same_wording"
    assert comparison["observations"]["report_1_only"] == ["Mild linear opacity."]
    assert comparison["observations"]["report_2_only"] == [
        "New opacity at the left base."
    ]


def test_report_comparison_does_not_compare_different_units_as_equal():
    comparison = compare_report_texts(
        "Glucose: 90 mg/dL",
        "Glucose: 5 mmol/L",
        _exact_text_similarity,
    )

    assert comparison["parameters"][0]["status"] == "units_not_comparable"


def test_report_comparison_includes_unlabeled_ocr_lines_and_aligns_exact_text():
    def unrelated_embeddings(first_texts, second_texts):
        return [[0.1 for _ in second_texts] for _ in first_texts]

    comparison = compare_report_texts(
        (
            "Patient Name: Example Person\n"
            "No focal air-space opacity.\n"
            "Mild cardiomegaly."
        ),
        (
            "Patient Name: Example Person\n"
            "No focal air-space opacity.\n"
            "Small left pleural effusion."
        ),
        unrelated_embeddings,
    )

    assert comparison["summary"]["related_observation_count"] == 1
    assert comparison["observations"]["related"][0]["report_1"] == (
        "No focal air-space opacity."
    )
    assert comparison["observations"]["report_1_only"] == ["Mild cardiomegaly."]
    assert comparison["observations"]["report_2_only"] == [
        "Small left pleural effusion."
    ]


def test_research_comparison_labels_evidence_without_generating_a_summary():
    evidence = {
        "paper_1": {
            "methodology": [
                {"citation": "[P1-C1]", "excerpt": "A randomized controlled trial."}
            ],
            "dataset": [],
        },
        "paper_2": {
            "methodology": [
                {"citation": "[P2-C1]", "excerpt": "A randomized controlled trial."}
            ],
            "dataset": [
                {"citation": "[P2-C2]", "excerpt": "A cohort of 120 participants."}
            ],
        },
    }

    result = compare_research_evidence(
        evidence,
        ("methodology", "dataset"),
        _exact_text_similarity,
    )

    assert result["methodology"]["status"] == "related_evidence"
    assert result["methodology"]["paper_1_citation"] == "[P1-C1]"
    assert result["dataset"]["status"] == "only_in_paper_2"
    assert "No relevant passage retrieved" in result["dataset"]["paper_1"]


def test_similarity_matrix_normalizes_local_embedding_vectors(tmp_path):
    service = ResearchRAGService(tmp_path / "indexes", "test-model", tmp_path / "uploads")
    service._encode = lambda texts: np.array(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]],
        dtype=np.float32,
    )

    scores = service.similarity_matrix(["first", "second"], ["third"])

    assert scores[0][0] == 0.0
    assert scores[1][0] == pytest.approx(2 ** -0.5)
