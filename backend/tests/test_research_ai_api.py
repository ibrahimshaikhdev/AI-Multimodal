import json
from datetime import date

import fitz
import numpy as np
import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import MedicalReport, Patient, ResearchPaper, ScanAsset, User
from backend.services.ai_service import AIServiceTimeout, AIServiceUnavailable
from backend.services.research_rag_service import ResearchRAGService


@pytest.fixture
def ai_workspace(tmp_path):
    app = create_app(testing=True)
    uploads = tmp_path / "uploads"
    app.config["UPLOAD_FOLDER"] = str(uploads)
    rag_service = ResearchRAGService(
        index_root=tmp_path / "indexes",
        embedding_model="test-embedding-model",
        upload_root=uploads,
    )
    rag_service._encode = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
    app.extensions["research_rag_service"] = rag_service

    with app.app_context():
        db.create_all()
        owner = User(
            first_name="Research",
            last_name="Owner",
            email="research-ai-owner@example.com",
            password_hash=generate_password_hash("ResearchPass123!"),
            role="patient",
        )
        other = User(
            first_name="Other",
            last_name="Owner",
            email="research-ai-other@example.com",
            password_hash=generate_password_hash("ResearchPass123!"),
            role="patient",
        )
        db.session.add_all([owner, other])
        db.session.commit()
        owner_id, other_id = owner.id, other.id

        with app.test_client() as client:
            yield client, app, owner_id, other_id, uploads

        db.session.remove()
        db.drop_all()


def _login(client, email):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "ResearchPass123!"},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def _pdf_with_text(text):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _add_paper(owner_id, uploads, title, content, suffix="pdf"):
    stored_filename = f"paper-{title.lower().replace(' ', '-')}.{suffix}"
    paper_directory = uploads / "research_papers" / str(owner_id)
    paper_directory.mkdir(parents=True, exist_ok=True)
    payload = (
        _pdf_with_text(content)
        if suffix == "pdf"
        else content
    )
    (paper_directory / stored_filename).write_bytes(payload)
    paper = ResearchPaper(
        user_id=owner_id,
        title=title,
        topic="Medical research",
        paper_date=date(2024, 1, 1),
        original_filename=f"{title}.{suffix}",
        stored_filename=stored_filename,
    )
    db.session.add(paper)
    db.session.commit()
    return paper.id


def test_research_search_extracts_real_local_text_and_is_account_scoped(ai_workspace):
    client, app, owner_id, other_id, uploads = ai_workspace
    owner_token = _login(client, "research-ai-owner@example.com")
    other_token = _login(client, "research-ai-other@example.com")
    with app.app_context():
        paper_id = _add_paper(
            owner_id,
            uploads,
            "Cardiac methods",
            "The study used echocardiography and reported a cohort of 42 participants.",
        )
        _add_paper(owner_id, uploads, "Other imaging", "The paper discusses computed tomography.")

    response = client.post(
        "/api/research/search",
        headers={"Authorization": f"Bearer {owner_token}"},
        json={"query": "echocardiography methods", "paper_ids": [paper_id]},
    )

    assert response.status_code == 200
    result = response.get_json()["results"][0]
    assert result["paper_id"] == paper_id
    assert result["citation"] == f"[P{paper_id}-C1]"
    assert "42 participants" in result["text"]

    other_response = client.post(
        "/api/research/search",
        headers={"Authorization": f"Bearer {other_token}"},
        json={"query": "echocardiography methods"},
    )
    assert other_response.status_code == 200
    assert other_response.get_json()["results"] == []
    private_paper_response = client.post(
        "/api/research/search",
        headers={"Authorization": f"Bearer {other_token}"},
        json={"query": "echocardiography methods", "paper_ids": [paper_id]},
    )
    assert private_paper_response.status_code == 404
    assert other_id != owner_id


def test_research_ask_uses_retrieved_excerpts_and_returns_citations(ai_workspace):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        selected_paper_id = _add_paper(
            owner_id,
            uploads,
            "Imaging findings",
            "The paper reports an imaging sensitivity of 91 percent.",
        )
        _add_paper(owner_id, uploads, "Excluded findings", "The paper reports a specificity of 75 percent.")

    captured = {}

    def answer(question, context):
        captured["question"] = question
        captured["context"] = context
        return "The retrieved paper reports 91 percent sensitivity."

    app.extensions["ai_service"].research_answer = answer
    response = client.post(
        "/api/research/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What sensitivity is reported?",
            "paper_ids": [selected_paper_id],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"].startswith("The retrieved paper")
    assert body["citations"][0]["citation"].startswith("[P")
    excerpts = captured["context"]["excerpts"]
    assert "91 percent" in excerpts[0]["excerpt"]
    assert captured["question"] == "What sensitivity is reported?"
    assert all(
        source["citation"].startswith(f"[P{selected_paper_id}-")
        for source in excerpts
    )


def test_local_research_ask_uses_three_chunks_and_returns_source_citations(
    ai_workspace,
    monkeypatch,
):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        paper_id = _add_paper(
            owner_id,
            uploads,
            "Local source study",
            "The study measured sensitivity at 91 percent.",
        )
    captured = {}
    rag_service = app.extensions["research_rag_service"]
    original_search = rag_service.search

    def search(user_id, papers, query, *, top_k=5, paper_ids=None):
        captured["top_k"] = top_k
        return original_search(
            user_id,
            papers,
            query,
            top_k=top_k,
            paper_ids=paper_ids,
        )

    monkeypatch.setattr(rag_service, "search", search)

    def local_answer(prompt, system=None):
        captured["prompt"] = prompt
        return "Sensitivity was 91 percent [S1]."

    monkeypatch.setattr("backend.routes.research_papers.local_generate", local_answer)
    response = client.post(
        "/api/research/ask-local",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What sensitivity is reported?", "paper_ids": [paper_id], "top_k": 5},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "Sensitivity was 91 percent [S1]."
    assert len(body["citations"]) <= 3
    assert body["citations"][0]["citation"].startswith("[S1] · ")
    assert f"[P{paper_id}-" in body["citations"][0]["citation"]
    assert captured["top_k"] == 3
    assert '"citation": "[S1]"' in captured["prompt"]
    assert "Cite every factual claim" in captured["prompt"]


def test_local_research_ask_returns_local_error_without_fallback(ai_workspace, monkeypatch):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        paper_id = _add_paper(owner_id, uploads, "Local error study", "A readable paper.")

    def failed_local_model(_prompt, system=None):
        raise RuntimeError("Local model failed (is llama-server running?): refused")

    monkeypatch.setattr(
        "backend.routes.research_papers.local_generate",
        failed_local_model,
    )
    response = client.post(
        "/api/research/ask-local",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What is reported?", "paper_ids": [paper_id]},
    )

    assert response.status_code == 502
    assert response.get_json() == {
        "success": False,
        "error": "Local model failed (is llama-server running?): refused",
    }


def test_research_ask_returns_ai_timeout(ai_workspace):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        _add_paper(owner_id, uploads, "Timeout study", "A paper with readable text.")
    def timeout(question, context):
        raise AIServiceTimeout()

    app.extensions["ai_service"].research_answer = timeout

    response = client.post(
        "/api/research/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What does the paper report?"},
    )

    assert response.status_code == 504
    assert "timed out" in response.get_json()["error"]


def test_research_comparison_returns_ai_and_local_fields_for_two_owned_papers(ai_workspace):
    client, app, owner_id, other_id, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        first_id = _add_paper(owner_id, uploads, "First study", "Methods used MRI.")
        second_id = _add_paper(owner_id, uploads, "Second study", "Methods used CT.")
        other_id = _add_paper(other_id, uploads, "Private paper", "Private evidence.")

    invalid = client.post(
        "/api/research/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"paper_ids": [first_id]},
    )
    assert invalid.status_code == 400

    forbidden = client.post(
        "/api/research/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"paper_ids": [first_id, other_id]},
    )
    assert forbidden.status_code == 404

    def low_similarity(first_texts, second_texts):
        return [[0.4 for _ in second_texts] for _ in first_texts]

    app.extensions["research_rag_service"].similarity_matrix = low_similarity

    captured = {}

    def compare_set(evidence):
        captured["evidence"] = evidence
        fields = ("methodology", "dataset", "model", "results", "limitations", "future_work")
        return json.dumps(
            {
                "papers": [
                    {
                        "id": paper["id"],
                        "title": paper["title"],
                        **{field: f"{field} from {paper['title']}" for field in fields},
                    }
                    for paper in evidence
                ],
                "overall": "Paper 1 used MRI, whereas Paper 2 used CT.",
            }
        )

    app.extensions["ai_service"].compare_research_set = compare_set
    response = client.post(
        "/api/research/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"paper_ids": [first_id, second_id]},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["papers"][0]["id"] == first_id
    assert payload["comparison"] is None
    assert payload["ai_comparison"]["overall"] == (
        "First study used MRI, whereas Second study used CT."
    )
    assert len(captured["evidence"]) == 2
    assert payload["citations"]


def test_local_research_comparison_parses_json_fences_and_limits_excerpt_context(
    ai_workspace,
    monkeypatch,
):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        first_id = _add_paper(
            owner_id,
            uploads,
            "First local study",
            "Methods and results from the first study. " * 30,
        )
        second_id = _add_paper(
            owner_id,
            uploads,
            "Second local study",
            "Methods and results from the second study. " * 30,
        )
    captured = {}
    fields = ("methodology", "dataset", "model", "results", "limitations", "future_work")

    def local_comparison(prompt, system=None):
        context = prompt.split("\n\nRequest:\n", maxsplit=1)[0].removeprefix("Context:\n")
        captured["context"] = json.loads(context)
        captured["system"] = system
        papers = captured["context"]["papers"]
        result = {
            "papers": [
                {
                    "id": paper["id"],
                    "title": paper["title"],
                    **{field: f"Supported {field}." for field in fields},
                }
                for paper in papers
            ],
            "overall": "The supplied excerpts support this comparison.",
        }
        return f"```json\n{json.dumps(result)}\n```"

    monkeypatch.setattr(
        "backend.routes.research_papers.local_generate",
        local_comparison,
    )
    response = client.post(
        "/api/research/compare-local",
        headers={"Authorization": f"Bearer {token}"},
        json={"paper_ids": [first_id, second_id]},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ai_comparison"]["overall"] == (
        "The supplied excerpts support this comparison."
    )
    assert body["ai_error"] is None
    assert body["citations"]
    for paper in captured["context"]["papers"]:
        excerpt_characters = sum(
            len(passage["excerpt"])
            for passages in paper["evidence"].values()
            for passage in passages
        )
        assert excerpt_characters <= 3500
    assert captured["system"] is None


def test_local_research_comparison_returns_error_for_invalid_json(ai_workspace, monkeypatch):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        paper_ids = [
            _add_paper(owner_id, uploads, title, "Readable study text.")
            for title in ("Malformed first study", "Malformed second study")
        ]
    monkeypatch.setattr(
        "backend.routes.research_papers.local_generate",
        lambda _prompt, system=None: "not json",
    )

    response = client.post(
        "/api/research/compare-local",
        headers={"Authorization": f"Bearer {token}"},
        json={"paper_ids": paper_ids},
    )

    assert response.status_code == 502
    body = response.get_json()
    assert body["success"] is False
    assert body["error"].startswith("Local model returned unreadable JSON:")


def test_research_comparison_preserves_local_fields_when_ai_provider_fails(ai_workspace):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        first_id = _add_paper(owner_id, uploads, "First paper", "First paper text.")
        second_id = _add_paper(owner_id, uploads, "Second paper", "Second paper text.")

    def unavailable_generation(_evidence):
        raise AIServiceUnavailable()

    app.extensions["ai_service"].compare_research_set = unavailable_generation

    response = client.post(
        "/api/research/compare",
        headers={"Authorization": f"Bearer {token}"},
        json={"paper_ids": [first_id, second_id]},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["comparison"]["methodology"]["status"] == "related_evidence"
    assert payload["ai_comparison"] is None
    assert payload["ai_error"] == "Configured AI providers are unavailable."


def test_research_comparison_supports_three_papers(ai_workspace):
    client, app, owner_id, _, uploads = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    with app.app_context():
        paper_ids = [
            _add_paper(owner_id, uploads, title, f"The study {title} used a prospective design.")
            for title in ("First study", "Second study", "Third study")
        ]

    app.extensions["research_rag_service"].similarity_matrix = (
        lambda first, second: [[0.5 for _ in second] for _ in first]
    )

    def compare_set(evidence):
        fields = ("methodology", "dataset", "model", "results", "limitations", "future_work")
        return json.dumps(
            {
                "papers": [
                    {
                        "id": paper["id"],
                        "title": paper["title"],
                        **{field: "Not stated in the retrieved excerpts." for field in fields},
                    }
                    for paper in evidence
                ],
                "overall": "The excerpts do not support further comparison.",
            }
        )

    app.extensions["ai_service"].compare_research_set = compare_set
    response = client.post(
        "/api/research/compare",
        headers={"Authorization": f"******"},
        json={"paper_ids": paper_ids},
        environ_overrides={"HTTP_AUTHORIZATION": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert [paper["id"] for paper in payload["ai_comparison"]["papers"]] == paper_ids
    assert len(payload["papers"]) == 3
    assert payload["comparison"] is None
    assert len(payload["citations"]) >= 3


def test_assistant_report_context_is_owner_checked_and_uses_saved_text(ai_workspace):
    client, app, owner_id, other_id, _ = ai_workspace
    token = _login(client, "research-ai-owner@example.com")
    other_token = _login(client, "research-ai-other@example.com")
    with app.app_context():
        patient = Patient(
            first_name="Patient",
            last_name="One",
            created_by_id=owner_id,
        )
        other_patient = Patient(
            first_name="Patient",
            last_name="Two",
            created_by_id=other_id,
        )
        db.session.add_all([patient, other_patient])
        db.session.flush()
        report = MedicalReport(
            patient_id=patient.id,
            report_type="Lab",
            extracted_text="Saved OCR: value 12 units.",
        )
        empty_report = MedicalReport(
            patient_id=patient.id,
            report_type="Unprocessed",
            extracted_text=None,
        )
        other_report = MedicalReport(
            patient_id=other_patient.id,
            report_type="Lab",
            extracted_text="Other user's private result.",
        )
        scan = ScanAsset(
            patient_id=patient.id,
            modality="MRI",
            body_region="Brain",
            original_filename="scan.png",
            file_reference="scan.png",
            processing_status="complete",
        )
        db.session.add_all([report, empty_report, other_report, scan])
        db.session.commit()
        patient_id = patient.id
        report_id, other_report_id = report.id, other_report.id

    captured = {}

    def query(question, context):
        captured["context"] = context
        return "Answer based on saved OCR."

    app.extensions["ai_service"].query = query
    response = client.post(
        "/api/ai/query",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What does the report say?",
            "context_type": "report",
            "report_id": report_id,
        },
    )
    assert response.status_code == 200
    assert "Saved OCR: value 12 units." in captured["context"]["sources"][0]["extracted_text"]

    timeline_response = client.post(
        "/api/ai/query",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "Summarize this timeline.",
            "context_type": "patient_timeline",
            "patient_id": patient_id,
        },
    )
    assert timeline_response.status_code == 200
    timeline_sources = captured["context"]["sources"]
    assert all(source.get("extracted_text") for source in timeline_sources if source["type"] == "report")
    assert any(source["type"] == "scan" for source in timeline_sources)

    unauthorized = client.post(
        "/api/ai/query",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What does this say?",
            "context_type": "report",
            "report_id": other_report_id,
        },
    )
    assert unauthorized.status_code == 404

    assert other_id != owner_id
