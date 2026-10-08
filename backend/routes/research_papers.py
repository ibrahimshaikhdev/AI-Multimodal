from datetime import date, datetime, timezone
from pathlib import Path

from flask import Blueprint, abort, current_app, g, jsonify, request, send_from_directory

from backend.auth import require_auth
from backend.extensions import db
from backend.models import ResearchPaper
from backend.services.file_storage import save_research_paper_file
from backend.services.audit_logging import record_audit_event
from backend.services.ai_service import AIServiceError
from backend.services.document_comparison import compare_research_evidence
from backend.services.research_rag_service import ResearchIndexError

research_papers_bp = Blueprint("research_papers", __name__)
RESEARCH_DISCLAIMER = (
    "AI-generated research assistance; verify claims against the original papers. "
    "Not medical advice or a clinical diagnosis."
)
RESEARCH_COMPARISON_DISCLAIMER = (
    "Comparison uses locally retrieved source passages and embedding similarity only; "
    "scores are not probabilities and do not determine scientific equivalence; review the papers."
)
RESEARCH_COMPARISON_FIELDS = (
    "methodology",
    "dataset",
    "model",
    "results",
    "limitations",
    "future_work",
)


def _serialize_research_paper(paper):
    return {
        "id": paper.id,
        "title": paper.title,
        "topic": paper.topic,
        "paper_date": paper.paper_date.isoformat() if paper.paper_date else None,
        "authors": paper.authors,
        "journal": paper.journal,
        "original_filename": paper.original_filename,
        "original_url": f"/api/research-papers/{paper.id}/original",
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }


@research_papers_bp.get("/research-papers")
@require_auth
def list_research_papers():
    papers = (
        db.session.query(ResearchPaper)
        .filter_by(user_id=g.current_user.id)
        .order_by(ResearchPaper.created_at.desc(), ResearchPaper.id.desc())
        .all()
    )
    record_audit_event(
        actor_id=g.current_user.id,
        action="research_paper.list",
        resource_type="research_paper",
        metadata={"count": len(papers)},
    )
    return jsonify({"papers": [_serialize_research_paper(paper) for paper in papers]}), 200


@research_papers_bp.post("/research-papers/upload")
@require_auth
def upload_research_paper():
    title = request.form.get("title", "").strip()
    topic = request.form.get("topic", "").strip()
    if not title or len(title) > 200:
        return jsonify({"error": "title is required and must be 200 characters or fewer"}), 400
    if not topic or len(topic) > 120:
        return jsonify({"error": "topic is required and must be 120 characters or fewer"}), 400

    authors = request.form.get("authors", "").strip() or None
    journal = request.form.get("journal", "").strip() or None
    if authors is not None and len(authors) > 255:
        return jsonify({"error": "authors must be 255 characters or fewer"}), 400
    if journal is not None and len(journal) > 255:
        return jsonify({"error": "journal must be 255 characters or fewer"}), 400

    paper_date_value = request.form.get("paper_date", "").strip()
    if paper_date_value:
        try:
            paper_date = datetime.strptime(paper_date_value, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "paper_date must use YYYY-MM-DD format"}), 400
        if paper_date.isoformat() != paper_date_value or paper_date > date.today():
            return jsonify({"error": "paper_date must be a valid date no later than today"}), 400
    else:
        paper_date = None

    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "A research paper PDF or DOCX file is required"}), 400

    try:
        original_filename, stored_filename = save_research_paper_file(
            uploaded_file,
            user_id=g.current_user.id,
            upload_root=current_app.config["UPLOAD_FOLDER"],
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    paper = ResearchPaper(
        user_id=g.current_user.id,
        title=title,
        topic=topic,
        paper_date=paper_date,
        authors=authors,
        journal=journal,
        original_filename=original_filename,
        stored_filename=stored_filename,
        created_at=datetime.now(timezone.utc),
    )
    db.session.add(paper)
    db.session.commit()
    record_audit_event(
        actor_id=g.current_user.id,
        action="research_paper.upload",
        resource_type="research_paper",
        resource_id=paper.id,
        metadata={"topic": paper.topic},
    )
    return jsonify({"paper": _serialize_research_paper(paper)}), 201


@research_papers_bp.get("/research-papers/<int:paper_id>/original")
@require_auth
def get_original_research_paper(paper_id):
    paper = (
        db.session.query(ResearchPaper)
        .filter_by(id=paper_id, user_id=g.current_user.id)
        .first()
    )
    if paper is None:
        return jsonify({"error": "Research paper not found"}), 404

    record_audit_event(
        actor_id=g.current_user.id,
        action="research_paper.view",
        resource_type="research_paper",
        resource_id=paper.id,
    )
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    paper_directory = upload_root / "research_papers" / str(g.current_user.id)
    if not (paper_directory / paper.stored_filename).is_file():
        abort(404)

    return send_from_directory(
        paper_directory,
        paper.stored_filename,
        as_attachment=True,
        download_name=paper.original_filename,
    )


def _owned_research_papers():
    return (
        db.session.query(ResearchPaper)
        .filter_by(user_id=g.current_user.id)
        .order_by(ResearchPaper.created_at.desc(), ResearchPaper.id.desc())
        .all()
    )


def _valid_question(data):
    question = data.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        return None
    return question.strip()


def _get_rag_service():
    return current_app.extensions["research_rag_service"]


@research_papers_bp.post("/research/search")
@require_auth
def search_research_papers():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400
    query = data.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        return jsonify({"success": False, "error": "query must be a non-empty string of at most 2000 characters."}), 400
    top_k = data.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        return jsonify({"success": False, "error": "top_k must be an integer between 1 and 10."}), 400

    papers = _owned_research_papers()
    if not papers:
        return jsonify({"success": True, "results": [], "message": "No research papers are available."}), 200
    try:
        results = _get_rag_service().search(
            g.current_user.id,
            papers,
            query,
            top_k=top_k,
        )
    except ResearchIndexError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        current_app.logger.exception("Research semantic search failed for user %s", g.current_user.id)
        return jsonify({"success": False, "error": "Research search is temporarily unavailable."}), 503

    return jsonify({"success": True, "results": results}), 200


@research_papers_bp.post("/research/ask")
@require_auth
def ask_research_question():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400
    question = _valid_question(data)
    if question is None:
        return jsonify({"success": False, "error": "question must be a non-empty string of at most 2000 characters."}), 400
    top_k = data.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        return jsonify({"success": False, "error": "top_k must be an integer between 1 and 10."}), 400

    papers = _owned_research_papers()
    if not papers:
        return jsonify({"success": False, "error": "Upload a research paper before asking a research question."}), 422
    try:
        results = _get_rag_service().search(
            g.current_user.id,
            papers,
            question,
            top_k=top_k,
        )
    except ResearchIndexError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        current_app.logger.exception("Research retrieval failed for user %s", g.current_user.id)
        return jsonify({"success": False, "error": "Research retrieval is temporarily unavailable."}), 503
    if not results:
        return jsonify({"success": False, "error": "No readable research excerpts were found for this question."}), 422

    context = [
        {
            "citation": result["citation"],
            "paper_title": result["paper_title"],
            "paper_date": result["paper_date"],
            "excerpt": result["text"],
        }
        for result in results
    ]
    try:
        answer = current_app.extensions["ai_service"].research_answer(question, context)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except AIServiceError as exc:
        return jsonify({"success": False, "error": str(exc)}), exc.status_code
    except Exception:
        current_app.logger.exception("Research answer generation failed for user %s", g.current_user.id)
        return jsonify({"success": False, "error": "Research answer generation failed."}), 502

    citations = [
        {
            "citation": result["citation"],
            "paper_id": result["paper_id"],
            "paper_title": result["paper_title"],
            "paper_date": result["paper_date"],
            "excerpt": result["text"],
        }
        for result in results
    ]
    return jsonify(
        {
            "success": True,
            "answer": answer,
            "citations": citations,
            "disclaimer": RESEARCH_DISCLAIMER,
        }
    ), 200


@research_papers_bp.post("/research/compare")
@require_auth
def compare_research_papers():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400
    paper_ids = data.get("paper_ids")
    if (
        not isinstance(paper_ids, list)
        or len(paper_ids) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in paper_ids)
        or paper_ids[0] == paper_ids[1]
    ):
        return jsonify({"success": False, "error": "paper_ids must contain exactly two distinct paper IDs."}), 400

    papers = (
        db.session.query(ResearchPaper)
        .filter(
            ResearchPaper.user_id == g.current_user.id,
            ResearchPaper.id.in_(paper_ids),
        )
        .all()
    )
    paper_by_id = {paper.id: paper for paper in papers}
    if len(paper_by_id) != 2:
        return jsonify({"success": False, "error": "One or more research papers were not found."}), 404
    ordered_papers = [paper_by_id[paper_id] for paper_id in paper_ids]
    rag_service = _get_rag_service()
    evidence: dict[str, dict[str, list[dict[str, str]]]] = {
        "paper_1": {},
        "paper_2": {},
    }
    citations = []
    field_queries = {
        "methodology": "study methodology methods design protocol procedures",
        "dataset": "dataset participants sample population data source",
        "model": "model algorithm architecture software intervention",
        "results": "results findings outcomes measurements performance",
        "limitations": "limitations weaknesses bias constraints",
        "future_work": "future work further research next steps",
    }
    try:
        for paper_index, paper in enumerate(ordered_papers, start=1):
            rag_service.paper_text(g.current_user.id, papers, paper.id)
            for field, query in field_queries.items():
                passages = rag_service.search(
                    g.current_user.id,
                    papers,
                    query,
                    top_k=1,
                    paper_ids={paper.id},
                )
                evidence[f"paper_{paper_index}"][field] = [
                    {
                        "citation": passage["citation"],
                        "excerpt": passage["text"],
                    }
                    for passage in passages
                ]
                citations.extend(
                    {
                        "citation": passage["citation"],
                        "paper_id": passage["paper_id"],
                        "paper_title": passage["paper_title"],
                        "field": field,
                    }
                    for passage in passages
                )
    except ResearchIndexError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        current_app.logger.exception("Research comparison retrieval failed for user %s", g.current_user.id)
        return jsonify({"success": False, "error": "Research comparison retrieval is temporarily unavailable."}), 503

    try:
        comparison = compare_research_evidence(
            evidence,
            RESEARCH_COMPARISON_FIELDS,
            _get_rag_service().similarity_matrix,
        )
    except Exception:
        current_app.logger.exception(
            "Local research comparison failed for user %s",
            g.current_user.id,
        )
        return jsonify(
            {
                "success": False,
                "error": "Local research comparison is temporarily unavailable.",
            }
        ), 503

    return jsonify(
        {
            "success": True,
            "papers": [
                {
                    "id": paper.id,
                    "title": paper.title,
                    "topic": paper.topic,
                    "paper_date": paper.paper_date.isoformat() if paper.paper_date else None,
                }
                for paper in ordered_papers
            ],
            "fields": list(RESEARCH_COMPARISON_FIELDS),
            "comparison": comparison,
            "citations": citations,
            "disclaimer": RESEARCH_COMPARISON_DISCLAIMER,
        }
    ), 200


__all__ = ["research_papers_bp"]
