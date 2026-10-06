from datetime import date, datetime, timezone
from pathlib import Path

from flask import Blueprint, abort, current_app, g, jsonify, request, send_from_directory

from backend.auth import require_auth
from backend.extensions import db
from backend.models import ResearchPaper
from backend.services.file_storage import save_research_paper_file

research_papers_bp = Blueprint("research_papers", __name__)


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


__all__ = ["research_papers_bp"]
