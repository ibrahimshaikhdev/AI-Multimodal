from datetime import date
from io import BytesIO

import pytest
from werkzeug.security import generate_password_hash

from backend.app import create_app
from backend.extensions import db
from backend.models import User


@pytest.fixture
def research_client(tmp_path):
    app = create_app(testing=True)
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "uploads")
    with app.app_context():
        db.create_all()
        owner = User(
            first_name="Research",
            last_name="Owner",
            email="research-owner@example.com",
            password_hash=generate_password_hash("ResearchPass123!"),
            role="patient",
        )
        other = User(
            first_name="Other",
            last_name="Researcher",
            email="other-researcher@example.com",
            password_hash=generate_password_hash("ResearchPass123!"),
            role="patient",
        )
        db.session.add_all([owner, other])
        db.session.commit()

        with app.test_client() as client:
            yield client, owner.id, other.id

        db.session.remove()
        db.drop_all()


def _login(client, email):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "ResearchPass123!"},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def test_workspace_library_paths_serve_the_shared_frontend(research_client):
    client, _, _ = research_client
    for path in (
        "/medicalreports.html",
        "/scan.html",
        "/healthtimeline.html",
        "/researchpapers.html",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert b'id="signedInView"' in response.data


def test_research_papers_upload_list_and_download_are_account_scoped(research_client):
    client, owner_id, other_id = research_client
    owner_token = _login(client, "research-owner@example.com")
    other_token = _login(client, "other-researcher@example.com")

    upload = client.post(
        "/api/research-papers/upload",
        headers={"Authorization": f"Bearer {owner_token}"},
        data={
            "title": "Cardiac imaging review",
            "topic": "Echocardiography",
            "paper_date": date.today().isoformat(),
            "authors": "A. Researcher",
            "journal": "Imaging Studies",
            "file": (BytesIO(b"%PDF-1.7\npaper"), "review paper.pdf"),
        },
        content_type="multipart/form-data",
    )

    assert upload.status_code == 201
    paper = upload.get_json()["paper"]
    assert paper["title"] == "Cardiac imaging review"
    assert paper["topic"] == "Echocardiography"
    assert paper["original_filename"] == "review_paper.pdf"

    listing = client.get(
        "/api/research-papers",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert listing.status_code == 200
    assert [item["id"] for item in listing.get_json()["papers"]] == [paper["id"]]

    download = client.get(
        paper["original_url"],
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert download.status_code == 200
    assert download.data == b"%PDF-1.7\npaper"
    assert "attachment" in download.headers["Content-Disposition"]
    assert "review_paper.pdf" in download.headers["Content-Disposition"]

    forbidden = client.get(
        paper["original_url"],
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert forbidden.status_code == 404
    assert other_id != owner_id


def test_research_paper_upload_rejects_missing_metadata_and_non_papers(research_client):
    client, _, _ = research_client
    token = _login(client, "research-owner@example.com")

    missing_topic = client.post(
        "/api/research-papers/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "A paper",
            "file": (BytesIO(b"%PDF-1.7\npaper"), "paper.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert missing_topic.status_code == 400
    assert "topic is required" in missing_topic.get_json()["error"]

    invalid_type = client.post(
        "/api/research-papers/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "A paper",
            "topic": "Research",
            "file": (BytesIO(b"not a pdf"), "paper.txt"),
        },
        content_type="multipart/form-data",
    )
    assert invalid_type.status_code == 400
