from scripts.build_kb import _markdown_section, build_project_kb


def test_markdown_section_preserves_nested_headings_until_next_peer():
    text = (
        "# First section\nFirst paragraph.\n\n"
        "## Nested section\nNested paragraph.\n\n"
        "# Second section\nOther content."
    )

    assert _markdown_section(text, "First section") == (
        "First paragraph.\n\n## Nested section\nNested paragraph."
    )


def test_build_project_kb_derives_sources_and_preserves_notes(tmp_path):
    (tmp_path / "backend" / "routes").mkdir(parents=True)
    (tmp_path / "backend" / "models").mkdir(parents=True)
    (tmp_path / "frontend").mkdir()
    (tmp_path / "README.md").write_text(
        "## Project purpose\nProject purpose text.\n\n"
        "## Workspace libraries\nWorkspace text.\n\n"
        "## Tech stack\nPython.\n\n## Local setup\nSetup text.",
        encoding="utf-8",
    )
    (tmp_path / "PROJECT_OVERVIEW.md").write_text(
        "# 1. What We Are Building\nA source-grounded research application.\n\n"
        "# 2. Main Goals\n1. Answer from sources.\n\n"
        "# 3. Official SRS Functional Requirements\n"
        "| FR-01 | Patient records |\n",
        encoding="utf-8",
    )
    (tmp_path / "frontend" / "index.html").write_text(
        '<button data-workspace-nav="patients">Patients</button>',
        encoding="utf-8",
    )
    (tmp_path / "backend" / "routes" / "example.py").write_text(
        '@bp.post("/demo")\ndef create_demo():\n    pass\n',
        encoding="utf-8",
    )
    (tmp_path / "backend" / "models" / "example.py").write_text(
        "class Example(db.Model):\n"
        '    __tablename__ = "examples"\n'
        "    id = db.Column(db.Integer, primary_key=True)\n",
        encoding="utf-8",
    )
    knowledge_directory = tmp_path / "project_kb"
    knowledge_directory.mkdir()
    notes = knowledge_directory / "notes.md"
    notes.write_text("hand-edited notes stay intact", encoding="utf-8")

    build_project_kb(tmp_path)

    overview = (knowledge_directory / "project_overview.md").read_text(
        encoding="utf-8"
    )
    features = (knowledge_directory / "features.md").read_text(encoding="utf-8")
    endpoints = (knowledge_directory / "api_endpoints.md").read_text(
        encoding="utf-8"
    )
    data_model = (knowledge_directory / "data_model.md").read_text(
        encoding="utf-8"
    )
    assert "A source-grounded research application." in overview
    assert "1. Answer from sources." in overview
    assert "| FR-01 | Patient records |" in features
    assert "`POST /api/demo`" in endpoints
    assert "`id` (db.Integer)" in data_model
    assert notes.read_text(encoding="utf-8") == "hand-edited notes stay intact"
