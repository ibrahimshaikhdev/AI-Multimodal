# Multimodal AI Research & Medical Report Intelligence

A Flask-based platform for patient record management, medical document processing, OCR, structured medical extraction, report comparison, research indexing, and grounded AI Q&A.

## Project purpose

This application is designed to help teams work with:

- medical document uploads
- OCR and text extraction
- structured patient/report data
- AI-based report summaries
- health timeline and comparison workflows
- research paper indexing and semantic search
- secure user access and audit-friendly operations

> This project is intended for research, experimentation, and internal team workflows. It is not a replacement for clinical judgment.

## Tech stack

- Python
- Flask
- SQLAlchemy
- PostgreSQL
- HTML / JavaScript frontend shell
- pytest for testing

## Repository status

This repo is being prepared as a team-friendly GitHub project. The local environment is isolated from secrets and personal machine-specific settings.

## Local setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

```powershell
cd "c:\Users\Fahad\OneDrive\Desktop\AI Multimodal"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Copy the example environment file and fill in your local settings:

```powershell
Copy-Item .env.example .env
```

5. Start the backend:

```powershell
python -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

6. Open the frontend in a browser:

```text
frontend/index.html
```

## Environment variables

The project uses values from `.env`. Keep real database credentials and secrets out of version control.

Example values are included in [.env.example](.env.example).

## Testing

Run the relevant checks:

```powershell
python -m pytest backend/tests/test_health.py backend/tests/test_user_model.py backend/tests/test_auth_register.py backend/tests/test_registration_ui.py -q
```

## Team workflow

- create a branch for each feature
- keep changes small and reviewable
- submit pull requests before merging
- do not commit actual secrets or environment files

## GitHub handoff

This project is ready for a GitHub repository once the remote URL is configured and the team has access to the repo.

## License

This project is currently set up for internal team use. Add a license before public release if needed.
