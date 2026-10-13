# How to use the application

Start the Flask backend from the repository root and open the local application in a browser. Sign in, then use the workspace navigation to manage patients, upload reports or scans, and open research tools.

Model checkpoints are stored with Git LFS. Install Git LFS before cloning so
the checkpoint files are downloaded with the repository:

```powershell
git lfs install
git clone https://github.com/ibrahimshaikhdev/AI-Multimodal-.git
cd AI-Multimodal-
git lfs pull
```

If the repository is already cloned, install Git LFS and run `git lfs pull`
from the repository root. The tracked patterns cover model weight files only;
sample images, notebooks, and uploaded patient files are not added by these
rules.

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
$env:SKIP_MODEL_WARMUP="1"
.\.venv\Scripts\python.exe -m flask --app backend.app run --host 127.0.0.1 --port 8000 --no-reload
```

Open **http://127.0.0.1:8000** in a browser. Flask serves the frontend and API
from the same origin; no second frontend port or hardcoded API host is needed.

The default database is `instance/medical_ai.db`; uploaded documents and scans
are stored under `instance/uploads/`. Back up the whole `instance/` folder.
Set `DATABASE_URL` only when intentionally selecting another persistent
database. If a legacy relative `medical_ai.db` or `app.db` does not exist but
`instance/medical_ai.db` does, the app uses the existing database rather than
silently creating a new empty database next to the project.

## Assistant context

The own-model Assistant retrieves project information from the Markdown and text files in `project_kb/`. If the user selects a report or patient timeline, the Assistant retrieves that signed-in user's selected report text or dated patient records from the database and includes them as PATIENT DATA, even when the question does not name the patient. If report text has not been extracted, it runs the existing document text extraction service on the uploaded file before answering. Patient records are retrieved live and are not copied into this shared knowledge base.
