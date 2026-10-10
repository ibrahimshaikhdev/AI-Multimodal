# Implementation Modules
## Multimodal AI Research & Medical Report Intelligence

> **Execution rule:** Implement one module at a time. Give Copilot/AI coding agent the module ID and tell it to implement only that module.

## Standard Command

```text
Read PROJECT_OVERVIEW.md and IMPLEMENTATION_MODULES.md.
Inspect the existing repository.
Implement Module MXX only.
Do not implement later modules.
Run relevant tests and report files changed, results, assumptions and blockers.
```

---

# PHASE 0 — FOUNDATION

## M00 — Project Foundation
**SRS:** Architecture / Technical Requirements

Build the minimum working foundation:

- Flask application
- Environment configuration
- PostgreSQL connection
- SQLAlchemy
- Basic project structure
- Basic health endpoint
- Frontend connection
- `.env.example`
- `.gitignore`
- Logging

**Done when:** Backend starts, database configuration works and frontend/backend can communicate.

---

# FR-01 — USER REGISTRATION

## M01 — User Model
Create the User database model.

Include identity, authentication, role/status and timestamps required by the application.

**Done when:** User table can be created successfully.

## M02 — Registration API
Implement:

```text
POST /api/auth/register
```

Add validation, duplicate handling and password hashing.

**Done when:** A valid user can register and plaintext passwords are never stored.

## M03 — Registration UI
Create the registration page/form and connect it to the API.

**Done when:** User can register through the UI.

---

# FR-02 — AUTHENTICATION & AUTHORIZATION

## M04 — Login API
Implement:

```text
POST /api/auth/login
```

Handle credentials and token/session creation.

**Done when:** Valid credentials authenticate and invalid credentials fail safely.

## M05 — Logout & Session Lifecycle
Implement:

```text
POST /api/auth/logout
```

Handle the selected JWT/session lifecycle.

**Done when:** Authentication can be terminated correctly.

## M06 — Authentication Middleware
Create reusable protection for authenticated endpoints.

**Done when:** Protected endpoints reject unauthenticated users.

## M07 — Role Authorization
Implement role/permission checks.

**Done when:** Users cannot perform actions outside their permissions.

## M07A — Authentication UI
Build the login and logout experience using the existing authentication APIs.

Include:

- Login form connected to `POST /api/auth/login`
- Logout action connected to `POST /api/auth/logout`
- Registration and login views that users can switch between
- A basic signed-in landing view; patient dashboard features remain in M10
- Loading, validation, API error and success states
- Keep the access token in memory only; do not store it in `localStorage` or `sessionStorage`

**Done when:** A user can register, log in, see the signed-in view, log out, and return to the login view. Reloading the page ends the browser-side session and requires logging in again.

---

# FR-03 — PATIENT MANAGEMENT

## M08 — Patient Model
Create the Patient model and relationships.

**Done when:** Patient records can be persisted.

## M09 — Patient CRUD API
Implement:

```text
GET/POST /api/patients
GET/PUT /api/patients/<id>
```

**Done when:** Authorized users can create, view and update patients.

## M10 — Patient UI
Build patient list, details, create and edit screens.

**Done when:** Patient management works end-to-end.

---

# FR-04 — MEDICAL DOCUMENT UPLOAD

## M11 — Medical Report Model
Create the Medical Report model.

Include patient relationship, type, date, file reference, extracted text and processing status.

**Done when:** Reports can be stored against patients.

## M12 — File Validation & Storage
Support:

```text
PDF
DOCX
JPG
PNG
```

Add size/type validation and safe storage.

**Done when:** Valid files are accepted and unsafe/unsupported files are rejected.

## M13 — Report Upload API
Implement:

```text
POST /api/reports/upload
```

**Done when:** Authorized users can upload a validated report.

## M14 — Report Upload UI
Build the frontend upload flow.

**Done when:** A user can select patient + file and upload successfully.

---

# FR-05 — OCR & TEXT EXTRACTION

## M15 — PDF Text Extraction
Use PyMuPDF or the existing approved parser.

**Done when:** Text-based PDFs produce usable text.

## M16 — OCR Service
Use Tesseract or EasyOCR for scanned/image documents.

**Done when:** Representative scanned documents produce usable text.

## M17 — OCR Preprocessing
Add useful image preprocessing such as resizing, grayscale, denoising, thresholding or deskewing where required.

**Done when:** OCR quality improves on representative samples.

## M18 — Text Cleaning
Normalize OCR/parser output while preserving useful structure.

**Done when:** Downstream NLP receives clean consistent text.

## M18A — Extraction Workflow and Results UI
Run extraction and text cleaning when a supported report is uploaded, persist the result and processing status, and show report history and extracted text in the patient view.

**Done when:** The uploader can view saved extracted text for selectable-text PDFs, scanned PDFs, images, and DOCX files, with a clear status when no text can be extracted.

---

# FR-06 — MEDICAL INFORMATION EXTRACTION

## M19 — Medical NLP Service
Create a model-independent medical NLP service interface.

Possible model:

- BioBERT
- PubMedBERT
- ClinicalBERT
- Equivalent transformer

**Done when:** NLP can be called without the application being tightly coupled to one model.

## M20 — Report Metadata Extraction
Extract explicitly labeled report dates and standard sections (such as Findings and Impression) from cleaned text. Preserve observation wording and its section; do not infer diagnoses or missing values. This initial implementation is deterministic and model-free so it can run on the current hardware; an NLP provider can improve coverage later without changing the output contract.

**Done when:** The upload API and patient report view return/display source-grounded dates, sections and verbatim observation excerpts from sample reports.

## M21 — Parameter / Value / Unit Extraction
Extract explicit numeric rows only when both a parameter and recognized unit are present. Preserve the source line and report date; leave confidence unscored for deterministic matches instead of presenting a fabricated probability. Unsupported or ambiguous values are omitted.

Return:

```text
Parameter
Value
Unit
Date
Confidence
Source
```

**Done when:** Extracted parameter rows and their source lines are available in the upload API and patient report view.

## M22 — Persist Extracted Information
Persist report parameter rows in a child table linked to the source report. Use a report-scoped fingerprint to make repeated processing idempotent; retain the exact source excerpt and keep confidence nullable when no calibrated score exists.

**Done when:** Extracted data is stored with the correct report, survives later reads, and repeated extraction does not create duplicate rows.

## M22A — Separate Scan Library and Original File Viewer (Implemented)
The patient view has a Scans section separate from written medical reports. Users can upload PDF, JPG/JPEG, and PNG scan files, associate them with a patient, and securely open the exact original through an authenticated endpoint. The region selector changes with modality: MRI offers Brain, Knee, Spine, Cardiac, and Other; CT offers Brain/head, Chest, Abdomen/pelvis, Spine, Cardiac, and Other; X-ray offers Chest, Bone/joint, Spine, Dental, and Other; ultrasound offers Abdomen, Obstetric, Cardiac/echocardiogram, Vascular, Thyroid, and Other. The API enforces the same taxonomy. Other is stored for reference but is explicitly ineligible for prediction. The Reports section also has an owner-authenticated original-file viewer alongside extracted text and parameter results. Named regions remain awaiting a compatible model.

The initial upload does not support DICOM study/series ingestion and does not perform image analysis. Do not claim per-scan "accuracy": a model's per-case score/confidence is distinct from accuracy measured on a labeled evaluation dataset. Add model predictions only after a selected model and its supported input format are integrated.

**Done when:** Users can upload scans separately from reports, see their patient-scoped scan list, and securely open original report and scan files; no diagnosis is implied before model integration.

## M22B — Reports and Scans Workspace UI (Implemented)
Present Reports and Scans as separate accessible tabs within the shared patient workspace. Preserve the patient context and existing upload/history flows; hide the document workspace while creating or editing a patient. Keep scan prediction status explicit until a compatible model is connected.

**Done when:** Users can switch between Reports and Scans without stacking both workflows, and keyboard navigation exposes the selected panel correctly.

---

# FR-07 — AI REPORT SUMMARY

## M23 — Summary Service
Create a report summarization service using the selected approved LLM/model.

Input:

```text
Source report text
+
Structured information
```

Output:

```text
Structured summary
```

**Done when:** A supported report generates a source-grounded summary.

## M24 — Summary API
Implement:

```text
POST /api/reports/<id>/summary
```

**Done when:** Summary can be requested through the backend.

## M25 — Summary UI
Display generated summary separately from source content.

**Done when:** User can view the report summary.

**Implemented:** Authenticated users can generate an on-demand, source-grounded summary from a report's saved extracted text using the configured local AI service.

---

# FR-08 — MEDICAL IMAGE ANALYSIS

## M26 — Medical Image Model & Storage
After M22A, define the model-facing image/study record around the actual input/output contract of the selected Kaggle-trained model. Keep model version, task, preprocessing version and evaluation metrics identifiable; do not guess a format or substitute sample confidence for measured accuracy.

**Done when:** Supported images are linked to patient/report records.

## M27 — Image Preprocessing
Prepare the selected supported modality for model inference.

**Done when:** Sample image reaches valid model input format.

## M28 — Computer Vision Integration
Integrate the selected model.

Possible:

- CNN
- ResNet
- EfficientNet
- ViT
- MONAI

Return the model's task-specific output and per-case score only as defined by that model. Keep dataset-level validation metrics separate and label both clearly; never present them as a confirmed diagnosis.

**Done when:** Supported sample runs through the model.

## M29 — Image Analysis Persistence
Create/store Image Analysis results and metadata.

**Done when:** Analysis results are stored and linked correctly.

---

# FR-09 — MULTIMODAL FUSION

## M30 — Unified Multimodal Representation
Create a structure for:

```text
Text context
+
Visual result
+
Source metadata
```

**Done when:** Text and visual outputs can be passed together.

## M31 — Research Context Connector
Allow research-derived context to be included where applicable.

**Done when:** Text, image and research context can coexist with provenance.

## M32 — Multimodal Analysis Service
Build the orchestration service.

```text
Text
 ↓
Vision
 ↓
Research Context
 ↓
Unified Analysis
```

**Done when:** The three information types can pass through one workflow.

---

# FR-10 — REPORT COMPARISON

## M33 — Report Comparison Engine
Compare two reports using explicit extracted parameters, source wording, and
local embedding similarity. Do not infer clinical equivalence or direction of
change from similarity alone.

Compare available:

- Parameters
- Values
- Units
- Findings

**Done when:** Differences are returned without invented values.

## M34 — Report Comparison API
Expose comparison through the Flask API.

**Done when:** Two report IDs can be compared through an API request.

## M35 — Report Comparison UI
Build report selection and structured comparison display.

**Done when:** User can compare two reports.

---

# FR-11 — HEALTH TIMELINE

## M36 — Timeline Service
Generate chronological events from patient reports and extracted dates/events.

**Done when:** Timeline events are ordered correctly.

## M37 — Timeline API
Implement:

```text
GET /api/timeline/<patient_id>
```

**Done when:** Backend returns chronological patient information.

## M38 — Timeline UI
Create a readable chronological timeline.

**Done when:** User can view a patient's history chronologically.

---

# FR-12 — RESEARCH PAPER UPLOAD

## M39 — Research Paper Model
Create Research Paper model and metadata.

**Done when:** Papers can be persisted independently.

## M40 — Research Upload & Parsing
Implement:

```text
POST /api/research/upload
```

Parse uploaded research papers.

**Done when:** Paper upload produces stored text/metadata.

## M41 — Research Chunking
Split extracted paper text into retrieval-friendly chunks while preserving paper/chunk metadata.

**Done when:** Each chunk can be traced back to its paper.

---

# FR-13 — RESEARCH RAG

## M42 — Embedding Service
Generate embeddings using Sentence Transformers.

**Done when:** Research chunks have vector representations.

## M43 — Vector Store
Choose ONE:

- FAISS
- ChromaDB
- Qdrant

Implement storage/retrieval integration.

**Done when:** Chunk vectors can be stored and retrieved.

## M44 — Query Embedding
Convert user queries into the same embedding space.

**Done when:** Query vectors can be generated.

## M45 — Semantic Retrieval
Retrieve relevant research chunks and preserve paper/chunk provenance.

**Done when:** A query returns relevant passages.

## M46 — Context Construction
Construct LLM context from retrieved chunks.

**Done when:** The generation layer receives controlled retrieved context.

## M47 — RAG Answer Generation
Implement:

```text
Question
 ↓
Query Embedding
 ↓
Retrieval
 ↓
Context
 ↓
LLM
 ↓
Grounded Answer
```

**Done when:** Answers use retrieved research context.

## M48 — Research Q&A API
Implement:

```text
POST /api/research/ask
```

**Done when:** Authorized users can ask questions over indexed papers.

## M49 — Research Search API
Implement semantic search:

```text
POST /api/research/search
```

**Done when:** Users can retrieve relevant papers/chunks without requiring LLM generation.

---

# FR-14 — RESEARCH COMPARISON

## M50 — Research Comparison Extraction
Extract/prepare comparison fields:

- Methodology
- Dataset
- Model
- Results
- Limitations
- Future work

**Done when:** Each paper has a structured comparison representation.

## M51 — Research Comparison Engine
Retrieve source passages across the configured fields for two or more
account-owned papers and generate a structured synthesis using the configured
AI provider. Bound the excerpts sent to the provider and validate the returned
paper IDs, titles, and fields. Show retrieved passages and citations. Generate
the local embedding-similarity comparison for the first two selected papers
only if AI generation fails, then display it as a temporary fallback.

If information is unavailable, mark it unavailable. Do not invent it.

**Done when:** A structured AI comparison is produced from retrieved evidence,
or a clear provider error is shown alongside the available local comparison.

## M52 — Research Comparison UI
Allow selecting at least two papers and display the AI field comparison,
overall synthesis, local evidence alignment, retrieved source passages, and
provider errors without hiding the local results.

**Done when:** User can compare papers from the UI.

---

# FR-15 — AI QUERY INTERFACE

## M53 — AI Query Backend
Implement:

```text
POST /api/ai/query
```

Create the controlled query request/response structure.

**Done when:** Authenticated users can submit supported AI queries.

## M54 — Query Context Router
Route a query to appropriate available project context:

```text
Medical Report
Research
Patient Timeline
Other indexed project content
```

Preserve source/context.

**Done when:** The query system selects the correct available context.

## M55 — AI Query Generation
Connect the query context to the selected LLM/model and produce a controlled response.

**Done when:** AI answers are based on available project context rather than unsupported invention.

## M56 — AI Query UI
Create the question-answer interface.

Include:

- Question input
- Loading state
- Answer
- Source/context indication where available
- Error handling

**Done when:** User can submit and receive an AI response through the UI.

---

# FR-16 — AUDIT LOGGING

## M57 — Audit Log Model
Create Audit Log storage.

Track important fields such as:

- Actor
- Action
- Resource
- Timestamp
- Result/status

**Done when:** Audit records can be stored.

## M58 — Audit Logging Service
Create a reusable service for recording security-sensitive and significant document/AI operations.

**Done when:** Other services can create audit entries without duplicating logic.

## M59 — Audit Integration
Integrate logging into key flows:

- Login/logout
- Patient operations
- Document upload
- AI operations
- Research operations
- Important administrative actions

**Done when:** Important operations generate audit entries.

## M60 — Audit View / Admin Endpoint
Create a protected endpoint or UI for authorized users to inspect audit records.

**Done when:** Authorized users can view audit activity without exposing it to unauthorized users.

**Implemented:** `GET /api/audit-logs` returns the signed-in user's events; administrators can inspect all events. The Activity page loads these records from the API.

---

# FINAL INTEGRATION

## M61 — Medical Pipeline Integration

Connect:

```text
Upload
 ↓
OCR / Parsing
 ↓
NLP
 ↓
Database
 ↓
Summary
```

Run an end-to-end medical report test.

---

## M62 — Research Pipeline Integration

Connect:

```text
Research Upload
 ↓
Parsing
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Store
 ↓
Retrieval
 ↓
LLM
```

Run an end-to-end RAG test.

---

## M63 — Multimodal Integration

Connect:

```text
Medical Text
+
Medical Image Result
+
Research Context
 ↓
Multimodal Analysis
```

Run an end-to-end test with supported sample data.

---

## M64 — Security Testing

Test:

- Authentication
- Authorization
- Password handling
- Protected endpoints
- Upload validation
- Access control
- Injection risks
- Sensitive error handling

---

## M65 — Performance / Async Testing

Test:

- Normal API requests
- OCR workload
- Embedding workload
- AI workload
- Celery/Redis jobs where implemented

Verify that long-running work does not unnecessarily block normal API operations.

---

## M66 — Full System Testing

Run:

- Unit tests
- API tests
- Integration tests
- Model checks
- Security tests
- Performance checks

Fix regressions without changing the SRS scope.

---

## M67 — Deployment Preparation

Prepare:

```text
Docker
Gunicorn
Nginx
PostgreSQL
Redis
Celery
Environment configuration
```

Create deployment documentation.

---

## M68 — Final Documentation

Update:

- README
- Setup instructions
- Environment variables
- API documentation
- Architecture notes
- AI model notes
- Testing instructions
- Known limitations

---

# MODULE EXECUTION RULE

For every module:

```text
1. Read PROJECT_OVERVIEW.md
2. Read this module
3. Inspect existing code
4. Implement only this module
5. Run relevant tests
6. Verify existing functionality
7. Report files changed
8. Report test results
9. Report blockers
10. Stop
```

## Important

Do **not** implement all modules in one request.

Example:

```text
Implement M01 only.
```

Then:

```text
Implement M02 only.
```

Then:

```text
Implement M03 only.
```

Continue sequentially.

The AI coding assistant is allowed to create the minimum supporting code required for the requested module, but it must not silently implement the scope of later modules.
