# Multimodal AI Research & Medical Report Intelligence
## Project Overview / AI Coding Context

**SRS:** SRS-MAIR-001  
**Version:** 1.0  
**Backend:** Python + Flask  
**Database:** PostgreSQL  
**Project type:** AI-powered medical-report + research-intelligence platform

> **Purpose of this file:** This is the master context file for GitHub Copilot / Claude Code / other AI coding assistants. Read this file before implementing any module.

---

# 1. What We Are Building

We are building an AI-powered platform that processes and organizes medical and research information.

The system combines:

- Document intelligence
- OCR
- NLP / Medical NLP
- Medical image analysis
- Multimodal fusion
- RAG
- LLMs
- Structured database storage
- Report comparison
- Patient health timeline
- Research-paper search and Q&A
- Audit logging

The system is intended to support information retrieval, document understanding, longitudinal record organization and research analysis.

**Important:** This is not intended to replace clinical judgment or provide autonomous medical diagnosis.

---

# 2. Main Goals

1. Upload and process medical reports and scanned documents.
2. Extract useful structured medical information.
3. Generate source-grounded AI summaries.
4. Analyze supported medical images.
5. Combine text, image and research information.
6. Compare medical reports across dates.
7. Build a chronological patient timeline.
8. Upload and index research papers.
9. Provide semantic research search.
10. Provide RAG-based research Q&A.
11. Compare research papers.
12. Provide a controlled AI query interface.
13. Maintain authentication, authorization and audit logs.

---

# 3. Official SRS Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | User Registration |
| FR-02 | Authentication and Authorization |
| FR-03 | Patient Management |
| FR-04 | Medical Document Upload |
| FR-05 | OCR and Text Extraction |
| FR-06 | Medical Information Extraction |
| FR-07 | AI Report Summary |
| FR-08 | Medical Image Analysis |
| FR-09 | Multimodal Fusion |
| FR-10 | Report Comparison |
| FR-11 | Health Timeline |
| FR-12 | Research Paper Upload |
| FR-13 | Research RAG |
| FR-14 | Research Comparison |
| FR-15 | AI Query Interface |
| FR-16 | Audit Logging |

The second `.md` file decomposes these requirements into implementation modules.

---

# 4. High-Level Architecture

```text
                    ┌─────────────────────┐
                    │      Frontend       │
                    │ React / HTML / JS   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Flask REST API    │
                    │ Auth / CRUD / APIs  │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌────────────┐    ┌────────────┐    ┌────────────┐
      │ Document   │    │ AI / ML    │    │    RAG     │
      │ Intelligence│   │ Services   │    │  Services  │
      └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
            │                  │                 │
            ▼                  ▼                 ▼
       OCR / Parsing      NLP / Vision      Embeddings
                                              Retrieval
                                                LLM
             └─────────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    PostgreSQL       │
                    │ Structured Data     │
                    └─────────────────────┘
```

---

# 5. Core Workflow

```text
User
 ↓
Upload
 ↓
Validation
 ↓
OCR / Parsing
 ↓
NLP / Vision
 ↓
Multimodal Fusion
 ↓
RAG / LLM
 ↓
Structured Results
 ↓
Dashboard / Timeline
```

---

# 6. Technology Stack

## Backend

- Python
- Flask
- Flask REST APIs / Blueprints
- SQLAlchemy

## Database

- PostgreSQL

## Authentication

- JWT or secure session management
- Password hashing
- Role-based authorization

## Frontend

The SRS allows:

- React.js
- OR HTML/CSS/JavaScript

Use whichever frontend is already selected in the repository. Do not replace an existing working frontend without a reason.

## Document Processing

- PyMuPDF
- Tesseract
- EasyOCR

## NLP

- Transformers
- spaCy
- BioBERT
- PubMedBERT
- ClinicalBERT
- Equivalent medical transformer where appropriate

## Computer Vision

- PyTorch
- OpenCV
- MONAI where appropriate
- CNN / ResNet / EfficientNet / ViT depending on supported task and dataset

## Embeddings

- Sentence Transformers

## Vector Database / Store

Choose one:

- FAISS
- ChromaDB
- Qdrant

Do not implement all three unless explicitly required.

## LLM

- Approved external API
- OR self-hosted LLM

The chosen LLM must receive retrieved/contextual information for grounded document/research answers where applicable.

## Async Processing

- Celery
- Redis

Use asynchronous processing for long-running OCR, embedding and AI workloads where appropriate.

## Deployment

- Linux/cloud VM
- Gunicorn
- Nginx
- Docker

## Version Control

- Git
- GitHub

---

# 7. Suggested Backend Structure

```text
backend/
├── app.py
├── config/
├── routes/
├── models/
├── schemas/
├── services/
│   ├── ocr/
│   ├── nlp/
│   ├── vision/
│   ├── rag/
│   ├── llm/
│   └── timeline/
├── ai/
├── utils/
├── uploads/
└── tests/
```

The exact structure can change if the repository already has a better working structure. AI coding agents should inspect the repository first.

---

# 8. Main Data Entities

The SRS identifies:

- User
- Patient
- Medical Report
- Report Parameters
- Medical Image
- Image Analysis
- Research Paper
- Research Chunk
- AI Query
- Audit Log

---

# 9. Medical Report Pipeline

```text
PDF / DOCX / JPG / PNG
        ↓
Validation
        ↓
Parsing
        ↓
OCR if required
        ↓
Text Cleaning
        ↓
Medical Information Extraction
        ↓
Parameter / Value / Unit / Date Extraction
        ↓
PostgreSQL
        ↓
AI Summary
        ↓
Comparison / Timeline / AI Query
```

---

# 10. Research RAG Pipeline

```text
Research Paper
      ↓
Upload
      ↓
PDF Text Extraction
      ↓
Chunking
      ↓
Sentence Transformer Embeddings
      ↓
Vector Store
      ↓
Semantic Retrieval
      ↓
Context Construction
      ↓
LLM
      ↓
Grounded Answer
```

---

# 11. Multimodal Pipeline

The multimodal workflow combines:

```text
Medical Text
     +
Medical Image Analysis
     +
Research Context
     ↓
Unified Analysis
```

Every important result should preserve its source/context.

Do not invent missing medical or research information.

---

# 12. API Areas

## Authentication

```text
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
```

## Patients

```text
GET/POST /api/patients
GET/PUT /api/patients/<id>
```

## Reports

```text
POST /api/reports/upload
GET /api/reports/<id>
POST /api/reports/<id>/summary
POST /api/reports/compare
```

## Images

```text
POST /api/images/upload
POST /api/images/<id>/analyze
```

## Research

```text
POST /api/research/upload
POST /api/research/search
POST /api/research/ask
```

## Timeline

```text
GET /api/timeline/<patient_id>
```

## AI

```text
POST /api/ai/query
```

Routes may be adjusted to fit the existing project structure, but avoid unnecessary breaking changes.

---

# 13. AI Model Rules

Use the SRS as the source of truth.

### Medical NLP

Possible:

- BioBERT
- PubMedBERT
- ClinicalBERT
- Equivalent transformer

### Vision

Possible:

- CNN
- ResNet
- EfficientNet
- ViT
- MONAI

The selected model must match the supported modality, dataset and task.

### OCR

- Tesseract
- EasyOCR

### Embeddings

- Sentence Transformers

### Vector Store

- FAISS
- ChromaDB
- Qdrant

### LLM

- Approved API
- Self-hosted LLM

---

# 14. Security Rules

- Never store plaintext passwords.
- Protect authenticated APIs.
- Validate uploaded files.
- Validate file type and size.
- Protect sensitive data in storage and transit.
- Keep secrets in environment variables.
- Do not expose API keys in frontend code.
- External AI services may only receive data permitted by the privacy policy.
- Important security and administrative operations must be audited.
- Do not expose sensitive information in error messages.

---

# 15. Data / Medical Safety Rules

Development and testing data should be:

- Public
- Synthetic
- Or appropriately de-identified

AI outputs can contain errors and require human review.

Medical-image models are limited to validated modalities and datasets.

The system must not present generated AI output as an autonomous diagnosis.

---

# 16. Async / Reliability Rules

Long-running tasks may use:

```text
Flask
 ↓
Celery
 ↓
Redis
 ↓
AI/OCR Worker
```

Failed background jobs should be detectable and retryable where safe.

Database operations should maintain transactional consistency.

---

# 17. Testing

Required categories:

- Unit tests
- API tests
- Integration tests
- Model evaluation
- Security tests
- Performance tests

Important integration tests:

```text
OCR → NLP → Database
```

```text
Research → Embeddings → Retrieval → LLM
```

---

# 18. Acceptance Targets

The project should be able to demonstrate:

- Secure registration/login/access control
- Medical PDF/document upload
- OCR of scanned documents
- Structured medical information storage
- Source-grounded report summary
- Report comparison
- Chronological health timeline
- Research-paper indexing
- Semantic research search
- RAG-based answers using retrieved context
- Research comparison
- Controlled AI querying
- Audit logs
- Proper success/error responses

---

# 19. AI Coding Assistant Master Instruction

Copy this when starting implementation:

```text
You are the primary coding agent for the Multimodal AI Research & Medical Report Intelligence project.

Read PROJECT_OVERVIEW.md and IMPLEMENTATION_MODULES.md before coding.

Use SRS-MAIR-001 as the source of truth.

First inspect the existing repository and understand its current structure.

Implement ONLY the module I explicitly request.

Do not implement future modules.
Do not redesign the entire project.
Do not create duplicate models, routes or services.
Reuse existing code when appropriate.
Keep AI models behind service interfaces.
Keep secrets in environment variables.
Follow the SRS security and medical-data constraints.
Do not invent medical values, diagnoses, sources or research findings.

After implementation:
1. Run relevant tests/checks.
2. Verify the application still starts.
3. Report files changed.
4. Report what was implemented.
5. Report tests/commands and results.
6. Report assumptions or blockers.

If a required dependency/module does not exist, create only the minimum foundation required for the requested module and clearly report it.
```

---

# 20. Preflight

Before M01:

- Repository initialized
- Python environment created
- Flask installed
- PostgreSQL available
- Environment configuration created
- Frontend initialized
- Git initialized
- Basic backend health endpoint works

Then begin:

```text
M01
↓
M02
↓
M03
...
```

