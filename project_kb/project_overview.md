# Project overview

This is the Multimodal AI Research & Medical Report Intelligence platform, a Flask web application with a browser-based HTML and JavaScript interface.

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

This application is designed to help teams work with:

- medical document uploads
- OCR and text extraction
- structured patient/report data
- AI-based report summaries
- health timeline and comparison workflows
- research paper indexing and semantic search
- secure user access and audit-friendly operations

> This project is intended for research, experimentation, and internal team workflows. It is not a replacement for clinical judgment.

The authenticated workspace includes dedicated Medical Reports and Medical
Scans views that collect the current user's patient uploads into searchable
libraries. Original files can be opened or downloaded through the existing
patient-scoped APIs. Health Timeline lists a patient's uploaded reports and
scans, ordered by upload time, with report/study dates shown when available.
It does not infer missing medical dates. The library screens also have
direct-entry paths at `/medicalreports.html`, `/scan.html`, and
`/healthtimeline.html`, served by the same authenticated frontend shell.

Research Papers accepts PDF and DOCX uploads with title, topic, publication
date, authors, and journal/source metadata. Paper files and records are private
to the signed-in account. The Research Search and RAG workflows retrieve
source excerpts from uploaded papers and provide citations; Research Comparison
summarizes selected papers. These research features do not provide clinical
advice.

## Main capabilities

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

## Technology stack

- Python
- Flask
- SQLAlchemy
- PostgreSQL
- HTML / JavaScript frontend shell
- pytest for testing

## Safety and privacy

The platform is a research and experimental information-management tool, not a clinical diagnosis system. Patient records are associated with user-owned patient accounts and must be accessed through the authenticated, ownership-checked application workflows.
