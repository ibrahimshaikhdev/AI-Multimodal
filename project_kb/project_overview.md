# Project knowledge

## Purpose
The project is a Flask web app that organises medical reports and research papers, extracts structured data with OCR, compares reports over time, and answers questions about research papers using RAG. It supports information retrieval and does not diagnose.

## Report comparison
Report comparison: two reports of one patient are ordered by date. Parameter differences (old value, new value, change, normal/high/low) are computed in Python; the AI only writes a short summary.

## Research comparison
Research comparison compares uploaded papers on methodology, dataset, model, results, limitations and future work and shows them in a table. Long papers are read in sections and merged before comparing.

## Research RAG
Research RAG splits uploaded papers into chunks, retrieves the most relevant chunks for a question, and answers only from them with citations like [S1].

## Technology stack
Tech stack: Python, Flask, PostgreSQL, SQLAlchemy, PyMuPDF, Tesseract OCR, scikit-learn TF-IDF retrieval, JWT authentication, and a plain HTML/JavaScript frontend.

## Privacy
Privacy: the project runs its own fine-tuned open model locally, so patient and project data is not sent to third-party AI services. Only synthetic or public data is used for training.

## Limitations
Limits: AI output can be wrong and needs human review. The project is a research prototype and is not for clinical use or diagnosis.

## OCR
OCR: PDFs are read with PyMuPDF, and scanned pages or images (JPG, PNG) fall back to Tesseract OCR. The extracted text is stored in the database.

## Health timeline
Health timeline: a patient's documents and extracted events are arranged chronologically.
