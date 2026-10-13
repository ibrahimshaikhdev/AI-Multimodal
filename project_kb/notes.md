# Project notes
- Report Comparison: pick a patient, choose two of their reports, click Extract text, then compare. The app orders the reports by date and itself computes each parameter's old value, new value, change and normal/high/low status; the AI only writes the short summary. Answers can come from Gemini, a local model, or our own model. Report reads use the shared, account-scoped database retrieval service.
- Research Papers: upload PDF or DOCX papers; they are private to the signed-in account.
- Research RAG: ask a question about uploaded papers; the app retrieves the most relevant passages and the AI answers only from them, citing sources like [S1].
- Research Comparison: select two or more papers; the AI compares methodology, dataset, model, results, limitations and future work.
- AI Assistant: routes greetings, project questions, counts, lab-term definitions, and account-scoped patient-record questions to intent-specific handlers. It uses live database retrieval for selected reports, lab values, timelines, scans, and report comparisons; timeline answers are formatted chronologically from saved dated records. New uploads are available on the next request. Patient records are not copied into the project knowledge base. Unmatched questions receive a scope reminder rather than an unsupported medical answer. It does not diagnose.
- Research RAG and Research Comparison use uploaded papers only; patient records are not mixed into research-paper context.
- Patient records are private to the signed-in user.
- Medical image model outputs are experimental and are not confirmed diagnoses.
- Own model: a small Qwen2.5-1.5B model fine-tuned by the team on synthetic data and run locally. The Gemini option sends text to Google; the own model does not.
