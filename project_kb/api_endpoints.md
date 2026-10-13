# API endpoints

These routes are discovered from Flask route decorators in `backend/routes/`. All listed API routes are registered under `/api`.

## Ai

- `POST /api/ai/query` — `query_project_context`
- `POST /api/ai/test` — `test_ai_connection`
## Audit Logs

- `GET /api/audit-logs` — `list_audit_logs`
## Auth

- `GET /api/auth/me` — `current_user`
- `POST /api/auth/login` — `login_user`
- `POST /api/auth/logout` — `logout_user`
- `POST /api/auth/register` — `register_user`
- `PUT /api/auth/me` — `update_current_user`
## Health

- `GET /api/health` — `health_check`
## Local Assistant

- `DELETE /api/ai/local-conversation` — `clear_local_assistant_conversation`
- `GET /api/ai/local-conversation` — `get_local_assistant_conversation`
- `POST /api/ai/query-local` — `query_local_assistant`
## Own Model

- `POST /api/reports/compare-own` — `compare_reports_own`
- `POST /api/research/ask-own` — `ask_research_question_own`
- `POST /api/research/compare-own` — `compare_research_papers_own`
## Patients

- `GET /api/patients` — `list_patients`
- `GET /api/patients/<int:patient_id>` — `get_patient`
- `POST /api/patients` — `create_patient`
- `PUT /api/patients/<int:patient_id>` — `update_patient`
## Reports

- `DELETE /api/reports/<int:report_id>` — `delete_report`
- `GET /api/reports` — `list_reports`
- `GET /api/reports/<int:report_id>/original` — `get_original_report`
- `POST /api/reports/<int:report_id>/extract-text` — `extract_report_text`
- `POST /api/reports/<int:report_id>/summary` — `summarize_report`
- `POST /api/reports/compare` — `compare_reports`
- `POST /api/reports/compare-local` — `compare_reports_local`
- `POST /api/reports/upload` — `upload_report`
## Research Papers

- `GET /api/research-papers` — `list_research_papers`
- `GET /api/research-papers/<int:paper_id>/original` — `get_original_research_paper`
- `POST /api/research-papers/upload` — `upload_research_paper`
- `POST /api/research/ask` — `ask_research_question`
- `POST /api/research/ask-local` — `ask_research_question_local`
- `POST /api/research/compare` — `compare_research_papers`
- `POST /api/research/compare-local` — `compare_research_papers_local`
- `POST /api/research/search` — `search_research_papers`
## Scans

- `DELETE /api/scans/<int:scan_id>` — `delete_scan`
- `GET /api/scans` — `list_scans`
- `GET /api/scans/<int:scan_id>/analysis/aorta-overlay` — `get_abdominal_aorta_overlay`
- `GET /api/scans/<int:scan_id>/analysis/chest-ct-overlay` — `get_chest_ct_overlay`
- `GET /api/scans/<int:scan_id>/analysis/ct-localization` — `get_ct_localization`
- `GET /api/scans/<int:scan_id>/analysis/overlay` — `get_cardiac_overlay`
- `GET /api/scans/<int:scan_id>/analysis/vascular-ultrasound-overlay` — `get_vascular_ultrasound_overlay`
- `GET /api/scans/<int:scan_id>/original` — `get_original_scan`
- `POST /api/scans/upload` — `upload_scan`
