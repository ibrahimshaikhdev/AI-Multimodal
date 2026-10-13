# Application features

The signed-in workspace is divided into these frontend sections:

- **⌂ Dashboard** (`dashboard`)
- **♙ Patients** (`patients`)
- **▤ Medical Reports** (`reports`)
- **▧ Medical Scans** (`scans`)
- **⇄ Report Comparison** (`report-comparison`)
- **◷ Health Timeline** (`timeline`)
- **✳ AI Assistant** (`assistant`)
- **▧ Research Papers** (`research-papers`)
- **⌕ Research RAG** (`research-rag`)
- **⇆ Research Comparison** (`research-comparison`)
- **◎ Profile** (`profile`)
- **⚙ Settings** (`settings`)
- **☷ Activity** (`activity`)
- **Medical reports Available Browse and download uploaded documents →** (`reports`)
- **Medical scans Available Browse and download uploaded imaging →** (`scans`)
- **Open patient workspace →** (`patients`)
- **Review medical reports →** (`reports`)
- **Review medical scans →** (`scans`)

## Documented capabilities

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

- Brain/head CT research analysis
- Chest CT slice segmentation (experimental)
- Interpreting MRI and dental model outputs
- Obstetric ultrasound view classification
- Abdominal aorta POCUS segmentation
- Cardiac echocardiogram view classification
- Carotid ultrasound segmentation
- Repository status
- Assistant project knowledge

## SRS functional requirements

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

Medical image-model outputs are experimental and must not be presented as confirmed diagnoses.
