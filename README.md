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

## Workspace libraries

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
to the signed-in account. Library search matches saved metadata only; full-text
indexing, semantic search, and generated research answers are not implemented.

## Tech stack

- Python
- Flask
- SQLAlchemy
- PostgreSQL
- HTML / JavaScript frontend shell
- pytest for testing

## Brain/head CT research analysis

The Brain/head CT subtype uses the free, MIT-licensed
[`ianpan/ct-head-hemorrhage-detection`](https://huggingface.co/ianpan/ct-head-hemorrhage-detection)
model locally. The pinned upstream inference implementation is included at
`backend/services/ct_head_inference.py`; its license is in
`backend/services/CT_HEAD_MODEL_LICENSE.txt`.

Upload either one ZIP containing a single non-contrast head CT DICOM series or
a 3D `.nii`/`.nii.gz` volume with values in Hounsfield units after header
scaling. Single JPG/PNG screenshots, multiframe DICOM, and arbitrary 2D
images are not supported for CT analysis. DICOM slices require rescale slope
and intercept metadata. Configure `MAX_CT_UPLOAD_SIZE` (default 512 MiB) for
the largest accepted CT file.

The model returns series and per-slice scores for epidural,
intraparenchymal, intraventricular, subarachnoid, subdural, and any
hemorrhage. It also generates a qualitative localization overlay for the
slice with the highest model `any` score. All results are AI/model output from
experimental research software, **not confirmed medical diagnoses**; they
must not be used to diagnose, exclude, triage, or manage hemorrhage.

## Chest CT slice segmentation (experimental)

The Chest CT subtype uses the supplied Keras model at
`models/chest ct scan model/best_chest_ct_model.keras`. It accepts one
model-ready JPG/PNG slice, not a complete 3D CT series. The slice must first
be converted from CT Hounsfield units by clipping to `[-1000, 400]`, scaling
with `(HU + 1000) / 1400`, and exporting the resulting grayscale values as an
image. The web service converts that image to grayscale, scales encoded pixel
values to `[0, 1]`, and resizes it to `256 × 256`.

The two output channels are Ground-glass opacity and Consolidation. The UI
shows a qualitative mask overlay and the number of pixels passing the
notebook's `0.5` threshold; these are not confidence estimates, physical
measurements, severity scores, or diagnoses. These terms describe image
patterns, not their cause, and the output does not determine COVID-19 status.

The supplied notebook reports test Dice of `0.1711`, which is low; its split
is a random image-level split rather than a documented patient-level
evaluation. This result has not been independently reproduced. Dataset
provenance and applicable licenses were not supplied with the model bundle.
Treat this integration as a research prototype only; masks may be inaccurate
and must not be used for clinical decisions.

## Interpreting MRI and dental model outputs

The Brain MRI model has four classes in the confirmed training order: Glioma,
Meningioma, No tumor, and Pituitary tumor. The API and UI display those names
and the model's scores (including mapping records saved with the former numeric
class names). Scores are experimental outputs, not necessarily calibrated
probabilities and not confirmed diagnoses.

Spine MRI uploads use the locally downloaded
[`mrimperium/Lumbar-Spine-Degenerative-Classification`](https://huggingface.co/mrimperium/Lumbar-Spine-Degenerative-Classification)
checkpoint (`models/lumbar_spine_mrimperium_best.pth`, 139,046,608 bytes;
SHA-256 `ffb9a1761f7530203eda289c438aaf7a6fd73c0c582774cebbf8a465f1c79007`).
It is a ResNet-18 model with a custom head and emits nine sigmoid scores: three
independent sigmoid scores from 0 to 1 for each combination of spinal canal
stenosis, neural foraminal narrowing, or subarticular stenosis and
Normal/Mild, Moderate, or Severe. These scores are not normalized to sum to
one within each condition.

The model accepts a **single JPG/PNG 2D image**, resized to 224×224 and
ImageNet-normalized. It is not a full-series or multi-view workflow. The
checkpoint's public README does not document validation performance or
training provenance, and the inference labels and preprocessing come from
its sample application. Scores are raw model outputs, not established
calibrated probabilities or medical diagnoses. Use only for experimental
demonstrations; professional review is required. PyTorch and timm are already
declared in `requirements.txt`; the checkpoint is downloaded locally and is
excluded from Git.

The Dental X-ray endpoint uses the detector from the free
[`Enosh729/oralguard` model card](https://huggingface.co/Enosh729/oralguard).
Its documented labels are caries, deep caries, periapical lesion, and impacted
tooth. Detector scores are not calibrated probabilities. The model card
reports training on 678 DENTEX 2023 panoramic images and per-class mAP@50 of
0.544, 0.431, 0.263, and 0.955 respectively; these dataset metrics are not
patient-specific accuracy or clinical validation. The UI presents detections
as candidate regions only.

## Obstetric ultrasound view classification

The Obstetric ultrasound subtype accepts one JPG/PNG image and uses the
[`Beijuka/ultrasound_plane_classification-swin-all-planes-class-weight-v1`](https://huggingface.co/Beijuka/ultrasound_plane_classification-swin-all-planes-class-weight-v1)
Swin-Base checkpoint. Its weights are 347,527,516 bytes and are expected at
`models/obstetric_ultrasound_swin/model.safetensors`; the checkpoint is
excluded from Git. The expected SHA-256 is
`923db1f77d315b0d302f274e2a479f5265ceedb73c83b002becfc2b35fb790de`.
Download it from the model's [direct checkpoint URL](https://huggingface.co/Beijuka/ultrasound_plane_classification-swin-all-planes-class-weight-v1/resolve/main/model.safetensors?download=true)
and save it to that path. The model card declares Apache-2.0 for the model. The model
uses one RGB image resized to 224×224 and ImageNet normalization; the
processor and class configuration are stored next to the weights.

The nine output labels are Placenta, Fetal brain, Fetal femur, Maternal cervix,
Fetal thorax, Fetal abdomen, Other, Fetal spine, and Fetal heart rate. This is
only experimental image-view classification: it does not assess fetal health
or diagnose a condition. The model card reports 87.8% accuracy and 84.1%
macro F1 on a 196-image evaluation split. Those publisher-reported results
are not independent or clinical validation, and some labels had very few
evaluation examples. Dataset access/provenance could not be independently
verified. Transformers 5.0.0 is required in addition to the existing PyTorch
and safetensors dependencies.

## Abdominal aorta POCUS segmentation

The Ultrasound / Abdomen option accepts one JPG or PNG and runs the
[`sumit-ai-ml/POCUS_Aorta_segmentation`](https://huggingface.co/sumit-ai-ml/POCUS_Aorta_segmentation)
YOLOv8 segmentation checkpoint on CPU. It is suitable only for abdominal
point-of-care ultrasound images showing the aorta (the publisher describes
short-axis aorta views); it is not a general abdominal scan analyzer and is
not an aneurysm detector. A returned colored overlay marks the model's
aorta-like segmentation. No mask means only that the model returned no mask;
it does not establish that the aorta is absent or abnormal.

The 6,740,216-byte `best.pt` checkpoint is expected at
`models/abdominal_aorta_ultrasound_best.pt` and is excluded from Git. Its
expected SHA-256 is
`ea582b9cc377088c20b336cc6047ffb6805d77283281ef7fdcfa641079639251`.
Download it from the [Hugging Face checkpoint URL](https://huggingface.co/sumit-ai-ml/POCUS_Aorta_segmentation/resolve/main/best.pt?download=true)
and save it to that path. The model repository declares Apache-2.0 and lists
one segmentation class, `Aorta`. The model card reports Dice 0.88, mIoU 0.85,
and CPU inference around 0.05 seconds; these are publisher-reported values,
not independently reproduced or clinically validated results. The
publisher's research describes a small, geographically limited dataset and
calls for broader validation. The inference runtime dependency
`ultralytics==8.4.173` is AGPL-3.0 licensed; review its terms for the intended
deployment.

All output is experimental segmentation only, not a diagnosis. Validate with
appropriate experts and representative images before research evaluation or
any clinical use.

## Cardiac echocardiogram view classification

The Ultrasound / Cardiac/echocardiogram option accepts one JPG or PNG still
image and uses the 47-view classifier from
[`thrive-2025/EchoView47`](https://huggingface.co/thrive-2025/EchoView47),
part of [EchoForge](https://github.com/thrive-centre/EchoForge). It labels the
image view (for example, an apical four-chamber view or a parasternal
long-axis view); the UI also explains that term in plain language. It does not
assess cardiac function, detect disease, or determine whether the heart is
normal. The output is experimental and is not a diagnosis.

The original Keras 2.15 HDF5 checkpoint is 251,648,352 bytes with SHA-256
`e74c6581b7da9acf82d9c1545543a13777fee544dc8b3cebafd9f74587c132a2`. It was
converted to the Keras 3 archive used by this app at
`models/echoview47/EchoView47.keras` (84,339,780 bytes; SHA-256
`5946e975664ace4de8b11385bb8d73bf4df9df2b9c1adfe51e243680cb673fe8`).
The converted checkpoint is excluded from Git. The conversion recreated the
224×224 RGB Rescaling(1/255) + Xception + global-average-pooling + 47-way
softmax architecture and copied the original weights. Its outputs on the
attributed four-chamber sample differed from the original by at most
1.35e-7. Runtime uses the existing `tensorflow-cpu==2.20.0` / Keras 3
dependency and safe-mode model loading.

On one visually identifiable apical four-chamber public-domain sample, the
top-ranked class was `a4ch-full` (“Apical 4-chamber view”) with an uncalibrated
model output of 22.8%. In plain language, that label means the image is viewed
from the tip of the heart and shows all four chambers together. This single smoke test is
not an accuracy estimate or independent validation. EchoForge's classifier
documentation reports 94.1% accuracy on its TTE47 test split; that is
publisher-reported and not clinical validation. EchoForge's classification
documentation states CC BY-NC-SA 4.0; review the license before commercial use
or redistribution.

Add the converted checkpoint at the documented path to enable inference. The
classifier analyzes still images only, not echocardiogram video or DICOM.

## Carotid ultrasound segmentation

The Ultrasound / Vascular option accepts one JPG or PNG still image and runs
the local Keras U-Net checkpoint at
`models/vascular carotid/best_carotid_ultrasound_model.keras`. It converts
the image to grayscale, resizes it to 256×256 using area interpolation,
normalizes pixels to 0–1, and thresholds the single-channel mask at 0.5. The
checkpoint is excluded from Git. The UI shows the predicted carotid-like
region overlay without displaying a model score.

The supplied model notes report test Dice 0.3921 and IoU 0.4073; these are
unverified model-development metrics, not clinical validation or patient
specific accuracy. In a local smoke test, the supplied sample produced a mask
of only two pixels at the configured threshold, so the overlay may be
effectively invisible for some inputs. An empty or tiny mask does not mean
that the artery is absent. The output does not assess stenosis, plaque,
blockage, DVT, blood flow, or vascular disease. Treat every result as an
experimental mask that may be inaccurate, not a confirmed medical diagnosis.

The accompanying notes identify the Kaggle dataset as
`orvile/carotid-ultrasound-images`; dataset licensing and provenance were not
independently verified.

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

## Environment variables

The project uses values from `.env`. Keep real database credentials and secrets out of version control.

Example values are included in [.env.example](.env.example).

## Cloud text generation

The shared AI service uses Gemini first and OpenRouter as a fallback. Configure
`GEMINI_API_KEY` and `OPENROUTER_API_KEY` in `.env` (never commit real keys).
`AI_PROVIDER` and `AI_FALLBACK_PROVIDER` select the order; `GEMINI_MODEL`,
`OPENROUTER_MODEL`, and `AI_TIMEOUT_SECONDS` configure models and request
timeouts. The example OpenRouter model is a free-tier route when available;
provider quotas, pricing, and availability are controlled by the providers and
are not guaranteed to remain free or continuously available.

Report summaries and research Q&A use the configured AI provider. Report
comparison uses explicit extracted values plus local embedding similarity;
research comparison aligns locally retrieved passages and their citations.
Those comparison workflows do not call a generative AI API. The embedding
model runs locally and may download its weights on first use. Similarity is a
text-matching aid, not a probability or evidence of clinical or scientific
equivalence.

The `/api/ai/test` endpoint requires a signed-in user and accepts:

```json
{"prompt": "Say hello in one sentence."}
```

It returns a generated answer or a JSON error if both configured cloud
providers fail or are unavailable. Summaries, research Q&A, and future
generative AI workflows use the same provider configuration and fallback
service.

## Testing

Run the relevant checks:

```powershell
python -m pytest backend/tests/test_health.py backend/tests/test_user_model.py backend/tests/test_auth_register.py backend/tests/test_registration_ui.py -q
python -m pytest backend/tests/test_research_papers_api.py backend/tests/test_report_upload_api.py backend/tests/test_scans_api.py -q
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
