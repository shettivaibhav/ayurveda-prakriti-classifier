# Ayurvedic Prakriti Classifier

Predict a person's dominant Ayurvedic dosha — **Kapha (K)**, **Pitta (P)**, or **Vata (V)** —
from four standardised hand photographs plus age and sex. The dataset was labelled by a
practising Ayurvedic clinician.

The deployed model is **fully automated** by default: the only inputs taken from the person
are age and sex; everything else is derived from the images. Practitioners can optionally
enter clinical observations in the assessment.

## What is deployed

The React app (`frontend/`) uses a local Python API (`mini/server.py`) to serve
**`mini/unified_prakriti_pipeline.joblib`** — a tuned
**Unified RBF-SVM (Balanced)** classifier over a 70-feature vector from four hand views,
age, and sex. Feature extraction and image quality checks are shared through
`mini/unified_cv.py`.

The earlier CNN checkpoints (`mini/model.pth`, `mini/model_v2.pth`) are kept for reference
but are **not** used by the current app.

## Repository layout

| Path | Purpose |
|------|---------|
| `frontend/` | React user interface for personal assessments, WhatsApp imports, and clinical review. |
| `mini/server.py` | Local API and built React app server; preserves the analysis and clinician workflows. |
| `mini/classifier_service.py` | Shared model loading, four-view inference, and clinical probability modulation. |
| `mini/unified_cv.py` | Four-view feature extraction, clinical marker extraction, and image quality checks. |
| `mini/palm_color.py` | Colour-patch feature extractor used by the earlier color-model experiments. |
| `mini/inference.py` | Original MobileNetV2 3-class architecture + CLI inference. |
| `model_v2/` | Data loading, model architectures, training, and tabular/image experiments. |
| `model_v2/models.py` | `WristClassifier` + `WristMultiViewClassifier` (4-view fusion). |
| `model_v2/avg/` | 4-view averaging experiments and the colour-model pipeline. |
| `model_v2/avg/train_color_model.py` | Trains and saves the deployed `model_color.joblib`. |
| `model_v2/avg/exp_tuned.py` | Nested-CV hyperparameter tuning that selects the deployed model. |

## Quick Start (Master Runner Script)

All project commands (running the web app, test suite, WhatsApp AI pipeline, and PDF documentation) can be launched directly using the master runner:

```bash
# Windows Batch (Interactive Menu or CLI)
run.bat               # Opens interactive menu
run.bat app           # Builds the React UI if needed, then launches the web app
run.bat test          # Runs End-to-End System Tests
run.bat docs          # Generates Technical Documentation PDFs
run.bat pipeline      # Runs WhatsApp Agentic AI Batch Pipeline
run.bat all           # Runs full verification suite (Tests + Docs)

# Or via Python directly:
python run_project.py --app
python run_project.py --test
python run_project.py --all
```

## Running the React app

```bash
npm --prefix frontend ci
npm --prefix frontend run build
python run_project.py --app
```

The app is available at `http://localhost:8501`. On Windows, `run.bat app` builds the
frontend on first launch when Node.js/npm are installed.

For frontend development, run the local Python API and Vite in two terminals:

```bash
python mini/server.py --api-only --port 8000
npm --prefix frontend run dev
```

Then open `http://localhost:5173`.

Install the Python project dependencies in the selected environment before launch:

```bash
python -m pip install -r mini/requirements.txt

# Set SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, and GROQ_API_KEY in the environment
# when those connected services are needed. Existing local secrets.toml files are also read.
```

## Dataset

The dataset (hand photographs, per-subject metadata, and the labelled feature spreadsheet)
is **human-subject data and is not included in this repository**. Training and experiment
scripts expect it locally. Most scripts resolve paths relative to the repository root; place
the data under the repo root (or adjust the `ROOT` in the relevant script) and regenerate the
manifests/embeddings as described in [`HANDOFF.md`](HANDOFF.md).

## Results summary

On the labelled set of 92 subjects (3-class dominant dosha):

- **Deployed colour model** (ExtraTrees on colour patches + age/sex): **~64% accuracy,
  ~53% balanced accuracy**, permutation test **p ≈ 0.005** (significant vs. the ~64%
  always-Kapha majority baseline).
- **Raw-image CNN features** perform near chance for this task, and do not improve with
  4-view averaging or stronger backbones.
- The practitioner's manually-assessed skin/palm features are the strongest signal
  (**~74% accuracy** with a tabular model), but require expert assessment and cannot be
  recovered reliably from uncalibrated phone photos.

See [`HANDOFF.md`](HANDOFF.md) for the full engineering notes and next steps.
