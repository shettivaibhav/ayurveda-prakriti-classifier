# Ayurvedic Prakriti Classifier

Predict a person's dominant Ayurvedic dosha — **Kapha (K)**, **Pitta (P)**, or **Vata (V)** —
from four standardised hand photographs plus age and sex. The dataset was labelled by a
practising Ayurvedic clinician.

The deployed model is **fully automated**: the only inputs taken from the person are age and
sex; everything else is derived from the images.

## What is deployed

The Streamlit app (`mini/app.py`) serves **`mini/model_color.joblib`** — a tuned
**Extremely Randomized Trees (ExtraTrees)** classifier over colour-patch features
(skin-colour from the back-of-hand view, palm-colour from the front-of-hand view; RGB +
CIELAB + chroma + redness) combined with age and sex. Feature extraction is shared between
training and inference in `mini/palm_color.py`.

The earlier CNN checkpoints (`mini/model.pth`, `mini/model_v2.pth`) are kept for reference
but are **not** used by the current app.

## Repository layout

| Path | Purpose |
|------|---------|
| `mini/` | Deployed Streamlit app, the deployed colour model, and the original CNN lineage. |
| `mini/app.py` | The app: collects 4 hand images + age/sex, predicts dominant dosha, stores results. |
| `mini/palm_color.py` | Colour-patch feature extractor (shared by app + training). |
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
run.bat app           # Launches Streamlit Web App
run.bat test          # Runs End-to-End System Tests
run.bat docs          # Generates Technical Documentation PDFs
run.bat pipeline      # Runs WhatsApp Agentic AI Batch Pipeline
run.bat all           # Runs full verification suite (Tests + Docs)

# Or via Python directly:
python run_project.py --app
python run_project.py --test
python run_project.py --all
```

## Running the app directly

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r mini/requirements.txt

# Supabase is used to store results. Provide your own credentials:
cp mini/.streamlit/secrets.toml.example mini/.streamlit/secrets.toml
#   ... then edit mini/.streamlit/secrets.toml with your project url + key

streamlit run mini/app.py
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
