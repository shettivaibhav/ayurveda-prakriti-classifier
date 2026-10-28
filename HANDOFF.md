# Developer Hand-off Notes

This document is the entry point for anyone continuing the project. It records the current
state, how the pieces fit together, how to set up data/environment, the main empirical
findings, and the known issues to address first.

## 1. Current state

- **Deployed model:** `mini/unified_prakriti_pipeline.joblib` — the four-view **Unified RBF-SVM**
  pipeline, served by the React frontend and `mini/server.py`.
- **Feature extraction:** `mini/unified_cv.py` supplies the 70-feature vector and clinical
  markers used by the React assessment API.
- **Older models (reference only, not used by the app):**
  - `mini/model.pth` — the original single-view MobileNetV2 3-class classifier.
  - `mini/model_v2.pth` — a frozen-backbone MobileNetV2 that averages predictions over the
    4 views.

## 2. Repository map

| Area | What lives there |
|------|------------------|
| `frontend/` | React assessment, WhatsApp batch import, and clinician review interface. |
| `mini/` | Local API, deployed `.joblib`, feature extraction, clinician backend, and original CNN files. |
| `model_v2/` | The rebuilt pipeline: `data.py` / `datasets.py` (data), `models.py` (architectures), `train.py` / `evaluate.py` (training), and `exp_*.py` (tabular, image-fusion, explainability experiments). |
| `model_v2/avg/` | 4-view averaging experiments and the colour-model line: `train_color_model.py` (builds the deployed model), `exp_tuned.py` (nested-CV model selection), `extract_*` (features), `verify_*` (deployment checks), `eval_oldmodel*` (old-vs-new comparisons). |

## 3. Environment & data setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r mini/requirements.txt        # app + colour model
pip install -r model_v2/requirements_v2.txt # torch/experiment pipeline (CPU)
```

**Data is not in the repo** (human-subject palm images + age/sex/health labels). To run the
training/experiment scripts you need to supply it locally. Scripts resolve `ROOT` relative to
the repository root (`Path(__file__).resolve().parents[2]` for `model_v2/avg/*`), so the
simplest setup is to place the dataset folders under the repo root.

Regeneration pipeline once the source spreadsheet + images are available:

1. `python model_v2/extract_all_from_xlsx.py` — extract labels and in-cell/legacy images
   into the per-subject dataset folder.
2. `python model_v2/build_manifest.py` — build the feature matrix + label framings
   (`manifest.csv`).
3. `python model_v2/avg/build_image_dataset.py` — build the per-image, subject-grouped
   manifest (one row per view).
4. `python model_v2/avg/extract_features_multi.py` — cache frozen CNN embeddings per backbone
   (`.npz`, git-ignored).
5. `python model_v2/avg/train_color_model.py` — train and save `mini/model_color.joblib`.
   `python model_v2/avg/exp_tuned.py` re-runs nested-CV model selection and redeploys the
   balanced-accuracy winner.

**Evaluation must use subject-level cross-validation** — all four views of a subject stay in
the same fold. Splitting by image leaks a subject's other views into training and inflates
scores. The scripts already do this; preserve it. Prefer **balanced accuracy** over raw
accuracy because the classes are imbalanced (~64% Kapha), so a majority-class guesser scores
~64% raw while learning nothing.

## 4. Key engineering findings

These come from repeated, leakage-controlled experiments on the 92-subject set and explain
why the deployed model looks the way it does:

- **Raw hand images carry little signal for this task.** Frozen features from five backbones
  (MobileNetV2, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, DINOv2-ViT-S) all land near chance
  on balanced accuracy, and this does **not** improve with 4-view averaging or per-side
  specialist models.
- **The signal is in colour + age/sex.** Sampling fixed colour patches (skin from the
  back-of-hand view, palm from the front view) and adding age/sex gives the best fully
  automated model: **~64% accuracy / ~53% balanced accuracy, p ≈ 0.005**. Winner selected by
  nested cross-validation was ExtraTrees.
- **The two strongest predictors are not recoverable from photos.** Palm colour and vein
  visibility dominate the practitioner's labelling, but both are near chance to recover from
  uncalibrated phone images. Consequently an `image → predicted-features → dosha` pipeline
  (~50%) does **worse** than classifying the image directly (~65%).
- **Dual-dosha labelling has a hard ceiling.** Even the practitioner's own ground-truth
  features predict the 7-class dual Prakriti only ~44%, versus ~74% for 3-class dominant
  dosha. Target 3-class dominant dosha; 7-class dual is not achievable at useful accuracy.

## 5. Known issues / TODO

1. **Rotate the Supabase key.** The project's Supabase URL + publishable key were previously
   committed to an older repository and stored in a local `secrets.toml`. Treat the key as
   compromised: rotate it in the Supabase dashboard, and provide replacement values through
   environment variables or a local secrets file. Never commit real credentials.
2. **Bring your own data.** See section 3 — the dataset is intentionally excluded. Cached
   embeddings (`*.npz`) and manifest CSVs are git-ignored and must be regenerated.
3. **The original training notebook** (`mini/training_pipeline.ipynb`) has had its outputs
   cleared and machine-specific paths genericised; update its `DATA_ROOT` before running.
4. **Accuracy is capped by capture conditions.** The most promising path to higher accuracy
   is standardised image capture (fixed lighting + a colour-reference card) so that palm
   colour and vein visibility become measurable, rather than a larger model.
5. **Example photos:** The React interface gives text guidance for each hand view and does
   not depend on bundled sample photographs.
