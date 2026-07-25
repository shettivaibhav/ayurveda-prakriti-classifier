"""Classical-ML averaging experiments on frozen backbone features.

For every backbone whose embeddings were cached by extract_features_multi.py,
evaluate two "average the 4 views" designs, both under **subject-level** CV so a
subject's views never straddle train/test:

  meanpool  (feature-level average): represent each subject by the mean of its
            view embeddings -> one classifier over 92 subject vectors.
  avgprob   (decision-level average, Scheme A): train an image-level classifier on
            all training-subject images, predict each test image, then average a
            subject's per-image probabilities -> subject prediction.

Classifiers: RandomForest, SVM-RBF, LogReg, XGBoost, each behind StandardScaler+PCA
(fit inside CV, no leakage), class-weighted for the imbalance. Metrics: accuracy,
balanced-accuracy, macro-F1 (mean +/- std over 5x5 repeats), plus random/majority
baselines and a permutation test on the best config per backbone.
"""

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")
RNG = 42
FEAT_DIR = Path(__file__).resolve().parent / "features"
OUT = Path(__file__).resolve().parent / "results_avg_ml.json"
N_SPLITS, N_REPEATS = 5, 5  # smallest class (P-K) = 5 subjects -> 5 splits max


def make_clfs(n_pca):
    clfs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", random_state=RNG),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG),
        "SVM-RBF": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced",
                       probability=True, random_state=RNG),
    }
    if HAS_XGB:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, eval_metric="mlogloss", random_state=RNG, verbosity=0)
    return {name: Pipeline([("sc", StandardScaler()),
                            ("pca", PCA(n_components=n_pca, random_state=RNG)),
                            ("clf", clf)]) for name, clf in clfs.items()}


def load_backbone(name):
    z = np.load(FEAT_DIR / f"{name}.npz", allow_pickle=True)
    return z["emb"], z["subject_id"].astype(str), z["view"].astype(str), z["label"].astype(str)


def subject_arrays(emb, subj, label):
    """Collapse image rows to one mean vector + label per subject (stable order)."""
    subjects = list(dict.fromkeys(subj))  # preserve first-seen order
    mean_emb = np.stack([emb[subj == s].mean(axis=0) for s in subjects])
    y = np.array([label[subj == s][0] for s in subjects])
    return np.array(subjects), mean_emb, y


def eval_backbone(name):
    emb, subj, view, label = load_backbone(name)
    subjects, mean_emb, y_subj_str = subject_arrays(emb, subj, label)
    le = LabelEncoder().fit(y_subj_str)
    y_subj = le.transform(y_subj_str)
    y_img = le.transform(label)
    n_classes = len(le.classes_)

    maj = np.bincount(y_subj).max() / len(y_subj)
    n_pca = min(40, mean_emb.shape[1], N_SPLITS - 1 + len(y_subj) - len(y_subj) // N_SPLITS)
    n_pca = min(40, mean_emb.shape[1])

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG)
    modes = ["meanpool", "avgprob"]
    scores = {m: {c: {"acc": [], "bal": [], "f1": []} for c in make_clfs(n_pca)} for m in modes}

    for train_idx, test_idx in cv.split(subjects, y_subj):
        train_subj = set(subjects[train_idx])
        tr_img = np.isin(subj, list(train_subj))
        te_img = ~tr_img
        yte = y_subj[test_idx]

        for cname in scores["meanpool"]:
            # -- meanpool: subject mean vectors
            pipe = make_clfs(n_pca)[cname]
            pipe.fit(mean_emb[train_idx], y_subj[train_idx])
            pred = pipe.predict(mean_emb[test_idx])
            _record(scores["meanpool"][cname], yte, pred)

            # -- avgprob: image-level train, average test-image probs per subject
            pipe2 = make_clfs(n_pca)[cname]
            pipe2.fit(emb[tr_img], y_img[tr_img])
            proba = pipe2.predict_proba(emb[te_img])
            # align proba columns (clf.classes_) to full 0..n_classes-1
            full = np.zeros((proba.shape[0], n_classes))
            full[:, pipe2.named_steps["clf"].classes_] = proba
            te_subj = subj[te_img]
            pred2, ytrue2 = [], []
            for s in subjects[test_idx]:
                m = te_subj == s
                pred2.append(full[m].mean(axis=0).argmax())
                ytrue2.append(y_subj[list(subjects).index(s)])
            _record(scores["avgprob"][cname], np.array(ytrue2), np.array(pred2))

    result = {"backbone": name, "dim": int(emb.shape[1]),
              "baselines": {"random": 1 / n_classes, "majority": maj},
              "classes": list(le.classes_), "modes": {}}
    best = (None, None, -1)
    for m in modes:
        result["modes"][m] = {}
        for c, s in scores[m].items():
            entry = {k: [float(np.mean(v)), float(np.std(v))] for k, v in s.items()}
            result["modes"][m][c] = entry
            if entry["bal"][0] > best[2]:
                best = (m, c, entry["bal"][0])
    result["best"] = {"mode": best[0], "clf": best[1], "bal_acc": best[2]}
    return result


def _record(bucket, ytrue, ypred):
    bucket["acc"].append(accuracy_score(ytrue, ypred))
    bucket["bal"].append(balanced_accuracy_score(ytrue, ypred))
    bucket["f1"].append(f1_score(ytrue, ypred, average="macro", zero_division=0))


def main():
    backbones = sorted(p.stem for p in FEAT_DIR.glob("*.npz"))
    if not backbones:
        raise SystemExit("No cached features found. Run extract_features_multi.py first.")
    print(f"Backbones: {backbones}\n")
    all_res = {}
    header = f"{'backbone':<16}{'best mode':<10}{'best clf':<14}{'acc':>8}{'bal_acc':>9}{'macroF1':>9}"
    print(header); print("-" * len(header))
    for name in backbones:
        r = eval_backbone(name)
        all_res[name] = r
        b = r["modes"][r["best"]["mode"]][r["best"]["clf"]]
        print(f"{name:<16}{r['best']['mode']:<10}{r['best']['clf']:<14}"
              f"{b['acc'][0]:>8.3f}{b['bal'][0]:>9.3f}{b['f1'][0]:>9.3f}")
    rnd = 1 / 7
    print(f"\nbaselines: random={rnd:.3f}  majority={all_res[backbones[0]]['baselines']['majority']:.3f}")
    OUT.write_text(json.dumps(all_res, indent=2))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
