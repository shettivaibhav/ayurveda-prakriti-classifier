"""Permutation test on the best image averaging config (MobileNetV2 mean-pooled
subject embeddings + StandardScaler/PCA/SVM-RBF). Answers: is the best image model
actually above chance? Subject-level, balanced-accuracy, 200 label permutations."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

FEAT = Path(__file__).resolve().parent / "features" / "mobilenet_v2.npz"


def main():
    z = np.load(FEAT, allow_pickle=True)
    emb, subj, label = z["emb"], z["subject_id"].astype(str), z["label"].astype(str)
    subjects = np.array(list(dict.fromkeys(subj)))
    X = np.stack([emb[subj == s].mean(0) for s in subjects])          # subject mean embedding
    y = LabelEncoder().fit_transform([label[subj == s][0] for s in subjects])

    pipe = Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=40, random_state=42)),
                     ("svm", SVC(kernel="rbf", C=2.0, gamma="scale",
                                 class_weight="balanced", random_state=42))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    score, perm, p = permutation_test_score(
        pipe, X, y, cv=cv, scoring="balanced_accuracy",
        n_permutations=200, n_jobs=-1, random_state=42)
    print(f"MobileNetV2 meanpool + SVM-RBF (subject-level, balanced-acc)")
    print(f"  observed balanced-acc : {score:.3f}")
    print(f"  permutation mean/std  : {perm.mean():.3f} / {perm.std():.3f}")
    print(f"  p-value               : {p:.3f}")
    print(f"  -> {'ABOVE chance (p<0.05)' if p < 0.05 else 'NOT distinguishable from chance (p>=0.05)'}")


if __name__ == "__main__":
    main()
