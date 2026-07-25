"""Explainability: which of the practitioner's features drive the prediction.

Global feature importance for the tabular RandomForest, via (a) impurity-based
importance and (b) permutation importance (model-agnostic, computed on held-out
CV predictions), aggregated over repeats. Answers the thesis question "does the
model reason like an Ayurvedic practitioner?" -> we expect palm/skin colour and
texture to dominate.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
RNG = 42
OUT = Path(__file__).resolve().parent.parent / "model_v2" / "results_full"
TAB = ["Age", "Sex_M", "Mole", "SC_Fair", "SC_Medium", "SC_Dark",
       "ST_Rough", "ST_Smooth", "ST_Dry", "ST_Oily", "SH_Thick", "SH_Thin",
       "SH_Med", "SHC_Brown", "SHC_Black", "VV_Low", "VV_Med", "VV_High",
       "PC_PinkRed", "PC_PinkWhite", "PC_Brown"]
GROUP = {c: c.split("_")[0] for c in TAB}  # SC_Fair -> SC ; Age -> Age


def importances(X, y_str, tag):
    y = LabelEncoder().fit_transform(y_str)
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample", random_state=RNG)
    # permutation importance on held-out folds (averaged) to avoid train leakage
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    perm = np.zeros(len(TAB))
    for tr, te in skf.split(X, y):
        rf.fit(X[tr], y[tr])
        r = permutation_importance(rf, X[te], y[te], n_repeats=10,
                                   scoring="balanced_accuracy", random_state=RNG,
                                   n_jobs=-1)
        perm += r.importances_mean
    perm /= skf.get_n_splits()
    rf.fit(X, y)
    imp = rf.feature_importances_

    per_feature = sorted(zip(TAB, imp, perm), key=lambda t: -t[2])
    # aggregate to annotation groups (SC, ST, SH, SHC, VV, PC, Age, Sex, Mole)
    grp = {}
    for f, gi, pm in zip(TAB, imp, perm):
        g = GROUP[f]
        grp.setdefault(g, [0.0, 0.0])
        grp[g][0] += gi
        grp[g][1] += pm
    grp_sorted = sorted(grp.items(), key=lambda t: -t[1][1])

    print(f"\n=== {tag}: top features (by permutation importance) ===")
    print(f"{'feature':<14}{'impurity':>10}{'perm_imp':>10}")
    for f, gi, pm in per_feature[:10]:
        print(f"{f:<14}{gi:>10.3f}{pm:>10.3f}")
    print(f"\n=== {tag}: by annotation group ===")
    print(f"{'group':<8}{'impurity':>10}{'perm_imp':>10}")
    for g, (gi, pm) in grp_sorted:
        print(f"{g:<8}{gi:>10.3f}{pm:>10.3f}")
    return {
        "per_feature": [{"feature": f, "impurity": gi, "perm": pm}
                        for f, gi, pm in per_feature],
        "by_group": {g: {"impurity": v[0], "perm": v[1]} for g, v in grp_sorted},
    }


def main():
    df = pd.read_csv(OUT / "manifest.csv")
    X = df[TAB].values.astype(float)
    out = {
        "7class": importances(X, df["y7"].values, "7-CLASS"),
        "3class_dominant": importances(X, df["ydom"].values, "3-CLASS DOMINANT"),
    }
    (OUT / "explain_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved -> {OUT/'explain_results.json'}")


if __name__ == "__main__":
    main()
