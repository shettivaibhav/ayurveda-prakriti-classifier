"""Generate thesis figures from the saved result JSONs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "model_v2" / "results_full"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def load(name):
    return json.loads((OUT / name).read_text())


def fig_approach_comparison(imgfus):
    """Balanced accuracy of each approach vs baselines, per framing."""
    for framing in ("7class", "3class_dominant"):
        d = imgfus[framing]
        sets = ["tabular", "image_single", "image_multi", "fusion"]
        bal = [d[s]["models"][d[s]["best_model"]]["bal_acc"][0] for s in sets]
        acc = [d[s]["models"][d[s]["best_model"]]["acc"][0] for s in sets]
        rand = d["tabular"]["baselines"]["random"]
        maj = d["tabular"]["baselines"]["majority"]
        x = np.arange(len(sets))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(x - 0.2, acc, 0.4, label="accuracy", color="#4C72B0")
        ax.bar(x + 0.2, bal, 0.4, label="balanced accuracy", color="#DD8452")
        ax.axhline(maj, ls="--", c="grey", label=f"majority baseline ({maj:.2f})")
        ax.axhline(rand, ls=":", c="black", label=f"random ({rand:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(sets, rotation=15)
        ax.set_ylabel("score"); ax.set_ylim(0, 1)
        ax.set_title(f"Approach comparison — {framing}")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(FIG / f"compare_{framing}.png", dpi=140)
        plt.close(fig)


def fig_confusion(tab):
    for framing, key in (("7-class", "7class"), ("3-class dominant", "3class_dominant")):
        r = tab[key]
        cm = np.array(r["best_confusion"]); labels = r["best_confusion_labels"]
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion ({framing}, {r['best_model']})\nrow-normalised recall")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(FIG / f"confusion_{key}.png", dpi=140)
        plt.close(fig)


def fig_importance(expl):
    for framing in ("7class", "3class_dominant"):
        grp = expl[framing]["by_group"]
        names = list(grp.keys()); perm = [grp[g]["perm"] for g in names]
        order = np.argsort(perm)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([names[i] for i in order], [perm[i] for i in order], color="#55A868")
        ax.set_xlabel("permutation importance (Δ balanced-acc)")
        ax.set_title(f"Feature-group importance — {framing}")
        fig.tight_layout(); fig.savefig(FIG / f"importance_{framing}.png", dpi=140)
        plt.close(fig)


def main():
    tab = load("tabular_results.json")
    imgfus = load("image_fusion_results.json")
    fig_approach_comparison(imgfus)
    fig_confusion(tab)
    try:
        expl = load("explain_results.json")
        fig_importance(expl)
    except FileNotFoundError:
        print("explain_results.json not ready — skipping importance figures")
    print("figures written to", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
