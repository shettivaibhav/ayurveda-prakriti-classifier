"""Palm/skin colour feature extraction — shared by training and the app so the
features are computed identically in both places.

Best fully-automated model (validated ~67% acc / 46% balanced vs the practitioner's
sheet, 3-class dominant dosha): two fixed anatomical colour patches + age/sex, into a
RandomForest. No CNN needed at inference.
  skin colour <- centre patch of the palm-open BACK view
  palm colour <- patch just-above-centre of the palm-open FRONT view
"""

import numpy as np
from skimage.color import rgb2lab

SKIN_VIEW = "palm_open_back"     # skin colour read from back of hand, centre
PALM_VIEW = "palm_open_front"    # palm colour read from palm, just above centre
SKIN_CY = 0.50
PALM_CY = 0.40
HALF = 0.16                      # patch half-size as a fraction of width/height

# feature order MUST match training (exp_best.py): [skin(10), palm(10), Age, Sex_M]
FEATURE_NAMES = (
    [f"SC_{c}" for c in ["R", "G", "B", "L", "a", "b", "cr", "cg", "cb", "red"]]
    + [f"PC_{c}" for c in ["R", "G", "B", "L", "a", "b", "cr", "cg", "cb", "red"]]
    + ["Age", "Sex_M"]
)


def patch_color(pil_img, cy_frac, half=HALF):
    """Mean colour descriptor (RGB, Lab, normalised-rgb chroma, redness) of a square
    patch centred at (width/2, height*cy_frac). 10 features."""
    im = np.asarray(pil_img.convert("RGB")).astype(float)
    H, W = im.shape[:2]
    cx, cy = W // 2, int(H * cy_frac)
    hw, hh = int(W * half), int(H * half)
    patch = im[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw].reshape(-1, 3) / 255.0
    rgb = patch.mean(0)
    lab = rgb2lab(patch.reshape(-1, 1, 3)).reshape(-1, 3).mean(0)
    s = patch.sum(1) + 1e-6
    chroma = np.stack([patch[:, 0] / s, patch[:, 1] / s, patch[:, 2] / s], 1).mean(0)
    return np.concatenate([rgb, lab, chroma, [(patch[:, 0] - patch[:, 1]).mean()]])


def extract_features(skin_img, palm_img, age, sex_m):
    """skin_img = palm_open_back PIL, palm_img = palm_open_front PIL. Returns (22,) vector."""
    sc = patch_color(skin_img, SKIN_CY)
    pc = patch_color(palm_img, PALM_CY)
    return np.concatenate([sc, pc, [float(age), float(sex_m)]])
