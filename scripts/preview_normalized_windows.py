"""Preview the plate windows under a documented per-panel luminance normalisation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])

CANDIDATES = [
    ("C3", 2176, 1152, "19.96"),
    ("C3", 3072, 1152, "18.26"),
    ("C6", 4480, 1024, "19.20"),
    ("C8", 640, 2304, "17.22"),
    ("C8", 4352, 1664, "23.13"),
    ("C1", 3456, 3072, "15.43"),
    ("C7", 8832, 1024, "19.05"),
    ("C10", 768, 640, "12.32"),
]
SIZE = 256


def normalize(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Common linear map from the panel luminance percentiles; no per-channel stretch."""
    a = arr.astype(np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    p_lo, p_hi = np.percentile(lum, [lo, hi])
    if p_hi - p_lo < 1e-3:
        return a.astype(np.uint8)
    out = (a - p_lo) / (p_hi - p_lo)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"]})
fig, axes = plt.subplots(2, len(CANDIDATES), figsize=(2.0 * len(CANDIDATES), 4.8))
for col, (scene, x, y, note) in enumerate(CANDIDATES):
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    pred = np.load(PREV["scenes"][scene]["predictions"]["Restormer / B1-5"]["prediction_path"])
    target = rgb[y:y + SIZE, x:x + SIZE]
    p = np.rint(pred[y:y + SIZE, x:x + SIZE] * 255).astype(np.uint8)
    axes[0, col].imshow(normalize(target), interpolation="none")
    axes[0, col].set_title(f"{scene} ({x},{y})\nPSNR {note}", fontsize=7)
    axes[1, col].imshow(normalize(p), interpolation="none")
    for row in range(2):
        axes[row, col].set_axis_off()
fig.tight_layout()
out = ROOT / "tmp" / "style_pass" / "candidate_windows_normalized.png"
fig.savefig(out, dpi=130, facecolor="white")
print("wrote", out)
