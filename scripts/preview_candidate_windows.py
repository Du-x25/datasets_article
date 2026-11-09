"""Render the palette of candidate plate windows for visual comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8")) \
    if False else None
import json
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])

CANDIDATES = [
    ("C3", 3072, 1152, "PSNR 18.26"),
    ("C3", 3456, 1152, "PSNR 25.91"),
    ("C6", 4480, 1024, "PSNR 19.20"),
    ("C6", 4352, 1024, "PSNR 23.50"),
    ("C8", 1536, 2304, "PSNR 15.46"),
    ("C8", 4352, 1664, "PSNR 23.13"),
    ("C1", 3456, 3072, "PSNR 15.43"),
    ("C7", 8832, 1024, "PSNR 19.05"),
]
SIZE = 256

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"]})
fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for col, (scene, x, y, note) in enumerate(CANDIDATES):
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    pred_path = PREV["scenes"][scene]["predictions"]["Restormer / B1-5"]["prediction_path"]
    pred = np.load(pred_path)
    target = rgb[y:y + SIZE, x:x + SIZE]
    p = pred[y:y + SIZE, x:x + SIZE]
    axes[0, col].imshow(target, interpolation="none")
    axes[0, col].set_title(f"{scene} ({x},{y})\n{note}\nmean={target.mean():.0f}", fontsize=7)
    axes[1, col].imshow(np.rint(p * 255).astype(np.uint8), interpolation="none")
    for row in range(2):
        axes[row, col].set_axis_off()
fig.tight_layout()
out = ROOT / "tmp" / "style_pass" / "candidate_windows.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, facecolor="white")
print("wrote", out)
