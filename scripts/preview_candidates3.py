"""Preview the finalist plate windows."""
from pathlib import Path
import json
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

CANDS = [("C2", 4992, 0), ("C3", 128, 1024), ("C4", 3072, 1280), ("C6", 4352, 1024),
         ("C9", 2304, 2560), ("C8", 0, 2176)]
SIZE = 256
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"]})
fig, axes = plt.subplots(4, len(CANDS), figsize=(2.4 * len(CANDS), 10))
for col, (scene, x, y) in enumerate(CANDS):
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    t = rgb[y:y+SIZE, x:x+SIZE]
    axes[0, col].imshow(t, interpolation="none")
    axes[0, col].set_title(f"{scene} ({x},{y})\nmean={t.mean():.0f}", fontsize=9)
    for row, model in enumerate(["Restormer / B1-5", "pix2pix / B1-5", "U-Net / B1-5"], start=1):
        p = np.load(PREV["scenes"][scene]["predictions"][model]["prediction_path"])
        axes[row, col].imshow(np.rint(p[y:y+SIZE, x:x+SIZE] * 255).astype(np.uint8), interpolation="none")
    for r in range(4):
        axes[r, col].set_axis_off()
fig.tight_layout()
out = ROOT / "tmp/style_pass/candidates3.png"
fig.savefig(out, dpi=115, facecolor="white")
print("wrote", out)
