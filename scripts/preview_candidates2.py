"""Preview candidate plate windows (RGB target vs pix2pix prediction) to choose by eye."""
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

CANDS = [("C3", 1792, 1152), ("C3", 2176, 1152), ("C3", 3072, 1152),
         ("C8", 2048, 2176), ("C8", 640, 2304), ("C8", 4352, 1664),
         ("C7", 1024, 1024), ("C6", 4352, 512), ("C6", 4480, 1024)]
SIZE = 256
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"]})
fig, axes = plt.subplots(2, len(CANDS), figsize=(2.2 * len(CANDS), 5.2))
for col, (scene, x, y) in enumerate(CANDS):
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    p2p = np.load(PREV["scenes"][scene]["predictions"]["pix2pix / B1-5"]["prediction_path"])
    rest = np.load(PREV["scenes"][scene]["predictions"]["Restormer / B1-5"]["prediction_path"])
    t = rgb[y:y+SIZE, x:x+SIZE]
    axes[0, col].imshow(t, interpolation="none")
    axes[0, col].set_title(f"{scene} ({x},{y})\nmean={t.mean():.0f}", fontsize=8)
    axes[1, col].imshow(np.rint(p2p[y:y+SIZE, x:x+SIZE] * 255).astype(np.uint8), interpolation="none")
    for r in range(2):
        axes[r, col].set_axis_off()
fig.tight_layout()
out = ROOT / "tmp/style_pass/candidates2.png"
fig.savefig(out, dpi=120, facecolor="white")
print("wrote", out)
