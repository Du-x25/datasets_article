"""Compare per-row display ranges of the three models against the RGB/B1 pair."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])
for scene, x, y in [("C8", 0, 2176), ("C9", 2304, 2560), ("C6", 4352, 1024)]:
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))[y:y+256, x:x+256]
    b1 = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["B1"]["path"]))[y:y+256, x:x+256]
    def lum(a):
        a = a.astype(np.float32)
        return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32) if a.ndim == 3 else a
    pair = np.concatenate([lum(rgb).ravel(), lum(b1).ravel()])
    print(f"--- {scene} ({x},{y})  RGB mean {rgb.mean():.1f}  B1 mean {b1.mean():.1f}  pair p2-p98 {np.percentile(pair,[2,98])}")
    for m in ["U-Net / B1-5", "pix2pix / B1-5", "Restormer / B1-5"]:
        p = np.load(PREV["scenes"][scene]["predictions"][m]["prediction_path"])[y:y+256, x:x+256]
        pl = lum(np.rint(p * 255))
        print(f"    {m:<20} pred mean {pl.mean():6.1f}  p2-p98 {np.percentile(pl,2):6.1f}-{np.percentile(pl,98):6.1f}")
