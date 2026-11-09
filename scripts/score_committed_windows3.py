"""Final window screens on the committed visual-review windows."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
SEL = json.loads((GEN / "outputs/experiment_review_20261005/visual_selection.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])
MODELS = ["U-Net / B1-5", "pix2pix / B1-5", "Restormer / B1-5"]

cands = []
for n in range(1, 11):
    s = f"C{n}"
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][s]["RGB"]["path"]))
    for crop in ["A", "B"]:
        x, y, w, h = SEL["scenes"][s][crop]["xywh"]
        c = rgb[y:y+h, x:x+w].astype(np.float32)
        tgt = np.rint(c / 255.0 * 255) / 255
        ps = {}
        for m in MODELS:
            p = np.load(PREV["scenes"][s]["predictions"][m]["prediction_path"])[y:y+h, x:x+w]
            ps[m] = 10*math.log10(1.0/max(((p-tgt)**2).mean(), 1e-12))
        cands.append(dict(scene=s, crop=crop, xywh=[x, y, w, h], mean=float(c.mean()),
                          p99=float(np.percentile(c, 99)), ps=ps))

mean_min, p99_min, floor = 60.0, 130.0, 14.0
passing = [c for c in cands if c["mean"] >= mean_min and c["p99"] >= p99_min
           and min(c["ps"].values()) >= floor]
passing.sort(key=lambda c: -c["ps"]["pix2pix / B1-5"])
print(f"screens: mean>={mean_min}, p99>={p99_min}, all models>={floor} dB -> {len(passing)} windows")
for c in passing:
    print(f"  {c['scene']} {c['crop']} {c['xywh'][:2]} mean={c['mean']:5.1f} p99={c['p99']:5.0f} "
          f"pix2pix={c['ps']['pix2pix / B1-5']:5.2f} Restormer={c['ps']['Restormer / B1-5']:5.2f} "
          f"U-Net={c['ps']['U-Net / B1-5']:5.2f}")

