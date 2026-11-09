"""How high does saved-prediction PSNR go at various exposure levels?"""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
SUPPORT_DIR = GEN / 'v3_pipeline/outputs/draft_support_v2_20261002'
SUP = json.loads((SUPPORT_DIR / 'manifest.json').read_text(encoding='utf-8'))
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])
SIZE, STRIDE = 256, 128

for scene in ["C3", "C6", "C7", "C8"]:
    mask = np.asarray(Image.open(SUPPORT_DIR / SUP["scenes"][scene]["mask"])) > 0
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    preds = {m: np.load(PREV["scenes"][scene]["predictions"][m]["prediction_path"])
             for m in ["pix2pix / B1-5", "Restormer / B1-5", "U-Net / B1-5"]}
    rows = []
    h, w = mask.shape
    for y in range(0, h - SIZE + 1, STRIDE):
        for x in range(0, w - SIZE + 1, STRIDE):
            if int(mask[y:y+SIZE, x:x+SIZE].sum()) != SIZE*SIZE:
                continue
            crop = rgb[y:y+SIZE, x:x+SIZE].astype(np.float32)
            tgt = np.rint(crop / 255.0 * 255) / 255
            ps = {m: 10*math.log10(1.0/max(((preds[m][y:y+SIZE, x:x+SIZE]-tgt)**2).mean(), 1e-12)) for m in preds}
            rows.append((ps["pix2pix / B1-5"], float(crop.mean()), np.percentile(crop,99), np.percentile(crop,1), x, y, min(ps.values())))
    rows.sort(reverse=True)
    print(f"--- {scene}: top pix2pix PSNR by exposure class")
    for thr, label in [((45,150,140,30), "exposure-screened"), ((35,170,120,45), "loosened"), ((0,256,0,256), "any")]:
        lo, hi, p99, p1 = thr
        ok = [r for r in rows if lo <= r[1] <= hi and r[2] >= p99 and r[3] <= p1]
        if ok:
            b = max(ok)
            print(f"   {label:<18} n={len(ok):>4} best pix2pix={b[0]:6.2f} worst-model={b[6]:6.2f} mean={b[1]:5.1f} at ({b[4]},{b[5]})")

