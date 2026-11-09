"""Score the 20 committed visual-review windows by display luminance and prediction fidelity.

Reads the already-saved predictions and the published mosaics; no training or
inference. Explains why the centre-of-canvas windows read as near-black and
identifies windows that are both well exposed and well predicted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
SEL = json.loads((GEN / "outputs/experiment_review_20261005/visual_selection.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])

rows = []
for n in range(1, 11):
    scene = f"C{n}"
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    for crop in ["A", "B"]:
        x, y, w, h = SEL["scenes"][scene][crop]["xywh"]
        c = rgb[y:y + h, x:x + w].astype(np.float32)
        lum = c.mean()
        lo, hi = np.percentile(c, [1, 99])
        entry = {"scene": scene, "crop": crop, "xywh": [x, y, w, h],
                 "rgb_mean": round(float(lum), 1),
                 "rgb_p1": round(float(lo), 1), "rgb_p99": round(float(hi), 1)}
        for label in ["Restormer / B1-5", "U-Net / B1-5", "pix2pix / B1-5"]:
            p = np.load(PREV["scenes"][scene]["predictions"][label]["prediction_path"])
            tgt = np.rint(c / 255.0 * 255) / 255
            err = ((p[y:y + h, x:x + w] - tgt) ** 2).mean()
            entry[label] = round(float(10 * np.log10(1.0 / max(err, 1e-12))), 2)
        rows.append(entry)

rows.sort(key=lambda r: r["rgb_mean"], reverse=True)
print(f"{'scene':<5}{'win':<4}{'x,y':<14}{'RGBmean':>8}{'p1':>6}{'p99':>6}"
      f"{'Restormer':>11}{'U-Net':>8}{'pix2pix':>9}")
for r in rows:
    print(f"{r['scene']:<5}{r['crop']:<4}{str(r['xywh'][:2]):<14}{r['rgb_mean']:>8}"
          f"{r['rgb_p1']:>6}{r['rgb_p99']:>6}{r['Restormer / B1-5']:>11}"
          f"{r['U-Net / B1-5']:>8}{r['pix2pix / B1-5']:>9}")

out = ROOT / "figures/source_data/window_display_scores_20261015.json"
out.write_text(json.dumps({"contract": "Display-luminance and saved-prediction PSNR of the 20 committed "
                                       "visual-review windows; read-only diagnostic, no new measurement.",
                           "windows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote", out)
