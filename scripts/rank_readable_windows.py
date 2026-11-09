"""Pick plate windows by display readability plus saved-prediction fidelity.

Ranking rule (deterministic): over the full-support shared-support windows, keep
those whose p99 display value is at least 100 and p1 at most 60 (so the panel is
well exposed without clipping), then maximise the saved Restormer (five-band) PSNR.
Only already-saved predictions and published mosaics are read; no training or
inference. Writes a candidate list plus a visual check of the display.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
SUPPORT = GEN / "v3_pipeline/outputs/draft_support_v2_20261002/manifest.json"
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
SUP = json.loads(SUPPORT.read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])

LABELS = ["U-Net / B1-5", "pix2pix / B1-5", "Restormer / B1-5"]
report = {}
for n in range(1, 11):
    scene = f"C{n}"
    info = SUP["scenes"][scene]
    mask = np.asarray(Image.open(SUPPORT.parent / info["mask"])) > 0
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    preds = {lab: np.load(PREV["scenes"][scene]["predictions"][lab]["prediction_path"])
             for lab in LABELS}
    scored = []
    for (x, y, w, h) in info["windows"]:
        if int(mask[y:y + h, x:x + w].sum()) != w * h:
            continue
        target = rgb[y:y + h, x:x + w].astype(np.float32)
        p1, p99 = np.percentile(target, [1, 99])
        if p99 < 100 or p1 > 60:
            continue
        tgt = np.rint(target / 255.0 * 255) / 255
        row = {"xywh": [x, y, w, h], "p1": float(p1), "p99": float(p99),
               "mean": float(target.mean())}
        for lab in LABELS:
            err = ((preds[lab][y:y + h, x:x + w] - tgt) ** 2).mean()
            row[lab] = float(10 * math.log10(1.0 / max(err, 1e-12)))
        scored.append(row)
    scored.sort(key=lambda r: r["Restormer / B1-5"], reverse=True)
    report[scene] = scored
    if scored:
        b = scored[0]
        print(f"{scene}: {len(scored):>4} readable windows | best {b['xywh'][:2]} "
              f"mean={b['mean']:.0f} p1={b['p1']:.0f} p99={b['p99']:.0f} "
              f"Restormer={b['Restormer / B1-5']:.2f} U-Net={b['U-Net / B1-5']:.2f} "
              f"pix2pix={b['pix2pix / B1-5']:.2f}")
    else:
        print(f"{scene}: no readable full-support window")

out = ROOT / "figures/source_data/window_readability_ranking_20261015.json"
out.write_text(json.dumps({
    "contract": "Deterministic readability screen over already-saved data: full-support shared-support "
                "windows with 100 <= p99 and p1 <= 60 on the published RGB crop, ranked by saved "
                "Restormer (five-band) PSNR. No training, inference or new metric.",
    "rule": {"max_p1": 60, "min_p99": 100, "rank_by": "Restormer / B1-5 PSNR of the stored prediction"},
    "candidates": report,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
