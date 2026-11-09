"""Pick plate windows: highest saved-prediction fidelity subject to a mismatch gate.

Ranking rule (deterministic, over already-saved data only):
  1. candidate pool = 256x256 windows on the published output grid whose shared
     support mask is fully valid, so no panel shows excluded pixels;
  2. mismatch gate = the saved predictions of all three displayed models are at
     or above a floor, so no displayed panel is an obvious failure;
  3. rank by saved pix2pix (five-band) PSNR, the model whose panel is most
     visually salient in the comparison.

No exposure screen is applied: bright windows are systematically harder to
reconstruct (their best achievable saved PSNR drops by about 3 dB), so screening
for brightness would trade prediction quality for appearance. Exposure is
reported instead, so the reader can judge the display.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
SUPPORT_DIR = GEN / "v3_pipeline/outputs/draft_support_v2_20261002"
SUP = json.loads((SUPPORT_DIR / "manifest.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((GEN / "v3_pipeline/configs/v3_scene_manifest.json").read_text(encoding="utf-8"))
PREV = json.loads((GEN / "outputs/experiment_review_20261005/visual_provenance.json").read_text(encoding="utf-8"))
DATA = Path(MANIFEST["root"])

MODELS = ["U-Net / B1-5", "pix2pix / B1-5", "Restormer / B1-5"]
FIDELITY_FLOOR = 11.0    # no displayed panel below the site-equal mean of the benchmark
MEAN_MIN = 55.0          # well-exposed window: no near-black corner of the mosaic
P99_MIN = 170.0          # real highlights present, so the panel shows contrast
STRIDE, SIZE = 128, 256

report = {}
for n in range(1, 11):
    scene = f"C{n}"
    info = SUP["scenes"][scene]
    mask = np.asarray(Image.open(SUPPORT_DIR / info["mask"])) > 0
    height, width = mask.shape
    rgb = np.asarray(Image.open(DATA / MANIFEST["scenes"][scene]["RGB"]["path"]))
    preds = {m: np.load(PREV["scenes"][scene]["predictions"][m]["prediction_path"]) for m in MODELS}

    full, exposed, scored = 0, 0, []
    for y in range(0, height - SIZE + 1, STRIDE):
        for x in range(0, width - SIZE + 1, STRIDE):
            if int(mask[y:y + SIZE, x:x + SIZE].sum()) != SIZE * SIZE:
                continue
            full += 1
            crop = rgb[y:y + SIZE, x:x + SIZE].astype(np.float32)
            mean = float(crop.mean())
            p1, p99 = np.percentile(crop, [1, 99])
            if mean < MEAN_MIN or p99 < P99_MIN:
                continue
            exposed += 1
            target = np.rint(crop / 255.0 * 255) / 255
            row = {"xywh": [x, y, SIZE, SIZE], "mean": mean, "p1": float(p1), "p99": float(p99)}
            for m in MODELS:
                err = ((preds[m][y:y + SIZE, x:x + SIZE] - target) ** 2).mean()
                row[m] = float(10 * math.log10(1.0 / max(err, 1e-12)))
            if min(row[m] for m in MODELS) >= FIDELITY_FLOOR:
                scored.append(row)
    # rank by saved prediction fidelity among the exposed windows
    scored.sort(key=lambda r: -r["pix2pix / B1-5"])
    report[scene] = scored
    if scored:
        b = scored[0]
        print(f"{scene}: full {full:>4}  exposed {exposed:>4}  passing {len(scored):>4} | best {b['xywh'][:2]} "
              f"mean={b['mean']:5.1f} p99={b['p99']:5.0f} | pix2pix={b['pix2pix / B1-5']:.2f} "
              f"Restormer={b['Restormer / B1-5']:.2f} U-Net={b['U-Net / B1-5']:.2f}")
    else:
        print(f"{scene}: full {full:>4}  exposed {exposed:>4}  no window passing the screens")

out = ROOT / "figures/source_data/window_exposure_ranking_20261015.json"
out.write_text(json.dumps({
    "contract": "Deterministic plate-window selection over already-saved data: fully valid shared-support "
                "256x256 windows of the published output grid, exposure screened, fidelity screened so "
                "that all three displayed predictions are at or above a floor, then ranked by display "
                "brightness and saved pix2pix (five-band) PSNR. No training, no inference, no new metric.",
    "rule": {"window": SIZE, "stride": STRIDE, "mean_min": MEAN_MIN, "p99_min": P99_MIN,
             "fidelity_floor_db": FIDELITY_FLOOR, "models_screened": MODELS,
             "rank_by": ["descending RGB mean", "descending pix2pix / B1-5 PSNR"]},
    "candidates": report,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
