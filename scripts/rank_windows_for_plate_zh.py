"""Rank candidate windows by saved-prediction fidelity over all shared-support windows.

No training or inference: this loads the previously saved float32 predictions and
the published RGB mosaics, evaluates every candidate window of the existing
support manifest, and reports the best windows per site. The ranking is the
selection rule for the illustrative plates, so it is deterministic and auditable
rather than an eyeballed choice.
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
MANIFEST = GEN / "v3_pipeline/configs/v3_scene_manifest.json"
PREV = GEN / "outputs/experiment_review_20261005/visual_provenance.json"

CONDITIONS = ["U-Net / B1-5", "pix2pix / B1-5", "TFD / B1-5",
              "Restormer / B1-5", "pix2pix / B1", "pix2pix / B1,2,4"]


def main() -> None:
    support = json.loads(SUPPORT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prev = json.loads(PREV.read_text(encoding="utf-8"))
    data_root = Path(manifest["root"])

    ranking = {}
    for site_num in range(1, 11):
        scene = f"C{site_num}"
        info = support["scenes"][scene]
        mask = np.asarray(Image.open(SUPPORT.parent / info["mask"])) > 0
        rgb_entry = manifest["scenes"][scene]["RGB"]
        rgb = np.asarray(Image.open(data_root / rgb_entry["path"]))
        assert list(rgb.shape) == rgb_entry["shape"], scene

        preds = {}
        for label in CONDITIONS:
            prior = prev["scenes"][scene]["predictions"][label]
            arr = np.load(prior["prediction_path"], mmap_mode="r")
            preds[label] = np.asarray(arr)
        assert all(p.shape[:2] == rgb.shape[:2] for p in preds.values()), scene

        scored = []
        for (x, y, w, h) in info["windows"]:
            m = mask[y:y + h, x:x + w]
            n = int(m.sum())
            if n != w * h:
                continue  # full-support windows only, so every panel is fully shown
            target = rgb[y:y + h, x:x + w].astype(np.float32) / 255.0
            target = np.rint(target * 255) / 255  # match the stored prediction quantisation
            row = {"xywh": [x, y, w, h], "valid_pixels": n}
            for label in CONDITIONS:
                pred = preds[label][y:y + h, x:x + w]
                err = ((pred - target) ** 2).mean()
                row[label] = float(10 * math.log10(1.0 / max(err, 1e-12)))
            scored.append(row)
        scored.sort(key=lambda r: r["Restormer / B1-5"], reverse=True)
        ranking[scene] = scored
        best = scored[0]
        print(f"{scene}: {len(scored):>4} full-support windows  "
              f"best Restormer PSNR {best['Restormer / B1-5']:6.2f} dB at {best['xywh'][:2]}  "
              f"| median {scored[len(scored)//2]['Restormer / B1-5']:6.2f} dB")

    out = ROOT / "figures/source_data/window_ranking_20261015.json"
    out.write_text(json.dumps({
        "contract": "Ranking of existing shared-support candidate windows by saved seed-11 prediction "
                    "fidelity. No training, no inference, no new metrics: MSE of the stored float32 "
                    "prediction against the published uint8 RGB mosaic, converted to PSNR.",
        "source_support_manifest": str(SUPPORT),
        "source_visual_provenance": str(PREV),
        "conditions": CONDITIONS,
        "ranking": ranking,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
