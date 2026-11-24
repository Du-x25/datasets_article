"""Per-scene DN histograms of the five SWIR bands and the RGB channel mean of the released panoramas (valid region, every 7th valid pixel).

Usage: quality_hist.py
Output: outputs/data_quality/band_hist.json  {scene: {band: [256 densities in %]}}
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("D:/dh/MSWIR_RGB/bench_20261124/outputs/data_quality")
RELEASE = Path("D:/dh/MSWIR_RGB/data/release")
GEOMETRY = Path("D:/dh/MSWIR_RGB/data/geometry")
SCENES = ["C%d" % i for i in range(1, 11)]


def main():
    out = {}
    for scene in SCENES:
        n = scene[1:]
        mask = np.array(Image.open(GEOMETRY / (scene + ".png"))) > 0
        bands = np.stack([np.array(Image.open(RELEASE / scene / ("scene%s_band%d.tif" % (n, i)))) for i in range(1, 6)], axis=-1)
        rgb = np.array(Image.open(RELEASE / scene / ("scene%s.tif" % n)).convert("RGB"))
        ys, xs = np.nonzero(mask)
        sel = np.arange(len(ys)) % 7 == 0
        b = bands[ys[sel], xs[sel]]
        r = rgb[ys[sel], xs[sel]]
        out[scene] = {}
        for i in range(5):
            out[scene]["B%d" % (i + 1)] = (np.bincount(b[:, i].ravel(), minlength=256) / b.shape[0] * 100).round(4).tolist()
        luma = np.round(r.mean(axis=1)).astype(int)  # channel mean, same reference as band_luma in data_quality_stats
        out[scene]["LUMA"] = (np.bincount(np.clip(luma.ravel(), 0, 255), minlength=256) / luma.size * 100).round(4).tolist()
        print(scene, "pixels", int(sel.sum()), flush=True)
    (OUT / "band_hist.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
