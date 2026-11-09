"""Diagnose the RGB/band source arrays used by the plate generator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
MANIFEST = GEN / "v3_pipeline/configs/v3_scene_manifest.json"
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
root = Path(manifest["root"])
print("manifest root:", root, "exists:", root.exists())

for scene, x, y in [("C8", 256, 2176), ("C2", 4992, 0), ("C3", 3456, 1152)]:
    for mode in ["RGB", "B1"]:
        entry = manifest["scenes"][scene][mode]
        path = root / entry["path"]
        img = Image.open(path)
        arr = np.asarray(img)
        crop = arr[y:y + 256, x:x + 256]
        print(f"{scene} {mode:<3} mode={img.mode} shape={arr.shape} dtype={arr.dtype} "
              f"declared={entry['shape']} path_exists={path.exists()}")
        print(f"        full  min/mean/max = {arr.min()}/{arr.mean():.1f}/{arr.max()}")
        print(f"        crop  min/mean/max = {crop.min()}/{crop.mean():.1f}/{crop.max()}")
