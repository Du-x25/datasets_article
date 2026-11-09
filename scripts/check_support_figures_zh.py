"""Verify the support/resolution figures quoted in the Chinese prose."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "figures/source_data"
REL = ROOT / "automation_completion_20261015/release/candidate_package"
GEN = ROOT / "generated"


def macro(name: str) -> str:
    import re
    for path in GEN.glob("*.tex"):
        for m in re.finditer(r"\\newcommand\{\\" + name + r"\}\{([^{}]+)\}", path.read_text(encoding="utf-8")):
            return m.group(1)
    return "?"


print("DatasetOutputScale =", macro("DatasetOutputScale"))
print("DatasetSceneCount  =", macro("DatasetSceneCount"))
print("QualifiedWindowCount =", macro("QualifiedWindowCount"))

stats = json.loads((SD / "support_statistics_20261015.json").read_text(encoding="utf-8"))
def find(obj, keys):
    """search nested json for first value whose key path ends with keys[-1]"""
    out = {}
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])
        else:
            out.setdefault(path[-1], node)
    walk(obj, [])
    return out

flat = find(stats, [])
for k in ["canvas_pixels", "common_support_pixels", "common_support_fraction",
          "ssim_center_pixels", "lpips_rectangles", "lpips_union_pixels",
          "lpips_union_fraction"]:
    hits = {kk: vv for kk, vv in flat.items() if kk == k}
    print(f"  {k}: {hits or 'not found'}")

print("\n--- prose claims to confirm ---")
print("canvas total 133,736,176 ; support 88,861,466 (66.45%) ; SSIM 58,532,167 ; LPIPS 1541 rects union 23,805,940 (26.79%)")
print("C10 exclusion 302,985 (2.630%) ; DN rule 44,571,725 (33.40%)")

excl = (SD.parent.parent / "generated/support_exclusions_zh.tex")
print("\n--- generated/support_exclusions_zh.tex ---")
print(excl.read_text(encoding="utf-8")[:1200])
