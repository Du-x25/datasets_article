"""Confirm the exact support counts quoted in the Chinese prose."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "figures/source_data"

with (SD / "support_statistics_20261015.csv").open(encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))

print("columns:", list(rows[0].keys()))
total = None
for row in rows:
    if row.get("scene") in {"合计", "total", "TOTAL"} or row.get(list(row.keys())[0]) in {"合计", "total"}:
        total = row
print("total row:", total)

# canvas from the provenance metadata
prov = list(csv.DictReader((SD / "acquisition_provenance.csv").open(encoding="utf-8-sig", newline="")))
canvas = sum(int(r["canvas_width"]) * int(r["canvas_height"]) for r in prov)
print("canvas pixels from provenance:", canvas)

c10 = next(r for r in prov if r["scene"] == "C10")
c10_px = int(c10["canvas_width"]) * int(c10["canvas_height"])
print(f"C10 canvas {c10['canvas_width']}x{c10['canvas_height']} = {c10_px}; 2.630% = {c10_px * 0.02630:.0f}")
print(f"all canvases 0.227% = {canvas * 0.00227:.0f}")
