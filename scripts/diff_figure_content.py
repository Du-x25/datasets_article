"""Prove the figure regeneration changed only the monitoring footer text.

For every regenerated PDF, compare the extracted text of the backup against the
current file, and compare the embedded raster pixels for the PNG previews. This
separates a real content change from the PDF CreationDate/ID metadata that
matplotlib writes on every run.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from pypdf import PdfReader
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT / "figures/zh"
BAK = ROOT / "tmp/style_pass/fig_backup"

STEMS = ["figV9_generalization_gap_zh", "figV10_baseline_comparison_zh",
         "figV11_misalignment_sensitivity_zh", "figV12_aggregation_zh",
         "v3_inventory_folds_zh", "fig_training_monitoring_zh"]

print(f"{'figure':<40} {'pdf text':<12} {'png pixels':<12}")
for stem in STEMS:
    a, b = BAK / f"{stem}.pdf", CUR / f"{stem}.pdf"
    ta = (PdfReader(str(a)).pages[0].extract_text() or "").split()
    tb = (PdfReader(str(b)).pages[0].extract_text() or "").split()
    text_state = "identical" if ta == tb else f"DIFFERS (+{len(tb)-len(ta)} tokens)"

    ia = Image.open(BAK / f"{stem}.png").convert("RGB")
    ib = Image.open(CUR / f"{stem}.png").convert("RGB")
    if ia.size != ib.size:
        png_state = f"size {ia.size}->{ib.size}"
    else:
        sa = hashlib.sha256(ia.tobytes()).hexdigest()
        sb = hashlib.sha256(ib.tobytes()).hexdigest()
        png_state = "identical" if sa == sb else "DIFFERS"
    print(f"{stem:<40} {text_state:<12} {png_state:<12}")
    if ta != tb:
        only_a = [t for t in ta if t not in tb]
        only_b = [t for t in tb if t not in ta]
        if only_a:
            print(f"    removed: {' '.join(only_a)[:160]}")
        if only_b:
            print(f"    added  : {' '.join(only_b)[:160]}")
