"""Measure the filled area of each PDF page to find wasted space."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
doc = pymupdf.open(ROOT / "build/zh/main_zh.pdf")
for i, page in enumerate(doc, 1):
    rect = page.rect
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    drawings = page.get_drawings()
    bboxes = [pymupdf.Rect(b[:4]) for b in blocks]
    for d in drawings:
        bboxes.append(d["rect"])
    for img in page.get_images(full=True):
        try:
            bboxes.extend(page.get_image_rects(img[0]))
        except Exception:
            pass
    if not bboxes:
        print(f"p{i:>3} EMPTY")
        continue
    top = min(b.y0 for b in bboxes)
    bottom = max(b.y1 for b in bboxes)
    height = rect.height
    print(f"p{i:>3} y {top:6.1f} -> {bottom:6.1f} of {height:6.1f}  "
          f"tail_gap {height - bottom:6.1f}  head_gap {top:6.1f}")
