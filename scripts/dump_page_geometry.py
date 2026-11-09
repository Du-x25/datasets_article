"""Dump line geometry of selected PDF pages to diagnose table layout problems."""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
doc = pymupdf.open(ROOT / "build/zh/main_zh.pdf")
pages = [int(a) for a in sys.argv[1:]] or [4, 5, 6, 7, 9, 10]
for pno in pages:
    page = doc[pno - 1]
    print("=" * 24, "page", pno, f"(width {page.rect.width:.0f}, height {page.rect.height:.0f})")
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            print(f"  y={y0:6.1f} x={x0:6.1f}-{x1:6.1f}  {text[:80]}")
