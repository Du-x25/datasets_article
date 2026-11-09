"""List figure/table captions with their page numbers in the built Chinese PDF.

Used to map the user's page-layout feedback ("图2 太小", "图4 占一整页")
onto concrete LaTeX floats.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
pdf = ROOT / "build/zh/main_zh.pdf"
reader = PdfReader(str(pdf))

pat = re.compile(r"(图|表)\s*(\d+)\s*[:：]?\s*([^\n]{0,40})")
for i, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    flat = re.sub(r"\s+", " ", text)
    seen = set()
    for m in pat.finditer(flat):
        key = (m.group(1), m.group(2))
        if key in seen:
            continue
        seen.add(key)
        print(f"p{i:>3}  {m.group(1)}{m.group(2):<3} {m.group(3)[:40]}")
