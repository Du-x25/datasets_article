"""Audit every page of the built PDF: how much of it is float (figure/table)."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
reader = PdfReader(str(ROOT / "build/zh/main_zh.pdf"))
for i, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    flat = re.sub(r"\s+", " ", text)
    imgs = 0
    try:
        imgs = len(page.images)
    except Exception:
        pass
    caps = re.findall(r"(图|表)\s*(\d+)\s*[:：]", flat)
    print(f"p{i:>3} imgs={imgs:<3} chars={len(flat):<5} caps={[a+b for a,b in caps]}  {flat[:70]}")
