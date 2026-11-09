"""Report the drawn width of every table rule, to check tables fill the text block."""
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
doc = pymupdf.open(ROOT / "build/zh/main_zh.pdf")
for i, page in enumerate(doc, 1):
    rules = [d["rect"] for d in page.get_drawings()
             if d["rect"].width > 60 and d["rect"].height < 3]
    if not rules:
        continue
    left = min(r.x0 for r in rules)
    right = max(r.x1 for r in rules)
    print(f"p{i:>3}  text block 58.1-537.2 pt   table rules {left:6.1f}-{right:6.1f}  width {right-left:6.1f}")
