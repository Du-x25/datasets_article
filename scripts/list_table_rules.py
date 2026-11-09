"""List individual horizontal rules on the pages whose tables looked narrow."""
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
doc = pymupdf.open(ROOT / "build/zh/main_zh.pdf")
for pno in [6, 20, 13]:
    page = doc[pno - 1]
    print("=" * 20, "page", pno)
    for d in page.get_drawings():
        r = d["rect"]
        if r.width > 20 and r.height < 3:
            print(f"   rule y={r.y0:6.1f} x={r.x0:6.1f}-{r.x1:6.1f} w={r.width:6.1f}")
    txt = " ".join((page.get_text() or "").split())
    print("   text head:", txt[:110])
