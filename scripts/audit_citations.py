"""Report which refs.bib entries are actually cited, and where the prose is heaviest."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "main_zh.tex").read_text(encoding="utf-8")
bib = (ROOT / "refs.bib").read_text(encoding="utf-8")

cited = Counter()
for m in re.finditer(r"\\cite\{([^}]*)\}", text):
    for key in m.group(1).split(","):
        cited[key.strip()] += 1
keys = re.findall(r"@\w+\{([^,]+),", bib)
print(f"bib entries: {len(keys)}   cited keys: {len(cited)}")
uncited = [k for k in keys if k not in cited]
print("uncited:", uncited)
print("\ncitation counts (ascending):")
for key, n in sorted(cited.items(), key=lambda kv: kv[1]):
    print(f"  {key:<26} {n}")

# prose weight per section
sections = re.split(r"\n(?=\\section|\\subsection)", text)
print("\nheaviest sections (CJK chars):")
rows = []
for chunk in sections:
    head = chunk.split("\n", 1)[0][:60]
    body = re.sub(r"\\(cite|ref|label|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", "", chunk)
    rows.append((len(re.findall(r"[\u4e00-\u9fff]", body)), head))
for n, head in sorted(rows, reverse=True)[:10]:
    print(f"  {n:>5}  {head}")
