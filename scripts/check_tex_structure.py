"""Structural sanity check for main_zh.tex after the 2026-10-15 style pass."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "main_zh.tex").read_text(encoding="utf-8")

ok = True
for env in ["table", "figure", "tabularx", "tabular", "tabular*", "abstract", "document"]:
    pat = r"\\begin\{" + re.escape(env) + r"\*?\}"
    end = r"\\end\{" + re.escape(env) + r"\*?\}"
    b = len(re.findall(pat, text))
    e = len(re.findall(end, text))
    flag = "OK " if b == e else "BAD"
    if b != e:
        ok = False
    print(f"{flag} {env:<10} begin={b} end={e}")

checks = {
    "dollar signs": text.count("$"),
    "cites": len(re.findall(r"\\cite\{", text)),
    "refs": len(re.findall(r"\\ref\{", text)),
    "braces {": text.count("{"),
    "braces }": text.count("}"),
}
for k, v in checks.items():
    print(f"    {k}: {v}")
print("brace balance:", "OK" if checks["braces {"] == checks["braces }"] else "BAD")
if checks["braces {"] != checks["braces }"]:
    ok = False

# duplicated adjacent CJK lines are a symptom of a botched edit
lines = text.splitlines()
dupes = [(i + 1, ln[:40]) for i, ln in enumerate(lines[1:], start=2) if ln.strip() and ln == lines[i - 2]]
print("adjacent duplicate lines:", dupes or "none")
if dupes:
    ok = False

sys.exit(0 if ok else 1)
