"""Structural sanity check for main_zh.tex and main_en.tex."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_file(filename: str) -> bool:
    path = ROOT / filename
    if not path.is_file():
        print(f"MISSING {filename}")
        return False
    text = path.read_text(encoding="utf-8")
    ok = True
    print(f"=== {filename} ===")
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
    if checks["braces {"] != checks["braces }"]:
        print("brace balance: BAD")
        ok = False
    else:
        print("brace balance: OK")
    if "**" in text:
        print("markdown bold residual (**): BAD")
        ok = False

    lines = text.splitlines()
    dupes = [(i + 1, ln[:40]) for i, ln in enumerate(lines[1:], start=2) if ln.strip() and ln == lines[i - 2]]
    print("adjacent duplicate lines:", dupes or "none")
    if dupes:
        ok = False
    return ok


def main():
    ok = all(check_file(name) for name in ("main_zh.tex", "supplementary_zh.tex", "main_en.tex", "supplementary_en.tex"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
