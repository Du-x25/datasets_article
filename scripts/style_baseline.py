"""Extract Chinese prose from main_zh.tex and report style metrics.

Standalone helper for the 2026-10-15 style pass. It does not touch the
manuscript or any generated asset; it only reads main_zh.tex.
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DROPPED_ENVIRONMENTS = (
    "tabularx", "tabular", "tabular*", "figure", "table", "verbatim",
    "thebibliography",
)


def strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    for env in DROPPED_ENVIRONMENTS:
        text = re.sub(
            r"\\begin\{" + re.escape(env) + r"\*?\}.*?\\end\{" + re.escape(env) + r"\*?\}",
            "\n",
            text,
            flags=re.S,
        )
    # keep prose of the abstract
    text = text.replace("\\begin{abstract}", "\n").replace("\\end{abstract}", "\n")
    text = re.sub(r"\\input\{[^}]*\}", "\n", text)
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(cite|ref|label)\{[^}]*\}", "", text)
    text = re.sub(r"\\(section|subsection|subsubsection|caption|footnotesize|small|centering|noindent|clearpage|maketitle|appendix|par|vspace|hspace|item)\*?(\[[^\]]*\])?(\{[^{}]*\})?", "\n", text)
    text = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?", "\n", text)
    text = re.sub(r"\\end\{[^}]*\}", "\n", text)
    text = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]+", "", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def sentences(text: str):
    out = []
    for line in text.splitlines():
        for piece in re.split(r"(?<=[。！？；])", line):
            piece = piece.strip()
            if len(piece) < 4:
                continue
            out.append(piece)
    return out


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "main_zh.tex"
    src = target.read_text(encoding="utf-8")
    prose = strip_tex(src)
    out = ROOT / "tmp" / "style_pass"
    out.mkdir(parents=True, exist_ok=True)
    stem = target.stem
    (out / f"{stem}_prose.txt").write_text(prose, encoding="utf-8")

    sents = sentences(prose)
    lengths = sorted(len(s) for s in sents)

    def pct(q):
        if not lengths:
            return 0
        idx = min(len(lengths) - 1, int(round(q * (len(lengths) - 1))))
        return lengths[idx]

    n = len(lengths)
    print(f"prose chars      : {len(re.sub(chr(10), '', prose))}")
    print(f"sentences        : {n}")
    print(f"median length    : {statistics.median(lengths) if lengths else 0}")
    print(f"P25/P75/P90      : {pct(0.25)}/{pct(0.75)}/{pct(0.90)}")
    print(f">=40 chars share : {sum(1 for x in lengths if x >= 40) / max(n,1) * 100:.1f}%")
    print(f"<15 chars share  : {sum(1 for x in lengths if x < 15) / max(n,1) * 100:.1f}%")
    for word in ["我们", "笔者", "然而", "但", "因此", "从而", "为此", "本文", "本研究", "本稿", "本轮", "当前", "尚未", "不等同", "不能", "不代表", "本稿不", "不在本稿范围"]:
        print(f"  {word:<8}: {prose.count(word)}")
    banned = ["完美", "极大", "革命性", "颠覆性", "填补空白", "state-of-the-art", "效果很好", "今后将进一步"]
    hits = {w: prose.count(w) for w in banned if prose.count(w)}
    print(f"banned hits      : {hits or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
