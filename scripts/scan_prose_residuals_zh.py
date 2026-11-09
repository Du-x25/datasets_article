"""Adversarial prose scan for the Chinese manuscript style pass.

Looks for residual AI-flavoured or popular-science markers, duplicated logical
connectives and sentence-initial filler that a regex linter does not catch.
Reads tmp/style_pass/main_zh_prose.txt (produced by scripts/style_baseline.py).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prose = (ROOT / "tmp/style_pass/main_zh_prose.txt").read_text(encoding="utf-8")
sentences = [s.strip() for s in re.split(r"(?<=[。！？；])", prose.replace("\n", "")) if len(s.strip()) > 3]

BANNED = [
    "值得注意的是", "需要指出", "需要说明", "众所周知", "简单来说", "也就是说",
    "换句话说", "不难看出", "不难发现", "可以看出", "不言而喻", "显而易见",
    "总的来说", "综上所述", "总而言之", "如上所述", "如前所述",
    "首先", "其次", "最后", "第一步", "第二步",
    "极大地", "大幅", "显著提升", "效果非常好", "十分", "非常",
    "本文将", "接下来", "下面我们", "让我们",
    "填补", "空白", "前所未有", "首创", "最优",
]
# duplicated causal/contrast connectives inside one sentence
DOUBLES = ["因此因此", "由于由于", "但是但", "然而但", "因此所以", "由于因为",
           "并且并且", "而且而且", "从而从而", "为此因此"]

DUPLICATE_SENTENCE_OPENERS = ["这说明", "这表明", "上述结果", "该结果"]

print(f"sentences: {len(sentences)}")
print("\n--- banned / filler markers ---")
for w in BANNED:
    hits = [i for i, s in enumerate(sentences) if w in s]
    if hits:
        print(f"  {w}: {len(hits)}")
        for i in hits[:3]:
            print(f"      {sentences[i][:70]}")

print("\n--- doubled connectives ---")
found = False
for w in DOUBLES:
    if w in prose:
        found = True
        print(f"  {w}")
if not found:
    print("  none")

print("\n--- repeated sentence openers (>=3) ---")
from collections import Counter
openers = Counter(s[:4] for s in sentences)
for opener, n in openers.most_common(12):
    if n >= 3:
        print(f"  {opener!r}: {n}")

print("\n--- sentences over 100 chars (segment candidates) ---")
for s in sentences:
    if len(s) > 100:
        print(f"  [{len(s)}] {s[:60]}...{s[-30:]}")

print("\n--- hedge density ---")
for w in ["可能", "或许", "似乎", "大致", "一定程度", "相对而言", "通常", "往往"]:
    print(f"  {w}: {prose.count(w)}")
