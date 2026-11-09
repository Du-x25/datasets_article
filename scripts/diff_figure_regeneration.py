"""Compare regenerated Chinese figure outputs against the pre-regeneration backup.

Used to prove that the monitoring-footer text edit changed only the intended
figures and left every quantitative data export byte-identical.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT / "figures/zh"
BAK = ROOT / "tmp/style_pass/fig_backup"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


rows = []
names = sorted({p.name for p in BAK.iterdir() if p.is_file()} | {p.name for p in CUR.iterdir() if p.is_file()})
for name in names:
    a, b = BAK / name, CUR / name
    if not a.is_file() or not b.is_file():
        rows.append((name, "MISSING", "", ""))
        continue
    ha, hb = sha(a), sha(b)
    rows.append((name, "same" if ha == hb else "CHANGED", a.stat().st_size, b.stat().st_size))

changed = [r for r in rows if r[1] != "same"]
same = [r for r in rows if r[1] == "same"]
print(f"unchanged: {len(same)}")
print(f"changed  : {len(changed)}")
for name, state, sa, sb in changed:
    print(f"  {state:<8} {name:<48} {sa} -> {sb}")

print("\ndata exports (must be unchanged):")
for name in ["training_monitoring_points.csv", "training_monitoring_runs.csv",
             "training_monitoring_contract.json", "chinese_result_figures.json",
             "v3_inventory_folds_zh.csv"]:
    a, b = BAK / name, CUR / name
    if a.is_file() and b.is_file():
        print(f"  {'same' if sha(a) == sha(b) else 'CHANGED':<8} {name}")
    else:
        print(f"  MISSING  {name}")
