"""Inspect which figure records carry hashes that regenerating figures would invalidate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

p = ROOT / "figures/zh/source_data/chinese_figures_audit.json"
d = json.loads(p.read_text(encoding="utf-8"))
print("chinese_figures_audit.json keys:", list(d.keys()))
def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]")
    else:
        if "sha" in path.lower() or "hash" in path.lower():
            print(f"  {path} = {node}")
walk(d)

for name in ["revision_results_20261015.json", "local_comparisons_20261015.json"]:
    q = ROOT / "figures/source_data" / name
    if q.is_file():
        j = json.loads(q.read_text(encoding="utf-8"))
        print(f"\n{name} top keys:", list(j.keys()) if isinstance(j, dict) else type(j))
        if isinstance(j, dict):
            for k, v in j.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"  {k}: list of {len(v)}, first keys {list(v[0].keys())[:8]}")
                    break
