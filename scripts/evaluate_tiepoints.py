"""Initialize and summarize blinded human tie-point geometric validation across all 15 modality pairs.

Generates a structured CSV template covering all 10 sites (C1-C10) and all 15 pairwise
combinations of {RGB, B1, B2, B3, B4, B5}, and computes per-site / per-group RMSE, median,
P95, and max residual error tables (JSON + LaTeX) once tie-points are annotated.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
import numpy as np

SCENES = [f"C{i}" for i in range(1, 11)]
MODALITIES = ["RGB", "B1", "B2", "B3", "B4", "B5"]
PAIRS = [f"{a}-{b}" for a, b in itertools.combinations(MODALITIES, 2)]
FIELDS = [
    "scene",
    "modality_pair",
    "point_id",
    "region_type",
    "x_mod_a_px",
    "y_mod_a_px",
    "x_mod_b_px",
    "y_mod_b_px",
    "annotator_id",
    "notes",
]


def init_template(out_csv: Path, points_per_pair: int = 3) -> dict:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    region_cycle = ["high_texture", "low_texture", "edge_or_dynamic"]
    for scene in SCENES:
        for pair in PAIRS:
            for k in range(1, points_per_pair + 1):
                rows.append({
                    "scene": scene,
                    "modality_pair": pair,
                    "point_id": f"{scene}_{pair}_{k:02d}",
                    "region_type": region_cycle[(k - 1) % len(region_cycle)],
                    "x_mod_a_px": "",
                    "y_mod_a_px": "",
                    "x_mod_b_px": "",
                    "y_mod_b_px": "",
                    "annotator_id": "",
                    "notes": "",
                })
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {"template": str(out_csv), "scenes": len(SCENES), "modality_pairs": len(PAIRS), "rows": len(rows)}


def summarize_tiepoints(in_csv: Path, out_json: Path | None = None, out_tex: Path | None = None) -> dict:
    by_scene: dict[str, list[float]] = {s: [] for s in SCENES}
    by_group: dict[str, list[float]] = {"RGB-SWIR": [], "SWIR-SWIR": []}
    all_err: list[float] = []

    with in_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not row["x_mod_a_px"].strip() or not row["x_mod_b_px"].strip():
                continue
            dx = float(row["x_mod_b_px"]) - float(row["x_mod_a_px"])
            dy = float(row["y_mod_b_px"]) - float(row["y_mod_a_px"])
            err = float(np.hypot(dx, dy))
            all_err.append(err)
            if row["scene"] in by_scene:
                by_scene[row["scene"]].append(err)
            group = "RGB-SWIR" if row["modality_pair"].startswith("RGB-") else "SWIR-SWIR"
            by_group[group].append(err)

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "rmse_px": None, "median_px": None, "p95_px": None, "max_px": None}
        a = np.asarray(vals, dtype=float)
        return {
            "n": int(a.size),
            "rmse_px": round(float(np.sqrt(np.mean(a ** 2))), 3),
            "median_px": round(float(np.median(a)), 3),
            "p95_px": round(float(np.percentile(a, 95)), 3),
            "max_px": round(float(np.max(a)), 3),
        }

    summary = {
        "overall": stats(all_err),
        "by_group": {k: stats(v) for k, v in by_group.items()},
        "by_scene": {k: stats(v) for k, v in by_scene.items()},
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if out_tex:
        lines = [
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Site & Points & RMSE (px) & Median (px) & P95 (px) & Max (px)\\",
            r"\midrule",
        ]
        for s in SCENES:
            st = summary["by_scene"][s]
            if st["n"] == 0:
                lines.append(f"{s} & 0 & -- & -- & -- & --\\\\")
            else:
                lines.append(f"{s} & {st['n']} & {st['rmse_px']:.2f} & {st['median_px']:.2f} & {st['p95_px']:.2f} & {st['max_px']:.2f}\\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-template", type=Path, default=None, help="Write blank 10-site x 15-modality-pair tie-point CSV template.")
    parser.add_argument("--points-per-pair", type=int, default=3)
    parser.add_argument("--evaluate", type=Path, default=None, help="Evaluate a completed tie-point CSV.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-tex", type=Path, default=None)
    args = parser.parse_args()

    if args.init_template:
        print(json.dumps(init_template(args.init_template, args.points_per_pair), indent=2))
        return 0
    if args.evaluate:
        print(json.dumps(summarize_tiepoints(args.evaluate, args.out_json, args.out_tex), indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
