#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild four result figures from saved measurements, without inference.

Figure contract: quantitative comparisons of nine conditions; a descriptive
first-fold window diagnostic separated from five-fold full-scene results;
signed response to displacement; and sensitivity to aggregation. The figures
use one seed, report no new statistical inference, and retain the existing
PDF/SVG/PNG outputs with editable vector text and a source-data manifest.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
SUM = Path(r"D:\USTC\Gen\v3_pipeline\outputs\summary_v2_20261005")
PIPELINE_OUTPUTS = SUM.parent
DIAGNOSTIC = Path(r"D:\USTC\Gen\outputs\fitting_diagnostic_20261015\fitting_diagnostic.json")
TRACE = {"revision_date": "2026-10-15", "scope": "Saved measurements only; no training, inference, or new statistical tests",
         "seed": 11, "benchmark_folds": 5, "benchmark_sites": 10}
SOURCE_PATHS = [SUM / "site_records.csv", SUM / "main_table.csv", SUM / "ex3_sensitivity.csv", DIAGNOSTIC]
GRID = "#E6E6E6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
})

C_MODEL = "#185FA5"
C_ABL = "#85B7EB"
C_LIN = "#BA7517"
C_CHAN = "#8A6A3D"
C_SEEN = "#0F6E56"
C_HELD = "#E24B4A"


def read_rows():
    with open(SUM / "site_records.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_main():
    with open(SUM / "main_table.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_ex3_summary():
    """汇总段（site_equal）位于第二个表头之后。"""
    rows = list(csv.reader(open(SUM / "ex3_sensitivity.csv", encoding="utf-8-sig")))
    for i, r in enumerate(rows):
        if r and r[0] == "mode" and len(r) > 2 and r[2] == "site_equal":
            return [dict(zip(r, x)) for x in rows[i + 1:] if x and x[0] in ("B1_only", "joint")]
    return []


LABEL = {
    "unet_bands1,2,3,4,5": "U-Net (5)",
    "pix2pix_bands1,2,3,4,5": "pix2pix (5)",
    "tfd_bands1,2,3,4,5": "TFD-GAN (5)",
    "restormer_bands1,2,3,4,5": "Restormer (5)",
    "pix2pix_bands1": "pix2pix (B1)",
    "pix2pix_bands1,2,4": "pix2pix (B124)",
    "reference_bands1,2,3,4,5": "Reference",
    "reference_bands1": "Reference (B1)",
    "reference_bands1,2,4": "Reference (B124)",
}
KIND = {
    "unet_bands1,2,3,4,5": "model",
    "pix2pix_bands1,2,3,4,5": "model",
    "tfd_bands1,2,3,4,5": "model",
    "restormer_bands1,2,3,4,5": "model",
    "pix2pix_bands1": "model",
    "pix2pix_bands1,2,4": "model",
}


def fig_generalization(rows):
    """Keep the diagnostic and benchmark evaluation units explicit."""
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    groups = [[r for r in diagnostic["per_site"] if r["group"] == name]
              for name in ("train_seen", "test_heldout")]
    assert {r["site"] for r in groups[0]} == set(diagnostic["train_sites"])
    assert {r["site"] for r in groups[1]} == set(diagnostic["test_sites"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # 左：两组均值 + B1 灰度复制参照
    seen, held = [[r["window_mean_psnr"] for r in records] for records in groups]
    ref_seen, ref_held = [float(np.mean([r["gray_copy_baseline_psnr"] for r in records])) for records in groups]
    means = [float(np.mean(seen)), float(np.mean(held))]
    gains = [value - reference for value, reference in zip(means, [ref_seen, ref_held])]
    gap = means[0] - means[1]
    TRACE["generalization"] = {"left_source": str(DIAGNOSTIC), "left_per_site": diagnostic["per_site"],
        "left_unit": "equal mean of within-site window-mean PSNR; first fold only",
        "left_source_precision": "saved per-site values have four decimal places",
        "left_model_means": means, "left_gray_means": [ref_seen, ref_held], "left_gains": gains, "left_gap": gap,
        "right_source": str(SUM / "site_records.csv"), "right_unit": "five-fold full-scene support PSNR per held-out site"}
    ax1.bar([0, 1], [np.mean(seen), np.mean(held)],
            color=["#93C7B9", "#E9A5A3"], width=0.55, zorder=3)
    ax1.scatter([0, 1], [ref_seen, ref_held], marker="_", s=420,
                color="#5F5E5A", linewidth=2.0, zorder=5)
    for i, (v, r) in enumerate([(np.mean(seen), ref_seen), (np.mean(held), ref_held)]):
        ax1.text(i + 0.24, v + 1.0, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        ax1.plot([i - 0.22, i + 0.22], [r, r], color="#2C2C2A", lw=1.6, zorder=5)
        ax1.text(i, r - 2.6, f"{r:.2f}", ha="center", fontsize=7.8, color="#2C2C2A")
        ax1.text(i, r - 5.2, f"+{v - r:.2f} dB", ha="center", fontsize=8.4,
                 color="#2C2C2A", fontweight="bold")
    for x, vals in zip([0, 1], [seen, held]):
        ax1.scatter([x] * len(vals), vals, s=18, color="white",
                    edgecolor="#2C2C2A", linewidth=0.6, zorder=4)
    ax1.text(0.5, 31.6, f"gap {gap:.2f} dB", ha="center", fontsize=9.0,
             fontweight="bold", color="#2C2C2A")
    ax1.text(0.5, -0.31, "Open circles: site means; C1: 1 window\nBlack marks: B1 grayscale-copy reference",
             transform=ax1.transAxes, ha="center", va="top", fontsize=7.2, color="#5F5E5A")
    ax1.set_xticks([0, 1])
    window_counts = [sum(r["n_windows"] for r in records) for records in groups]
    ax1.set_xticklabels([f"Training\n3 sites / {window_counts[0]} windows", f"Held-out\n2 sites / {window_counts[1]} windows"], fontsize=8)
    ax1.set_title("a  First fold: fixed windows", loc="left", fontsize=9, fontweight="bold", pad=10)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_ylim(0, 34)
    ax1.set_xlim(-0.45, 1.65)
    ax1.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # 右：全部条件的逐地点分布
    sites = [f"C{i}" for i in range(1, 11)]
    x = np.arange(10)
    for cond, col, mk, lab in [
        ("restormer_bands1,2,3,4,5", C_MODEL, "o", "Restormer (5)"),
        ("pix2pix_bands1,2,3,4,5", "#1D9E75", "s", "pix2pix (5)"),
    ]:
        v = [next(float(r["psnr"]) for r in rows
                  if r["condition"] == cond and r["site"] == s) for s in sites]
        ax2.plot(x, v, marker=mk, color=col, lw=1.1, ms=3.6, label=lab, zorder=3)
    lr = [next(float(r["psnr"]) for r in rows
               if r["condition"] == "reference_bands1,2,3,4,5" and r["kind"] == "linear_reference"
               and r["site"] == s) for s in sites]
    ax2.plot(x, lr, marker="^", color=C_LIN, lw=1.1, ms=4.0,
             label="Linear reference", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sites)
    ax2.set_xlabel("Held-out site")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper right", fontsize=7)
    ax2.set_title("b  Five folds: full-scene support", loc="left", fontsize=9, fontweight="bold", pad=10)
    ax2.set_ylim(10.5, 19.8)
    ax2.text(0.5, -0.31, "Each site is tested in its assigned fold\nA different evaluation unit from panel a",
             transform=ax2.transAxes, ha="center", va="top", fontsize=7.2, color="#5F5E5A")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.28, wspace=0.35)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"figV9_generalization_gap.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_baseline(rows):
    """图4：九条件 PSNR 对照。"""
    order = [
        ("reference_bands1,2,3,4,5", "linear_reference", "Linear reference (5)", C_LIN),
        ("reference_bands1,2,4", "linear_reference", "Linear reference (B124)", "#D9A05B"),
        ("restormer_bands1,2,3,4,5", "model", "Restormer (5)", C_MODEL),
        ("reference_bands1", "linear_reference", "Linear reference (B1)", "#E0BE8E"),
        ("pix2pix_bands1,2,4", "model", "pix2pix (B124)", C_ABL),
        ("pix2pix_bands1", "model", "pix2pix (B1)", "#B8D8F5"),
        ("tfd_bands1,2,3,4,5", "model", "TFD-GAN (5)", "#2E8B8B"),
        ("unet_bands1,2,3,4,5", "model", "U-Net (5)", "#4A7FBF"),
        ("pix2pix_bands1,2,3,4,5", "model", "pix2pix (5)", "#7FB3E0"),
    ]
    vals = []
    for cond, kind, lab, col in order:
        v = np.mean([float(r["psnr"]) for r in rows
                     if r["condition"] == cond and r["kind"] == kind])
        vals.append((v, lab, col))

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    y = np.arange(len(vals))
    vs = [v for v, _, _ in vals]
    for i, ((v, _, color), (_, kind, _, _)) in enumerate(zip(vals, order)):
        ax.scatter(v, i, color=color, marker="s" if kind == "linear_reference" else "o", s=35, zorder=3)
    for i, v in enumerate(vs):
        ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=8.4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3}, zorder=5)
    learned = [(v, lab) for (v, lab, _), (_, kind, _, _) in zip(vals, order) if kind == "model"]
    best, best_label = max(learned)
    TRACE["baseline"] = {"source": str(SUM / "site_records.csv"),
        "values": [{"condition": cond, "kind": kind, "label": lab, "psnr": float(v)}
                   for (cond, kind, lab, _), (v, _, _) in zip(order, vals)],
        "best_learned_label": best_label, "best_learned_psnr": float(best)}
    ax.axvline(best, color="#A32D2D", lw=0.9, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l, _ in vals])
    ax.invert_yaxis()
    ax.set_xlabel("site-equal PSNR (dB)")
    ax.set_xlim(12.2, 14.9)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], linestyle="", marker="s", color=C_LIN),
               Line2D([], [], linestyle="", marker="o", color=C_MODEL),
               Line2D([], [], linestyle="", marker="o", color=C_ABL),
               Line2D([], [], linestyle="", marker="o", color="#2E8B8B")]
    ax.legend(handles, ["training-only linear reference", "learned model (5 bands)",
                        "learned model (band ablation)", "TFD-GAN"],
              frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.40), fontsize=7.5)
    ax.text(0.5, 1.08, f"Dashed line: best learned model, Restormer ({best:.4f} dB)",
            transform=ax.transAxes, ha="center", fontsize=7.5)

    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.17, top=0.72)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"figV10_baseline_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity():
    """图5：受控错位响应。"""
    rows = read_ex3_summary()
    fig, ax = plt.subplots(figsize=(4.3, 3.15))
    TRACE["sensitivity"] = {"source": str(SUM / "ex3_sensitivity.csv"), "values": rows,
        "definition": "PSNR drop = baseline minus perturbed; positive means degradation",
        "aggregation": "four-direction average per site, then ten-site equal mean",
        "zero": "zero displacement shown explicitly; drop is definitionally zero"}
    for mode, col, mk, lab in [("joint", C_MODEL, "o", "all SWIR bands"),
                               ("B1_only", "#D9A05B", "s", "B1 only")]:
        pts = sorted([(float(r["magnitude_px"]), float(r["psnr_drop"]))
                      for r in rows if r["mode"] == mode])
        ax.plot([0] + [p[0] for p in pts], [0] + [p[1] for p in pts],
                marker=mk, color=col, lw=1.3, ms=4.4, label=lab, zorder=3)
        ax.annotate(f"{pts[-1][1]:.3f}", pts[-1], xytext=(-6, 7), textcoords="offset points",
                    ha="right", fontsize=8, color=col)
    ax.axhline(0, color="#909090", lw=0.7, zorder=1)
    ax.annotate("−0.004", (0.5, -0.004), xytext=(9, 13), textcoords="offset points",
                ha="left", fontsize=7.2, color="#A76D16",
                arrowprops={"arrowstyle": "-", "color": "#A76D16", "lw": 0.6})
    ax.set_xlabel("Displacement (pixels, B1 reference grid)")
    ax.set_ylabel("PSNR drop (dB; baseline − perturbed)")
    ax.set_xticks([0, 0.5, 1, 2, 3, 4])
    ax.set_xlim(-0.12, 4.15)
    ax.set_ylim(-0.028, 0.50)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"figV11_misalignment_sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_aggregation():
    """图6：两种聚合口径的排序对比。"""
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    groups = [[], []]
    support = json.loads((SUM / "export_metadata.json").read_text(encoding="utf-8"))["support_sha256"]
    SOURCE_PATHS.append(SUM / "export_metadata.json")
    for fold in range(1, 6):
        ref_path = PIPELINE_OUTPUTS / "reference_baselines_v2_20261002" / f"fold{fold}_bands1-2-3-4-5.json"
        model_path = PIPELINE_OUTPUTS / "draft_runs_v2_20261002" / f"restormer_fold{fold}_seed11_bands1-2-3-4-5" / "result.json"
        SOURCE_PATHS.extend([ref_path, model_path])
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        model = json.loads(model_path.read_text(encoding="utf-8"))
        assert reference["support_sha256"] == model["config"]["support_sha256"] == support
        groups[0].extend({"site": r["site"], "psnr": r["psnr"], "valid_pixels": r["valid_pixels"],
                          "mse": r["rmse"] ** 2, "mse_source": "saved full-precision RMSE squared"}
                         for r in reference["per_site"] if r["kind"] == "linear_reference")
        groups[1].extend({"site": site, "psnr": r["psnr"], "valid_pixels": r["valid_pixels"],
                          "mse": r["mse"], "mse_source": "saved full-precision MSE"}
                         for site, r in model["per_site"].items())
    for records in groups:
        assert len(records) == 10 and {r["site"] for r in records} == {f"C{i}" for i in range(1, 11)}
        records.sort(key=lambda r: int(r["site"][1:]))
    assert all(a["site"] == b["site"] and a["valid_pixels"] == b["valid_pixels"] for a, b in zip(*groups))
    eq = [float(np.mean([r["psnr"] for r in records])) for records in groups]
    pooled = [-10 * math.log10(sum(r["valid_pixels"] * r["mse"] for r in records) /
                               sum(r["valid_pixels"] for r in records)) for records in groups]
    TRACE["aggregation"] = {"condition_order": ["linear_reference_five", "restormer_five"],
        "per_site": groups, "site_equal_psnr": eq, "pixel_pooled_psnr": pooled,
        "definition": "pooled PSNR = -10 log10(sum(n_pixels * site_MSE) / sum(n_pixels)); intensity range 1",
        "precision": "full-precision saved metrics; no pooling from rounded CSV RMSE"}
    for index, color, marker, label in [(0, C_LIN, "s", "Linear reference (5)"),
                                         (1, C_MODEL, "o", "Restormer (5)")]:
        values = [eq[index], pooled[index]]
        ax.plot([0, 1], values, color=color, marker=marker, lw=1.2, ms=5, label=label, zorder=3)
        for x, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (x, value), xytext=(-7 if x == 0 else 7, 0),
                        textcoords="offset points", ha="right" if x == 0 else "left",
                        va="center", fontsize=8.2, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Site-equal\n(main analysis)", "Pixel-pooled MSE"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(-0.33, 1.33)
    ax.set_ylim(13.22, 14.28)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              handletextpad=0.4, columnspacing=0.9, fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"figV12_aggregation.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = read_rows()
    fig_generalization(rows)
    fig_baseline(rows)
    fig_sensitivity()
    fig_aggregation()
    TRACE["sources"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCE_PATHS]
    TRACE["export"] = {"backend": "Python/matplotlib", "formats": ["pdf", "svg", "png"],
                       "svg_fonttype": "none", "pdf_fonttype": 42, "grid": GRID,
                       "uncertainty": "No new intervals or significance tests; open dots are site means"}
    source_data = OUT / "source_data" / "revision_results_20261015.json"
    source_data.parent.mkdir(parents=True, exist_ok=True)
    source_data.write_text(json.dumps(TRACE, ensure_ascii=False, indent=2), encoding="utf-8")
    print("图表已生成：")
    for p in sorted(OUT.glob("fig*.pdf")):
        print("  ", p.name)
