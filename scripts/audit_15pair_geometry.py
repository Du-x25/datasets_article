"""Independent non-SIFT gradient-domain sub-pixel geometric registration audit across all 15 modality pairs.

Uses Sobel gradient magnitude (polarity-inversion invariant for VIS-SWIR) and
DFT sub-pixel phase cross-correlation (Guizar-Sicairos et al. 2008, upsample_factor=10)
on stride-256 valid mosaic tiles across all 10 sites and all 15 pairwise combinations of
{RGB, B1, B2, B3, B4, B5}, stratified by texture energy.

Outputs portable summary data, bilingual LaTeX tables/macros, and bilingual figures.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENES = [f"C{i}" for i in range(1, 11)]
MODALITIES = ["RGB", "B1", "B2", "B3", "B4", "B5"]
PAIRS = list(itertools.combinations(MODALITIES, 2))
PAIR_NAMES = [f"{a}-{b}" for a, b in PAIRS]

PATCH = 256
STRIDE = 256
MIN_MASK_FRAC = 0.95
MAX_SHIFT = 16.0
MIN_GRAD_CORR = 0.25
UPSAMPLE = 10


def load_site_modalities(source_root: Path, site: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    num = site[1:]
    mods: dict[str, np.ndarray] = {}
    rgb_path = source_root / site / f"scene{num}.tif"
    with Image.open(rgb_path) as im:
        rgb = np.asarray(im, dtype=np.float32)
    mods["RGB"] = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    for b in range(1, 6):
        bp = source_root / site / f"scene{num}_band{b}.tif"
        with Image.open(bp) as im:
            mods[f"B{b}"] = np.asarray(im, dtype=np.float32)
    mask_path = source_root / "usability_masks" / site / "joint.png"
    with Image.open(mask_path) as im:
        mask = np.asarray(im) > 0
    return mods, mask


def grad_mag(img: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    gx = ndimage.sobel(img, axis=1, mode="reflect")
    gy = ndimage.sobel(img, axis=0, mode="reflect")
    return np.hypot(gx, gy).astype(np.float32)


def shifted_ncc(a: np.ndarray, b: np.ndarray, dy: float, dx: float) -> float:
    from scipy import ndimage
    b_shift = ndimage.shift(b, shift=(dy, dx), order=1, mode="nearest")
    m = 16
    ac = a[m:-m, m:-m]
    bc = b_shift[m:-m, m:-m]
    ac = ac - ac.mean()
    bc = bc - bc.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(ac * bc) / denom)


def run_audit(source_root: Path) -> dict:
    from skimage.registration import phase_cross_correlation
    hann1d = np.hanning(PATCH).astype(np.float32)
    hann2d = np.outer(hann1d, hann1d)

    records: list[dict] = []
    site_tile_counts: dict[str, int] = {}

    for site in SCENES:
        mods, mask = load_site_modalities(source_root, site)
        h, w = mask.shape
        grads = {m: grad_mag(arr) for m, arr in mods.items()}

        tiles = [
            (y, x)
            for y in range(0, h - PATCH + 1, STRIDE)
            for x in range(0, w - PATCH + 1, STRIDE)
            if mask[y : y + PATCH, x : x + PATCH].mean() >= MIN_MASK_FRAC
        ]
        site_tile_counts[site] = len(tiles)
        if not tiles:
            continue

        # Texture energy per tile using mean of RGB & B1 gradient SD
        tex_scores = np.array(
            [0.5 * (grads["RGB"][y : y + PATCH, x : x + PATCH].std() + grads["B1"][y : y + PATCH, x : x + PATCH].std())
             for y, x in tiles],
            dtype=np.float32,
        )
        tex_med = float(np.median(tex_scores))

        for (y, x), tex_val in zip(tiles, tex_scores):
            tex_tier = "high" if tex_val >= tex_med else "low"
            patch_grads = {m: grads[m][y : y + PATCH, x : x + PATCH] for m in MODALITIES}
            patch_win = {m: (g - g.mean()) * hann2d for m, g in patch_grads.items()}

            for ma, mb in PAIRS:
                shift, _, _ = phase_cross_correlation(
                    patch_win[ma],
                    patch_win[mb],
                    upsample_factor=UPSAMPLE,
                    normalization="phase",
                )
                dy, dx = float(shift[0]), float(shift[1])
                mag = float(np.hypot(dy, dx))
                if mag <= MAX_SHIFT:
                    r_peak = shifted_ncc(patch_grads[ma], patch_grads[mb], dy, dx)
                else:
                    r_peak = 0.0
                judgeable = bool(mag <= MAX_SHIFT and r_peak >= MIN_GRAD_CORR)
                records.append({
                    "site": site,
                    "pair": f"{ma}-{mb}",
                    "group": "RGB-SWIR" if ma == "RGB" else "SWIR-SWIR",
                    "y": y,
                    "x": x,
                    "tex_tier": tex_tier,
                    "dy_px": round(dy, 3),
                    "dx_px": round(dx, 3),
                    "err_px": round(mag, 3),
                    "r_grad": round(r_peak, 4),
                    "judgeable": judgeable,
                })
        print(f"Audited {site}: {len(tiles)} valid 256x256 tiles x 15 pairs = {len(tiles) * 15} checks", flush=True)

    def summarize(rows: list[dict]) -> dict:
        n_total = len(rows)
        jud = [r["err_px"] for r in rows if r["judgeable"]]
        n_jud = len(jud)
        if not jud:
            return {"n_total": n_total, "n_judgeable": 0, "judgeable_pct": 0.0,
                    "median_px": None, "rmse_px": None, "p95_px": None, "within_2px_pct": None}
        a = np.asarray(jud, dtype=float)
        return {
            "n_total": n_total,
            "n_judgeable": n_jud,
            "judgeable_pct": round(100.0 * n_jud / max(1, n_total), 1),
            "median_px": round(float(np.median(a)), 2),
            "rmse_px": round(float(np.sqrt(np.mean(a ** 2))), 2),
            "p95_px": round(float(np.percentile(a, 95)), 2),
            "within_2px_pct": round(float(100.0 * np.mean(a <= 2.0)), 1),
        }

    by_pair = {p: summarize([r for r in records if r["pair"] == p]) for p in PAIR_NAMES}
    by_group = {g: summarize([r for r in records if r["group"] == g]) for g in ("RGB-SWIR", "SWIR-SWIR")}
    by_group_tex = {
        f"{g}_{t}": summarize([r for r in records if r["group"] == g and r["tex_tier"] == t])
        for g in ("RGB-SWIR", "SWIR-SWIR")
        for t in ("high", "low")
    }
    by_site = {
        s: {
            "tiles": site_tile_counts[s],
            "all": summarize([r for r in records if r["site"] == s]),
            "RGB-SWIR": summarize([r for r in records if r["site"] == s and r["group"] == "RGB-SWIR"]),
            "SWIR-SWIR": summarize([r for r in records if r["site"] == s and r["group"] == "SWIR-SWIR"]),
        }
        for s in SCENES
    }
    matrix_median = {
        s: {
            p: summarize([r for r in records if r["site"] == s and r["pair"] == p])["median_px"]
            for p in PAIR_NAMES
        }
        for s in SCENES
    }

    return {
        "protocol": {
            "method": "Independent non-SIFT Sobel gradient-magnitude DFT sub-pixel phase cross-correlation",
            "patch_px": PATCH,
            "stride_px": STRIDE,
            "min_mask_fraction": MIN_MASK_FRAC,
            "upsample_factor": UPSAMPLE,
            "max_shift_px": MAX_SHIFT,
            "min_grad_ncc": MIN_GRAD_CORR,
            "note": "Completely independent of pipeline SIFT keypoints; automated sub-pixel audit (not human annotation).",
        },
        "overall": summarize(records),
        "by_group": by_group,
        "by_group_texture": by_group_tex,
        "by_pair": by_pair,
        "by_site": by_site,
        "matrix_median_px": matrix_median,
    }


def write_tables_and_macros(summary: dict) -> None:
    gen = REPO_ROOT / "generated"
    gen.mkdir(parents=True, exist_ok=True)

    # 1. Macros
    rgb_s = summary["by_group"]["RGB-SWIR"]
    swir_s = summary["by_group"]["SWIR-SWIR"]
    ov = summary["overall"]
    rgb_hi = summary["by_group_texture"]["RGB-SWIR_high"]
    rgb_lo = summary["by_group_texture"]["RGB-SWIR_low"]
    sw_hi = summary["by_group_texture"]["SWIR-SWIR_high"]
    sw_lo = summary["by_group_texture"]["SWIR-SWIR_low"]

    macros = [
        f"\\newcommand{{\\AuditTotalTiles}}{{{sum(v['tiles'] for v in summary['by_site'].values())}}}",
        f"\\newcommand{{\\AuditTotalChecks}}{{{ov['n_total']}}}",
        f"\\newcommand{{\\AuditJudgeablePct}}{{{ov['judgeable_pct']:.1f}}}",
        f"\\newcommand{{\\AuditRgbSwirMed}}{{{rgb_s['median_px']:.2f}}}",
        f"\\newcommand{{\\AuditRgbSwirRmse}}{{{rgb_s['rmse_px']:.2f}}}",
        f"\\newcommand{{\\AuditRgbSwirPnineFive}}{{{rgb_s['p95_px']:.2f}}}",
        f"\\newcommand{{\\AuditRgbSwirJudPct}}{{{rgb_s['judgeable_pct']:.1f}}}",
        f"\\newcommand{{\\AuditSwirSwirMed}}{{{swir_s['median_px']:.2f}}}",
        f"\\newcommand{{\\AuditSwirSwirRmse}}{{{swir_s['rmse_px']:.2f}}}",
        f"\\newcommand{{\\AuditSwirSwirPnineFive}}{{{swir_s['p95_px']:.2f}}}",
        f"\\newcommand{{\\AuditSwirSwirJudPct}}{{{swir_s['judgeable_pct']:.1f}}}",
        f"\\newcommand{{\\AuditRgbHiMed}}{{{rgb_hi['median_px']:.2f}}}",
        f"\\newcommand{{\\AuditRgbLoMed}}{{{rgb_lo['median_px']:.2f}}}",
        f"\\newcommand{{\\AuditSwirHiMed}}{{{sw_hi['median_px']:.2f}}}",
        f"\\newcommand{{\\AuditSwirLoMed}}{{{sw_lo['median_px']:.2f}}}",
    ]
    (gen / "geom_15pair_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    # 2. Compact 15-pair table (ZH and EN)
    def make_pair_table(lang: str) -> str:
        if lang == "zh":
            head = (
                "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
                "模态对 & 类别 & 可判定率/\\% & 中位误差/px & RMSE/px & P95/px & $\\le$2\\,px占比/\\%\\\\\n\\midrule\n"
            )
            gmap = {"RGB-SWIR": "跨模态", "SWIR-SWIR": "SWIR波段间"}
            sum_rgb, sum_swir = "RGB--SWIR汇总（5对）", "SWIR--SWIR汇总（10对）"
        else:
            head = (
                "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
                "Modality pair & Group & Judgeable/\\% & Median/px & RMSE/px & P95/px & $\\le$2\\,px/\\%\\\\\n\\midrule\n"
            )
            gmap = {"RGB-SWIR": "Cross-modal", "SWIR-SWIR": "Intra-SWIR"}
            sum_rgb, sum_swir = "RGB--SWIR pooled (5 pairs)", "SWIR--SWIR pooled (10 pairs)"

        rows = []
        for p in PAIR_NAMES:
            st = summary["by_pair"][p]
            grp = "RGB-SWIR" if p.startswith("RGB-") else "SWIR-SWIR"
            rows.append(
                f"{p.replace('-', '--')} & {gmap[grp]} & {st['judgeable_pct']:.1f} & "
                f"{st['median_px']:.2f} & {st['rmse_px']:.2f} & {st['p95_px']:.2f} & {st['within_2px_pct']:.1f}\\\\"
            )
        rows.append("\\midrule")
        for label, grp_key in ((sum_rgb, "RGB-SWIR"), (sum_swir, "SWIR-SWIR")):
            st = summary["by_group"][grp_key]
            rows.append(
                f"\\textbf{{{label}}} & -- & \\textbf{{{st['judgeable_pct']:.1f}}} & "
                f"\\textbf{{{st['median_px']:.2f}}} & \\textbf{{{st['rmse_px']:.2f}}} & "
                f"\\textbf{{{st['p95_px']:.2f}}} & \\textbf{{{st['within_2px_pct']:.1f}}}\\\\"
            )
        return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"

    (gen / "geom_15pair_zh.tex").write_text(make_pair_table("zh"), encoding="utf-8")
    (gen / "geom_15pair_en.tex").write_text(make_pair_table("en"), encoding="utf-8")


def plot_figures(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.array([[summary["matrix_median_px"][s][p] for p in PAIR_NAMES] for s in SCENES], dtype=float)

    for lang in ("zh", "en"):
        if lang == "zh":
            plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                                 "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8})
            out_path = REPO_ROOT / "figures/zh/geom_15pair_audit_zh.pdf"
            title_a = "a  全15对模态逐地点梯度相位相关中位残差（输出像素）"
            title_b = "b  高纹理 vs. 低纹理分层中位残差"
            cbar_lbl = "中位亚像素位移 / px"
            ylabel_b = "中位亚像素位移 / px"
            cats = ["RGB-SWIR\n高纹理", "RGB-SWIR\n低纹理", "SWIR-SWIR\n高纹理", "SWIR-SWIR\n低纹理"]
        else:
            plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],
                                 "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8})
            out_path = REPO_ROOT / "figures/en/geom_15pair_audit_en.pdf"
            title_a = "a  Non-SIFT gradient phase-correlation median residual across 15 modality pairs (output px)"
            title_b = "b  Texture-stratified median residual"
            cbar_lbl = "Median sub-pixel shift / px"
            ylabel_b = "Median sub-pixel shift / px"
            cats = ["RGB-SWIR\nhigh tex.", "RGB-SWIR\nlow tex.", "SWIR-SWIR\nhigh tex.", "SWIR-SWIR\nlow tex."]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.6), gridspec_kw={"width_ratios": [2.35, 1.2]}, layout="constrained")
        im = ax1.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(3.5, float(np.nanpercentile(mat, 95))))
        ax1.set_xticks(range(len(PAIR_NAMES)), PAIR_NAMES, rotation=45, ha="right", fontsize=7)
        ax1.set_yticks(range(len(SCENES)), SCENES, fontsize=7.5)
        ax1.axvline(4.5, color="#b91c1c", lw=1.2, ls="--")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                if np.isfinite(val):
                    txt_color = "white" if val > 2.0 else "#111827"
                    ax1.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=6, color=txt_color)
        ax1.set_title(title_a, fontsize=8.5, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax1, shrink=0.88, pad=0.025)
        cbar.set_label(cbar_lbl, fontsize=7.5)

        keys = ["RGB-SWIR_high", "RGB-SWIR_low", "SWIR-SWIR_high", "SWIR-SWIR_low"]
        vals = [summary["by_group_texture"][k]["median_px"] for k in keys]
        p95s = [summary["by_group_texture"][k]["p95_px"] for k in keys]
        colors = ["#1f5553", "#6aa5a1", "#2b5c8f", "#8cb1d1"]
        bars = ax2.bar(range(4), vals, color=colors, width=0.56)
        for i, (v, p95) in enumerate(zip(vals, p95s)):
            ax2.text(i, v + 0.06, f"{v:.2f}\n(P95:{p95:.1f})", ha="center", va="bottom", fontsize=6.5)
        ax2.set_xticks(range(4), cats, fontsize=6.3)
        ax2.set_ylim(0, max(vals) * 1.48)
        ax2.set_ylabel(ylabel_b, fontsize=7.5)
        ax2.set_title(title_b, fontsize=8.5, fontweight="bold")
        ax2.grid(axis="y", color="#dddddd", lw=0.6)
        ax2.set_axisbelow(True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v3"))
    parser.add_argument("--from-json", action="store_true", help="Regenerate tables/figures from saved geom_15pair_audit.json without reading TIFFs.")
    args = parser.parse_args()

    out_json = REPO_ROOT / "figures/source_data/geom_15pair_audit.json"
    if args.from_json:
        summary = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        summary = run_audit(args.source_root)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_tables_and_macros(summary)
    plot_figures(summary)
    print(json.dumps({
        "overall": summary["overall"],
        "by_group": summary["by_group"],
        "by_group_texture": summary["by_group_texture"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
