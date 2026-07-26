from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DATASET = Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v2")
AUDIT = Path(r"D:\A_projects\swir_rgb_dataset_pipeline\artifacts\current_audit")
FIGURES = REPO / "figures"
SOURCE_DATA = FIGURES / "source_data"
GENERATED = REPO / "generated"

BLUE = "#2F6B8A"
GOLD = "#D8A03D"
PURPLE = "#7A6FA3"
SLATE = "#59636E"
PALE_BLUE = "#DCEAF1"
PALE_GOLD = "#F5E8C8"

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def save_figure(fig: plt.Figure, stem: str, dpi: int) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": dpi}),
        (".png", {"dpi": 220}),
    ):
        fig.savefig(FIGURES / f"{stem}{suffix}", bbox_inches="tight", **kwargs)


def display_array(path: Path) -> tuple[np.ndarray, float, float]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB" if image.mode == "RGB" else "L"))
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max() or 1)
    shown = np.clip((array.astype(np.float32) - low) / (high - low), 0, 1)
    return shown, float(low), float(high)


def figure_scene_plate() -> None:
    modalities = ("RGB", "B1", "B2", "B3", "B4", "B5")
    records: list[dict[str, object]] = []
    fig, axes = plt.subplots(10, 6, figsize=(7.2, 10.1), constrained_layout=True)
    for row, number in enumerate(range(1, 11)):
        scene = f"C{number}"
        for column, modality in enumerate(modalities):
            filename = (
                f"scene{number}.tif"
                if modality == "RGB"
                else f"scene{number}_band{modality[1:]}.tif"
            )
            path = DATASET / scene / filename
            shown, low, high = display_array(path)
            ax = axes[row, column]
            ax.imshow(shown, cmap=None if modality == "RGB" else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(modality, fontweight="bold", color=BLUE if modality == "RGB" else SLATE)
            if column == 0:
                ax.set_ylabel(scene, rotation=0, labelpad=14, va="center", fontweight="bold")
            records.append(
                {
                    "scene": scene,
                    "modality": modality,
                    "filename": filename,
                    "display_percentile_low": 1,
                    "display_percentile_high": 99,
                    "display_value_low": low,
                    "display_value_high": high,
                }
            )
    fig.suptitle(
        "Ten paired scenes with RGB and five SWIR bands",
        fontsize=10,
        fontweight="bold",
        y=1.006,
    )
    save_figure(fig, "fig1_scene_modality_overview", dpi=300)
    plt.close(fig)
    pd.DataFrame(records).to_csv(
        SOURCE_DATA / "fig1_display_provenance.csv", index=False, encoding="utf-8-sig"
    )


def figure_scale_quality() -> pd.DataFrame:
    audit = pd.read_csv(AUDIT / "scene_summary.csv")
    quality = pd.read_csv(DATASET / "metadata" / "scene_quality_summary.csv")
    data = audit.merge(
        quality[
            [
                "scene",
                "rgb_b1_holdout_p95_preview_px",
                "rgb_b1_identity_corner_deviation_preview_px",
            ]
        ],
        on="scene",
        how="left",
    )
    data["scene_number"] = data["scene"].str.removeprefix("C").astype(int)
    data = data.sort_values("scene_number")
    data.to_csv(SOURCE_DATA / "fig2_scene_quality.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1), hspace=0.38, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    colors = [GOLD if value else BLUE for value in data["excluded_patch_count"] > 0]
    ax_a.bar(data["scene"], data["usable_patch_count"], color=colors, width=0.72)
    ax_a.set_ylabel("Usable 256×256 patches")
    ax_a.set_title("a  Scene-level patch inventory", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color="#E6E9EC", linewidth=0.6)
    for x, y in zip(data["scene"], data["usable_patch_count"]):
        ax_a.text(x, y + 7, f"{int(y)}", ha="center", va="bottom", fontsize=5.5)

    size = np.sqrt(data["usable_patch_count"]) * 7
    ax_b.scatter(data["width"], data["height"], s=size, color=BLUE, alpha=0.82, edgecolor="white", linewidth=0.5)
    for _, row in data.iterrows():
        ax_b.annotate(row["scene"], (row["width"], row["height"]), xytext=(3, 2), textcoords="offset points", fontsize=5.5)
    ax_b.set_xlabel("Width (pixels)")
    ax_b.set_ylabel("Height (pixels)")
    ax_b.set_title("b  Scene extent", loc="left", fontweight="bold")
    ax_b.grid(color="#E6E9EC", linewidth=0.6)

    x = np.arange(len(data))
    ax_c.plot(x, data["rgb_b1_holdout_p95_preview_px"], marker="o", color=BLUE, label="RGB–B1 held-out P95")
    ax_c.plot(
        x,
        data["rgb_b1_identity_corner_deviation_preview_px"],
        marker="s",
        color=PURPLE,
        label="RGB–B1 corner deviation",
    )
    ax_c.set_xticks(x, data["scene"])
    ax_c.set_ylabel("Automated residual (pixels)")
    ax_c.set_title("c  Registration diagnostics", loc="left", fontweight="bold")
    ax_c.grid(axis="y", color="#E6E9EC", linewidth=0.6)
    ax_c.legend(loc="upper left")

    fig.suptitle("Dataset scale and reproducible quality diagnostics", fontsize=10, fontweight="bold")
    save_figure(fig, "fig2_scale_quality_diagnostics", dpi=600)
    plt.close(fig)
    return data


def figure_pipeline() -> None:
    stages = [
        ("Paired acquisition", PALE_BLUE),
        ("Source inventory\n& SHA-256", PALE_BLUE),
        ("Registration,\nmosaicking & crop", PALE_GOLD),
        ("Scene-level\nquality gates", PALE_GOLD),
        ("Privacy-aware\npatches & splits", "#E8E2F2"),
        ("Benchmark &\npaper evidence", "#E8E2F2"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.55)
    ax.axis("off")
    centers = np.linspace(0.52, 5.48, len(stages))
    for index, ((label, fill), x) in enumerate(zip(stages, centers)):
        box = FancyBboxPatch(
            (x - 0.42, 0.54),
            0.84,
            0.48,
            boxstyle="round,pad=0.04,rounding_size=0.04",
            facecolor=fill,
            edgecolor=SLATE,
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(x, 0.78, label, ha="center", va="center", fontsize=6.4, fontweight="bold")
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(centers[index + 1] - 0.45, 0.78),
                xytext=(x + 0.45, 0.78),
                arrowprops={"arrowstyle": "-|>", "color": SLATE, "lw": 1.0},
            )
    ax.text(
        3,
        1.27,
        "Read-only sources → validated data products → traceable computational evidence",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        3,
        0.23,
        "Every published count, table and quantitative panel is generated from manifests or experiment outputs.",
        ha="center",
        va="center",
        fontsize=6.5,
        color=SLATE,
    )
    save_figure(fig, "fig3_data_evidence_pipeline", dpi=600)
    plt.close(fig)


def write_macros(data: pd.DataFrame) -> None:
    train = {"C1", "C3", "C4", "C5", "C6", "C7", "C10"}
    validation = {"C8"}
    test = {"C2", "C9"}
    counts = data.set_index("scene")["usable_patch_count"].to_dict()
    privacy = int(data["excluded_patch_count"].sum())
    lines = [
        "% Generated by scripts/generate_dataset_figures.py.",
        r"\newcommand{\DatasetSceneCount}{" + str(len(data)) + "}",
        r"\newcommand{\DatasetTiffCount}{" + str(len(data) * 6) + "}",
        r"\newcommand{\DatasetUsablePatchCount}{" + str(int(data["usable_patch_count"].sum())) + "}",
        r"\newcommand{\DatasetPrivacyExcludedCount}{" + str(privacy) + "}",
        r"\newcommand{\DatasetFixedTrainCount}{" + str(sum(int(counts[x]) for x in train)) + "}",
        r"\newcommand{\DatasetFixedValidationCount}{" + str(sum(int(counts[x]) for x in validation)) + "}",
        r"\newcommand{\DatasetFixedTestCount}{" + str(sum(int(counts[x]) for x in test)) + "}",
    ]
    (GENERATED / "dataset_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    split_lookup = {
        **{scene: "Train" for scene in train},
        **{scene: "Validation" for scene in validation},
        **{scene: "Test" for scene in test},
    }
    table_rows_en = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Scene-level data inventory generated from the validated manifest.}",
        r"\label{tab:scene-inventory}",
        r"\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Scene & Width & Height & TIFFs & Usable patches & Fixed split \\",
        r"\midrule",
    ]
    table_rows_zh = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{由验证清单自动生成的场景级数据统计。}",
        r"\label{tab:scene-inventory-zh}",
        r"\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"场景 & 宽度 & 高度 & TIFF数 & 有效图块 & 固定划分 \\",
        r"\midrule",
    ]
    split_zh = {"Train": "训练", "Validation": "验证", "Test": "测试"}
    for _, row in data.iterrows():
        values = (
            row["scene"],
            int(row["width"]),
            int(row["height"]),
            int(row["tiff_count"]),
            int(row["usable_patch_count"]),
            split_lookup[row["scene"]],
        )
        table_rows_en.append(
            f"{values[0]} & {values[1]} & {values[2]} & {values[3]} & {values[4]} & {values[5]} \\\\"
        )
        table_rows_zh.append(
            f"{values[0]} & {values[1]} & {values[2]} & {values[3]} & {values[4]} & {split_zh[values[5]]} \\\\"
        )
    table_tail = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GENERATED / "scene_inventory_en.tex").write_text(
        "\n".join(table_rows_en + table_tail) + "\n", encoding="utf-8"
    )
    (GENERATED / "scene_inventory_zh.tex").write_text(
        "\n".join(table_rows_zh + table_tail) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Refresh macros and scene tables without regenerating figure files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        data = pd.read_csv(SOURCE_DATA / "fig2_scene_quality.csv")
    else:
        figure_scene_plate()
        data = figure_scale_quality()
        figure_pipeline()
    write_macros(data)
    print(f"Generated manuscript data products in {REPO}")


if __name__ == "__main__":
    main()
