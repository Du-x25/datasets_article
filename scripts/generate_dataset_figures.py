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
DATASET = Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v3")
AUDIT = Path(r"D:\A_projects\swir_rgb_dataset_pipeline\artifacts\audit_v3")
TASK_VALIDATION = Path(
    r"D:\Data_dir\SWIR-RGB\M-SWIR-RGB\tasks_v3\task_validation.json"
)
FIGURES = REPO / "figures"
SOURCE_DATA = FIGURES / "source_data"
GENERATED = REPO / "generated"

# Quality-gate thresholds, mirrored from
# D:/A_projects/swir_rgb_scene_mosaic/configs/rawdata/base.yaml (validation block).
PREVIEW_MAXIMUM_DIMENSION = 1200
GATE_HOLDOUT_P95_PREVIEW_PX = 2.5
GATE_CORNER_DEVIATION_PREVIEW_PX = 8.0
GATE_INTRA_SWIR_OUTPUT_P95_PX = 8.0

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
                "maximum_intra_swir_output_p95_px",
            ]
        ],
        on="scene",
        how="left",
    )
    data["scene_number"] = data["scene"].str.removeprefix("C").astype(int)
    data = data.sort_values("scene_number").reset_index(drop=True)

    # Registration gates are defined on 1200-pixel preview images, so converting
    # them to released pixels needs the per-scene preview scale factor.
    data["preview_scale"] = data[["width", "height"]].max(axis=1) / PREVIEW_MAXIMUM_DIMENSION
    data["rgb_b1_holdout_p95_output_px"] = (
        data["rgb_b1_holdout_p95_preview_px"] * data["preview_scale"]
    )
    data["rgb_b1_identity_corner_deviation_output_px"] = (
        data["rgb_b1_identity_corner_deviation_preview_px"] * data["preview_scale"]
    )
    data["gate_holdout_p95_output_px"] = (
        GATE_HOLDOUT_P95_PREVIEW_PX * data["preview_scale"]
    )
    data["gate_corner_deviation_output_px"] = (
        GATE_CORNER_DEVIATION_PREVIEW_PX * data["preview_scale"]
    )
    data["gate_intra_swir_output_p95_px"] = GATE_INTRA_SWIR_OUTPUT_P95_PX
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
    ax_c.plot(x, data["rgb_b1_holdout_p95_output_px"], marker="o", color=BLUE, label="RGB–B1 held-out P95")
    ax_c.plot(
        x,
        data["rgb_b1_identity_corner_deviation_output_px"],
        marker="s",
        color=PURPLE,
        label="RGB–B1 corner deviation",
    )
    ax_c.plot(
        x,
        data["maximum_intra_swir_output_p95_px"],
        marker="^",
        color=GOLD,
        label="max intra-SWIR P95",
    )
    ax_c.axhline(
        GATE_INTRA_SWIR_OUTPUT_P95_PX,
        color=GOLD,
        linestyle=":",
        linewidth=0.9,
        label=f"intra-SWIR gate ({GATE_INTRA_SWIR_OUTPUT_P95_PX:g} px)",
    )
    gate_low = float(data["gate_corner_deviation_output_px"].min())
    gate_high = float(data["gate_corner_deviation_output_px"].max())
    ax_c.axhspan(gate_low, gate_high, color=PURPLE, alpha=0.10)
    ax_c.text(
        0.02,
        gate_low,
        "corner-deviation gate band (scene dependent)",
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=PURPLE,
    )
    ax_c.set_yscale("log")
    ax_c.set_ylim(1.2, gate_high * 1.25)
    ax_c.set_xticks(x, data["scene"])
    ax_c.set_ylabel("Residual (released output pixels, log)")
    ax_c.set_title("c  Registration diagnostics", loc="left", fontweight="bold")
    ax_c.grid(axis="y", color="#E6E9EC", linewidth=0.6, which="both")
    ax_c.legend(loc="lower left", fontsize=5.2, framealpha=0.9)

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


def load_acquisition() -> pd.DataFrame:
    acquisition = pd.read_csv(DATASET / "metadata" / "acquisition_provenance.csv")
    acquisition["scene_number"] = (
        acquisition["scene"].str.removeprefix("C").astype(int)
    )
    return acquisition.sort_values("scene_number").reset_index(drop=True)


def write_macros(data: pd.DataFrame, acquisition: pd.DataFrame) -> None:
    train = {"C1", "C3", "C4", "C5", "C6", "C7", "C10"}
    validation = {"C8"}
    test = {"C2", "C9"}
    counts = data.set_index("scene")["usable_patch_count"].to_dict()
    privacy = int(data["excluded_patch_count"].sum())
    task_validation = json.loads(TASK_VALIDATION.read_text(encoding="utf-8"))
    if not task_validation["passed"]:
        raise RuntimeError("Refusing to write manuscript macros from invalid tasks")
    if not acquisition["scene_validation_passed"].all():
        raise RuntimeError("Refusing to write manuscript macros from failed scene builds")
    coverage = acquisition["common_coverage_fraction_uncropped"]
    build_seconds = acquisition["build_seconds"]
    lines = [
        "% Generated by scripts/generate_dataset_figures.py.",
        r"\newcommand{\DatasetSceneCount}{" + str(len(data)) + "}",
        r"\newcommand{\DatasetTiffCount}{" + str(len(data) * 6) + "}",
        r"\newcommand{\DatasetUsablePatchCount}{" + str(int(data["usable_patch_count"].sum())) + "}",
        r"\newcommand{\DatasetPrivacyExcludedCount}{" + str(privacy) + "}",
        r"\newcommand{\DatasetFixedTrainCount}{" + str(sum(int(counts[x]) for x in train)) + "}",
        r"\newcommand{\DatasetFixedValidationCount}{" + str(sum(int(counts[x]) for x in validation)) + "}",
        r"\newcommand{\DatasetFixedTestCount}{" + str(sum(int(counts[x]) for x in test)) + "}",
        r"\newcommand{\DatasetTaskFileCount}{"
        + str(int(task_validation["actual_file_count"]))
        + "}",
        r"\newcommand{\DatasetRawFileCount}{"
        + str(int(acquisition["raw_file_count"].sum()))
        + "}",
        r"\newcommand{\DatasetSourcePositionCount}{"
        + str(int(acquisition["source_positions"].sum()))
        + "}",
        r"\newcommand{\DatasetRgbFrameCount}{"
        + str(int(acquisition["rgb_source_frames"].sum()))
        + "}",
        r"\newcommand{\DatasetSwirFrameCount}{"
        + str(int(acquisition["swir_source_frames"].sum()))
        + "}",
        r"\newcommand{\DatasetPoseEdgeCount}{"
        + str(int(acquisition["accepted_pose_edges"].sum()))
        + "}",
        r"\newcommand{\DatasetOutputScale}{"
        + f"{float(acquisition['output_scale'].iloc[0]):g}"
        + "}",
        r"\newcommand{\DatasetAcquisitionDateMin}{"
        + str(acquisition["acquisition_date"].min())
        + "}",
        r"\newcommand{\DatasetAcquisitionDateMax}{"
        + str(acquisition["acquisition_date"].max())
        + "}",
        r"\newcommand{\DatasetCoverageMin}{" + f"{coverage.min():.3f}" + "}",
        r"\newcommand{\DatasetCoverageMax}{" + f"{coverage.max():.3f}" + "}",
        r"\newcommand{\DatasetCoverageMean}{" + f"{coverage.mean():.3f}" + "}",
        r"\newcommand{\DatasetBuildSecondsMin}{" + f"{build_seconds.min():.1f}" + "}",
        r"\newcommand{\DatasetBuildSecondsMax}{" + f"{build_seconds.max():.1f}" + "}",
        r"\newcommand{\DatasetRedactedSceneUsablePatchCount}{"
        + str(int(counts["C10"]))
        + "}",
        r"\newcommand{\RegHoldoutMinPx}{"
        + f"{data['rgb_b1_holdout_p95_output_px'].min():.2f}"
        + "}",
        r"\newcommand{\RegHoldoutMedianPx}{"
        + f"{data['rgb_b1_holdout_p95_output_px'].median():.2f}"
        + "}",
        r"\newcommand{\RegHoldoutMeanPx}{"
        + f"{data['rgb_b1_holdout_p95_output_px'].mean():.2f}"
        + "}",
        r"\newcommand{\RegHoldoutMaxPx}{"
        + f"{data['rgb_b1_holdout_p95_output_px'].max():.2f}"
        + "}",
        r"\newcommand{\RegCornerMinPx}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].min():.2f}"
        + "}",
        r"\newcommand{\RegCornerMedianPx}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].median():.2f}"
        + "}",
        r"\newcommand{\RegCornerMeanPx}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].mean():.2f}"
        + "}",
        r"\newcommand{\RegCornerMaxPx}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].max():.2f}"
        + "}",
        r"\newcommand{\RegCornerWorstScene}{"
        + str(
            data.loc[
                data["rgb_b1_identity_corner_deviation_output_px"].idxmax(), "scene"
            ]
        )
        + "}",
        r"\newcommand{\RegCornerRunnerUpPx}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].nlargest(2).iloc[1]:.2f}"
        + "}",
        r"\newcommand{\RegCornerOutlierRatio}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].nlargest(1).iloc[0] / data['rgb_b1_identity_corner_deviation_output_px'].nlargest(2).iloc[1]:.1f}"
        + "}",
        r"\newcommand{\RegCornerOtherMeanPx}{"
        + f"{data['rgb_b1_identity_corner_deviation_output_px'].drop(data['rgb_b1_identity_corner_deviation_output_px'].idxmax()).mean():.2f}"
        + "}",
        r"\newcommand{\RegCornerGateMinPx}{"
        + f"{data['gate_corner_deviation_output_px'].min():.1f}"
        + "}",
        r"\newcommand{\RegCornerGateMaxPx}{"
        + f"{data['gate_corner_deviation_output_px'].max():.1f}"
        + "}",
        r"\newcommand{\IntraSwirMinPx}{"
        + f"{data['maximum_intra_swir_output_p95_px'].min():.2f}"
        + "}",
        r"\newcommand{\IntraSwirMeanPx}{"
        + f"{data['maximum_intra_swir_output_p95_px'].mean():.2f}"
        + "}",
        r"\newcommand{\IntraSwirMaxPx}{"
        + f"{data['maximum_intra_swir_output_p95_px'].max():.2f}"
        + "}",
        r"\newcommand{\IntraSwirGatePx}{"
        + f"{GATE_INTRA_SWIR_OUTPUT_P95_PX:g}"
        + "}",
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


def write_acquisition_tables(acquisition: pd.DataFrame) -> None:
    header_en = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-scene acquisition and mosaic-construction record, generated from "
        r"\texttt{metadata/acquisition\_provenance.csv}. Dates were recovered from the "
        r"modification times of the read-only source files and each scene maps to a single "
        r"calendar date. Pose edges are the accepted inter-position feature matches entering "
        r"the pose graph; coverage is the fraction of the uncropped canvas on which all six "
        r"modalities are valid.}",
        r"\label{tab:acquisition}",
        r"\small",
        r"\begin{tabular}{@{}llrrrrlrr@{}}",
        r"\toprule",
        r"Scene & Date & Pos. & RGB frames & SWIR frames & Pose edges & Pose model "
        r"& Canvas (px) & Coverage \\",
        r"\midrule",
    ]
    header_zh = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{由 \texttt{metadata/acquisition\_provenance.csv} 自动生成的逐场景采集与"
        r"拼接构建记录。日期由只读原始文件的修改时间恢复，每个场景对应唯一日历日期。"
        r"位姿边为进入位姿图的已接受位置间特征匹配数；覆盖率为未裁剪画布上六模态均有效的比例。}",
        r"\label{tab:acquisition-zh}",
        r"\small",
        r"\begin{tabular}{@{}llrrrrlrr@{}}",
        r"\toprule",
        r"场景 & 日期 & 位置数 & RGB帧 & SWIR帧 & 位姿边 & 位姿模型 & 画布（像素） & 覆盖率 \\",
        r"\midrule",
    ]
    rows_en, rows_zh = [], []
    for _, row in acquisition.iterrows():
        bundle = "+BA" if bool(row["bundle_adjustment_enabled"]) else ""
        pose = f"{row['pose_model']}{bundle}"
        burst = int(row["rgb_frames_per_position"])
        rgb_cell = f"{int(row['rgb_source_frames'])}" + (
            f" ({burst}/pos)" if burst > 1 else ""
        )
        values = (
            row["scene"],
            str(row["acquisition_date"]),
            int(row["source_positions"]),
            rgb_cell,
            int(row["swir_source_frames"]),
            int(row["accepted_pose_edges"]),
            pose,
            f"{int(row['canvas_width'])}$\\times${int(row['canvas_height'])}",
            f"{float(row['common_coverage_fraction_uncropped']):.3f}",
        )
        line = " & ".join(str(value) for value in values) + r" \\"
        rows_en.append(line)
        rows_zh.append(line)
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GENERATED / "scene_acquisition_en.tex").write_text(
        "\n".join(header_en + rows_en + tail) + "\n", encoding="utf-8"
    )
    (GENERATED / "scene_acquisition_zh.tex").write_text(
        "\n".join(header_zh + rows_zh + tail) + "\n", encoding="utf-8"
    )
    acquisition.to_csv(
        SOURCE_DATA / "acquisition_provenance.csv", index=False, encoding="utf-8-sig"
    )


def figure_registration_zoom() -> None:
    """Central 256x256 crop of every scene, RGB next to B1, without cherry-picking."""
    patch = 256
    records: list[dict[str, object]] = []
    fig, axes = plt.subplots(5, 4, figsize=(5.6, 7.4))
    fig.subplots_adjust(
        left=0.01, right=0.995, top=0.945, bottom=0.005, wspace=0.06, hspace=0.17
    )
    for row in range(5):
        for column in range(4):
            scene = f"C{2 * row + (1 if column < 2 else 2)}"
            number = int(scene.removeprefix("C"))
            is_rgb = column % 2 == 0
            path = (
                DATASET / scene / f"scene{number}.tif"
                if is_rgb
                else DATASET / scene / f"scene{number}_band1.tif"
            )
            with Image.open(path) as image:
                width, height = image.size
                x0, y0 = (width - patch) // 2, (height - patch) // 2
                array = np.asarray(
                    image.crop((x0, y0, x0 + patch, y0 + patch)).convert(
                        "RGB" if is_rgb else "L"
                    )
                )
            shown, low, high = display_array_from_array(array)
            ax = axes[row, column]
            ax.imshow(shown, cmap=None if is_rgb else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(
                0.03,
                0.97,
                "RGB" if is_rgb else "B1",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.5, ec="none"),
            )
            if column % 2 == 0:
                ax.text(
                    1.03,
                    1.02,
                    scene,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
            records.append(
                {
                    "scene": scene,
                    "modality": "RGB" if is_rgb else "B1",
                    "crop_x0": x0,
                    "crop_y0": y0,
                    "crop_size": patch,
                    "scene_width": width,
                    "scene_height": height,
                    "selection_rule": "central_non_overlapping_256px_crop",
                    "display_percentile_low": 1,
                    "display_percentile_high": 99,
                    "display_value_low": low,
                    "display_value_high": high,
                }
            )
    fig.suptitle(
        "Central paired crop of every scene: RGB and SWIR B1",
        fontsize=9,
        fontweight="bold",
        y=0.985,
    )
    save_figure(fig, "fig7_registration_zoom", dpi=400)
    plt.close(fig)
    pd.DataFrame(records).to_csv(
        SOURCE_DATA / "fig7_zoom_provenance.csv", index=False, encoding="utf-8-sig"
    )


def display_array_from_array(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max() or 1)
    shown = np.clip((array.astype(np.float32) - low) / (high - low), 0, 1)
    return shown, float(low), float(high)


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
        figure_registration_zoom()
    acquisition = load_acquisition()
    write_macros(data, acquisition)
    write_acquisition_tables(acquisition)
    print(f"Generated manuscript data products in {REPO}")


if __name__ == "__main__":
    main()
