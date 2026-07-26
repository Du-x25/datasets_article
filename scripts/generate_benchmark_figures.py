"""Generate manuscript benchmark figures only from complete three-seed results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
SOURCE_DATA = FIGURES / "source_data"

BLUE = "#2F6B8A"
GOLD = "#D8A03D"
PURPLE = "#7A6FA3"
SLATE = "#59636E"
GRID = "#E6E9EC"
INPUT_ORDER = ["B1", "B2", "B3", "B4", "B5", "B124", "5B"]
MODEL_ORDER = ["unet", "resunet", "resnet"]
MODEL_LABELS = {"unet": "U-Net", "resunet": "ResUNet", "resnet": "ResNet"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 220}),
    ):
        fig.savefig(FIGURES / f"{stem}{suffix}", bbox_inches="tight", **kwargs)


def require_complete(frame: pd.DataFrame, expected_rows: int, stage: str) -> None:
    if len(frame) != expected_rows:
        raise SystemExit(
            f"{stage} summary is incomplete: {len(frame)} rows != {expected_rows}"
        )
    if not (frame["N_Seeds"] == 3).all():
        raise SystemExit(f"{stage} summary contains a group without three seeds")


def metric_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    colors: list[str],
) -> None:
    x = np.arange(len(frame))
    mean = frame[f"{metric}_Mean"].to_numpy(float)
    sd = frame[f"{metric}_SD"].to_numpy(float)
    ax.bar(x, mean, color=colors, width=0.72, zorder=2)
    ax.errorbar(
        x,
        mean,
        yerr=sd,
        fmt="none",
        ecolor=SLATE,
        elinewidth=0.8,
        capsize=2.5,
        zorder=3,
    )
    ax.set_xticks(x, frame["Input"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)


def figure_input(summary: pd.DataFrame) -> None:
    frame = summary[summary["Stage"] == "input"].copy()
    require_complete(frame, 7, "input")
    frame["Input"] = pd.Categorical(frame["Input"], INPUT_ORDER, ordered=True)
    frame = frame.sort_values("Input").reset_index(drop=True)
    frame["Input"] = frame["Input"].astype("string")
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        SOURCE_DATA / "fig4_input_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )
    colors = [
        PURPLE if name == "5B" else GOLD if name == "B124" else BLUE
        for name in frame["Input"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    metric_panel(
        axes[0, 0],
        frame,
        "validation_psnr",
        "a  Validation PSNR used for selection",
        "PSNR (dB)",
        colors,
    )
    metric_panel(
        axes[0, 1],
        frame,
        "PSNR",
        "b  Held-out test PSNR",
        "PSNR (dB)",
        colors,
    )
    metric_panel(
        axes[1, 0],
        frame,
        "SSIM",
        "c  Held-out test SSIM",
        "SSIM",
        colors,
    )
    metric_panel(
        axes[1, 1],
        frame,
        "CIEDE2000",
        "d  Held-out test colour difference",
        "CIEDE2000 (lower is better)",
        colors,
    )
    fig.suptitle(
        "SWIR input comparison with a fixed U-Net",
        fontsize=10,
        fontweight="bold",
    )
    save_figure(fig, "fig4_input_benchmark")
    plt.close(fig)


def figure_model(summary: pd.DataFrame) -> None:
    input_rows = summary[
        (summary["Stage"] == "input")
        & (summary["Input"].isin(["B1", "B124", "5B"]))
    ].copy()
    input_rows["Stage"] = "model"
    model_rows = summary[summary["Stage"] == "model"].copy()
    frame = pd.concat([input_rows, model_rows], ignore_index=True)
    require_complete(frame, 9, "model comparison")
    frame["Model"] = pd.Categorical(frame["Model"], MODEL_ORDER, ordered=True)
    frame["Input"] = pd.Categorical(
        frame["Input"], ["B1", "B124", "5B"], ordered=True
    )
    frame = frame.sort_values(["Input", "Model"]).reset_index(drop=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        SOURCE_DATA / "fig5_model_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    metrics = (
        ("validation_psnr", "a  Validation PSNR", "PSNR (dB)"),
        ("PSNR", "b  Held-out test PSNR", "PSNR (dB)"),
        ("SSIM", "c  Held-out test SSIM", "SSIM"),
        ("CIEDE2000", "d  Held-out colour difference", "CIEDE2000"),
    )
    width = 0.23
    x = np.arange(3)
    model_colors = (BLUE, GOLD, PURPLE)
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        for offset, (model, color) in enumerate(zip(MODEL_ORDER, model_colors)):
            values = frame[frame["Model"] == model].sort_values("Input")
            means = values[f"{metric}_Mean"].to_numpy(float)
            sd = values[f"{metric}_SD"].to_numpy(float)
            position = x + (offset - 1) * width
            ax.bar(
                position,
                means,
                width=width,
                color=color,
                label=MODEL_LABELS[model],
                zorder=2,
            )
            ax.errorbar(
                position,
                means,
                yerr=sd,
                fmt="none",
                ecolor=SLATE,
                elinewidth=0.7,
                capsize=2,
                zorder=3,
            )
        ax.set_xticks(x, ["B1", "B124", "5B"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axes[0, 0].legend(ncol=3, loc="best")
    fig.suptitle(
        "Model-family comparison across representative SWIR inputs",
        fontsize=10,
        fontweight="bold",
    )
    save_figure(fig, "fig5_model_benchmark")
    plt.close(fig)


def figure_scene(summary: pd.DataFrame) -> None:
    frame = summary[summary["Stage"] == "scene"].copy()
    require_complete(frame, 10, "scene")
    frame["scene_number"] = (
        frame["Test_Scene"].str.removeprefix("C").astype(int)
    )
    frame = frame.sort_values("scene_number").reset_index(drop=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        SOURCE_DATA / "fig6_scene_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )
    x = np.arange(len(frame))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.15),
        sharex=True,
        constrained_layout=True,
    )
    for ax, metric, title, ylabel, color in (
        (
            axes[0],
            "PSNR",
            "a  Leave-one-scene-out PSNR",
            "PSNR (dB)",
            BLUE,
        ),
        (
            axes[1],
            "SSIM",
            "b  Leave-one-scene-out SSIM",
            "SSIM",
            PURPLE,
        ),
    ):
        mean = frame[f"{metric}_Mean"].to_numpy(float)
        sd = frame[f"{metric}_SD"].to_numpy(float)
        ax.errorbar(
            x,
            mean,
            yerr=sd,
            color=color,
            marker="o",
            linewidth=1.4,
            capsize=3,
        )
        ax.axhline(mean.mean(), color=SLATE, linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    axes[1].set_xticks(x, frame["Test_Scene"])
    axes[1].set_xlabel("Held-out source scene")
    fig.suptitle(
        "Cross-scene variability under leave-one-scene-out evaluation",
        fontsize=10,
        fontweight="bold",
    )
    save_figure(fig, "fig6_scene_benchmark")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("input", "model", "scene"),
        default=("input",),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.benchmark_dir.resolve() / "summary_metrics.csv"
    summary = pd.read_csv(summary_path)
    for stage in args.stages:
        if stage == "input":
            figure_input(summary)
        elif stage == "model":
            figure_model(summary)
        else:
            figure_scene(summary)
    print(f"Generated benchmark figures for: {', '.join(args.stages)}")


if __name__ == "__main__":
    main()
