"""Draw the same-configuration protocol ladder from the saved activation records."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures/source_data/reproduction_20261112/summary.json"
OUT = ROOT / "figures/zh/results_20261112"

font_manager.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = d["ladder"]
    ref = d["reference"]

    labels = [r["label"] for r in rows]
    values = [r["psnr"] for r in rows]
    colors = ["#b8b8b8", "#9fb3c8", "#7f9db9", "#3f8f8a", "#2f7370", "#1f5553"]

    fig, ax = plt.subplots(figsize=(7.0, 3.1), layout="constrained")
    y = range(len(rows))
    bars = ax.barh(list(y), values, color=colors, height=0.62)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 34)
    ax.set_xlabel("测试 PSNR / dB")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)

    ax.axvspan(ref["our_repro_psnr"][0], ref["chen_paper_psnr"], color="#e08a3c", alpha=0.18, zorder=0)
    ax.set_ylim(len(rows) - 0.5, -1.0)
    ax.text((ref["our_repro_psnr"][0] + ref["chen_paper_psnr"]) / 2, -0.92,
            f"陈论文 {ref['chen_paper_psnr']:.2f} / 复现 {ref['our_repro_psnr'][0]:.2f}–{ref['our_repro_psnr'][1]:.2f}",
            color="#a05a1e", fontsize=7, ha="center", va="top")

    for bar, value in zip(bars, values):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", fontsize=7.5)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "protocol_ladder_zh.pdf")
    print("wrote", OUT / "protocol_ladder_zh.pdf")


if __name__ == "__main__":
    main()
