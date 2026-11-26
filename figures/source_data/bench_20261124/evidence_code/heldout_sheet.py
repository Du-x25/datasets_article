"""留出场景的输出示例: 每折取留出窗口最多的两个场景, 各取场景划分模型上 PSNR 最接近该场景中位数的窗口
(在同时属于随机划分测试集的留出窗口中选取), 给出输入、随机窗口划分的输出、场景划分模型的输出与 RGB 真值。

用法: heldout_sheet.py   -> paper_out/heldout/heldout_examples_zh.pdf / .png
aggregate_results.py 用 select() 把所选窗口写入随稿源数据。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

ROOT = "D:/dh/MSWIR_RGB/bench_20261124"
FONT = "C:/Windows/Fonts/simhei.ttf"
RANDOM = ROOT + "/outputs/ev_mswir_ep100/ep100"  # first run of the same-configuration comparison, 100 epochs
FOLDS = {"A": (180, ["C6", "C7", "C8", "C9", "C10"]), "B": (220, ["C1", "C2", "C3", "C4", "C5"])}
OUT = ROOT + "/paper_out/heldout"
COLS = ["SWIR输入", "随机窗口\n划分输出", "场景划分\n模型输出", "RGB真值"]


def select():
    random_csv = RANDOM + "/per_image.csv"
    if not os.path.isfile(random_csv):
        return []
    random = pd.read_csv(random_csv).set_index("image")
    rows = []
    for fold, (epoch, scenes) in FOLDS.items():
        path = "%s/outputs/e4_scene%s/ep%d/per_image.csv" % (ROOT, fold, epoch)
        if not os.path.isfile(path):
            continue
        per = pd.read_csv(path)
        per["scene"] = per.image.str.split("_").str[0]
        per = per[per.scene.isin(scenes) & per.image.isin(random.index)]
        for sc in per.scene.value_counts().index[:2]:
            g = per[per.scene == sc]
            r = g.iloc[(g.psnr - g.psnr.median()).abs().argsort().iloc[0]]
            _, _, y, x = r.image[:-4].split("_")
            rows.append({"fold": fold, "scene": sc, "window": r.image, "top": int(y[1:]), "left": int(x[1:]), "size": 256,
                         "psnr_held": float(r.psnr), "psnr_random": float(random.loc[r.image, "psnr"])})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    rows = select()
    fig, axes = plt.subplots(2, 10, figsize=(6.75, 2.0), dpi=300,
                             gridspec_kw=dict(width_ratios=[0.62, 1, 1, 1, 1] * 2, wspace=0.04, hspace=0.06))
    for r in range(2):
        for c in (0, 5):
            axes[r, c].axis("off")
    for r, fold in enumerate(FOLDS):
        data = "%s/datasets/scene_fold%s" % (ROOT, fold)
        run = "%s/outputs/e4_scene%s/ep%d" % (ROOT, fold, FOLDS[fold][0])
        for b, row in enumerate([x for x in rows if x["fold"] == fold]):
            axes[r, 5 * b].text(1.0, 0.5, "划分 %s\n%s" % (fold, row["scene"]), transform=axes[r, 5 * b].transAxes,
                                ha="right", va="center", fontsize=6.5, linespacing=1.2)
            paths = (data + "/test_A/" + row["window"], RANDOM + "/pred/" + row["window"],
                     run + "/pred/" + row["window"], data + "/test_B/" + row["window"])
            for c, p in enumerate(paths):
                ax = axes[r, c + 1 + 5 * b]
                ax.imshow(Image.open(p).convert("RGB"), interpolation="nearest")
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(COLS[c], fontsize=6.5, pad=2, linespacing=1.0)
                if c in (1, 2):
                    ax.text(0.03, 0.03, "%.2f dB" % row["psnr_random" if c == 1 else "psnr_held"], transform=ax.transAxes,
                            fontsize=6, va="bottom", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
            print(fold, row["scene"], row["window"], "held %.2f random %.2f" % (row["psnr_held"], row["psnr_random"]), flush=True)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig("%s/heldout_examples_zh.%s" % (OUT, ext))


if __name__ == "__main__":
    main()
