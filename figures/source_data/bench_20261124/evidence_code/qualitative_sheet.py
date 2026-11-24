"""定性图候选：每个场景在测试窗口中按 RGB 真值的梯度能量排序，输出候选总览（只看真值，不看预测）。

用法: qualitative_sheet.py            -> paper_out/qualitative/candidates_gt.png
      qualitative_sheet.py C1:12 C3:0 -> 按给定（场景:名次）输出对比图 paper_out/qualitative/qualitative_zh.pdf
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "D:/dh/MSWIR_RGB/bench_20261124"
DATA = "D:/dh/chen_repro/datasets/mswir_random_4to1_3552"
PRED = ROOT + "/outputs/ev_mswir_ep100/ep100/pred"
OUT = ROOT + "/paper_out/qualitative"
SCENES = ["C%d" % i for i in range(1, 11)]
FONT = "C:/Windows/Fonts/simhei.ttf"


def names(scene):
    import os
    return sorted(n for n in os.listdir(DATA + "/test_B") if n.startswith(scene + "_"))


def grad_energy(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    gy, gx = np.gradient(g)
    return float((gx * gx + gy * gy).mean())


def ranked(scene):
    return sorted(names(scene), key=lambda n: -grad_energy(DATA + "/test_B/" + n))


def sheet(k=6):
    cell, pad = 128, 4
    img = Image.new("RGB", (60 + k * (cell + pad), len(SCENES) * (cell + pad)), "white")
    font = ImageFont.truetype(FONT, 18)
    d = ImageDraw.Draw(img)
    for r, s in enumerate(SCENES):
        d.text((4, r * (cell + pad) + 50), s, fill="black", font=font)
        for c, n in enumerate(ranked(s)[:k]):
            t = Image.open(DATA + "/test_B/" + n).convert("RGB").resize((cell, cell), Image.LANCZOS)
            img.paste(t, (60 + c * (cell + pad), r * (cell + pad)))
            d.text((60 + c * (cell + pad), r * (cell + pad)), str(c), fill="yellow", font=font)
    img.save(OUT + "/candidates_gt.png")
    for s in SCENES:
        print(s, [n for n in ranked(s)[:k]], flush=True)


PICKS = {"C1": ("energy", 0), "C2": ("energy", 1), "C3": ("sorted", 5), "C4": ("sorted", 4), "C5": ("energy", 0),
         "C6": ("energy", 1), "C7": ("sorted", 45), "C8": ("energy", 1), "C9": ("energy", 0), "C10": ("energy", 0)}


def chosen(scene):
    kind, k = PICKS[scene]
    return (ranked(scene) if kind == "energy" else names(scene))[k]


def panel(scenes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    rows = [(s, chosen(s)) for s in scenes]
    half = (len(rows) + 1) // 2
    fig, axes = plt.subplots(half, 7, figsize=(6.75, 1.12 * half + 0.3), dpi=260,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.14, 1, 1, 1], wspace=0.04, hspace=0.04))
    for ax in axes[:, 3]:
        ax.axis("off")
    cols = ["SWIR输入（B1--B3）", "TFD-GAN输出", "RGB真值"]
    for i, (s, n) in enumerate(rows):
        block, r = divmod(i, half)
        _, _, y, x = n[:-4].split("_")
        for c, p in enumerate((DATA + "/test_A/" + n, PRED + "/" + n, DATA + "/test_B/" + n)):
            ax = axes[r, 3 * block + c + block]
            ax.imshow(Image.open(p).convert("RGB"), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, pad=2)
            if c == 0:
                ax.set_ylabel("%s\n%s, %s" % (s, y[1:], x[1:]), fontsize=7, linespacing=1.1, labelpad=2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(OUT + "/qualitative_zh." + ext)
    for s, n in rows:
        print(s, n, flush=True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1:
        panel(sys.argv[1:])
    else:
        sheet()