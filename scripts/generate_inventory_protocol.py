"""Regenerate the English and Chinese inventory/fold figures from the current protocol.

Contract: ten independent physical sites, 140 acquisition positions, five
site-disjoint folds; panel a is a count bar chart and panel b a role matrix.
Python/Matplotlib only; editable PDF/SVG and a PNG preview, no inference.
"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from plot_fonts import select_plot_font

ROOT = Path(__file__).resolve().parents[1]


def render(protocol, inventory, sites, counts, matrix, lang):
    roles_zh = ("训", "验", "测")
    roles_en = ("Tr", "V", "Te")
    if lang == "zh":
        family, _ = select_plot_font()
        plt.rcParams.update({
            "font.family": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        })
        labels = roles_zh
        suptitle = "10个不同物理地点 · 140个采集视野"
        subtitle = "仅展示采集规模与划分协议，不含模型结果"
        ylabel = "采集视野数"
        title_a = "a  逐地点采集规模"
        fold_labels = [f"第{fold['fold']}折" for fold in protocol["folds"]]
        title_b = "b  地点隔离五折协议"
        legend_line = "训：训练（6地点）    验：验证（2地点）    测：测试（2地点）"
        seed_line = f"地点隔离五折基准使用训练种子：{', '.join(map(str, protocol['seeds']))}"
        out_dir = ROOT / "figures/zh"
        stem = "v3_inventory_folds_zh"
    else:
        plt.rcParams.update({
            "font.family": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        })
        labels = roles_en
        suptitle = "10 distinct physical sites · 140 acquisition positions"
        subtitle = "Inventory and site-disjoint protocol only — no model outcomes"
        ylabel = "Acquisition positions"
        title_a = "a  Site inventory"
        fold_labels = [f"Fold {fold['fold']}" for fold in protocol["folds"]]
        title_b = "b  Site-disjoint five-fold protocol"
        legend_line = "Tr = train (6 sites)    V = validation (2 sites)    Te = test (2 sites)"
        seed_line = f"Site-disjoint five-fold benchmark seed: {', '.join(map(str, protocol['seeds']))}"
        out_dir = ROOT / "figures"
        stem = "v3_inventory_folds"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, grid) = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(left=.06, right=.99, top=.78, bottom=.22, wspace=.20)
    fig.suptitle(suptitle, fontsize=12, y=.98)
    fig.text(.5, .875, subtitle, ha="center", color="#59636E", fontsize=9)
    ax.bar(sites, counts, color="#397D86", width=.75)
    ax.set_ylim(0, 34)
    ax.set_ylabel(ylabel)
    ax.set_title(title_a, loc="left", fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EAEC", linewidth=.5)
    for i, count in enumerate(counts):
        ax.text(i, count + .7, str(count), ha="center", fontsize=8)
    grid.imshow(matrix, cmap=ListedColormap(["#DFEBED", "#ECE3CF", "#397D86"]), vmin=0, vmax=2, aspect="auto")
    grid.set_xticks(range(10), sites)
    grid.set_yticks(range(5), fold_labels)
    grid.set_title(title_b, loc="left", fontweight="bold")
    grid.tick_params(length=0)
    for r in range(5):
        for c in range(10):
            grid.text(c, r, labels[matrix[r, c]], ha="center", va="center", fontsize=8,
                      color="white" if matrix[r, c] == 2 else "#24383C")
    for spine in grid.spines.values():
        spine.set_visible(False)
    fig.text(.5, .135, legend_line, ha="center", fontsize=9)
    fig.text(.5, .045, seed_line, ha="center", fontweight="bold", fontsize=9)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    protocol = json.loads((ROOT / "docs/current_protocol.json").read_text(encoding="utf-8"))
    with (ROOT / "figures/source_data/acquisition_provenance.csv").open(encoding="utf-8-sig") as f:
        inventory = {row["scene"]: int(row["source_positions"]) for row in csv.DictReader(f)}
    sites = [f"C{i}" for i in range(1, 11)]
    counts = [inventory[site] for site in sites]
    assert sum(counts) == 140 and len(protocol["folds"]) == 5
    roles = ("train", "validation", "test")
    matrix = np.empty((5, 10), dtype=int)
    rows = []
    for r, fold in enumerate(protocol["folds"]):
        members = [site for role in roles for site in fold[role]]
        assert sorted(members) == sorted(sites)
        for c, site in enumerate(sites):
            role = next(role for role in roles if site in fold[role])
            matrix[r, c] = roles.index(role)
            rows.append(dict(site=site, acquisition_positions=inventory[site], fold=fold["fold"], role=role))
    render(protocol, inventory, sites, counts, matrix, "en")
    render(protocol, inventory, sites, counts, matrix, "zh")
    with (ROOT / "figures/source_data/v3_inventory_folds.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    zh_csv = ROOT / "figures/zh/source_data/v3_inventory_folds_zh.csv"
    if zh_csv.parent.exists():
        with zh_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Inventory/fold figures regenerated (en + zh) from current protocol.")


if __name__ == "__main__":
    main()
