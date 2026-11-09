"""Regenerate the inventory/fold figure from the current protocol (no outcomes).

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

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "docs/current_protocol.json").read_text(encoding="utf-8"))
    with (ROOT / "figures/source_data/acquisition_provenance.csv").open(encoding="utf-8-sig") as f:
        inventory = {row["scene"]: int(row["source_positions"]) for row in csv.DictReader(f)}
    sites = [f"C{i}" for i in range(1, 11)]
    counts = [inventory[site] for site in sites]
    assert sum(counts) == 140 and len(protocol["folds"]) == 5
    roles = ("train", "validation", "test")
    labels = ("Tr", "V", "Te")
    matrix = np.empty((5, 10), dtype=int)
    rows = []
    for r, fold in enumerate(protocol["folds"]):
        members = [site for role in roles for site in fold[role]]
        assert sorted(members) == sorted(sites)
        for c, site in enumerate(sites):
            role = next(role for role in roles if site in fold[role])
            matrix[r, c] = roles.index(role)
            rows.append(dict(site=site, acquisition_positions=inventory[site], fold=fold["fold"], role=role))
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (ax, grid) = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(left=.06, right=.99, top=.78, bottom=.22, wspace=.20)
    fig.suptitle("10 distinct physical sites · 140 acquisition positions", fontsize=12, y=.98)
    fig.text(.5, .875, "Inventory and protocol only — no model outcomes", ha="center", color="#59636E", fontsize=9)
    ax.bar(sites, counts, color="#397D86", width=.75)
    ax.set_ylim(0, 34)
    ax.set_ylabel("Acquisition positions")
    ax.set_title("a  Site inventory", loc="left", fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EAEC", linewidth=.5)
    for i, count in enumerate(counts):
        ax.text(i, count + .7, str(count), ha="center", fontsize=8)
    grid.imshow(matrix, cmap=ListedColormap(["#DFEBED", "#ECE3CF", "#397D86"]), vmin=0, vmax=2, aspect="auto")
    grid.set_xticks(range(10), sites)
    grid.set_yticks(range(5), [f"Fold {fold['fold']}" for fold in protocol["folds"]])
    grid.set_title("b  Site-disjoint five-fold protocol", loc="left", fontweight="bold")
    grid.tick_params(length=0)
    for r in range(5):
        for c in range(10):
            grid.text(c, r, labels[matrix[r, c]], ha="center", va="center", fontsize=8,
                      color="white" if matrix[r, c] == 2 else "#24383C")
    for spine in grid.spines.values():
        spine.set_visible(False)
    fig.text(.5, .135, "Tr = train (6 sites)    V = validation (2 sites)    Te = test (2 sites)", ha="center", fontsize=9)
    current_seed = ", ".join(map(str, protocol["seeds"]))
    fig.text(.5, .045, f"Reported training seed: {current_seed}", ha="center", fontweight="bold", fontsize=9)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"figures/v3_inventory_folds.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    with (ROOT / "figures/source_data/v3_inventory_folds.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Inventory/fold figure regenerated from current protocol.")


if __name__ == "__main__":
    main()
