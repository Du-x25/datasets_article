"""两套数据的全样本统计对照（只读源数据）。

images : 逐图统计 —— 辐射（均值、标准差、饱和）、纹理尺度（lag-1 自相关、PSD 重心、梯度/标准差）、
                   通道相关、输入-真值跨模态相关及 ±16 像素内的对齐峰
bands  : 发布全景逐场景 × 逐波段的辐射统计与波段相关（有效区域内抽样像素）

用法: data_quality_stats.py images|bands|all
输出: outputs/data_quality/
"""
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OUT = Path("D:/dh/MSWIR_RGB/bench_20261124/outputs/data_quality")
SETS = {
    "chen": Path("D:/chen/pix2pixhd-p-noshuffle/datasets/new_seg/train124_rgb"),
    "mswir_4to1": Path("D:/dh/chen_repro/datasets/mswir_random_4to1_3552"),
    "mswir_native": Path("D:/dh/chen_repro/datasets/mswir_random_4000"),
}
RELEASE = Path("D:/dh/MSWIR_RGB/data/release")
GEOMETRY = Path("D:/dh/MSWIR_RGB/data/geometry")
SCENES = ["C%d" % i for i in range(1, 11)]
R = 16
PAIRS = ((0, 1), (0, 2), (1, 2))


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def gray(x):
    return x.astype(np.float64).mean(axis=2)


def lag1(g):
    return 0.5 * (corr(g[:, :-1], g[:, 1:]) + corr(g[:-1, :], g[1:, :]))


def psd_centroid(g):
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    p = np.abs(np.fft.rfft2((g - g.mean()) * win)) ** 2
    f = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.rfftfreq(w)[None, :] ** 2)
    return float((p * f).sum() / p.sum())


def grad_ratio(g):
    s = g.std()
    if s < 1e-6:
        return float("nan")
    return float(np.sqrt(np.gradient(g, axis=0) ** 2 + np.gradient(g, axis=1) ** 2).mean() / s)


def chan_corr(x):
    return float(np.nanmean([corr(x[..., i], x[..., j]) for i, j in PAIRS]))


def align(ga, gb):
    h, w = ga.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    wa, wb = (ga - ga.mean()) * win, (gb - gb.mean()) * win
    xc = np.fft.irfft2(np.fft.rfft2(wa) * np.conj(np.fft.rfft2(wb)), s=ga.shape)
    allowed = np.zeros((h, w), bool)
    allowed[:R + 1, :R + 1] = allowed[:R + 1, w - R:] = allowed[h - R:, :R + 1] = allowed[h - R:, w - R:] = True
    iy, ix = np.unravel_index(np.argmax(np.where(allowed, xc, -np.inf)), xc.shape)
    peak = float(xc[iy, ix] / np.sqrt((wa * wa).sum() * (wb * wb).sum()))
    return corr(ga, gb), peak, int(ix if ix <= R else ix - w), int(iy if iy <= R else iy - h)


def one(task):
    name, set_name, split, root = task
    row = dict(set=set_name, split=split, image=name, scene=re.split("[-_]", name)[0])
    for tag, sub in (("a", "_A"), ("b", "_B")):
        x = np.asarray(Image.open(root / (split + sub) / name).convert("RGB"))
        g = gray(x)
        for c in range(3):
            v = x[..., c]
            row["%s_mean%d" % (tag, c + 1)] = float(v.mean())
            row["%s_std%d" % (tag, c + 1)] = float(v.std())
            row["%s_sat%d" % (tag, c + 1)] = float((v >= 255).mean() * 100)
        row[tag + "_lag1"] = lag1(g)
        row[tag + "_psd"] = psd_centroid(g)
        row[tag + "_grad"] = grad_ratio(g)
        row[tag + "_cc"] = chan_corr(x)
        if tag == "a":
            ga = g
        else:
            r0, peak, dx, dy = align(ga, g)
            row.update(ab_r0=r0, ab_peak=peak, ab_dx=dx, ab_dy=dy,
                       ab_shift=int(round((dx ** 2 + dy ** 2) ** 0.5)))
    return row


def summarize(rows, keys):
    out = {}
    for key in keys:
        sub = [r for r in rows if all(r[k] == v for k, v in key.items())]
        s = {"n": len(sub)}
        for m in sub[0]:
            if m in ("set", "split", "image", "scene"):
                continue
            v = np.array([r[m] for r in sub], dtype=np.float64)
            v = v[~np.isnan(v)]
            s[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "p5": float(np.percentile(v, 5)),
                    "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
        sh = np.array([r["ab_shift"] for r in sub])
        s["ab_peak_within_1px_pct"] = float((sh <= 1).mean() * 100)
        s["ab_peak_within_2px_pct"] = float((sh <= 2).mean() * 100)
        out["/".join(str(v) for v in key.values())] = s
    return out


def build_images():
    tasks = []
    for set_name, root in SETS.items():
        for split in ("train", "test"):
            a = {p.name for p in (root / (split + "_A")).iterdir() if p.suffix.lower() == ".png"}
            b = {p.name for p in (root / (split + "_B")).iterdir() if p.suffix.lower() == ".png"}
            tasks += [(n, set_name, split, root) for n in sorted(a & b)]
    print("images:", len(tasks), flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, tasks, chunksize=32))
    with open(OUT / "image_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ([{"set": s, "split": "all"} for s in SETS]
            + [{"set": s, "split": sp} for s in SETS for sp in ("train", "test")])
    agg = summarize(rows + [dict(r, split="all") for r in rows], keys)
    (OUT / "image_stats_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    metrics = [m for m, v in agg["chen/all"].items() if isinstance(v, dict)]
    qs = ("mean", "sd", "p5", "p50", "p95")
    with open(OUT / "image_stats_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "split", "n"] + ["%s_%s" % (m, q) for m in metrics for q in qs])
        for key, s in agg.items():
            set_name, split = key.split("/")
            w.writerow([set_name, split, s["n"]] + [s[m][q] for m in metrics for q in qs])
    return agg


def build_bands():
    rows, corr_out = [], {}
    for scene in SCENES:
        n = scene[1:]
        mask = np.array(Image.open(GEOMETRY / (scene + ".png"))) > 0
        bands = np.stack([np.array(Image.open(RELEASE / scene / ("scene%s_band%d.tif" % (n, i))))
                          for i in range(1, 6)], axis=-1)
        rgb = np.array(Image.open(RELEASE / scene / ("scene%s.tif" % n)).convert("RGB"))
        ys, xs = np.nonzero(mask)
        sel = np.arange(len(ys)) % 7 == 0
        b = bands[ys[sel], xs[sel]].astype(np.float64)
        r = rgb[ys[sel], xs[sel]].astype(np.float64)
        for name, v in [("B%d" % (i + 1), b[:, i]) for i in range(5)] + [(c, r[:, i]) for i, c in enumerate("RGB")]:
            rows.append({"scene": scene, "band": name, "pixels": int(len(v)), "mean": v.mean(), "std": v.std(),
                         "p1": np.percentile(v, 1), "p5": np.percentile(v, 5), "p50": np.percentile(v, 50),
                         "p95": np.percentile(v, 95), "p99": np.percentile(v, 99),
                         "sat255_pct": (v >= 255).mean() * 100, "sat0_pct": (v <= 0).mean() * 100})
        corr_out[scene] = {"bands": np.round(np.corrcoef(b.T), 4).tolist(),
                           "band_luma": [round(corr(b[:, i], r.mean(axis=1)), 4) for i in range(5)]}
        print("bands", scene, "valid %.3f" % mask.mean(), flush=True)
    with open(OUT / "band_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "band_corr.json").write_text(json.dumps(corr_out, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] in ("bands", "all"):
        build_bands()
    if sys.argv[1] in ("images", "all"):
        build_images()


if __name__ == "__main__":
    main()
