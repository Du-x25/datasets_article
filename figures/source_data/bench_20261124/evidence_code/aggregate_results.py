"""收集 bench_20261124 的全部结果并渲染论文表格、数值宏与图。

数字来源: outputs/<任务>/ep<轮次>/summary.json 与 per_image.csv（统一 8 位 PNG 口径），以及 chen_repro
中已有的同口径结果。本脚本只负责收集与配对统计；表格、宏与图由 build_bench_20261124.py 根据
paper_out/results_bench_20261124.json 生成，论文仓库内用同一模块由随稿源数据重建。
缺失的运行在表格与宏中以 -- 占位，可随时重跑。

用法: aggregate_results.py [--no-figures]
"""
import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import build_bench_20261124 as B

ROOT = Path("D:/dh/MSWIR_RGB/bench_20261124")
REPRO = Path("D:/dh/chen_repro")
OUT = ROOT / "paper_out"
GEN = OUT / "generated"
SRC = OUT / "source_data"
FIG = OUT / "figures"
METRICS = B.METRICS
SINGLES, TRIPLES = B.SINGLES, B.TRIPLES
ITERS, TRAIN, MATCHED, EPOCHS = B.ITERS, B.TRAIN, B.MATCHED, B.EPOCHS
PEAK_MIN = 0.2


def scene_of(names):
    s = pd.Series(names)
    return s.str.extract(r"^(C\d+)_", expand=False).fillna(s.str.split("-").str[0]).values


class Run:
    def __init__(self, path):
        s = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        self.path = str(path).replace("\\", "/")
        self.n = s["n"]
        self.mean = {m: s[m]["mean"] for m in METRICS}
        self.per = pd.read_csv(path / "per_image.csv")
        self.per["scene"] = scene_of(self.per["image"].tolist())
        self.scene_mean = self.per.groupby("scene")["psnr"].mean()
        self.scene_means = self.per.groupby("scene")[list(METRICS)].mean()
        self.scene_n = self.per.groupby("scene").size()


def collect():
    r = {}

    def add(key, path):
        if (path / "summary.json").is_file():
            r[key] = Run(path)

    add("mswir_s1_ep100", REPRO / "outputs/mswir_random_4to1_png/4to1_ep100")
    add("mswir_s1_ep200", ROOT / "outputs/ev_mswir_ep200/ep200")
    add("mswir_s1_ep500", REPRO / "outputs/mswir_random_4to1_png/4to1_ep500")
    add("chen_s1_ep100", REPRO / "outputs/png/b4_ep100")
    add("chen_s1_ep200", REPRO / "outputs/png/b4_ep200")
    add("chen_c1_ep500", REPRO / "outputs/png/b4_500_a")
    add("chen_c2_ep500", REPRO / "outputs/png/b4_500_b")
    add("chen_c1_ep490", REPRO / "outputs/png/b4_500_a_ep490")
    add("chen_c2_ep490", REPRO / "outputs/png/b4_500_b_ep490")
    for s in ("s2", "s3"):
        for data in ("mswir", "chen"):
            for e in (100, 200):
                add("%s_%s_ep%d" % (data, s, e), ROOT / "outputs" / ("e1_%s_%s" % (data, s)) / ("ep%d" % e))
    for e in (100, 200):
        add("native_s1_ep%d" % e, ROOT / "outputs" / ("ev_native_ep%d" % e) / ("ep%d" % e))
    for v in SINGLES + TRIPLES:
        if v != "B123":
            add("e2_%s_ep100" % v, ROOT / "outputs" / ("e2_%s" % v) / "ep100")
    for name, eps in EPOCHS.items():
        for e in eps:
            add("e3_%s_ep%d" % (name, e), ROOT / "outputs" / ("e3_%s" % name) / ("ep%d" % e))
    add("e4_sceneA_ep180", ROOT / "outputs/e4_sceneA/ep180")
    add("e4_sceneB_ep220", ROOT / "outputs/e4_sceneB/ep220")
    return r


def stat(runs):
    """若干次训练的均值、样本标准差与共同测试窗口数。"""
    if not runs:
        return None
    a = np.array([[x.mean[m] for m in METRICS] for x in runs])
    ns = sorted({x.n for x in runs})
    return {"mean": a.mean(0), "sd": a.std(0, ddof=1) if len(runs) > 1 else None, "k": len(runs),
            "n": ns[0] if len(ns) == 1 else None}


def boot(diffs, n_boot=20000):
    """按场景整群抽样的均值 95% 置信区间；diffs: 场景 -> 逐窗口差值。"""
    keys = sorted(diffs)
    sums = np.array([diffs[k].sum() for k in keys])
    cnts = np.array([len(diffs[k]) for k in keys])
    idx = np.random.RandomState(20261124).randint(0, len(keys), size=(n_boot, len(keys)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(sums.sum() / cnts.sum()), "lo": float(lo), "hi": float(hi),
            "scenes": len(keys), "scenes_pos": int(sum(diffs[k].mean() > 0 for k in keys))}


def paired(run, refs, metric="psnr"):
    """run 相对若干参照训练（逐窗口先平均）的配对差。"""
    ref = pd.concat([x.per.set_index("image")[metric] for x in refs], axis=1).mean(1)
    both = pd.concat([run.per.set_index("image")[metric], ref], axis=1, keys=["a", "b"]).dropna()
    d = both["a"] - both["b"]
    return boot({s: g.values for s, g in d.groupby(scene_of(both.index.tolist()))})


def seed_scenes(runs, keys):
    """逐场景指标在若干次训练间的均值与标准差。"""
    keys = [k for k in keys if k in runs]
    out = {}
    for sc in sorted(set().union(*[set(runs[k].scene_mean.index) for k in keys])) if keys else []:
        a = np.array([[runs[k].scene_means.loc[sc, m] for m in METRICS] for k in keys])
        out[sc] = {"n": int(runs[keys[0]].scene_n[sc]), "mean": a.mean(0),
                   "sd": a.std(0, ddof=1) if len(keys) > 1 else None, "k": len(keys)}
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    return None if isinstance(o, float) and not np.isfinite(o) else o


def build(runs):
    d = {"runs": {k: {"n": v.n, **v.mean} for k, v in runs.items()}, "scene_seeds": {}}
    for data in ("mswir", "chen"):
        for e in (100, 200):
            keys = ["%s_%s_ep%d" % (data, s, e) for s in B.SEEDS if "%s_%s_ep%d" % (data, s, e) in runs]
            keys.sort(key=lambda k: B.SEEDS.index(k.split("_")[1]))
            d["%s_ep%d" % (data, e)] = stat([runs[k] for k in keys])
            d["%s_ep%d_runs" % (data, e)] = keys
            if e == 100:
                d["scene_seeds"][data] = seed_scenes(runs, keys)
    d["scene_rank_rho"] = {}  # lowest Spearman correlation of per-scene PSNR between any two runs
    for data in ("mswir", "chen"):
        keys = d["%s_ep100_runs" % data]
        rho = [runs[a].scene_means["psnr"].corr(runs[b].scene_means["psnr"], method="spearman")
               for a, b in itertools.combinations(keys, 2)]
        d["scene_rank_rho"][data] = float(min(rho)) if rho else None
    refs = [runs[k] for k in d["mswir_ep100_runs"]]
    d["band_ref"] = d["mswir_ep100"]
    scenes = sorted(set().union(*[set(x.scene_mean.index) for x in refs])) if refs else []
    d["band_ref_scene_mean"] = {sc: float(np.mean([x.scene_mean[sc] for x in refs])) for sc in scenes}
    d["band"] = {}
    for v in SINGLES + TRIPLES:
        k = "e2_%s_ep100" % v
        if k in runs:
            d["band"][v] = {"run": stat([runs[k]]), "paired": paired(runs[k], refs) if refs else None,
                            "scene_mean": runs[k].scene_mean.to_dict()}
    for name, pool in (("single", SINGLES), ("triple", [t for t in TRIPLES if t != "B123"])):
        done = {v: d["band"][v]["run"]["mean"][0] for v in pool if v in d["band"]}
        if done:
            best, worst = max(done, key=done.get), min(done, key=done.get)
            d["best_" + name], d["worst_" + name] = [best, float(done[best])], [worst, float(done[worst])]
    d["budget"] = {k: stat([v]) for k, v in runs.items() if k.startswith("e3_")}
    d["native"] = {"%d" % e: stat([runs["native_s1_ep%d" % e]]) if "native_s1_ep%d" % e in runs else None for e in (100, 200)}
    d["mswir_ep500"] = stat([runs["mswir_s1_ep500"]]) if "mswir_s1_ep500" in runs else None
    d["chen_ep500"] = stat([runs[k] for k in ("chen_c1_ep500", "chen_c2_ep500") if k in runs])
    late = [abs(runs[a].mean["psnr"] - runs[b].mean["psnr"])
            for a, b in (("chen_c1_ep500", "chen_c1_ep490"), ("chen_c2_ep500", "chen_c2_ep490")) if a in runs and b in runs]
    d["chen_late_swing"] = max(late) if late else None
    d["gap"] = None
    if all(d["%s_ep%d" % (x, e)] for x in ("mswir", "chen") for e in (100, 200)):
        d["gap"] = {"%d" % e: d["mswir_ep%d" % e]["mean"][0] - d["chen_ep%d" % e]["mean"][0] for e in (100, 200)}
    d["scene"] = scene_block(runs, d)
    d["protocol"] = {
        "code": "TFD-GAN, pix2pixhd-p-noshuffle; 29 source files sha256-identical to the reference copy",
        "train_args": "--label_nc 0 --no_instance --loadSize 256 --batchSize 4; Adam lr 2e-4, betas (0.5, 0.999); "
                      "constant lr for niter epochs",
        "reference_budget_updates": ITERS["full"] * 100,
        "iters_per_epoch": ITERS, "train_windows": TRAIN, "matched_stop_epoch": MATCHED,
        "folds": {"A": {"train": "C1-C5", "test": "C6-C10", "run": "e4_sceneA_ep180", "train_windows": 1558, "test_windows": 1618},
                  "B": {"train": "C6-C10", "test": "C1-C5", "run": "e4_sceneB_ep220", "train_windows": 1284, "test_windows": 1934}},
        "evaluation": "8-bit PNG output; PSNR; skimage SSIM (channel_axis=2, data_range=1); LPIPS-Alex; "
                      "mean over test windows",
        "seeds": "the reference code sets no random seed; each run is an independent initialisation and data order",
        "bootstrap": "scene-cluster bootstrap of per-window differences vs the mean of the reference runs, "
                     "20000 resamples, seed 20261124",
    }
    return d


def scene_block(runs, d):
    out = {"foldA": stat([runs["e4_sceneA_ep180"]]) if "e4_sceneA_ep180" in runs else None,
           "foldB": stat([runs["e4_sceneB_ep220"]]) if "e4_sceneB_ep220" in runs else None, "per_scene": {}}
    rand, held = {}, {}
    for k in d["mswir_ep100_runs"]:
        for sc, v in runs[k].scene_mean.items():
            rand.setdefault(sc, []).append(v)
    for key, fold in (("e4_sceneA_ep180", "A"), ("e4_sceneB_ep220", "B")):
        if key in runs:
            held.update({sc: (float(v), fold) for sc, v in runs[key].scene_mean.items()})
    scenes = sorted(set(rand) | set(held))
    out["per_scene"] = {sc: {"random": float(np.mean(rand[sc])) if sc in rand else None,
                             "held": held[sc][0] if sc in held else None,
                             "fold": held[sc][1] if sc in held else None} for sc in scenes}
    both = [sc for sc in scenes if sc in rand and sc in held]
    if both:
        out["scenes"] = len(both)
        out["random_scene_equal"] = float(np.mean([out["per_scene"][sc]["random"] for sc in both]))
        out["held_scene_equal"] = float(np.mean([out["per_scene"][sc]["held"] for sc in both]))
        out["drop_scene_equal"] = out["random_scene_equal"] - out["held_scene_equal"]
        out["scenes_lower"] = int(sum(out["per_scene"][sc]["held"] < out["per_scene"][sc]["random"] for sc in both))
    if d["mswir_ep100"] and out["foldA"] and out["foldB"]:
        out["drop_fold_mean"] = d["mswir_ep100"]["mean"][0] - (out["foldA"]["mean"][0] + out["foldB"]["mean"][0]) / 2
    return out


def load_quality():
    raw = json.loads((ROOT / "outputs/data_quality/image_stats_summary.json").read_text(encoding="utf-8"))
    per = pd.read_csv(ROOT / "outputs/data_quality/image_stats.csv")
    per["a_std"] = per[["a_std1", "a_std2", "a_std3"]].mean(axis=1)
    out = {}
    for key, s in raw.items():
        name, split = key.split("/")
        if split != "all":
            continue
        row = {**{"%s_%s" % (m, k): v[k] for m, v in s.items() if isinstance(v, dict) for k in ("mean", "p5", "p50", "p95")},
               **{k: v for k, v in s.items() if not isinstance(v, dict)}}
        g = per[per["set"] == name]
        strong = g[g["ab_peak"] >= PEAK_MIN]
        weak = g["ab_peak"] < PEAK_MIN
        row.update({"ab_peak_weak_pct": float(weak.mean() * 100),
                    "ab_within1px_strong_pct": float((strong["ab_shift"] <= 1).mean() * 100),
                    "ab_within4px_strong_pct": float((strong["ab_shift"] <= 4).mean() * 100),
                    "ab_within4px_pct": float((g["ab_shift"] <= 4).mean() * 100),
                    "weak_std_mean": float(g.loc[weak, "a_std"].mean()),
                    "strong_std_mean": float(g.loc[~weak, "a_std"].mean()),
                    "ab_ratio_pct": float(100.0 * g["ab_r0"].mean() / g["ab_peak"].mean())})
        out[name] = {"all": row}
    return out


def quality_scene():
    stats = pd.read_csv(ROOT / "outputs/data_quality/band_stats.csv")
    per = pd.read_csv(ROOT / "outputs/data_quality/image_stats.csv")
    per = per[per["set"] == "mswir_4to1"]
    out = {}
    for sc in B.SCENES:
        s = stats[stats.scene == sc].set_index("band")
        g = per[per.scene == sc]
        strong = g[g["ab_peak"] >= PEAK_MIN]
        out[sc] = {"b1_mean": float(s.loc["B1", "mean"]), "b1_sat": float(s.loc["B1", "sat255_pct"]),
                   "b3_std": float(s.loc["B3", "std"]), "b5_std": float(s.loc["B5", "std"]),
                   "rgb_sat": {c: float(s.loc[c, "sat255_pct"]) for c in "RGB"},
                   "rgb0": {c: float(s.loc[c, "sat0_pct"]) for c in "RGB"},
                   "rgb_luma": float(np.mean([s.loc[c, "mean"] for c in "RGB"])),
                   "windows": int(len(g)), "ab_r0_med": float(g["ab_r0"].median()),
                   "weak": float((g["ab_peak"] < PEAK_MIN).mean() * 100),
                   "within1": float((strong["ab_shift"] <= 1).mean() * 100)}
    return out


def quality_bands():
    stats = pd.read_csv(ROOT / "outputs/data_quality/band_stats.csv")
    corr = json.loads((ROOT / "outputs/data_quality/band_corr.json").read_text(encoding="utf-8"))
    out = {}
    for b in SINGLES:
        s = stats[stats.band == b]
        w = s.pixels.values / s.pixels.values.sum()
        i = int(b[1]) - 1
        others = [c["bands"][i][j] for c in corr.values() for j in range(5) if j != i]
        luma = [c["band_luma"][i] for c in corr.values()]
        out[b] = {"mean_min": float(s["mean"].min()), "mean_max": float(s["mean"].max()),
                  "std_min": float(s["std"].min()), "std_max": float(s["std"].max()),
                  "sat": float((s.sat255_pct.values * w).sum()), "sat_max": float(s.sat255_pct.max()),
                  "sat0_max": float(s.sat0_pct.max()), "corr_min": min(others), "corr_max": max(others),
                  "luma_min": min(luma), "luma_max": max(luma),
                  "luma_by_scene": {scene: c["band_luma"][i] for scene, c in corr.items()}}
    pairs = {}
    for i, j in itertools.combinations(range(5), 2):
        v = [c["bands"][i][j] for c in corr.values()]
        pairs["B%d%d" % (i + 1, j + 1)] = {"min": min(v), "median": float(np.median(v)), "max": max(v)}
    within = [pairs[k] for k in ("B12", "B14", "B24", "B35")]
    cross = [v for k, v in pairs.items() if k not in ("B12", "B14", "B24", "B35")]
    groups = {"pairs": pairs,
              "within": [min(p["min"] for p in within), float(np.median([p["median"] for p in within])), max(p["max"] for p in within)],
              "cross": [min(p["min"] for p in cross), float(np.median([p["median"] for p in cross])), max(p["max"] for p in cross)]}
    return out, groups


def quality_hist():
    return json.loads((ROOT / "outputs/data_quality/band_hist.json").read_text(encoding="utf-8"))


def qualitative_block(runs):
    """图板所用窗口（仅按真值选定）及其在参照训练中的逐窗口PSNR。"""
    if "mswir_s1_ep100" not in runs:
        return None
    import qualitative_sheet as Q
    per = runs["mswir_s1_ep100"].per.set_index("image")
    rows = []
    for sc in Q.SCENES:
        name = Q.chosen(sc)
        _, _, y, x = name[:-4].split("_")
        rows.append({"scene": sc, "window": name, "top": int(y[1:]), "left": int(x[1:]), "size": 256,
                     "psnr_s1_ep100": float(per.loc[name, "psnr"])})
    return {"windows": rows, "psnr_min": min(r["psnr_s1_ep100"] for r in rows),
            "psnr_max": max(r["psnr_s1_ep100"] for r in rows)}


def source_data(d, runs):
    pd.DataFrame([{"run": k, **v} for k, v in d["runs"].items()]).to_csv(SRC / "runs_summary.csv", index=False, encoding="utf-8-sig")
    rows = [{"input": v, "psnr": b["run"]["mean"][0], "ssim": b["run"]["mean"][1], "lpips": b["run"]["mean"][2],
             "d_psnr_vs_b123": b["paired"]["mean"] if b["paired"] else None,
             "ci_lo": b["paired"]["lo"] if b["paired"] else None, "ci_hi": b["paired"]["hi"] if b["paired"] else None,
             "scenes_lower_than_b123": b["paired"]["scenes"] - b["paired"]["scenes_pos"] if b["paired"] else None}
            for v, b in d["band"].items()]
    if d["band_ref"]:
        r = d["band_ref"]
        rows.append({"input": "B123 (mean of %d runs)" % r["k"], "psnr": r["mean"][0], "ssim": r["mean"][1], "lpips": r["mean"][2]})
    pd.DataFrame(rows).sort_values("input").to_csv(SRC / "band_ablation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"scene": sc, "random_split_psnr": v["random"], "held_out_psnr": v["held"], "held_out_fold": v["fold"]}
                  for sc, v in d["scene"]["per_scene"].items()]).to_csv(SRC / "scene_split_per_scene.csv", index=False, encoding="utf-8-sig")
    rows = [{"run": k, "fraction": k.split("_")[1], "epoch": int(k.rsplit("_ep", 1)[1]),
             "train_windows": TRAIN[k.split("_")[1]], "updates": ITERS[k.split("_")[1]] * int(k.rsplit("_ep", 1)[1]),
             "psnr": g["mean"][0], "ssim": g["mean"][1], "lpips": g["mean"][2]} for k, g in d["budget"].items()]
    for e in (100, 200):
        g = d["mswir_ep%d" % e]
        if g:
            rows.append({"run": "full_ep%d" % e, "fraction": "full", "epoch": e, "train_windows": TRAIN["full"],
                         "updates": ITERS["full"] * e, "psnr": g["mean"][0], "ssim": g["mean"][1], "lpips": g["mean"][2],
                         "psnr_sd": g["sd"][0] if g["sd"] else None})
    pd.DataFrame(rows).to_csv(SRC / "budget_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"dataset": data, "scene": sc, "test_windows": v["n"], "psnr": v["mean"][0], "ssim": v["mean"][1],
                   "lpips": v["mean"][2], "psnr_sd": v["sd"][0] if v["sd"] else None, "runs": v["k"]}
                  for data, scenes in d["scene_seeds"].items() for sc, v in scenes.items()]).to_csv(
        SRC / "scene_seeds.csv", index=False, encoding="utf-8-sig")
    per = pd.read_csv(ROOT / "outputs/data_quality/image_stats.csv")
    keep = ["set", "split", "image", "scene", "a_lag1", "b_lag1", "a_psd", "b_psd", "ab_r0", "ab_peak", "ab_shift"]
    per[keep].round(4).to_csv(SRC / "data_quality_windows.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(ROOT / "outputs/data_quality/band_stats.csv", SRC / "data_quality_bands.csv")
    shutil.copyfile(ROOT / "outputs/data_quality/band_corr.json", SRC / "data_quality_band_corr.json")
    shutil.copyfile(ROOT / "outputs/data_quality/band_hist.json", SRC / "data_quality_band_hist.json")
    rows = []
    for key, run in runs.items():
        p = run.per[["image", "psnr", "ssim_sk", "lpips_alex"]].copy()
        p.insert(0, "run", key)
        rows.append(p)
    pd.concat(rows).round(5).to_csv(SRC / "per_window_metrics.csv", index=False, encoding="utf-8-sig")
    import qualitative_sheet as Q
    rows = d["qualitative"]["windows"] if d.get("qualitative") else []
    pd.DataFrame(rows).to_csv(SRC / "qualitative_windows.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    for p in (OUT, GEN, SRC):
        p.mkdir(parents=True, exist_ok=True)
    runs = collect()
    d = build(runs)
    d["quality"] = load_quality()
    d["quality_bands"], d["quality_corr"] = quality_bands()
    d["quality_scene"] = quality_scene()
    d["quality_hist"] = quality_hist()
    d["qualitative"] = qualitative_block(runs)
    d = json.loads(json.dumps(clean(d), ensure_ascii=False))
    (OUT / "results_bench_20261124.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, body in B.tables(d).items():
        (GEN / name).write_text(body, encoding="utf-8")
    if not args.no_figures:
        B.figures(d, FIG)
    source_data(d, runs)
    print("runs:", len(runs), "| outputs in", OUT)


if __name__ == "__main__":
    main()
