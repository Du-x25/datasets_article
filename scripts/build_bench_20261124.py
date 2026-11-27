"""Build the 2026-11-24 benchmark tables, macros and figures from bundled source data.

Input : figures/source_data/bench_20261124/results.json
Output: generated/bench_*_20261124_zh.tex, figures/zh/results_20261124/*.pdf

The JSON holds per-run means, paired contrasts and quality statistics collected by the
experiment workspace. This script only formats them, so every committed table can be
regenerated and compared byte for byte (check_tables); no training or inference happens here.
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures/source_data/bench_20261124/results.json"
GEN = ROOT / "generated"
FIG = ROOT / "figures/zh/results_20261124"
SUFFIX = "_20261124_zh.tex"
METRICS = ("psnr", "ssim_sk", "lpips_alex")
SINGLES = ["B%d" % i for i in range(1, 6)]
TRIPLES = ["B%d%d%d" % t for t in itertools.combinations(range(1, 6), 3)]
BANDS = {"B1": "1000--1050", "B2": "1050--1100", "B3": "1225--1275", "B4": "1525--1575", "B5": "1625--1675"}
ITERS = {"frac25": 177, "frac50": 355, "frac75": 533, "full": 710}
TRAIN = {"frac25": 710, "frac50": 1422, "frac75": 2132, "full": 2842}
MATCHED = {"frac25": 400, "frac50": 200, "frac75": 130}
EPOCHS = {"frac25": (100, 200, 300, 400), "frac50": (50, 100, 150, 200), "frac75": (50, 100, 130)}
SCENES = ["C%d" % i for i in range(1, 11)]
SEEDS = ["s1", "s2", "s3"]
CHEN_SCENES = [str(i) for i in range(1, 8)]
WORDS = ["", "One", "Two", "Three", "Four", "Five"]
DEC = ((0, 2), (1, 3), (2, 3))
FOLDS = {"A": ("C1--C5", "C6--C10", "e4_sceneA_ep180"), "B": ("C6--C10", "C1--C5", "e4_sceneB_ep220")}
HELD = {"A": 1618, "B": 1934}  # test windows of the held-out scenes, fixed by the fold manifests
QUALITY_SETS = (("chen", "对照数据集图块"), ("mswir_4to1", "本数据集 4:1 窗口"), ("mswir_native", "本数据集 1:1 窗口"))


def fmt(x, dec):
    return "--" if x is None or not np.isfinite(x) else ("%." + str(dec) + "f") % x


def pm(g, i, dec):
    if g is None:
        return "--"
    sd = g.get("sd")
    if sd is None or sd[i] is None or not np.isfinite(sd[i]):
        return fmt(g["mean"][i], dec)
    return "%s$\\pm$%s" % (fmt(g["mean"][i], dec), fmt(sd[i], dec))


def cells(g):
    return [pm(g, i, dec) for i, dec in DEC]


def ci(p):
    return "--" if p is None else "$%+.2f$ [%+.2f, %+.2f]" % (p["mean"], p["lo"], p["hi"])


def signed(x, dec=2):
    return "--" if x is None else ("%+." + str(dec) + "f") % x


def thousands(n):
    return "{:,}".format(int(n)).replace(",", "{,}")


def rng(a, b, dec):
    lo, hi = ("%." + str(dec) + "f") % a, ("%." + str(dec) + "f") % b
    return ("$%s$--%s" % (lo, hi)) if a < 0 else "%s--%s" % (lo, hi)


def tabular(cols, header, rows):
    return "\n".join(["\\begin{tabular}{@{}%s@{}}" % cols, "\\toprule", header, "\\midrule", *rows,
                      "\\bottomrule", "\\end{tabular}"]) + "\n"


HEAD = "PSNR/dB$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$"


def latex_main(d):
    rows = ["%s & 100 epoch & %s & %s & %s \\\\" % ("对照数据集\\cite{chen2026thesis}", *cells(d["chen_ep100"])),
            " & 200 epoch & %s & %s & %s \\\\" % tuple(cells(d["chen_ep200"])), "\\midrule",
            "M-SWIR-RGB（4:1 窗口） & 100 epoch & %s & %s & %s \\\\" % tuple(cells(d["mswir_ep100"])),
            " & 200 epoch & %s & %s & %s \\\\" % tuple(cells(d["mswir_ep200"]))]
    return tabular("llrrr", "数据 & 训练预算 & " + HEAD + "\\\\", rows)


def latex_scene_seeds(d):
    rows = []
    for sc in SCENES:
        v = d["scene_seeds"]["mswir"].get(sc)
        if v is not None:
            rows.append("%s & %d & %s & %s & %s \\\\" % (sc, v["n"], pm(v, 0, 2), fmt(v["mean"][1], 3), fmt(v["mean"][2], 3)))
    rows.append("\\midrule")
    allr = d["mswir_ep100"]
    rows.append("十个场景合并 & %d & %s & %s & %s \\\\" % (d["runs"][d["mswir_ep100_runs"][0]]["n"], pm(allr, 0, 2),
                                                   fmt(allr["mean"][1], 3), fmt(allr["mean"][2], 3)))
    return tabular("lrrrr", "场景 & 测试窗口 & " + HEAD + "\\\\", rows)


def latex_band(d):
    ref = d["mswir_ep100"]
    rows = []
    for v in SINGLES + TRIPLES:
        if v == "B123":
            rows.append("B123（参照，%d 次训练） & %s & %s & %s & 参照 \\\\" % (ref["k"], *cells(ref)))
        elif v in d["band"]:
            b = d["band"][v]
            rows.append("%s & %s & %s & %s & %s \\\\" % (v, fmt(b["run"]["mean"][0], 2), fmt(b["run"]["mean"][1], 3),
                                                     fmt(b["run"]["mean"][2], 3), ci(b["paired"])))
        else:
            rows.append("%s & -- & -- & -- & -- \\\\" % v)
    return tabular("lrrrr", "输入 & " + HEAD + " & $\\Delta$PSNR 及 95\\% 置信区间\\\\", rows)


def band_complete(d):
    return all(v in d["band"] for v in SINGLES + TRIPLES if v != "B123")


def latex_budget(d):
    rows = []
    for name in ("frac25", "frac50", "frac75"):
        g = d["budget"].get("e3_%s_ep%d" % (name, MATCHED[name]))
        rows.append("%d\\%% & %d & %s & %s & %s & %s \\\\" % (
            int(name[4:]), TRAIN[name], thousands(ITERS[name] * MATCHED[name]), *cells(g)))
    g = d["mswir_ep100"]
    rows.append("100\\%% & %d & %s & %s & %s & %s \\\\" % (TRAIN["full"], thousands(ITERS["full"] * 100), *cells(g)))
    return tabular("rrlrrr", "训练集 & 训练窗口 & 更新次数 & " + HEAD + "\\\\", rows)


def latex_scene(d):
    s = d["scene"]
    diag = s.get("diag", {})
    held = {f: "留出的 %s 共 %d 个窗口" % (FOLDS[f][1], HELD[f]) for f in "AB"}

    def row(name, test, g):
        return "%s & %s & %s & %s & %s \\\\" % (name, test, *cells(g))
    rows = [row("随机窗口划分（%d 次训练）" % d["mswir_ep100"]["k"], "十个场景共 %d 个窗口" % d["runs"][d["mswir_ep100_runs"][0]]["n"],
                d["mswir_ep100"]), "\\midrule"]
    for fold in ("A", "B"):
        train, test, key = FOLDS[fold]
        g = s["fold" + fold]
        rows.append(row("场景划分 %s（训练 %s）" % (fold, train), held[fold], g))
    if all(all(k in diag.get(f, {}) for k in ("train", "shift_global", "baseline")) for f in "AB"):
        n = {f: sum(diag[f]["train_scene_n"].values()) for f in "AB"}
        rows.append("\\midrule")
        for fold in "AB":
            rows.append(row("划分 %s 的模型，自身训练窗口" % fold, "%s 的 %d 个窗口" % (FOLDS[fold][0], n[fold]), diag[fold]["train"]))
        for fold in "AB":
            rows.append(row("划分 %s 的模型，留出输入强度对齐" % fold, held[fold], diag[fold]["shift_global"]))
        rows.append("\\midrule")
        for name, key in (("输入的三通道均值（不训练）", "gray"), ("训练目标的通道均值（不训练）", "const")):
            for fold in "AB":
                rows.append("%s & %s & %s & -- & -- \\\\" % (name, held[fold], fmt(diag[fold]["baseline"][key], 2)))
    return tabular("llrrr", "模型 & 测试窗口 & " + HEAD + "\\\\", rows)


def latex_per_scene(d):
    s, rows = d["scene"], []
    for sc in SCENES:
        v = s["per_scene"].get(sc)
        if v is None:
            continue
        diff = None if v["held"] is None else v["held"] - v["random"]
        rows.append("%s & %s & %s & %s \\\\" % (sc, fmt(v["random"], 2), fmt(v["held"], 2), signed(diff)))
    if s.get("drop_scene_equal") is not None:
        rows += ["\\midrule", "十场景等权 & %s & %s & %s \\\\" % (fmt(s["random_scene_equal"], 2), fmt(s["held_scene_equal"], 2),
                                                         signed(s["held_scene_equal"] - s["random_scene_equal"]))]
    return tabular("lrrr", "场景 & 随机窗口划分 & 留出场景 & 差值\\\\", rows)


def latex_quality_scale(q):
    rows = []
    for key, label in QUALITY_SETS:
        c = q[key]["all"]
        near = c["ab_within1px_strong_pct"] if key != "mswir_native" else c["ab_within4px_strong_pct"]
        rows.append("%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %s \\\\" % (
            label, int(c["n"]), c["a_lag1_mean"], c["a_psd_mean"], c["b_lag1_mean"], c["b_psd_mean"],
            c["ab_r0_mean"], c["ab_peak_mean"], fmt(near, 1) + ("" if key != "mswir_native" else "$^{\\dagger}$")))
    head = ["图像集 & 数量 & \\multicolumn{2}{c}{SWIR 输入} & \\multicolumn{2}{c}{RGB 目标} & \\multicolumn{3}{c}{模态间灰度相关}\\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
            " & & lag-1 & 频谱重心 & lag-1 & 频谱重心 & 零位移 & $\\pm$16 像素内峰值 & 峰在指定邻域$^{*}$/\\%\\\\"]
    return "\n".join(["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule", *head, "\\midrule", *rows, "\\bottomrule",
                      "\\end{tabular}"]) + "\n"



def triple(v):
    return "/".join("%.1f" % v[c] for c in "RGB")


def latex_quality_scene(q):
    rows = ["%s & %d & %.1f & %.0f & %s & %s & %.2f & %.1f & %.1f \\\\" % (
        sc, v["windows"], v["b1_sat"], v["rgb_luma"], triple(v["rgb_sat"]), triple(v["rgb0"]),
        v["ab_r0_med"], v["weak"], v["within1"]) for sc, v in q.items()]
    head = ["场景 & 窗口 & B1取255/\\% & RGB均值 & RGB取255/\\% & RGB取0/\\% & 零位移相关 & 不可判定/\\% & 峰在中心邻域$^{*}$/\\%\\\\"]
    return "\n".join(["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule", *head, "\\midrule", *rows, "\\bottomrule",
                      "\\end{tabular}"]) + "\n"


def latex_quality_corr(groups):
    rows = ["B%s--B%s & %.3f & %.3f & %.3f \\\\" % (k[1], k[2], p["min"], p["median"], p["max"]) for k, p in groups["pairs"].items()]
    return tabular("lrrr", "波段对 & 十场景最小 & 中位数 & 最大\\\\", rows)


def latex_quality_bands(q):
    rows = ["%s & %s & %s & %s & %.2f & %.1f & %s & %s \\\\" % (
        b, BANDS[b], rng(v["mean_min"], v["mean_max"], 0), rng(v["std_min"], v["std_max"], 1), v["sat"], v["sat_max"],
        rng(v["corr_min"], v["corr_max"], 3), rng(v["luma_min"], v["luma_max"], 2)) for b, v in q.items()]
    head = ["波段 & 名义通带/nm & 场景均值 & 场景标准差 & \\multicolumn{2}{c}{DN=255 比例/\\%} & 与其余波段 & 与RGB亮度\\\\",
            "\\cmidrule(lr){5-6}", " & & 范围 & 范围 & 合并 & 最大场景 & 相关范围 & 相关范围\\\\"]
    return "\n".join(["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule", *head, "\\midrule", *rows, "\\bottomrule",
                      "\\end{tabular}"]) + "\n"


def latex_macros(d):
    out = []

    def add(name, value):
        assert name.isalpha(), name  # LaTeX control-sequence names may only contain letters
        out.append("\\newcommand{\\%s}{%s}" % (name, value))

    def group(prefix, g):
        for n, (i, dec) in zip(("Psnr", "Ssim", "Lpips"), DEC):
            add(prefix + n, "--" if g is None else fmt(g["mean"][i], dec))
            add(prefix + n + "Sd", "--" if g is None or g.get("sd") is None else fmt(g["sd"][i], dec))
            add(prefix + n + "Pm", pm(g, i, dec))

    for pre, g in (("MsEpHundred", d["mswir_ep100"]), ("MsEpTwoHundred", d["mswir_ep200"]),
                   ("ChEpHundred", d["chen_ep100"]), ("ChEpTwoHundred", d["chen_ep200"]),
                   ("NativeEpHundred", d["native"].get("100")), ("NativeEpTwoHundred", d["native"].get("200")),
                   ("MsEpFiveHundred", d.get("mswir_ep500")), ("ChEpFiveHundred", d.get("chen_ep500")),
                   ("FracTwentyFive", d["budget"].get("e3_frac25_ep400")), ("FracFifty", d["budget"].get("e3_frac50_ep200")),
                   ("FracSeventyFive", d["budget"].get("e3_frac75_ep130")),
                   ("FracTwentyFiveEpHundred", d["budget"].get("e3_frac25_ep100")),
                   ("FracFiftyEpHundred", d["budget"].get("e3_frac50_ep100")),
                   ("FracSeventyFiveEpHundred", d["budget"].get("e3_frac75_ep100")),
                   ("SceneFoldA", d["scene"].get("foldA")), ("SceneFoldB", d["scene"].get("foldB"))):
        group(pre, g)
    for pre, g in (("MsEpHundred", d["mswir_ep100"]), ("MsEpTwoHundred", d["mswir_ep200"]),
                   ("ChEpHundred", d["chen_ep100"]), ("ChEpTwoHundred", d["chen_ep200"])):
        add(pre + "Seeds", "--" if g is None else str(g["k"]))
    for pre, keys in (("MsEpHundred", d["mswir_ep100_runs"]), ("MsEpTwoHundred", d["mswir_ep200_runs"]),
                      ("ChEpHundred", d["chen_ep100_runs"]), ("ChEpTwoHundred", d["chen_ep200_runs"])):
        if len(keys) > 1:
            ps = [d["runs"][k]["psnr"] for k in keys]
            add(pre + "RunMin", "%.2f" % min(ps))
            add(pre + "RunMax", "%.2f" % max(ps))
            add(pre + "RunSpan", "%.2f" % (max(ps) - min(ps)))
    for pre, data in (("Ms", "mswir"), ("Ch", "chen")):
        keys100, keys200 = d["%s_ep100_runs" % data], d["%s_ep200_runs" % data]
        low = min(keys100, key=lambda k: d["runs"][k]["psnr"]) if len(keys100) > 1 else None
        later = [k for k in keys200 if low and k.rsplit("_ep", 1)[0] == low.rsplit("_ep", 1)[0]]
        if later:
            add(pre + "EpHundredLowRunGap", "%.2f" % (max(d["runs"][k]["psnr"] for k in keys100) - d["runs"][low]["psnr"]))
            add(pre + "EpTwoHundredLowRunGap", "%.2f" % (max(d["runs"][k]["psnr"] for k in keys200) - d["runs"][later[0]]["psnr"]))
    for name, label in (("100", "Hundred"), ("200", "TwoHundred")):
        full, native = d["mswir_ep%s" % name], d["native"].get(name)
        if full and native:
            add("NativeDrop" + label, "%.2f" % (full["mean"][0] - native["mean"][0]))
    if d.get("chen_late_swing") is not None:
        add("ChLateSwing", "%.2f" % d["chen_late_swing"])
    if d.get("gap"):
        add("GapEpHundred", signed(d["gap"]["100"]))
        add("GapEpTwoHundred", signed(d["gap"]["200"]))
        add("GapEpHundredPlain", "%.2f" % abs(d["gap"]["100"]))
        add("GapEpTwoHundredPlain", "%.2f" % abs(d["gap"]["200"]))
    full = d["mswir_ep100"]
    for name, key in (("TwentyFive", "e3_frac25_ep400"), ("Fifty", "e3_frac50_ep200"), ("SeventyFive", "e3_frac75_ep130")):
        part = d["budget"].get(key)
        if full is not None and part is not None:
            add("BudgetGain" + name, signed(full["mean"][0] - part["mean"][0]))
    s = d["scene"]
    for key, name, sg in (("held_scene_equal", "SceneHeldEqualPsnr", False), ("random_scene_equal", "SceneRandomEqualPsnr", False),
                          ("drop_scene_equal", "SceneDropEqualAbs", False),
                          ("drop_scene_equal", "SceneDropEqualPsnr", True), ("drop_fold_mean", "SceneDropFoldPsnr", True)):
        if s.get(key) is not None:
            add(name, signed(s[key]) if sg else "%.2f" % s[key])
    if s.get("scenes") is not None:
        add("SceneHeldLowerCount", str(s["scenes_lower"]))
        add("SceneHeldSceneCount", str(s["scenes"]))
    held = {sc: v["held"] for sc, v in s["per_scene"].items() if v["held"] is not None}
    for tag, pick in (("Min", min), ("Max", max)):
        if held:
            name = pick(held, key=held.get)
            add("SceneHeld" + tag + "Name", name)
            add("SceneHeld" + tag + "Psnr", "%.2f" % held[name])
    for fold in ("A", "B"):
        w = (s.get("same_windows") or {}).get(fold)
        if w:
            add("SceneSameWindowCount" + fold, str(w["n"]))
            add("SceneSameRandom" + fold + "Psnr", "%.2f" % w["random"])
            add("SceneSameHeld" + fold + "Psnr", "%.2f" % w["held"])
        v = (s.get("diag") or {}).get(fold)
        if not v:
            continue
        counts = sorted(v["train_scene_n"].items(), key=lambda kv: (-kv[1], kv[0]))
        if len(counts) > 1:
            total = sum(c for _, c in counts)
            add("SceneTrainMainName" + fold, counts[0][0])
            add("SceneTrainMainShare" + fold, "%.0f" % (100.0 * counts[0][1] / total))
            add("SceneTrainSecondName" + fold, counts[1][0])
            add("SceneTrainSecondShare" + fold, "%.0f" % (100.0 * counts[1][1] / total))
        for key, label in (("train", "SelfTrain"), ("shift_global", "ShiftGlobal"), ("shift_perimg", "ShiftPerimg")):
            if key in v:
                group("Scene" + label + fold, v[key])
        if "baseline" in v:
            add("SceneGrayBase" + fold + "Psnr", "%.2f" % v["baseline"]["gray"])
            add("SceneConstBase" + fold + "Psnr", "%.2f" % v["baseline"]["const"])
    changes = [abs(s["diag"][f][k]["mean"][0] - s["fold" + f]["mean"][0])
               for f in ("A", "B") if f in (s.get("diag") or {}) for k in ("shift_global", "shift_perimg") if k in s["diag"][f]]
    if changes:
        add("SceneShiftMaxChange", "%.2f" % max(changes))
    for key, label in (("best_single", "BandBestSingle"), ("worst_single", "BandWorstSingle"),
                       ("best_triple", "BandBestTriple"), ("worst_triple", "BandWorstTriple")):
        if d.get(key):
            add(label, d[key][0])
            add(label + "Psnr", "%.2f" % d[key][1])
    ref = d["mswir_ep100"]
    variants = [v for v in SINGLES + TRIPLES if v != "B123" and d["band"].get(v, {}).get("paired")]
    if band_complete(d) and ref is not None and ref.get("sd") is not None and variants:
        ps = {v: d["band"][v]["run"]["mean"][0] for v in variants}
        runs = [d["runs"][k]["psnr"] for k in d["mswir_ep100_runs"]]
        low = [v for v in variants if ps[v] < min(runs)]  # below every reference run
        rest = [v for v in variants if v not in low]
        p = d["band"]
        add("BandLowNames", "、".join(sorted(low, key=variants.index)))
        add("BandLowDropMin", "%.2f" % (ref["mean"][0] - max(ps[v] for v in low)))
        add("BandLowDropMax", "%.2f" % (ref["mean"][0] - min(ps[v] for v in low)))
        add("BandLowScenePosMax", str(max(p[v]["paired"]["scenes_pos"] for v in low)))
        add("BandRestCount", str(len(rest)))
        add("BandRestDiffMax", "%.2f" % max(abs(ps[v] - ref["mean"][0]) for v in rest))
        add("BandRestRangeExcess", "%.2f" % max(max(ps[v] - max(runs), min(runs) - ps[v], 0.0) for v in rest))
        excl = [v for v in variants if p[v]["paired"]["lo"] > 0 or p[v]["paired"]["hi"] < 0]
        add("BandCiExcludingZero", str(len(excl)))
        add("BandCiExcludingRest", str(len([v for v in excl if v not in low])))
        b3 = [v for v in rest if "3" in v]
        add("BandBThreeRestNames", "、".join(sorted(b3, key=ps.get)))
        add("BandBThreeRestDiffMax", "%.2f" % max(abs(ps[v] - ref["mean"][0]) for v in b3))
    for v in SINGLES:
        if v in d["band"]:
            add("Band" + WORDS[int(v[1])] + "SinglePsnr", "%.2f" % d["band"][v]["run"]["mean"][0])
    if all(v in d["band"] for v in ("B3", "B5")):
        add("BandBThreeFiveSingleDiff", "%.2f" % abs(d["band"]["B5"]["run"]["mean"][0] - d["band"]["B3"]["run"]["mean"][0]))
    for data, label, scenes in (("mswir", "Ms", SCENES), ("chen", "Ch", CHEN_SCENES)):
        vals = [d["scene_seeds"][data][sc] for sc in scenes if sc in d["scene_seeds"][data]]
        if vals:
            ps = [v["mean"][0] for v in vals]
            add(label + "SceneMinPsnr", "%.2f" % min(ps)), add(label + "SceneMaxPsnr", "%.2f" % max(ps))
            add(label + "SceneSpan", "%.2f" % (max(ps) - min(ps)))
            add(label + "SceneMinName", scenes[ps.index(min(ps))])
            sds = [v["sd"][0] for v in vals if v.get("sd")]
            if sds:
                add(label + "SceneSdMax", "%.2f" % max(sds))
            rho = d.get("scene_rank_rho", {}).get("mswir" if label == "Ms" else "chen")
            if rho is not None:
                add(label + "SceneRankRhoMin", "%.2f" % rho)
    q = d["quality"]
    for key, label in (("chen", "ScaleChen"), ("mswir_4to1", "ScaleMs"), ("mswir_native", "ScaleNative")):
        for suf, col in (("LagOne", "a_lag1_mean"), ("Psd", "a_psd_mean"), ("BLagOne", "b_lag1_mean"),
                         ("BPsd", "b_psd_mean"), ("AbCorr", "ab_r0_mean"), ("AbPeak", "ab_peak_mean")):
            add(label + suf, "%.3f" % q[key]["all"][col])
        add(label + "AbRatio", "%.0f" % q[key]["all"]["ab_ratio_pct"])
        add(label + "PeakWithinOnePx", "%.1f" % q[key]["all"]["ab_peak_within_1px_pct"])
        add(label + "PeakWeakPct", "%.1f" % q[key]["all"]["ab_peak_weak_pct"])
        add(label + "PeakWithinOnePxStrong", "%.1f" % q[key]["all"]["ab_within1px_strong_pct"])
        add(label + "RgbChanCorr", "%.3f" % q[key]["all"]["b_cc_mean"])
        add(label + "RgbSat", "/".join("%.1f" % q[key]["all"]["b_sat%d_mean" % i] for i in (1, 2, 3)))
    add("ScaleNativePeakWithinFourPx", "%.1f" % q["mswir_native"]["all"]["ab_within4px_strong_pct"])
    qb = d["quality_bands"]
    for b, v in qb.items():
        if b in SINGLES:
            name = "BandB" + WORDS[int(b[1])]  # control-sequence names may only contain letters
            add(name + "MeanMin", "%.0f" % v["mean_min"])
            add(name + "MeanMax", "%.0f" % v["mean_max"])
            add(name + "Sat", "%.2f" % v["sat"])
            add(name + "SatMax", "%.1f" % v["sat_max"])
            add(name + "CorrMin", "%.3f" % v["corr_min"])
            add(name + "CorrMax", "%.3f" % v["corr_max"])
            add(name + "LumaMin", "%+.2f" % v["luma_min"])
            add(name + "LumaMax", "%+.2f" % v["luma_max"])
    for name, key in (("Within", "within"), ("Cross", "cross")):
        lo, mid, hi = d["quality_corr"][key]
        add("Corr" + name + "Min", "%.3f" % lo)
        add("Corr" + name + "Median", "%.3f" % mid)
        add("Corr" + name + "Max", "%.3f" % hi)
    qs = d["quality_scene"]
    top = sorted(qs, key=lambda sc: -qs[sc]["b1_sat"])
    add("BandBOneSatTopName", top[0])
    add("BandBOneSatTop", "%.1f" % qs[top[0]]["b1_sat"])
    add("BandBOneSatSecondName", top[1])
    add("BandBOneSatSecond", "%.1f" % qs[top[1]]["b1_sat"])
    flat = {(sc, c): v for sc, q in qs.items() for c, v in q["rgb_sat"].items()}
    sat_name = max(flat, key=flat.get)[0]
    add("RgbSatMaxName", sat_name)
    add("RgbSatMax", "%.1f" % max(flat.values()))
    add("RgbSatSecond", "%.1f" % max(v for (sc, c), v in flat.items() if sc != sat_name))
    flat = {(sc, c): v for sc, q in qs.items() for c, v in q["rgb0"].items()}
    name, c = max(flat, key=flat.get)
    add("RgbZeroMaxName", name)
    add("RgbZeroMaxChannel", {"R": "红", "G": "绿", "B": "蓝"}[c])
    add("RgbZeroMax", "%.1f" % flat[(name, c)])
    add("SceneAlignMinName", min(qs, key=lambda sc: qs[sc]["within1"]))
    add("SceneAlignMin", "%.1f" % min(v["within1"] for v in qs.values()))
    add("SceneAlignMaxName", max(qs, key=lambda sc: qs[sc]["within1"]))
    add("SceneAlignMax", "%.1f" % max(v["within1"] for v in qs.values()))
    add("SceneCorrMin", "%.2f" % min(v["ab_r0_med"] for v in qs.values()))
    add("SceneCorrMax", "%.2f" % max(v["ab_r0_med"] for v in qs.values()))
    weak = sorted(qs, key=lambda sc: -qs[sc]["weak"])
    add("SceneWeakMaxName", weak[0])
    add("SceneWeakMax", "%.1f" % qs[weak[0]]["weak"])
    add("SceneWeakSecondName", weak[1])
    add("SceneWeakSecond", "%.1f" % qs[weak[1]]["weak"])
    add("ScaleMsWeakStd", "%.1f" % q["mswir_4to1"]["all"]["weak_std_mean"])
    add("ScaleMsStrongStd", "%.1f" % q["mswir_4to1"]["all"]["strong_std_mean"])
    add("BandDone", "1" if band_complete(d) else "0")
    add("SceneDone", "1" if d["scene"].get("foldA") and d["scene"].get("foldB") else "0")
    add("BudgetDone", "1" if all(d["budget"].get("e3_%s_ep%d" % (n, MATCHED[n])) for n in MATCHED) else "0")
    if d.get("qualitative"):
        add("QualPsnrMin", "%.2f" % d["qualitative"]["psnr_min"])
        add("QualPsnrMax", "%.2f" % d["qualitative"]["psnr_max"])
    return "\n".join(out) + "\n"


def tables(d):
    return {
        "bench_main" + SUFFIX: latex_main(d), "bench_scene_seeds" + SUFFIX: latex_scene_seeds(d),
        "bench_band" + SUFFIX: latex_band(d), "bench_budget" + SUFFIX: latex_budget(d),
        "bench_scene" + SUFFIX: latex_scene(d), "bench_per_scene" + SUFFIX: latex_per_scene(d),
        "bench_quality_scale" + SUFFIX: latex_quality_scale(d["quality"]),
        "bench_quality_bands" + SUFFIX: latex_quality_bands(d["quality_bands"]),
        "bench_quality_scene" + SUFFIX: latex_quality_scene(d["quality_scene"]),
        "bench_quality_corr" + SUFFIX: latex_quality_corr(d["quality_corr"]),
        "bench_macros" + SUFFIX: latex_macros(d),
    }


def figures(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_fonts import select_plot_font
    family, _ = select_plot_font()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grey, teal, dark = "#9fb3c8", "#4f9f9a", "#1f5553"

    ref = d["mswir_ep100"]
    order = [v for v in SINGLES + TRIPLES if v == "B123" or v in d["band"]]
    if ref is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        for y, v in enumerate(order):
            if v == "B123":
                per = np.array(list(d["band_ref_scene_mean"].values()) or [np.nan])
                ax.barh(y, ref["mean"][0], color=dark, height=0.62, xerr=0.0 if ref["sd"] is None else ref["sd"][0],
                        error_kw={"elinewidth": 0.8, "capsize": 2})
            else:
                per = np.array([d["band"][v]["scene_mean"][sc] for sc in d["band_ref_scene_mean"]])
                ax.barh(y, d["band"][v]["run"]["mean"][0], color=grey if v in SINGLES else teal, height=0.62)
            ax.scatter(per, np.full(len(per), y), s=7, color="black", alpha=0.5, linewidths=0, zorder=3)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlabel("测试 PSNR / dB（点为各场景内的窗口均值）")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.savefig(out / "band_ablation_zh.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.9, 3.0), layout="constrained")
    style = {"frac25": ("25%", grey, "o"), "frac50": ("50%", "#5f8fb0", "s"), "frac75": ("75%", "#3f8f8a", "^"),
             "full": ("100%", dark, "D")}
    for name, (label, color, marker) in style.items():
        if name == "full":
            pts = [(ITERS["full"] * e, g["mean"][0]) for e, g in ((100, d["mswir_ep100"]), (200, d["mswir_ep200"]))
                   if g is not None]
        else:
            pts = [(ITERS[name] * int(k.rsplit("_ep", 1)[1]), g["mean"][0])
                   for k, g in d["budget"].items() if k.startswith("e3_%s_" % name)]
        if pts:
            ax.plot([p[0] / 1000.0 for p in pts], [p[1] for p in pts], marker=marker, ms=4, lw=1.1, color=color,
                    label="%s（%d 个训练窗口）" % (label, TRAIN[name]))
    if ref is not None and ref["sd"] is not None:
        ax.errorbar([71.0], [ref["mean"][0]], yerr=ref["sd"][0], color=dark, fmt="none", capsize=2, lw=0.9)
    ax.axvline(71.0, color="#a05a1e", lw=0.9, ls="--")
    ax.set_xlabel("参数更新次数 / 千次")
    ax.set_ylabel("测试 PSNR / dB")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(out / "budget_curve_zh.pdf")
    plt.close(fig)

    hist = d.get("quality_hist")
    if hist:
        fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.3), layout="constrained", sharex=True, sharey=True)
        cmap = plt.get_cmap("turbo")
        x = np.arange(257)
        for ax, b in zip(axes.flat, ["B1", "B2", "B3", "B4", "B5", "LUMA"]):
            for i, sc in enumerate(SCENES):
                v = np.array(hist[sc][b] + [hist[sc][b][-1]], dtype=float)
                ax.step(x, np.where(v > 1e-4, v, np.nan), where="post", lw=0.8, color=cmap(i / 9.0), label=sc)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 30)
            ax.set_xlim(0, 256)
            ax.set_title("RGB 均值（参照）" if b == "LUMA" else b, fontsize=8)
            ax.grid(color="#eeeeee", linewidth=0.5)
            ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel("像素比例 / %", fontsize=6.5)
        for ax in axes[1]:
            ax.set_xlabel("数字量")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=10, frameon=False, fontsize=7, loc="outside lower center")
        fig.savefig(out / "band_histograms_zh.pdf")
        plt.close(fig)

    s = d["scene"]
    if ref is not None:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
        ax = axes[0]
        names, vals, errs, colors = [], [], [], []
        for label, g, color in (("随机窗口", ref, grey), ("场景划分 A", s.get("foldA"), teal), ("场景划分 B", s.get("foldB"), teal)):
            if g is not None:
                names.append(label)
                vals.append(float(g["mean"][0]))
                errs.append(0.0 if g["sd"] is None else float(g["sd"][0]))
                colors.append(color)
        ax.bar(range(len(names)), vals, color=colors, width=0.6, yerr=errs, error_kw={"elinewidth": 0.8, "capsize": 2})
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.6, "%.2f" % v, ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), names)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.18)
        ax.set_ylabel("测试 PSNR / dB")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax = axes[1]
        both = [sc for sc in SCENES if s["per_scene"].get(sc) and s["per_scene"][sc]["held"] is not None]
        if both:
            xs = np.arange(len(both))
            r_vals = [s["per_scene"][sc]["random"] for sc in both]
            h_vals = [s["per_scene"][sc]["held"] for sc in both]
            for x, sc in zip(xs, both):
                ax.plot([x, x], [s["per_scene"][sc]["held"], s["per_scene"][sc]["random"]], color="#999999", lw=0.8)
            ax.scatter(xs, r_vals, s=18, color=grey, zorder=3, label="随机窗口划分（%d 次训练均值）" % ref["k"])
            ax.scatter(xs, h_vals, s=20, color=teal, marker="^", zorder=3, label="留出场景")
            ax.set_xticks(xs, both)
            ax.set_ylim(min(h_vals) - 1.5, max(r_vals) + 4.8)
            ax.set_ylabel("场景内 PSNR / dB")
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
        fig.savefig(out / "scene_boundary_zh.pdf")
        plt.close(fig)


def figures_en(d, out):
    import csv as _csv
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from pypdf import PdfReader

    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "font.size": 8})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grey, teal, dark = "#9fb3c8", "#4f9f9a", "#1f5553"

    ref = d["mswir_ep100"]
    order = [v for v in SINGLES + TRIPLES if v == "B123" or v in d["band"]]
    if ref is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        for y, v in enumerate(order):
            if v == "B123":
                per = np.array(list(d["band_ref_scene_mean"].values()) or [np.nan])
                ax.barh(y, ref["mean"][0], color=dark, height=0.62, xerr=0.0 if ref["sd"] is None else ref["sd"][0],
                        error_kw={"elinewidth": 0.8, "capsize": 2})
            else:
                per = np.array([d["band"][v]["scene_mean"][sc] for sc in d["band_ref_scene_mean"]])
                ax.barh(y, d["band"][v]["run"]["mean"][0], color=grey if v in SINGLES else teal, height=0.62)
            ax.scatter(per, np.full(len(per), y), s=7, color="black", alpha=0.5, linewidths=0, zorder=3)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlabel("Test PSNR / dB (dots show per-scene window means)")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.savefig(out / "band_ablation_en.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.9, 3.0), layout="constrained")
    style = {"frac25": ("25%", grey, "o"), "frac50": ("50%", "#5f8fb0", "s"), "frac75": ("75%", "#3f8f8a", "^"),
             "full": ("100%", dark, "D")}
    for name, (label, color, marker) in style.items():
        if name == "full":
            pts = [(ITERS["full"] * e, g["mean"][0]) for e, g in ((100, d["mswir_ep100"]), (200, d["mswir_ep200"]))
                   if g is not None]
        else:
            pts = [(ITERS[name] * int(k.rsplit("_ep", 1)[1]), g["mean"][0])
                   for k, g in d["budget"].items() if k.startswith("e3_%s_" % name)]
        if pts:
            ax.plot([p[0] / 1000.0 for p in pts], [p[1] for p in pts], marker=marker, ms=4, lw=1.1, color=color,
                    label="%s (%d train windows)" % (label, TRAIN[name]))
    if ref is not None and ref["sd"] is not None:
        ax.errorbar([71.0], [ref["mean"][0]], yerr=ref["sd"][0], color=dark, fmt="none", capsize=2, lw=0.9)
    ax.axvline(71.0, color="#a05a1e", lw=0.9, ls="--")
    ax.set_xlabel("Parameter updates / thousands")
    ax.set_ylabel("Test PSNR / dB")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(out / "budget_curve_en.pdf")
    plt.close(fig)

    hist = d.get("quality_hist")
    if hist:
        fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.3), layout="constrained", sharex=True, sharey=True)
        cmap = plt.get_cmap("turbo")
        x = np.arange(257)
        for ax, b in zip(axes.flat, ["B1", "B2", "B3", "B4", "B5", "LUMA"]):
            for i, sc in enumerate(SCENES):
                v = np.array(hist[sc][b] + [hist[sc][b][-1]], dtype=float)
                ax.step(x, np.where(v > 1e-4, v, np.nan), where="post", lw=0.8, color=cmap(i / 9.0), label=sc)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 30)
            ax.set_xlim(0, 256)
            ax.set_title("RGB mean (ref.)" if b == "LUMA" else b, fontsize=8)
            ax.grid(color="#eeeeee", linewidth=0.5)
            ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Pixel share / %", fontsize=6.5)
        for ax in axes[1]:
            ax.set_xlabel("Digital number (DN)")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=10, frameon=False, fontsize=7, loc="outside lower center")
        fig.savefig(out / "band_histograms_en.pdf")
        plt.close(fig)

    s = d["scene"]
    if ref is not None:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
        ax = axes[0]
        names, vals, errs, colors = [], [], [], []
        for label, g, color in (("Random\nwindow", ref, grey), ("Scene\nsplit A", s.get("foldA"), teal), ("Scene\nsplit B", s.get("foldB"), teal)):
            if g is not None:
                names.append(label)
                vals.append(float(g["mean"][0]))
                errs.append(0.0 if g["sd"] is None else float(g["sd"][0]))
                colors.append(color)
        ax.bar(range(len(names)), vals, color=colors, width=0.6, yerr=errs, error_kw={"elinewidth": 0.8, "capsize": 2})
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.6, "%.2f" % v, ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), names)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.18)
        ax.set_ylabel("Test PSNR / dB")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax = axes[1]
        both = [sc for sc in SCENES if s["per_scene"].get(sc) and s["per_scene"][sc]["held"] is not None]
        if both:
            xs = np.arange(len(both))
            r_vals = [s["per_scene"][sc]["random"] for sc in both]
            h_vals = [s["per_scene"][sc]["held"] for sc in both]
            for x_pos, sc in zip(xs, both):
                ax.plot([x_pos, x_pos], [s["per_scene"][sc]["held"], s["per_scene"][sc]["random"]], color="#999999", lw=0.8)
            ax.scatter(xs, r_vals, s=18, color=grey, zorder=3, label="Random window split (%d-run mean)" % ref["k"])
            ax.scatter(xs, h_vals, s=20, color=teal, marker="^", zorder=3, label="Held-out scene")
            ax.set_xticks(xs, both)
            ax.set_ylim(min(h_vals) - 1.5, max(r_vals) + 4.8)
            ax.set_ylabel("Within-site PSNR / dB")
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
        fig.savefig(out / "scene_boundary_en.pdf")
        plt.close(fig)

    # Re-label embedded lossless panels from heldout_examples_zh.pdf and qualitative_zh.pdf into English.
    held_src = FIG / "heldout_examples_zh.pdf"
    held_csv = SRC.parent / "heldout_examples_windows.csv"
    if held_src.is_file() and held_csv.is_file():
        imgs = [Image.open(_io.BytesIO(im.data)).convert("RGB") for im in PdfReader(held_src).pages[0].images]
        with held_csv.open(encoding="utf-8-sig") as f:
            hrows = list(_csv.DictReader(f))
        if len(imgs) == 16 and len(hrows) == 4:
            fig, axes = plt.subplots(2, 10, figsize=(6.75, 2.0), dpi=300,
                                     gridspec_kw=dict(width_ratios=[0.62, 1, 1, 1, 1] * 2, wspace=0.04, hspace=0.06))
            for r in range(2):
                for c in (0, 5):
                    axes[r, c].axis("off")
            cols = ["SWIR input", "Random\nsplit output", "Scene-split\noutput", "RGB target"]
            for idx, row in enumerate(hrows):
                r, b = divmod(idx, 2)
                axes[r, 5 * b].text(1.0, 0.5, "Split %s\n%s" % (row["fold"], row["scene"]),
                                    transform=axes[r, 5 * b].transAxes, ha="right", va="center", fontsize=6.5, linespacing=1.2)
                for c in range(4):
                    ax = axes[r, c + 1 + 5 * b]
                    ax.imshow(imgs[idx * 4 + c], interpolation="nearest")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    if r == 0:
                        ax.set_title(cols[c], fontsize=6.5, pad=2, linespacing=1.0)
                    if c in (1, 2):
                        ax.text(0.03, 0.03, "%.2f dB" % float(row["psnr_random" if c == 1 else "psnr_held"]),
                                transform=ax.transAxes, fontsize=6, va="bottom",
                                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
            fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.005)
            fig.savefig(out / "heldout_examples_en.pdf")
            plt.close(fig)

    qual_src = FIG / "qualitative_zh.pdf"
    qual_csv = SRC.parent / "qualitative_windows.csv"
    if qual_src.is_file() and qual_csv.is_file():
        imgs = [Image.open(_io.BytesIO(im.data)).convert("RGB") for im in PdfReader(qual_src).pages[0].images]
        with qual_csv.open(encoding="utf-8-sig") as f:
            qrows = list(_csv.DictReader(f))
        if len(imgs) == 30 and len(qrows) == 10:
            half = 5
            fig, axes = plt.subplots(half, 7, figsize=(6.75, 1.12 * half + 0.3), dpi=260,
                                     gridspec_kw=dict(width_ratios=[1, 1, 1, 0.14, 1, 1, 1], wspace=0.04, hspace=0.04))
            for ax in axes[:, 3]:
                ax.axis("off")
            cols = ["SWIR (B1--B3)", "TFD-GAN output", "RGB target"]
            for i, row in enumerate(qrows):
                block, r = divmod(i, half)
                row_major_idx = r * 2 + block
                for c in range(3):
                    ax = axes[r, 3 * block + c + block]
                    ax.imshow(imgs[row_major_idx * 3 + c], interpolation="nearest")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    if r == 0:
                        ax.set_title(cols[c], fontsize=7.5, pad=2)
                    if c == 0:
                        ax.set_ylabel("%s\n%04d, %04d" % (row["scene"], int(row["top"]), int(row["left"])),
                                      fontsize=7, linespacing=1.1, labelpad=2)
            fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.005)
            fig.savefig(out / "qualitative_en.pdf")
            plt.close(fig)


def tables_en(d):
    def latex_main_en(d):
        rows = ["%s & 100 epochs & %s & %s & %s \\\\" % ("Reference archive~\\cite{chen2026thesis}", *cells(d["chen_ep100"])),
                " & 200 epochs & %s & %s & %s \\\\" % tuple(cells(d["chen_ep200"])), "\\midrule",
                "M-SWIR-RGB (4:1 windows) & 100 epochs & %s & %s & %s \\\\" % tuple(cells(d["mswir_ep100"])),
                " & 200 epochs & %s & %s & %s \\\\" % tuple(cells(d["mswir_ep200"]))]
        return tabular("llrrr", "Dataset & Training budget & " + HEAD + "\\\\", rows)

    def latex_scene_seeds_en(d):
        rows = []
        for sc in SCENES:
            v = d["scene_seeds"]["mswir"].get(sc)
            if v is not None:
                rows.append("%s & %d & %s & %s & %s \\\\" % (sc, v["n"], pm(v, 0, 2), fmt(v["mean"][1], 3), fmt(v["mean"][2], 3)))
        rows.append("\\midrule")
        allr = d["mswir_ep100"]
        rows.append("All ten sites pooled & %d & %s & %s & %s \\\\" % (d["runs"][d["mswir_ep100_runs"][0]]["n"], pm(allr, 0, 2),
                                                                      fmt(allr["mean"][1], 3), fmt(allr["mean"][2], 3)))
        return tabular("lrrrr", "Site & Test windows & " + HEAD + "\\\\", rows)

    def latex_band_en(d):
        ref = d["mswir_ep100"]
        rows = []
        for v in SINGLES + TRIPLES:
            if v == "B123":
                rows.append("B123 (ref., %d runs) & %s & %s & %s & Reference \\\\" % (ref["k"], *cells(ref)))
            elif v in d["band"]:
                b = d["band"][v]
                rows.append("%s & %s & %s & %s & %s \\\\" % (v, fmt(b["run"]["mean"][0], 2), fmt(b["run"]["mean"][1], 3),
                                                             fmt(b["run"]["mean"][2], 3), ci(b["paired"])))
            else:
                rows.append("%s & -- & -- & -- & -- \\\\" % v)
        return tabular("lrrrr", "Input & " + HEAD + " & $\\Delta$PSNR and 95\\% CI\\\\", rows)

    def latex_budget_en(d):
        rows = []
        for name in ("frac25", "frac50", "frac75"):
            g = d["budget"].get("e3_%s_ep%d" % (name, MATCHED[name]))
            rows.append("%d\\%% & %d & %s & %s & %s & %s \\\\" % (
                int(name[4:]), TRAIN[name], thousands(ITERS[name] * MATCHED[name]), *cells(g)))
        g = d["mswir_ep100"]
        rows.append("100\\%% & %d & %s & %s & %s & %s \\\\" % (TRAIN["full"], thousands(ITERS["full"] * 100), *cells(g)))
        return tabular("rrlrrr", "Train fraction & Train windows & Updates & " + HEAD + "\\\\", rows)

    def latex_scene_en(d):
        s = d["scene"]
        diag = s.get("diag", {})
        held = {f: "Held-out %s (%d windows)" % (FOLDS[f][1], HELD[f]) for f in "AB"}
        def row(name, test, g):
            return "%s & %s & %s & %s & %s \\\\" % (name, test, *cells(g))
        rows = [row("Random window split (%d runs)" % d["mswir_ep100"]["k"],
                    "All 10 sites (%d windows)" % d["runs"][d["mswir_ep100_runs"][0]]["n"], d["mswir_ep100"]), "\\midrule"]
        for fold in ("A", "B"):
            train, _, _ = FOLDS[fold]
            rows.append(row("Scene split %s (train %s)" % (fold, train), held[fold], s["fold" + fold]))
        if all(all(k in diag.get(f, {}) for k in ("train", "shift_global", "baseline")) for f in "AB"):
            n = {f: sum(diag[f]["train_scene_n"].values()) for f in "AB"}
            rows.append("\\midrule")
            for fold in "AB":
                rows.append(row("Split %s model, own train windows" % fold, "%s (%d windows)" % (FOLDS[fold][0], n[fold]), diag[fold]["train"]))
            for fold in "AB":
                rows.append(row("Split %s model, intensity-aligned input" % fold, held[fold], diag[fold]["shift_global"]))
            rows.append("\\midrule")
            for name, key in (("Three-channel input mean (untrained)", "gray"), ("Train target channel mean (untrained)", "const")):
                for fold in "AB":
                    rows.append("%s & %s & %s & -- & -- \\\\" % (name, held[fold], fmt(diag[fold]["baseline"][key], 2)))
        return tabular("llrrr", "Model & Test windows & " + HEAD + "\\\\", rows)

    def latex_per_scene_en(d):
        s, rows = d["scene"], []
        for sc in SCENES:
            v = s["per_scene"].get(sc)
            if v is None:
                continue
            diff = None if v["held"] is None else v["held"] - v["random"]
            rows.append("%s & %s & %s & %s \\\\" % (sc, fmt(v["random"], 2), fmt(v["held"], 2), signed(diff)))
        if s.get("drop_scene_equal") is not None:
            rows += ["\\midrule", "Ten-site equal mean & %s & %s & %s \\\\" % (
                fmt(s["random_scene_equal"], 2), fmt(s["held_scene_equal"], 2), signed(s["held_scene_equal"] - s["random_scene_equal"]))]
        return tabular("lrrr", "Site & Random split & Held-out site & Difference\\\\", rows)

    def latex_quality_scale_en(q):
        labels_en = (("chen", "Archive patches~\\cite{chen2026thesis}"), ("mswir_4to1", "M-SWIR-RGB 4:1 windows"), ("mswir_native", "M-SWIR-RGB 1:1 windows"))
        rows = []
        for key, label in labels_en:
            c = q[key]["all"]
            near = c["ab_within1px_strong_pct"] if key != "mswir_native" else c["ab_within4px_strong_pct"]
            rows.append("%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %s \\\\" % (
                label, int(c["n"]), c["a_lag1_mean"], c["a_psd_mean"], c["b_lag1_mean"], c["b_psd_mean"],
                c["ab_r0_mean"], c["ab_peak_mean"], fmt(near, 1) + ("" if key != "mswir_native" else "$^{\\dagger}$")))
        head = ["Window set & Count & \\multicolumn{2}{c}{SWIR input} & \\multicolumn{2}{c}{RGB target} & \\multicolumn{3}{c}{Cross-modal greyscale correlation}\\\\",
                "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
                " & & lag-1 & PSD centroid & lag-1 & PSD centroid & Zero-shift & Peak ($\\pm$16 px) & Peak in zone$^{*}$/\\%\\\\"]
        return "\n".join(["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule", *head, "\\midrule", *rows, "\\bottomrule",
                          "\\end{tabular}"]) + "\n"

    def latex_quality_scene_en(q):
        rows = ["%s & %d & %.1f & %.0f & %s & %s & %.2f & %.1f & %.1f \\\\" % (
            sc, v["windows"], v["b1_sat"], v["rgb_luma"], triple(v["rgb_sat"]), triple(v["rgb0"]),
            v["ab_r0_med"], v["weak"], v["within1"]) for sc, v in q.items()]
        head = ["Site & Windows & B1=255/\\% & RGB mean & RGB=255/\\% & RGB=0/\\% & Median $r_0$ & Weak peak/\\% & Peak in $3\\times 3^{*}$/\\%\\\\"]
        return "\n".join(["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule", *head, "\\midrule", *rows, "\\bottomrule",
                          "\\end{tabular}"]) + "\n"

    def latex_quality_corr_en(groups):
        rows = ["B%s--B%s & %.3f & %.3f & %.3f \\\\" % (k[1], k[2], p["min"], p["median"], p["max"]) for k, p in groups["pairs"].items()]
        return tabular("lrrr", "Band pair & Site min & Median & Max\\\\", rows)

    def latex_quality_bands_en(q):
        rows = ["%s & %s & %s & %s & %.2f & %.1f & %s & %s \\\\" % (
            b, BANDS[b], rng(v["mean_min"], v["mean_max"], 0), rng(v["std_min"], v["std_max"], 1), v["sat"], v["sat_max"],
            rng(v["corr_min"], v["corr_max"], 3), rng(v["luma_min"], v["luma_max"], 2)) for b, v in q.items()]
        head = ["Band & Nominal passband/nm & Site mean & Site SD & \\multicolumn{2}{c}{DN=255 share/\\%} & Inter-band & vs. RGB luma\\\\",
                "\\cmidrule(lr){5-6}", " & & range & range & Pooled & Max site & corr. range & corr. range\\\\"]
        return "\n".join(["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule", *head, "\\midrule", *rows, "\\bottomrule",
                          "\\end{tabular}"]) + "\n"

    macros_en = latex_macros(d)
    macros_en = macros_en.replace("、", ", ").replace("{蓝}", "{blue}").replace("{红}", "{red}").replace("{绿}", "{green}")
    suf = "_20261124_en.tex"
    return {
        "bench_main" + suf: latex_main_en(d), "bench_scene_seeds" + suf: latex_scene_seeds_en(d),
        "bench_band" + suf: latex_band_en(d), "bench_budget" + suf: latex_budget_en(d),
        "bench_scene" + suf: latex_scene_en(d), "bench_per_scene" + suf: latex_per_scene_en(d),
        "bench_quality_scale" + suf: latex_quality_scale_en(d["quality"]),
        "bench_quality_bands" + suf: latex_quality_bands_en(d["quality_bands"]),
        "bench_quality_scene" + suf: latex_quality_scene_en(d["quality_scene"]),
        "bench_quality_corr" + suf: latex_quality_corr_en(d["quality_corr"]),
        "bench_macros" + suf: macros_en,
    }


def check_tables():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[/\\]", text), "results.json must not contain local paths"
    d = json.loads(text)
    for name, body in tables(d).items():
        assert (GEN / name).read_text(encoding="utf-8") == body, "stale generated table: %s" % name
    for done, figure in ((band_complete(d), "band_ablation_zh.pdf"),
                         (d["scene"].get("foldA") is not None and d["scene"].get("foldB") is not None, "scene_boundary_zh.pdf")):
        assert not done or (FIG / figure).stat().st_mtime >= SRC.stat().st_mtime, "stale figure: %s" % figure
    return sorted(tables(d))


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    for name, body in tables(d).items():
        (GEN / name).write_text(body, encoding="utf-8")
    for name, body in tables_en(d).items():
        (GEN / name).write_text(body, encoding="utf-8")
    figures(d, FIG)
    figures_en(d, ROOT / "figures/en/results_20261124")
    print("wrote %d zh + %d en tables and figures from %s" % (len(tables(d)), len(tables_en(d)), SRC))


if __name__ == "__main__":
    main()
