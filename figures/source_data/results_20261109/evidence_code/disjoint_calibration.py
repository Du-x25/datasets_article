"""Spatially disjoint target-site calibration check.

Each scene is split along its longer axis.  For both directions, ten 256x256
patches are fitted in one half and the joint RGB quadratic adapter is scored in
the other half.  A same-half control uses ten patches from the evaluation half
and the identical evaluation pixels, so the only change is calibration
location, not scene content or evaluation region.
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t, ttest_rel, wilcoxon
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Conversion from CIE-LAB")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

sys.path.insert(0, r"D:\dh\MSWIR_RGB")
from mswir.data import SceneStore
from mswir.runtime import OfflineLPIPS
import torch

P = Path(r"D:\dh\MSWIR_RGB")
RUN = P / "runs/three_channel_v2_16680_b8/formal"
OUT = RUN / "report/calibration/disjoint_calibration.csv"
SUMMARY = RUN / "report/calibration/disjoint_calibration_summary.csv"
STATS = RUN / "report/calibration/disjoint_calibration_stats.csv"
FIGURE = RUN / "report/figures/F12_disjoint_calibration"
BANDS = (1, 3, 4)
PATCH, HALO, EVAL_PIXELS, CHUNK, MAX_LPIPS = 256, 16, 500_000, 2_000_000, 16
SITES = sys.argv[1:] or ["C%d" % i for i in range(1, 11)]

store = SceneStore(P/"data/release", P/"data/manifest.json", P/"data/support", cache_sites=1)
lpips_fn = OfflineLPIPS(P/"weights", "cuda" if torch.cuda.is_available() else "cpu")


def quad_features(x):
    columns = [x, x ** 2]
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            columns.append((x[:, i] * x[:, j])[:, None])
    columns.append(np.ones((len(x), 1), np.float32))
    return np.concatenate(columns, 1)


def fit(features, target):
    return np.linalg.lstsq(features, target, rcond=None)[0]


def predict(flat_bands, coefficient, shape):
    result = np.empty((flat_bands.shape[0], 3), np.float32)
    for start in range(0, flat_bands.shape[0], CHUNK):
        stop = min(start + CHUNK, flat_bands.shape[0])
        result[start:stop] = quad_features(flat_bands[start:stop]) @ coefficient
    return np.clip(result.reshape(shape + (3,)), 0, 1)


def metrics(prediction, target, lab_target, indices):
    pred = prediction.reshape(-1, 3)[indices]
    targ = target.reshape(-1, 3)[indices]
    lp = rgb2lab(pred)
    lg = lab_target.reshape(-1, 3)[indices]
    cp, cg = np.sqrt(lp[:, 1] ** 2 + lp[:, 2] ** 2), np.sqrt(lg[:, 1] ** 2 + lg[:, 2] ** 2)
    hue = np.abs(np.arctan2(lp[:, 2], lp[:, 1]) - np.arctan2(lg[:, 2], lg[:, 1]))
    hue = np.minimum(hue, 2 * np.pi - hue) * 180 / np.pi
    mse = float(np.mean((pred - targ) ** 2))
    return {"psnr": 10 * np.log10(1 / mse),
            "deltaE00": float(deltaE_ciede2000(lp, lg).mean()),
            "dL": float(np.mean(np.abs(lp[:, 0] - lg[:, 0]))),
            "dC": float(np.mean(np.abs(cp - cg))),
            "hue_error_deg": float(hue.mean()),
            "saturation_ratio": float(cp.mean() / cg.mean())}


def lpips_rectangles(site, evaluation):
    height, width = evaluation.shape
    candidates = []
    for x, y, w, h in store.support["scenes"][site]["lpips_rectangles"]:
        if x + w > width or y + h > height:
            continue
        if evaluation[y:y + h, x:x + w].all():
            candidates.append((x, y, w, h))
    if len(candidates) > MAX_LPIPS:
        picks = np.linspace(0, len(candidates) - 1, MAX_LPIPS).astype(int)
        candidates = [candidates[i] for i in picks]
    return candidates


def lpips_score(prediction, target, rectangles):
    if not rectangles:
        return float("nan"), 0
    values = [lpips_fn(prediction[y:y + h, x:x + w], target[y:y + h, x:x + w])
              for x, y, w, h in rectangles]
    return float(np.mean(values)), len(values)


def evenly(boxes, count=10):
    if len(boxes) < count:
        return None
    return [boxes[i] for i in np.linspace(0, len(boxes) - 1, count).astype(int)]


rows = []
for site in SITES:
    raw = store.raw(site); mask = store.masks[site]
    height, width = mask.shape
    axis = 0 if height >= width else 1
    length = height if axis == 0 else width
    middle = length // 2
    bands = np.stack([raw["B%d" % b] for b in BANDS], -1).astype(np.float32) / 255
    target = raw["RGB"].astype(np.float32) / 255
    flat_bands = bands.reshape(-1, 3)
    lab_target = rgb2lab(target)
    grid = [(x, y) for y in range(0, height - PATCH + 1, PATCH)
            for x in range(0, width - PATCH + 1, PATCH)
            if mask[y:y + PATCH, x:x + PATCH].mean() >= 0.3]

    regions = []
    for half in (0, 1):
        region = np.zeros_like(mask)
        if axis == 0:
            if half == 0:
                region[:middle] = True
            else:
                region[middle:] = True
        else:
            if half == 0:
                region[:, :middle] = True
            else:
                region[:, middle:] = True
        regions.append(region)

    for fit_half in (0, 1):
        evaluation_half = 1 - fit_half
        fit_region, evaluation_region = regions[fit_half], regions[evaluation_half]
        fit_boxes = [b for b in grid if fit_region[b[1]:b[1] + PATCH, b[0]:b[0] + PATCH].all()]
        evaluation_boxes = [b for b in grid
                            if evaluation_region[b[1]:b[1] + PATCH, b[0]:b[0] + PATCH].all()]
        fit_chosen, evaluation_chosen = evenly(fit_boxes), evenly(evaluation_boxes)
        if fit_chosen is None or evaluation_chosen is None:
            print(site, "half", fit_half, "insufficient patches", flush=True)
            continue

        calibration_cross = np.zeros_like(mask)
        for x, y in fit_chosen:
            calibration_cross[y:y + PATCH, x:x + PATCH] |= mask[y:y + PATCH, x:x + PATCH]
        calibration_within = np.zeros_like(mask)
        calibration_global = np.zeros_like(mask)
        blocked = np.zeros_like(mask)
        for x, y in evaluation_chosen:
            calibration_within[y:y + PATCH, x:x + PATCH] |= mask[y:y + PATCH, x:x + PATCH]
            blocked[max(0, y - HALO):y + PATCH + HALO, max(0, x - HALO):x + PATCH + HALO] = True
        global_chosen = evenly(grid)
        for x, y in global_chosen:
            calibration_global[y:y + PATCH, x:x + PATCH] |= mask[y:y + PATCH, x:x + PATCH]
            blocked[max(0, y - HALO):y + PATCH + HALO, max(0, x - HALO):x + PATCH + HALO] = True
        evaluation = mask & evaluation_region & ~blocked
        indices = np.flatnonzero(evaluation.ravel())
        if len(indices) > EVAL_PIXELS:
            indices = np.random.default_rng(20261028 + 10 * int(site[1:]) + fit_half).choice(
                indices, EVAL_PIXELS, replace=False)
        rectangles = lpips_rectangles(site, evaluation)

        calibrations = {"within": calibration_within, "global": calibration_global,
                        "cross": calibration_cross}
        predictions = {}
        for method, calibration in calibrations.items():
            coefficient = fit(quad_features(bands[calibration]), target[calibration])
            predictions[method] = predict(flat_bands, coefficient, mask.shape)

        for method in ("within", "global", "cross"):
            prediction = predictions[method]
            lpips_value, n_rectangles = lpips_score(prediction, target, rectangles)
            values = {"site": site, "axis": axis, "fit_half": fit_half, "method": method,
                      "calibration_fraction": float(calibrations[method].sum() / mask.sum()),
                      "n_eval": len(indices), **metrics(prediction, target, lab_target, indices),
                      "lpips": lpips_value, "n_lpips": n_rectangles}
            rows.append(values)
            print(site, fit_half, method, "%.3f" % values["psnr"],
                  "%.3f" % values["deltaE00"], "%.4f" % lpips_value, flush=True)
    del lab_target, bands, target, flat_bands

with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)

keys = ["psnr", "deltaE00", "dL", "dC", "hue_error_deg", "saturation_ratio", "lpips"]
summary = []
for site in SITES:
    selected = [r for r in rows if r["site"] == site]
    if not selected:
        continue
    entry = {"site": site, "axis": selected[0]["axis"], "directions": len(selected) // 3}
    for key in keys:
        for method in ("within", "global", "cross"):
            entry[method + "_" + key] = float(np.nanmean(
                [r[key] for r in selected if r["method"] == method]))
    entry["n_eval"] = int(np.mean([r["n_eval"] for r in selected]))
    summary.append(entry)

overall = {"site": "ALL", "axis": "", "directions": int(np.mean([r["directions"] for r in summary]))}
for key in keys:
    for method in ("within", "global", "cross"):
        overall[method + "_" + key] = float(np.nanmean([r[method + "_" + key] for r in summary]))
overall["n_eval"] = int(np.mean([r["n_eval"] for r in summary]))
summary.append(overall)

with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
    writer.writeheader(); writer.writerows(summary)

stats_rows = []
for metric, lower_better in (("psnr", False), ("deltaE00", True), ("lpips", True)):
    for first, second, label in (
            ("within", "global", "same-half -> global-spread"),
            ("global", "cross", "global-spread -> other-half"),
            ("within", "cross", "same-half -> other-half")):
        first_values = np.array([r[first + "_" + metric] for r in summary if r["site"] != "ALL"], float)
        second_values = np.array([r[second + "_" + metric] for r in summary if r["site"] != "ALL"], float)
        finite = np.isfinite(first_values) & np.isfinite(second_values)
        first_values, second_values = first_values[finite], second_values[finite]
        if len(first_values) < 2:
            continue
        differences = second_values - first_values
        n = len(differences)
        critical = float(student_t.ppf(0.975, n - 1))
        rng = np.random.default_rng(20261028)
        bootstrap = rng.choice(differences, (10_000, n), replace=True).mean(1)
        try:
            t_p = float(ttest_rel(second_values, first_values).pvalue)
        except ValueError:
            t_p = float("nan")
        try:
            w_p = float(wilcoxon(differences).pvalue) if n >= 6 else float("nan")
        except ValueError:
            w_p = 1.0
        stats_rows.append({
            "comparison": label, "metric": metric, "n": n,
            "first_mean": float(first_values.mean()), "second_mean": float(second_values.mean()),
            "mean_diff_second_minus_first": float(differences.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "t_p": t_p, "wilcoxon_p": w_p,
            "wins_second": int(np.sum(differences < 0)) if lower_better else int(np.sum(differences > 0)),
            "better": "lower" if lower_better else "higher"})
with STATS.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(stats_rows[0].keys()))
    writer.writeheader(); writer.writerows(stats_rows)

plt.rcParams.update({"font.size": 9})
figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
sites = [r["site"] for r in summary if r["site"] != "ALL"]
positions = np.arange(len(sites))
for axis, key, ylabel in ((axes[0], "psnr", "PSNR (dB)"),
                          (axes[1], "deltaE00", "CIEDE2000")):
    within = [next(r for r in summary if r["site"] == s)["within_" + key] for s in sites]
    spread = [next(r for r in summary if r["site"] == s)["global_" + key] for s in sites]
    cross = [next(r for r in summary if r["site"] == s)["cross_" + key] for s in sites]
    for index in positions:
        axis.plot([index] * 3, [within[index], spread[index], cross[index]],
                  color="#b0b0b0", linewidth=1.2, zorder=1)
    axis.scatter(positions, within, color="#1f4e79", s=34, label="10 patches in same half", zorder=2)
    axis.scatter(positions, spread, color="#117864", s=34, marker="D", label="10 patches spread over scene", zorder=3)
    axis.scatter(positions, cross, color="#c0392b", s=34, marker="s", label="10 patches in other half", zorder=4)
    axis.set_xticks(positions); axis.set_xticklabels(sites)
    axis.set_ylabel(ylabel); axis.grid(alpha=0.25)
mean_within = float(np.mean([r["within_psnr"] for r in summary if r["site"] != "ALL"]))
mean_spread = float(np.mean([r["global_psnr"] for r in summary if r["site"] != "ALL"]))
mean_cross = float(np.mean([r["cross_psnr"] for r in summary if r["site"] != "ALL"]))
axes[0].set_title("PSNR: within %.2f, spread %.2f, other-half %.2f dB"
                  % (mean_within, mean_spread, mean_cross))
mean_within = float(np.mean([r["within_deltaE00"] for r in summary if r["site"] != "ALL"]))
mean_spread = float(np.mean([r["global_deltaE00"] for r in summary if r["site"] != "ALL"]))
mean_cross = float(np.mean([r["cross_deltaE00"] for r in summary if r["site"] != "ALL"]))
axes[1].set_title("CIEDE2000: within %.2f, spread %.2f, other-half %.2f"
                  % (mean_within, mean_spread, mean_cross))
axes[0].legend(fontsize=8)
figure.suptitle("Spatial generalisation of target-site calibration (10 patches, RGB quadratic)", fontsize=11)
figure.tight_layout()
figure.savefig(FIGURE.with_suffix(".png"), bbox_inches="tight")
figure.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
for row in summary:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
print(stats_rows)
print("wrote", OUT)
