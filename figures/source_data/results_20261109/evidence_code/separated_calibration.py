"""Separated lightness/chroma calibration at a fixed 10-patch target-site budget.

All methods use the same 10 spatially spread 256x256 patches and the same
evaluation mask (calibration patches plus a 16 px halo excluded).  Predictions
are scored and released one at a time to keep memory bounded.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000

sys.path.insert(0, r"D:\dh\MSWIR_RGB")
from mswir.data import SceneStore
from mswir.protocols import FOLDS

P = Path(r"D:\dh\MSWIR_RGB")
RUN = P / "runs/three_channel_v2_16680_b8/formal"
OUT = RUN / "report/calibration/separated_calibration.csv"
SUMMARY = RUN / "report/calibration/separated_calibration_summary.csv"
BANDS = (1, 3, 4)
PATCH, HALO = 256, 16
EVAL_PIXELS = 500_000

METHODS = ["cross_rgb", "target_rgb", "target_rgb_quad", "target_lab_lin",
           "target_lab_quad", "deep_raw", "deep_rgb3", "deep_rgb_quad",
           "deep_tone_chroma", "deepL_bandC", "bandL_deepC"]

store = SceneStore(P/"data/release", P/"data/manifest.json", P/"data/support", cache_sites=1)


def quad_features(x):
    columns = [x, x ** 2]
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            columns.append((x[:, i] * x[:, j])[:, None])
    columns.append(np.ones((len(x), 1), np.float32))
    return np.concatenate(columns, 1)


def lin_features(x):
    return np.concatenate([x, np.ones((len(x), 1), np.float32)], 1)


def fit(features, target):
    return np.linalg.lstsq(features, target, rcond=None)[0]


def lab_chroma_features(lab, quadratic):
    a, b, l = lab[:, 1:2], lab[:, 2:3], lab[:, 0:1]
    if not quadratic:
        return np.concatenate([a, b, l, np.ones((len(lab), 1), np.float32)], 1)
    return np.concatenate([a, b, l, a ** 2, b ** 2, l ** 2, a * b, a * l, b * l,
                           np.ones((len(lab), 1), np.float32)], 1)


def metrics(prediction, target, lab_target, evaluation_indices):
    pred = prediction.reshape(-1, 3)[evaluation_indices]
    targ = target.reshape(-1, 3)[evaluation_indices]
    lp = rgb2lab(pred)
    lg = lab_target.reshape(-1, 3)[evaluation_indices]
    cp, cg = np.sqrt(lp[:, 1] ** 2 + lp[:, 2] ** 2), np.sqrt(lg[:, 1] ** 2 + lg[:, 2] ** 2)
    hue = np.abs(np.arctan2(lp[:, 2], lp[:, 1]) - np.arctan2(lg[:, 2], lg[:, 1]))
    hue = np.minimum(hue, 2 * np.pi - hue) * 180 / np.pi
    mse = float(np.mean((pred - targ) ** 2))
    psnr = 10 * np.log10(1 / mse)
    fix_chroma = np.clip(lab2rgb(np.stack([lp[:, 0], lg[:, 1], lg[:, 2]], -1)), 0, 1)
    fix_luma = np.clip(lab2rgb(np.stack([lg[:, 0], lp[:, 1], lp[:, 2]], -1)), 0, 1)

    def psnr_of(image):
        return 10 * np.log10(1 / np.mean((image - targ) ** 2))
    return {"psnr": psnr,
            "deltaE00": float(deltaE_ciede2000(lp, lg).mean()),
            "dL": float(np.mean(np.abs(lp[:, 0] - lg[:, 0]))),
            "dC": float(np.mean(np.abs(cp - cg))),
            "hue_error_deg": float(hue.mean()),
            "saturation_ratio": float(cp.mean() / cg.mean()),
            "gain_fix_chroma": psnr_of(fix_chroma) - psnr,
            "gain_fix_luma": psnr_of(fix_luma) - psnr}


def cross_fit(fold):
    xs, ys = [], []
    for site in ["C%d" % s for s in FOLDS[fold - 1][0]]:
        raw = store.raw(site); mask = store.masks[site]
        yy, xx = np.indices(mask.shape)
        selected = mask & (yy % 16 == 0) & (xx % 16 == 0)
        bands = np.stack([raw["B%d" % b] for b in BANDS], -1).astype(np.float32) / 255
        xs.append(bands[selected]); ys.append((raw["RGB"].astype(np.float32) / 255)[selected])
    return fit(lin_features(np.concatenate(xs)), np.concatenate(ys))


rows = []
cross_cache = {}
sites = sys.argv[1:] or ["C%d" % i for i in range(1, 11)]
for site in sites:
    fold = next(i for i, (tr, va, te) in enumerate(FOLDS, 1) if int(site[1:]) in te)
    raw = store.raw(site); mask = store.masks[site]
    grid = [(x, y) for y in range(0, mask.shape[0] - PATCH + 1, PATCH)
            for x in range(0, mask.shape[1] - PATCH + 1, PATCH)
            if mask[y:y + PATCH, x:x + PATCH].mean() >= 0.3]
    chosen = [grid[i] for i in np.linspace(0, len(grid) - 1, 10).astype(int)]
    calibration = np.zeros_like(mask)
    blocked = np.zeros_like(mask)
    for x, y in chosen:
        calibration[y:y + PATCH, x:x + PATCH] |= mask[y:y + PATCH, x:x + PATCH]
        blocked[max(0, y - HALO):y + PATCH + HALO, max(0, x - HALO):x + PATCH + HALO] = True
    evaluation = mask & ~blocked
    evaluation_indices = np.flatnonzero(evaluation.ravel())
    if len(evaluation_indices) > EVAL_PIXELS:
        evaluation_indices = np.random.default_rng(20261027 + int(site[1:])).choice(
            evaluation_indices, EVAL_PIXELS, replace=False)
    fraction = float(calibration.sum() / mask.sum())

    bands = np.stack([raw["B%d" % b] for b in BANDS], -1).astype(np.float32) / 255
    target = raw["RGB"].astype(np.float32) / 255
    flat_bands = bands.reshape(-1, 3)
    y_cal = target[calibration]
    lab_target = rgb2lab(target)
    lab_cal = lab_target[calibration]

    def record(method, prediction):
        rows.append({"site": site, "fold": fold, "method": method,
                     "calibration_fraction": fraction,
                     "n_eval": len(evaluation_indices),
                     **metrics(prediction.astype(np.float32), target, lab_target, evaluation_indices)})
        print(site, method, "%.3f" % rows[-1]["psnr"], flush=True)
        print(site, method, "%.3f" % rows[-1]["psnr"], flush=True)

    if fold not in cross_cache:
        cross_cache[fold] = cross_fit(fold)
    record("cross_rgb", np.clip((lin_features(flat_bands) @ cross_cache[fold]).reshape(mask.shape + (3,)), 0, 1))

    x_cal = bands[calibration]
    record("target_rgb", np.clip((lin_features(flat_bands) @ fit(lin_features(x_cal), y_cal)).reshape(mask.shape + (3,)), 0, 1))
    record("target_rgb_quad", np.clip((quad_features(flat_bands) @ fit(quad_features(x_cal), y_cal)).reshape(mask.shape + (3,)), 0, 1))

    lab_band = None
    for quadratic, name in ((False, "target_lab_lin"), (True, "target_lab_quad")):
        feature_fn = quad_features if quadratic else lin_features
        l_coef = fit(feature_fn(x_cal), lab_cal[:, 0:1])
        ab_coef = fit(feature_fn(x_cal), lab_cal[:, 1:3])
        l_pred = (feature_fn(flat_bands) @ l_coef).reshape(mask.shape)
        ab_pred = (feature_fn(flat_bands) @ ab_coef).reshape(mask.shape + (2,))
        lab_pred = np.clip(np.concatenate([l_pred[..., None], ab_pred], -1),
                           [0, -128, -128], [100, 127, 127])
        prediction = np.clip(lab2rgb(lab_pred), 0, 1)
        record(name, prediction)
        if name == "target_lab_quad":
            lab_band = rgb2lab(prediction)

    deep = np.clip(np.asarray(np.load(RUN/"tasks"/("EX2_drunet_T245_f%d_s11" % fold)
                                      /"evaluation/predictions"/(site+".npy"), mmap_mode="r")), 0, 1)
    record("deep_raw", deep)
    flat_deep = deep.reshape(-1, 3)
    deep_cal = deep[calibration]
    record("deep_rgb3", np.clip((lin_features(flat_deep) @ fit(lin_features(deep_cal), y_cal)).reshape(mask.shape + (3,)), 0, 1))
    record("deep_rgb_quad", np.clip((quad_features(flat_deep) @ fit(quad_features(deep_cal), y_cal)).reshape(mask.shape + (3,)), 0, 1))

    lab_deep = rgb2lab(deep)
    lab_deep_cal = lab_deep[calibration]
    tone = np.polyfit(lab_deep_cal[:, 0], lab_cal[:, 0], 2)
    tone_l = np.clip(np.polyval(tone, lab_deep[..., 0]), 0, 100)
    chroma_coef = fit(lab_chroma_features(lab_deep_cal, True), lab_cal[:, 1:3])
    chroma = (lab_chroma_features(lab_deep.reshape(-1, 3), True) @ chroma_coef).reshape(mask.shape + (2,))
    separated = np.concatenate([tone_l[..., None], chroma], -1)
    record("deep_tone_chroma", np.clip(lab2rgb(np.clip(separated, [0, -128, -128], [100, 127, 127])), 0, 1))

    record("deepL_bandC", np.clip(lab2rgb(np.stack([lab_deep[..., 0], lab_band[..., 1], lab_band[..., 2]], -1)), 0, 1))
    record("bandL_deepC", np.clip(lab2rgb(np.stack([lab_band[..., 0], lab_deep[..., 1], lab_deep[..., 2]], -1)), 0, 1))
    del deep, lab_deep, lab_band, lab_target

with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)

keys = ["psnr", "deltaE00", "dL", "dC", "hue_error_deg", "saturation_ratio",
        "gain_fix_chroma", "gain_fix_luma"]
summary = []
for method in METHODS:
    selected = [r for r in rows if r["method"] == method]
    summary.append({"method": method, "sites": len(selected),
                    **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
    writer.writeheader(); writer.writerows(summary)

for row in summary:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
print("wrote", OUT)