"""Diagnostic for the scene folds: where do trivial baselines sit on the held-out windows, and does matching the
input intensity to the training sites change the held-out result? Transformed inputs go to outputs/diag/."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("D:/dh/MSWIR_RGB/bench_20261124")


def load(d):
    files = sorted(Path(d).glob("*.png"))
    return files, np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def moments(x):
    s, s2, per_m, per_s = np.zeros(3), np.zeros(3), np.zeros((len(x), 3)), np.zeros((len(x), 3))
    for i, im in enumerate(x):
        f = im.reshape(-1, 3).astype(np.float64)
        s += f.sum(0)
        s2 += (f * f).sum(0)
        per_m[i], per_s[i] = f.mean(0), f.std(0)
    n = x.shape[0] * x.shape[1] * x.shape[2]
    m = s / n
    return m, np.sqrt(np.maximum(s2 / n - m * m, 0)), per_m, per_s


def by_scene(names, per_m):
    scenes = np.array([f.name.split("_")[0] for f in names])
    return {s: per_m[scenes == s].mean(0).round(1).tolist() for s in sorted(set(scenes))}


def psnr_row(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / mse)


def save(names, te, out, m_from, s_from, m_to, s_to):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(names):
        m_from_i = m_from[i] if m_from.ndim == 2 else m_from
        s_from_i = s_from[i] if s_from.ndim == 2 else s_from
        im = (te[i].astype(np.float64) - m_from_i) * (s_to / np.maximum(s_from_i, 1.0)) + m_to
        Image.fromarray(np.clip(np.rint(im), 0, 255).astype(np.uint8)).save(out / f.name)


def main(fold):
    d = ROOT / "datasets" / ("scene_fold%s" % fold)
    tr_names, tr = load(d / "train_A")
    m_tr, s_tr, pm_tr, _ = moments(tr)
    scene_in_tr = by_scene(tr_names, pm_tr)
    img_range_tr = (pm_tr.min(0), pm_tr.max(0))
    del tr
    _, tr_b = load(d / "train_B")
    m_trb, _, pm_trb, _ = moments(tr_b)
    scene_out_tr = by_scene(tr_names, pm_trb)
    del tr_b
    te_names, te = load(d / "test_A")
    m_te, s_te, pm_te, ps_te = moments(te)
    _, gt = load(d / "test_B")
    m_gtb, _, pm_gtb, _ = moments(gt)
    r = lambda v: np.round(v, 1).tolist()
    print("fold", fold)
    print("input mean/std  train", r(m_tr), r(s_tr), " held-out", r(m_te), r(s_te))
    print("target mean     train", r(m_trb), " held-out", r(m_gtb))
    print("per-scene input mean  train", scene_in_tr)
    print("per-scene input mean  held-out", by_scene(te_names, pm_te))
    print("per-scene target mean train", scene_out_tr)
    print("per-scene target mean held-out", by_scene(te_names, pm_gtb))
    print("per-image input mean range  train", r(img_range_tr[0]), r(img_range_tr[1]),
          " held-out", r(pm_te.min(0)), r(pm_te.max(0)))
    save(te_names, te, ROOT / "outputs/diag" / ("e4_scene%s_shift_global" % fold) / "test_A", m_te, s_te, m_tr, s_tr)
    save(te_names, te, ROOT / "outputs/diag" / ("e4_scene%s_shift_perimg" % fold) / "test_A",
         pm_te, ps_te, m_tr, s_tr)
    gray_v = [psnr_row(np.repeat(te[i].mean(2, keepdims=True), 3, 2), gt[i]) for i in range(len(te))]
    const_v = [psnr_row(np.broadcast_to(np.rint(m_trb).astype(np.uint8), gt[i].shape), gt[i]) for i in range(len(te))]
    for name, v in (("input as gray", gray_v), ("constant training-mean colour", const_v)):
        print("baseline %-32s PSNR mean %.2f median %.2f" % (name, np.mean(v), np.median(v)))


if __name__ == "__main__":
    main(sys.argv[1])
