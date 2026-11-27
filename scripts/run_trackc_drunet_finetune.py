"""Track C few-shot neural fine-tuning benchmark (DRUNet UNetRes on B1/B3/B4).

Strictly replicates the exact 10-patch target-site calibration geometry and
500,000 held-out pixel evaluation mask of `separated_calibration.py`:
- Input bands: B1, B3, B4 (scaled to [0, 1])
- Target: RGB (scaled to [0, 1])
- 10 evenly spaced 256x256 calibration patches per site (mask fraction >= 0.3)
- Strict exclusion of the 10 calibration patches plus a 16-px halo before scoring
- Identical RNG seed (20261027 + site_id) for 500,000 held-out evaluation pixels

Compares three neural regimes on the exact same held-out pixels as Track C affine/quadratic:
1. DRUNet zero-shot: pre-trained on the fold's 6 training sites (500 updates, batch 8, D4 aug)
2. DRUNet scratch-10: trained from scratch on the 10 target patches (250 updates, batch 4, D4 aug)
3. DRUNet fine-tuned-10: initialized from fold zero-shot weights, fine-tuned on the 10 target patches
   (150 updates, batch 4, lr=5e-5 cosine decay, D4 aug)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
MSWIR_CODE = Path(r"D:\USTC\Gen\MSWIR_RGB")
sys.path.insert(0, str(MSWIR_CODE))
from mswir.chen_drunet.network_unet import UNetRes

SCENES = [f"C{i}" for i in range(1, 11)]
FOLDS = [
    ([3, 4, 5, 6, 7, 10], [2, 9], [1, 8]),
    ([1, 3, 4, 6, 7, 8], [5, 10], [2, 9]),
    ([1, 2, 4, 7, 8, 9], [3, 6], [5, 10]),
    ([1, 2, 5, 8, 9, 10], [4, 7], [3, 6]),
    ([2, 3, 5, 6, 9, 10], [1, 8], [4, 7]),
]
BANDS = (1, 3, 4)
PATCH = 256
HALO = 16
EVAL_PIXELS = 500_000
SEED = 11


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_site(source_root: Path, site: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num = site[1:]
    with Image.open(source_root / site / f"scene{num}.tif") as im:
        rgb = np.asarray(im, dtype=np.float32) / 255.0
    swir = np.stack(
        [np.asarray(Image.open(source_root / site / f"scene{num}_band{b}.tif"), dtype=np.float32) / 255.0
         for b in BANDS],
        axis=-1,
    )
    with Image.open(source_root / "usability_masks" / site / "joint.png") as im:
        mask = np.asarray(im) > 0
    return swir, rgb, mask


def build_drunet(device: torch.device) -> nn.Module:
    model = UNetRes(
        in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4,
        act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose", bias=False,
    ).to(device)
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, gain=0.2)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return model


def d4_augment(x: np.ndarray, y: np.ndarray, m: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k)
        y = np.rot90(y, k)
        m = np.rot90(m, k)
    if rng.random() < 0.5:
        x = np.fliplr(x)
        y = np.fliplr(y)
        m = np.fliplr(m)
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(m)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float()).abs() * mask.float()
    return (err.sum(dim=(1, 2, 3)) / (mask.float().sum(dim=(1, 2, 3)) * 3.0 + 1e-6)).mean()


def sample_valid_windows(mask: np.ndarray, stride: int = 128, min_valid: float = 0.5) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [
        (x, y)
        for y in range(0, h - PATCH + 1, stride)
        for x in range(0, w - PATCH + 1, stride)
        if mask[y : y + PATCH, x : x + PATCH].mean() >= min_valid
    ]


def trackc_split(site: str, mask: np.ndarray):
    grid = [
        (x, y)
        for y in range(0, mask.shape[0] - PATCH + 1, PATCH)
        for x in range(0, mask.shape[1] - PATCH + 1, PATCH)
        if mask[y : y + PATCH, x : x + PATCH].mean() >= 0.3
    ]
    chosen = [grid[i] for i in np.linspace(0, len(grid) - 1, 10).astype(int)]
    calibration = np.zeros_like(mask)
    blocked = np.zeros_like(mask)
    for x, y in chosen:
        calibration[y : y + PATCH, x : x + PATCH] |= mask[y : y + PATCH, x : x + PATCH]
        blocked[max(0, y - HALO) : y + PATCH + HALO, max(0, x - HALO) : x + PATCH + HALO] = True
    evaluation = mask & ~blocked
    eval_idx = np.flatnonzero(evaluation.ravel())
    if len(eval_idx) > EVAL_PIXELS:
        eval_idx = np.random.default_rng(20261027 + int(site[1:])).choice(eval_idx, EVAL_PIXELS, replace=False)
    cal_frac = float(calibration.sum() / mask.sum())
    return chosen, calibration, eval_idx, cal_frac


def train_steps(
    model: nn.Module,
    sample_fn,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)

    for _ in range(steps):
        xs, ys, ms = [], [], []
        for _ in range(batch_size):
            x, y, m = sample_fn(rng)
            x, y, m = d4_augment(x, y, m, rng)
            xs.append(x.transpose(2, 0, 1))
            ys.append(y.transpose(2, 0, 1))
            ms.append(m[None, :, :].astype(np.float32))
        xt = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
        yt = torch.from_numpy(np.stack(ys)).to(device, non_blocking=True)
        mt = torch.from_numpy(np.stack(ms)).to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt)
            loss = masked_l1(pred, yt, mt)
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def infer_mosaic(model: nn.Module, swir: np.ndarray, device: torch.device, stride: int = 128) -> np.ndarray:
    model.eval()
    h, w, _ = swir.shape
    ys = list(range(0, max(1, h - PATCH + 1), stride))
    xs = list(range(0, max(1, w - PATCH + 1), stride))
    if ys[-1] != h - PATCH:
        ys.append(h - PATCH)
    if xs[-1] != w - PATCH:
        xs.append(w - PATCH)

    win1d = np.sin(np.linspace(0.0, math.pi, PATCH, dtype=np.float32)) ** 2
    win1d = np.maximum(win1d, 1e-3)
    win2d = np.outer(win1d, win1d)[..., None]

    acc = np.zeros((h, w, 3), dtype=np.float32)
    wgt = np.zeros((h, w, 1), dtype=np.float32)

    coords = [(x, y) for y in ys for x in xs]
    bs = 16
    for i in range(0, len(coords), bs):
        batch_c = coords[i : i + bs]
        batch_x = np.stack([swir[y : y + PATCH, x : x + PATCH].transpose(2, 0, 1) for x, y in batch_c])
        xt = torch.from_numpy(batch_x).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(xt).float().clamp(0.0, 1.0).cpu().numpy()
        for (x, y), pred in zip(batch_c, out):
            p = pred.transpose(1, 2, 0)
            acc[y : y + PATCH, x : x + PATCH] += p * win2d
            wgt[y : y + PATCH, x : x + PATCH] += win2d
    return np.clip(acc / np.maximum(wgt, 1e-6), 0.0, 1.0)


def score_prediction(pred: np.ndarray, target: np.ndarray, lab_target: np.ndarray, eval_idx: np.ndarray) -> dict:
    p = pred.reshape(-1, 3)[eval_idx]
    t = target.reshape(-1, 3)[eval_idx]
    mse = float(np.mean((p - t) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    mae = float(np.mean(np.abs(p - t)))
    lp = rgb2lab(p)
    lg = lab_target.reshape(-1, 3)[eval_idx]
    de00 = float(deltaE_ciede2000(lp, lg).mean())
    return {"psnr": round(psnr, 3), "deltaE00": round(de00, 3), "mae": round(mae, 4)}


def verify_trackc_alignment(source_root: Path, sites_data: dict) -> tuple[float, float]:
    """Quick sanity check that our evaluation indices reproduce Track C affine (14.113 / 16.502 dB)."""
    cross_psnrs, target_psnrs = [], []
    for fold_idx, (tr_ids, _, te_ids) in enumerate(FOLDS, 1):
        xs, ys = [], []
        for sid in tr_ids:
            sw, rg, mk = sites_data[f"C{sid}"]
            yy, xx = np.indices(mk.shape)
            sel = mk & (yy % 16 == 0) & (xx % 16 == 0)
            xs.append(sw[sel])
            ys.append(rg[sel])
        x_tr = np.concatenate(xs)
        y_tr = np.concatenate(ys)
        w_cross = np.linalg.lstsq(np.concatenate([x_tr, np.ones((len(x_tr), 1), np.float32)], 1), y_tr, rcond=None)[0]
        for sid in te_ids:
            site = f"C{sid}"
            sw, rg, mk = sites_data[site]
            _, cal_mask, eval_idx, _ = trackc_split(site, mk)
            flat_sw = sw.reshape(-1, 3)
            pred_c = np.clip(np.concatenate([flat_sw, np.ones((len(flat_sw), 1), np.float32)], 1) @ w_cross, 0, 1)
            mse_c = float(np.mean((pred_c[eval_idx] - rg.reshape(-1, 3)[eval_idx]) ** 2))
            cross_psnrs.append(10.0 * np.log10(1.0 / mse_c))

            x_cal = sw[cal_mask]
            y_cal = rg[cal_mask]
            w_t = np.linalg.lstsq(np.concatenate([x_cal, np.ones((len(x_cal), 1), np.float32)], 1), y_cal, rcond=None)[0]
            pred_t = np.clip(np.concatenate([flat_sw, np.ones((len(flat_sw), 1), np.float32)], 1) @ w_t, 0, 1)
            mse_t = float(np.mean((pred_t[eval_idx] - rg.reshape(-1, 3)[eval_idx]) ** 2))
            target_psnrs.append(10.0 * np.log10(1.0 / mse_t))
    c_mean, t_mean = float(np.mean(cross_psnrs)), float(np.mean(target_psnrs))
    assert abs(c_mean - 14.113) < 0.01 and abs(t_mean - 16.502) < 0.01, f"Track C eval mismatch: {c_mean}, {t_mean}"
    return c_mean, t_mean


def run_benchmark(source_root: Path, pretrain_steps: int, ft_steps: int, scratch_steps: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU required"
    t_start = time.time()

    sites_data = {s: load_site(source_root, s) for s in SCENES}
    c_chk, t_chk = verify_trackc_alignment(source_root, sites_data)
    print(f"Verified exact Track C evaluation mask match: cross_affine={c_chk:.3f} dB, target_affine={t_chk:.3f} dB", flush=True)

    train_windows = {s: sample_valid_windows(sites_data[s][2], stride=128, min_valid=0.5) for s in SCENES}
    rows: list[dict] = []

    for fold_idx, (tr_ids, _, te_ids) in enumerate(FOLDS, 1):
        tr_sites = [f"C{i}" for i in tr_ids]
        te_sites = [f"C{i}" for i in te_ids]

        # 1. Pre-train DRUNet on the 6 training sites (site-balanced sampling)
        t0 = time.time()
        set_seed(SEED + fold_idx * 100)
        base_model = build_drunet(device)

        def sample_source(rng: np.random.Generator):
            s = tr_sites[int(rng.integers(0, len(tr_sites)))]
            x, y = train_windows[s][int(rng.integers(0, len(train_windows[s])))]
            sw, rg, mk = sites_data[s]
            return sw[y : y + PATCH, x : x + PATCH], rg[y : y + PATCH, x : x + PATCH], mk[y : y + PATCH, x : x + PATCH]

        train_steps(base_model, sample_source, steps=pretrain_steps, batch_size=8, lr=1e-4, device=device, seed=SEED + fold_idx * 100)
        base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        print(f"Fold {fold_idx} pre-training ({pretrain_steps} steps on {tr_sites}) done in {time.time()-t0:.1f}s", flush=True)

        # 2. Evaluate zero-shot, scratch-10, and fine-tuned-10 on each held-out test site
        for site in te_sites:
            sw, rg, mk = sites_data[site]
            chosen, _, eval_idx, cal_frac = trackc_split(site, mk)
            lab_target = rgb2lab(rg)

            # (a) Zero-shot DRUNet
            base_model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
            pred_zs = infer_mosaic(base_model, sw, device)
            m_zs = score_prediction(pred_zs, rg, lab_target, eval_idx)
            rows.append({"site": site, "fold": fold_idx, "method": "drunet_zeroshot", "cal_frac": round(cal_frac, 4), **m_zs})

            def sample_target_10(rng: np.random.Generator):
                x, y = chosen[int(rng.integers(0, len(chosen)))]
                return sw[y : y + PATCH, x : x + PATCH], rg[y : y + PATCH, x : x + PATCH], mk[y : y + PATCH, x : x + PATCH]

            # (b) Few-shot fine-tuned DRUNet (from zero-shot weights on 10 target patches ONLY)
            ft_model = build_drunet(device)
            ft_model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
            train_steps(ft_model, sample_target_10, steps=ft_steps, batch_size=4, lr=5e-5, device=device, seed=SEED + int(site[1:]) * 17)
            pred_ft = infer_mosaic(ft_model, sw, device)
            m_ft = score_prediction(pred_ft, rg, lab_target, eval_idx)
            rows.append({"site": site, "fold": fold_idx, "method": "drunet_ft10", "cal_frac": round(cal_frac, 4), **m_ft})
            del ft_model

            # (c) Target-only scratch DRUNet (random init on 10 target patches ONLY)
            set_seed(SEED + int(site[1:]) * 31)
            sc_model = build_drunet(device)
            train_steps(sc_model, sample_target_10, steps=scratch_steps, batch_size=4, lr=1e-4, device=device, seed=SEED + int(site[1:]) * 31)
            pred_sc = infer_mosaic(sc_model, sw, device)
            m_sc = score_prediction(pred_sc, rg, lab_target, eval_idx)
            rows.append({"site": site, "fold": fold_idx, "method": "drunet_scratch10", "cal_frac": round(cal_frac, 4), **m_sc})
            del sc_model
            torch.cuda.empty_cache()

            print(
                f"  {site} (fold {fold_idx}, cal={cal_frac*100:.1f}%): "
                f"ZS={m_zs['psnr']:.3f} dB (dE={m_zs['deltaE00']:.2f}) | "
                f"Scratch10={m_sc['psnr']:.3f} dB (dE={m_sc['deltaE00']:.2f}) | "
                f"FT10={m_ft['psnr']:.3f} dB (dE={m_ft['deltaE00']:.2f})",
                flush=True,
            )

    # Aggregate 10-site equal means
    summary_by_method = {}
    for meth in ("drunet_zeroshot", "drunet_scratch10", "drunet_ft10"):
        sub = [r for r in rows if r["method"] == meth]
        summary_by_method[meth] = {
            "sites": len(sub),
            "psnr": round(float(np.mean([r["psnr"] for r in sub])), 3),
            "deltaE00": round(float(np.mean([r["deltaE00"] for r in sub])), 3),
            "mae": round(float(np.mean([r["mae"] for r in sub])), 4),
        }

    return {
        "protocol": {
            "bands": "B1/B3/B4 (T134)",
            "architecture": "DRUNet (UNetRes [64,128,256,512], nb=4, masked L1, D4 aug, bf16)",
            "seed": SEED,
            "pretrain_steps": pretrain_steps,
            "ft_steps": ft_steps,
            "scratch_steps": scratch_steps,
            "eval_pixels_per_site": EVAL_PIXELS,
            "halo_px": HALO,
            "verified_trackc_cross_affine_psnr": round(c_chk, 3),
            "verified_trackc_target_affine_psnr": round(t_chk, 3),
            "wall_time_sec": round(time.time() - t_start, 1),
            "gpu": torch.cuda.get_device_name(0),
        },
        "summary": summary_by_method,
        "per_site": rows,
    }


def write_outputs(data: dict) -> None:
    out_json = REPO_ROOT / "figures/source_data/trackc_drunet_finetune.json"
    out_csv = REPO_ROOT / "figures/source_data/trackc_drunet_finetune.csv"
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data["per_site"][0].keys()))
        w.writeheader()
        w.writerows(data["per_site"])

    gen = REPO_ROOT / "generated"
    s = data["summary"]
    zs, sc, ft = s["drunet_zeroshot"], s["drunet_scratch10"], s["drunet_ft10"]
    gain_ft_vs_zs = ft["psnr"] - zs["psnr"]
    gain_ft_vs_quad = ft["psnr"] - 16.715

    macros = [
        f"\\newcommand{{\\TrackCDrZsPsnr}}{{{zs['psnr']:.3f}}}",
        f"\\newcommand{{\\TrackCDrZsDe}}{{{zs['deltaE00']:.3f}}}",
        f"\\newcommand{{\\TrackCDrScPsnr}}{{{sc['psnr']:.3f}}}",
        f"\\newcommand{{\\TrackCDrScDe}}{{{sc['deltaE00']:.3f}}}",
        f"\\newcommand{{\\TrackCDrFtPsnr}}{{{ft['psnr']:.3f}}}",
        f"\\newcommand{{\\TrackCDrFtDe}}{{{ft['deltaE00']:.3f}}}",
        f"\\newcommand{{\\TrackCDrFtGainZs}}{{{gain_ft_vs_zs:+.3f}}}",
        f"\\newcommand{{\\TrackCDrFtGainQuad}}{{{gain_ft_vs_quad:+.3f}}}",
    ]
    (gen / "trackc_drunet_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    # Combined Track C table (pixel-level + neural few-shot on identical 500,000 held-out pixels)
    zh_tbl = "\n".join([
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"拟合/适应条件（固定B1/B3/B4输入） & PSNR/dB$\uparrow$ & \dE{}$\downarrow$\\",
        r"\midrule",
        r"训练地点仿射（零样本） & 14.113 & 19.719\\",
        f"DRUNet零样本（6训练地点预训练） & {zs['psnr']:.3f} & {zs['deltaE00']:.3f}\\\\",
        r"\midrule",
        r"目标地点仿射（10块拟合） & 16.502 & 12.864\\",
        r"目标地点二次（10块拟合） & 16.715 & 12.089\\",
        f"DRUNet从头训练（仅目标10块） & {sc['psnr']:.3f} & {sc['deltaE00']:.3f}\\\\",
        f"DRUNet少样本微调（预训练+目标10块） & {ft['psnr']:.3f} & {ft['deltaE00']:.3f}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    en_tbl = "\n".join([
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Calibration / adaptation regime (fixed B1/B3/B4) & PSNR/dB$\uparrow$ & \dE{}$\downarrow$\\",
        r"\midrule",
        r"Training-site affine (zero-shot) & 14.113 & 19.719\\",
        f"DRUNet zero-shot (6-site pre-trained) & {zs['psnr']:.3f} & {zs['deltaE00']:.3f}\\\\",
        r"\midrule",
        r"Target-site affine (10 patches) & 16.502 & 12.864\\",
        r"Target-site quadratic (10 patches) & 16.715 & 12.089\\",
        f"DRUNet from scratch (10 target patches) & {sc['psnr']:.3f} & {sc['deltaE00']:.3f}\\\\",
        f"DRUNet few-shot fine-tuned (pre-trained + 10 patches) & {ft['psnr']:.3f} & {ft['deltaE00']:.3f}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    (gen / "trackc_drunet_zh.tex").write_text(zh_tbl, encoding="utf-8")
    (gen / "trackc_drunet_en.tex").write_text(en_tbl, encoding="utf-8")

    # Also plot per-site comparison figure (ZH + EN)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read reference affine/quad per-site PSNR from separated_calibration.csv
    sep_csv = REPO_ROOT / "figures/source_data/results_20261109/separated_calibration.csv"
    ref_psnr = {}
    with sep_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ("cross_rgb", "target_rgb_quad"):
                ref_psnr[(r["site"], r["method"])] = float(r["psnr"])

    ft_map = {r["site"]: r["psnr"] for r in data["per_site"] if r["method"] == "drunet_ft10"}
    sc_map = {r["site"]: r["psnr"] for r in data["per_site"] if r["method"] == "drunet_scratch10"}
    zs_map = {r["site"]: r["psnr"] for r in data["per_site"] if r["method"] == "drunet_zeroshot"}

    xs = np.arange(len(SCENES))
    w = 0.21
    for lang in ("zh", "en"):
        if lang == "zh":
            plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                                 "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8})
            out_fig = REPO_ROOT / "figures/zh/trackc_drunet_zh.pdf"
            labels = ["DRUNet 零样本", "目标地点二次映射", "DRUNet 目标10块从头训练", "DRUNet 10块少样本微调"]
            ylabel = "保留评价像素 PSNR / dB"
            title = "目标地点10图块少样本适应：像素级校准 vs. DRUNet微调（逐地点对比）"
        else:
            plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],
                                 "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8})
            out_fig = REPO_ROOT / "figures/en/trackc_drunet_en.pdf"
            labels = ["DRUNet zero-shot", "Target quadratic", "DRUNet scratch (10 patches)", "DRUNet fine-tuned (10 patches)"]
            ylabel = "Held-out evaluation PSNR / dB"
            title = "Ten-patch target-site adaptation: pixel-level calibration vs. DRUNet fine-tuning"

        fig, ax = plt.subplots(figsize=(8.6, 3.2), layout="constrained")
        v_zs = [zs_map[sc_name] for sc_name in SCENES]
        v_qd = [ref_psnr[(sc_name, "target_rgb_quad")] for sc_name in SCENES]
        v_sc = [sc_map[sc_name] for sc_name in SCENES]
        v_ft = [ft_map[sc_name] for sc_name in SCENES]

        ax.bar(xs - 1.5 * w, v_zs, width=w, color="#9fb3c8", label=f"{labels[0]} ({zs['psnr']:.2f} dB)")
        ax.bar(xs - 0.5 * w, v_qd, width=w, color="#5b8fb9", label=f"{labels[1]} (16.72 dB)")
        ax.bar(xs + 0.5 * w, v_sc, width=w, color="#4f9f9a", label=f"{labels[2]} ({sc['psnr']:.2f} dB)")
        ax.bar(xs + 1.5 * w, v_ft, width=w, color="#1f5553", label=f"{labels[3]} ({ft['psnr']:.2f} dB)")
        ax.set_xticks(xs, SCENES)
        ax.set_ylabel(ylabel)
        ax.set_ylim(8, max(max(v_ft), max(v_qd), max(v_sc)) + 3.0)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v3"))
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--ft-steps", type=int, default=150)
    parser.add_argument("--scratch-steps", type=int, default=250)
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()

    out_json = REPO_ROOT / "figures/source_data/trackc_drunet_finetune.json"
    if args.from_json:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        data = run_benchmark(args.source_root, args.pretrain_steps, args.ft_steps, args.scratch_steps)

    write_outputs(data)
    print(json.dumps({"protocol": data["protocol"], "summary": data["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
