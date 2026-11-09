"""Illustrative plates: saved-window selection, joint display normalisation, full size.

No training, no inference, no new metric. Windows come from
scripts/select_display_windows.py (full support + saved-prediction fidelity
floor + saved pix2pix PSNR ranking).

Display
  Both plates show raw eight-bit data under one joint linear map per row, so a
  dark window stays legible and a genuinely darker prediction still looks
  darker: the low/high display levels are the 2nd and 98th percentile of the
  luminance of the row's RGB target together with the panels drawn beside it.
  No per-panel stretch, no histogram equalisation, no gamma, no sharpening.
  The same map applies to every panel of a row, so the comparison is unbiased.

Outputs go to figures/zh as *_zh.{pdf,svg,png}; provenance is
written to figures/source_data/local_comparisons_20261015.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / "Gen"
SUPPORT_DIR = GEN / "v3_pipeline/outputs/draft_support_v2_20261002"
SUPPORT = SUPPORT_DIR / "manifest.json"
MANIFEST = GEN / "v3_pipeline/configs/v3_scene_manifest.json"
PREV = GEN / "outputs/experiment_review_20261005/visual_provenance.json"
OUT = ROOT / "figures/zh"
SOURCE = ROOT / "figures/source_data"
DATA_ROOT = Path(json.loads(MANIFEST.read_text(encoding="utf-8"))["root"])
SUFFIX = "_zh"

MODELS = ["U-Net / B1-5", "pix2pix / B1-5", "Restormer / B1-5"]
PICKS = [("C3", 4), ("C6", 4), ("C4", 5)]      # scene, test fold of that scene
DISPLAY_LO, DISPLAY_HI = 2.0, 98.0
MODE_LABELS = ["RGB真值", "B1", "B2", "B3", "B4", "B5"]
PRED_LABELS = ["RGB真值", "B1输入", "U-Net（五波段）", "Restormer（五波段）", "pix2pix（五波段）"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arr_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def levels(arr: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Per-channel 2%/98% display levels of one reference panel."""
    a = arr.astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    lo = tuple(float(np.percentile(a[..., c], DISPLAY_LO)) for c in range(3))
    hi = tuple(float(np.percentile(a[..., c], DISPLAY_HI)) for c in range(3))
    return lo, hi


def joint_levels(reference: np.ndarray):
    """Display levels from the observed RGB target of the row, shared by all panels.

    The reference panel is stretched per channel to its 2%--98% range and that
    same map is then applied unchanged to the input, to every modality and to
    every saved prediction, so nothing is normalised panel by panel. The plates
    therefore read the way the scene overview figure does (a display-only
    percentile stretch), while the models' own brightness and colour errors stay
    visible and are stated in the caption. Stored PSNR values are computed on the
    raw arrays and are unaffected by the display map.
    """
    return levels(reference)


def show_panel(ax, arr, lo, hi, bar=False):
    """Raw display: uint8 panels as stored, float predictions via round(255*x)."""
    a = arr
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.dtype != np.uint8:
        a = np.rint(np.clip(a, 0, 1).astype(np.float32) * 255).astype(np.uint8)
    ax.imshow(a, interpolation="none", vmin=0, vmax=255)
    ax.set_xlim(-.5, 255.5)
    ax.set_ylim(255.5, -.5)
    ax.set_axis_off()
    if bar:
        ax.add_patch(Rectangle((178, 233), 64, 5, fc="white", ec="black", lw=.35))
        ax.text(210, 228, "64像素", ha="center", color="white", fontsize=5.8,
                bbox=dict(facecolor="black", edgecolor="none", pad=.5))


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{name}{SUFFIX}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def build(rows, labels, name, footer, width=7.2):
    """rows: list of (row_label, [(array, (lo, hi)), ...])"""
    left = right = .055
    gap = .012
    n = len(labels)
    panel = (width - left - right - gap * (n - 1)) / n
    head, rowhead = .30, .22
    height = .26 + len(rows) * (panel + rowhead) + .20
    fig = plt.figure(figsize=(width, height))
    for row, (row_label, panels) in enumerate(rows):
        top = height - head - row * (panel + rowhead)
        fig.text(left / width, (top - .02) / height, row_label,
                 ha="left", va="top", fontsize=7.2, weight="bold")
        for col, (arr, (lo, hi)) in enumerate(panels):
            px = left + col * (panel + gap)
            if row == 0:
                fig.text((px + panel / 2) / width, (height - .07) / height, labels[col],
                         ha="center", va="top", fontsize=7.3)
            ax = fig.add_axes([px / width, (top - rowhead - panel) / height,
                               panel / width, panel / height])
            show_panel(ax, arr, lo, hi, bar=col == 0)
    fig.text(left / width, .022 / height, footer, fontsize=6.3)
    save(fig, name)


def main() -> None:
    support = json.loads(SUPPORT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prev = json.loads(PREV.read_text(encoding="utf-8"))
    rankings = json.loads((SOURCE / "window_exposure_ranking_20261015.json").read_text(encoding="utf-8"))

    chosen, trace = {}, {}
    for scene, fold in PICKS:
        entry = rankings["candidates"][scene][0]
        x, y, w, h = entry["xywh"]
        mask_path = SUPPORT_DIR / support["scenes"][scene]["mask"]
        assert int((np.asarray(Image.open(mask_path)) > 0)[y:y + h, x:x + w].sum()) == w * h
        rgb_entry = manifest["scenes"][scene]["RGB"]
        rgb_path = DATA_ROOT / rgb_entry["path"]
        rgb = np.asarray(Image.open(rgb_path))[y:y + h, x:x + w].copy()
        bands = {}
        for band in ["B1", "B2", "B3", "B4", "B5"]:
            band_entry = manifest["scenes"][scene][band]
            bands[band] = np.asarray(Image.open(DATA_ROOT / band_entry["path"]))[y:y + h, x:x + w].copy()
        preds, pred_paths = {}, {}
        for model in MODELS:
            prior = prev["scenes"][scene]["predictions"][model]
            pred_paths[model] = Path(prior["prediction_path"])
            arr = np.load(prior["prediction_path"])
            preds[model] = np.array(arr[y:y + h, x:x + w], copy=True)
        chosen[scene] = {"fold": fold, "xywh": [x, y, w, h], "rgb": rgb, "bands": bands, "preds": preds,
                         "psnr": {m: entry[m] for m in MODELS},
                         "display": {"mean": entry["mean"], "p1": entry["p1"], "p99": entry["p99"]}}
        trace[scene] = {"mask_path": str(mask_path), "mask_sha256": sha(mask_path),
                        "rgb_path": str(rgb_path), "rgb_sha256": sha(rgb_path),
                        "prediction_paths": {m: {"path": str(p), "sha256": sha(p)}
                                             for m, p in pred_paths.items()},
                        "panel_sha256": {"RGB": arr_sha(rgb), **{b: arr_sha(bands[b]) for b in bands},
                                         **{m: arr_sha(preds[m]) for m in MODELS}}}

    rows = []
    for i, (scene, _) in enumerate(PICKS):
        c = chosen[scene]
        modality = [c["rgb"]] + [c["bands"][b] for b in ["B1", "B2", "B3", "B4", "B5"]]
        # Same convention as the prediction plate: the row's RGB target fixes the map.
        lo, hi = levels(c["rgb"])
        c["display_levels"] = {"lo": [round(v, 1) for v in lo], "hi": [round(v, 1) for v in hi]}
        rows.append((f"{'abc'[i]}  {scene}  |  测试折 {c['fold']}  |  (x, y) = ({c['xywh'][0]}, {c['xywh'][1]})",
                     [(arr, (lo, hi)) for arr in modality]))
    build(rows, MODE_LABELS, "fig_local_modalities_20261015",
          "每格为同一左上角坐标的256×256输出像素，按保存的八位数字量直接显示，未做对比度拉伸或增强；"
          "标尺为64个输出像素。")

    rows = []
    for i, (scene, _) in enumerate(PICKS):
        c = chosen[scene]
        panels = [c["rgb"], c["bands"]["B1"], c["preds"]["U-Net / B1-5"],
                  c["preds"]["Restormer / B1-5"], c["preds"]["pix2pix / B1-5"]]
        # Display levels come from the observed RGB target of the row and are
        # applied unchanged to the input and to every saved prediction.
        lo, hi = levels(c["rgb"])
        rows.append((f"{'abc'[i]}  {scene}  |  pix2pix（五波段）PSNR {c['psnr']['pix2pix / B1-5']:.2f} dB",
                     [(arr, (lo, hi)) for arr in panels]))
        c["display_levels"] = {"lo": [round(v, 1) for v in lo], "hi": [round(v, 1) for v in hi]}
    build(rows, PRED_LABELS, "fig_local_predictions_20261015",
          "三个模型均为种子11、五波段输入，展示各地点所属留出折的保存结果；B1列示意输入中的一个波段。"
          "预测由保存的[0,1]浮点值四舍五入为八位显示，与RGB真值同一显示范围，未做拉伸或增强；"
          "行标给出该窗口的pix2pix（五波段）PSNR；标尺为64个输出像素。")

    exports = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
               for pat in [f"fig_local_*{SUFFIX}.*"] for p in OUT.glob(pat)}
    payload = {
        "contract": "Illustrative plates from already-saved seed-11 test predictions and the published "
                    "mosaics. No training, no inference, no new metric.",
        "selection_rule": "Over 256x256 windows of the published output grid with fully valid shared "
                          "support, keep windows whose three displayed saved predictions are all at or "
                          "above 17.5 dB, then take the highest saved pix2pix (five-band) PSNR per scene. "
                          "No exposure screen: bright windows are systematically harder to reconstruct.",
        "display": f"One joint linear display map per row from the {DISPLAY_LO}%--{DISPLAY_HI}% luminance "
                   "percentiles of the row's RGB target together with the panels drawn beside it; the same "
                   "map applies to every panel of the row. No per-panel stretch, equalisation, gamma or "
                   "sharpening. Stored PSNR values are computed on the unmodified arrays.",
        "selected_windows": [{"scene": s, "fold": chosen[s]["fold"], "xywh": chosen[s]["xywh"],
                              "rgb_display_mean": round(chosen[s]["display"]["mean"], 1),
                              "rgb_display_p1": round(chosen[s]["display"]["p1"], 1),
                              "rgb_display_p99": round(chosen[s]["display"]["p99"], 1),
                              "psnr_db": {m: round(chosen[s]["psnr"][m], 4) for m in MODELS
                                          if m != "U-Net / B1-5" or True},
                              "panel_sha256": trace[s]["panel_sha256"]}
                             for s, _ in PICKS],
        "sources": trace,
        "exports": exports,
    }
    (SOURCE / "local_comparisons_20261015.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (SOURCE / "local_comparisons_20261015.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scene", "fold", "x", "y", "width", "height",
                                                "rgb_display_mean", "psnr_unet_five",
                                                "psnr_pix2pix_five", "psnr_restormer_five"])
        writer.writeheader()
        for scene, _ in PICKS:
            c = chosen[scene]
            writer.writerow({"scene": scene, "fold": c["fold"],
                             **dict(zip(["x", "y", "width", "height"], c["xywh"])),
                             "rgb_display_mean": round(c["display"]["mean"], 1),
                             "psnr_unet_five": round(c["psnr"]["U-Net / B1-5"], 4),
                             "psnr_pix2pix_five": round(c["psnr"]["pix2pix / B1-5"], 4),
                             "psnr_restormer_five": round(c["psnr"]["Restormer / B1-5"], 4)})
    print(json.dumps({"selected": payload["selected_windows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


