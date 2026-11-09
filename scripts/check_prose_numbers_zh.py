"""Cross-check every arithmetic claim made in main_zh.tex prose against saved tables.

The QA script (scripts/qa_manuscripts.py) verifies macros and table cells; this
helper verifies the *derived* statements written into the prose during the
2026-10-15 style pass, so a rewritten sentence cannot silently misstate a value.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT.parents[1] / "Gen/v3_pipeline/outputs/summary_v2_20261005"
SENS = ROOT / "automation_completion_20261015/release/candidate_package/results/summary"

rows = {}
with (SUMMARY / "main_table.csv").open(encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        rows[(row["condition"], row["kind"])] = row

sens = {}
with (SENS / "sensitivity_aggregate_full_precision.csv").open(encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        sens[(row["mode"], row["magnitude_px"])] = row


def num(key, field):
    return float(rows[key][field])


models_five = [
    "unet_bands1,2,3,4,5",
    "pix2pix_bands1,2,3,4,5",
    "tfd_bands1,2,3,4,5",
    "restormer_bands1,2,3,4,5",
]
channel_five = num(("reference_bands1,2,3,4,5", "channel_reference"), "psnr_site_equal")

d_psnr = [num((m, "model"), "psnr_site_equal") - channel_five for m in models_five]
d_de = [num(("reference_bands1,2,3,4,5", "channel_reference"), "deltaE00_mean_site_equal")
        - num((m, "model"), "deltaE00_mean_site_equal") for m in models_five]

print("claim: 五波段学习模型相对通道参照 PSNR 高出 2.00--3.20 dB")
print(f"       actual {min(d_psnr):.4f} -- {max(d_psnr):.4f}")
for model, delta in sorted(zip(models_five, d_psnr), key=lambda kv: kv[1]):
    print(f"         {model:<32} {delta:+.4f} dB")
print("claim: dE 由 29.9596 降至 20.52--22.47")
print(f"       actual {num(('reference_bands1,2,3,4,5','channel_reference'),'deltaE00_mean_site_equal'):.4f} -> "
      f"{min(num((m,'model'),'deltaE00_mean_site_equal') for m in models_five):.4f}--"
      f"{max(num((m,'model'),'deltaE00_mean_site_equal') for m in models_five):.4f}")
print(f"       deltaE improvement {min(d_de):.4f}--{max(d_de):.4f}")

rest = num(("restormer_bands1,2,3,4,5", "model"), "psnr_site_equal")
lin = num(("reference_bands1,2,3,4,5", "linear_reference"), "psnr_site_equal")
print(f"claim: Restormer - linear = -0.3122 ; actual {rest - lin:.4f}")

pix5 = num(("pix2pix_bands1,2,3,4,5", "model"), "deltaE00_mean_site_equal")
pix1 = num(("pix2pix_bands1", "model"), "deltaE00_mean_site_equal")
s5 = num(("pix2pix_bands1,2,3,4,5", "model"), "ssim_site_equal")
s1 = num(("pix2pix_bands1", "model"), "ssim_site_equal")
print(f"claim: 五波段相对B1 dE 降低 1.7584 ; actual {pix1 - pix5:.4f}")
print(f"claim: SSIM 下降 0.0430 ; actual {s5 - s1:.4f}")
print(f"claim: PSNR 下降约 0.40 dB ; actual "
      f"{num(('pix2pix_bands1','model'),'psnr_site_equal') - num(('pix2pix_bands1,2,3,4,5','model'),'psnr_site_equal'):.4f}")

print(f"claim: 4px 联合扰动 PSNR -0.4451 / dE +0.8212 ; actual "
      f"{-float(sens[('joint','4.0')]['psnr_drop']):.4f} / {float(sens[('joint','4.0')]['deltaE00_rise']):.4f}")
print(f"claim: 仅B1 4px PSNR -0.2112 / dE +0.7877 ; actual "
      f"{-float(sens[('B1_only','4.0')]['psnr_drop']):.4f} / {float(sens[('B1_only','4.0')]['deltaE00_rise']):.4f}")
print(f"claim: 仅B1 0.5px PSNR -0.0040 ; actual "
      f"{-float(sens[('B1_only','0.5')]['psnr_drop']):.4f}")

# resolution support figures quoted in the prose
print("claim: 十地点 PSNR 范围 11.30--14.26 dB (pix2pix five-band)")
print(f"       actual {num(('pix2pix_bands1,2,3,4,5','model'),'psnr_min'):.4f}--"
      f"{num(('pix2pix_bands1,2,3,4,5','model'),'psnr_max'):.4f}")
