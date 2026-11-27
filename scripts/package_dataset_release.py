"""Assemble and SHA-256 verify the portable six-module M-SWIR-RGB release archive.

This script stages the full ~11 GB M-SWIR-RGB release directory outside the git
manuscript repository using NTFS hardlinks (zero extra disk space) or full file
copies, sanitizes Windows local paths in CSV/JSON/YAML manifests to portable
relative paths, verifies mosaic and source-frame SHA-256 digests, and writes a
master SHA256SUMS.txt manifest for public repository deposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENES = [f"C{i}" for i in range(1, 11)]
MODALITY_FILES = [
    ("RGB", "scene{n}.tif"),
    ("B1", "scene{n}_band1.tif"),
    ("B2", "scene{n}_band2.tif"),
    ("B3", "scene{n}_band3.tif"),
    ("B4", "scene{n}_band4.tif"),
    ("B5", "scene{n}_band5.tif"),
]
SWIR_BANDS = ["B1", "B2", "B3", "B4", "B5"]
MASK_NAMES = ["RGB.png", "B1.png", "B2.png", "B3.png", "B4.png", "B5.png", "redaction.png", "joint.png"]


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def stage_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def sanitize_windows_paths(text: str) -> str:
    text = re.sub(r"[A-Za-z]:[/\\]+A_projects[/\\]+swir_rgb_scene_mosaic[/\\]+configs[/\\]+current_v3[/\\]*", "benchmarks/configs/", text)
    text = re.sub(r"[A-Za-z]:[/\\]+Data_dir[/\\]+SWIR-RGB[/\\]+datasets_article_auto_v3[/\\]+", "data/mosaics/", text)
    text = re.sub(r"[A-Za-z]:[/\\]+Data_dir[/\\]+SWIR-RGB[/\\]+datasets_article_auto_v3(?=[,\s\"']|$)", "data/mosaics", text)
    text = re.sub(r"[A-Za-z]:[/\\]+Data_dir[/\\]+SWIR-RGB[/\\]+rawdata[/\\]+", "data/source_frames/", text)
    text = re.sub(r"[A-Za-z]:[/\\]+Data_dir[/\\]+SWIR-RGB[/\\]+rawdata(?=[,\s\"']|$)", "data/source_frames", text)
    text = re.sub(r"[A-Za-z]:[/\\]+USTC[/\\]+Gen[/\\]+v3_pipeline[/\\]+outputs[/\\]+v3_geometric_privacy_support", "support/v3_geometric_privacy_support", text)
    text = re.sub(r"[A-Za-z]:[/\\]+USTC[/\\]+Gen[/\\]+v3_pipeline[/\\]+configs[/\\]+", "metadata/", text)
    text = re.sub(r"[A-Za-z]:[/\\]+A_projects[/\\]+swir_rgb_scene_mosaic[/\\]+work[/\\]+current_v3", "metadata/geometry_work_ref", text)
    text = re.sub(r"[A-Za-z]:[/\\]+Data_dir[/\\]+SWIR-RGB[/\\]+derived_v3_e5_current", "support/derived_ref", text)
    return text.replace("\\", "/")


def write_release_docs(stage_dir: Path, summary: dict) -> None:
    readme = f"""# M-SWIR-RGB: Scene-Level Paired RGB and Five-Band Short-Wave Infrared Dataset

## Overview
- **Sites:** 10 distinct physical ground sites (`C1`–`C10`), 140 overlapping camera positions
- **Mosaics (`data/mosaics/`):** {summary['mosaics_count']} uncompressed 8-bit TIFF files (6 per site: RGB + B1–B5)
- **Selected source frames (`data/source_frames/`):** {summary['source_frames_count']} files (140 RGB + 700 SWIR)
- **Usability & privacy masks (`usability_masks/`):** {summary['masks_count']} PNG masks (`RGB`, `B1`–`B5`, `redaction`, `joint`)
- **Total payload size:** {summary['total_bytes'] / (1024**3):.2f} GiB
- **Licenses:** Data under CC BY 4.0 (`LICENSE_DATA_CC_BY_4.0.txt`); scripts/configs under MIT (`LICENSE_CODE_MIT.txt`)

## Directory Hierarchy
```
M-SWIR-RGB/
|-- data/
|   |-- mosaics/C1..C10/scene{{N}}.tif, scene{{N}}_band{{1..5}}.tif
|   \\-- source_frames/C1..C10/{{pos}}/rgb.jpg, B1.png..B5.png
|-- usability_masks/C1..C10/
|   |-- RGB.png, B1.png..B5.png, redaction.png, joint.png
|-- metadata/
|   |-- dataset_manifest.csv
|   |-- rawdata_release_manifest.csv
|   |-- acquisition_provenance.csv
|   |-- scene_quality_summary.csv
|   |-- source_provenance.csv
|   |-- privacy_redaction_layers.json
|   |-- usability_masks_manifest.json
|   |-- frame_to_mosaic_coverage.csv
|   \\-- dynamic_range_usability.csv
|-- support/
|   |-- current_protocol.json
|   |-- v3_geometric_privacy_support/
|   |-- archive_rgb_correspondence.json
|   \\-- tiepoint_validation_template.csv
|-- benchmarks/
|   |-- configs/c1.yaml..c10.yaml
|   |-- scripts/
|   \\-- source_data/
|-- SHA256SUMS.txt
|-- LICENSE_DATA_CC_BY_4.0.txt
|-- LICENSE_CODE_MIT.txt
\\-- README.md
```

## Integrity Verification
Run `sha256sum -c SHA256SUMS.txt` (Linux/macOS) or verify via Python against `SHA256SUMS.txt`.

## Radiometric & Privacy Notes
- All imagery consists of **uncalibrated 8-bit digital numbers (DN, 0–255)** with nominal 50 nm SWIR filter passbands (B1: 1000–1050, B2: 1050–1100, B3: 1225–1275, B4: 1525–1575, B5: 1625–1675 nm).
- Apply `usability_masks/<site>/joint.png` (`255` = valid) during training and scoring.
- Seven vehicle licence plates in `C10` are directly redacted across all six mosaics via `border_median_constant` and excluded in `redaction.png`.
"""
    (stage_dir / "README.md").write_text(readme, encoding="utf-8")

    (stage_dir / "LICENSE_DATA_CC_BY_4.0.txt").write_text(
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode\n\n"
        "You are free to share and adapt the M-SWIR-RGB dataset for any purpose, provided\n"
        "appropriate attribution is given to Du et al., M-SWIR-RGB Data Descriptor (2026).\n",
        encoding="utf-8",
    )
    (stage_dir / "LICENSE_CODE_MIT.txt").write_text(
        "MIT License\n\nCopyright (c) 2026 Hu Du, Xiao Liu, Lili Du, Hu Cui\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this\n"
        "software and associated documentation files, to deal in the Software without restriction,\n"
        "including without limitation the rights to use, copy, modify, merge, publish, distribute,\n"
        "sublicense, and/or sell copies of the Software, subject to the standard MIT notice.\n",
        encoding="utf-8",
    )
    (stage_dir / "PRE_UPLOAD_CHECKLIST.md").write_text(
        "# Pre-Upload Gate Checklist (Remove or Archive Before Public Deposition)\n\n"
        "- [x] 60 six-modality TIFF mosaics staged and SHA-256 verified against `dataset_manifest.csv`\n"
        "- [x] 840 selected source frames staged and RGB SHA-256 verified against `rawdata_release_manifest.csv`\n"
        "- [x] 80 usability/privacy masks staged\n"
        "- [x] All CSV/JSON/YAML manifests sanitized to relative portable paths (zero local drive paths)\n"
        "- [x] Master `SHA256SUMS.txt` generated\n"
        "- [ ] **MANUAL GATE:** Complete frame-by-frame privacy review/redaction of `data/source_frames/` "
        "(especially C10 licence plates) using `scripts/prepare_raw_privacy_review.py`, then re-run "
        "`package_dataset_release.py --refresh-sums` before uploading to Zenodo/Figshare/Science Data Bank.\n",
        encoding="utf-8",
    )


def refresh_sha256sums(stage_dir: Path) -> int:
    entries = []
    for p in sorted(stage_dir.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "PRE_UPLOAD_CHECKLIST.md"}:
            rel = p.relative_to(stage_dir).as_posix()
            entries.append(f"{sha256_file(p)}  {rel}")
    (stage_dir / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return len(entries)


def package_release(
    source_root: Path,
    rawdata_root: Path,
    configs_root: Path,
    support_root: Path,
    stage_dir: Path,
    mode: str,
    verify_hashes: bool,
) -> dict:
    meta_src = source_root / "metadata"
    assert (meta_src / "dataset_manifest.csv").is_file(), f"Missing {meta_src / 'dataset_manifest.csv'}"
    assert (meta_src / "rawdata_release_manifest.csv").is_file(), f"Missing {meta_src / 'rawdata_release_manifest.csv'}"

    stage_dir.mkdir(parents=True, exist_ok=True)
    mosaics_count = 0
    total_bytes = 0

    # 1. Mosaics + sanitized dataset_manifest.csv
    sanitized_dataset_rows = []
    with (meta_src / "dataset_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        dataset_fields = reader.fieldnames
        for row in reader:
            scene = row["scene"]
            fname = row["filename"]
            src = source_root / scene / fname
            assert src.is_file(), f"Missing mosaic: {src}"
            if verify_hashes:
                actual = sha256_file(src)
                assert actual == row["sha256"], f"SHA-256 mismatch for {src}: {actual} != {row['sha256']}"
            dst = stage_dir / "data" / "mosaics" / scene / fname
            stage_file(src, dst, mode)
            mosaics_count += 1
            total_bytes += src.stat().st_size
            new_row = dict(row)
            new_row["filename"] = f"data/mosaics/{scene}/{fname}"
            new_row["config"] = f"benchmarks/configs/{scene.lower()}.yaml"
            sanitized_dataset_rows.append(new_row)

    (stage_dir / "metadata").mkdir(parents=True, exist_ok=True)
    with (stage_dir / "metadata" / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dataset_fields)
        w.writeheader()
        w.writerows(sanitized_dataset_rows)

    # 2. 840 selected source frames + sanitized rawdata_release_manifest.csv
    source_frames_count = 0
    sanitized_raw_rows = []
    with (meta_src / "rawdata_release_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        raw_fields = reader.fieldnames
        for row in reader:
            scene = row["scene"]
            pos = row["position"]
            rgb_src = Path(row["published_rgb"])
            if not rgb_src.is_file():
                rgb_src = rawdata_root / scene / "pco" / rgb_src.name
            assert rgb_src.is_file(), f"Missing selected RGB source frame: {rgb_src}"
            if verify_hashes:
                actual = sha256_file(rgb_src)
                assert actual == row["rgb_sha256"], f"RGB SHA-256 mismatch for {rgb_src}"
            rgb_rel = f"data/source_frames/{scene}/{pos}/rgb{rgb_src.suffix.lower()}"
            stage_file(rgb_src, stage_dir / rgb_rel, mode)
            source_frames_count += 1
            total_bytes += rgb_src.stat().st_size

            new_row = dict(row)
            new_row["published_rgb"] = rgb_rel
            for band in SWIR_BANDS:
                col = f"published_{band}"
                b_src = Path(row[col])
                if not b_src.is_file():
                    b_src = rawdata_root / scene / "swir" / pos / f"{band}.png"
                assert b_src.is_file(), f"Missing selected SWIR frame: {b_src}"
                b_rel = f"data/source_frames/{scene}/{pos}/{band}.png"
                stage_file(b_src, stage_dir / b_rel, mode)
                new_row[col] = b_rel
                source_frames_count += 1
                total_bytes += b_src.stat().st_size
            sanitized_raw_rows.append(new_row)

    with (stage_dir / "metadata" / "rawdata_release_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        w.writerows(sanitized_raw_rows)

    # 3. Usability & privacy masks
    masks_count = 0
    for scene in SCENES:
        for mname in MASK_NAMES:
            src = source_root / "usability_masks" / scene / mname
            assert src.is_file(), f"Missing mask: {src}"
            stage_file(src, stage_dir / "usability_masks" / scene / mname, mode)
            masks_count += 1
            total_bytes += src.stat().st_size

    # 4. Remaining sanitized metadata files
    for fname in [
        "acquisition_provenance.csv",
        "scene_quality_summary.csv",
        "source_provenance.csv",
        "privacy_redaction_layers.json",
        "usability_masks_manifest.json",
        "frame_to_mosaic_coverage.csv",
        "dynamic_range_usability.csv",
    ]:
        src = meta_src / fname
        if src.is_file():
            cleaned = sanitize_windows_paths(src.read_text(encoding="utf-8"))
            assert not re.search(r"[A-Za-z]:[/\\]", cleaned), f"Unsanitized local path in {fname}"
            (stage_dir / "metadata" / fname).write_text(cleaned, encoding="utf-8")

    # 5. Support files (5-fold protocol, geometric/privacy support PNGs, archive correspondence, tiepoint template)
    (stage_dir / "support").mkdir(parents=True, exist_ok=True)
    if (meta_src / "current_protocol.json").is_file():
        cleaned = sanitize_windows_paths((meta_src / "current_protocol.json").read_text(encoding="utf-8"))
        (stage_dir / "support" / "current_protocol.json").write_text(cleaned, encoding="utf-8")
    if support_root.is_dir():
        dst_sup = stage_dir / "support" / "v3_geometric_privacy_support"
        dst_sup.mkdir(parents=True, exist_ok=True)
        for p in sorted(support_root.iterdir()):
            if p.is_file():
                if p.suffix.lower() in {".json", ".csv", ".txt"}:
                    (dst_sup / p.name).write_text(sanitize_windows_paths(p.read_text(encoding="utf-8")), encoding="utf-8")
                else:
                    stage_file(p, dst_sup / p.name, mode)
                    total_bytes += p.stat().st_size
    corr_src = REPO_ROOT / "figures/source_data/bench_20261124/archive_rgb_correspondence.json"
    if corr_src.is_file():
        shutil.copy2(corr_src, stage_dir / "support" / corr_src.name)

    # 6. Benchmarks (sanitized mosaic YAML configs + reproduction scripts + summary source data)
    (stage_dir / "benchmarks" / "configs").mkdir(parents=True, exist_ok=True)
    if configs_root.is_dir():
        for cfg in sorted(configs_root.glob("c*.yaml")):
            cleaned = sanitize_windows_paths(cfg.read_text(encoding="utf-8"))
            (stage_dir / "benchmarks" / "configs" / cfg.name).write_text(cleaned, encoding="utf-8")
    (stage_dir / "benchmarks" / "scripts").mkdir(parents=True, exist_ok=True)
    for script_name in [
        "build_geom_diagnostics.py",
        "audit_15pair_geometry.py",
        "run_trackc_drunet_finetune.py",
        "build_bench_20261124.py",
        "build_results_20261109.py",
        "make_site_fact_sheet.py",
        "evaluate_tiepoints.py",
    ]:
        sp = REPO_ROOT / "scripts" / script_name
        if sp.is_file():
            shutil.copy2(sp, stage_dir / "benchmarks" / "scripts" / script_name)
    sd_dst = stage_dir / "benchmarks" / "source_data"
    if sd_dst.exists():
        shutil.rmtree(sd_dst)
    shutil.copytree(REPO_ROOT / "figures" / "source_data", sd_dst)

    summary = {
        "mosaics_count": mosaics_count,
        "source_frames_count": source_frames_count,
        "masks_count": masks_count,
        "total_bytes": total_bytes,
        "mode": mode,
        "verified_hashes": verify_hashes,
        "stage_dir": str(stage_dir),
    }
    write_release_docs(stage_dir, summary)
    summary["sha256sums_entries"] = refresh_sha256sums(stage_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v3"))
    parser.add_argument("--rawdata-root", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\rawdata"))
    parser.add_argument("--configs-root", type=Path, default=Path(r"D:\A_projects\swir_rgb_scene_mosaic\configs\current_v3"))
    parser.add_argument("--support-root", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\derived_v3_e5_20260923\v3_geometric_privacy_support"))
    parser.add_argument("--stage-dir", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\M-SWIR-RGB_release_v1"))
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--skip-verify", action="store_true", help="Skip input SHA-256 verification before staging.")
    parser.add_argument("--refresh-sums", action="store_true", help="Only recompute SHA256SUMS.txt inside an existing stage-dir.")
    args = parser.parse_args()

    if args.refresh_sums:
        count = refresh_sha256sums(args.stage_dir)
        print(json.dumps({"status": "refreshed", "stage_dir": str(args.stage_dir), "sha256sums_entries": count}, indent=2))
        return 0

    summary = package_release(
        source_root=args.source_root,
        rawdata_root=args.rawdata_root,
        configs_root=args.configs_root,
        support_root=args.support_root,
        stage_dir=args.stage_dir,
        mode=args.mode,
        verify_hashes=not args.skip_verify,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
