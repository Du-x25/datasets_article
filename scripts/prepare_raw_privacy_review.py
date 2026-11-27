"""Generate visual privacy contact sheets and checklist for the 840 selected raw source frames.

Also supports applying border_median_constant redaction boxes to staged release copies
(never modifying immutable originals in D:\\Data_dir\\SWIR-RGB\\rawdata) once the author
annotates bounding boxes for raw source frames (e.g., C10 plates).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_contact_sheet(items: list[tuple[str, Path]], out_path: Path, thumb_size: int = 384, cols: int = 5) -> None:
    if not items:
        return
    rows = math.ceil(len(items) / cols)
    pad = 12
    label_h = 26
    canvas_w = cols * thumb_size + (cols + 1) * pad
    canvas_h = rows * (thumb_size + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img_path) in enumerate(items):
        r, c = divmod(idx, cols)
        x0 = pad + c * (thumb_size + pad)
        y0 = pad + r * (thumb_size + label_h + pad)
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            ox = x0 + (thumb_size - im_rgb.width) // 2
            oy = y0 + label_h + (thumb_size - im_rgb.height) // 2
            sheet.paste(im_rgb, (ox, oy))
        draw.rectangle([x0, y0, x0 + thumb_size, y0 + label_h], fill=(30, 41, 59))
        draw.text((x0 + 8, y0 + 4), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def apply_border_median_constant(arr: np.ndarray, box: list[int], border: int = 4) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return out
    bx1, by1 = max(0, x1 - border), max(0, y1 - border)
    bx2, by2 = min(w, x2 + border), min(h, y2 + border)
    patch = out[by1:by2, bx1:bx2]
    mask = np.ones((by2 - by1, bx2 - bx1), dtype=bool)
    mask[(y1 - by1):(y2 - by1), (x1 - bx1):(x2 - bx1)] = False
    ring = patch[mask]
    if ring.size > 0:
        fill = np.median(ring, axis=0).astype(out.dtype)
        out[y1:y2, x1:x2] = fill
    return out


def generate_review_pack(manifest_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = out_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list[tuple[str, Path]]] = {}
    checklist_rows = []
    with manifest_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scene = row["scene"]
            pos = row["position"]
            rgb_path = Path(row["published_rgb"])
            by_scene.setdefault(scene, []).append((f"{scene} pos {pos} ({rgb_path.name})", rgb_path))
            status = "NEEDS_PLATE_REDACTION" if scene == "C10" else "PENDING_VISUAL_CHECK"
            checklist_rows.append({
                "scene": scene,
                "position": pos,
                "rgb_file": f"data/source_frames/{scene}/{pos}/rgb{rgb_path.suffix.lower()}",
                "swir_files": f"data/source_frames/{scene}/{pos}/B1..B5.png",
                "status": status,
                "boxes_xyxy_json": "[]",
                "reviewer_notes": "C10 parking lot: verify and redact vehicle licence plates in raw RGB/SWIR frames" if scene == "C10" else "",
            })

    for scene, items in sorted(by_scene.items()):
        make_contact_sheet(items, sheets_dir / f"{scene}_rgb_sheet.jpg", thumb_size=400, cols=5)
        if scene == "C10":
            make_contact_sheet(items, sheets_dir / "C10_rgb_detail_sheet.jpg", thumb_size=768, cols=3)

    checklist_path = out_dir / "privacy_review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checklist_rows[0].keys()))
        w.writeheader()
        w.writerows(checklist_rows)

    template_boxes = {
        "instructions": "Add pixel bounding boxes [x1, y1, x2, y2] keyed by relative path under stage_dir/data/source_frames/...",
        "method": "border_median_constant",
        "boxes": {
            "data/source_frames/C10/01/rgb.jpg": [],
        },
    }
    (out_dir / "raw_redaction_boxes_template.json").write_text(json.dumps(template_boxes, indent=2), encoding="utf-8")

    return {
        "scenes_rendered": len(by_scene),
        "positions_logged": len(checklist_rows),
        "frames_covered": len(checklist_rows) * 6,
        "checklist": str(checklist_path),
        "contact_sheets_dir": str(sheets_dir),
    }


def apply_redactions(stage_dir: Path, boxes_json: Path) -> dict:
    spec = json.loads(boxes_json.read_text(encoding="utf-8"))
    modified = []
    for rel_path, boxes in spec.get("boxes", {}).items():
        if not boxes:
            continue
        target = stage_dir / rel_path
        assert target.is_file(), f"Missing staged file to redact: {target}"
        with Image.open(target) as im:
            arr = np.array(im)
        for box in boxes:
            arr = apply_border_median_constant(arr, box)
        # Break any hardlink before writing so original rawdata is NEVER altered.
        tmp = target.with_suffix(target.suffix + ".tmp")
        Image.fromarray(arr).save(tmp, quality=95 if target.suffix.lower() in {".jpg", ".jpeg"} else None)
        target.unlink()
        tmp.rename(target)
        modified.append({"file": rel_path, "boxes": len(boxes), "new_sha256": sha256_file(target)})
    return {"modified_files": modified, "count": len(modified)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v3\metadata\rawdata_release_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\privacy_review_840frames"))
    parser.add_argument("--stage-dir", type=Path, default=Path(r"D:\Data_dir\SWIR-RGB\M-SWIR-RGB_release_v1"))
    parser.add_argument("--apply-boxes", type=Path, default=None, help="Apply border_median_constant redaction boxes JSON to staged release files.")
    args = parser.parse_args()

    if args.apply_boxes is not None:
        res = apply_redactions(args.stage_dir, args.apply_boxes)
        print(json.dumps(res, indent=2))
        return 0

    res = generate_review_pack(args.manifest, args.out_dir)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
