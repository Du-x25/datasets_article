"""从已有 4:1 窗口清单派生三类数据集，不改动原数据集。

bands      5 个单波段（同一波段复制为 3 通道）+ 其余 9 个三波段组合；窗口与原数据集逐一同名
fractions  25/50/75% 训练子集（按场景分层、嵌套），测试集不变
scenes     二折场景划分（C1-C5 与 C6-C10 互为训练与测试），测试场景的全部窗口作测试

用法: prepare_bench_datasets.py bands|fractions|scenes|all
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE = Path("D:/dh/chen_repro/datasets/mswir_random_4to1_3552")
SRC = Path("D:/dh/MSWIR_RGB/data/release")
OUT = Path("D:/dh/MSWIR_RGB/bench_20261124/datasets")
FOV, SIZE, BATCH, REF_UPDATES = 1024, 256, 4, 71000
NAME = re.compile(r"^(C\d+)_(\d+)_y(\d+)_x(\d+)\.png$")
SINGLES = [("B%d" % i, (i,)) for i in range(1, 6)]
TRIPLES = [("B%d%d%d" % t, t) for t in itertools.combinations(range(1, 6), 3) if t != (1, 2, 3)]
FOLDS = {"scene_foldA": (["C1", "C2", "C3", "C4", "C5"], ["C6", "C7", "C8", "C9", "C10"]),
         "scene_foldB": (["C6", "C7", "C8", "C9", "C10"], ["C1", "C2", "C3", "C4", "C5"])}


def windows(split):
    a = sorted(p.name for p in (BASE / (split + "_A")).iterdir())
    b = sorted(p.name for p in (BASE / (split + "_B")).iterdir())
    if a != b:
        raise SystemExit("%s_A and %s_B differ" % (split, split))
    out = []
    for name in a:
        m = NAME.match(name)
        if not m:
            raise SystemExit("unexpected file name %s" % name)
        out.append((name, m.group(1), int(m.group(3)), int(m.group(4))))
    return out


def new_dir(name):
    d = OUT / name
    for sub in ("train_A", "train_B", "test_A", "test_B"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def copy_pairs(d, split, names, src_split=None):
    src = src_split or split
    for n in names:
        shutil.copy2(BASE / (src + "_A") / n, d / (split + "_A") / n)
        shutil.copy2(BASE / (src + "_B") / n, d / (split + "_B") / n)


def per_scene(names):
    c = {}
    for n in names:
        c[n.split("_")[0]] = c.get(n.split("_")[0], 0) + 1
    return dict(sorted(c.items()))


def write_manifest(d, **kw):
    tr = sorted(p.name for p in (d / "train_A").iterdir())
    te = sorted(p.name for p in (d / "test_A").iterdir())
    info = {
        "schema": "mswir-bench-20261124-v1",
        "parent": str(BASE),
        "source_root": str(SRC),
        "source_fov": FOV,
        "network_size": SIZE,
        "scale": "4:1, Pillow BICUBIC (same as parent)",
        "train": len(tr),
        "test": len(te),
        "train_per_scene": per_scene(tr),
        "test_per_scene": per_scene(te),
        "iters_per_epoch": len(tr) // BATCH,
        "window_sha256": {"train": hashlib.sha256("\n".join(tr).encode()).hexdigest(),
                          "test": hashlib.sha256("\n".join(te).encode()).hexdigest()},
    }
    info.update(kw)
    (d / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def report(name, info):
    it = info["iters_per_epoch"]
    epochs = min(range(10, 2000, 10), key=lambda t: abs(t * it - REF_UPDATES))
    print("%-11s train %4d (%3d it/ep) test %4d | budget epoch %d -> %d updates (%+.1f%% vs %d)"
          % (name, info["train"], it, info["test"], epochs, epochs * it,
             (epochs * it / REF_UPDATES - 1) * 100, REF_UPDATES), flush=True)


def build_bands():
    train, test = windows("train"), windows("test")
    variants = SINGLES + TRIPLES
    dirs = {v: new_dir(v) for v, _ in variants}
    by_scene = {}
    for split, lst in (("train", train), ("test", test)):
        for w in lst:
            by_scene.setdefault(w[1], []).append((split,) + w)
    for scene in sorted(by_scene):
        bands = [np.array(Image.open(SRC / scene / ("scene%s_band%d.tif" % (scene[1:], i))))
                 for i in range(1, 6)]
        for split, name, _, y, x in by_scene[scene]:
            chans = {i: np.array(Image.fromarray(bands[i - 1][y:y + FOV, x:x + FOV])
                                 .resize((SIZE, SIZE), Image.BICUBIC)) for i in range(1, 6)}
            ref = np.array(Image.open(BASE / (split + "_A") / name).convert("RGB"))
            again = np.stack([chans[1], chans[2], chans[3]], axis=-1)
            if not np.array_equal(again, ref):
                raise SystemExit("window mapping mismatch for %s: max abs diff %d"
                                 % (name, int(np.abs(again.astype(int) - ref.astype(int)).max())))
            for v, idx in variants:
                c = [chans[i] for i in idx]
                Image.fromarray(np.stack(c * 3 if len(c) == 1 else c, axis=-1)).save(dirs[v] / (split + "_A") / name)
        del bands
        print("bands", scene, len(by_scene[scene]), "windows", flush=True)
    for v, idx in variants:
        for split, lst in (("train", train), ("test", test)):
            for w in lst:
                shutil.copy2(BASE / (split + "_B") / w[0], dirs[v] / (split + "_B") / w[0])
        report(v, write_manifest(dirs[v], variant="band", input_A=idx,
                                 input_note="same band replicated to 3 channels" if len(idx) == 1
                                 else "stack of the listed bands in ascending order",
                                 target_B="copied from parent"))


def build_fractions(seed):
    train, test = windows("train"), windows("test")
    by_scene = {}
    for w in train:
        by_scene.setdefault(w[1], []).append(w)
    rng, order = random.Random(seed), {}
    for sc in sorted(by_scene):
        order[sc] = sorted(by_scene[sc])
        rng.shuffle(order[sc])
    for frac, name in ((0.25, "frac25"), (0.5, "frac50"), (0.75, "frac75")):
        keep = [w[0] for sc in sorted(order) for w in order[sc][:int(round(len(order[sc]) * frac))]]
        d = new_dir(name)
        copy_pairs(d, "train", keep)
        copy_pairs(d, "test", [w[0] for w in test])
        report(name, write_manifest(d, variant="fraction", fraction=frac, subset_seed=seed,
                                    subset_rule="per-scene shuffle with subset_seed, nested 25<50<75",
                                    target_B="copied from parent"))


def build_scenes():
    train, test = windows("train"), windows("test")
    for name, (tr_sc, te_sc) in FOLDS.items():
        d = new_dir(name)
        copy_pairs(d, "train", [w[0] for w in train if w[1] in tr_sc])
        for split, lst in (("train", train), ("test", test)):
            copy_pairs(d, "test", [w[0] for w in lst if w[1] in te_sc], src_split=split)
        report(name, write_manifest(d, variant="scene", train_scenes=tr_sc, test_scenes=te_sc,
                                    target_B="copied from parent; test holds every window of the test scenes",
                                    note="windows named *_B remain B123 input + RGB target"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bands", "fractions", "scenes", "all"])
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()
    if args.what in ("fractions", "all"):
        build_fractions(args.seed)
    if args.what in ("scenes", "all"):
        build_scenes()
    if args.what in ("bands", "all"):
        build_bands()


if __name__ == "__main__":
    main()
