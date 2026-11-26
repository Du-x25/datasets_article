"""Reference points of the scene folds: the fold checkpoints on their own training windows and on held-out inputs
whose intensity is matched to the training scenes (diag_e4_shift.py). Results go to outputs/diag/ and are read by
aggregate_results.py. A variant with an existing summary.json is skipped.

用法: diag_e4_eval.py <gpu> [A|B]   (默认两折)
"""
import os
import subprocess
import sys
import time

ROOT = "D:/dh/MSWIR_RGB/bench_20261124"
PY = "D:/dh/MSWIR_RGB/.venv_mswir/Scripts/python.exe"
SCRIPTS = "D:/dh/chen_repro/scripts"
EPOCH = {"A": 180, "B": 220}


def main(gpu, folds):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for fold in folds:
        run = "%s/train/e4_scene%s" % (ROOT, fold)
        ckpt = "%s/checkpoints/e4_scene%s/%d_net_G.pth" % (run, fold, EPOCH[fold])
        data = "%s/datasets/scene_fold%s" % (ROOT, fold)
        diag = ROOT + "/outputs/diag"
        tests = {"train": (data + "/train_A", data + "/train_B"),
                 "shift_global": ("%s/e4_scene%s_shift_global/test_A" % (diag, fold), data + "/test_B"),
                 "shift_perimg": ("%s/e4_scene%s_shift_perimg/test_A" % (diag, fold), data + "/test_B")}
        for key, (test_a, gt) in tests.items():
            out = "%s/e4_scene%s_%s" % (diag, fold, key)
            if os.path.isfile(out + "/summary.json"):
                print("skip", out, flush=True)
                continue
            for step, cmd in (
                ("predict", [PY, SCRIPTS + "/predict_tfd_variant.py", "--repo", run + "/code", "--ckpt", ckpt,
                             "--test-a", test_a, "--out", out + "/pred", "--gpu", gpu]),
                ("metrics", [PY, SCRIPTS + "/metrics_from_dirs.py", "--pred", out + "/pred", "--gt", gt,
                             "--out", out, "--name", "diag_e4_scene%s_%s" % (fold, key), "--gpu", gpu]),
            ):
                t = time.time()
                with open("%s.%s.log" % (out, step), "w", encoding="utf-8") as log:
                    log.write(" ".join(cmd) + "\n")
                    log.flush()
                    rc = subprocess.run(cmd, cwd=run, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                print("fold %s %-12s %-7s rc=%d %.1f min" % (fold, key, step, rc, (time.time() - t) / 60), flush=True)
                if rc != 0:
                    raise SystemExit("step failed: %s" % cmd[1])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["A", "B"])
