"""TFD 基准队列执行器：每个 (GPU, slot) 一个进程，从 jobs.json 领任务并执行训练与统一 PNG 评测。

用法: runner.py --gpu 0 --slot 0
任务字段: id, dataroot, niter, niter_decay, eval_epochs, [stop_after], [gpu], [priority]
评测任务: id, eval_only, dataroot, ckpt, epoch

所有新文件只写入 ROOT；REPRO 与陈的数据目录仅读取。
"""
import argparse
import hashlib
import json
import msvcrt
import os
import shutil
import subprocess
import time
from datetime import datetime

ROOT = "D:/dh/MSWIR_RGB/bench_20261124"
REPRO = "D:/dh/chen_repro"
PY = "D:/dh/MSWIR_RGB/.venv_mswir/Scripts/python.exe"
REF_CODE = REPRO + "/train/tfd_mswir_random_4to1_b4"
CODE_FILES = [
    "LICENSE.txt", "WTConv.py", "eca.py", "train.py", "test.py",
    "data/__init__.py", "data/aligned_dataset.py", "data/base_data_loader.py", "data/base_dataset.py",
    "data/custom_dataset_data_loader.py", "data/data_loader.py", "data/image_folder.py",
    "models/__init__.py", "models/base_model.py", "models/models.py", "models/network_drunet.py",
    "models/networks.py", "models/pix2pixHD_model.py", "models/ui_model.py",
    "options/__init__.py", "options/base_options.py", "options/test_options.py", "options/train_options.py",
    "util/__init__.py", "util/html.py", "util/image_pool.py", "util/metrics.py", "util/util.py",
    "util/visualizer.py",
]
TRAIN_ARGS = ["--label_nc", "0", "--no_instance", "--loadSize", "256", "--batchSize", "4"]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def event(msg):
    line = "[%s] %s" % (now(), msg)
    with open(ROOT + "/events.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


class StateLock:
    def __init__(self):
        self.f = open(ROOT + "/state.lock", "a+b")
        while True:
            self.f.seek(0)
            try:
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.f.close()


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def claim(gpu):
    with StateLock():
        jobs = load_json(ROOT + "/jobs.json", [])
        state = load_json(ROOT + "/state.json", {})
        pending = [j for j in jobs
                   if state.get(j["id"], {}).get("status", "pending") == "pending"
                   and j.get("gpu", None) in (None, gpu)]
        if not pending:
            return None
        pending.sort(key=lambda j: j.get("priority", 100))
        job = pending[0]
        state[job["id"]] = {"status": "running", "gpu": gpu, "runner_pid": os.getpid(),
                            "start": now(), "evals": {}}
        save_json(ROOT + "/state.json", state)
        return job


def update(job_id, fn):
    with StateLock():
        state = load_json(ROOT + "/state.json", {})
        fn(state.setdefault(job_id, {}))
        save_json(ROOT + "/state.json", state)


def evaluated(job_id, epoch):
    return str(epoch) in load_json(ROOT + "/state.json", {}).get(job_id, {}).get("evals", {})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_workspace(job, gpu):
    ws = "%s/train/%s" % (ROOT, job["id"])
    if os.path.isdir(ws):
        os.rename(ws, ws + ".prev_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    code = ws + "/code"
    hashes = {}
    for f in CODE_FILES:
        dst = os.path.join(code, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(REF_CODE, f), dst)
        hashes[f] = sha256(dst)
    os.makedirs(ws + "/checkpoints")
    os.makedirs(ws + "/logs")
    manifest = {
        "job": job,
        "workspace": ws,
        "gpu": gpu,
        "created": now(),
        "code_source": REF_CODE,
        "code_identical_to_source": "D:/chen/pix2pixhd-p-noshuffle (29 files, sha256 verified 2026-11-24)",
        "code_sha256": hashes,
        "training": {
            "args": " ".join(TRAIN_ARGS + ["--niter %d --niter_decay %d" % (job.get("niter", 0), job.get("niter_decay", 0))]),
            "lr": "2e-4, constant for niter epochs, then linear decay to zero over niter_decay epochs",
            "seed": "none (Chen's code sets no seed); each run is an independent random init",
            "note": "only data, schedule length, name, checkpoints_dir and gpu differ from the reference launcher",
        },
        "evaluation": "predict_tfd_variant.py -> 8-bit PNG -> metrics_from_dirs.py (PSNR, skimage SSIM, LPIPS-Alex)",
        "environment": {"python": "3.8.19", "torch": "2.4.1+cu121", "gpu": "NVIDIA A100-PCIE-40GB"},
    }
    save_json(ws + "/repro_manifest.json", manifest)
    return ws


def ckpt_ready(ws, name, epoch):
    # D is saved after G, so an older D file means the G checkpoint is complete
    d = "%s/checkpoints/%s/%d_net_D.pth" % (ws, name, epoch)
    try:
        return time.time() - os.path.getmtime(d) > 20
    except OSError:
        return False


def kill_tree(proc):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def run_eval(job, ws, gpu, epoch, ckpt=None):
    ckpt = ckpt or "%s/checkpoints/%s/%d_net_G.pth" % (ws, job["id"], epoch)
    out = "%s/outputs/%s/ep%d" % (ROOT, job["id"], epoch)
    data = job["dataroot"]
    if not os.path.isfile(out + "/summary.json"):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with open(ws + "/logs/eval_ep%d.log" % epoch, "a", encoding="utf-8") as log:
            for cmd in (
                [PY, REPRO + "/scripts/predict_tfd_variant.py", "--repo", ws + "/code", "--ckpt", ckpt,
                 "--test-a", data + "/test_A", "--out", out + "/pred", "--gpu", str(gpu)],
                [PY, REPRO + "/scripts/metrics_from_dirs.py", "--pred", out + "/pred", "--gt", data + "/test_B",
                 "--out", out, "--name", "%s_ep%d" % (job["id"], epoch), "--gpu", str(gpu)],
            ):
                log.write(" ".join(cmd) + "\n")
                log.flush()
                rc = subprocess.run(cmd, cwd=ws, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                if rc != 0:
                    raise RuntimeError("eval step failed rc=%d: %s" % (rc, cmd[1]))
    s = load_json(out + "/summary.json", {})
    return {"psnr": s["psnr"]["mean"], "ssim_sk": s["ssim_sk"]["mean"],
            "lpips_alex": s["lpips_alex"]["mean"], "n": s["n"], "summary": out + "/summary.json"}


def report_eval(job, epoch, res):
    update(job["id"], lambda st: st.setdefault("evals", {}).__setitem__(str(epoch), res))
    event("EVAL %s ep%d PSNR %.4f SSIM %.4f LPIPS %.4f n=%d"
          % (job["id"], epoch, res["psnr"], res["ssim_sk"], res["lpips_alex"], res["n"]))


def run_job(job, gpu):
    jid, t0 = job["id"], time.time()
    event("START %s gpu=%d%s" % (jid, gpu, " (eval only)" if job.get("eval_only") else ""))
    ws = make_workspace(job, gpu)
    try:
        if job.get("eval_only"):
            res = run_eval(job, ws, gpu, job["epoch"], job["ckpt"])
            update(jid, lambda st: st.update(status="done", end=now(), evals={str(job["epoch"]): res}))
            event("DONE %s %.0fs" % (jid, time.time() - t0))
            return

        cmd = [PY, "train.py", "--name", jid, "--dataroot", job["dataroot"],
               "--checkpoints_dir", ws + "/checkpoints", *TRAIN_ARGS, "--gpu_ids", str(gpu),
               "--niter", str(job["niter"]), "--niter_decay", str(job["niter_decay"])]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        stop = job.get("stop_after")
        with open(ws + "/logs/train.log", "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ws + "/code", stdout=log, stderr=subprocess.STDOUT, env=env)
            update(jid, lambda st: st.update(train_pid=proc.pid, workspace=ws, command=" ".join(cmd)))
            status = "failed"
            while True:
                rc = proc.poll()
                for e in job["eval_epochs"]:
                    if not evaluated(jid, e) and ckpt_ready(ws, jid, e):
                        report_eval(job, e, run_eval(job, ws, gpu, e))
                if rc is not None:
                    status = "done" if rc == 0 else "failed"
                    break
                if stop and ckpt_ready(ws, jid, stop):
                    kill_tree(proc)
                    status = "done"
                    break
                time.sleep(30)
    except Exception as e:
        status = "failed"
        update(jid, lambda st: st.update(error=repr(e)))
        event("FAIL %s %r" % (jid, e))
    update(jid, lambda st: st.update(status=status, end=now(), hours=round((time.time() - t0) / 3600, 2)))
    event("DONE %s status=%s %.2fh" % (jid, status, (time.time() - t0) / 3600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--slot", type=int, required=True)
    args = ap.parse_args()
    os.makedirs(ROOT + "/logs", exist_ok=True)
    event("RUNNER up gpu=%d slot=%d pid=%d" % (args.gpu, args.slot, os.getpid()))
    while True:
        job = claim(args.gpu)
        if job is None:
            time.sleep(60)
            continue
        run_job(job, args.gpu)


if __name__ == "__main__":
    main()
