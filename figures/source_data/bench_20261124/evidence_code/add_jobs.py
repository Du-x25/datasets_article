"""把实验 2/3/4 的任务追加到 jobs.json（持 StateLock，原子替换）。重复 id 不会重复加入。"""
import json
import os
import sys

import runner

DS = runner.ROOT + "/datasets"
E2 = ["B1", "B2", "B3", "B4", "B5", "B124", "B125", "B134", "B135", "B145", "B234", "B235", "B245", "B345"]

NEW = [
    {"id": "e3_frac25", "group": "e3", "priority": 20, "dataroot": DS + "/frac25",
     "niter": 400, "niter_decay": 400, "stop_after": 400, "eval_epochs": [100, 200, 300, 400]},
    {"id": "e3_frac50", "group": "e3", "priority": 20, "dataroot": DS + "/frac50",
     "niter": 200, "niter_decay": 200, "stop_after": 200, "eval_epochs": [50, 100, 150, 200]},
    {"id": "e3_frac75", "group": "e3", "priority": 20, "dataroot": DS + "/frac75",
     "niter": 130, "niter_decay": 130, "stop_after": 130, "eval_epochs": [50, 100, 130]},
    {"id": "e4_sceneA", "group": "e4", "priority": 30, "dataroot": DS + "/scene_foldA",
     "niter": 180, "niter_decay": 180, "stop_after": 180, "eval_epochs": [180]},
    {"id": "e4_sceneB", "group": "e4", "priority": 30, "dataroot": DS + "/scene_foldB",
     "niter": 220, "niter_decay": 220, "stop_after": 220, "eval_epochs": [220]},
]
for b in E2:
    NEW.append({"id": "e2_" + b, "group": "e2", "priority": 40, "dataroot": DS + "/" + b,
                "niter": 100, "niter_decay": 100, "stop_after": 100, "eval_epochs": [100]})

if __name__ == "__main__":
    with runner.StateLock():
        jobs = runner.load_json(runner.ROOT + "/jobs.json", [])
        have = {j["id"] for j in jobs}
        added = [j for j in NEW if j["id"] not in have]
        jobs += added
        runner.save_json(runner.ROOT + "/jobs.json", jobs)
    print("added %d jobs, total %d" % (len(added), len(jobs)))
    for j in NEW:
        ok = all(os.path.isdir(j["dataroot"] + "/" + s) for s in ("train_A", "train_B", "test_A", "test_B"))
        print("%-14s prio %d  %-46s dirs=%s" % (j["id"], j["priority"], j["dataroot"], ok))
    sys.stdout.flush()
