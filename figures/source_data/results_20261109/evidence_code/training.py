"""One auditable three-channel task, called by the PyCharm train entry point.

The scheduler owns device assignment; this module owns one task's scientific
identity, optimizers, deterministic sampling, validation and durable recovery.
No formal training starts by importing this module.
"""
import contextlib
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .checkpoint import (CheckpointManager, IdentityMismatch, atomic_torch_save,
                         capture_rng, restore_rng)
from .io_utils import (append_jsonl, atomic_destination, atomic_json, file_lock,
                       read_json, sha256, signature, utc_now)
from .protocols import SiteBalancedSampler


def seed_everything(seed, device=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolved_parameters(task, config):
    model = task["model"].lower()
    default_lr = 1e-4 if model == "drunet" else 2e-4
    default_betas = (0.9, 0.999) if model == "drunet" else (0.5, 0.999)
    max_updates = config.get("max_updates")
    if not isinstance(max_updates, int) or isinstance(max_updates, bool) or max_updates < 1:
        raise ValueError("Set and freeze a positive integer max_updates before starting")
    batch_size = int(config.get("batch_size", 4))
    validation_every = int(config.get("validation_every", min(2000, max_updates)))
    if min(batch_size, validation_every) < 1:
        raise ValueError("batch_size and validation_every must be positive")
    lr = float(config.get("model_learning_rates", {}).get(model, default_lr))
    betas = list(config.get("model_betas", {}).get(model, default_betas))
    if lr <= 0 or len(betas) != 2 or any(not 0 <= value < 1 for value in betas):
        raise ValueError("Invalid Adam learning rate or beta values")
    clip_norm = config.get("gradient_clip_norm")
    if clip_norm is not None and (not math.isfinite(float(clip_norm)) or float(clip_norm) <= 0):
        raise ValueError("gradient_clip_norm must be positive and finite when specified")
    return {"max_updates": max_updates, "batch_size": batch_size,
            "validation_every": validation_every, "optimizer": "Adam", "lr": lr,
            "betas": betas, "weight_decay": float(config.get("weight_decay", 0.0)),
            "schedule": "half_constant_half_linear", "amp_bf16": bool(config.get("amp_bf16", False)),
            "augmentation": "D4_uniform_8", "model_config": config.get("model_config", {}),
            "checkpoint_selection": "fixed_validation_windows_site_equal_psnr",
            "gradient_clip_norm": clip_norm}


def training_identity(task, config, protocol, data_identity):
    """Paths, logging frequency and GPU number do not change scientific identity."""
    parameters = _resolved_parameters(task, config)
    source_identity = config.get("source_identity")
    if source_identity is None:
        root = Path(__file__).parent
        source_identity = {name: sha256(root / name) for name in
                           ("training.py", "checkpoint.py", "io_utils.py", "modeling.py", "data.py", "protocols.py")
                           if (root / name).exists()}
        vendor = root.parent / "vendor"
        if vendor.exists():
            source_identity.update({str(path.relative_to(root.parent)).replace("\\", "/"): sha256(path)
                                    for path in sorted(vendor.rglob("*.py"))})
    identity_fields = {"schema": "mswir-training-v1", "task": task,
                       "parameters": parameters, "data_identity": data_identity,
                       "protocol": protocol, "source_identity": source_identity,
                       "weights_identity": config.get("weights_identity", {}),
                       "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__)}}
    return signature(identity_fields), identity_fields


def _to_scalar(value):
    if torch.is_tensor(value):
        value = value.detach().float().item()
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("A loss component is non-finite")
    return value


def _range_to_zero_one(tensor, input_range):
    if input_range == "minus_one_one":
        return (tensor.float() + 1.0) / 2.0
    if input_range == "zero_one":
        return tensor.float()
    raise ValueError("Unknown model input range: " + str(input_range))


def _as_tensor(value):
    if torch.is_tensor(value):
        return value.float()
    return torch.from_numpy(np.asarray(value).copy()).float()


def _batch(store, windows, channel_map, input_range, device, transforms=None):
    triples = [store.load_window(window, channel_map=channel_map, input_range=input_range,
                                transform=0 if transforms is None else transforms[index])
               for index, window in enumerate(windows)]
    x, y, mask = (torch.stack([_as_tensor(item[column]) for item in triples]).to(device)
                  for column in range(3))
    if x.ndim != 4 or y.ndim != 4 or x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError("Every model requires BCHW three-channel input and RGB target")
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape != (x.shape[0], 1, x.shape[2], x.shape[3]) or x.shape != y.shape:
        raise ValueError("Input, target and mask shapes differ")
    if not all(torch.isfinite(value).all().item() for value in (x, y, mask)):
        raise ValueError("Input data contains non-finite values")
    if not all(float(item.sum()) > 0 for item in mask):
        raise ValueError("An input window has no valid target pixels")
    return x, y, mask


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _check_gradient(parameters, clip_norm=None):
    # Infinity means compute and validate the norm without changing gradients.
    norm = torch.nn.utils.clip_grad_norm_(parameters,
                                          float("inf") if clip_norm is None else float(clip_norm),
                                          error_if_nonfinite=True)
    return float(norm.detach().cpu())


def _state_finite(module):
    return all(not (value.is_floating_point() or value.is_complex()) or
               bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def training_step(bundle, x, y, mask, g_optimizer, d_optimizer, device, amp, clip_norm=None):
    """Validate both losses and gradients before either optimizer is advanced.

    D receives detached fake images inside the model bundle. G then receives
    gradients through the same frozen D; both updates use the pre-update D.
    This explicit common update order applies to both adversarial baselines.
    """
    g_optimizer.zero_grad(set_to_none=True)
    discriminator = bundle.discriminator
    metrics = {}
    if d_optimizer is not None:
        d_optimizer.zero_grad(set_to_none=True)
        discriminator.requires_grad_(True)
    with _autocast(device, amp):
        fake = bundle.generator(x)
        if fake.shape != y.shape:
            raise ValueError("Generator output shape does not match RGB target")
        if d_optimizer is not None:
            d_loss, d_parts = bundle.discriminator_loss(fake, x, y, mask)
        else:
            d_loss, d_parts = None, {}
    if d_loss is not None:
        metrics["d_total"] = _to_scalar(d_loss)
        metrics.update({"d_" + key: _to_scalar(value) for key, value in d_parts.items()})
        d_loss.backward()
        metrics["d_gradient_norm"] = _check_gradient(discriminator.parameters(), clip_norm)
        discriminator.requires_grad_(False)
    try:
        with _autocast(device, amp):
            g_loss, g_parts = bundle.generator_loss(fake, x, y, mask)
        metrics["g_total"] = _to_scalar(g_loss)
        metrics.update({"g_" + key: _to_scalar(value) for key, value in g_parts.items()})
        g_loss.backward()
        metrics["g_gradient_norm"] = _check_gradient(bundle.generator.parameters(), clip_norm)
    finally:
        if discriminator is not None:
            discriminator.requires_grad_(True)
    if d_optimizer is not None:
        d_optimizer.step()
    g_optimizer.step()
    return metrics


def validate(bundle, store, windows, channel_map, device, amp=False, preview_count=3):
    if not windows:
        raise ValueError("Fixed validation windows must not be empty")
    previous_mode = bundle.generator.training
    bundle.generator.eval()
    totals, previews = {}, []
    selected_ids = []
    for site in sorted({window["site"] for window in windows}):
        selected_ids.append(next(window["id"] for window in windows if window["site"] == site))
    selected_ids += [window["id"] for window in windows if window["id"] not in selected_ids]
    selected_ids = set(selected_ids[:preview_count])
    try:
        with torch.no_grad():
            for window in windows:
                x, y, mask = _batch(store, [window], channel_map, bundle.input_range, device)
                with _autocast(device, amp):
                    fake = bundle.generator(x)
                pred = _range_to_zero_one(fake, bundle.input_range)
                target = _range_to_zero_one(y, bundle.input_range)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("Non-finite validation prediction")
                error = (pred.clamp(0, 1).double() - target.double()).square() * mask.double()
                squared, count = totals.setdefault(window["site"], [0.0, 0])
                totals[window["site"]] = [squared + float(error.sum().cpu()),
                                           count + int(mask.sum().cpu()) * 3]
                if window["id"] in selected_ids:
                    previews.append({"window": window,
                                     "input": _range_to_zero_one(x[0], bundle.input_range).cpu().numpy(),
                                     "target": target[0].cpu().numpy(),
                                     "prediction": pred[0].clamp(0, 1).cpu().numpy(),
                                     "mask": mask[0].cpu().numpy()})
    finally:
        bundle.generator.train(previous_mode)
    scores = {site: -10 * math.log10(max(squared / count, 1e-16))
              for site, (squared, count) in totals.items()}
    return {"site_equal_psnr": float(np.mean(list(scores.values()))),
            "per_site_psnr": scores, "per_site_channel_samples": {site: v[1] for site, v in totals.items()},
            "window_count": len(windows), "purpose": "validation_checkpoint_selection_only"}, previews


def _save_previews(run_dir, update, previews, validation):
    directory = run_dir / "previews"
    directory.mkdir(exist_ok=True)
    rows = []
    arrays = {}
    for index, preview in enumerate(previews):
        tiles = [preview["input"], preview["target"], preview["prediction"],
                 np.repeat(preview["mask"], 3, axis=0)]
        rows.append(np.concatenate(tiles, axis=2))
        arrays.update({str(index) + "_" + name: preview[name] for name in ("input", "target", "prediction", "mask")})
    stem = "validation_%08d" % update
    if rows:
        canvas = (np.clip(np.concatenate(rows, axis=1).transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        with atomic_destination(directory / (stem + ".png")) as temporary:
            Image.fromarray(canvas).save(temporary, format="PNG")
        with atomic_destination(directory / (stem + ".npz")) as temporary:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
    atomic_json(directory / (stem + ".json"), {"update": update, "validation": validation,
                 "columns": ["input_composite", "rgb_target", "rgb_prediction", "valid_mask"],
                 "windows": [preview["window"] for preview in previews]})


def _reconcile_log(path, update):
    """Keep failed-attempt log rows separately; scientific curves use replayed rows."""
    path = Path(path)
    if not path.exists():
        return
    retained, discarded = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            import json
            record = json.loads(line)
            (retained if int(record.get("update", -1)) <= update else discarded).append(line)
        except (ValueError, TypeError):
            discarded.append(line)
    if discarded:
        with (path.parent / (path.stem + ".interrupted_rows.jsonl")).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(discarded) + "\n")
        with atomic_destination(path) as temporary:
            temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")


def _new_interval():
    return {"start_update": None, "updates": 0, "samples": 0,
            "loss_sums": {}, "step_seconds_sum": 0.0, "lr_used_first": None}


def _accumulate_interval(interval, update, losses, step_seconds, samples, lr_used):
    if interval["updates"] == 0:
        interval["start_update"] = update
        interval["lr_used_first"] = lr_used
    elif set(interval["loss_sums"]) != set(losses):
        raise ValueError("Training loss component names changed inside a logging interval")
    for name, value in losses.items():
        interval["loss_sums"][name] = interval["loss_sums"].get(name, 0.0) + value
    interval["updates"] += 1
    interval["samples"] += samples
    interval["step_seconds_sum"] += step_seconds


def _interval_record(interval, update, lr_used_last, lr_next, elapsed, windows, transforms, input_range):
    count = interval["updates"]
    if count < 1:
        raise ValueError("Cannot log an empty training interval")
    return {"update": update, "interval_start_update": interval["start_update"],
            "interval_end_update": update, "interval_updates": count,
            "interval_samples": interval["samples"], "aggregation": "mean_over_completed_updates",
            "input_range": input_range,
            **{name: value / count for name, value in interval["loss_sums"].items()},
            "step_seconds": interval["step_seconds_sum"] / count,
            "interval_step_seconds": interval["step_seconds_sum"],
            "elapsed_seconds": elapsed,
            "lr_used": interval["lr_used_first"], "lr_used_last": lr_used_last, "lr_next": lr_next,
            "lr_summary": "first_used_last_used_and_next_update_boundaries_not_averages",
            "sample_ids": [window["id"] for window in windows], "transforms": transforms,
            "sample_trace_scope": "last_batch_only"}


def _complete_verified(run_dir, identity):
    status_path = run_dir / "training_status.json"
    if not status_path.exists():
        return None
    status = read_json(status_path)
    if status.get("identity") != identity:
        raise IdentityMismatch("Existing task has a different scientific identity")
    if status.get("status") != "training_complete":
        return None
    artifacts = status.get("artifacts", {})
    if not artifacts or status.get("best_file") not in artifacts:
        raise RuntimeError("Completed training has no required best-model artifact")
    for name, digest in artifacts.items():
        path = run_dir / name
        if not path.is_file() or sha256(path) != digest:
            if name.startswith("checkpoints/") or name == status.get("best_file"):
                # A completed record is not authority to skip damaged states.
                # Let the same verified previous-generation recovery run below.
                return None
            raise RuntimeError("Completed training artifact missing or corrupt: " + name)
    return dict(status, status="already_trained", best_path=str(run_dir / status["best_file"]))


def train_task(task, config, store, protocol, run_dir, device, bundle_factory=None, stop_after=None):
    """Train/resume one task; optional factory/stop_after are bounded test hooks.

    config.max_updates is mandatory. `stop_after` pauses after a complete update,
    without altering the scientific budget or learning-rate schedule.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    with file_lock(run_dir / ".task.lock"):
        return _train_locked(task, config, store, protocol, run_dir, device, bundle_factory, stop_after)


def _train_locked(task, config, store, protocol, run_dir, device, bundle_factory, stop_after):
    channel_map = list(task["channel_map"])
    if len(channel_map) != 3 or not all(int(band) in range(1, 6) for band in channel_map):
        raise ValueError("channel_map must contain exactly three B1..B5 band numbers")
    if not protocol.get("train_windows") or not protocol.get("val_windows"):
        raise ValueError("Training and fixed validation windows are both required")
    data_identity = config.get("data_identity", getattr(store, "identity", None))
    if data_identity is None:
        raise ValueError("A verified data-content identity is required")
    parameters = _resolved_parameters(task, config)
    identity, identity_record = training_identity(task, config, protocol, data_identity)
    complete = _complete_verified(run_dir, identity)
    if complete:
        return complete
    identity_path = run_dir / "training_identity.json"
    if identity_path.exists() and read_json(identity_path).get("identity") != identity:
        raise IdentityMismatch("Configuration/code/data changed; refusing to mix experiments")
    manager = CheckpointManager(run_dir / "checkpoints", identity)
    status_path = run_dir / "training_status.json"
    if status_path.exists() and not manager.exists:
        raise RuntimeError("An interrupted task has no committed checkpoint; refusing silent restart")
    atomic_json(identity_path, {"identity": identity, "scientific": identity_record})
    atomic_json(run_dir / "resolved_config.json", {"task": task, "scientific_parameters": parameters,
                "operational": {key: config.get(key) for key in
                                ("checkpoint_seconds", "log_every", "weights_dir", "stop_file")}})
    seed = int(task.get("train_seed", 11))
    seed_everything(seed, device)
    if bundle_factory is None:
        from .modeling import build_bundle
        bundle_factory = build_bundle
    bundle = bundle_factory(task["model"], device, config.get("weights_dir"),
                            parameters["model_config"], training=True)
    if bundle.input_range not in ("zero_one", "minus_one_one"):
        raise ValueError("Model bundle must declare its intensity range")
    amp = parameters["amp_bf16"] and device.type == "cuda"
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but this CUDA device does not support it")
    generator, discriminator = bundle.generator, bundle.discriminator
    options = {"lr": parameters["lr"], "betas": tuple(parameters["betas"]),
               "weight_decay": parameters["weight_decay"]}
    g_optimizer = torch.optim.Adam(generator.parameters(), **options)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), **options) if discriminator is not None else None
    budget = parameters["max_updates"]
    midpoint = budget // 2
    lr_factor = lambda completed: 1.0 if completed <= midpoint else max(0.0, (budget - completed) / max(1, budget - midpoint))
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lr_factor)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lr_factor) if d_optimizer is not None else None
    sampler = SiteBalancedSampler(protocol["train_windows"], seed)
    augmentation_rng = random.Random(seed + 100003)
    best, history, done, elapsed_offset = None, [], 0, 0.0
    metric_interval = _new_interval()
    if manager.exists:
        state = manager.load(map_location=device)
        if state.get("device_type") != device.type:
            raise IdentityMismatch("CPU/GPU execution mode changed; refusing a numerically different resume")
        if bool(state.get("amp_enabled")) != amp:
            raise IdentityMismatch("Actual AMP mode changed; refusing incompatible resume")
        generator.load_state_dict(state["generator"], strict=True)
        g_optimizer.load_state_dict(state["g_optimizer"])
        g_scheduler.load_state_dict(state["g_scheduler"])
        if discriminator is not None:
            discriminator.load_state_dict(state["discriminator"], strict=True)
            d_optimizer.load_state_dict(state["d_optimizer"])
            d_scheduler.load_state_dict(state["d_scheduler"])
        if hasattr(bundle, "load_state_dict_extra"):
            bundle.load_state_dict_extra(state.get("bundle_extra", {}))
        sampler.load_state_dict(state["sampler"])
        augmentation_rng.setstate(state["augmentation_rng"])
        restore_rng(state["rng"])
        best, history, done, elapsed_offset = state["best"], state["validation_history"], state["updates_done"], state["elapsed_seconds"]
        metric_interval = state["metric_interval"]
        for name in ("train_metrics.jsonl", "validation_metrics.jsonl"):
            _reconcile_log(run_dir / name, done)
        append_jsonl(run_dir / "events.jsonl", {"event": "resume", "utc": utc_now(), "update": done,
                      "recovery": manager.loaded_from})
        del state
    generator.train()
    if discriminator is not None:
        discriminator.train()
    started, last_saved_at = time.perf_counter(), time.perf_counter()
    checkpoint_seconds = float(config.get("checkpoint_seconds", 600.0))
    log_every = int(config.get("log_every", 50))
    if checkpoint_seconds <= 0 or log_every < 1:
        raise ValueError("Checkpoint seconds and log interval must be positive")
    milestone_updates = {max(1, math.ceil(budget * part / 4)) for part in range(1, 5)}
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    def save_status(status, **extra):
        record = {"status": status, "identity": identity, "task_id": task.get("task_id"),
                  "updated_utc": utc_now(), "updates_done": done, "max_updates": budget,
                  "last_committed_update": manager.loaded_from["updates_done"] if manager.loaded_from else None,
                  "best": best, **extra}
        atomic_json(status_path, record)
        return record

    def save_state():
        nonlocal last_saved_at
        if not _state_finite(generator) or (discriminator is not None and not _state_finite(discriminator)):
            raise FloatingPointError("Non-finite model state; retaining the preceding valid checkpoint")
        state = {"schema": "mswir-training-state-v1", "identity": identity, "updates_done": done,
                 "device_type": device.type, "amp_enabled": amp,
                 "generator": generator.state_dict(), "g_optimizer": g_optimizer.state_dict(),
                 "g_scheduler": g_scheduler.state_dict(),
                 "discriminator": discriminator.state_dict() if discriminator is not None else None,
                 "d_optimizer": d_optimizer.state_dict() if d_optimizer is not None else None,
                 "d_scheduler": d_scheduler.state_dict() if d_scheduler is not None else None,
                 "rng": capture_rng(include_cuda=device.type == "cuda"), "sampler": sampler.state_dict(),
                 "augmentation_rng": augmentation_rng.getstate(), "best": best,
                 "metric_interval": metric_interval,
                 "validation_history": history, "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                 "bundle_extra": bundle.state_dict_extra() if hasattr(bundle, "state_dict_extra") else {}}
        manager.save(state)
        last_saved_at = time.perf_counter()

    def save_model(name):
        if not _state_finite(generator):
            raise FloatingPointError("Refusing to save a non-finite generator")
        path = models_dir / name
        digest = atomic_torch_save(path, {"generator": generator.state_dict(), "identity": identity,
                    "task": task, "step": done, "input_range": bundle.input_range})
        return {"file": str(path.relative_to(run_dir)).replace("\\", "/"), "sha256": digest, "update": done}

    def stop_requested():
        return (run_dir / "STOP_REQUESTED").exists() or bool(config.get("stop_file") and Path(config["stop_file"]).exists())

    if not manager.exists:
        save_state()  # Initial state makes even a first-update interruption recoverable.
    save_status("training")
    try:
        for update in range(done + 1, budget + 1):
            if stop_requested():
                save_state()
                return save_status("paused")
            step_started = time.perf_counter()
            windows = sampler.sample(parameters["batch_size"])
            transforms = [augmentation_rng.randrange(8) for _ in windows]
            x, y, mask = _batch(store, windows, channel_map, bundle.input_range, device, transforms)
            lr_used = float(g_optimizer.param_groups[0]["lr"])
            losses = training_step(bundle, x, y, mask, g_optimizer, d_optimizer, device, amp,
                                   parameters["gradient_clip_norm"])
            g_scheduler.step()
            if d_scheduler is not None:
                d_scheduler.step()
            done = update
            if device.type == "cuda":
                # Timing measures a completed update, not asynchronous enqueue time.
                torch.cuda.synchronize(device)
            _accumulate_interval(metric_interval, update, losses, time.perf_counter() - step_started,
                                 len(windows), lr_used)
            validation_due = update % parameters["validation_every"] == 0 or update == budget
            if update % log_every == 0 or update == 1 or validation_due:
                append_jsonl(run_dir / "train_metrics.jsonl",
                             _interval_record(metric_interval, update, lr_used,
                                float(g_optimizer.param_groups[0]["lr"]),
                                elapsed_offset + time.perf_counter() - started,
                                windows, transforms, bundle.input_range))
                metric_interval = _new_interval()
            if validation_due:
                metrics, previews = validate(bundle, store, protocol["val_windows"], channel_map, device, amp,
                                              int(config.get("preview_count", 3)))
                record = {"update": update, **metrics}
                history.append(record)
                append_jsonl(run_dir / "validation_metrics.jsonl", record)
                _save_previews(run_dir, update, previews, metrics)
                if best is None or metrics["site_equal_psnr"] > best["site_equal_psnr"]:
                    best = dict(save_model("best_%08d.pt" % update), site_equal_psnr=metrics["site_equal_psnr"])
                print("%s update %d/%d validation PSNR %.4f" % (task.get("task_id", "task"), update, budget,
                                                               metrics["site_equal_psnr"]), flush=True)
            if update in milestone_updates:
                artifact = save_model("milestone_%08d.pt" % update)
                atomic_json(models_dir / ("milestone_%08d.json" % update), artifact)
            # Validation commits its selected best model immediately. Routine
            # full-state saves otherwise follow wall time, not a heavy step rule.
            if validation_due or update in milestone_updates or time.perf_counter() - last_saved_at >= checkpoint_seconds:
                save_state()
                save_status("training")
            if stop_after is not None and update >= stop_after and update < budget:
                if manager.loaded_from["updates_done"] != done:
                    save_state()
                return save_status("paused")
        if manager.loaded_from["updates_done"] != done:
            save_state()
        if best is None:
            raise RuntimeError("Training completed without a validation-selected generator")
        atomic_json(run_dir / "validation_history.json", history)
        atomic_json(models_dir / "best.json", best)
        artifacts = {name: sha256(run_dir / name) for name in
                     (best["file"], "training_identity.json", "validation_history.json", "train_metrics.jsonl",
                      "validation_metrics.jsonl", "checkpoints/index.json")}
        latest = manager.loaded_from
        artifacts["checkpoints/" + latest["file"]] = latest["sha256"]
        result = save_status("training_complete", artifacts=artifacts, best_file=best["file"],
                             best_path=str(run_dir / best["file"]))
        return result
    except BaseException as error:
        append_jsonl(run_dir / "events.jsonl", {"event": "interrupted", "utc": utc_now(), "update": done,
                    "last_committed_update": manager.loaded_from["updates_done"] if manager.loaded_from else None,
                    "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        save_status("interrupted", error_type=type(error).__name__, error=str(error))
        raise
