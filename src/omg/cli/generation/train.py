"""Train the Bones-Seed committed-prefix generator using torchrun."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from omg.core.paths import repo_root
from omg.data.bones import BonesWindows, encode_bones_window
from omg.generation.bones_text import BonesTextCache
from omg.generation.flow import MotionFlow
from omg.generation.flow_reference import FlowReferenceGenerator
from omg.generation.flow_seam import replan_seam_metrics


def batches(num_windows, batch_size, start, stop, seed, rank, world):
    """Step-addressable uniform draws with replacement, independent on each rank."""
    generator = torch.Generator()
    for step in range(start, stop):
        generator.manual_seed(seed + step * world + rank)
        yield torch.randint(num_windows, (batch_size,), generator=generator).tolist()


class WindowBatches:
    def __init__(self, dataset, cache):
        self.dataset = dataset
        self.cache = cache

    def __getitem__(self, indices):
        batch = self.dataset.get_batch(indices)
        text, mask = self.cache.lookup(batch["text"])
        return batch["qpos"], text, mask


@torch.inference_mode()
def evaluate(flow, dataset, cfg, mean, std, rank, world, device):
    # Reuse identical windows, Gaussian draws and times at every checkpoint.
    count = min(len(dataset.dataset), cfg["val_windows"])
    count -= count % world
    if count < world:
        raise ValueError("Need at least one distinct validation window per rank")
    ids = torch.randperm(len(dataset.dataset), generator=torch.Generator().manual_seed(271828))[:count][rank::world]
    generator = torch.Generator(device=device).manual_seed(314159 + rank)
    past, horizon = cfg["history"], cfg["horizon"]
    prefixes = sorted({0, min(25, horizon - 1), min(cfg["model"]["max_prefix_length"], horizon - 1)})
    reference_model = FlowReferenceGenerator(flow, past, horizon, mean, std)
    metrics = {}
    for offset in range(0, len(ids), 16):
        raw, text, mask = dataset[ids[offset:offset+16]]
        poses = encode_bones_window(raw.to(device), past, mean, std)
        history, future = poses[:, :past], poses[:, past:]
        kwargs = {"text_tokens": text.to(device), "text_mask": mask.to(device)}
        noise = torch.randn(future.shape, device=device, generator=generator)
        times = torch.rand(len(future), device=device, generator=generator)
        if offset == 0:
            replan_noise = torch.randn(future.shape, device=device, generator=generator)
            metrics.update(replan_seam_metrics(reference_model, raw.to(device), noise, replan_noise,
                                               steps=cfg["sample_steps"], **kwargs))
        for prefix in prefixes:
            lengths = torch.full((len(future),), prefix, device=device, dtype=torch.long)
            loss = flow(history, future, prefix_lengths=lengths, time=times, noise=noise, **kwargs)
            name = f"flow_mse_p{prefix}"
            metrics[name] = metrics.get(name, 0) + loss * len(future)
            if offset == 0:
                plan = flow.sample(history, future[:, :prefix], horizon, steps=cfg["sample_steps"], noise=noise, **kwargs)
                if not torch.equal(plan[:, :prefix], future[:, :prefix]) or not torch.isfinite(plan).all():
                    raise RuntimeError("Non-finite generated plan or modified committed prefix")
                delta = (plan[:, prefix:] - future[:, prefix:]) * std
                metrics[f"sample_joint_mae_rad_p{prefix}"] = delta[:, :, 9:].abs().mean()
                metrics[f"sample_root_l2_m_p{prefix}"] = delta[:, :, :3].norm(dim=-1).mean()
    vec = torch.stack([value / len(ids) if key.startswith("flow_mse_") else value for key, value in metrics.items()])
    if world > 1:
        dist.all_reduce(vec)
        vec /= world
    if not torch.isfinite(vec).all():
        raise RuntimeError("Non-finite held-out metrics")
    return dict(zip(metrics, vec.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--config", type=Path, default=repo_root() / "configs/flow/bones.yaml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Restore latest.pt and continue at its next update")
    parser.add_argument("--stop-after", type=int, help="Absolute pilot stop; preserves the full-run LR schedule")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    gpu = args.device == "cuda"
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if gpu else torch.device("cpu")
    if gpu:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if gpu else "gloo")
    torch.manual_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(exist_ok=True, parents=True)
    last = args.output / "latest.pt"
    if last.exists() and not args.resume:
        raise FileExistsError(f"Existing run at {args.output}; pass --resume to continue")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if not manifest["complete"] and args.stop_after is None:
        raise ValueError("Full pretraining requires a complete BONES-SEED conversion")
    cache = BonesTextCache(args.text_cache)
    data_cfg = dict(history=cfg["history"], horizon=cfg["horizon"])
    train = BonesWindows(args.data, "train", stride=cfg["train_stride"], **data_cfg)
    val = BonesWindows(args.data, "val", stride=cfg["eval_stride"], **data_cfg)
    stats_path = args.data / "train_stats.json"
    stats = json.loads(stats_path.read_text())
    manifest_sha = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    if (stats["format"], stats["split"], stats["fps"], stats["history"], stats["horizon"], stats["manifest_sha256"]) != (
        "bones-native38-stats-v1", "train", 50, cfg["history"], cfg["horizon"], manifest_sha
    ):
        raise ValueError("Native38 statistics must match this dataset, history and horizon")
    mean = torch.tensor(stats["mean"], device=device)
    std = torch.tensor(stats["std"], device=device)
    if mean.shape != (38,) or std.shape != (38,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Native38 statistics require 38 finite means and positive finite scales")
    flow = MotionFlow(**cfg["model"]).to(device)
    model = DistributedDataParallel(flow, device_ids=[device.index] if gpu else None) if world > 1 else flow
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=gpu)
    provenance = dict(manifest_sha256=manifest_sha,
                      text_manifest_sha256=cache.signature, stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                      world_size=world, device=args.device)
    start, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["config"] != cfg or checkpoint["provenance"] != provenance:
            raise ValueError("Cannot resume with different data, configuration or world size")
        flow.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["step"], checkpoint["best_validation"]
        del checkpoint
    stop = cfg["steps"] if args.stop_after is None else args.stop_after
    if not start < stop <= cfg["steps"]:
        raise ValueError(f"Require {start} < stop <= {cfg['steps']}, got {stop}")
    batch_size = cfg["batch_size"]
    loader = DataLoader(WindowBatches(train, cache), batch_size=None,
                        sampler=batches(len(train), batch_size, start, stop, cfg["seed"], rank, world),
                        num_workers=cfg["workers"], pin_memory=gpu)
    validation = WindowBatches(val, cache)
    def log(event, **values):
        if rank == 0:
            row = json.dumps(dict(event=event, time=time.time(), **values), allow_nan=False)
            print(row, flush=True)
            with (args.output / "metrics.jsonl").open("a") as file:
                file.write(row + "\n")
    source = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True)
    revision = os.environ.get("OMG_FLOW_REVISION") or source.stdout.strip()
    log("start", start=start, stop=stop, source_revision=revision, train_windows=len(train), val_windows=len(val),
        trainable=sum(p.numel() for p in flow.parameters()), config=cfg, provenance=provenance)
    clock, logged_step = time.perf_counter(), start
    for step, (qpos, text, mask) in enumerate(loader, start + 1):
        # Replaying this update uses the same dropout, prefix length, time and noise.
        torch.manual_seed(cfg["seed"] + step * world + rank)
        poses = encode_bones_window(qpos.to(device, non_blocking=True), cfg["history"], mean, std)
        history, future = poses[:, :cfg["history"]], poses[:, cfg["history"]:]
        warm = cfg["warmup"]
        lr = (cfg["lr"] * step / warm if step <= warm else
              cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) *
              (1 + math.cos(math.pi * (step - warm) / (cfg["steps"] - warm))))
        optim.param_groups[0]["lr"] = lr
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device.type, torch.bfloat16, enabled=gpu):
            loss = model(history, future, text_tokens=text.to(device, non_blocking=True), text_mask=mask.to(device, non_blocking=True))
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        optim.step()
        if step == start + 1 or step % cfg["log_every"] == 0 or step == stop:
            reduced = loss.detach().clone()
            if world > 1:
                dist.all_reduce(reduced)
            reduced = reduced.item() / world
            now = time.perf_counter()
            log("train", step=step, loss=reduced, grad_norm=grad.item(), lr=lr,
                windows_per_sec=(step-logged_step)*batch_size*world / (now-clock),
                peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30 if gpu else None)
            clock, logged_step = now, step
        validating = step % cfg["eval_every"] == 0 or step == stop
        improved = False
        if validating:
            flow.eval()
            with torch.autocast(device.type, torch.bfloat16, enabled=gpu):
                metrics = evaluate(flow, validation, cfg, mean, std, rank, world, device)
            flow.train()
            losses = [value for key, value in metrics.items() if key.startswith("flow_mse_")]
            score = sum(losses) / len(losses)
            improved = score < best
            best = min(score, best)
            log("val", step=step, **metrics)
        if step % cfg["save_every"] == 0 or validating:
            if rank == 0:
                payload = dict(model=flow.state_dict(), optimizer=optim.state_dict(), step=step, best_validation=best,
                               config=cfg, model_config=cfg["model"], mean=mean.cpu(), std=std.cpu(), provenance=provenance,
                               source_revision=revision)
                tmp = last.with_suffix(".tmp")
                torch.save(payload, tmp)
                tmp.replace(last)
                if improved:
                    # A hard link keeps the best copy without rewriting the optimizer.
                    tmp = args.output / "best.tmp"
                    tmp.unlink(missing_ok=True)
                    os.link(last, tmp)
                    tmp.replace(args.output / "best.pt")
                log("checkpoint", step=step, bytes=last.stat().st_size, best=improved)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
