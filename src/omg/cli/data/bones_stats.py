"""Compute native38 preconditioning from every eligible training window on a fixed grid."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from omg.data.bones import BonesWindows, encode_bones_window


@torch.inference_mode()
def compute(data, history=10, horizon=100, stride=25, batch_size=256):
    windows = BonesWindows(data, "train", history=history, horizon=horizon, stride=stride)
    if len(windows) == 0:
        raise ValueError("No full training windows for native38 statistics")
    total = torch.zeros(38, dtype=torch.float64)
    squares = torch.zeros_like(total)
    for start in range(0, len(windows), batch_size):
        indices = range(start, min(len(windows), start + batch_size))
        qpos = windows.get_batch(indices)["qpos"]
        features = encode_bones_window(qpos, history, torch.zeros(38), torch.ones(38)).double()
        total += features.sum((0, 1))
        squares += features.square().sum((0, 1))
    frames = len(windows) * (history + horizon)
    mean = total / frames
    std = (squares / frames - mean.square()).sqrt()
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Training windows must span all 38 features with finite positive variance")
    return dict(format="bones-native38-stats-v1", split="train", fps=50, history=history, horizon=horizon,
                stride=stride, windows=len(windows), frames=frames, mean=mean.tolist(), std=std.tolist(),
                manifest_sha256=hashlib.sha256((Path(data) / "manifest.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=25)
    args = parser.parse_args()
    stats = compute(args.data, stride=args.stride)
    temporary = args.data / "train_stats.tmp"
    temporary.write_text(json.dumps(stats, indent=2) + "\n")
    temporary.replace(args.data / "train_stats.json")
    print(f"Calibrated from {stats['windows']} train windows, {stats['frames']} frame observations.")
