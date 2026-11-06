"""BONES-SEED G1 at 50 Hz. Windows are confined to one clip and one actor split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from omg.motion.feature_codec import heading_quaternions
from omg.utils.rotation_conversions import (
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_matrix,
)


def encode_bones_window(qpos: torch.Tensor, history: int, mean: torch.Tensor, std: torch.Tensor):
    """All frames share the xyz and heading anchor at the last history frame."""
    if not 1 <= history <= qpos.shape[-2] or qpos.shape[-1] != 36:
        raise ValueError("Expected [..., time, 36] qpos and a nonempty history inside the window")
    anchor = qpos[..., history - 1 : history, :]
    _, inverse = heading_quaternions(anchor[..., 3:7])
    positions = quaternion_apply(inverse, qpos[..., :3] - anchor[..., :3])
    rotations = quaternion_to_matrix(quaternion_multiply(inverse, qpos[..., 3:7]))
    features = torch.cat((positions, matrix_to_rotation_6d(rotations), qpos[..., 7:]), -1)
    return (features - mean) / std


def load_bones_captions(root: str | Path) -> list[str]:
    return json.loads((Path(root) / "captions.json").read_text())


class BonesWindows(Dataset):
    def __init__(self, root: str | Path, split: str, history=10, horizon=100, stride=25):
        import pyarrow.parquet as pq

        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["format"] != "bones-g1-native50-v1" or manifest["fps"] != 50:
            raise ValueError("Expected a BONES-SEED native50 dataset")
        if split not in ("train", "val", "test") or min(history, horizon, stride) < 1:
            raise ValueError("Expected train/val/test and positive history, horizon and stride")
        self.nframes = history + horizon
        self.stride = stride
        clips = pq.read_table(
            root / "clips.parquet",
            columns=["clip_id", "offset", "frames", "caption_ids"],
            filters=[("split", "=", split), ("frames", ">=", self.nframes)],
        )
        self.ids = clips["clip_id"].to_pylist()
        self.offset = clips["offset"].to_numpy()
        self.ends = np.cumsum(1 + (clips["frames"].to_numpy() - self.nframes) // stride)
        self.caption_ids = np.asarray(clips["caption_ids"].to_pylist(), dtype=np.int32).reshape(-1, 4)
        self.captions = load_bones_captions(root)
        count = manifest["frames"]
        if (root / "qpos.f32").stat().st_size != count * 36 * 4:
            raise ValueError("Packed qpos size differs from the manifest")
        self.qpos = torch.from_file(str(root / "qpos.f32"), size=count * 36, dtype=torch.float32).reshape(-1, 36)
        self.times = torch.arange(self.nframes)
        self.training = split == "train"

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def get_batch(self, indices):
        """Vectorized random windows; gathers one contiguous tensor for device transfer."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("BONES-SEED window index out of range")
        clip = np.searchsorted(self.ends, idx, side="right")
        local = idx - np.where(clip > 0, self.ends[clip - 1], 0)
        start = local * self.stride
        row = torch.from_numpy(self.offset[clip] + start)
        caption = self.caption_ids[clip, (local + clip) % 4 if self.training else 0]
        return {
            "qpos": self.qpos[row[:, None] + self.times],
            "text": [self.captions[i] for i in caption],
            "clip_id": [self.ids[i] for i in clip],
            "start": torch.from_numpy(start),
        }

    def __getitem__(self, index):
        return {k: v[0] for k, v in self.get_batch([index]).items()}
