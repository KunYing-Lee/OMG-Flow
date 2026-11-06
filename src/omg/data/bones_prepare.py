"""Convert the pinned BONES-SEED G1 release into an indexed native50 qpos file."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from omg.robots.g1.constants import G1_JOINT_NAMES

REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
ARCHIVE_SHA = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"
METADATA_SHA = "7cbe5624f91ae19af3feee69cff74b2cf553fb26367c8edc4b674c3d8ea97e7d"
CAPTION_COLUMNS = [f"content_natural_desc_{i}" for i in range(1, 5)]
CSV_COLUMNS = (
    ["Frame"] + [f"root_translate{axis}" for axis in "XYZ"] + [f"root_rotate{axis}" for axis in "XYZ"]
    + [f"{name}_dof" for name in G1_JOINT_NAMES]
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(4 << 20):
            h.update(block)
    return h.hexdigest()


class HashedReader:
    """Hash exactly the compressed source stream supplied to the tar reader."""

    def __init__(self, file):
        self.file = file
        self.sha = hashlib.sha256()

    def read(self, size):
        block = self.file.read(size)
        self.sha.update(block)
        return block


def convert_g1_csv(payload: bytes, nframes: int):
    # Index by joint name, never assume CSV column order.
    df = pd.read_csv(io.BytesIO(payload), dtype=np.float64)
    if set(df.columns) != set(CSV_COLUMNS):
        raise ValueError(f"Unexpected G1 columns: {df.columns.tolist()}")
    data = df[CSV_COLUMNS].to_numpy()
    if (len(data) != nframes or not np.array_equal(data[:, 0], np.arange(nframes))
            or not np.isfinite(data).all()):
        raise ValueError("Frame index, catalog duration or numeric values do not match a complete 120 Hz clip")
    # Exact 120→50 grid, last target timestamp <= last source timestamp.
    times = np.arange((nframes - 1) * 5 // 12 + 1, dtype=np.int64) * 12
    lo, fraction = times // 5, (times % 5 / 5)[:, None]
    hi = np.minimum(lo + 1, nframes - 1)
    xyz = data[:, 1:4] * 0.01
    angles = np.deg2rad(data[:, 7:])
    rot = Rotation.from_euler("xyz", data[:, 4:7], degrees=True)
    quat = Slerp(np.arange(nframes), rot)(times / 5).as_quat()[:, [3, 0, 1, 2]]
    qpos = np.column_stack((
        xyz[lo] + fraction * (xyz[hi] - xyz[lo]), quat,
        angles[lo] + fraction * (angles[hi] - angles[lo]),
    )).astype("<f4")
    return qpos


def prepare(archive, metadata, root, workers=4, limit=None):
    if sha256_file(metadata) != METADATA_SHA:
        raise ValueError("Metadata SHA256 differs from the pinned BONES-SEED revision")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"An existing dataset is already published in {root}")
    md = pd.read_parquet(metadata)
    if (md.move_g1_path.duplicated().any() or md.move_name.duplicated().any()
            or not md.actor_uid.equals(md.take_actor)):
        raise ValueError("Nonunique clip paths/IDs or conflicting actor identity")
    md = md.set_index("move_g1_path")
    captions = sorted({s.strip() for c in CAPTION_COLUMNS for s in md[c]})
    if "" in captions:
        raise ValueError("An empty natural language annotation was found")
    caption_ids = {s: i for i, s in enumerate(captions)}
    (root / "captions.json").write_text(json.dumps(captions, ensure_ascii=False))
    # A whole actor (including every original and its mirror) appears in one split.
    actors = sorted(set(md.actor_uid), key=lambda a: hashlib.sha256(f"bones-actor-v1/{a}".encode()).digest())
    n = len(actors) // 20
    actor_split = {a: ("val" if i < n else "test" if i < 2 * n else "train") for i, a in enumerate(actors)}
    seen = set()
    records = []
    offset = 0
    data_sha = hashlib.sha256()
    stream = nullcontext(sys.stdin.buffer) if str(archive) == "-" else Path(archive).open("rb")
    with stream as file:
        reader = HashedReader(file)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar, (root / "qpos.f32").open("wb") as out, ThreadPoolExecutor(workers) as pool:
            pending = []
            for member in tar:
                if not member.isfile():
                    continue
                path = member.name.removeprefix("./")
                if path not in md.index or path in seen:
                    raise ValueError(f"Unexpected or duplicate archive member {path}")
                seen.add(path)
                row = md.loc[path]
                pending.append((row, path, pool.submit(convert_g1_csv, tar.extractfile(member).read(), row.move_duration_frames)))
                if len(pending) < workers * 2 and len(seen) != (limit or len(md)):
                    continue
                for row, path, future in pending:
                    try:
                        poses = future.result()
                    except ValueError as error:
                        raise ValueError(f"{path}: {error}") from error
                    payload = memoryview(poses).cast("B")
                    data_sha.update(payload)
                    out.write(payload)
                    records.append({
                        "clip_id": row.move_name, "source_path": path, "actor": row.actor_uid,
                        "original_take": row.take_org_name, "take_date": int(row.take_date),
                        "mirror": bool(row.is_mirror), "source_frames": int(row.move_duration_frames),
                        "offset": offset, "frames": len(poses), "split": actor_split[row.actor_uid],
                        "pose_sha256": hashlib.sha256(payload).hexdigest(),
                        "caption_ids": [caption_ids[row[c].strip()] for c in CAPTION_COLUMNS],
                    })
                    offset += len(poses)
                pending.clear()
                if len(seen) % 10000 < workers * 2:
                    print(f"{len(seen)} clips, {offset} frames", flush=True)
                if limit is not None and len(seen) == limit:
                    break
        if limit is None:
            # Tar parsing stops at its end marker, before all gzip padding has necessarily been read.
            while reader.read(4 << 20):
                pass
            if reader.sha.hexdigest() != ARCHIVE_SHA:
                raise ValueError("G1 archive SHA256 differs from the pinned BONES-SEED revision")
    if len(records) != (limit or len(md)):
        raise ValueError(f"Converted {len(records)} clips, expected {limit or len(md)}")
    clips = pd.DataFrame(records)
    if (clips.groupby("pose_sha256").split.nunique() != 1).any():
        raise ValueError("Identical pose sequences occur in different actor splits")
    clips.to_parquet(root / "clips.parquet", index=False)
    counts = clips.groupby("split").agg(clips=("clip_id","count"), actors=("actor","nunique"), frames=("frames","sum"))
    manifest = {
        "format": "bones-g1-native50-v1", "fps": 50, "source_fps": 120, "frames": offset, "clips": len(clips),
        "source": "bones-studio/seed", "revision": REVISION, "complete": limit is None,
        "archive_sha256": ARCHIVE_SHA if limit is None else None, "metadata_sha256": METADATA_SHA,
        "qpos_sha256": data_sha.hexdigest(), "index_sha256": sha256_file(root / "clips.parquet"),
        "captions_sha256": sha256_file(root / "captions.json"), "splits": counts.to_dict("index"),
    }
    # Publish the manifest only after the packed buffer and index are complete.
    (root / "manifest.tmp").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "manifest.tmp").replace(root / "manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", required=True, help="Pinned g1.tar.gz, or - for a compressed stream on stdin")
    cli.add_argument("--metadata", required=True)
    cli.add_argument("--root", required=True)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--limit", type=int, help="First N clips for an explicitly incomplete smoke dataset")
    prepare(**vars(cli.parse_args()))
