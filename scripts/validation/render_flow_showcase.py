"""Render curated 190.5k physics rollouts; selections and input hashes are recorded."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from omg.tracking.holomotion.runtime import (
    build_g1_state_handles,
    resolve_robot_xml,
    set_g1_qpos,
)

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("docs/media/flow"))
args = parser.parse_args()
src = args.artifacts
out = args.output
out.mkdir(parents=True, exist_ok=True)
m = mujoco.MjModel.from_xml_path(str(resolve_robot_xml(None)))
d = mujoco.MjData(m)
h = build_g1_state_handles(m)
r = mujoco.Renderer(m, height=480, width=640)
c = mujoco.MjvCamera()
c.distance = 3.3
c.azimuth = 135
c.elevation = -15
manifest = json.loads((src / "manifest.json").read_text())
metrics = json.loads((src / "tracking.json").read_text())
records = []
for name, i in [("dance", 7), ("turn", 15), ("jump", 12)]:
    q = np.load(src / f"best_executed_{i}.npy")
    case = manifest["cases"][i]
    metric = next(x for x in metrics if x["label"] == "best" and x["case"] == i)
    writer = imageio.get_writer(
        out / f"{name}.mp4", fps=25, codec="libx264", quality=8, macro_block_size=1
    )
    for t in range(0, 1000, 2):
        set_g1_qpos(m, d, h, q[t])
        mujoco.mj_forward(m, d)
        c.lookat[:] = [q[t, 0], q[t, 1], 0.85]
        r.update_scene(d, c)
        im = r.render().copy()
        cv2.rectangle(im, (0, 0), (640, 64), (244, 246, 248), -1)
        cv2.putText(
            im,
            {
                "dance": "Dance choreography",
                "turn": "Repeated turns",
                "jump": "Upward reach",
            }[name],
            (18, 26),
            0,
            0.7,
            (25, 35, 45),
            2,
        )
        cv2.putText(
            im,
            f"Flow 190.5k | HoloMotion simulation | {t / 50:05.2f}s",
            (18, 51),
            0,
            0.43,
            (65, 75, 85),
            1,
        )
        writer.append_data(im)
    writer.close()
    peak = int(np.argmax(q[:, 2]))
    start = (
        max(0, min(12, peak / 50 - 4))
        if name == "jump"
        else (4 if name == "dance" else 0)
    )
    records.append(
        {
            "name": name,
            "case": case,
            "metrics": metric,
            "checkpoint_sha256": manifest["checkpoints"]["best"]["sha256"],
            "source_sha256": hashlib.sha256(
                (src / f"best_executed_{i}.npy").read_bytes()
            ).hexdigest(),
            "preview_start_s": start,
            "preview_duration_s": 8,
            "full_duration_s": 20,
            "playback_speed": 1,
        }
    )
    print(name, "preview start", start, "peak", peak / 50, flush=True)
r.close()
(out / "manifest.json").write_text(
    json.dumps(
        {
            "selection": "190.5k only; lowest mean root tracking error among three evaluated seeds per displayed category",
            "rendering": "MuJoCo pose rendering of recorded physics execution, no trajectory edits",
            "clips": records,
        },
        indent=2,
    )
)
