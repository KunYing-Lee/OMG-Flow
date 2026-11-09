"""Render the selected gallery from saved HoloMotion execution, without simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np

from omg.tracking.holomotion.runtime import build_g1_state_handles, set_g1_qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=Path("docs/media/flow"))
    parser.add_argument("--scene", type=Path, default=Path("assets/holomotion/g1_29dof/scene_29dof.xml"))
    args = parser.parse_args()
    cases = json.loads((args.media / "cases.json").read_text())
    manifest = json.loads((args.media / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    handles = build_g1_state_handles(model)
    pelvis = model.body("pelvis").id
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.6, 135, -12
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for case in cases:
            path = args.media / case["name"]
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                executed = saved["executed"]
                if executed.shape != (int(case["duration"] * 50), 36) or not np.isfinite(executed).all():
                    raise ValueError(f"Invalid 50 Hz execution trajectory: {path}")
                np.testing.assert_array_equal(np.diff(saved["timestamps_ns"]), 20_000_000)
            with imageio.get_writer(
                str(path.with_suffix(".mp4")), fps=25, codec="libx264", quality=None,
                ffmpeg_params=["-crf", "26", "-preset", "slow", "-movflags", "+faststart"],
                macro_block_size=2,
            ) as writer:
                for qpos in executed[::2]:
                    set_g1_qpos(model, data, handles, qpos)
                    camera.lookat[:] = data.xpos[pelvis]
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(case["preview_start_s"]),
                "-i", str(path.with_suffix(".mp4")), "-t", "8", "-vf",
                "fps=8,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3", "-final_delay", "12", str(path.with_suffix(".gif")),
            ], check=True)
            clip = next(clip for clip in manifest["clips"] if clip["name"] == case["name"])
            clip["files"] = {
                suffix: {"bytes": path.with_suffix(suffix).stat().st_size,
                         "sha256": hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest()}
                for suffix in (".mp4", ".gif", ".npz", ".json")
            }
            print(case["name"], "rendered", flush=True)
    manifest["presentation"] = {
        "renderer": "scripts/validation/render_gallery.py",
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mujoco": mujoco.__version__,
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "camera": {"target": "executed pelvis world xyz each frame", "distance": 2.6,
                   "azimuth": 135, "elevation": -12},
        "mp4": {"width": 480, "height": 360, "fps": 25, "duration_s": 20, "crf": 26},
        "gif": {"width": 320, "height": 240, "fps": 8, "duration_s": 8, "colors": 64},
        "note": "Recorded execution only; no overlays, pose edits, smoothing or speed changes."
    }
    (args.media / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
