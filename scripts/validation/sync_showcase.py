"""Regenerate the fixed README suite on one idle GPU; stop on any foreign process.

Run from the repository root. Every case executes deployment.run_sync, including
its actual tracker/physics loop. No seed search, early termination or pose edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("deployment/sync_flow.json"))
    parser.add_argument("--cases", type=Path, default=Path("docs/media/flow/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sync-showcase"))
    parser.add_argument("--gpu", type=int, required=True, help="Physical nvidia-smi index; must have no process")
    parser.add_argument("--names", nargs="+", help="Run an explicit subset; never select by outcome")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if args.names:
        if set(args.names) - {case["name"] for case in cases}:
            parser.error("Unknown case name")
        cases = [case for case in cases if case["name"] in args.names]
    config = json.loads(args.config.read_text())
    if (config["backend"], config["replan_period"], config["lowpass_hz"], config["steps"]) != ("tensorrt-fp16", 0.2, 8.0, 10):
        parser.error("README suite requires TRT FP16, 0.2 s, 8 Hz suffix filter and Euler-10")
    args.output.mkdir(parents=True, exist_ok=True)
    uuid = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu),
                   "MUJOCO_EGL_DEVICE_ID": str(args.gpu), "MUJOCO_GL": "egl",
                   "OMP_NUM_THREADS": "1", "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "TOKENIZERS_PARALLELISM": "false"}
    manifest = {"config": config, "gpu_uuid": uuid, "physical_gpu": args.gpu,
                "initialization": "tracker default standing pose; no dataset history",
                "selection": "all fixed cases and seeds; no retries selected by motion outcome",
                "rendering": "generated reference | recorded MuJoCo execution, matched world camera, 25 fps, normal speed",
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted([*Path('deployment').glob('*.py'),
                                                   *Path('src').rglob('*.py'), Path(__file__)])},
                "clips": []}
    with (args.output / "gpu_guard.jsonl").open("a", buffering=1) as guard:
        for case in cases:
            command = [sys.executable, "-m", "deployment.run_sync", "--config", str(args.config),
                       "--output", str(args.output / case["name"]), "--prompt", case["prompt"],
                       "--seed", str(case["seed"]), "--duration", str(case["duration"]), "--showcase"]
            process = None
            with (args.output / (case["name"] + ".log")).open("w") as log:
                while True:
                    snapshot = ET.fromstring(subprocess.check_output([
                        "nvidia-smi", "-i", str(args.gpu), "-q", "-x",
                    ], text=True))
                    # XML includes both compute and graphics processes. All GPU
                    # work in run_sync (including EGL) belongs to its one PID.
                    pids = [int(pid.text) for pid in snapshot.findall(".//process_info/pid")]
                    foreign = [pid for pid in pids if process is None or pid != process.pid]
                    guard.write(json.dumps({"time_ns": time.time_ns(), "case": case["name"],
                                            "gpu_uuid": uuid, "pids": pids, "foreign": foreign}) + "\n")
                    if foreign:
                        if process is not None and process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait()
                        raise SystemExit(f"GPU {args.gpu} occupied by {foreign}; stopped without accepting this case")
                    if process is None:
                        print("Running", case["name"], "on", uuid, flush=True)
                        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                    elif process.poll() is not None:
                        if process.returncode:
                            raise SystemExit(f"{case['name']} failed ({process.returncode}); see its log")
                        break
                    time.sleep(0.5)
            report = json.loads((args.output / (case["name"] + ".json")).read_text())
            if not manifest["clips"]:
                # Check availability and start the child before large model I/O.
                # The generator already hashes its checkpoint while owning CUDA.
                manifest["asset_sha256"] = {
                    "checkpoint": report["checkpoint"]["checkpoint_sha256"],
                    **{key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                       for key in ("tracker", "scene")},
                    "tracker_external": hashlib.sha256(Path(config["tracker"]).with_name(
                        "4de98d3c-9d3c-11f1-aff4-06f301228322").read_bytes()).hexdigest(),
                }
            import imageio_ffmpeg

            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(case.get("preview_start_s", 0)), "-i",
                str(args.output / (case["name"] + ".mp4")), "-t", "8", "-vf",
                "fps=8,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                str(args.output / (case["name"] + ".gif")),
            ], check=True)
            manifest["clips"].append({**case, "metrics": {key: report[key] for key in (
                "first_failure_s", "failure_screen", "root_height_min_m", "joint_rmse_rad",
                "root_position_rmse_m", "control_overruns", "deadline_misses", "generation_ms")},
                "files": {suffix: {"bytes": (args.output / (case["name"] + suffix)).stat().st_size,
                                   "sha256": hashlib.sha256((args.output / (case["name"] + suffix)).read_bytes()).hexdigest()}
                          for suffix in (".mp4", ".gif", ".npz", ".json")}})
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(case["name"], "first failure:", report["first_failure_s"], flush=True)


if __name__ == "__main__":
    main()
