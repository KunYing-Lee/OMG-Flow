"""Run a synchronous Flow control loop, then render its recorded states offline.

Usage: python -m deployment.run_sync --config deployment/sync_flow.json --output docs/deploy_sync/sync_02s
"""

import argparse
import json
import logging
import os
from pathlib import Path
from time import monotonic_ns, perf_counter_ns, sleep

import numpy as np

from deployment.sync_flow import FlowGenerator, SyncScheduler, TICK_NS


def run_control(generator, tracker, robot, standing, *, duration, prefix_length,
                replan_period, seed, epoch_ns, overrun="raise", pace=True):
    """Same read → synchronous replan → tracker → write path for sim and SDK I/O.

    Render/encode only after this function returns. Simulation advances exactly
    20 ms per write; wall-clock overruns remain visible and are never hidden by
    dropping control ticks, shifting reference times, or catching up physics.
    """
    if overrun not in ("raise", "log") or duration <= 0:
        raise ValueError("Require positive duration and overrun=raise|log")
    if tracker.metadata.context_length > generator.history:
        raise ValueError("Flow history must cover tracker reference context")
    scheduler = SyncScheduler(standing, generator.history, generator.horizon, prefix_length,
                              replan_period, tracker.metadata.n_fut_frames, seed, epoch_ns)
    references, executed, targets, ticks, replans = [], [], [], [], []
    features = None
    for tick in range(round(duration * 50)):
        deadline = epoch_ns + (tick + 1) * TICK_NS
        if pace:
            sleep(max(0, (deadline - TICK_NS - monotonic_ns()) / 1e9))
        started = perf_counter_ns()
        state = robot.read_state()
        read_ms = (perf_counter_ns() - started) / 1e6
        generation_ms = reference_features_ms = 0.0
        if scheduler.replan_due:
            generation_ms = scheduler.replan(generator)
            feature_started = perf_counter_ns()
            features = tracker.reference_features(scheduler.qpos)
            reference_features_ms = (perf_counter_ns() - feature_started) / 1e6
            planning_ms = (perf_counter_ns() - started) / 1e6
            replans.append({
                "tick": tick, "timestamp_ns": int(scheduler.timestamps_ns[scheduler.cursor]),
                "committed_end_ns": int(scheduler.timestamps_ns[scheduler.cursor + prefix_length - 1]),
                "suffix_start_ns": int(scheduler.timestamps_ns[scheduler.cursor + prefix_length]),
                "generation_ms": generation_ms, "planning_ms": planning_ms,
                "reference_features_ms": reference_features_ms,
                "splice_joint_step_max_rad": float(np.max(np.abs(np.diff(
                    scheduler.qpos[scheduler.cursor + prefix_length - 1:scheduler.cursor + prefix_length + 1, 7:], axis=0)))),
            })
            if planning_ms > 20:
                message = f"REPLAN OVERRUN tick={tick}: generation={generation_ms:.3f} ms planning={planning_ms:.3f} ms > 20 ms"
                if overrun == "raise":
                    raise RuntimeError(message)
                logging.error(message)
        target = tracker.targets(state, features, scheduler.cursor)
        write_started = perf_counter_ns()
        robot.write_joint_targets(target)
        write_ms = (perf_counter_ns() - write_started) / 1e6
        elapsed_ms = (perf_counter_ns() - started) / 1e6
        late_ms = max(0, (monotonic_ns() - deadline) / 1e6) if pace else None
        if elapsed_ms > 20 or (late_ms is not None and late_ms > 0):
            message = f"CONTROL OVERRUN tick={tick}: compute={elapsed_ms:.3f} ms deadline_lateness={late_ms} ms"
            if overrun == "raise":
                raise RuntimeError(message)
            logging.error(message)
        # Compare state and reference at the SAME time, before applying this tick.
        references.append(scheduler.qpos[scheduler.cursor].copy())
        executed.append(state.qpos.copy())
        targets.append(target)
        ticks.append({"tick": tick, "timestamp_ns": int(scheduler.timestamps_ns[scheduler.cursor]),
                      "state_timestamp_ns": state.timestamp_ns, "compute_ms": elapsed_ms, "late_ms": late_ms,
                      "generation_ms": generation_ms, "reference_features_ms": reference_features_ms,
                      "observation_ms": tracker.observation_ms, "tracker_onnx_ms": tracker.onnx_ms,
                      "feature_build_ms": reference_features_ms + tracker.observation_ms,
                      "physics_io_ms": read_ms + write_ms,
                      "other_ms": elapsed_ms - generation_ms - reference_features_ms - tracker.observation_ms
                                  - tracker.onnx_ms - read_ms - write_ms})
        scheduler.advance()
    return np.asarray(references), np.asarray(executed), np.asarray(targets), ticks, replans


def render_video(robot, reference, executed, replans, ticks, path, period):
    import imageio.v2 as imageio
    import mujoco

    from omg.tracking.holomotion.runtime import set_g1_qpos
    from omg.tracking.holomotion.video import draw_overlay

    data = mujoco.MjData(robot.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.6, 90, -15
    by_tick = {row["tick"]: row for row in replans}
    latest = replans[0]
    with mujoco.Renderer(robot.model, height=480, width=640) as renderer, imageio.get_writer(
        str(path), fps=50, codec="libx264", quality=7, ffmpeg_log_level="warning", macro_block_size=16,
    ) as writer:
        for tick, (ref, actual) in enumerate(zip(reference, executed)):
            latest = by_tick.get(tick, latest)
            panels = []
            # Matched world camera preserves root-position error in the paired view.
            camera.lookat[:] = [ref[0], ref[1], 0.85]
            for label, qpos in (("Generated reference (kinematic)", ref), ("HoloMotion executing (MuJoCo)", actual)):
                set_g1_qpos(robot.model, data, robot.g1, qpos)
                renderer.update_scene(data, camera=camera)
                panels.append(draw_overlay(renderer.render().copy(), [
                    label, f"SIM t={tick / 50:.2f}s  tick={tick}  replan={period:.1f}s",
                    f"{'REPLAN NOW' if tick in by_tick else 'last replan'} @{latest['tick']}: {latest['generation_ms']:.2f} ms",
                    f"Full tick {ticks[tick]['compute_ms']:.2f} ms: {'OVERRUN' if ticks[tick]['compute_ms'] > 20 else 'within 20 ms'}",
                    "50 Hz simulation time; see JSON for wall-clock deadlines",
                ]))
            writer.append_data(np.concatenate(panels, axis=1))


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    from deployment.sync_robot import HoloTracker, MujocoRobot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period", type=float)
    parser.add_argument("--steps", type=int, choices=(5, 10))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--backend", choices=("torch-fp32", "torch-fp16", "tensorrt-fp16"))
    parser.add_argument("--lowpass-hz", type=float, help="0 disables filtering")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key, value in (("replan_period", args.period), ("steps", args.steps), ("duration", args.duration), ("backend", args.backend)):
        if value is not None:
            config[key] = value
    if args.lowpass_hz is not None:
        config["lowpass_hz"] = args.lowpass_hz or None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = FlowGenerator(config)
    tracker = HoloTracker(config["tracker"])
    robot = MujocoRobot(config["scene"], tracker.metadata, 0, config["standing_height"])
    # Warm tracker kernels/cache allocation before starting its real sequence.
    features = tracker.reference_features(np.repeat(robot.standing[None], generator.history + generator.horizon, axis=0))
    for _ in range(3):
        tracker.targets(robot.read_state(), features, generator.history)
    tracker.tracker.reset()
    tracker.last_action.fill(0)
    tracker.robot_history.clear()
    epoch_ns = monotonic_ns()
    robot.epoch_ns = epoch_ns
    reference, executed, targets, ticks, replans = run_control(
        generator, tracker, robot, robot.standing, duration=config["duration"],
        prefix_length=config["prefix_length"], replan_period=config["replan_period"], seed=config["seed"],
        epoch_ns=epoch_ns, overrun=config["overrun"], pace=config["pace"],
    )
    # Raw synchronized trajectories allow independent metric/video reconstruction.
    np.savez_compressed(args.output.with_suffix(".npz"), reference=reference, executed=executed, targets=targets,
                        timestamps_ns=np.array([row["timestamp_ns"] for row in ticks]))
    report = {"config": config, "backend": generator.backend, "checkpoint": generator.contract,
              "device": __import__("torch").cuda.get_device_name(), "ticks": ticks, "replans": replans}
    for key in ("generation_ms", "planning_ms"):
        values = np.array([row[key] for row in replans])
        report[key] = dict(zip(("p50", "p95", "max"), map(float, np.percentile(values, [50, 95, 100]))))
    report["latency_ms"] = {}
    for group, rows in (("all_ticks", ticks), ("replan_ticks", [r for r in ticks if r["tick"] % round(config["replan_period"] * 50) == 0])):
        report["latency_ms"][group] = {
            key: {"n": len(rows), **dict(zip(("p50", "p95", "max"), map(float, np.percentile([r[key] for r in rows], [50, 95, 100]))))}
            for key in ("compute_ms", "generation_ms", "feature_build_ms", "reference_features_ms", "observation_ms", "tracker_onnx_ms", "physics_io_ms", "other_ms")
        }
    if generator.trt is not None:
        report["engine"] = {"path": str(generator.trt.path), "cache_hit": generator.trt.cache_hit,
                            "contract": generator.trt.contract}
    fallen = executed[:, 2] < 0.45
    report.update({
        "duration_s": len(ticks) / 50, "control_overruns": sum(row["compute_ms"] > 20 for row in ticks),
        "deadline_misses": sum(row["late_ms"] is not None and row["late_ms"] > 0 for row in ticks),
        "replan_overruns": sum(row["planning_ms"] > 20 for row in replans),
        "root_height_min_m": float(executed[:, 2].min()),
        "first_height_below_045m_s": float(np.flatnonzero(fallen)[0] / 50) if fallen.any() else None,
        "joint_rmse_rad": float(np.sqrt(np.mean((reference[:, 7:] - executed[:, 7:]) ** 2))),
        "root_position_rmse_m": float(np.sqrt(np.mean(np.sum((reference[:, :3] - executed[:, :3]) ** 2, axis=1)))),
    })
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    render_video(robot, reference, executed, replans, ticks, args.output.with_suffix(".mp4"), config["replan_period"])
    print(json.dumps({key: value for key, value in report.items() if key not in ("ticks", "replans")}, indent=2))


if __name__ == "__main__":
    main()
