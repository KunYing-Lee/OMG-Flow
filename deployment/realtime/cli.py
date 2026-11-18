"""Serve, bake, or exercise the native realtime wire contract."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from .schedule import Schedule, motion
from .config import load_config
from .server import serve
from .wire import RobotStateRequest, MotionPlanChunk, decode_message


def request(history, plan_id, schedule, caption):
    switch = schedule.switch_frame(plan_id)
    return RobotStateRequest(history, 50., schedule.request_frame(plan_id),
        prompt=caption, metadata=dict(plan_id=plan_id, switch_frame=switch, period_frames=schedule.period))


def bake(source, output, *, after_end="hold"):
    q = source.bake()
    np.savez(output, qpos_36=q, fps=np.array([50.], np.float32),
        play_frames=np.asarray(source.play_frames), history=source.start_history,
        metadata=np.array([json.dumps({**source.metadata(), "after_end": after_end}, sort_keys=True)]))
    if source.config.robot_xml is not None:
        source.export(q, output)
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("serve", "bake"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--backend", choices=["torch-fp32", "torch-fp16", "tensorrt-fp16"])
        cmd.add_argument("--engine-directory", type=Path)
        cmd.add_argument("--device")
        cmd.add_argument("--switch-seconds", type=float)
        cmd.add_argument("--after-end", choices=["hold", "error"])
        if command == "serve":
            cmd.add_argument("--bind", required=True)
            cmd.add_argument("--log", type=Path, required=True)
            cmd.add_argument("--max-plans", type=int)
            cmd.add_argument("--allow-start-mismatch", action="store_true")
        else:
            cmd.add_argument("--output", type=Path, required=True)
    client = sub.add_parser("client")
    client.add_argument("--connect", required=True)
    client.add_argument("--history", type=Path, required=True, help="npz containing history")
    client.add_argument("--text", required=True)
    client.add_argument("--plans", type=int, default=3)
    client.add_argument("--switch-seconds", type=float, default=.5)
    args = p.parse_args()
    if args.command == "client":
        import zmq
        if args.plans < 1:
            p.error("--plans must be positive")
        schedule = Schedule(args.switch_seconds)
        with np.load(args.history, allow_pickle=False) as z:
            history = z["history"]
        with zmq.Context() as context, context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(args.connect)
            for k in range(args.plans):
                if k:
                    time.sleep(max(0, origin + (k * schedule.period - 25) / 50 - time.perf_counter()))
                r = request(history, k, schedule, args.text)
                start = time.perf_counter()
                socket.send_multipart(r.to_message())
                if not socket.poll(30000 if k == 0 else max(0, int(1000 * (
                    origin + schedule.deadline_frame(k) / 50 - time.perf_counter())))):
                    raise TimeoutError("Planner deadline missed")
                result = MotionPlanChunk.from_message(*decode_message(*socket.recv_multipart()))
                motion(result.qpos_36, 100)
                if (result.fps, result.plan_id, result.request_id, result.metadata["switch_frame"],
                    result.request_tracker_frame, result.metadata["period_frames"]) != (
                    50., k, r.request_id, k * schedule.period, r.tracker_frame, schedule.period):
                    raise ValueError("Mismatched response")
                print(json.dumps(dict(plan_id=k, client_total_ms=(time.perf_counter()-start)*1000)))
                history = result.qpos_36[schedule.period-10:schedule.period].copy()
                if k == 0:
                    origin = time.perf_counter()
        return
    overrides = {key: getattr(args, key) for key in
                 ("backend", "device", "switch_seconds", "after_end") if getattr(args, key) is not None}
    if args.engine_directory is not None:
        overrides["engine_directory"] = str(args.engine_directory.resolve())
    config = load_config(args.config, **overrides)
    if args.command == "serve" and args.max_plans is not None and args.max_plans < 1:
        p.error("--max-plans must be positive")
    from .source import FlowSource

    source = FlowSource(config)
    schedule = Schedule(config.switch_seconds)
    if args.command == "bake":
        bake(source, args.output, after_end=config.after_end)
    else:
        # Warm model kernels/engine before the client's initialization timeout.
        source.reset()
        source.chunk(0, 0)
        serve(source, args.bind, log=args.log, schedule=schedule, after_end=config.after_end,
              max_plans=args.max_plans, allow_start_mismatch=args.allow_start_mismatch, start_gate=config.start_gate)


if __name__ == "__main__":
    main()
