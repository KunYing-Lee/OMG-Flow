"""Isolated cce4ad3/main/runtime captures, fixed inputs, and numerical comparison.

Run this script with --source pointing at a git worktree. It inserts that
worktree before importing omg; the oracle never imports the new runtime.
All generated artifacts belong in --output/--input, not in the source tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    golden = sub.add_parser("golden")
    golden.add_argument("--output", type=Path, required=True)
    golden.add_argument("--seed", type=int, default=20261116)
    golden.add_argument("--prefix", type=int, default=25, choices=(0, 25))
    capture = sub.add_parser("capture")
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--implementation", choices=("oracle", "runtime", "sync"), required=True)
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--checkpoint-sha256", default="3f8d9f8a9cbc96cb034e6c0cb2b1bfccd38816a8dca66c68df3f24b7f49f9cde")
    capture.add_argument("--text-cache", type=Path, required=True)
    capture.add_argument("--prompt", default="A person walks forward.")
    capture.add_argument("--text-encoder", type=Path, help="Only for the original sync implementation")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--backend", default="torch-fp32", choices=("torch-fp32", "torch-fp16", "tensorrt-fp16"))
    capture.add_argument("--device", default="cuda")
    capture.add_argument("--steps", type=int, default=10)
    capture.add_argument("--lowpass-hz", type=float)
    capture.add_argument("--engine-cache", default="tensorrt_engine_cache/flow_sync")
    capture.add_argument("--no-cuda-graph", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--tolerance", type=float, default=1e-5)
    compare.add_argument("--report-only", action="store_true", help="Report reduced-precision error without asserting FP32 tolerance")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "golden":
        rng = np.random.default_rng(args.seed)
        t = np.arange(10 + args.prefix, dtype=np.float32) / 50
        q = np.zeros((len(t), 36), dtype=np.float32)
        q[:, :3] = np.stack([.12 * t + 1, .03 * np.sin(t) - .3, .79 + .01 * np.cos(t)], -1)
        # Smooth roll/pitch/yaw exercise gravity projection and body-frame velocities.
        roll, pitch, yaw = .05 * np.sin(2 * t), .08 * np.cos(t), .2 + .15 * t
        cr, cp, cy = np.cos(roll / 2), np.cos(pitch / 2), np.cos(yaw / 2)
        sr, sp, sy = np.sin(roll / 2), np.sin(pitch / 2), np.sin(yaw / 2)
        q[:, 3:7] = np.stack([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                              cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy], -1)
        q[:, 7:] = .1 * np.sin(t[:, None] * 2 + rng.uniform(-1, 1, 29))
        np.savez(args.output, history_qpos=q[:10], committed_qpos=q[10:],
                 noise=rng.standard_normal((100, 38), dtype=np.float32))
        return
    if args.command == "compare":
        receipts = [json.loads(path.with_suffix(".json").read_text()) for path in (args.reference, args.candidate)]
        for key in ("checkpoint_sha256", "input_sha256", "text_cache_signature", "prompt", "steps", "prefix", "lowpass_hz", "validation_script_sha256"):
            if receipts[0][key] != receipts[1][key]:
                raise ValueError(f"Capture contracts differ: {key}")
        with np.load(args.reference, allow_pickle=False) as ref, np.load(args.candidate, allow_pickle=False) as candidate:
            if set(ref.files) != set(candidate.files):
                raise ValueError("Capture fields differ")
            metrics = {}
            for name in ref.files:
                a, b = ref[name].astype(np.float64), candidate[name].astype(np.float64)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Invalid arrays: {name}")
                delta = np.abs(a - b)
                metrics[name] = {"max_abs": float(delta.max(initial=0)),
                                 "rmse": float(np.sqrt(np.mean(delta ** 2))),
                                 "bitwise_equal": ref[name].tobytes() == candidate[name].tobytes()}
            for name in ("qpos", "body_positions"):
                delta = (ref[name][..., :3] - candidate[name][..., :3] if name == "qpos"
                         else ref[name] - candidate[name])
                distances = np.linalg.norm(delta.astype(np.float64), axis=-1) * 1000
                metrics[name]["position_mm"] = {"max": float(distances.max()),
                    "p95": float(np.percentile(distances, 95)), "rms": float(np.sqrt(np.mean(distances ** 2)))}
            passed = all(metrics[name]["max_abs"] <= args.tolerance for name in ("qpos", "qpos_replay"))
            result = dict(reference=str(args.reference), candidate=str(args.candidate), tolerance=args.tolerance,
                          qpos_passed=passed, metrics=metrics)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not args.report_only and not passed:
            raise SystemExit("qpos parity failed")
        return

    if args.device.startswith("cuda"):
        subprocess.run(["nvidia-smi"], check=True)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                          "--format=csv,noheader"], text=True).strip()
        if active:
            raise SystemExit(f"GPU occupied; no inference started: {active}")
    source = args.source.resolve()
    sys.path[:0] = [str(source / "src"), str(source)]
    import torch
    from omg.generation.flow_reference import FlowReferenceGenerator
    from omg.generation.bones_text import BonesTextCache
    from omg.robots.g1.kinematics import G1Kinematics
    from omg.tracking.holomotion.reference import native_qpos, precompute_reference_features
    from omg.utils import rotation_conversions as rotations

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    with np.load(args.input, allow_pickle=False) as data:
        history, committed, noise = data["history_qpos"], data["committed_qpos"], data["noise"]
    cache = BonesTextCache(args.text_cache, verify_files=True)
    tokens, mask = cache.lookup([args.prompt])
    text = dict(text_tokens=tokens.to(args.device).float(), text_mask=mask.to(args.device))
    with torch.inference_mode():
        if args.implementation == "oracle":
            if args.backend != "torch-fp32" or args.lowpass_hz is not None:
                raise ValueError("oracle is the unmodified FP32 reference API, no filter")
            ref = FlowReferenceGenerator.from_checkpoint(args.checkpoint, args.device).requires_grad_(False)
            qpos = ref.generate_suffix(torch.from_numpy(history)[None].to(args.device),
                torch.from_numpy(committed)[None].to(args.device), torch.from_numpy(noise)[None].to(args.device),
                steps=args.steps, **text)[0].cpu().numpy()
        else:
            config = dict(checkpoint=args.checkpoint, prompt=args.prompt, text_cache=args.text_cache,
                text_encoder=None if args.text_encoder is None else args.text_encoder.resolve(), device=args.device, steps=args.steps, prefix_length=len(committed),
                backend=args.backend, lowpass_hz=args.lowpass_hz, engine_cache=args.engine_cache,
                cuda_graph=not args.no_cuda_graph)
            if args.implementation == "runtime":
                from deployment.flow_runtime import FlowRuntime
                runtime = FlowRuntime(config)
            else:
                from deployment.sync_flow import FlowGenerator
                runtime = FlowGenerator(config)
                # Hold text fixed even though the old constructor loads T5.
                runtime.text_tokens.copy_(text["text_tokens"])
                runtime.text_mask.copy_(text["text_mask"])
            qpos = runtime.generate(history, committed, noise)
        # Second call reuses the runtime/graphs with changed history, orientation,
        # committed poses and noise. An independent oracle process sees the same changes.
        changed = torch.from_numpy(np.concatenate([history, committed]))[None].to(args.device)
        yaw = rotations.axis_angle_to_quaternion(changed.new_tensor([[[0., 0., .6]]]))
        changed = torch.cat([rotations.quaternion_apply(yaw, changed[..., :3]) + changed.new_tensor([.7, -.4, .2]),
                             rotations.quaternion_multiply(yaw, changed[..., 3:7]), changed[..., 7:]], dim=-1)
        changed_noise = np.roll(noise, 7, axis=0).copy()
        if args.implementation == "oracle":
            replay = ref.generate_suffix(changed[:, :10], changed[:, 10:],
                torch.from_numpy(changed_noise)[None].to(args.device), steps=args.steps, **text)[0].cpu().numpy()
        else:
            replay = runtime.generate(changed[0, :10].cpu().numpy(), changed[0, 10:].cpu().numpy(), changed_noise)
        body = G1Kinematics().to(args.device).eval().forward_body_positions(
            torch.from_numpy(qpos)[None].to(args.device))[0].cpu().numpy()
        # Probe changed rotation functions independently from network output.
        angles = torch.tensor([[0, 0, 0], [1e-9, -1e-8, 1e-7], [.1, -.3, 1.2], [3.1415925, 0, 0]],
                              device=args.device, dtype=torch.float32)
        quat = rotations.axis_angle_to_quaternion(angles)
        arrays = dict(qpos=qpos, qpos_replay=replay, body_positions=body,
            rotation_axis_angle=quat.cpu().numpy(),
            rotation_inverse=rotations.quaternion_invert(quat).cpu().numpy(),
            rotation_matrix_to_quaternion=rotations.matrix_to_quaternion(rotations.quaternion_to_matrix(quat)).cpu().numpy())
    # Fixed input, rather than generated qpos, isolates tracking changes.
    known = np.concatenate([history, committed])
    arrays["native_qpos"] = native_qpos(known, 50)
    arrays.update({"tracking_" + k: v for k, v in precompute_reference_features(known, 50, np.arange(29)).items()})
    np.savez(args.output, **arrays)
    receipt = dict(validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source=str(source), source_commit=subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        implementation=args.implementation, backend=args.backend, steps=args.steps, prefix=len(committed),
        lowpass_hz=args.lowpass_hz, cuda_graph=not args.no_cuda_graph, checkpoint_sha256=checkpoint_sha,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), text_cache_signature=cache.signature,
        prompt=args.prompt, torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
        tf32=False, rotation_source=rotations.__file__,
        source_files_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in (
            "src/omg/generation/flow_reference.py", "src/omg/generation/flow.py", "src/omg/utils/rotation_conversions.py",
            "src/omg/pipeline/checkpoint.py", "src/omg/tracking/holomotion/reference.py")})
    if args.implementation == "runtime":
        receipt["runtime_contract"] = runtime.contract
        receipt["runtime_source_sha256"] = hashlib.sha256((source / "deployment/flow_runtime.py").read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
