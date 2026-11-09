"""Matched qpos/noise comparisons: TRT graph/eager, Torch FP32 and autocast FP16."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deployment.sync_flow import FlowGenerator, filter_suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("docs/deploy_sync/sync_02s.npz"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tape = np.load(args.reference)["reference"]
    tape = np.concatenate([np.repeat(tape[:1], 10, axis=0), tape])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for steps in (10, 5):
        generator = FlowGenerator({**config, "steps": steps, "backend": "tensorrt-fp16"})
        for case, start in enumerate(np.linspace(0, len(tape) - generator.history - generator.prefix, 20, dtype=int)):
            history = tape[start:start + generator.history].copy()
            committed = tape[start + generator.history:start + generator.history + generator.prefix].copy()
            noise = np.random.default_rng(np.random.SeedSequence([config["seed"], case])).standard_normal(
                (generator.horizon, 38), dtype=np.float32)
            actual = generator.generate(history, committed, noise)
            np.testing.assert_array_equal(actual, generator.generate(history, committed, noise))
            outputs = {"trt": actual}
            with torch.inference_mode():
                for precision in ("fp32", "fp16", "trt-eager"):
                    window = generator.forward(*generator.inputs.values(),
                                               precision="fp32" if precision == "trt-eager" else precision,
                                               use_trt=precision == "trt-eager")
                    suffix = window[:, generator.history + generator.prefix:]
                    if generator.lowpass_hz is not None:
                        suffix = torch.as_tensor(filter_suffix(window.cpu().numpy(), generator.history,
                                                 generator.prefix, generator.lowpass_hz), device="cuda", dtype=torch.float32)
                    outputs[precision] = generator.decode(suffix)[0].cpu().numpy()
                    if precision != "trt-eager":
                        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
                            upstream = generator.reference.generate_suffix(*generator.inputs.values(), steps=steps,
                                text_tokens=generator.text_tokens, text_mask=generator.text_mask)[0].cpu().numpy()
                        unfiltered = generator.decode(window[:, generator.history + generator.prefix:])[0].cpu().numpy()
                        np.testing.assert_allclose(unfiltered, upstream, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(actual, outputs["trt-eager"], atol=1e-5, rtol=1e-5)
            row = {"steps": steps, "case": case, "start": int(start), "repeat_exact": True, "pairs": {}}
            for left, right in (("trt", "fp32"), ("trt", "fp16"), ("fp16", "fp32"), ("trt", "trt-eager")):
                delta = np.abs(outputs[left] - outputs[right])
                assert np.isfinite(delta).all()
                row["pairs"][left + "_vs_" + right] = {
                    "max_qpos_diff": float(delta.max()), "root_xyz_m": float(delta[:, :3].max()),
                    "quaternion_component": float(delta[:, 3:7].max()), "joints_rad": float(delta[:, 7:].max()),
                }
            rows.append(row)
        contract = generator.trt.contract
        # Release captured graphs and execution context before constructing E5.
        del generator
        torch.cuda.empty_cache()
    report = {"config": config, "engine_contract": contract,
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "cases": rows, "worst": {}}
    for steps in (10, 5):
        report["worst"][str(steps)] = {
            pair: {metric: max(row["pairs"][pair][metric] for row in rows if row["steps"] == steps)
                   for metric in rows[0]["pairs"][pair]} for pair in rows[0]["pairs"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["worst"], indent=2))


if __name__ == "__main__":
    main()
