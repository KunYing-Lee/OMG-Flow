"""Compare the deployed backend with eager Flow on identical inputs and noise."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deployment.sync_flow import FlowGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator = FlowGenerator(json.loads(args.config.read_text()))
    rng = np.random.default_rng(42)
    history = np.zeros((generator.history, 36), dtype=np.float32)
    history[:, 2:4] = [0.79, 1]
    committed = np.repeat(history[-1:], generator.prefix, axis=0)
    rows = []
    for _ in range(3):
        noise = rng.standard_normal((generator.horizon, 38), dtype=np.float32)
        actual = generator.generate(history, committed, noise)
        with torch.inference_mode():
            expected = generator.forward(*generator.inputs.values())[0].cpu().numpy()
        repeated = generator.generate(history, committed, noise)
        np.testing.assert_array_equal(actual, repeated)
        np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0.01)
        rows.append({"max_abs_qpos_error": float(np.max(np.abs(actual - expected))), "repeat_exact": True})
        # Exercise nontrivial world orientation, translation and moving history.
        history = actual[:generator.history].copy()
        history[:, :2] += [1.0, -0.5]
        committed = actual[generator.history:generator.history + generator.prefix].copy()
        committed[:, :2] += [1.0, -0.5]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"backend": generator.backend, "contract": generator.contract,
                                      "atol": 0.01, "rtol": 0.01, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
