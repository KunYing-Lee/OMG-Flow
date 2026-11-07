# 200k long-continuation evaluation — 2026-11-07

The 190,500-step validation-best checkpoint and final 200,000-step checkpoint each completed 18 paired 20-second HoloMotion physics replays without crossing the specified height/tilt failure screen. This demonstrates stability in this small offline suite, not general physical acceptance or successful real-time deployment. The final checkpoint reduced mean root tracking error from 0.369 m to 0.337 m, with mixed results across actions. Keep both checkpoints; neither dominates all metrics.

## Protocol

- Six clips from held-out test actors, selected before generation: longest clip whose first caption contains each of `walk`, `run`, `dance`, `kick`, `jump`, `turn`, tie-broken by clip ID. Four distinct actors; this is a diagnostic suite, not a representative population estimate.
- Seeds 1234, 2345, 3456; identical initial 10 mocap frames, captions, Gaussian noise streams, and 10-step Euler solver for both checkpoints. Frozen pinned T5 embeddings; BF16 inference.
- 50 Hz, 20 seconds, replan every 25 frames with 25 immutable committed frames. After initialization, history and prefix come exclusively from the generated reference. Each call returns only the editable suffix; no future ground truth, pose smoothing, clipping, blending, or reference stabilization.
- HoloMotion v1.4.1 `model_16200.onnx`, repository G1 scene, native 50 Hz, ten future frames, ten simulation substeps. Identical observed pose and finite-difference velocity initialization. Existing tracker action limit remains 10 for all references; no new intervention or controller tuning.
- All trajectories replayed completely, including after any failure. Failure screen is pelvis z below 0.35 m or root tilt over 60 degrees. This operational screen is explicitly separate from tracking and semantic success.
- Original test clips also replayed, up to their available length minus 11 preview/derivative frames; all six stayed above the same screen. Their durations differ from the generated 20-second tests.
- This is offline replay of autoregressively generated plans. It does not measure online scheduling deadlines, executed-state conditioning, robustness to perturbations, or physical hardware.

## Results

All means below average the 18 generated trials equally. Joint acceleration is the absolute second difference at 50 Hz, averaged over joints and time. Seam values include both stencils crossing every newly generated suffix boundary.

| Metric | best 190.5k | final 200k |
|---|---:|---:|
| Completed without height/tilt failure | 18/18 | 18/18 |
| Joint tracking MAE (rad) | 0.06882 | 0.06893 |
| Root tracking error (m) | 0.36864 | 0.33694 |
| Reference seam acceleration (rad/s²) | 16.104 | 16.237 |
| Reference overall acceleration (rad/s²) | 17.108 | 16.975 |
| Reference acceleration over matched original-clip durations (rad/s²) | 17.446 | 17.277 |

Original mocap acceleration over these matching durations averages 6.462 rad/s². The generated references are about 2.7 times higher; lower seam values than whole-trajectory values do not establish natural motion. Short-horizon validation loss did not expose this long-rollout tracking gap.

| Caption category | best root error (m) | final root error (m) |
|---|---:|---:|
| Walk | 0.267 | 0.300 |
| Run | 1.175 | 0.881 |
| Dance | 0.063 | 0.067 |
| Kick | 0.401 | 0.484 |
| Jump | 0.181 | 0.169 |
| Turn | 0.125 | 0.121 |

These are descriptive paired results over three seeds per clip. They do not show statistically established improvement across the dataset. Upright completion does not establish correct running, a successful kick, or matching the caption throughout a 20-second extension of a shorter source action.

## Reproduction and artifacts

Use `scripts/validation/flow_long_eval.py` stages `generate`, `track`, and `render` with `PYTHONPATH=src`. Set `MUJOCO_GL=disable` for headless physics. Rendering requires a working OpenGL context. CPU tracking uses one ONNX thread; generation used one idle RTX 5090. The recorded batch-of-18 replan latency was about 102 ms, which is not a single-stream deadline measurement.

Example remote inputs:

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/leeky-omg-flow/venv/bin/python scripts/validation/flow_long_eval.py generate \
  --data /tmp/leeky-omg-flow/bones --text /tmp/leeky-omg-flow/bones-text \
  --run /root/leeky/omg-flow/runs/bones-native50-4x5090-20261107 \
  --out /tmp/leeky-omg-flow/eval-200k
```

Run stage `track` with `--out`, `--onnx`, and `MUJOCO_GL=disable`. Run stage `render` with only `--out` on the saved output directory. Render defaults to seed 1234 for all six categories, chosen in advance rather than by outcome. Videos show best/final reference on the top row and their physics execution below; each execution camera follows the corresponding reference and shows root error.

- Durable G3 artifacts: `/root/leeky/omg-flow/evaluations/eval-200k`.
- Local artifacts and six full 20-second videos: `/Users/mac/Developer/OMG-Flow/outputs/eval-200k`.
- `manifest.json` records exact clip IDs, captions, seeds, checkpoint/data/T5/tracker hashes, physics dependencies, and XML hashes.
- `tracking.json` contains all 42 replay records; `summary.json` contains aggregates. Raw reference, execution, and action arrays are preserved.
- Best checkpoint SHA256: `3f8d9f8a9cbc96cb034e6c0cb2b1bfccd38816a8dca66c68df3f24b7f49f9cde`.
- Final checkpoint SHA256: `c3427e83cdf6fe6785694a0678c66eaf30aebef643ab5df1188ba7108e484856`.

Existing Flow/prefix regression checks: 11 passed. All 36 generated outputs were finite, with unit root quaternions; all committed buffers were retained exactly. Generation used Torch 2.8.0+cu128; physics used MuJoCo 3.3.7 and ONNX Runtime 1.23.2. Videos are pose re-renderings on local MuJoCo 3.16.0; physical integration results come solely from the recorded remote 3.3.7 rollouts.

The next diagnostic should focus on the source of root-velocity/position tracking mismatch in the moving references, with the existing tracker and initialization held fixed. Simply increasing training steps is not justified by this comparison.
