# Recording contract (realtime chain)

Every run records **the four module outputs plus the transforms' parameters and the
event timeline**. Everything else (timeline, tape, tracker observation, target) is
derivable from those.

| # | Module | Output recorded | File | Rate |
|---|---|---|---|---|
| M1 | main model (Lite HF, SFT step400, future models) | plan block `100×36` per accepted plan | `<base>.plans.npz` | 1 / switch |
| M2 | corrector (residual, future correctors) | joint delta `25×29` per decision | `<base>.decisions.npz` | 1 / switch |
| M3 | HoloMotion tracker | action `29` per frame | `<base>.frames.csv` | 50 Hz |
| M4 | robot | `q/dq/tau 29`, `quat 4`, `gyro 3` per frame | `<base>.frames.csv` | 50 Hz |
| — | events | request/accept frames, wall times, latency | `<base>.plans.npz` | 1 / switch |
| — | transforms | `switch_seconds`, `action_scale`, `default_angles`, `onnx_to_real`, residual manifest, tracker metadata | `<base>.meta.json` | once |

Layers that are intentionally **not** recorded because they are derivable:

- timeline (`PlanTimeline.qpos`) = plans spliced at `switch_frame`;
- tape (corrected reference) = timeline + committed deltas;
- tracker observation (522/604) = features(13-row window) + robot state + last action;
- target (`/humanoid/action`) = action × action_scale + default_angles, permuted by `onnx_to_real`;
- reference features and the 3455 residual observation: large and cheap to recompute.

## Files

| File | Content |
|---|---|
| `<base>.frames.csv` | `frame, t, action0..28, q0..28, dq0..28, tau0..28, quat0..3, gyro0..2, residual, read_ms, tick_ms, encode_ms, consume_ms, step_ms` (131 columns) |
| `<base>.plans.npz` | `plan_id, request_frame, accept_frame, switch_frame, deadline_frame, request_time, accept_time, latency_ms, blocks[N,100,36]` |
| `<base>.decisions.npz` | `frame, start, latent_norm, delta[N,25,29]` |
| `<base>.meta.json` | record version, node file, args, planner endpoint, prompt, switch, residual name/manifest, action scale/default/permutation, tracker metadata, counts |

## Usage

```bash
python deployment/native50_policy_node.py ... --record /workspace/logs/run-<ts>
```

Files appear as `run-<ts>.frames.csv`, `run-<ts>.plans.npz`, `run-<ts>.decisions.npz`,
`run-<ts>.meta.json`. The CSV is flushed per row; plans/decisions are rewritten on each
event, so a killed process still leaves usable data.

## Cost

- frames: 131 fields ≈ 0.9 KB/row → ~45 KB/s → **2.7 MB/min**;
- plans: 100×36 floats compressed ≈ 6 KB per plan → at 0.5 s switch ≈ **12 KB/s**;
- decisions: 25×29 floats compressed ≈ 4 KB per decision → ≈ **8 KB/s**;
- writer overhead: one `csv.writer` call + `flush` per frame (no fsync), expected
  well under 0.5 ms/frame; **not yet measured on the robot**.

`scripts/capture_lowstate.py` remains useful as an independent `/lowstate` capture
(`q/dq/tau` + IMU) and for velocity-mode segments before B.
