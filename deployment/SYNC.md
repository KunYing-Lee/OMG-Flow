# Synchronous OMG-Flow deployment

Run from the repository root with `src` on `PYTHONPATH`. This is the default deployment entrypoint. It loads `MotionFlow` checkpoints
through `FlowReferenceGenerator` and executes generation and tracking in the
same thread.

```bash
# Check processes first; select one genuinely idle GPU.
nvidia-smi
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.run_sync --config deployment/sync_flow.json \
  --output docs/deploy_sync/sync_02s

# Identical prompt, seed, prefix, integrator and checkpoint; only cadence changes.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.run_sync --config deployment/sync_flow.json \
  --period 0.5 --output docs/deploy_sync/sync_05s

python -m pytest -q tests/generation/test_sync_deployment.py
```

`sync_flow.json` supplies checkpoint, prompt, Euler steps (10 or 5), prefix
length, replan period, explicit backend, engine cache directory, low-pass cutoff,
tracker, T5, scene, seed, and duration.
Set the `/models` paths to your Flow checkpoint, local T5-base and HoloMotion
v1.4.1 assets. Install the repository with `[train,tracking,export,render,dev]`
extras in a CUDA environment. The README suite pins the original 190.5k
checkpoint hash in its manifest; the earlier timing study used front-kick SFT. The ONNX graph must sit beside external weights
`4de98d3c-9d3c-11f1-aff4-06f301228322`. No model files are committed.

## Timing and commitment

At tick `k`, the generator receives reference history `[k-H,k)` and clean
committed reference `[k,k+P)`. Its Euler suffix replaces exactly `[k+P,k+T)`.
History comes from the reference already tracked, not measured robot motion.
All poses retain their world frame and integer monotonic timestamps
`epoch_ns + tick * 20,000,000`; the Flow API itself uses their fixed 50 Hz order.
The model has no extra absolute-time embedding. The initial history and first
`P` committed frames are the tracker default standing joint pose with configurable
root height. With `P=25`, generation starts after a 0.5 s standing prefix.

HoloMotion sees future poses before executing them. The scheduler preserves
that preview **and the extra pose used by its velocity stencil**, and rejects
prefixes shorter than `n_fut_frames + 1`. Tests check both poses and derived
tracker features. A clean prefix prevents retrospective edits; it does **not**
mathematically guarantee a smooth learned first suffix frame. The report records
the splice joint step. The configured low-pass acts only on new suffixes; there
is no blend, reset or stabilization.

Generation runs in the same Python thread and control tick, with no request lead
or background generation. The torch backend uses FP16 autocast (or explicit FP32) with FP32 world
geometry. Both Torch and TensorRT use fixed-shape CUDA graph replay of Euler
sampling and a separate decode graph, with the CPU filter between them.
Noise is supplied explicitly from a seed and replan index. Kernel warmup, text
encoding, graph capture and engine loading happen before the control epoch.
Three codec operations use equivalent fixed-shape tensor expressions so capture
does not trigger dynamic indexing or host-to-device constant copies. Tracker
reference features use batched SciPy rotations, checked against the scalar
formulas; no motion processing is added.

`generation_ms` measures noise, input transfer, inference, synchronized output,
validation and splice. `planning_ms` also includes state read and reference
feature construction. Each tick separately measures generation, reference features, observation
construction, tracker ONNX (including its transfers), and state read + physics/write.
`feature_build_ms` is reference features plus observation construction; `other_ms`
accounts for remaining loop work. Reports include P50/P95/max for all ticks
and separately for replan ticks (500/50 for the default ten-second run,
1,000/100 for the twenty-second gallery). Generation is zero on other ticks.
`overrun=raise` fails on a missed 20 ms budget; `log` records every overrun for
diagnostic simulation. `pace=true` uses absolute 50 Hz wall deadlines and reports
lateness. Ticks/timestamps are never dropped or shifted to conceal a miss.

MuJoCo advances ten 2 ms physics steps per control write. During an overrun,
physics waits for the synchronous computation; a physical robot would continue
moving. Thus an overrunning run is an offline tracking demonstration, not proof
of real-time control. Videos play simulation time at 50 fps, or 25 fps with
`--showcase`, and are rendered after measurement from paired pre-action states
at identical timestamps.

## Suffix filter

Default `lowpass_hz: 8.0`; JSON `null` or CLI `--lowpass-hz 0` turns it off.
Other cutoffs must lie strictly between 0 and 25 Hz. This is the exact filter from
OMG-Posttrain `4f7ff92`, `src/omg_posttrain/generation/flow.py` and
`docs/flow_vs_lite_ab_20261109.md`:

1. Encode history and committed frames into normalized anchor-relative 38-D
   features (xyz, rot6d, joints), then append the generated suffix.
2. `butter(4, cutoff_hz / 25)` followed by float64 `filtfilt(..., axis=1,
   padtype="odd", padlen=15)` over all 110 frames at 50 Hz.
3. Keep only rows `10 + P:` and convert to FP32. De-normalize and decode normally.

The committed context supplies the left boundary condition, but its filtered
values are discarded. Original world history/prefix and timestamps remain exact.
This is noncausal within the available generated window; it requires no future
sensor data or extra control-tick lookahead. World quaternions are never filtered.

## TensorRT

Set `backend` explicitly to `tensorrt-fp16`, `torch-fp16`, or `torch-fp32`.
There is no automatic backend substitution. The default is TensorRT in the
checked-in config; a missing cache builds once before the control epoch.

The export follows Posttrain `4f7ff92:scripts/benchmark_trt_deploy.py`: CPU FP32
model copy, ONNX opset 18, static batch one, TensorRT FP16 with a 4 GiB workspace.
Only `_velocity` is exported. Inputs are state `[1,100,38]`, time `[1]`, committed
mask `[1,100]`, projected history/text context, and context mask. Float I/O stays
FP32 and masks stay boolean. FP32 context projection, Euler accumulation and the
world codec remain outside TRT. Euler uses `t = step / steps`, never `t = 1`.

The cache is nested under checkpoint SHA256 and GPU UUID. Its key also binds GPU
model/capability, CUDA/Torch/TRT versions, shapes/dtypes, export/network source
hashes and build settings. A sidecar verifies the contract and engine SHA256.
Euler count, prefix, cutoff and prompt values are runtime inputs/settings; the
same engine is reused for E5/E10 and prompts with matching context shape.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.export_sync --config deployment/sync_flow.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.check_sync_backend --config deployment/sync_flow.json \
  --reference docs/deploy_sync/sync_02s.npz \
  --output docs/deploy_sync/backend_trt_check.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.run_sync --config deployment/sync_flow.json \
  --steps 10 --output docs/deploy_sync/sync_02s_trt_e10_lp8
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.run_sync --config deployment/sync_flow.json \
  --steps 5 --output docs/deploy_sync/sync_02s_trt_e5_lp8
```

The checker uses 20 fixed windows from the prior reference tape at each step
count, with identical noise and the same filter across precision paths. It reports
maximum qpos/root/quaternion/joint differences for TRT vs FP32, TRT vs FP16 and
FP16 vs FP32. It requires exact repeatability, checks CUDA replay against eager
TRT, and compares the unfiltered Torch adapter against the original reference API
(atol=rtol=1e-5). FP16 approximation errors are measured, not a physical safety gate.

## Real robot boundary

`run_control` accepts any `RobotIO`: `read_state()` returns one timestamped state;
`write_joint_targets()` takes 29 canonical G1 joint angles. `RobotSDK` deliberately
raises until the actual SDK is bound. Its TODOs cover native joint order, world
odometry, body-frame gyro, sensor synchronization/freshness, metadata PD gains,
enable/watchdog/emergency-stop integration. Run with `overrun="raise"` and
`pace=True`. Physical operation and SDK acceptance are not implemented or claimed.

The simulator uses the HoloMotion scene, metadata PD gains/scales, and MuJoCo
actuator limits. Beyond the specified suffix filter, it adds no action clipping, balance forces,
or reset-on-fall. Raw reference, execution and target arrays are saved in `.npz`;
`.json` contains tick timestamps, replan timing, splice probes and tracking metrics.
Reports retain the first root height below 0.45 m and separately apply the
historical README failure screen: root height below 0.35 m or tilt above
60 degrees, with no early termination. Neither establishes prompt compliance
or full contact/foot-slip quality.
