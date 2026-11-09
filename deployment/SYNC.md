# Synchronous OMG-Flow deployment

Run from the repository root with `src` on `PYTHONPATH`. This entrypoint loads
**only** `MotionFlow` checkpoints through `FlowReferenceGenerator`; it does not
use the existing Lite planner, stream server, ROS adapter, or diffusion exporter.

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
length, replan period, engine path, tracker, T5, scene, seed, and duration.
The checked-in paths point to the front-kick SFT and HoloMotion v1.4.1 assets on
the shared HQ `/root`. Install the repository with `[train,tracking,render,dev]`
extras in a CUDA environment. The ONNX graph must sit beside external weights
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
the raw splice joint step. There is no blend, filter, reset or stabilization.

Generation runs in the same Python thread and control tick, with no request lead
or background generation. The torch backend uses FP16 autocast with FP32 world
geometry and fixed-shape CUDA graph replay of the existing Euler implementation.
Noise is supplied explicitly from a seed and replan index. Kernel warmup, text
encoding, graph capture and engine loading happen before the control epoch.
Three codec operations use equivalent fixed-shape tensor expressions so capture
does not trigger dynamic indexing or host-to-device constant copies. Tracker
reference features use batched SciPy rotations, checked against the scalar
formulas; no motion processing is added.

`generation_ms` measures noise, input transfer, inference, synchronized output,
validation and splice. `planning_ms` also includes state read and reference
feature construction. Each tick separately measures read/replan/tracker/write.
`overrun=raise` fails on a missed 20 ms budget; `log` records every overrun for
diagnostic simulation. `pace=true` uses absolute 50 Hz wall deadlines and reports
lateness. Ticks/timestamps are never dropped or shifted to conceal a miss.

MuJoCo advances ten 2 ms physics steps per control write. During an overrun,
physics waits for the synchronous computation; a physical robot would continue
moving. Thus an overrunning run is an offline tracking demonstration, not proof
of real-time control. Videos play simulation time at 50 fps and are rendered
after measurement from paired pre-action states at identical timestamps.

## TensorRT

If `engine_path` exists, it is mandatory: load/binding/contract errors fail.
Otherwise torch FP16 is selected. The engine has three FP32 I/O tensors
`history_qpos [1,H,36]`, `committed_qpos [1,P,36]`, `noise [1,T,38]` and FP32
output `suffix [1,T-P,36]`, with FP16 network computation. The sidecar `.json`
binds the checkpoint SHA256, prompt, prefix, history, horizon and Euler steps.
Build a new engine after changing any of those fields:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.export_sync --config deployment/sync_flow.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m deployment.check_sync_backend --config deployment/sync_flow.json \
  --output docs/deploy_sync/backend_check.json
```

The backend check compares three sequential world-reference continuations
against eager FP16 with identical noise, reports numerical differences, requires
exact repeatability, and rejects differences beyond `atol=rtol=0.01`.
These are numerical smoke tolerances, not physical acceptance criteria.

## Real robot boundary

`run_control` accepts any `RobotIO`: `read_state()` returns one timestamped state;
`write_joint_targets()` takes 29 canonical G1 joint angles. `RobotSDK` deliberately
raises until the actual SDK is bound. Its TODOs cover native joint order, world
odometry, body-frame gyro, sensor synchronization/freshness, metadata PD gains,
enable/watchdog/emergency-stop integration. Run with `overrun="raise"` and
`pace=True`. Physical operation and SDK acceptance are not implemented or claimed.

The simulator uses the HoloMotion scene, metadata PD gains/scales, and MuJoCo
actuator limits. It adds no action clipping, reference smoothing, balance forces,
or reset-on-fall. Raw reference, execution and target arrays are saved in `.npz`;
`.json` contains tick timestamps, replan timing, splice probes and tracking metrics.
The simple survival diagnostic is first root height below 0.45 m; it is not a
full contact/foot-slip or kick-quality evaluator.
