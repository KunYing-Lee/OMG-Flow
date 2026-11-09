# OMG-Flow synchronous deployment evidence — 2026-11-09

The Flow-only synchronous loop is implemented and recorded. **Physical deployment
is not accepted:** the SDK is deliberately unbound, ten-step control ticks miss
deadlines, and tracking is unstable in these standing-start simulations. Five
Euler steps meet all measured deadlines in this run but still produce a fall.

All runs use one RTX 5090, physical GPU 3 on `hq-g1`, checked process-free before
launch. Other GPUs remained occupied and untouched. The checkpoint is the
front-kick SFT at
`/root/leeky/projects/OMG-Posttrain/flow-ab-20261108/sft/latest/sft.pt`, SHA256
`87fee467ef75589ccc39216e08a567fc6a4888b08b3c74200e6775764cece578`.
Prompt: **“A classical dancer kicks one leg forward.”** Each run is 10 s / 500
control ticks, seed 1234, 10 history frames, 100-frame horizon, 25 committed
frames, and tracker-default standing joints at 0.79 m root height. Only cadence
or Euler steps differ. The first 25 reference frames remain standing.

| Video / configuration | Generation P50 / P95 / max, ms | Planning P50 / P95 / max, ms | Compute overruns / deadline misses (500 ticks) | Joint RMSE, rad | Root position RMSE, m |
|---|---:|---:|---:|---:|---:|
| [0.2 s, Euler 10](sync_02s.mp4) | 14.831 / 14.869 / 15.826 | 15.581 / 15.720 / 16.683 | 31 / 38 | 0.218 | 0.499 |
| [0.5 s, Euler 10](sync_05s.mp4) | 17.002 / 25.924 / 25.994 | 17.749 / 26.678 / 26.748 | 17 / 21 | 0.283 | 0.775 |
| [0.2 s, Euler 5](sync_02s_e5.mp4) | 8.044 / 8.096 / 9.121 | 8.809 / 8.928 / 9.984 | 0 / 0 | 0.301 | 0.856 |

Generation includes explicit noise, input transfer, synchronous inference, output
transfer, validation and splice. Planning additionally includes state read and
reference-feature construction. Compute overruns count complete control work
over 20 ms; deadline misses also include late starts/scheduling. The maximum full
control computation was respectively **22.079, 56.090 and 15.470 ms**. There were
0/8/0 planning overruns among 50/20/50 replans. Every overrun was logged; none was
concealed by changing the reference clock, dropping ticks or switching backends.

The 0.2 s / ten-step robot dips below the 0.45 m diagnostic height at **5.58 s**,
reaches 0.339 m, then recovers upright by 9 s (final root height 0.810 m). Video
inspection shows a deep crouch and substantial tracking error; this threshold
crossing is not an unrecoverable fall. The 0.5 s / ten-step and 0.2 s / five-step
runs cross the threshold at **5.72 s** and **5.58 s**, respectively, and finish
lying on the ground, around 0.061 m root height. RMSE covers the entire 10 s,
including those failures, and compares reference/state at identical timestamps.

The videos are reference (kinematic) | HoloMotion execution, 1280×480, 50 fps,
500 frames each. They mark replan ticks and generation latency. Sizes are
**4.22, 3.97 and 4.22 MB**, all under 20 MB. Rendering occurred after measurement.
Videos play simulation time: MuJoCo waits while synchronous computation overruns;
a physical robot would continue moving. They therefore do not demonstrate
real-time physical robustness. Representative frames at 1/4/6/9 s and both
fallen endpoints were visually inspected.

Files and reproduction:

- [Deployment guide](../../deployment/SYNC.md) and [config](../../deployment/sync_flow.json).
- [Generator and scheduler](../../deployment/sync_flow.py): strict Flow checkpoint
  loading, absolute monotonic timestamps, immutable prefix, Euler 10/5, indexed
  deterministic noise, TensorRT selection or torch FP16 CUDA graph replay.
- [Tracker / robot I/O](../../deployment/sync_robot.py): v1.4.1 ONNX plus required
  external weights, canonical joint ordering, native metadata PD, MuJoCo, SDK TODOs.
- [Control runner and renderer](../../deployment/run_sync.py): synchronous 50 Hz
  path, deadline checks, paired state capture, metrics, offline video.
- [TensorRT exporter](../../deployment/export_sync.py) and
  [backend parity checker](../../deployment/check_sync_backend.py).
- [CPU tests](../../tests/generation/test_sync_deployment.py). Two existing shared
  geometry files received equivalent fixed-shape/batched implementations to
  permit CUDA capture and remove per-frame Python rotation overhead.
- [Environment/source/asset/media manifest](manifest.json),
  [GPU backend check](backend_torch_check.json), and each video's matching `.json`
  (all tick/replan timings) and `.npz` (reference, execution, joint targets, times).

The measured backend is **torch FP16 with CUDA graph replay**. No compatible
TensorRT engine was present; the optional TensorRT loader/exporter was not
exercised. GPU replay matched eager FP16 exactly in three sequential continuation
checks, with exact repeated output. The targeted CPU suite passed **28 tests**,
including prefix/derived-feature invariance, 10/25-tick schedules, reproducible
noise, deadline rejection before writes, and rotation/gradient checks. All
500 state timestamps exactly match reference timestamps in each run. Local
source and scene hashes match the measured remote files.

Committed-prefix preservation is exact; learned suffix smoothness is not
guaranteed. Maximum first-suffix joint steps were **0.171, 0.169 and 0.160 rad**.
No reference blend, low-pass filter, action clip, balance force or reset was
added. These are single-seed diagnostic runs, not a controlled multi-seed quality
comparison. GPU clocks were not locked and the host was shared; do not attribute
the 0.5 s latency difference solely to cadence. Software versions and hashes are
in the manifest (GPU: Python 3.10.22, torch 2.7.1+cu128, MuJoCo 3.11.0).

On this container, missing EGL GLVND libraries and OpenCV were installed only in
`/root/leeky/projects/OMG-Flow-deploy-sync/runtime/`. The run used
`PYTHONPATH=src:runtime/python`, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=3`, and
the local `runtime/usr/lib/x86_64-linux-gnu` in `LD_LIBRARY_PATH`. The generic
commands in the deployment guide work in an environment with the declared
render/tracking dependencies already installed.
