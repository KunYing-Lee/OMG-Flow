# TensorRT FP16 + 8 Hz suffix filter — 2026-11-09

The velocity-network TensorRT path is implemented and numerically checked.
Both requested 10 s / 500-tick simulations use the same front-kick SFT, prompt,
seed 1234, 0.2 s replanning, P25, standing start and tracker as `d3d92cc`.
All timing below is from hq-g1 physical GPU 3 (RTX 5090), one GPU at a time.
It was process-free before each run and monitored for foreign processes every
0.5 s through export, parity, simulation and rendering. No foreign process was
observed in the accepted runs. Clocks were not locked and the host remained shared.

Source: OMG-Posttrain branch `claude/silly-banach-d8f8c6`, commit `4f7ff92`:
`docs/flow_vs_lite_ab_20261109.md`, `src/omg_posttrain/generation/flow.py`,
`scripts/benchmark_trt_deploy.py`, and `scripts/analyze_jitter.py`.

## Exact suffix filter and velocity export

Fourth-order Butterworth, `butter(4, cutoff_hz / 25)`, 50 Hz; float64
`filtfilt(..., axis=1, padtype="odd", padlen=15)` over normalized native38
`[history10 | committed P | generated 100-P]`. Only filtered suffix rows are
retained, cast to FP32, de-normalized and decoded. World quaternions are never
filtered. Committed world poses/timestamps remain bit-identical. The context
provides the left boundary condition; its filtered values are discarded.
Default cutoff is 8 Hz; JSON `null` or `--lowpass-hz 0` disables it.

The A/B export approach is retained: CPU FP32 copy, static batch-one ONNX
opset 18, TensorRT FP16 velocity network, 4 GiB builder workspace. FP32
context projection, Euler accumulation and world codec remain outside TRT.
Exact CUDA graph replay wraps sampling and decoding separately, with SciPy
filtering between them. No alternate integrator or motion correction was added.

Cache keys include checkpoint SHA256, GPU UUID/model/capability, software
versions, input shapes/dtypes, build settings and export/network source hashes.
The sidecar checks the engine SHA256. E10 and E5 loaded the same cached engine;
step count, cutoff and prefix values do not require different velocity engines.
Backend selection is explicit and errors do not select a different backend.

## Numerical comparison

20 fixed windows per Euler count from the original Torch reference tape, same
noise per case and same LP8 filter on every precision path. These are matched
input comparisons, not independently evolving trajectories. TRT replay versus
eager TRT is bit-exact in every case; repeated calls are exact. Unfiltered Torch
adapter output also matches the upstream reference API at atol=rtol=1e-5.
No physical acceptance threshold is inferred from FP16 approximation errors.

| Steps | TRT vs Torch FP32 max qpos | TRT vs Torch FP16 max qpos | Torch FP16 vs FP32 max qpos | TRT vs FP32 max root xyz, m |
|---|---:|---:|---:|---:|
| E10 | 0.020167 | 0.020622 | 0.004178 | 0.002727 |
| E5 | 0.019942 | 0.019959 | 0.004903 | 0.002405 |

The largest qpos differences above occur in joints (radians). Raw qpos mixes
metres, quaternion components and radians; component maxima are in
[backend_trt_check.json](backend_trt_check.json).

## Complete control timing

P50 / P95 / max in milliseconds. This first table uses only the **50 replan
ticks**, so generation is not hidden by the 450 ticks with no generator call.
Feature build includes reference precomputation plus per-tick observation
construction. ONNX includes tracker transfers/KV handling. Physics/I/O includes
state read and ten MuJoCo 2 ms steps with PD writes. Other includes remaining
read-to-write loop work. Quantiles of stages do not sum to total quantiles.

| Replan-tick stage | TRT E10 + LP8 | TRT E5 + LP8 |
|---|---:|---:|
| Generation | 6.945 / 7.259 / 8.794 | 4.565 / 4.690 / 6.133 |
| Feature build | 1.261 / 1.324 / 1.372 | 1.268 / 1.300 / 1.612 |
| Tracker ONNX | 1.543 / 1.597 / 1.806 | 1.496 / 1.525 / 1.653 |
| Physics / I/O | 2.299 / 2.647 / 3.098 | 2.227 / 2.666 / 3.399 |
| Other | 0.107 / 0.122 / 0.169 | 0.104 / 0.114 / 0.159 |
| Full control tick | 12.170 / 12.610 / 14.968 | 9.692 / 10.217 / 11.947 |

All **500 control ticks**, including non-replan ticks:

| Backend | Full tick P50 / P95 / max, ms | Ticks >20 ms | Wall deadline misses |
|---|---:|---:|---:|
| TRT E10 + LP8 | 4.598 / 12.167 / 14.968 | 0/500 | 0/500 |
| TRT E5 + LP8 | 4.507 / 9.685 / 11.947 | 0/500 | 0/500 |
| Earlier Torch FP16 CUDA graph E10, filter off | 4.474 / 20.066 / 22.079 | 31/500 | 38/500 |
| Earlier Torch FP16 CUDA graph E5, filter off | 4.621 / 13.357 / 15.470 | 0/500 | 0/500 |

The earlier runs used the same physical GPU 3 and motion settings but occurred
earlier under different shared-host load and had filtering off. This is the
requested historical comparison, not an isolated measurement of the backend's
causal speedup. All tick-level stage data and separate all-tick/replan-tick
summaries are preserved in the JSON files. Sleep, trace append, logging and
offline rendering are outside measured read-to-write control work.

## Motion and media

| Run / MP4 | First root height <0.45 m | Minimum root height, m | Final root height, m | Joint RMSE, rad | Root position RMSE, m |
|---|---:|---:|---:|---:|---:|
| [E10 + LP8](sync_02s_trt_e10_lp8.mp4) | 5.62 s | 0.059 | 0.061 | 0.298 | 0.814 |
| [E5 + LP8](sync_02s_trt_e5_lp8.mp4) | 5.72 s | 0.059 | 0.061 | 0.308 | 0.754 |

Both robots fall and finish on the ground. Passing a compute deadline does not
establish tracking or hardware acceptance. Filtering and FP16 change the
trajectory relative to the earlier single-seed run; these runs cannot isolate
the cause of that difference. No reset, balance force or action clipping is used.
Videos are paired kinematic reference | HoloMotion execution, 1280×480, 50 fps,
500 frames; they show full-tick latency and are rendered after timing. MuJoCo
waits if computation overruns, whereas a physical robot keeps moving.

## Evidence and reproduction

- [Deployment guide and commands](../../deployment/SYNC.md).
- [E10 trace](sync_02s_trt_e10_lp8.json), [E10 arrays](sync_02s_trt_e10_lp8.npz).
- [E5 trace](sync_02s_trt_e5_lp8.json), [E5 arrays](sync_02s_trt_e5_lp8.npz).
- [Matched numerical checks](backend_trt_check.json).
- [Source, checkpoint, engine, package and artifact hashes](trt_manifest.json).
- [GPU process audit](gpu_guard.jsonl).

33 targeted CPU tests passed: Flow/reference, prefix/timestamp/preview
invariance, exact filter/boundary specification, spectral/phase response,
cutoff validation, timing accounting, geometry and tracker tests. Accepted GPU
runs verified cached-engine reuse and all 500 state/reference timestamps match.
Media frame counts, dimensions and representative frames were inspected locally.

An initial GPU 0 run started idle, but another session subsequently launched a
workload there. Its E5 measurement had 2/500 compute overruns and was excluded.
All initial GPU 0 reports/media are preserved under the remote run directory's
`gpu0_initial/`; both conditions and parity were rerun on monitored GPU 3.
No other workload was stopped or modified. Mars5090 was not used.
