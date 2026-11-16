# Batch-one plan runtime

`FlowRuntime` owns only generation: checkpoint normalization, text conditioning,
Euler integration, optional suffix filtering and world-qpos decode. It imports
no simulator, tracker, transport or robot controller. Training code and configs
are unchanged. `sync_flow.FlowGenerator` reexports it; the existing 50 Hz scheduler,
0.2 s replanning, prefix 25 and deployment JSON defaults are unchanged.

Run from a checkout with `PYTHONPATH=src` (or install `.[dev]`). Torch FP32 supports
CPU and CUDA; Torch FP16 autocast and TensorRT FP16 require CUDA. CUDA uses the
existing separate sampling/decode graphs by default. `cuda_graph=False` selects
eager execution explicitly. TensorRT retains the engine cache and hash contract
in `flow_trt.py`; source paths are now independent of the caller's working directory.
The tested CUDA stack is Torch 2.8.0+cu128 / TensorRT 10.13.3.9 on RTX 5090.

```python
from deployment.flow_runtime import FlowRuntime

runtime = FlowRuntime({
    "checkpoint": "<checkpoint>", "device": "cuda",
    "backend": "tensorrt-fp16", "engine_cache": "tensorrt_engine_cache/flow_sync",
    "steps": 10, "prefix_length": 25, "lowpass_hz": 8.0,
    "prompt": "A person walks forward.", "text_cache": "<text-cache>",
})
suffix_qpos = runtime.generate(history_qpos, committed_qpos, noise)
# Alternatively: runtime.generate(history_qpos, committed_qpos, seed=1234, replan_index=3)
```

Inputs are unbatched `history_qpos [10,36]`, `committed_qpos [P,36]`, and full
`noise [100,38]`. Output is a fresh NumPy FP32 `suffix_qpos [100-P,36]` array.
Qpos is root xyz, unit root quaternion **wxyz**, and 29 G1 joints. Arrays or Torch
tensors are accepted, converted to FP32 on the configured device at the boundary.
The instance has a fixed P within the checkpoint's trained prefix range; use
P=0 for an initial plan, P=25 for the standard replan. Instances own mutable
buffers and are not reentrant or thread-safe.

Only noise rows **P:** influence the result. Rows `:P` are unused, including
when they contain NaNs; no noise is regenerated, scaled or masked into the suffix.
This exactly matches Posttrain `FlowSeededMotionGenerator.decode` and
`FlowNoiseDecoder.decode`; PPO's used mask contains `(100-P)*38` coordinates.
The caller keeps the original committed world frames. No blending or seam repair
is applied. Optional 8 Hz filtering is the existing float64 fourth-order
Butterworth `filtfilt` over normalized `[history|prefix|suffix]`; only the suffix
is retained and decoded in FP32. The runtime default is no filter; the existing
synchronous deployment config continues to request 8 Hz.

Seed mode preserves the deployment NumPy
`default_rng(SeedSequence([seed,replan_index])).standard_normal((100,38), dtype=float32)`.
This is **not** Posttrain's Torch device RNG seeded with `seed+plan_index`.
For paired comparisons or policy noise, pass the same external noise explicitly.
`sample(...)` returns a borrowed device tensor of normalized suffix features;
it is overwritten by the next call. It is available for timing without qpos decode.

## Text cache and offline robot use

The cache is the existing Posttrain-compatible **directory**, not a new format:
`text.json`, `tokens.npy` (FP16 `[captions,50,768]`), `lengths.npy` (uint8).
`BonesTextCache(..., verify_files=True)` enforces encoder and file checksums;
Posttrain's `caption_tokens(text_cache, caption)` reads it unchanged. The cache
path requires no T5 model or Transformers installation. Prompt lookup is exact.

```bash
PYTHONPATH=src python -m deployment.flow_runtime text-cache \
  --prompt 'A person walks forward.' --text-encoder "<text-encoder>" \
  --device cuda --output "<text-cache>"
```

Cache creation needs `.[train]` and the pinned T5 snapshot verified by
`prepare_cache`. CPU encoding uses FP32; CUDA encoding uses the existing BF16
cache preparation path, with FP16 storage. Direct `--text-encoder` loads local
T5 in FP32 as the original synchronous deployment did. These text preparation
paths can give different embeddings: numerical parity must hold the cache fixed.
Use either `text_cache` or `text_encoder` with a prompt; direct tensor injection
is available through `text_tokens=[1,L,D]`, `text_mask=[1,L]` constructor keywords.

## Plan and latency CLI

Create an NPZ containing `history_qpos`, `committed_qpos`, and optional `noise`.
If noise is omitted, supply `--seed`; supplying both is an error. The parity
script can create a small deterministic example without a checkpoint.

```bash
python scripts/validation/parity_cce4ad3.py golden --prefix 25 --output inputs.npz
PYTHONPATH=src python -m deployment.flow_runtime plan \
  --checkpoint "<checkpoint>" --text-cache "<text-cache>" \
  --prompt 'A person walks forward.' --input inputs.npz \
  --backend tensorrt-fp16 --lowpass-hz 8 --output plan.npz
PYTHONPATH=src python -m deployment.flow_runtime benchmark \
  --checkpoint "<checkpoint>" --text-cache "<text-cache>" \
  --prompt 'A person walks forward.' --input inputs.npz \
  --backend tensorrt-fp16 --lowpass-hz 8 --warmup 10 --iterations 100 \
  --output latency.json
```

Plan NPZ stores `suffix_qpos`, original `committed_qpos`, and `contract_json`.
Benchmark JSON reports batch-one p50/p95/max milliseconds with and without world
qpos decode. Measurements synchronize CUDA around each call, include validation,
input transfer and optional filter, and exclude checkpoint/T5 loading, engine
building, warmup and graph capture. With-decode timing includes the output copy
to CPU. These are generator measurements, not end-to-end robot control deadlines.

See [validation evidence](../docs/deploy_runtime/VALIDATION.md) for fixed-checkpoint
results and reproduction commands.
