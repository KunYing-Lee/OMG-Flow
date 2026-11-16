# Runtime validation — 2026-11-16

Implementation commit: `aa92092bca48456d963614f11ec527680adbdd04`.
Base main: `6dbed001a53cfd3cd5ea02b597142b96114ae3bf`.
Oracle: `cce4ad3` in a separate `git worktree add --detach` checkout.

## Checkpoint and inputs

- Checkpoint SHA256: `3f8d9f8a9cbc96cb034e6c0cb2b1bfccd38816a8dca66c68df3f24b7f49f9cde`.
- Environment: single RTX 5090 workstation; CUDA `12.8`, torch `2.8.0+cu128`, TensorRT `10.13.3.9`; TF32 disabled for captures.
- Euler-10, batch 1, history10/horizon100, P=0 and P=25. Unfiltered FP32 parity.
- Prompt: `A person walks forward.` One pinned T5 cache shared by every implementation; signature `8c7d1e0e74149251634fa5100e97330830362c78970f901802b1c8c975e38a85`.
- Golden seed 20261116 creates smooth xyz, roll/pitch/yaw and joint poses plus full `[100,38]` noise. Replays rotate/translate both history and prefix, and roll noise rows, on the same runtime instance.
- Captures execute in separate processes with the selected source checkout inserted before imports. Published receipts preserve commit, source file hashes, input hash, validation script hash and cache signature; local source paths are redacted. The oracle calls its unmodified `FlowReferenceGenerator.generate_suffix`, the path used by Posttrain's plain Euler decoder.

## Numerical results

| Comparison | qpos max abs, first / changed-input replay | Body positions max abs | Result |
|---|---:|---:|---|
| cce4ad3 → runtime FP32, P=0 | 0 / 0 | 0 | Bitwise identical |
| cce4ad3 → runtime FP32, P=25 | 0 / 0 | 0 | Bitwise identical |
| cce4ad3 → main reference API, P=25 | 0 / 0 | 0 | Bitwise identical |
| main synchronous generator → runtime FP32, P=25, 8 Hz | 0 / 0 | 0 | Bitwise identical |

The original synchronous constructor requires local T5. Its text buffers were
then replaced with the same fixed cache tokens before the comparison, so this
test measures runtime extraction, not T5/cache quantization differences.

Reduced precision versus runtime FP32, P=25:

| Backend / filter | qpos max abs first / replay | Root position max, mm | Body position max / p95 / RMS, mm |
|---|---:|---:|---:|
| Torch FP16 / none | 0.0005862 / 0.0007539 | 0.1753 | 0.3424 / 0.2027 / 0.1211 |
| TRT FP16 / none | 0.0036504 / 0.0120416 | 1.0945 | 1.5448 / 1.0999 / 0.6658 |
| TRT FP16 / 8 Hz | 0.0032604 / 0.0119137 | 0.8402 | 1.3299 / 1.0193 / 0.6359 |

Qpos max is a componentwise maximum across xyz in meters, unit quaternion
components and joints in radians; it is not a distance. Body/root distances
above are Euclidean errors on the **first** capture only; changed-input replay
qpos errors are reported separately. These fixed cases establish implementation
parity, not a motion-quality distribution, safety bound or physical acceptance.
FP16 errors are reported, not asserted to meet the FP32 `1e-5` threshold.
The compare command gates both qpos arrays; utility metrics remain separately reported.

## Latency and smoke

Batch 1, P=25, Euler-10, 8 Hz, CUDA graphs, 10 warmups and 100 measured calls;
wall-clock synchronized on both sides. Input validation/copies and CPU filtering
are included; with-decode also includes output transfer. Model loading, T5,
engine build and graph capture are excluded.

| Backend | Without decode p50 / p95 / max, ms | With decode p50 / p95 / max, ms |
|---|---:|---:|
| Torch FP32 | 25.143 / 25.915 / 26.327 | 25.335 / 25.447 / 25.952 |
| Torch FP16 | 21.318 / 22.150 / 22.485 | 21.460 / 21.643 / 22.143 |
| TRT FP16 | 6.150 / 6.236 / 6.699 | 6.403 / 6.481 / 6.552 |

TRT plan CLI produced finite `(75,36)` suffix output, bitwise identical to its
matched capture. This does not measure simulation, tracking or robot I/O; the
Torch configurations exceed a 20 ms control tick in this measurement.

## Reproduction

Replace `<workdir>` and `<checkpoint>` with local inputs. Use an idle GPU; capture
rejects existing compute processes. Create the fixed text cache using the
[cache command](../../deployment/RUNTIME.md#text-cache-and-offline-robot-use).

```bash
cd "<workdir>"
python -m pytest -q tests/deployment tests/generation
mkdir -p tmp .cache artifacts
export PYTHONPATH=src OMP_NUM_THREADS=1 TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface" CUDA_CACHE_PATH="$PWD/.cache/cuda"
git worktree add --detach oracle-cce4ad3 cce4ad3
python scripts/validation/parity_cce4ad3.py golden --prefix 25 --output artifacts/golden-p25.npz
# Create the cache with deployment.flow_runtime text-cache (see deployment/RUNTIME.md).
python scripts/validation/parity_cce4ad3.py capture \
  --source oracle-cce4ad3 --implementation oracle \
  --checkpoint "<checkpoint>" --text-cache artifacts/text-cache \
  --input artifacts/golden-p25.npz --output artifacts/cce4ad3-p25.npz
python scripts/validation/parity_cce4ad3.py capture \
  --source . --implementation runtime \
  --checkpoint "<checkpoint>" --text-cache artifacts/text-cache \
  --input artifacts/golden-p25.npz --output artifacts/runtime-fp32-p25.npz
python scripts/validation/parity_cce4ad3.py compare \
  --reference artifacts/cce4ad3-p25.npz --candidate artifacts/runtime-fp32-p25.npz \
  --output artifacts/parity-p25.json
```

Repeat with P=0. For the main reference, first run `git worktree add --detach oracle-main 6dbed001a53cfd3cd5ea02b597142b96114ae3bf`, then use `--source oracle-main --implementation oracle`.
Use `--backend tensorrt-fp16` and `compare --report-only` for reduced precision;
compare equal `--lowpass-hz` settings only. Captures reject mismatched checkpoint,
input, text, steps, prefix, filtering and validation-script contracts. See
[plan/cache/benchmark smoke commands](../../deployment/RUNTIME.md).
