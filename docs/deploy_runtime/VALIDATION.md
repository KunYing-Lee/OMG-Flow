# Runtime validation — 2026-11-16

Implementation commit: `aa92092bca48456d963614f11ec527680adbdd04`.
Base main: `6dbed001a53cfd3cd5ea02b597142b96114ae3bf`.
Oracle: `cce4ad3` in a separate `git worktree add --detach` checkout.
No `src/` or training `configs/` changes. No training or merge was performed.

## Checkpoint and inputs

- Checkpoint SHA256: `3f8d9f8a9cbc96cb034e6c0cb2b1bfccd38816a8dca66c68df3f24b7f49f9cde`.
- Copied with `scp -3 hq-g2:/root/leeky/omg-flow/runs/bones-native50-4x5090-20261107/best.pt charlie-5090:~/flow-handoff-20261116/assets/omg_flow_best.pt`; destination SHA verified.
- GPU: one RTX 5090; Torch `2.8.0+cu128`, TensorRT `10.13.3.9`, Python 3.10; Torch TF32 disabled for captures. Exact [environment](results/environment.txt).
- Euler-10, batch 1, history10/horizon100, P=0 and P=25. Unfiltered FP32 parity.
- Prompt: `A person walks forward.` One pinned T5 cache shared by every implementation; signature `8c7d1e0e74149251634fa5100e97330830362c78970f901802b1c8c975e38a85`.
- Golden seed 20261116 creates smooth xyz, roll/pitch/yaw and joint poses plus full `[100,38]` noise. Replays rotate/translate both history and prefix, and roll noise rows, on the same runtime instance.
- Captures execute in separate processes with the selected source checkout inserted before imports. Receipts preserve actual source path, commit, source file hashes, input hash, validation script hash and cache signature. The oracle calls its unmodified `FlowReferenceGenerator.generate_suffix`, the path used by Posttrain's plain Euler decoder.

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

Changed main utilities were examined separately on identical input:

- `rotation_conversions.py`: axis-angle quaternion, inversion and matrix-to-quaternion probes are bitwise identical, including zero, small angles and near pi. End-to-end FP32 qpos and kinematics are also identical on the fixed cases.
- `tracking/holomotion/reference.py`: `native_qpos` is identical. Batched tracking features are **not bitwise identical**: max differences are quaternion `5.96e-8`, gravity `2.98e-7`, base linear velocity `1.49e-8 m/s`, and base angular velocity `2.46e-6 rad/s`. The tracked poses themselves are unchanged. These features are outside Flow sampling.
- `pipeline/checkpoint.py`: inspection confirms it is the diffusion `CheckpointPlanner`, outside the Flow call graph. It was **not executed** with the Flow checkpoint and has no role in these inference outputs; only the shared `native_qpos` validation path was probed. This is not an empirical equivalence claim for diffusion checkpoint inference.

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

CPU validation with a small random model requires no checkpoint: P=0/25,
Euler-5/10, padded text masks, external unused-prefix NaNs, seed equivalence,
filter-only-suffix behavior and verified cache loading without T5.

```bash
HF_HOME=/private/tmp/flow-hf-cache MPLCONFIGDIR=/private/tmp/flow-matplotlib \
  .venv/bin/python -m pytest -q
# 259 passed, 1 skipped, 7 warnings (16.77 s)
.venv/bin/python -m pytest -q tests/generation tests/deployment
# 190 passed, 1 skipped (initial required-suite run)
```

The full-suite skip is the CUDA variant of `test_training_observability` on the CPU-only Mac. Initial collection lacked
data/render extras; after installing them, five data tests encountered a sandbox
cache permission error. Setting `HF_HOME` inside `/private/tmp` resolved it without
changing code or tests. Independent read-only code review found no blocking issue.

## Reproduction

Use an idle device: check `nvidia-smi` before **each** GPU command. Capture itself
prints `nvidia-smi` and rejects existing compute processes. All remote source,
venv, worktrees, caches and result files are under
`~/flow-handoff-20261116/omgflow/`; model assets are in the explicitly allowed
`~/flow-handoff-20261116/assets/`.

```bash
cd ~/flow-handoff-20261116/omgflow
export PYTHONPATH=src OMP_NUM_THREADS=1 TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface" CUDA_CACHE_PATH="$PWD/.cache/cuda"
git worktree add --detach oracle-cce4ad3 cce4ad3
python scripts/validation/parity_cce4ad3.py golden --prefix 25 --output artifacts/golden-p25.npz
# Create the cache with deployment.flow_runtime text-cache (see deployment/RUNTIME.md).
python scripts/validation/parity_cce4ad3.py capture \
  --source oracle-cce4ad3 --implementation oracle \
  --checkpoint ../assets/omg_flow_best.pt --text-cache artifacts/text-cache \
  --input artifacts/golden-p25.npz --output artifacts/cce4ad3-p25.npz
python scripts/validation/parity_cce4ad3.py capture \
  --source . --implementation runtime \
  --checkpoint ../assets/omg_flow_best.pt --text-cache artifacts/text-cache \
  --input artifacts/golden-p25.npz --output artifacts/runtime-fp32-p25.npz
python scripts/validation/parity_cce4ad3.py compare \
  --reference artifacts/cce4ad3-p25.npz --candidate artifacts/runtime-fp32-p25.npz \
  --output artifacts/parity-p25.json
```

Repeat with P=0; use `--source oracle-main --implementation oracle` for main.
Use `--backend tensorrt-fp16` and `compare --report-only` for reduced precision;
compare equal `--lowpass-hz` settings only. Captures reject mismatched checkpoint,
input, text, steps, prefix, filtering and validation-script contracts. See
[plan/cache/benchmark smoke commands](../../deployment/RUNTIME.md).

## Remote working-copy audit

Read-only archive streams from all three requested directories were downloaded;
symlinked `src` was followed and Python bytecode excluded. Compared against main
`6dbed00` in `deployment/`, `src/`, `tests/`, `scripts/`.
Full [difference list](remote_audit.json) and [per-file SHA256 / matching historical commits](remote_provenance.json).

| Source | Files inspected | Differing or remote-only files | Decision |
|---|---:|---:|---|
| `OMG-Flow-deploy-sync-trt-20261109` working tree | 161 | 37 | All match prior main ancestors; do not restore |
| `OMG-Flow-deploy-sync` working tree | 160 | 41 | Historical code plus stale SYNC.md; do not restore |
| `OMG-Flow-default-20261109` working tree | 184 | 2 | Older SYNC.md and pre-preview-offset showcase script; retain main |
| default `code.tar.gz` | 184 | 5 | Earlier sync/showcase code/tests; retain main |

The retired async/residual/controller/realtime files were explicitly removed
in `b84dfa3`. They are not missing new Flow runtime work. No remote code was
imported. Three `source.tar.gz`/`source.tgz` files fail gzip integrity checks;
their SHA256 matches the actual remote files, confirming source truncation,
not transfer damage. Their inaccessible tails cannot be audited. Valid working
trees and default `code.tar.gz` were compared in full. No writes were made to hq-g2.

## Remote artifacts

[File inventory](results/file-inventory.json) gives task-owned groups and byte
counts; unrelated concurrent `posttrain/` work is excluded. Approximate logical
sizes: venv 11.85 GB; checkpoint 522.8 MB; local T5 893.8 MB; TRT engine/ONNX/receipt
266.1 MB; git 78.2 MB; transferred source archive 75.6 MB; oracle worktrees 43.2 MB;
final numerical artifacts ~0.91 MB, initial yaw-only evidence ~0.70 MB. No checkpoint,
engine, large NPZ or other file over 1 MB is added to git.

The provided local clone's `.git` was sandbox read-only. The review branch lives
in `/private/tmp/omg-flow-runtime-20261116`; the original checkout is untouched.
Notion guide/search/skill listing all returned reauthentication errors; this
handoff is recorded in the repository and PR, **not saved to Notion**.
