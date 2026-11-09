# Historical diffusion benchmark

The default text protocol uses native **50 Hz**, **100 predicted frames (2 s)**,
10 history frames, the 38D representation and `exp=50m`. Sampling uses the
checkpoint's 50-step DDIM configuration with CFG 2.5; seed 0 and batch size 32.
Generation and deployment settings are unchanged.

Prepare the fixed test set once from the pinned native50 dataset, then reuse it
for all checkpoint comparisons:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples
PYTHONPATH=src python -m omg.cli.generation.benchmark \
  --ckpt_path models/generation/model.ckpt \
  --evaluator_checkpoint models/evaluator/step_007000.pt
```

- Default manifest: `outputs/benchmark_samples/native50/text_test_1024.jsonl`.
  Preparation selects 1,024 captioned windows, balanced across the 12 text
  cohorts in `benchmarks/protocol.py`. The summary records counts and SHA-256.
- Generation and FID/KID reference use the same fixed manifest by default.
  A different reference split requires `--reference-samples-path`. Dataset
  revision, split, window identity and frame count must match; no resampling.
- Text metrics are enabled by default and require the evaluator's T5-3B text
  encoder. GT, generated and per-sample retrieval share seeded, dataset-stratified
  groups of 32 candidates. Reuse the same manifests, seed and evaluator for comparisons.
- FID/KID use all 100 native frames without temporal resampling. This protocol
  is distinct from OMG's 30 Hz / 60-frame release benchmark; scores are not
  directly comparable. The default evaluator path is also configurable with
  `OMG_MODELS_ROOT` or `--evaluator_checkpoint`.
- Multimodality requires `--multimodality-repeats > 1`. A single 100-frame chunk
  has no autoregressive seam, so transition metrics are reported as disabled.
  Longer-window or tracker-executed evaluations are separate protocols.

For Flow deployment parity use `python -m deployment.check_sync_backend`.
The fixed [sync showcase](media/flow/README.md) executes generation and tracking
online in the same control loop; the [long evaluation](flow-long-eval.md) is a
separate offline checkpoint-comparison protocol.
