# Configuration

- Flow training: [configs/flow/bones.yaml](../configs/flow/bones.yaml), consumed by
  `omg.cli.generation.train` / `make train`.
- Deployment: [deployment/sync_flow.json](../deployment/sync_flow.json), consumed
  by `deployment.run_sync` / `make deploy`. Set local checkpoint, T5 and tracker
  paths. Defaults are 50 Hz, 0.2 s replanning, P25, Euler-10, TensorRT FP16 and
  an 8 Hz suffix low-pass. `overrun=log` is for diagnostic simulation;
  `overrun=raise` rejects missed budgets. `pace=true` measures wall deadlines.
- Media cases: [docs/media/flow/cases.json](media/flow/cases.json), fixed prompts,
  seeds and durations for `make showcase`.

The diffusion Hydra configs under `configs/generation` remain for offline
training and benchmarks, including the pinned multimodal dataset. Their
normalization and architecture must match their checkpoints.
