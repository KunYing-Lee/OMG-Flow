# OMG-Flow

OMG-Flow generates 50 Hz Unitree G1 reference motion conditioned on text, ten frames of history, and the future frames already promised to a tracker. It uses the native38 pose representation and transformer blocks from [OMG-Lite](https://github.com/KunYing-Lee/OMG-Lite/commit/1ab545f74be5a0f1470af269fa96d64b27c6066c), derived from [OMG](https://github.com/Tsinghua-MARS-Lab/OMG). Upstream MIT licensing and history are retained.

## What is learned

A training window consists of ten observed frames followed by a 100-frame plan. For each example, sample a committed length uniformly from 0 through 50 inclusive. The first P future frames are clean conditioning tokens, identified by a binary mask and endpoint time 1. The remaining frames follow the straight conditional flow path from Gaussian noise to the normalized pose. The transformer regresses its velocity on the suffix alone. Each example contributes its mean loss over editable coordinates. A deterministic Euler solver integrates only those coordinates. This adapts the clean-prefix formulation in [HALO’s trainer](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/generator/training/trainer.py).

Training starts from random weights. Positions, rotations and joint angles use the coordinate system anchored to the last history frame, and are standardized using Bones-Seed training actors alone. The model sees frozen T5-base token embeddings for the corresponding Bones-Seed clip. The four source descriptions supply caption variants.

## Prepare Bones-Seed

Use an account with access to the [Bones Studio release](https://huggingface.co/datasets/bones-studio/seed). The converter expects the official G1 archive and v004 metadata at revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`. The code checks both source hashes. It reads CSV columns by joint name, converts cm to m and degrees to radians, and resamples from 120 to 50 Hz with quaternion SLERP for root orientation. Each actor and all their mirrors belong to one split. Validation and test each hold out 26 actors; these are project-defined splits. A window requires 110 real frames from one clip. The manifest records byte hashes, counts and split membership summaries.

From the repository root with the training and data dependencies installed:

```sh
export PYTHONPATH=src
python -m omg.data.bones_prepare --archive /data/source/g1.tar.gz \
  --metadata /data/source/seed_metadata_v004.parquet --root /data/bones
python -m omg.cli.data.bones_stats --data /data/bones
python -m omg.cli.data.cache_bones_text --data /data/bones --output /data/bones-text \
  --t5 /models/t5-base --device cuda
```

The text cache is built once from [the pinned T5-base snapshot](https://huggingface.co/google-t5/t5-base/tree/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1). Only the model, data index and cache are needed in the training loop.

## Validate and train

```sh
export PYTHONPATH=src
python -m pytest -q tests/data/test_bones.py tests/generation/test_flow*.py
export FLOW_DATA=/data/bones FLOW_TEXT_CACHE=/data/bones-text FLOW_OUTPUT=/runs/bones-flow
make smoke
make train TRAIN_ARGS='--resume --stop-after 16'
make train TRAIN_ARGS=--resume
```

For a dedicated four-GPU allocation, `scripts/train_bones.sh` verifies the packed files and caption cache, runs the two pilot stages, and proceeds automatically when both succeed. Set `BONES_DATA`, `BONES_TEXT_CACHE` and `FLOW_OUTPUT` to the paths mounted in that allocation.

The smoke target trains four ranks to step 12, checks held-out losses and conditional samples, and writes a checkpoint. The second restores optimizer state and advances to step 16 with the full-run learning-rate schedule. The third continues to the configured step count. The preset is in `configs/flow/bones.yaml`. A scratch pilot can also use a conversion with `--limit` and an independent output directory. Full pretraining checks the manifest's completeness. Metrics are appended to `metrics.jsonl`. Atomic `latest.pt` and `best.pt` contain parameters, optimizer state, normalization, step and run provenance.

## Continue an exposed reference

```python
from omg.generation.flow_reference import FlowReferenceGenerator

generator = FlowReferenceGenerator.from_checkpoint(checkpoint_path, device="cuda")
suffix = generator.generate_suffix(
    history_qpos, committed_qpos, gaussian_noise,
    steps=10, text_tokens=t5_tokens, text_mask=t5_mask,
)
```

Inputs are world qpos36 in xyz, wxyz and the OMG G1 joint order. Pass the ten frames immediately preceding the committed prefix. The explicit Gaussian tensor is always shaped `[batch, 100, 38]`. At P=25 the result contains 75 new world-space poses; the caller keeps the original 25 poses in its buffer. The same noise and conditioning reproduce a plan in evaluation mode. Tests cover a second plan made after the history anchor moves. Physical tracking and the quality of the first new frame require separate evaluation. The inherited diffusion reference trainer is available at `omg.cli.generation.train_diffusion`; OMG-Flow checkpoints use the flow generator above.
