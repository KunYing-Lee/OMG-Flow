<div align="center">
  <h1>OMG-Flow</h1>
  <p>Native 50 Hz · Flow Matching · Committed-Prefix Continuation</p>
  <p>
    Built on <a href="https://github.com/KunYing-Lee/OMG-Lite">OMG-Lite</a> and
    <a href="https://github.com/Tsinghua-MARS-Lab/OMG">OMG: Omni-Modal Motion Generation for Generalist Humanoid Control</a>.
  </p>
  <p>
    <a href="https://github.com/KunYing-Lee/OMG-Lite"><img src="https://img.shields.io/badge/GitHub-OMG--Lite-181717?logo=github" alt="OMG-Lite"></a>
    <a href="https://huggingface.co/datasets/bones-studio/seed"><img src="https://img.shields.io/badge/Dataset-BONES--SEED-FFD21E" alt="BONES-SEED"></a>
    <a href="docs/flow-method.md"><img src="https://img.shields.io/badge/Docs-Method-007EC6" alt="Method"></a>
    <a href="docs/flow-long-eval.md"><img src="https://img.shields.io/badge/Results-Physics_Replay-377D53" alt="Evaluation"></a>
  </p>
</div>

OMG-Flow generates text-conditioned Unitree G1 motion at **native 50 Hz**
in a **38D pose space**, using flow matching to continue an already committed reference.

- **Flow matching:** learn a velocity field and generate motion with a deterministic Euler solver.
- **Clean committed prefix:** preserve the frames already exposed to the tracker and generate only the editable suffix.
- **BONES-SEED training:** actor-separated data, train-only normalization, and frozen T5-base text embeddings.

<table>
  <tr>
    <td align="center" width="33%"><a href="docs/media/flow/dance.mp4"><img src="docs/media/flow/dance.gif" alt="Generated dance tracked in simulation" width="100%"></a><br>Dance choreography</td>
    <td align="center" width="33%"><a href="docs/media/flow/turn.mp4"><img src="docs/media/flow/turn.gif" alt="Generated turns tracked in simulation" width="100%"></a><br>Repeated turns</td>
    <td align="center" width="33%"><a href="docs/media/flow/jump.mp4"><img src="docs/media/flow/jump.gif" alt="Generated upward reach tracked in simulation" width="100%"></a><br>Upward reach</td>
  </tr>
  <tr>
    <td align="center" width="33%"><a href="docs/media/flow/walk.mp4"><img src="docs/media/flow/walk.gif" alt="Walking" width="100%"></a><br>Walking</td>
    <td align="center" width="33%"><a href="docs/media/flow/run.mp4"><img src="docs/media/flow/run.gif" alt="Backward movement" width="100%"></a><br>Backward movement</td>
    <td align="center" width="33%"><a href="docs/media/flow/kick.mp4"><img src="docs/media/flow/kick.gif" alt="Stepping and kicking" width="100%"></a><br>Stepping and kicking</td>
  </tr>
</table>

## 1. Install

```bash
git clone https://github.com/KunYing-Lee/OMG-Flow.git
cd OMG-Flow
make venv
source .venv/bin/activate
make install
export PYTHONPATH=src
```

Use `make install-cn` for China mainland networks. For RTX 5090 training,
create the environment with `make venv VENV_PYTHON=3.11` and use
`make install-blackwell` instead of `make install` (PyTorch 2.8 / CUDA 12.8).
See [environment details](docs/flow-method.md#install).

## 2. Prepare BONES-SEED

Obtain the official G1 archive and v004 metadata from
[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed), and the
[pinned T5-base snapshot](https://huggingface.co/google-t5/t5-base/tree/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1).

```bash
python -m omg.data.bones_prepare --archive /data/source/g1.tar.gz \
  --metadata /data/source/seed_metadata_v004.parquet --root /data/bones
python -m omg.cli.data.bones_stats --data /data/bones
python -m omg.cli.data.cache_bones_text --data /data/bones \
  --output /data/bones-text --t5 /models/t5-base --device cuda
```

Conversion produces native 50 Hz clips with actor-separated train, validation and
test splits. See [data provenance](docs/data/bones.md) and
[preparation details](docs/flow-method.md#prepare-bones-seed).

## 3. Train

```bash
export FLOW_DATA=/data/bones
export FLOW_TEXT_CACHE=/data/bones-text
export FLOW_OUTPUT=/runs/bones-flow
make smoke
make train TRAIN_ARGS='--resume --stop-after 16'
make train TRAIN_ARGS=--resume
```

The default uses four GPUs, ten history frames and a 100-frame future window.
Each example samples a committed-prefix length from 0–50; that prefix stays
clean and masked, and only the suffix contributes to flow-matching loss.
The [preset](configs/flow/bones.yaml) runs for 200k updates.

Checkpoints include weights, optimizer, normalization and data provenance.
**The current trainer retains only rolling `latest.pt` and validation-loss
`best.pt`; it does not archive every step.** See
[training details](docs/flow-method.md#validate-and-train) and
[validation evidence](docs/flow-validation.md).

## 4. Generate a Continuation

Use a checkpoint produced by the Flow trainer. Flow weights are not bundled
with this repository; the upstream OMG diffusion checkpoint is not compatible.

```python
from omg.generation.flow_reference import FlowReferenceGenerator

generator = FlowReferenceGenerator.from_checkpoint("/runs/bones-flow/best.pt", device="cuda")
suffix = generator.generate_suffix(
    history_qpos, committed_qpos, gaussian_noise,
    steps=10, text_tokens=t5_tokens, text_mask=t5_mask,
)
```

Inputs are world-space G1 poses `[B, T, 36]`: xyz, wxyz, and 29 joint angles.
Pass ten history frames immediately before the committed prefix, plus explicit
noise `[B, 100, 38]` and cached T5 tokens. With a 25-frame prefix, the returned
suffix contains 75 new poses; keep the original prefix in the reference buffer.
See [the generation contract](docs/flow-method.md#continue-an-exposed-reference).

## 5. Simulate and Evaluate

Use [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) tracking artifacts
separately. The paired evaluation uses fixed test clips, text and noise seeds,
20-second autoregressive continuations, and unmodified native 50 Hz physics replay.

```bash
python scripts/validation/flow_long_eval.py generate \
  --data /data/bones --text /data/bones-text \
  --run /runs/bones-flow --out outputs/flow-eval
MUJOCO_GL=disable python scripts/validation/flow_long_eval.py track \
  --out outputs/flow-eval --onnx /models/holomotion/model_16200.onnx
python scripts/validation/flow_long_eval.py render \
  --out outputs/flow-eval --with-gt
```

Rendering requires a working OpenGL context. See
[protocol, metrics and limitations](docs/flow-long-eval.md).
In the six-clip, three-seed suite, both evaluated checkpoints completed all
18 rollouts without crossing the height/tilt failure screen. Root tracking
drift and excess joint acceleration remain; real-time and hardware acceptance
have not been established.

## Acknowledgements

Based on [OMG](https://github.com/Tsinghua-MARS-Lab/OMG) and
[OMG-Lite](https://github.com/KunYing-Lee/OMG-Lite), with motion data from
[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed), tracking from
[HoloMotion](https://github.com/HorizonRobotics/HoloMotion), and clean-prefix
training inspired by [HALO](https://apple-aiml-research.github.io/ml-halo/).

[MIT License](LICENSE)
