# Synchronous OMG-Flow deployment

The default is a single-thread 50 Hz control loop, replanning every 0.2 s with
TensorRT FP16, Euler-10, a 25-frame clean committed prefix and an 8 Hz suffix
low-pass. The same loop runs generation, tracker inference and MuJoCo physics.

1. Install the training, tracking, export and rendering dependencies.
2. Set the Flow checkpoint, T5 and HoloMotion paths in [sync_flow.json](sync_flow.json).
3. Select a process-free GPU, then run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 make deploy PYTHON=.venv/bin/python
# Override prompt, seed or duration explicitly:
CUDA_VISIBLE_DEVICES=0 make deploy DEPLOY_ARGS='--prompt "A person waves both arms." --seed 1234 --duration 20'
```

`make export-sync` builds the velocity engine ahead of the first rollout.
`make showcase SHOWCASE_ARGS='--gpu 0'` regenerates the fixed README suite and
stops its own child process if another process appears on that GPU.

Outputs are paired MP4, raw synchronized NPZ and JSON with configuration,
checkpoint/engine provenance, timing and height/tilt failure diagnostics.
The physical `RobotSDK` binding remains unimplemented; simulation is not
hardware acceptance. See [the control and inference contract](SYNC.md).
