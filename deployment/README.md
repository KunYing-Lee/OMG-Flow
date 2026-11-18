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

## Realtime planner

The realtime server runs generation separately from the consuming node, using
protocol v2 at 50 Hz. Run the commands below from the repository root.

1. Install the realtime extra. TensorRT setup also needs the existing export
   dependencies and TensorRT on the CUDA deployment system:

   ```bash
   python -m pip install -e '.[realtime,export,tracking]'
   ```

   For CPU Torch-only use, `python -m pip install -e '.[realtime]'` is sufficient.
   Prepare a local text cache using the [runtime command](RUNTIME.md#text-cache-and-offline-robot-use),
   or install `.[train]` and configure a local text encoder.

2. Export/build the TensorRT velocity engine with the existing exporter. Write
   `engine_config.json` with paths resolved from the working directory:

   ```json
   {
     "checkpoint": "<checkpoint>",
     "text_cache": "<text_cache>",
     "prompt": "<prompt>",
     "device": "cuda",
     "steps": 10,
     "prefix_length": 25,
     "lowpass_hz": 8.0,
     "engine_cache": "<engine_dir>"
   }
   ```

   ```bash
   python -m deployment.export_sync --config engine_config.json
   ```

   This exports ONNX and builds or validates the cached FP16 velocity engine.
   The same engine serves prefix lengths 0 and 25 when checkpoint, text shapes
   and device match. Set `engine_directory` in the realtime config to the same
   `<engine_dir>`. `engine_cache` is the exporter's field name. Torch-only use
   can skip this step; choose `torch-fp32` and `device=cpu`.

3. Copy [the example config](realtime/example_config.json) to `planner.json`,
   replace the asset paths and prompt, and provide a fixed float32 `[10,36]`
   start history in an NPZ. JSON paths resolve relative to `planner.json`.
   Add `noise_policy` only for a compatible sampled policy; `robot_xml` is
   needed only for an additional HoloMotion clip export.

4. Bake the reference before serving:

   ```bash
   python -m deployment.realtime bake --config planner.json --output baked.npz
   ```

5. Start the server with an explicit endpoint and log:

   ```bash
   python -m deployment.realtime serve --config planner.json \
     --bind 'tcp://<host>:<port>' --log server.jsonl
   ```

6. Connect a node implementing [protocol v2 and the START CONTRACT](realtime/README.md#schedule-and-start-contract).
   Plan zero uses the configured model history; the node's measured ten frames
   serve only the configurable start gate. Later histories must match the
   already-sent reference bitwise. To exercise the connection without a robot:

   ```bash
   python -m deployment.realtime client --connect 'tcp://<host>:<port>' \
     --history baked.npz --text '<prompt>' --plans 3 --switch-seconds 0.5
   ```

Use the same switch period on server and client. See the [realtime reference](realtime/README.md)
for config validation, deadlines, terminal preview, policy schema and CLI overrides.
