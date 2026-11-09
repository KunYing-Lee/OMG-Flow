# Inference contract

The default is the synchronous OMG-Flow controller documented in
[deployment/SYNC.md](../deployment/SYNC.md): 50 Hz control, 0.2 s replanning,
TensorRT FP16 velocity inference, Euler-10, clean P25 commitment and an 8 Hz
suffix low-pass. No backend substitution or approximate denoiser cache is used.

Flow checkpoints carry model configuration, normalization and provenance.
World-space G1 poses are `[xyz, wxyz, 29 joints]`; the model uses anchor-relative
38D features with ten history and 100 future frames. Explicit indexed noise is
independent of warmup. Tracker-visible poses and their velocity stencil remain
unchanged when a new suffix is installed.

Cold start/export, text encoding and CUDA graph warmup precede the control epoch.
Reports separate generation, reference/observation features, tracker ONNX,
physics/I/O and deadline misses. Simulation waits during compute overruns;
50 Hz video playback does not establish real-time or physical acceptance.

The retained diffusion export and benchmark code is for offline experiments,
not the deployment default. Flow and diffusion weights are not interchangeable.
