# Generation and simulation

The default deployment/inference path is `python -m deployment.run_sync`, or
`make deploy`. Configure model paths in `deployment/sync_flow.json`.
The loop runs at 50 Hz and replans synchronously every ten ticks (0.2 s), using
TensorRT FP16 Euler-10 and an 8 Hz suffix low-pass with a clean 25-frame prefix.

```bash
CUDA_VISIBLE_DEVICES=0 make deploy DEPLOY_ARGS='--prompt "A person walks forward."'
```

See [deployment](../deployment/README.md) for setup and
[SYNC.md](../deployment/SYNC.md) for timing, noise, commitment and filter details.
No planner server, background generation or request/lead scheduler is used.

For programmatic offline continuation use `FlowReferenceGenerator` as described
in [the Flow method](flow-method.md#continue-an-exposed-reference).
The diffusion generation/export tools remain for training and benchmark
reproduction; their checkpoints are not accepted by the Flow deployment path.
