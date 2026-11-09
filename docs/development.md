# Development

Run `make test` in the full project environment. Sync tests cover indexed noise,
immutable committed poses and timestamps, preview/velocity-stencil coverage,
control ordering, timing accounting, and the specified suffix filter.
Training, data, benchmark and HoloMotion tests remain independent of deployment.

For GPU validation use `deployment.check_sync_backend` and the guarded
`scripts/validation/sync_showcase.py`. Retain raw reference/execution arrays,
model hashes, fixed prompts/seeds and latency reports. Inspect sampled video
frames as well as failure metrics; upright survival alone is not prompt success.
