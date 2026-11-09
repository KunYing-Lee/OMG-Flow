# Flow and historical diffusion artifacts

Flow deployment requires a Flow trainer checkpoint; set its path in
`deployment/sync_flow.json`. No Flow weights are bundled. The README video
manifest records the exact 190.5k checkpoint hash.

## Historical diffusion bundle (offline benchmarks only)

Use the [38D / 50Hz bundle](https://huggingface.co/THU-MARS/OMG/tree/main/checkpoints/38D).
The best-validation checkpoint is `sstep=075500.ckpt`, SHA256
`8bdd19cc6a47131dacb8f29a2a440c6a3a405dfcd335ecfffc3054828bd59fe7`.

Download the whole directory, including `config.yaml`, normalization stats,
`g1_kinematics.json`, `g1_29dof.urdf`, `seed_stand.npz`, and `infer.py`.
The script resolves paths relative to the bundle, verifies hashes and strictly
loads the checkpoint. T5-base is the generation text encoder; install the
training extras and provide a local T5-base directory for offline operation.
Follow the bundle README or the repository quickstart.

The bundle's validation record reports a 100-frame native50Hz generation from
standing history, with exact output parity against the original training config.
It is not tracker or physical-robot acceptance. The released checkpoint was
trained with diffusion loss only. Optional FK loss remains available for new runs.
