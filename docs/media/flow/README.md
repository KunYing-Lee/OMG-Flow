# Flow showcase

User-selected checkpoint: **190,500 steps**. These are selected HoloMotion physics executions, not reference-only animation. Each full MP4 contains 20 seconds at normal speed. Previews are eight-second GIF excerpts. This showcase choice does not establish an overall best checkpoint.

Within each displayed category we selected the lowest full-rollout mean root tracking error among the three previously evaluated seeds. No poses, actions, or simulation results were modified. Camera tracking and overlays affect presentation only. The GIF loop is a playback loop, not a tested continuation across its end.

## Dance choreography

Original prompt: person performs four reps of a complex dance choreography in one place

- Clip: `dancing_routine_opt_2_001__A039`; seed: 2345.
- Full-rollout root error: 0.050 m; joint MAE: 0.062 rad.
- GIF excerpt: 4.00 to 12.00 seconds.
- [Full video](dance.mp4)

## Repeated turns

Original prompt: individual in a neutral position makes a 90 degree turn to their left, stops, then repeats whole action many times

- Clip: `idle_turn_090_R_long_002__A549`; seed: 1234.
- Full-rollout root error: 0.072 m; joint MAE: 0.070 rad.
- GIF excerpt: 0.00 to 8.00 seconds.
- [Full video](turn.mp4)

## Upward reach

Original prompt: a jump reaching upwards as high as possible

- Clip: `reach_jump_R_002__A217`; seed: 1234.
- Full-rollout root error: 0.143 m; joint MAE: 0.073 rad.
- GIF excerpt: 0.00 to 8.00 seconds.
- [Full video](jump.mp4)

The executed upward reach is shown here; the tracker does not fully reproduce the jumping amplitude in the source prompt. This is not a claim of successful jump tracking.

[Machine-readable provenance](manifest.json) records checkpoint and input trajectory hashes. [Full evaluation](../../flow-long-eval.md) includes all six categories and three seeds per checkpoint, including weaker results. Original OMG-Lite GIFs elsewhere in `docs/media` are inherited assets and are not used as OMG-Flow results.

Re-render the full videos with `PYTHONPATH=src python scripts/validation/render_flow_showcase.py --artifacts outputs/eval-200k`. The source trajectories must be present; rendering does not rerun physics. GIFs use the recorded excerpt offsets, 12.5 fps, 384-pixel width, and FFmpeg palette generation.
