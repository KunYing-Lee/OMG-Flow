# Synchronous simulation gallery

All 16 clips were recorded on November 9, 2026 using the default **50 Hz synchronous controller, 0.2 s replanning, TensorRT FP16 Euler-10, P25, and 8 Hz suffix low-pass**. The left panel is the generated reference; the right is HoloMotion executing in MuJoCo. Each MP4 is 20 seconds at 25 fps, 960 × 360, normal simulation speed, at most 1.38 MB. GIFs are 8-second excerpts at 8 fps, 480 pixels wide, at most 2.05 MB; their explicit offsets are in [cases.json](cases.json).

The checkpoint is the historical README step **190,500**, SHA-256 `3f8d9f8a9cbc96cb034e6c0cb2b1bfccd38816a8dca66c68df3f24b7f49f9cde`. All eight old prompts and seeds are preserved, and eight new cases use seed 1234. Every case starts from the tracker standing pose, with no dataset history. Historical source clip IDs/offsets remain in cases.json for provenance only. This is a regeneration, not a controlled comparison with the old dataset-initialized gallery. No seed search, pose edits or outcome-based selection was used.

## Observed behavior

Walking, repeated turning, arm raises and low leg extensions are visible. The complex dance, pickup and punch instructions are not fully realized. Jump references rise substantially higher than the executed motion. Running and side stepping develop large position drift and can leave the matched camera. No topple was observed in the sampled frames. The squat triggers the height screen at 7.90 s because of a deep crouch, remains supported in the inspected frames, and does not fully stand back up by 20 s.

The diagnostic failure screen is pelvis height < 0.35 m or pelvis tilt > 60 degrees. It flags squat only; it is not a semantic-success or contact-based fall detector. All cases continue for the full 20 seconds. Sample times and observations are recorded per clip in [manifest.json](manifest.json).

Across 16,000 ticks, reports contain 6 compute overruns and 10 deadline misses. Simulation waits during overruns; playback follows simulation time. These are simulation results, not physical-robot acceptance. The recording environment used MuJoCo 3.11.0; the package currently pins 3.7.x, so the gallery is not a validation of that older simulator version. Exact runtime, source/asset hashes and GPU process logs are retained in the manifest.

## Fixed cases

| Clip | Seed | First height/tilt flag | Root RMSE (m) | Compute overruns |
|---|---:|---:|---:|---:|
| [dance](dance.mp4) | 2345 | none | 0.056 | 0 |
| [turn](turn.mp4) | 1234 | none | 0.039 | 0 |
| [jump](jump.mp4) | 1234 | none | 0.180 | 0 |
| [walk](walk.mp4) | 3456 | none | 0.069 | 0 |
| [run](run.mp4) | 1234 | none | 1.305 | 0 |
| [kick](kick.mp4) | 3456 | none | 0.296 | 0 |
| [energetic_dance](energetic_dance.mp4) | 2345 | none | 0.077 | 2 |
| [crouch_rebound](crouch_rebound.mp4) | 1234 | none | 0.289 | 2 |
| [walking_forward](walking_forward.mp4) | 1234 | none | 0.350 | 0 |
| [side_step](side_step.mp4) | 1234 | none | 1.258 | 0 |
| [wave_arms](wave_arms.mp4) | 1234 | none | 0.045 | 0 |
| [squat](squat.mp4) | 1234 | 7.90 s | 0.054 | 0 |
| [jump_in_place](jump_in_place.mp4) | 1234 | none | 0.110 | 0 |
| [front_kick](front_kick.mp4) | 1234 | none | 0.378 | 1 |
| [punch](punch.mp4) | 1234 | none | 0.037 | 0 |
| [spin](spin.mp4) | 1234 | none | 0.263 | 1 |

### dance

> person performs four reps of a complex dance choreography in one place

Upright small steps and arm motions; close tracking, but not the requested complex choreography.

[MP4](dance.mp4) · [GIF](dance.gif) · [Raw trajectory](dance.npz) · [Timing and metrics](dance.json)

### turn

> individual in a neutral position makes a 90 degree turn to their left, stops, then repeats whole action many times

Visible repeated turning with close reference/execution agreement; no fall in sampled frames.

[MP4](turn.mp4) · [GIF](turn.gif) · [Raw trajectory](turn.npz) · [Timing and metrics](turn.json)

### jump

> a jump reaching upwards as high as possible

Upward arm reach is visible; reference pelvis peaks at 1.11 m while execution peaks at 0.84 m. The large requested jump is not reproduced.

[MP4](jump.mp4) · [GIF](jump.gif) · [Raw trajectory](jump.npz) · [Timing and metrics](jump.json)

### walk

> walk forward, lean down to pick up an object, then continue the walk

Forward walking is visible and tracked; the requested bend/pickup phase is not apparent.

[MP4](walk.mp4) · [GIF](walk.gif) · [Raw trajectory](walk.npz) · [Timing and metrics](walk.json)

### run

> character waves back over left shoulder and talks with a child then starts to run backwards

Backward-motion prompt produces locomotion, but large world-position drift develops; execution moves partly out of the matched camera. Root-position RMSE is 1.31 m.

[MP4](run.mp4) · [GIF](run.gif) · [Raw trajectory](run.npz) · [Timing and metrics](run.json)

### kick

> individual with an injured torso kicks an item on the ground and then takes three steps

Low leg extensions/kicks and stepping are visible; no strong injured-torso style or exact three-step sequence is established.

[MP4](kick.mp4) · [GIF](kick.gif) · [Raw trajectory](kick.npz) · [Timing and metrics](kick.json)

### energetic_dance

> person performs four reps of a complex dance choreography in one place

Small stepping and arm gestures, with a late arm spread. The complex four-repetition choreography is not established; the historical energetic name is retained.

[MP4](energetic_dance.mp4) · [GIF](energetic_dance.gif) · [Raw trajectory](energetic_dance.npz) · [Timing and metrics](energetic_dance.json)

### crouch_rebound

> high jump over a moving pole in one place

A crouch and rebound are visible. The high-jump reference reaches 1.06 m pelvis height; execution reaches only 0.83 m and shows substantial height/orientation mismatch.

[MP4](crouch_rebound.mp4) · [GIF](crouch_rebound.gif) · [Raw trajectory](crouch_rebound.npz) · [Timing and metrics](crouch_rebound.json)

### walking_forward

> A person walks forward.

Sustained forward walking is visible; execution remains upright but world-position drift accumulates (0.35 m root RMSE).

[MP4](walking_forward.mp4) · [GIF](walking_forward.gif) · [Raw trajectory](walking_forward.npz) · [Timing and metrics](walking_forward.json)

### side_step

> A person takes steps sideways to the left.

Side-stepping motion is visible, but execution develops large positional drift (1.26 m root RMSE) and partly leaves the matched camera.

[MP4](side_step.mp4) · [GIF](side_step.gif) · [Raw trajectory](side_step.npz) · [Timing and metrics](side_step.json)

### wave_arms

> A person waves both arms overhead.

Both arms rise, followed by asymmetric waving. Reference and execution agree closely; continuous symmetric overhead waving is not maintained.

[MP4](wave_arms.mp4) · [GIF](wave_arms.gif) · [Raw trajectory](wave_arms.npz) · [Timing and metrics](wave_arms.json)

### squat

> A person performs a squat and stands back up.

Deep squat crosses the 0.35 m height screen at 7.90 s (minimum 0.31 m). No visible topple: pelvis tilt stays below 47 degrees. The robot does not fully stand back up by 20 s.

[MP4](squat.mp4) · [GIF](squat.gif) · [Raw trajectory](squat.npz) · [Timing and metrics](squat.json)

### jump_in_place

> A person jumps in place.

Jump attempts are visible, but execution reaches only 0.84 m pelvis height versus 1.07 m in the reference; jump amplitude is not reproduced.

[MP4](jump_in_place.mp4) · [GIF](jump_in_place.gif) · [Raw trajectory](jump_in_place.npz) · [Timing and metrics](jump_in_place.json)

### front_kick

> A person kicks one leg forward.

Low forward leg lifts/extensions appear late, with stepping and positional drift (0.38 m root RMSE); no strong front kick is established.

[MP4](front_kick.mp4) · [GIF](front_kick.gif) · [Raw trajectory](front_kick.npz) · [Timing and metrics](front_kick.json)

### punch

> A person throws a punch forward.

Raised-arm gestures are tracked closely, but a clear forward punch is not apparent.

[MP4](punch.mp4) · [GIF](punch.gif) · [Raw trajectory](punch.npz) · [Timing and metrics](punch.json)

### spin

> A person spins around in place.

Repeated rotation is visible and remains upright; unwrapped pelvis yaw changes by about 936 degrees in the reference and 899 degrees in execution.

[MP4](spin.mp4) · [GIF](spin.gif) · [Raw trajectory](spin.npz) · [Timing and metrics](spin.json)

## Validation

The full suite passes on the GPU host: **253 passed** (one warning). macOS: **252 passed, one CUDA-only skip**. The [backend comparison](backend_check.json) covers 40 fixed windows (20 each at Euler-10 and Euler-5): repeats are exact, graph and eager TRT outputs are identical, and the unfiltered Torch adapter agrees with the upstream generator within `atol=rtol=1e-5`. FP16 approximation error is measured separately: at Euler-10, the worst TRT-versus-FP32 difference is **0.0132 m root position / 0.1133 rad joint angle**. These are numerical measurements, not an accuracy or physical-safety acceptance threshold. The audit verifies all fixed inputs, hashes, 16,000 ticks, 1,600 replans, failure-screen calculations and media limits.

## Reproduce

Set local asset paths in `deployment/sync_flow.json`, then run from the repository root:

```bash
make showcase SHOWCASE_ARGS="--gpu 0"
```

The command refuses an occupied GPU and stops its own child if another process appears. The archived queue log includes interrupted attempts; accepted clips have no foreign processes in their recorded guard samples. Source commit `af52b1d` produced all raw videos/trajectories; the later presentation change only selects GIF excerpts. The previous gallery is available in [Git history](https://github.com/KunYing-Lee/OMG-Flow/tree/e23d7a9/docs/media/flow).
