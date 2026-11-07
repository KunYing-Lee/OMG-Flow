# Showcase sources

All six videos use step 190,500 and recorded HoloMotion physics execution. Full videos are 20 seconds; GIFs are eight-second excerpts at normal speed. No text overlays or trajectory edits.

## dance

person performs four reps of a complex dance choreography in one place

Seed 2345; clip `dancing_routine_opt_2_001__A039`. [Full video](dance.mp4).

## turn

individual in a neutral position makes a 90 degree turn to their left, stops, then repeats whole action many times

Seed 1234; clip `idle_turn_090_R_long_002__A549`. [Full video](turn.mp4).

## jump

a jump reaching upwards as high as possible

Seed 1234; clip `reach_jump_R_002__A217`. [Full video](jump.mp4).

## walk

walk forward, lean down to pick up an object, then continue the walk

Seed 3456; clip `injured_torso_grab_walk_ff_180_002__A549`. [Full video](walk.mp4).

## run

character waves back over left shoulder and talks with a child then starts to run backwards

Seed 1234; clip `wave_right_hand_000_R_talking_with_child_jog_ff_start_360_R_001__A479`. [Full video](run.mp4).

## kick

individual with an injured torso kicks an item on the ground and then takes three steps

Seed 3456; clip `injured_torso_kick_trash_R_002__A217`. [Full video](kick.mp4).

Selected by lowest mean root tracking error within each category. The upward reach does not fully reproduce the jumping amplitude of its prompt. [Full evaluation](../../flow-long-eval.md) · [Provenance](manifest.json).

## Dynamic replacements

Walking and stepping/kicking in the main gallery were replaced by energetic dance and crouch/rebound. Both are new 190.5k float32 CPU generations, using clean P25 continuation and the same unmodified tracker. Six candidates (three held-out source clips, two seeds each) were evaluated. The selected cases completed 20 seconds without the height/tilt failure screen.

- [energetic_dance](energetic_dance.mp4): person performs four reps of a complex dance choreography in one place; seed 2345; root error 0.107 m; mean absolute joint speed 0.542 rad/s.
- [crouch_rebound](crouch_rebound.mp4): high jump over a moving pole in one place; seed 1234; root error 0.239 m; mean absolute joint speed 0.394 rad/s.

The high-jump prompt produces crouching and rebound in physics, not a verified high jump. See [replacement provenance](dynamic_manifest.json). The original walk/kick files are retained for audit but are no longer featured.
