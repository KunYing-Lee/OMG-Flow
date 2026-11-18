# Realtime planner

Run `python -m deployment.realtime {serve,bake,client}` from the repository root.
Install `.[realtime]` for sockets. Wire, schedule, server, timeline, config and
CLI help need only NumPy; constructing a Flow source loads Torch and the runtime.
See the [deployment walkthrough](../README.md#realtime-planner) for engine setup.

## Wire protocol v2

Use ZMQ REQ on the node and REP on the server. Every request and reply is exactly
two multipart frames, in order:

1. UTF-8 JSON header.
2. Compressed NPZ payload, decoded with `allow_pickle=False`.

Both headers contain `version` (integer `2`), `kind`, `created_time` (wall-clock
seconds), `request_id` (string), `prompt` (string), and `metadata` (object).

| Message | Additional header fields | NPZ array |
| --- | --- | --- |
| `robot_state_request` | `history_fps=50`, `tracker_frame`, `buffer_remaining_frames` (default 0), `last_plan_id` (nullable) | `qpos_36_history`, float32 `[10,36]` |
| `motion_plan_chunk` | `fps=50`, `plan_id`, `request_tracker_frame`, `planning_latency_seconds` | `qpos_36`, float32 `[100,36]` |

Motion columns are root XYZ in metres, root quaternion **WXYZ**, and 29 joint
positions in radians. All values must be finite. Quaternion norms must differ
from one by at most `1e-4`. Shape, dtype, nonfinite values, nonunit quaternions,
unsupported version/kind and object arrays requiring pickle are rejected.
The decoded arrays retain their float32 bit patterns.

Request metadata must include integer `plan_id`, `switch_frame`, and
`period_frames`. The plan ID is nonnegative, the switch must agree with the
schedule, and the prompt must equal the configured caption exactly. Plan zero
resets the session; subsequent IDs must be sequential. Invalid requests close
the server without a reply and are recorded in the JSONL log. There is no error
reply or implicit retry protocol.

Reply metadata preserves these keys:

- `switch_frame`, `period_frames`, `committed_frames`.
- `provisional_from`, `padded_from`: absolute stream frame indices, or null.
- `source` (`flow` or `flow-noise`), `caption`, `backend`, `seed`, `episode_id`.
- `play_frames`, `preview_frames=11`, `replan_terminal_preview=true`.
- `internal_replans`, `generated_end`, `runtime_contract`, `policy_sha256` (nullable).
- `engine`: null or `{path, cache_hit}`; reserved `package` is null.
- `realtime_transport`: `server_recv_wall_time`, `server_recv_perf_time`,
  `server_send_wall_time`, `server_send_perf_time`. Performance timestamps belong
  to the server's clock; do not subtract a timestamp from another process.

The reply echoes `request_id`, `prompt`, `plan_id` and `request_tracker_frame`.
Latency is server computation time, excluding transport. The client enforces
the delivery deadline.

## Schedule and START CONTRACT

Let wire period `P = 50 * switch_seconds`: an integer from 25 through 100.
Valid switch intervals are 0.5 through 2.0 seconds in 20 ms increments.
Plan zero initializes before playback with `tracker_frame=0`, `switch_frame=0`.
For plan ID `n >= 1`:

- Switch at `S = n * P`.
- Request at `S - 25`; lead is 25 frames.
- Accept by `S - 11`, before tracker preview needs the plan. The budget is
  14 frames, or 280 ms at 50 Hz.

**START CONTRACT:**

1. Plan zero's model history is the fixed ten-frame start history loaded from
   the config, **never measured robot state**.
2. The ten frames the node sends with plan zero are used **only for the safety
   gate**. Reject if any joint difference exceeds `start_gate.joint_rad`
   (default 0.3 rad), any root-height difference exceeds `start_gate.height_m`
   (default 0.10 m), or any wrapped yaw difference exceeds `start_gate.yaw_deg`
   (default 10 degrees). Compare every row with the corresponding fixed row.
   Horizontal translation is diagnostic only. `--allow-start-mismatch` bypasses
   rejection; model history still comes from the fixed config history.
3. For plan IDs >= 1 the request history must equal the frames already sent at
   `[S-10, S)` **bitwise**, including signed zero. These are committed reference
   frames, not fresh robot measurements.

The model's internal replan interval is always ten frames, independently of wire
P. It conditions on history10 plus committed prefix25. Tick `k` replaces only
`[k+25, k+100)`. For a reply at S, execute the needed ticks through `S+P-26`,
capped at `max(0, floor((play_frames-15)/10)*10)`. Initial generation uses request
index 0; tick k uses index `k/10`. These indices key noise independently of the
wire plan ID, so changing the transport period preserves the generated tape.

Only the first P reply rows are committed. Later rows may be provisional.
`provisional_from` marks where a future internal replan can replace rows. When
generated support does not fill the reply, `padded_from` marks repeated copies
of the last generated row. Padding exists only in the reply, never in model
context, and must not overlap the committed prefix. `validate_wire_schedule`
rejects insufficient terminal support before loading models or serving.

There are eleven terminal preview frames. `after_end=hold` retains that preview
and then repeats the last **played** frame. A final preview-only replan can occur
at the playback boundary. `after_end=error` rejects wire requests with
`S >= play_frames`. Baking uses the same internal schedule and exports
`play_frames+11` rows; `after_end` governs subsequent wire requests, not bake length.

## Configuration

Start from [example_config.json](example_config.json) and replace its relative
asset placeholders. Paths in JSON resolve against the config file's directory.
Unknown fields, missing required fields and incorrect types raise `ValueError`.
An engine directory is required for TensorRT; there is no backend substitution.

| Field | Contract |
| --- | --- |
| `checkpoint` | Required checkpoint path string |
| `text_cache` / `text_encoder` | Exactly one non-null path; cache directory or local text encoder |
| `prompt` | Required nonempty caption string; exact cache lookup |
| `steps` | Required positive integer Euler step count |
| `lowpass_hz` | Required; null disables filtering, otherwise number strictly between 0 and 25 Hz |
| `start_history` | Required object with exactly `path` (NPZ) and `key`; selected array must be float32 `[10,36]` |
| `play_frames` | Required positive integer played duration at 50 Hz |
| `seed`, `episode_id` | Required integers in `[0, 2**63)` |
| `backend` | Required: `torch-fp32`, `torch-fp16`, or `tensorrt-fp16` |
| `device` | Required device string; FP16 requires CUDA |
| `engine_directory` | Required path for TensorRT; otherwise optional/null |
| `noise_policy` | Optional/null sampled noise-policy checkpoint path |
| `robot_xml` | Optional/null XML path; enables additional HoloMotion clip export during bake |
| `start_gate` | Optional object: `joint_rad=0.3`, `height_m=0.10`, `yaw_deg=10.0`; each finite and nonnegative |
| `switch_seconds` | Optional, default 0.5; valid schedule period as above |
| `after_end` | Optional `hold` (default) or `error` |

`serve` and `bake` require `--config`. Both accept overrides `--backend`,
`--engine-directory`, `--device`, `--switch-seconds`, and `--after-end`.
CLI engine-directory paths resolve against the working directory.
`serve` additionally requires `--bind` and `--log`; optional `--max-plans` is a
positive integer and `--allow-start-mismatch` explicitly bypasses the start gate.
`bake` requires `--output`, storing `qpos_36`, `fps`, `play_frames`, `history`,
and JSON `metadata`. With robot XML configured it also writes a sibling
`.holomotion.npz` clip. No address or port is supplied by default.

The diagnostic `client` requires `--connect`, `--history` (NPZ with `history`),
and `--text`; it accepts `--plans` (default 3) and `--switch-seconds` (default 0.5).
It waits up to 30 seconds for initialization and enforces the frame deadline
for later replies. It exercises transport and reference continuity; it does not
control a robot.

## Noise-policy checkpoint and sampling

The checkpoint is loaded with `torch.load(..., weights_only=True)`. Its top level
must be a mapping containing `actor`. Optional `format_version` must be integer
`1`; omission is accepted. Other top-level metadata is unused. The actor state
dict must have exactly the following finite float32 tensors:

| Key | Shape |
| --- | --- |
| `log_std` | `[3800]` |
| `backbone.0.weight`, `backbone.0.bias` | `[512,1331]`, `[512]` |
| `backbone.2.weight`, `backbone.2.bias` | `[512,512]`, `[512]` |
| `mean_head.weight`, `mean_head.bias` | `[3800,512]`, `[3800]` |

Missing/extra actor keys, wrong shapes or dtypes, nonfinite parameters and
unsupported versions raise `ValueError` before `load_state_dict(strict=True)`.
The actor has SiLU after each hidden layer and state-independent log standard
deviation. No training package is needed at inference time.

Observation is the existing `encode_bones_window` encoding, anchored at the last
history frame and normalized with the Flow checkpoint mean/std. History10 and
optional prefix25 are padded with zeros to 35 rows, flattened, then appended
with the prefix-length/25 indicator, yielding 1331 values.

Paired epsilon packs `(seed, episode_id, request_id, 0)` as four little-endian
unsigned 64-bit integers. BLAKE2b uses an 8-byte digest and personalization
`omg-noise-eval1`; its little-endian integer is masked to 63 bits. A fresh CPU
Torch generator draws 3800 float32 standard normals and transfers them to the
actor device. Sample `mean + exp(log_std) * epsilon`, then zero rows before the
prefix length (0 initially, 25 on replans). Used rows are always sampled.
Without a policy, generation retains the runtime's NumPy seed/index path.

The [numerical fixture](../../tests/deployment/fixtures/README.md) checks
observation bits, full actor means and masked samples, and keyed epsilon slices
and sums against an independent training implementation, with no trained weights.
