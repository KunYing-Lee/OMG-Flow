"""Native-50 Hz residual actor for OMG-Lite -- a *reference corrector*, not an action adder.

Verified against the training source `KunYing-Lee/OMG-Posttrain` @ `8d36862`
(2026-10-26, `feat/unified-noise-residual-ppo` merged), not reverse-engineered:

  * shapes: `learning/ppo/train.py:160` -> action_frames = period (25),
    observation_frames = period + 11 (36); `train.py:176` -> residual period = period.
  * observation: `policies/observations/v3.py:82-127` (preview) +
    `policies/trajectory30.py:26-45` (committed queue, `reference_hz=50`):
        610   = 5 x 122 proprio
        1404  = 36 x 39 raw-plan preview   [frame .. frame+35]
        1404  = 36 x 39 corrected tape, zeroed where not locked
        36    = locked flags
        1     = phase (always 0 at 50 Hz)
  * decision timing: `reference/fixed50.py:221-249` + `learning/ppo/trajectory_training.py:72-127`
        decisions at frame 0 (reset) and at switch-11 (the preview deadline);
        the correction is written to [frame+11, frame+36) and the *corrected tape*
        (not the raw plan) is what the tracker consumes.
  * queue source: the corrected tape, NOT the preview. The two differ exactly
    where previous corrections were applied; that is why the checkpoint RMS shows
    a queue-row-10 velocity spike and ~9x larger height variance.
  * "previous residual" fed to proprioception = corrected_tape[j] - raw_plan[j]
    at the previous control frame (`trajectory_training.py:125`).

The actor never modifies the tracker action: it rewrites reference joints that
go into the tracker observation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

FPS = 50.0
PREVIEW = 10
DECISION_LEAD = PREVIEW + 1          # 11: commit starts at decision_frame + 11
PROPRIO_DIM = 122
REFERENCE_DIM = 39
HISTORY_STEPS = 5
FUTURE_FRAMES = 35
ACTION_FRAMES = 25
RAW_WIDTH = HISTORY_STEPS * PROPRIO_DIM + (FUTURE_FRAMES + 1) * REFERENCE_DIM * 2 + (FUTURE_FRAMES + 1) + 1
LOCKED_FRAMES = PREVIEW + 1
PHASE = 0.0

assert RAW_WIDTH == 3455, RAW_WIDTH


class StateNotReady(RuntimeError):
    """Raised instead of guessing when the residual contract cannot be honoured."""


# --------------------------------------------------------------------------- #
# geometry / observation
# --------------------------------------------------------------------------- #
def rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body->world rotation from wxyz quaternion(s): (4,)->(3,3), (N,4)->(N,3,3)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    single = q.ndim == 1
    q = q.reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1)
    if not np.isfinite(norm).all() or np.any(norm < 1e-8):
        raise ValueError("Invalid root quaternion")
    w, x, y, z = (q / norm[:, None]).T
    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - w * z)
    out[:, 0, 2] = 2 * (x * z + w * y)
    out[:, 1, 0] = 2 * (x * y + w * z)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - w * x)
    out[:, 2, 0] = 2 * (x * z - w * y)
    out[:, 2, 1] = 2 * (y * z + w * x)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out[0] if single else out


def proprio_sample(
    robot_state: dict[str, np.ndarray],
    previous_residual: np.ndarray,
    last_action_onnx: np.ndarray,
) -> np.ndarray:
    """122-dim proprioception sample: gravity, gyro, q, dq, previous residual, last action."""
    quat = np.asarray(robot_state["root_quat_wxyz"], np.float32).reshape(4)
    gyro = np.asarray(robot_state["gyro"], np.float32).reshape(3)
    q = np.asarray(robot_state["joint_pos"], np.float32).reshape(29)
    dq = np.asarray(robot_state["joint_vel"], np.float32).reshape(29)
    previous = np.asarray(previous_residual, np.float32).reshape(29)
    action = np.asarray(last_action_onnx, np.float32).reshape(29)
    gravity = (-rotation_matrix(quat)[2, :]).astype(np.float32)
    sample = np.concatenate((gravity, gyro, q, dq, previous, action))
    if sample.shape != (PROPRIO_DIM,) or not np.isfinite(sample).all():
        raise ValueError("Invalid proprioception sample")
    return sample


def reference_block(window_36: np.ndarray, robot_quat_wxyz: np.ndarray) -> np.ndarray:
    """(36,39) reference features: absolute joints, relative rot6d, height, local velocity."""
    window = np.asarray(window_36, dtype=np.float64)
    if window.shape != (FUTURE_FRAMES + 1, 36):
        raise ValueError(f"reference window must be (36,36), got {window.shape}")
    quats = window[:, 3:7]
    if not np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-3):
        raise ValueError("Reference quaternions are not normalized")

    position = window[:, :3]
    velocity = np.empty_like(position)
    velocity[:-1] = (position[1:] - position[:-1]) * FPS
    velocity[-1] = velocity[-2]

    body = rotation_matrix(robot_quat_wxyz)
    yaw = float(np.arctan2(body[1, 0], body[0, 0]))
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    local_velocity = np.stack(
        (
            c * velocity[:, 0] + s * velocity[:, 1],
            -s * velocity[:, 0] + c * velocity[:, 1],
            velocity[:, 2],
        ),
        axis=-1,
    )
    ref_rotation = rotation_matrix(quats)
    relative = np.einsum("ij,tjk->tik", body.T, ref_rotation)
    rot6 = np.concatenate((relative[:, 0, :], relative[:, 1, :]), axis=-1)
    block = np.concatenate(
        (window[:, 7:], rot6, window[:, 2:3], local_velocity), axis=-1
    )
    if block.shape != (FUTURE_FRAMES + 1, REFERENCE_DIM):
        raise ValueError("Invalid reference block shape")
    return block.astype(np.float32)


def build_observation(
    proprio_history: np.ndarray,
    preview: np.ndarray,
    locked: np.ndarray,
    queue: np.ndarray,
    phase: float = PHASE,
) -> np.ndarray:
    """Assemble the 3455-dim actor input.

    `preview` is the raw-plan block (36,39); `queue` is the corrected-tape block
    (36,39), already masked by `locked` (36,).
    """
    history = np.asarray(proprio_history, np.float32)
    preview = np.asarray(preview, np.float32)
    locked = np.asarray(locked, np.float32).reshape(-1)
    queue = np.asarray(queue, np.float32)
    if history.shape != (HISTORY_STEPS, PROPRIO_DIM):
        raise ValueError(f"proprio history must be (5,122), got {history.shape}")
    if preview.shape != (FUTURE_FRAMES + 1, REFERENCE_DIM):
        raise ValueError(f"preview must be (36,39), got {preview.shape}")
    if queue.shape != (FUTURE_FRAMES + 1, REFERENCE_DIM):
        raise ValueError(f"queue must be (36,39), got {queue.shape}")
    if locked.shape != (FUTURE_FRAMES + 1,):
        raise ValueError(f"locked must be (36,), got {locked.shape}")
    obs = np.concatenate(
        (
            history.reshape(-1),
            preview.reshape(-1),
            queue.reshape(-1),
            locked,
            np.array([phase], np.float32),
        )
    )
    if obs.shape != (RAW_WIDTH,) or not np.isfinite(obs).all():
        raise ValueError("Invalid residual observation")
    return obs.astype(np.float32)[None, :]


# --------------------------------------------------------------------------- #
# bounded joint residual
# --------------------------------------------------------------------------- #
def bounded_joint_residual(
    base_qpos: np.ndarray,
    latent: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    residual_scale: float,
    residual_latent_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint-limit-aware residual (exact port of omg_posttrain.policies.residual)."""
    base = np.asarray(base_qpos, np.float32)
    latent = np.asarray(latent, np.float32)
    lower = np.asarray(lower, np.float32).reshape(29)
    upper = np.asarray(upper, np.float32).reshape(29)
    if base.ndim != 3 or base.shape[-1] != 36 or base.shape[-2] != ACTION_FRAMES:
        raise ValueError(f"base must be (1,25,36), got {base.shape}")
    if latent.shape != (1, ACTION_FRAMES, 29):
        raise ValueError(f"latent must be (1,25,29), got {latent.shape}")
    if not (np.isfinite(base).all() and np.isfinite(latent).all()):
        raise ValueError("Non-finite residual input")
    base_joints = base[..., 7:]
    normalized = np.tanh(np.float32(residual_latent_gain) * latent)
    admissible_lower = np.minimum(lower, base_joints)
    admissible_upper = np.maximum(upper, base_joints)
    available = np.where(
        normalized >= 0.0, admissible_upper - base_joints, base_joints - admissible_lower
    )
    delta = np.float32(residual_scale) * normalized * available
    corrected = np.concatenate((base[..., :7], base_joints + delta), axis=-1)
    return corrected.astype(np.float32), delta.astype(np.float32)


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    required = {
        "onnx_sha256", "raw_width", "action_frames", "future_frames",
        "residual_period_frames", "residual_scale", "residual_latent_gain",
        "joint_lower_limits", "joint_upper_limits", "switch_seconds",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"{path}: manifest missing {sorted(missing)}")
    if int(manifest["raw_width"]) != RAW_WIDTH:
        raise ValueError(f"{path}: raw_width {manifest['raw_width']} != {RAW_WIDTH}")
    if int(manifest["action_frames"]) != ACTION_FRAMES:
        raise ValueError(f"{path}: action_frames {manifest['action_frames']} != {ACTION_FRAMES}")
    if int(manifest["future_frames"]) != FUTURE_FRAMES:
        raise ValueError(f"{path}: future_frames {manifest['future_frames']} != {FUTURE_FRAMES}")
    if int(manifest["residual_period_frames"]) != ACTION_FRAMES:
        raise ValueError(f"{path}: residual period must equal the switch period")
    for key in ("joint_lower_limits", "joint_upper_limits"):
        if np.asarray(manifest[key], np.float32).shape != (29,):
            raise ValueError(f"{path}: {key} must have 29 entries")
    if not np.all(np.asarray(manifest["joint_lower_limits"]) < np.asarray(manifest["joint_upper_limits"])):
        raise ValueError(f"{path}: joint limits are inverted")
    return manifest


class ResidualActor:
    """ONNX actor wrapper. `session` is injected so tests need no onnxruntime."""

    def __init__(self, manifest: dict[str, Any], session: Any):
        self.manifest = manifest
        self.session = session
        self.lower = np.asarray(manifest["joint_lower_limits"], np.float32)
        self.upper = np.asarray(manifest["joint_upper_limits"], np.float32)
        self.residual_scale = float(manifest["residual_scale"])
        self.residual_latent_gain = float(manifest["residual_latent_gain"])
        self.period = int(manifest["residual_period_frames"])

    def act(self, raw: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if raw.shape != (1, RAW_WIDTH) or base.shape != (1, ACTION_FRAMES, 36):
            raise ValueError("Invalid actor input shapes")
        latent, corrected, delta = self.session.run(None, {"raw": raw, "base": base})
        latent = np.asarray(latent, np.float32)
        corrected = np.asarray(corrected, np.float32)
        delta = np.asarray(delta, np.float32)
        if latent.shape != (1, ACTION_FRAMES * 29) or corrected.shape != (1, ACTION_FRAMES, 36) \
                or delta.shape != (1, ACTION_FRAMES, 29):
            raise ValueError("Actor returned unexpected shapes")
        if not (np.isfinite(latent).all() and np.isfinite(corrected).all() and np.isfinite(delta).all()):
            raise ValueError("Actor returned non-finite values")
        return latent, corrected, delta


def open_onnx_session(entry: "ResidualEntry") -> Any:
    import onnxruntime as ort

    onnx_path = entry.directory / entry.manifest.get("onnx_file", "ppo_actor.onnx")
    got = sha256_file(onnx_path)
    if got != entry.manifest["onnx_sha256"]:
        raise ValueError(f"{onnx_path}: sha256 mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CUDAExecutionProvider"]
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("Residual actor requires the CUDA execution provider")
    return session


# --------------------------------------------------------------------------- #
# registry / switch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResidualEntry:
    name: str
    directory: Path
    manifest: dict[str, Any]

    @property
    def prompt(self) -> str:
        return str(self.manifest.get("prompt", ""))


class ResidualRegistry:
    """Directory of named residual models; selection is explicit or prompt-matched."""

    def __init__(self, entries: dict[str, ResidualEntry]):
        self.entries = entries

    @classmethod
    def discover(cls, root: str | Path) -> "ResidualRegistry":
        root = Path(root)
        entries: dict[str, ResidualEntry] = {}
        if not root.is_dir():
            return cls(entries)
        for manifest_path in sorted(root.glob("*/actor-export.json")):
            try:
                manifest = load_manifest(manifest_path)
            except Exception as error:  # noqa: BLE001 - a bad entry must not block others
                print(f"[residual] skip {manifest_path}: {error}")
                continue
            names = {manifest_path.parent.name, str(manifest.get("name", ""))}
            for name in names:
                if name:
                    entries[name] = ResidualEntry(name, manifest_path.parent, manifest)
        return cls(entries)

    def names(self) -> list[str]:
        return sorted(self.entries)

    def resolve(
        self,
        mode: str,
        prompt: str | None = None,
        *,
        warn: Callable[[str], None] | None = None,
    ) -> ResidualEntry | None:
        """mode: 'off' | 'auto' | <entry name>. Returns None when disabled."""
        mode = (mode or "off").strip()
        if mode in ("", "off", "none"):
            return None
        if mode == "auto":
            matches = [e for e in self.entries.values() if e.prompt == prompt]
            if len(matches) == 1:
                return matches[0]
            if warn is not None:
                warn(f"residual=auto: {len(matches)} entries match prompt {prompt!r}; staying off")
            return None
        entry = self.entries.get(mode)
        if entry is None:
            raise KeyError(f"unknown residual {mode!r}; available: {self.names()}")
        return entry


# --------------------------------------------------------------------------- #
# per-episode state machine
# --------------------------------------------------------------------------- #
@dataclass
class Correction:
    frame: int
    start: int                 # first frame of the corrected window (absolute)
    joints_g1: np.ndarray      # (25,29) corrected joints to write into the tape
    latent: np.ndarray         # (725,)
    delta: np.ndarray          # (25,29)


class ResidualCorrector:
    """Decision + proprio history for the native50 fixed-switch protocol.

    The caller owns two reference buffers:
      * `raw_reference`   -- the planner timeline (`PlanTimeline.qpos` in Lite);
      * `corrected_tape`  -- the reference actually sent to the tracker, i.e. the
                             tape that decisions overwrite.
    Call `step()` every control frame (proprio history is updated each frame);
    it returns None except at the reset frame and at `switch - 11`.
    """

    def __init__(self, actor: ResidualActor, *, period: int | None = None):
        self.actor = actor
        self.period = int(period or actor.period)
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros((HISTORY_STEPS, PROPRIO_DIM), np.float32)
        self._history_valid = False
        self._last_frame = -1
        self.decisions = 0

    def _observe(self, sample: np.ndarray, frame: int) -> np.ndarray:
        if frame != self._last_frame:
            self._history = np.concatenate((self._history[1:], sample[None]), axis=0)
            self._last_frame = frame
        if not self._history_valid:
            self._history = np.tile(sample, (HISTORY_STEPS, 1))
            self._history_valid = True
        return self._history

    def _previous_residual(
        self, frame: int, raw_reference: np.ndarray, corrected_tape: np.ndarray
    ) -> np.ndarray:
        if frame <= 0:
            return np.zeros(29, np.float32)
        return (corrected_tape[frame - 1, 7:36] - raw_reference[frame - 1, 7:36]).astype(np.float32)

    def decision_due(self, frame: int) -> bool:
        """Reset frame 0, then every `switch - 11` (preview deadline)."""
        return frame == 0 or frame % self.period == self.period - DECISION_LEAD

    def step(
        self,
        *,
        frame: int,
        raw_reference: np.ndarray,
        corrected_tape: np.ndarray,
        robot_state: dict[str, np.ndarray],
        last_action_onnx: np.ndarray,
    ) -> Correction | None:
        frame = int(frame)
        raw_reference = np.asarray(raw_reference, np.float32)
        corrected_tape = np.asarray(corrected_tape, np.float32)
        for name, array in (("raw_reference", raw_reference), ("corrected_tape", corrected_tape)):
            if array.ndim != 2 or array.shape[1] != 36:
                raise ValueError(f"{name} must be (T,36)")
            if len(array) < frame + FUTURE_FRAMES + 1:
                raise StateNotReady(f"{name} doesn't cover frame {frame}+36")

        previous = self._previous_residual(frame, raw_reference, corrected_tape)
        sample = proprio_sample(robot_state, previous, last_action_onnx)
        history = self._observe(sample, frame)

        if not self.decision_due(frame):
            return None

        reset = frame == 0
        start = 0 if reset else frame + DECISION_LEAD
        if start + ACTION_FRAMES > len(raw_reference):
            raise StateNotReady("raw plan doesn't cover the correction window")
        locked = (
            np.zeros(FUTURE_FRAMES + 1, np.float32)
            if reset
            else (np.arange(FUTURE_FRAMES + 1) < LOCKED_FRAMES).astype(np.float32)
        )
        quat = robot_state["root_quat_wxyz"]
        preview = reference_block(raw_reference[frame: frame + FUTURE_FRAMES + 1], quat)
        if reset:
            # nothing is locked on the reset decision; skip one 36-frame block build
            queue = np.zeros((FUTURE_FRAMES + 1, REFERENCE_DIM), np.float32)
        else:
            queue = reference_block(
                corrected_tape[frame: frame + FUTURE_FRAMES + 1], quat
            ) * locked[:, None]
        raw = build_observation(history, preview, locked, queue)
        base = raw_reference[start: start + ACTION_FRAMES]
        latent, corrected, delta = self.actor.act(raw, base[None])
        self.decisions += 1
        return Correction(
            frame=frame,
            start=start,
            joints_g1=corrected[0, :, 7:].copy(),
            latent=latent[0].copy(),
            delta=delta[0].copy(),
        )
