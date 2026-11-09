"""G1 state/target boundary, HoloMotion v1.4.1 tracker, and MuJoCo robot I/O."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omg.robots.g1.constants import G1_JOINT_NAMES
from omg.tracking.holomotion.reference import build_holomotion_obs, precompute_reference_features
from omg.tracking.holomotion.runtime import (
    HoloMotionTrackerSession, build_g1_state_handles, build_holomotion_handles,
    build_onnx_session, extract_g1_qpos, load_holomotion_metadata, set_g1_state,
)


@dataclass
class RobotState:
    timestamp_ns: int  # Same monotonic clock as the control schedule.
    qpos: np.ndarray  # World xyz, wxyz, canonical G1 29 joints.
    joint_velocity: np.ndarray  # Canonical G1 order, rad/s.
    gyro: np.ndarray  # Pelvis/body frame, rad/s.


class RobotIO(ABC):
    @abstractmethod
    def read_state(self) -> RobotState:
        """Return one coherent, timestamped state sample."""

    @abstractmethod
    def write_joint_targets(self, targets: np.ndarray):
        """Canonical G1 joint radians; native PD uses the tracker metadata gains."""


class RobotSDK(RobotIO):
    def read_state(self):
        # TODO: bind SDK sensor reads, synchronize odometry/IMU/joints, convert
        # native order and frames, validate freshness against the monotonic clock.
        raise NotImplementedError("Bind the actual robot SDK state reader")

    def write_joint_targets(self, targets):
        # TODO: bind SDK joint targets and metadata kp/kd; integrate native enable,
        # watchdog and emergency stop. Never infer or invent SDK motor commands.
        raise NotImplementedError("Bind the actual robot SDK target writer")


class HoloTracker:
    def __init__(self, model_path):
        model_path = Path(model_path).resolve()
        if model_path.name != "model_16200_dynamic_batch.onnx":
            raise ValueError("Use HoloMotion v1.4.1 model_16200_dynamic_batch.onnx")
        if not model_path.with_name("4de98d3c-9d3c-11f1-aff4-06f301228322").is_file():
            raise FileNotFoundError("HoloMotion ONNX external-data file must be beside the graph")
        self.session = build_onnx_session(model_path, ["CUDAExecutionProvider"])
        self.metadata = load_holomotion_metadata(self.session)
        if set(self.metadata.joint_names) != set(G1_JOINT_NAMES) or len(self.metadata.joint_names) != 29:
            raise ValueError("Tracker must control exactly the 29 canonical G1 joints")
        self.order = np.array([G1_JOINT_NAMES.index(name) for name in self.metadata.joint_names])
        self.handles = SimpleNamespace(onnx_to_g1=self.order, default_joint_pos=self.metadata.default_joint_pos)
        self.tracker = HoloMotionTrackerSession(self.session, self.metadata)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.robot_history = {}

    def reference_features(self, qpos):
        return precompute_reference_features(qpos, fps=50, onnx_to_g1=self.order)

    def targets(self, state, features, cursor):
        obs = build_holomotion_obs(
            features, cursor, self.metadata.n_fut_frames, None, None, self.handles, self.last_action,
            context_length=self.metadata.context_length, obs_schema_version=self.metadata.obs_schema_version,
            robot_history=self.robot_history, robot_state={
                "root_quat_wxyz": state.qpos[3:7], "joint_pos": state.qpos[7:],
                "joint_vel": state.joint_velocity, "gyro": state.gyro,
            },
        )
        self.last_action = self.tracker.run(obs)
        if not np.isfinite(self.last_action).all():
            raise RuntimeError("Nonfinite tracker action")
        targets = np.empty(29, dtype=np.float32)
        targets[self.order] = self.metadata.default_joint_pos + self.metadata.action_scale * self.last_action
        return targets


class MujocoRobot(RobotIO):
    def __init__(self, scene, metadata, epoch_ns, standing_height):
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.g1 = build_g1_state_handles(self.model)
        self.handles = build_holomotion_handles(self.model, metadata)
        self.epoch_ns = epoch_ns
        self.standing = np.zeros(36, dtype=np.float32)
        self.standing[2:4] = standing_height, 1
        self.standing[7 + self.handles.onnx_to_g1] = metadata.default_joint_pos
        set_g1_state(self.model, self.data, self.g1, self.standing)

    def read_state(self):
        return RobotState(
            self.epoch_ns + round(self.data.time * 1e9), extract_g1_qpos(self.data, self.g1),
            self.data.qvel[self.g1["joint_dof_adr"]].astype(np.float32),
            self.data.sensordata[self.g1["pelvis_gyro_adr"]:self.g1["pelvis_gyro_adr"] + 3].astype(np.float32),
        )

    def write_joint_targets(self, targets):
        import mujoco

        for _ in range(10):
            self.data.ctrl[self.handles.actuator_ids] = (
                self.handles.kp * (targets[self.handles.onnx_to_g1] - self.data.qpos[self.handles.joint_qpos_adr])
                - self.handles.kd * self.data.qvel[self.handles.joint_dof_adr]
            )
            mujoco.mj_step(self.model, self.data)
