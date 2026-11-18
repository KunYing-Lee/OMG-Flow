"""OMG native realtime protocol v2. The codec needs only NumPy; the server socket needs pyzmq."""
import io
import json
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

MESSAGE_VERSION = 2


def motion(q, frames):
    if q.dtype != np.float32 or q.shape != (frames, 36) or not np.isfinite(q).all():
        raise ValueError(f"Expected finite float32 ({frames},36) motion")
    if np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)) > 1e-4:
        raise ValueError("Root quaternions must be unit length within 1e-4")
    return q


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def encode_message(header, arrays=None):
    payload = io.BytesIO()
    np.savez_compressed(payload, **{str(k): np.asarray(v) for k, v in (arrays or {}).items()})
    return json.dumps(_jsonable(dict(header)), sort_keys=True, separators=(",", ":")).encode(), payload.getvalue()


def decode_message(header_bytes, payload_bytes):
    header = json.loads(header_bytes)
    if not isinstance(header, dict) or type(header.get("version")) is not int or header["version"] != MESSAGE_VERSION:
        raise ValueError("Expected realtime protocol version 2")
    if header.get("kind") not in ("robot_state_request", "motion_plan_chunk"):
        raise ValueError("Expected robot_state_request or motion_plan_chunk")
    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in payload.files}
    return header, arrays


@dataclass
class RobotStateRequest:
    qpos_36_history: np.ndarray
    history_fps: float
    tracker_frame: int
    buffer_remaining_frames: int = 0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    last_plan_id: int | None = None
    prompt: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_message(self):
        return encode_message(dict(version=2, kind="robot_state_request", created_time=time.time(),
            request_id=self.request_id, history_fps=float(self.history_fps), tracker_frame=self.tracker_frame,
            buffer_remaining_frames=self.buffer_remaining_frames, last_plan_id=self.last_plan_id,
            prompt=self.prompt, metadata=self.metadata), {"qpos_36_history": self.qpos_36_history})

    @classmethod
    def from_message(cls, header, arrays):
        if header.get("version") != MESSAGE_VERSION or header.get("kind") != "robot_state_request":
            raise ValueError("Expected robot_state_request")
        return cls(motion(arrays["qpos_36_history"], 10), header["history_fps"], header["tracker_frame"],
                   header.get("buffer_remaining_frames", 0), header["request_id"],
                   header.get("last_plan_id"), header.get("prompt"), header.get("metadata") or {})


@dataclass
class MotionPlanChunk:
    qpos_36: np.ndarray
    fps: float
    request_id: str
    plan_id: int
    request_tracker_frame: int
    planning_latency_seconds: float = 0.
    prompt: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_message(self):
        return encode_message(dict(version=2, kind="motion_plan_chunk", created_time=time.time(),
            request_id=self.request_id, plan_id=self.plan_id, request_tracker_frame=self.request_tracker_frame,
            fps=float(self.fps), planning_latency_seconds=self.planning_latency_seconds,
            prompt=self.prompt, metadata=self.metadata), {"qpos_36": self.qpos_36})

    @classmethod
    def from_message(cls, header, arrays):
        if header.get("version") != MESSAGE_VERSION or header.get("kind") != "motion_plan_chunk":
            raise ValueError("Expected motion_plan_chunk")
        return cls(motion(arrays["qpos_36"], 100), header["fps"], header["request_id"], header["plan_id"],
                   header["request_tracker_frame"], header.get("planning_latency_seconds", 0.),
                   header.get("prompt"), header.get("metadata") or {})


class ZmqPlanServer:
    def __init__(self, bind):
        import zmq

        self.socket = zmq.Context.instance().socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)

    def recv_request(self):
        frames = self.socket.recv_multipart()
        self.received = time.perf_counter()
        if len(frames) != 2:
            raise ValueError("Expected two multipart frames")
        request = RobotStateRequest.from_message(*decode_message(*frames))
        request.metadata["realtime_transport"] = dict(server_recv_wall_time=time.time(),
                                                       server_recv_perf_time=self.received)
        return request

    def send_plan(self, plan):
        plan.metadata.setdefault("realtime_transport", {}).update(
            server_send_wall_time=time.time(), server_send_perf_time=time.perf_counter())
        self.socket.send_multipart(plan.to_message())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.socket.close()
