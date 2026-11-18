"""A sequential session owns exactly the stream already sent to its client."""
import json
import time
from typing import Protocol
import numpy as np
from .config import StartGate
from .schedule import Schedule, motion
from .wire import ZmqPlanServer


class StreamSource(Protocol):
    start_history: np.ndarray
    play_frames: int
    def reset(self): ...
    def chunk(self, plan_id: int, switch_frame: int) -> np.ndarray: ...
    def metadata(self) -> dict: ...


def _yaw(wxyz):
    w, x, y, z = (np.asarray(wxyz, np.float64)[..., i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class StreamSession:
    def __init__(self, source: StreamSource, *, schedule=Schedule(), after_end="hold", allow_start_mismatch=False, start_gate=StartGate()):
        if after_end not in ("hold", "error"):
            raise ValueError("after_end must be hold or error")
        if hasattr(source, "validate_wire_schedule"):
            source.validate_wire_schedule()
        self.source, self.schedule, self.after_end = source, schedule, after_end
        self.allow_start_mismatch = allow_start_mismatch
        self.start_gate = start_gate
        if source.metadata().get("period_frames", schedule.period) != schedule.period:
            raise ValueError("Source and server periods must match")
        motion(source.start_history, 10)
        self.support_frames = source.play_frames + source.metadata().get("preview_frames", 0)
        self.next_plan = 0
        self.timeline = np.empty((0, 36), np.float32)

    def plan(self, request, record):
        plan_id, switch = self.schedule.validate(request)
        record.update(plan_id=plan_id, switch=switch, history_check="not_checked")
        caption = self.source.metadata().get("caption")
        if caption is not None and request.prompt != caption:
            raise ValueError("Request caption differs from the fixed trained caption")
        start = time.perf_counter()
        if plan_id == 0:
            difference = np.abs(request.qpos_36_history - self.source.start_history)
            record["plan0_history_max_diff"] = float(difference.max())
            record["plan0_joint_max_diff_rad"] = difference[:, 7:].max(axis=0).tolist()
            record["plan0_root_position_max_diff_m"] = difference[:, :3].max(axis=0).tolist()
            # Heading matters: the tracker sees the reference root orientation in the
            # source's world frame, so the robot's yaw estimate must match it.
            yaw = np.angle(np.exp(1j * (_yaw(request.qpos_36_history[:, 3:7]) - _yaw(self.source.start_history[:, 3:7]))))
            record["plan0_yaw_max_diff_rad"] = float(np.abs(yaw).max())
            mismatch = bool(np.any(difference[:, 7:] > self.start_gate.joint_rad)
                            or np.any(difference[:, 2] > self.start_gate.height_m)
                            or np.any(np.abs(yaw) > np.deg2rad(self.start_gate.yaw_deg)))
            record["start_mismatch"] = mismatch
            record["allow_start_mismatch"] = self.allow_start_mismatch
            if mismatch and not self.allow_start_mismatch:
                record["history_check"] = "start_mismatch_refused"
                raise ValueError(f"Fixed start mismatch: limits joint={self.start_gate.joint_rad} rad, "
                                 f"height={self.start_gate.height_m} m, yaw={self.start_gate.yaw_deg} deg "
                                 f"(joint {difference[:, 7:].max():.3f} rad, "
                                 f"height {difference[:, 2].max():.3f} m, yaw {np.rad2deg(np.abs(yaw).max()):.1f} deg)")
            record["history_check"] = "ignored_fixed_start"
            self.source.reset()
            self.next_plan = 0
            self.timeline = np.empty((0, 36), np.float32)
        elif plan_id != self.next_plan:
            raise ValueError(f"Nonsequential plan: expected {self.next_plan}, got {plan_id}")
        else:
            equal = self.timeline[switch - 10:switch].tobytes() == request.qpos_36_history.tobytes()
            record["history_check"] = "bitwise_equal" if equal else "mismatch"
            if not equal:
                raise ValueError("Request history differs from the corrected stream already sent")
        if switch >= self.source.play_frames and self.after_end == "error":
            raise ValueError("Request is past play_frames")
        generate = switch < self.source.play_frames or (
            switch < self.support_frames and self.source.metadata().get("replan_terminal_preview", False))
        if generate:
            q = self.source.chunk(plan_id, switch).copy()
        else:
            q = np.repeat(self.timeline[self.source.play_frames - 1][None], 100, axis=0)
        if self.after_end == "hold" and switch + 100 > self.support_frames:
            last = self.source.play_frames - 1
            held = q[last - switch] if last >= switch else self.timeline[last]
            q[max(0, self.support_frames - switch):] = held
            if not generate and switch < self.support_frames:
                count = self.support_frames - switch
                q[:count] = self.timeline[switch:switch + count]
        record["server_compute_ms"] = (time.perf_counter() - start) * 1000
        record["internal_replans"] = self.source.metadata().get("internal_replans", 0) if generate else 0
        metadata = {**self.source.metadata(), "internal_replans": record["internal_replans"]}
        if "padded_from" in metadata:
            for key in ("provisional_from", "padded_from"):
                if not generate or (metadata[key] is not None and metadata[key] >= self.support_frames):
                    metadata[key] = None  # Terminal holding replaced this part of the reply.
            assert metadata["padded_from"] is None or metadata["padded_from"] >= switch + self.schedule.period
            record.update({key: metadata[key] for key in ("committed_frames", "provisional_from", "padded_from")})
        response = self.schedule.response(request, q, record["server_compute_ms"] / 1000,
            metadata)
        self.timeline = np.concatenate((self.timeline[:switch], q))
        self.next_plan = plan_id + 1
        return response


def serve(source, bind, *, log, schedule=Schedule(), after_end="hold", max_plans=None, allow_start_mismatch=False, start_gate=StartGate()):
    session = StreamSession(source, schedule=schedule, after_end=after_end, allow_start_mismatch=allow_start_mismatch, start_gate=start_gate)
    with open(log, "a", buffering=1) as output, ZmqPlanServer(bind) as server:
        count = 0
        while max_plans is None or count < max_plans:
            record = {}
            try:
                request = server.recv_request()
                response = session.plan(request, record)
                server.send_plan(response)
                record["recv_send_ms"] = (time.perf_counter() - server.received) * 1000
            except Exception as error:
                # Fail closed: log the diagnostic, close REP without replying,
                # and preserve a nonzero process exit for the supervisor.
                record["error"] = str(error)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                raise
            output.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"plan {record['plan_id']} | compute_ms={record['server_compute_ms']:.2f} "
                  f"total_ms={record['recv_send_ms']:.2f} | history={record['history_check']}", flush=True)
            count += 1
