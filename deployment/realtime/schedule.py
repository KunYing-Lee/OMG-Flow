"""Server boundary for the native fixed-switch schedule."""
from dataclasses import dataclass
import math
from .wire import MotionPlanChunk, motion


@dataclass(frozen=True)
class Schedule:
    switch_seconds: float = 0.5

    def __post_init__(self):
        if type(self.switch_seconds) not in (int, float):
            raise ValueError("switch_seconds must be a number")
        p = self.switch_seconds * 50
        if not math.isfinite(p) or not 25 <= p <= 100 or abs(p - round(p)) > 1e-9:
            raise ValueError("switch_seconds must be 0.5..2.0 in 20ms increments")

    @property
    def period(self):
        return round(self.switch_seconds * 50)

    def switch_frame(self, plan_id):
        return plan_id * self.period

    def request_frame(self, plan_id):
        return 0 if plan_id == 0 else self.switch_frame(plan_id) - 25

    def deadline_frame(self, plan_id):
        """Plan zero initializes before playback and has no playback deadline."""
        return None if plan_id == 0 else self.switch_frame(plan_id) - 11

    def validate(self, request):
        if request.history_fps != 50 or not request.prompt:
            raise ValueError("Native 50Hz history and text required")
        motion(request.qpos_36_history, 10)
        if request.metadata.get("period_frames") != self.period:
            raise ValueError("Planner and client schedules must match")
        plan_id, switch = request.metadata["plan_id"], request.metadata["switch_frame"]
        if type(plan_id) is not int or type(switch) is not int or plan_id < 0:
            raise ValueError("Plan and switch must be nonnegative integers")
        if switch != self.switch_frame(plan_id) or request.tracker_frame != self.request_frame(plan_id):
            raise ValueError("Invalid future-history schedule")
        return plan_id, switch

    def response(self, request, qpos, latency, metadata):
        return MotionPlanChunk(motion(qpos, 100), 50., request.request_id,
            request.metadata["plan_id"], request.tracker_frame, latency, request.prompt,
            {**metadata, "switch_frame": request.metadata["switch_frame"], "period_frames": self.period,
             "realtime_transport": request.metadata.get("realtime_transport", {}).copy()})
