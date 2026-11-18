"""CPU protocol, timeline and session contract tests."""
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import uuid

import numpy as np
import pytest

from deployment.realtime.cli import request
from deployment.realtime.config import StartGate
from deployment.realtime.schedule import Schedule
from deployment.realtime.server import StreamSession, serve
from deployment.realtime.timeline import Sync50Timeline
from deployment.realtime.wire import RobotStateRequest, MotionPlanChunk, decode_message, encode_message

class DummySource:
    play_frames = 140
    caption = "dummy"
    start_history = np.zeros((10, 36), np.float32)
    start_history[:, 3] = 1

    def reset(self):
        self.next = 0

    def chunk(self, plan_id, switch_frame):
        assert plan_id == self.next
        self.next += 1
        q = np.zeros((100, 36), np.float32)
        q[:, 3] = 1
        q[:, 0] = np.minimum(np.arange(switch_frame, switch_frame + 100), self.play_frames - 1).astype(np.float32) / 1000
        return q

    def metadata(self):
        return dict(source="dummy")


def test_golden():
    fixture = Path(__file__).parent / "fixtures"
    frames = ((fixture / "request.json").read_bytes(), (fixture / "request.npz").read_bytes())
    h, a = decode_message(*frames)
    assert encode_message(h, a) == frames
    assert h["metadata"]["period_frames"] == 25
    assert a["qpos_36_history"].shape == (10, 36)


@pytest.mark.parametrize("seconds", [0, .49, 2.01, float("nan"), .501])
def test_bad_schedule(seconds):
    with pytest.raises(ValueError):
        Schedule(seconds)


@pytest.mark.parametrize("period", [25, 50, 100])
def test_schedule(period):
    s = Schedule(period / 50)
    for k in (0, 1, 3):
        r = request(DummySource.start_history, k, s, "dummy")
        assert s.validate(r) == (k, k * period)
        r.tracker_frame += 1
        with pytest.raises(ValueError):
            s.validate(r)


def test_response_boundary():
    s = Schedule()
    source = DummySource()
    source.reset()
    r = request(source.start_history, 0, s, "dummy")
    q = source.chunk(0, 0)
    for bad in (q.astype(np.float64), q[:99], q * np.nan, q * 2):
        with pytest.raises(ValueError):
            s.response(r, bad, 0., {})
    r.metadata["period_frames"] = 50
    with pytest.raises(ValueError):
        s.validate(r)


def test_after_end_error():
    session = StreamSession(DummySource(), after_end="error")
    history = DummySource.start_history
    for k in range(6):
        result = session.plan(request(history, k, Schedule(), "dummy"), {})
        history = result.qpos_36[15:25]
    with pytest.raises(ValueError, match="past play_frames"):
        session.plan(request(history, 6, Schedule(), "dummy"), {})


def test_hold_clips_unplayed_suffix_and_preserves_explicit_preview():
    class FiniteSource(DummySource):
        play_frames = 20
        def chunk(self, plan_id, switch_frame):
            q = super().chunk(plan_id, switch_frame)
            q[:, 0] = np.arange(100, dtype=np.float32)
            return q
        def metadata(self):
            return {"preview_frames": 11}
    source = FiniteSource()
    session = StreamSession(source)
    response = session.plan(request(source.start_history, 0, Schedule(), "dummy"), {})
    np.testing.assert_array_equal(response.qpos_36[:31, 0], np.arange(31))
    assert np.all(response.qpos_36[31:, 0] == 19)
    response = session.plan(request(response.qpos_36[15:25], 1, Schedule(), "dummy"), {})
    np.testing.assert_array_equal(response.qpos_36[:6, 0], np.arange(25, 31))
    assert np.all(response.qpos_36[6:, 0] == 19)


def test_history_comparison_is_bitwise_including_signed_zero():
    session = StreamSession(DummySource())
    first = session.plan(request(DummySource.start_history, 0, Schedule(), "dummy"), {})
    history = first.qpos_36[15:25].copy()
    history[0, 1] = -0.
    with pytest.raises(ValueError, match="history differs"):
        session.plan(request(history, 1, Schedule(), "dummy"), {})


def history():
    q = np.zeros((10, 36), np.float32)
    q[:, 3] = 1
    return q


def generate(context, request_id):
    q = np.zeros((100 if request_id == 0 else 75, 36), np.float32)
    q[:, 3] = 1
    # Sensitive to the full history AND prefix, while remaining bounded.
    q[:, 7:] = np.random.default_rng(request_id).uniform(-.1, .1, (len(q), 29)).astype(np.float32)
    q[:, 7:] += np.mean(context[:, 7:], axis=0)
    return q


@pytest.mark.parametrize("period", [25, 50])
def test_finality_and_no_extra_replans(period):
    frames = 364
    oracle = np.empty((440, 36), np.float32)
    oracle[:100] = generate(history(), 0)
    for k in range(10, 350, 10):
        oracle[k+25:k+100] = generate(oracle[k-10:k+25].copy(), k//10)
    timeline = Sync50Timeline(history(), generate, period=period)
    timeline.reset()
    sent = np.empty((0, 36), np.float32)
    for switch in range(0, frames, period):
        timeline.advance(switch)
        assert timeline.tape[:len(sent)].tobytes() == sent.tobytes()
        stop = min(switch + period, frames)
        assert timeline.tape[switch:stop].tobytes() == oracle[switch:stop].tobytes()
        sent = timeline.tape[:stop].copy()
        limit = min(switch + period - 26, 340)
        assert timeline.next_tick == max(10, (limit // 10 + 1) * 10)
    assert sent.tobytes() == oracle[:frames].tobytes()


class FixedSource:
    start_history = history()
    play_frames = 100
    def reset(self):
        self.was_reset = True
    def chunk(self, plan_id, switch_frame):
        return np.repeat(self.start_history[-1:], 100, axis=0)
    def metadata(self):
        return {}


@pytest.mark.parametrize("column,delta", [(7, .301), (35, -.301), (2, .101)])
def test_start_mismatch_refused_before_generation(column, delta):
    source = FixedSource()
    measured = history()
    measured[4, column] += delta
    record = {}
    with pytest.raises(ValueError, match="Fixed start mismatch"):
        StreamSession(source).plan(request(measured, 0, Schedule(), "dummy"), record)
    assert not hasattr(source, "was_reset")
    assert record["history_check"] == "start_mismatch_refused"
    assert len(record["plan0_joint_max_diff_rad"]) == 29
    assert len(record["plan0_root_position_max_diff_m"]) == 3


def test_start_yaw_mismatch_refused():
    measured = history()
    half = np.deg2rad(12.0) / 2
    measured[:, 3:7] = [np.cos(half), 0, 0, np.sin(half)]
    record = {}
    with pytest.raises(ValueError, match="Fixed start mismatch"):
        StreamSession(FixedSource()).plan(request(measured, 0, Schedule(), "dummy"), record)
    assert record["plan0_yaw_max_diff_rad"] == pytest.approx(np.deg2rad(12.0), abs=1e-5)
    small = history()
    half = np.deg2rad(8.0) / 2
    small[:, 3:7] = [np.cos(half), 0, 0, np.sin(half)]
    record = {}
    StreamSession(FixedSource()).plan(request(small, 0, Schedule(), "dummy"), record)
    assert record["history_check"] == "ignored_fixed_start"


def test_start_override_and_translation_diagnostic():
    measured = history()
    measured[:, 7] = 1
    measured[:, 0] = 99
    record = {}
    result = StreamSession(FixedSource(), allow_start_mismatch=True).plan(request(measured, 0, Schedule(), "dummy"), record)
    assert record["start_mismatch"] and record["allow_start_mismatch"]
    assert record["plan0_root_position_max_diff_m"] == [99, 0, 0]
    assert np.all(result.qpos_36[:, 7] == 0)


def test_start_threshold_inclusive():
    measured = history()
    measured[:, 7] = .3
    measured[:, 2] = .1
    StreamSession(FixedSource()).plan(request(measured, 0, Schedule(), "dummy"), {})


def test_p25_padding_is_only_an_unused_reply_copy():
    timeline = Sync50Timeline(history(), generate, period=25)
    timeline.reset()
    timeline.chunk(0, 0)
    q = timeline.chunk(1, 25)
    assert timeline.next_tick == 30
    assert timeline.valid_end == 120
    assert timeline.transport_metadata() == dict(committed_frames=25, provisional_from=55, padded_from=120)
    assert q[:95].tobytes() == timeline.tape[25:120].tobytes()
    assert q[95:].tobytes() == np.repeat(timeline.tape[119:120], 5, axis=0).tobytes()
    q[:] = 0  # Neither generated rows nor padding can mutate the model's tape.
    assert np.any(timeline.tape[25:120])


@pytest.mark.parametrize("period", [25, 50, 100])
def test_full_wire_session_where_horizon_is_generated(period):
    class Source(Sync50Timeline):
        def metadata(self):
            return dict(**self.transport_metadata(), period_frames=self.period, preview_frames=11)
    source = Source(history(), generate, period=period)
    session = StreamSession(source, schedule=Schedule(period/50))
    oracle = np.empty((440,36), np.float32)
    oracle[:100] = generate(history(), 0)
    for k in range(10,350,10):
        oracle[k+25:k+100] = generate(oracle[k-10:k+25].copy(), k//10)
    measured = history()
    played = []
    for plan in range(364//period+1):
        response = session.plan(request(measured, plan, session.schedule, "fake"), {})
        played.append(response.qpos_36[:period].copy())
        measured = response.qpos_36[period-10:period].copy()
    assert np.concatenate(played)[:364].tobytes() == oracle[:364].tobytes()


def test_terminal_committed_prefix_preflight():
    class Source(Sync50Timeline):
        def metadata(self):
            return {}
    source = Source(history(), generate, play_frames=100, period=100)
    with pytest.raises(ValueError, match="terminal generated support"):
        StreamSession(source, schedule=Schedule(2.))
    assert not hasattr(source, "tape")


def test_terminal_preview_request_at_exact_play_boundary():
    class Source(Sync50Timeline):
        def metadata(self):
            return dict(period_frames=self.period, preview_frames=11, replan_terminal_preview=True,
                        internal_replans=getattr(self, "internal_replans", 0))
    source = Source(history(), generate, play_frames=100, period=50)
    session = StreamSession(source, schedule=Schedule(1.))
    measured = history()
    for plan in range(3):
        record = {}
        response = session.plan(request(measured, plan, session.schedule, "fake"), record)
        measured = response.qpos_36[40:50].copy()
    assert source.next_tick == 90
    assert record["internal_replans"] == 1  # k80 finalizes preview frames105..110
    oracle = np.empty((180,36),np.float32)
    oracle[:100] = generate(history(),0)
    for k in range(10,90,10):
        oracle[k+25:k+100] = generate(oracle[k-10:k+25].copy(),k//10)
    assert session.timeline[:111].tobytes() == oracle[:111].tobytes()
    assert np.all(session.timeline[111:] == oracle[99])


@pytest.mark.parametrize("period", [25, 50, 100])
def test_schedule_deadlines(period):
    schedule = Schedule(period / 50)
    assert schedule.request_frame(0) == 0
    assert schedule.deadline_frame(0) is None
    for plan in (1, 2, 7):
        assert schedule.switch_frame(plan) == plan * period
        assert schedule.request_frame(plan) == plan * period - 25
        assert schedule.deadline_frame(plan) == plan * period - 11
        assert schedule.deadline_frame(plan) - schedule.request_frame(plan) == 14


def test_wire_roundtrip():
    original = request(history(), 0, Schedule(), "dummy")
    decoded = RobotStateRequest.from_message(*decode_message(*original.to_message()))
    assert decoded.qpos_36_history.tobytes() == original.qpos_36_history.tobytes()
    assert decoded.metadata == original.metadata
    assert decoded.request_id == original.request_id
    response = StreamSession(FixedSource()).plan(decoded, {})
    copied = MotionPlanChunk.from_message(*decode_message(*response.to_message()))
    assert copied.qpos_36.tobytes() == response.qpos_36.tobytes()
    assert copied.metadata == response.metadata
    assert copied.plan_id == 0 and copied.request_id == decoded.request_id


@pytest.mark.parametrize("kind,frames,key,cls", [
    ("robot_state_request", 10, "qpos_36_history", RobotStateRequest),
    ("motion_plan_chunk", 100, "qpos_36", MotionPlanChunk),
])
@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "inf", "quaternion", "pickle"])
def test_wire_rejects_bad_motion(kind, frames, key, cls, bad):
    r = request(history(), 0, Schedule(), "dummy")
    message = r if frames == 10 else StreamSession(FixedSource()).plan(r, {})
    header, arrays = decode_message(*message.to_message())
    q = arrays[key]
    if bad == "shape":
        q = q[:-1]
    elif bad == "dtype":
        q = q.astype(np.float64)
    elif bad in ("nan", "inf"):
        q[0, 0] = float(bad)
    elif bad == "quaternion":
        q[0, 3] = 2
    else:
        q = q.astype(object)
    with pytest.raises(ValueError):
        cls.from_message(*decode_message(*encode_message(header, {key: q})))


@pytest.mark.parametrize("change", [{"version": 1}, {"version": 2.0}, {"kind": "unknown"}])
def test_wire_rejects_header(change):
    header, arrays = decode_message(*request(history(), 0, Schedule(), "dummy").to_message())
    with pytest.raises(ValueError):
        decode_message(*encode_message({**header, **change}, arrays))


def test_wrong_message_class():
    frames = request(history(), 0, Schedule(), "dummy").to_message()
    with pytest.raises(ValueError, match="motion_plan_chunk"):
        MotionPlanChunk.from_message(*decode_message(*frames))


@pytest.mark.parametrize("field,column,low,high", [
    ("joint_rad", 7, 0.12, 0.18), ("height_m", 2, 0.02, 0.08), ("yaw_deg", None, 2., 8.),
])
def test_custom_start_gate(field, column, low, high):
    measured = history()
    if column is None:
        half = np.deg2rad(5.) / 2
        measured[:, 3:7] = [np.cos(half), 0, 0, np.sin(half)]
    else:
        measured[:, column] = (low + high) / 2
    with pytest.raises(ValueError, match="Fixed start mismatch"):
        StreamSession(FixedSource(), start_gate=StartGate(**{field: low})).plan(
            request(measured, 0, Schedule(), "dummy"), {})
    StreamSession(FixedSource(), start_gate=StartGate(**{field: high})).plan(
        request(measured, 0, Schedule(), "dummy"), {})


@pytest.mark.parametrize("column,delta", [(7, .29), (35, -.29), (2, .09)])
def test_start_gate_accepts_small_difference(column, delta):
    measured = history()
    measured[4, column] += delta
    result = StreamSession(FixedSource()).plan(request(measured, 0, Schedule(), "dummy"), {})
    assert result.qpos_36.tobytes() == np.repeat(history()[-1:], 100, axis=0).tobytes()


def test_caption_sequence_and_session_configuration():
    class CaptionSource(FixedSource):
        def metadata(self):
            return dict(caption="fixed", period_frames=25)
    source = CaptionSource()
    session = StreamSession(source)
    with pytest.raises(ValueError, match="caption"):
        session.plan(request(history(), 0, Schedule(), "wrong"), {})
    with pytest.raises(ValueError, match="Nonsequential"):
        session.plan(request(history(), 2, Schedule(), "fixed"), {})
    with pytest.raises(ValueError, match="after_end"):
        StreamSession(source, after_end="other")
    with pytest.raises(ValueError, match="periods"):
        StreamSession(source, schedule=Schedule(1.))


@pytest.mark.parametrize("period,counts", [(25, [0, 2, 2, 3]), (50, [2, 5, 5, 5])])
def test_internal_replans_and_bake_stream_parity(period, counts):
    class Source(Sync50Timeline):
        def metadata(self):
            return dict(**self.transport_metadata(), period_frames=self.period, preview_frames=11,
                        replan_terminal_preview=True, internal_replans=self.internal_replans)
    source = Source(history(), generate, period=period)
    baked = source.bake()
    session = StreamSession(source, schedule=Schedule(period / 50))
    measured = history()
    for plan in range((source.play_frames + 10) // period + 1):
        record = {}
        result = session.plan(request(measured, plan, session.schedule, "dummy"), record)
        measured = result.qpos_36[period-10:period].copy()
        if plan < len(counts):
            assert record["internal_replans"] == counts[plan]
        for key in ("provisional_from", "padded_from"):
            assert result.metadata[key] is None or result.metadata[key] >= (plan + 1) * period
    assert baked.tobytes() == session.timeline[:source.play_frames + 11].tobytes()
    assert source.provisional_from is None
    assert source.bake().tobytes() == baked.tobytes()


def test_zmq_roundtrip_and_reset(tmp_path):
    zmq = pytest.importorskip("zmq")
    endpoint = "inproc://" + uuid.uuid4().hex
    log = tmp_path / "server.jsonl"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(serve, DummySource(), endpoint, log=log, max_plans=9)
        with zmq.Context.instance().socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(endpoint)
            measured = history()
            for plan in [*range(8), 0]:
                socket.send_multipart(request(measured, plan, Schedule(), "dummy").to_message())
                assert socket.poll(5000)
                result = MotionPlanChunk.from_message(*decode_message(*socket.recv_multipart()))
                measured = result.qpos_36[15:25].copy()
                if plan >= 6:
                    assert np.all(result.qpos_36[:, 0] == np.float32(.139))
                assert "server_send_wall_time" in result.metadata["realtime_transport"]
        future.result(timeout=5)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 9
    assert records[0]["history_check"] == records[-1]["history_check"] == "ignored_fixed_start"
    assert all(row["history_check"] == "bitwise_equal" for row in records[1:8])


def test_zmq_start_refusal_logs_and_exits(tmp_path):
    zmq = pytest.importorskip("zmq")
    endpoint = "inproc://" + uuid.uuid4().hex
    log = tmp_path / "server.jsonl"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(serve, FixedSource(), endpoint, log=log, max_plans=1,
                             start_gate=StartGate(joint_rad=.01))
        with zmq.Context.instance().socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(endpoint)
            measured = history()
            measured[:, 7] = .02
            socket.send_multipart(request(measured, 0, Schedule(), "dummy").to_message())
            with pytest.raises(ValueError, match="Fixed start mismatch"):
                future.result(timeout=5)
            assert not socket.poll(10)
    record = json.loads(log.read_text())
    assert record["history_check"] == "start_mismatch_refused"
    assert "Fixed start mismatch" in record["error"]


@pytest.mark.parametrize("args", [[], ["serve"], ["bake"], ["client"]])
def test_help_without_model_dependencies(args):
    # Block imports explicitly: this exercises a NumPy-only environment even
    # when the test runner itself has Torch installed.
    code = '''
import sys
from importlib.abc import MetaPathFinder
class BlockModels(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'omg', 'zmq'}:
            raise ImportError('model or socket dependency imported by help')
sys.meta_path.insert(0, BlockModels())
import deployment.realtime.wire, deployment.realtime.schedule
import deployment.realtime.server, deployment.realtime.timeline
import deployment.realtime.config, deployment.realtime.source
from deployment.realtime.cli import main
sys.argv = ['realtime', *ARGS, '--help']
main()
'''
    result = subprocess.run([sys.executable, "-c", "ARGS=" + repr(args) + "\n" + code],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
