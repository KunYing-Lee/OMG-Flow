"""Pure NumPy sync50 timeline with history10 and committed prefix25."""
import numpy as np

from .schedule import Schedule, motion


class Sync50Timeline:
    """Synchronous generation state shared by offline and REP callers.

    request(context, request_id) returns a world-space suffix.
    Replans stop at the last active preview request, k+14 < play_frames.
    """
    def __init__(self, history, request, *, play_frames=364, period=25):
        motion(history, 10)
        Schedule(period / 50)
        if type(play_frames) is not int or play_frames < 1:
            raise ValueError("play_frames must be a positive integer")
        self.start_history = history.copy()
        self.request = request
        self.play_frames, self.period = play_frames, period
        self.last_tick = max(0, (play_frames - 15) // 10 * 10)

    def validate_wire_schedule(self):
        # A very wide terminal request can outlive generated support. Reject it
        # before arming: transport padding may never enter the committed prefix.
        for switch in range(0, self.play_frames + 11, self.period):
            tick = max(0, min(self.last_tick, (switch + self.period - 26) // 10 * 10))
            if tick + 100 < switch + self.period:
                raise ValueError(f"P={self.period} at S={switch}: terminal generated support "
                                 "does not cover the committed prefix")

    def reset(self):
        self.tape = np.empty((self.play_frames + 110, 36), np.float32)
        self.tape[:100] = motion(self.request(self.start_history.copy(), 0), 100)
        self.next_tick, self.valid_end, self.internal_replans = 10, 100, 0
        self.provisional_from = self.padded_from = None

    def advance(self, switch):
        limit = min(switch + self.period - 1 - 25, self.last_tick)
        self.internal_replans = 0
        while self.next_tick <= limit:
            k = self.next_tick
            assert k + 25 <= self.valid_end
            self.tape[k + 25:k + 100] = motion(self.request(self.tape[k - 10:k + 25].copy(), k // 10), 75)
            self.valid_end = k + 100
            self.next_tick += 10
            self.internal_replans += 1
        # Both completeness AND no unrequested lookahead, including P=50 plan zero.
        assert self.next_tick == max(10, (limit // 10 + 1) * 10)
        assert self.next_tick > switch + self.period - 1 - 25 or self.next_tick > self.last_tick

    def chunk(self, plan_id, switch_frame):
        assert switch_frame == plan_id * self.period
        self.advance(switch_frame)
        end = switch_frame + 100
        self.provisional_from = self.next_tick + 25 if self.next_tick <= self.last_tick else None
        self.padded_from = self.valid_end if self.valid_end < end else None
        q = self.tape[switch_frame:min(end, self.valid_end)].copy()
        if len(q) < 100:
            # Only the reply copy is padded. These rows never enter model context.
            assert self.padded_from >= switch_frame + self.period
            q = np.concatenate((q, np.repeat(self.tape[self.valid_end - 1][None], 100 - len(q), axis=0)))
        assert self.next_tick > switch_frame + self.period - 26 or self.next_tick > self.last_tick
        return q

    def transport_metadata(self):
        return dict(committed_frames=self.period,
                    provisional_from=getattr(self, "provisional_from", None),
                    padded_from=getattr(self, "padded_from", None))


    def bake(self):
        self.validate_wire_schedule()
        self.reset()
        for switch in range(0, self.play_frames + 11, self.period):
            self.advance(switch)
        return self.tape[:self.play_frames + 11].copy()
