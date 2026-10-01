"""Signal fusion — turns raw readings into one AttentionState (Role 2).

Rules (in priority order):
  1. Recent keyboard/mouse input       -> ENGAGED  (looking down at the keyboard is not distraction)
  2. Camera sees a face:
       eyes closed or head turned away -> DISTRACTED
       otherwise                       -> ENGAGED
  3. Camera sees no face:
       long idle (or no input sensor)  -> AWAY
  4. No camera at all (fallback):      idle time alone decides
A majority vote over the last N states removes single-frame flicker.
"""
from __future__ import annotations

from collections import Counter, deque

from fido.config import SensingConfig
from fido.contracts import AttentionState, SensorReading

RECENT_INPUT_S = 4.0


def classify(r: SensorReading, cfg: SensingConfig) -> AttentionState:
    idle = r.seconds_since_input

    if idle is not None and idle < RECENT_INPUT_S:
        return AttentionState.ENGAGED

    if r.face_present is True:
        closed = r.eyes_closed_prob is not None and r.eyes_closed_prob >= cfg.eyes_closed_prob_threshold
        if closed or r.gaze_away:
            return AttentionState.DISTRACTED
        return AttentionState.ENGAGED

    if r.face_present is False:
        if idle is None or idle >= cfg.idle_threshold_s:
            return AttentionState.AWAY
        return AttentionState.DISTRACTED

    # camera unavailable -> keyboard/mouse fallback
    if idle is None:
        return AttentionState.UNKNOWN
    if idle >= cfg.away_idle_s:
        return AttentionState.AWAY
    if idle >= cfg.idle_threshold_s:
        return AttentionState.DISTRACTED
    return AttentionState.ENGAGED


class Fusion:
    def __init__(self, cfg: SensingConfig, window: int = 5):
        self.cfg = cfg
        self.history: deque[AttentionState] = deque(maxlen=window)

    def update(self, reading: SensorReading) -> AttentionState:
        self.history.append(classify(reading, self.cfg))
        return Counter(self.history).most_common(1)[0][0]
