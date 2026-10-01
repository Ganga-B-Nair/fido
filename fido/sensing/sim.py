"""Simulated user — lets the whole team develop and demo without the camera or Pi.

The simulated user has hidden preferences: each nudge works with a different probability,
and that probability depends on context (a sound works when you've walked away; a head-tilt
only works if you're at the desk to see it). The bandit doesn't know these numbers — watching
it discover them is the learning demo.
"""
from __future__ import annotations

import random

from fido.contracts import AttentionState, Nudge, SensorReading

# P(re-engage | nudge, context). Edit these to show the bandit adapting to a different "user".
DEFAULT_PREFERENCES = {
    "distracted": {Nudge.CHIRP: 0.30, Nudge.ALARM: 0.40, Nudge.HEAD_TILT: 0.75,
                   Nudge.EYES_FLASH: 0.45, Nudge.FACE_SAD: 0.60},
    "away":       {Nudge.CHIRP: 0.45, Nudge.ALARM: 0.80, Nudge.HEAD_TILT: 0.10,
                   Nudge.EYES_FLASH: 0.10, Nudge.FACE_SAD: 0.05},
}


class SimulatedUser:
    def __init__(self, preferences=None, seed: int | None = None,
                 drift_rate: float = 0.04, away_share: float = 0.35, self_return_rate: float = 0.01):
        self.prefs = preferences or DEFAULT_PREFERENCES
        self.rng = random.Random(seed)
        self.drift_rate = drift_rate              # P(engaged -> drift) per second
        self.away_share = away_share              # share of drifts that are walk-aways
        self.self_return_rate = self_return_rate  # P(returns on their own) per second
        self.state = AttentionState.ENGAGED
        self.last_input = 0.0
        self._pending_return: float | None = None

    def receive_nudge(self, nudge: Nudge, now: float):
        ctx = self.state.value
        p = self.prefs.get(ctx, {}).get(nudge, 0.0)
        if self.state != AttentionState.ENGAGED and self.rng.random() < p:
            self._pending_return = now + self.rng.uniform(1.0, 5.0)  # reaction delay

    def step(self, now: float, dt: float) -> SensorReading:
        if self.state == AttentionState.ENGAGED:
            if self.rng.random() < self.drift_rate * dt:
                self.state = (AttentionState.AWAY if self.rng.random() < self.away_share
                              else AttentionState.DISTRACTED)
            elif self.rng.random() < 0.5 * dt:
                self.last_input = now  # typing now and then
        else:
            returned = (self._pending_return is not None and now >= self._pending_return)
            if returned or self.rng.random() < self.self_return_rate * dt:
                self.state = AttentionState.ENGAGED
                self._pending_return = None
                self.last_input = now

        idle = max(0.0, now - self.last_input)
        if self.state == AttentionState.ENGAGED:
            return SensorReading(True, 0.05, False, idle, now)
        if self.state == AttentionState.DISTRACTED:
            eyes_closed = self.rng.random() < 0.5
            return SensorReading(True, 0.9 if eyes_closed else 0.1, not eyes_closed, idle, now)
        return SensorReading(False, None, None, idle, now)
