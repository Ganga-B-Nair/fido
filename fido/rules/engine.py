"""Rule engine — the "agent that follows a set of rules" half of the brief (Role 2).

Deterministic, explainable, no learning. It decides WHEN to nudge; the bandit decides HOW.

Rules:
  R1  ENGAGED                                   -> reset timer, never nudge
  R2  DISTRACTED for >= distracted_trigger_s    -> nudge (context "distracted")
  R3  AWAY for >= away_trigger_s                -> nudge (context "away")
  R4  Within cooldown_s of the last nudge       -> suppress
  R5  max_nudges_per_hour reached               -> suppress
  R6  A nudge is still awaiting its outcome     -> suppress (one experiment at a time)
  R7  UNKNOWN state (sensors not ready)         -> never nudge
"""
from __future__ import annotations

from collections import deque

from fido.config import RuleConfig
from fido.contracts import AttentionState, Trigger


class RuleEngine:
    def __init__(self, cfg: RuleConfig):
        self.cfg = cfg
        self.state = AttentionState.UNKNOWN
        self.state_since: float | None = None
        self.last_nudge_at: float | None = None
        self.nudge_times: deque[float] = deque()
        self.last_reason = "starting"

    def update(self, state: AttentionState, now: float, nudge_pending: bool = False) -> Trigger | None:
        if state != self.state:
            self.state, self.state_since = state, now
        elapsed = now - self.state_since if self.state_since is not None else 0.0

        while self.nudge_times and now - self.nudge_times[0] > 3600:
            self.nudge_times.popleft()

        if state in (AttentionState.ENGAGED, AttentionState.UNKNOWN):
            self.last_reason = f"R1/R7: {state.value}"
            return None

        threshold = (self.cfg.distracted_trigger_s if state == AttentionState.DISTRACTED
                     else self.cfg.away_trigger_s)
        if elapsed < threshold:
            self.last_reason = f"{state.value} {elapsed:.0f}s / {threshold:.0f}s"
            return None
        if nudge_pending:
            self.last_reason = "R6: waiting on previous nudge"
            return None
        if self.last_nudge_at is not None and now - self.last_nudge_at < self.cfg.cooldown_s:
            self.last_reason = "R4: cooldown"
            return None
        if len(self.nudge_times) >= self.cfg.max_nudges_per_hour:
            self.last_reason = "R5: hourly cap"
            return None

        self.last_nudge_at = now
        self.nudge_times.append(now)
        self.last_reason = f"{'R2' if state == AttentionState.DISTRACTED else 'R3'}: trigger"
        return Trigger(state=state, distracted_for=elapsed, context=state.value, timestamp=now)
