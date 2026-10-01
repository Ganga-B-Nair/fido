"""Learning agent — contextual multi-armed bandit (Role 3).

Context = the attention state that triggered the nudge ("distracted" or "away").
Arms    = the Nudge types.
Reward  = 1.0 if the user re-engages fast, decaying to 0.3 at the edge of the window,
          0.0 if they don't re-engage at all.

Two selection policies, switchable in config:
  epsilon-greedy : explore a random arm with prob epsilon, else the best mean
  UCB1           : mean + c * sqrt(ln N / n)  — explores arms it's unsure about
State is saved to JSON so FIDO keeps what it learned between runs.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from fido.config import AgentConfig
from fido.contracts import AttentionState, Nudge, NudgeOutcome, Trigger


class ContextualBandit:
    def __init__(self, cfg: AgentConfig, arms: list[Nudge] | None = None, seed: int | None = None):
        self.cfg = cfg
        self.arms = arms or list(Nudge)
        self.rng = random.Random(seed)
        # stats[context][arm] = [pulls, mean_reward]
        self.stats: dict[str, dict[str, list[float]]] = {}

    # --- core ---------------------------------------------------------------
    def _ctx(self, context: str) -> dict[str, list[float]]:
        return self.stats.setdefault(context, {a.value: [0, 0.0] for a in self.arms})

    def select(self, context: str) -> Nudge:
        s = self._ctx(context)
        untried = [a for a in self.arms if s[a.value][0] == 0]
        if untried:                                   # try every arm once first
            return self.rng.choice(untried)
        if self.cfg.algorithm == "epsilon":
            if self.rng.random() < self.cfg.epsilon:
                return self.rng.choice(self.arms)
            return max(self.arms, key=lambda a: s[a.value][1])
        total = sum(s[a.value][0] for a in self.arms)
        return max(self.arms, key=lambda a: s[a.value][1]
                   + self.cfg.ucb_c * math.sqrt(math.log(total) / s[a.value][0]))

    def update(self, context: str, arm: Nudge, reward: float):
        n, mean = self._ctx(context)[arm.value]
        n += 1
        mean += (reward - mean) / n                   # incremental average
        self.stats[context][arm.value] = [n, mean]

    def best(self, context: str) -> tuple[Nudge, float] | None:
        s = self.stats.get(context)
        if not s:
            return None
        arm = max(self.arms, key=lambda a: s[a.value][1])
        return arm, s[arm.value][1]

    def summary(self) -> str:
        lines = []
        for ctx, s in self.stats.items():
            ranked = sorted(s.items(), key=lambda kv: -kv[1][1])
            lines.append(f"[{ctx}] " + "  ".join(f"{k}:{v[1]:.2f}(n={int(v[0])})" for k, v in ranked))
        return "\n".join(lines) or "(no data yet)"

    # --- persistence --------------------------------------------------------
    def save(self, path: Path | None = None):
        path = Path(path or self.cfg.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"algorithm": self.cfg.algorithm, "stats": self.stats}, indent=2))

    def load(self, path: Path | None = None) -> bool:
        path = Path(path or self.cfg.state_path)
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.stats = data.get("stats", {})
        for ctx in self.stats:                        # tolerate arms added after saving
            for a in self.arms:
                self.stats[ctx].setdefault(a.value, [0, 0.0])
        return True


class NudgeTracker:
    """Watches what happens after a nudge and turns it into a reward."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.pending: tuple[Nudge, str, float] | None = None  # (nudge, context, sent_at)
        self._log_ready = False

    @property
    def busy(self) -> bool:
        return self.pending is not None

    def start(self, nudge: Nudge, trigger: Trigger, now: float):
        self.pending = (nudge, trigger.context, now)

    def reward_for(self, seconds: float | None) -> float:
        if seconds is None:
            return 0.0
        frac = min(seconds / self.cfg.reengage_window_s, 1.0)
        return 1.0 - 0.7 * frac                       # 1.0 instant -> 0.3 at the deadline

    def check(self, state: AttentionState, now: float) -> NudgeOutcome | None:
        if self.pending is None:
            return None
        nudge, ctx, sent = self.pending
        waited = now - sent
        if state == AttentionState.ENGAGED:
            outcome = NudgeOutcome(nudge, ctx, True, waited, self.reward_for(waited), now)
        elif waited >= self.cfg.reengage_window_s:
            outcome = NudgeOutcome(nudge, ctx, False, None, 0.0, now)
        else:
            return None
        self.pending = None
        self._log(outcome)
        return outcome

    def _log(self, o: NudgeOutcome):
        path = Path(self.cfg.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(o).keys()))
            if new:
                w.writeheader()
            row = asdict(o)
            row["nudge"] = o.nudge.value
            w.writerow(row)
