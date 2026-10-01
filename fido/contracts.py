"""Shared types — the contract between the four modules.

Agree on this file on Day 1 and avoid changing it without telling the team.

Data flow:
    sensing  -> SensorReading -> fusion -> AttentionState
    rules    -> AttentionState -> Trigger | None
    agent    -> Trigger -> Nudge   (and later: reward -> update)
    actuation-> Nudge -> LEDs / servo / buzzer / face
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class AttentionState(str, Enum):
    ENGAGED = "engaged"        # face present, eyes open, recent input
    DISTRACTED = "distracted"  # face present but eyes closed / looking away / idle
    AWAY = "away"              # no face and no input
    UNKNOWN = "unknown"        # sensors not ready


class Nudge(str, Enum):
    """The bandit's arms. Add or remove arms here; everything else adapts."""
    CHIRP = "chirp"            # short soft buzzer beep
    ALARM = "alarm"            # longer, louder buzzer pattern
    HEAD_TILT = "head_tilt"    # servo wiggle
    EYES_FLASH = "eyes_flash"  # LED blink pattern
    FACE_SAD = "face_sad"      # on-screen face turns sad / calls the user


@dataclass
class SensorReading:
    """One snapshot from the sensing layer (owned by Role 1)."""
    face_present: bool | None = None     # None = camera unavailable
    eyes_closed_prob: float | None = None  # 0..1 from the eye-state CNN (or EAR fallback)
    gaze_away: bool | None = None        # head turned away from screen
    seconds_since_input: float | None = None  # keyboard/mouse idle time (fallback signal)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trigger:
    """Emitted by the rule engine when a nudge is warranted (Role 2)."""
    state: AttentionState
    distracted_for: float          # seconds in a non-engaged state
    context: str                   # bandit context key, e.g. "distracted" or "away"
    timestamp: float = field(default_factory=time.time)


@dataclass
class NudgeOutcome:
    """Reward record for the learning agent (Role 3)."""
    nudge: Nudge
    context: str
    reengaged: bool
    seconds_to_reengage: float | None
    reward: float
    timestamp: float = field(default_factory=time.time)
