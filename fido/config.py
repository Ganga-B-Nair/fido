"""All tunable numbers live here so tuning on Day 6 never means hunting through code."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SensingConfig:
    camera_backend: str = "auto"        # "auto" | "picamera2" (Pi Camera Module) | "opencv" (USB webcam)
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 10                       # Pi 5 handles 10 fps MediaPipe comfortably
    eye_model_path: Path = ROOT / "models" / "eye_state.tflite"
    eye_input_size: int = 64            # must match ml/train_eye_cnn.py
    face_landmarker_path: Path = ROOT / "models" / "face_landmarker.task"  # newer MediaPipe only
    ear_closed_threshold: float = 0.21  # EAR fallback when no model file yet
    eyes_closed_prob_threshold: float = 0.6
    idle_threshold_s: float = 20.0      # keyboard/mouse idle counts as distracted
    away_idle_s: float = 60.0


@dataclass
class RuleConfig:
    distracted_trigger_s: float = 30.0  # rule: distracted this long -> nudge
    away_trigger_s: float = 90.0        # rule: away this long -> nudge
    cooldown_s: float = 45.0            # rule: never nudge twice within this window
    max_nudges_per_hour: int = 12       # rule: hard cap so FIDO never becomes spam
    smoothing_window: int = 5           # readings to majority-vote over (debounce)


@dataclass
class AgentConfig:
    algorithm: str = "ucb"              # "ucb" or "epsilon"
    epsilon: float = 0.15
    ucb_c: float = 0.5                  # rewards are 0..1, so a small c converges in a demo-length run
    reengage_window_s: float = 20.0     # how long we wait to see if the nudge worked
    state_path: Path = ROOT / "logs" / "bandit_state.json"
    log_path: Path = ROOT / "logs" / "outcomes.csv"


@dataclass
class ActuationConfig:
    # BCM pin numbers. Change to match your wiring.
    led_left_pin: int = 17
    led_right_pin: int = 27
    buzzer_pin: int = 22
    servo_pin: int = 18
    show_face: bool = True
    face_size: tuple = (480, 320)


@dataclass
class Config:
    sensing: SensingConfig = field(default_factory=SensingConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)
    tick_s: float = 0.5                 # main loop period


def demo_config() -> Config:
    """Short thresholds so the learning is visible in a 5-minute demo."""
    c = Config()
    c.rules.distracted_trigger_s = 6.0
    c.rules.away_trigger_s = 10.0
    c.rules.cooldown_s = 8.0
    c.rules.max_nudges_per_hour = 200
    c.agent.reengage_window_s = 8.0
    c.sensing.idle_threshold_s = 6.0
    return c
