"""Unit tests for the parts that don't need hardware.  Run:  python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fido.agent.bandit import ContextualBandit, NudgeTracker  # noqa: E402
from fido.config import AgentConfig, RuleConfig, SensingConfig  # noqa: E402
from fido.contracts import AttentionState as S, Nudge, SensorReading, Trigger  # noqa: E402
from fido.rules.engine import RuleEngine  # noqa: E402
from fido.sensing.fusion import Fusion, classify  # noqa: E402

CFG = SensingConfig()


# --- fusion -------------------------------------------------------------------
def test_recent_typing_is_engaged_even_if_eyes_look_closed():
    assert classify(SensorReading(True, 0.95, False, 1.0), CFG) == S.ENGAGED

def test_closed_eyes_is_distracted():
    assert classify(SensorReading(True, 0.95, False, 30.0), CFG) == S.DISTRACTED

def test_looking_away_is_distracted():
    assert classify(SensorReading(True, 0.05, True, 30.0), CFG) == S.DISTRACTED

def test_no_face_and_idle_is_away():
    assert classify(SensorReading(False, None, None, 120.0), CFG) == S.AWAY

def test_camera_missing_falls_back_to_idle_time():
    assert classify(SensorReading(None, None, None, 1.0), CFG) == S.ENGAGED
    assert classify(SensorReading(None, None, None, CFG.idle_threshold_s + 1), CFG) == S.DISTRACTED
    assert classify(SensorReading(None, None, None, CFG.away_idle_s + 1), CFG) == S.AWAY

def test_fusion_smooths_single_frame_flicker():
    f = Fusion(CFG, window=5)
    for _ in range(4):
        f.update(SensorReading(True, 0.05, False, 30.0))
    assert f.update(SensorReading(True, 0.95, False, 30.0)) == S.ENGAGED


# --- rules --------------------------------------------------------------------
def rules():
    return RuleEngine(RuleConfig(distracted_trigger_s=10, away_trigger_s=20, cooldown_s=30, max_nudges_per_hour=3))

def test_no_trigger_before_threshold_then_trigger():
    r = rules()
    assert r.update(S.DISTRACTED, 0) is None
    assert r.update(S.DISTRACTED, 9) is None
    t = r.update(S.DISTRACTED, 10)
    assert t is not None and t.context == "distracted"

def test_engaged_resets_timer():
    r = rules()
    r.update(S.DISTRACTED, 0)
    r.update(S.ENGAGED, 8)
    assert r.update(S.DISTRACTED, 9) is None
    assert r.update(S.DISTRACTED, 18) is None

def test_cooldown_and_pending_suppress():
    r = rules()
    r.update(S.DISTRACTED, 0)
    assert r.update(S.DISTRACTED, 10)
    assert r.update(S.DISTRACTED, 20) is None           # cooldown
    assert r.update(S.DISTRACTED, 45, nudge_pending=True) is None
    assert r.update(S.DISTRACTED, 45) is not None

def test_hourly_cap():
    r = rules()
    fired = 0
    r.update(S.AWAY, 0)
    for t in range(20, 600, 31):
        fired += r.update(S.AWAY, t) is not None
    assert fired == 3

def test_unknown_never_triggers():
    r = rules()
    r.update(S.UNKNOWN, 0)
    assert r.update(S.UNKNOWN, 1000) is None


# --- bandit -------------------------------------------------------------------
def test_bandit_finds_best_arm_per_context():
    import random
    rng = random.Random(0)
    truth = {"distracted": {Nudge.HEAD_TILT: 0.8}, "away": {Nudge.ALARM: 0.8}}
    for algo in ("ucb", "epsilon"):
        b = ContextualBandit(AgentConfig(algorithm=algo), seed=1)
        for _ in range(400):
            for ctx in truth:
                arm = b.select(ctx)
                b.update(ctx, arm, 1.0 if rng.random() < truth[ctx].get(arm, 0.2) else 0.0)
        assert b.best("distracted")[0] == Nudge.HEAD_TILT, algo
        assert b.best("away")[0] == Nudge.ALARM, algo

def test_bandit_save_load_roundtrip(tmp_path):
    cfg = AgentConfig(state_path=tmp_path / "s.json")
    b = ContextualBandit(cfg)
    b.update("away", Nudge.ALARM, 1.0)
    b.save()
    b2 = ContextualBandit(cfg)
    assert b2.load() and b2.stats["away"]["alarm"] == [1, 1.0]

def test_tracker_rewards(tmp_path):
    cfg = AgentConfig(reengage_window_s=10, log_path=tmp_path / "o.csv")
    tr = NudgeTracker(cfg)
    tr.start(Nudge.CHIRP, Trigger(S.DISTRACTED, 30, "distracted"), now=0)
    assert tr.check(S.DISTRACTED, 5) is None
    out = tr.check(S.ENGAGED, 5)
    assert out.reengaged and abs(out.reward - 0.65) < 1e-9
    tr.start(Nudge.CHIRP, Trigger(S.AWAY, 30, "away"), now=100)
    out = tr.check(S.AWAY, 110)
    assert not out.reengaged and out.reward == 0.0
    assert (tmp_path / "o.csv").read_text().count("\n") == 3
