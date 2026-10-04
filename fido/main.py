"""FIDO main loop — wires the four modules together (Role 4 owns integration).

Run from the project root:
    python -m fido.main --sim            # no hardware, simulated user, fast clock (anyone, any laptop)
    python -m fido.main --sim --no-face  # same, terminal only
    python -m fido.main --demo           # real sensors, short thresholds, face + live camera panel
    python -m fido.main --demo --no-camera-view   # same, face only
    python -m fido.main                  # real sensors, normal thresholds

Each tick:
    sensors -> SensorReading -> Fusion -> AttentionState
    RuleEngine(state) -> Trigger?            (rules: WHEN to nudge)
    Bandit.select(context) -> Nudge          (learning: HOW to nudge)
    Actuator.perform(nudge)
    NudgeTracker.check(state) -> reward -> Bandit.update
"""
from __future__ import annotations

import argparse
import logging
import time

from fido.agent.bandit import ContextualBandit, NudgeTracker
from fido.actuation.hardware import Actuator, make_hardware
from fido.config import Config, demo_config
from fido.contracts import AttentionState, SensorReading
from fido.rules.engine import RuleEngine
from fido.sensing.fusion import Fusion

log = logging.getLogger("fido")


class RealClock:
    def __init__(self, tick: float):
        self.tick = tick

    def now(self) -> float:
        return time.time()

    def sleep(self):
        time.sleep(self.tick)


class SimClock:
    """Advances simulated time by `tick` each step but only sleeps `real_sleep` — runs ~speedx faster."""

    def __init__(self, tick: float, speed: float):
        self.t, self.tick, self.real_sleep = 0.0, tick, tick / speed

    def now(self) -> float:
        return self.t

    def sleep(self):
        self.t += self.tick
        time.sleep(self.real_sleep)


def build_sensors(cfg: Config, args):
    """Returns read(now) -> SensorReading, cleanup, an optional sim user, and the camera (or None)."""
    if args.sim:
        from fido.sensing.sim import SimulatedUser
        user = SimulatedUser(seed=args.seed)
        return (lambda now: user.step(now, cfg.tick_s)), (lambda: None), user, None

    from fido.sensing.activity import ActivitySensor
    activity = ActivitySensor().start()
    camera = None
    if not args.no_camera:
        try:
            from fido.sensing.camera import CameraSensor
            camera = CameraSensor(cfg.sensing).start()
        except Exception as e:  # noqa: BLE001
            log.warning("Camera unavailable (%s) — keyboard/mouse fallback only", e)

    def read(now):
        cam = camera.latest() if camera else {}
        return SensorReading(face_present=cam.get("face_present"),
                             eyes_closed_prob=cam.get("eyes_closed_prob"),
                             gaze_away=cam.get("gaze_away"),
                             seconds_since_input=activity.seconds_since_input(),
                             timestamp=now)

    def cleanup():
        activity.stop()
        if camera:
            camera.stop()

    return read, cleanup, None, camera


def wait_and_render(clock, face, state, status, camera, fps: float = 25.0):
    """Sleep until the next logic tick, redrawing the window meanwhile so video and animations
    stay smooth (the rule engine and bandit only need to run every tick_s)."""
    if face is None or isinstance(clock, SimClock):
        if face:
            face.update(state, status, None)
        clock.sleep()
        return
    end = time.time() + clock.tick
    while time.time() < end and not face.quit_requested:
        face.update(state, status, camera.latest() if camera else None)
        time.sleep(1.0 / fps)


def run(args):
    cfg = demo_config() if (args.demo or args.sim) else Config()
    if args.algorithm:
        cfg.agent.algorithm = args.algorithm
    if args.sim:  # keep simulated learning separate from what FIDO learns about a real user
        cfg.agent.state_path = cfg.agent.state_path.with_name("bandit_state_sim.json")
        cfg.agent.log_path = cfg.agent.log_path.with_name("outcomes_sim.csv")

    read, cleanup_sensors, sim_user, camera = build_sensors(cfg, args)
    clock = SimClock(cfg.tick_s, args.speed) if args.sim else RealClock(cfg.tick_s)

    fusion = Fusion(cfg.sensing, window=cfg.rules.smoothing_window)
    rules = RuleEngine(cfg.rules)
    bandit = ContextualBandit(cfg.agent, seed=args.seed)
    tracker = NudgeTracker(cfg.agent)
    if not args.fresh and bandit.load():
        log.info("Loaded learned policy from %s", cfg.agent.state_path)

    face = None
    if not args.no_face:
        try:
            from fido.actuation.face import Face
            face = Face(cfg.actuation.face_size,
                        show_camera=camera is not None and not args.no_camera_view,
                        closed_threshold=cfg.sensing.eyes_closed_prob_threshold)
        except Exception as e:  # noqa: BLE001
            log.warning("Face window unavailable (%s)", e)
    hw = make_hardware(cfg.actuation, force_mock=args.sim)
    actuator = Actuator(hw, face)

    nudges = successes = 0
    log.info("FIDO running (%s, %s). Ctrl+C to stop.", "sim" if args.sim else "live", cfg.agent.algorithm)
    try:
        while True:
            now = clock.now()
            state = fusion.update(read(now))

            outcome = tracker.check(state, now)
            if outcome:
                bandit.update(outcome.context, outcome.nudge, outcome.reward)
                successes += outcome.reengaged
                log.info("  -> %s %s (reward %.2f) | success rate %d/%d",
                         outcome.nudge.value, "WORKED" if outcome.reengaged else "ignored",
                         outcome.reward, successes, nudges)

            if args.max_nudges and nudges >= args.max_nudges and not tracker.busy:
                break

            trigger = rules.update(state, now, nudge_pending=tracker.busy)
            if trigger:
                nudge = bandit.select(trigger.context)
                nudges += 1
                log.info("NUDGE #%d  context=%s after %.0fs  ->  %s",
                         nudges, trigger.context, trigger.distracted_for, nudge.value)
                actuator.perform(nudge)
                tracker.start(nudge, trigger, now)
                if sim_user:
                    sim_user.receive_nudge(nudge, now)

            wait_and_render(clock, face, state, rules.last_reason, camera)
            if face and face.quit_requested:
                break
    except KeyboardInterrupt:
        pass
    finally:
        bandit.save()
        log.info("Saved policy to %s\n%s", cfg.agent.state_path, bandit.summary())
        cleanup_sensors()
        hw.close()
        if face:
            face.close()


def main():
    p = argparse.ArgumentParser(description="FIDO — Focus-Inducing Distraction Observer")
    p.add_argument("--sim", action="store_true", help="simulated user, no hardware needed")
    p.add_argument("--demo", action="store_true", help="short thresholds for a live demo")
    p.add_argument("--no-face", action="store_true", help="don't open the pygame face window")
    p.add_argument("--no-camera", action="store_true", help="keyboard/mouse fallback only")
    p.add_argument("--no-camera-view", action="store_true",
                   help="use the camera but hide the live video panel")
    p.add_argument("--algorithm", choices=["ucb", "epsilon"])
    p.add_argument("--fresh", action="store_true", help="ignore saved policy, start learning from zero")
    p.add_argument("--speed", type=float, default=20.0, help="sim speed multiplier")
    p.add_argument("--max-nudges", type=int, default=0, help="stop after N nudges (0 = run forever)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    run(args)


if __name__ == "__main__":
    main()
