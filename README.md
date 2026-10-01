# FIDO — Focus-Inducing Distraction Observer

A Raspberry Pi 5 desktop pet that notices when you drift off task and nudges you back.
A **rule engine** decides *when* to nudge; a **contextual bandit** learns *how* to nudge each person.
An **eye-state CNN** (trained on the MRL Eye Dataset) gives it its perception.

```
camera ─► MediaPipe ─► eye-state CNN ─┐
                                      ├─► Fusion ─► AttentionState ─► Rule Engine ─► Trigger
keyboard / mouse idle time ───────────┘                                               │
                                                                                      ▼
      LEDs · servo · buzzer · face  ◄── Actuator ◄── Nudge ◄── Contextual Bandit (learns)
                                                                      ▲
                                    did the user re-engage? ── reward ┘
```

## Try it in 1 minute (any laptop, no hardware)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                         # 14 tests, all should pass
python -m fido.main --sim                   # simulated user + face window, 20x speed
python scripts/simulate_learning.py --plot logs/curve.png   # learning curve vs random baseline
```

`--sim` uses a simulated user with hidden preferences (see `fido/sensing/sim.py`), so you can watch
the bandit discover that a head-tilt works when the user is distracted at the desk, and the buzzer
works when they've walked away.

## Project layout and owners

| Path | What | Owner |
|---|---|---|
| `fido/contracts.py` | Shared types — **the contract between modules. Agree Day 1, don't change silently.** | All |
| `fido/config.py` | Every threshold, pin number and timing in one place | All |
| `fido/sensing/camera.py` | Picamera2/OpenCV capture, MediaPipe FaceMesh, eye-state CNN, gaze | **Role 1 — Sensing & Perception** |
| `ml/train_eye_cnn.py` | Train the CNN on MRL Eye Dataset, export TFLite | Role 1 |
| `scripts/test_camera.py` | Live preview of what FIDO sees | Role 1 |
| `fido/sensing/activity.py` | Keyboard/mouse idle time (evdev on Pi, pynput on laptops) | **Role 2 — Rules & Fusion** |
| `fido/sensing/fusion.py` | Readings → ENGAGED / DISTRACTED / AWAY, with smoothing | Role 2 |
| `fido/rules/engine.py` | Rules R1–R7: when to nudge, cooldown, hourly cap | Role 2 |
| `fido/agent/bandit.py` | UCB / ε-greedy contextual bandit, reward tracker, persistence | **Role 3 — Learning Agent** |
| `fido/sensing/sim.py` | Simulated user for testing the bandit | Role 3 |
| `scripts/simulate_learning.py` | Learning curve vs random baseline (demo slide) | Role 3 |
| `fido/actuation/hardware.py` | LEDs, buzzer, servo via gpiozero; mock fallback | **Role 4 — Actuation & Integration** |
| `fido/actuation/face.py` | Pygame animated face | Role 4 |
| `fido/main.py` | Main loop wiring everything together | Role 4 |
| `scripts/test_actuators.py` | Fire every nudge pattern on the real pins | Role 4 |

Every module degrades gracefully: no camera → keyboard/mouse only; no CNN file → Eye Aspect Ratio;
no GPIO → mock prints; no display → `--no-face`. Nobody is blocked waiting on someone else.

## Raspberry Pi 5 setup

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-gpiozero python3-lgpio python3-opencv
sudo usermod -aG input,gpio $USER          # keyboard/mouse + GPIO access; log out and back in
git clone <your repo> fido && cd fido
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
python scripts/download_models.py          # MediaPipe face model (newer MediaPipe needs it)
python -m pytest -q
python scripts/test_actuators.py           # Day 4: check wiring
python scripts/test_camera.py              # Day 2: check camera + thresholds
python -m fido.main --demo                 # the real thing, demo timings
```

Notes:
- **Pi 5 GPIO:** `RPi.GPIO` does not work on the Pi 5. gpiozero with lgpio (installed above) does.
- **Pi Camera Module** goes through Picamera2; a **USB webcam** goes through OpenCV. `camera_backend="auto"` tries both.
- **Keyboard/mouse on the Pi** use evdev because the Pi 5 desktop runs Wayland, where pynput can't see global input.

## Wiring (BCM pins — change in `fido/config.py`)

| Part | Pi pin | Notes |
|---|---|---|
| LED left (eye) | GPIO17 → 220 Ω → LED → GND | |
| LED right (eye) | GPIO27 → 220 Ω → LED → GND | |
| Buzzer (active) | GPIO22 → + , − → GND | |
| Servo SG90 signal | GPIO18 | V+ to 5 V (pin 2), GND to GND. External 5 V with shared GND if it jitters |

## Training the eye-state model (Role 1, laptop or Colab)

1. Download the MRL Eye Dataset from Kaggle (search "MRL Eye Dataset") and unzip it into `data/`.
2. `pip install -r ml/requirements-train.txt`
3. `python ml/train_eye_cnn.py --data data/<folder> --epochs 8` (add `--limit 20000` for a quick first run)
4. Copy `models/eye_state.tflite` to the Pi's `fido/models/`. FIDO picks it up automatically.

## Run options

```
python -m fido.main [--sim] [--demo] [--no-face] [--no-camera]
                    [--algorithm ucb|epsilon] [--fresh] [--speed 20] [--max-nudges N] [-v]
```

- Learned policy is saved to `logs/bandit_state.json` (sim runs use `bandit_state_sim.json`); `--fresh` starts from zero.
- Every nudge outcome is logged to `logs/outcomes.csv` — use it for the report's results section.

## Week plan

| Day | Goal | Who |
|---|---|---|
| 1 | Pi set up, repo cloned, `pytest` + `--sim` running on everyone's laptop, contracts agreed | All |
| 2 | Camera + MediaPipe live on the Pi; start CNN training | Role 1 |
| 3 | Activity sensor on the Pi, fusion and rules tuned against real readings | Role 1 + 2 |
| 4 | LEDs, buzzer, servo wired; `test_actuators.py` passes; face looks good | Role 4 |
| 5 | Bandit tuned with `simulate_learning.py`; reward window tuned on real use | Role 3 |
| 6 | Full `--demo` run end to end; record the learning curve | All |
| 7 | Buffer, rehearse demo, report and slides | All |

If Days 2–3 overrun, ship with the EAR fallback and keep training the CNN in parallel — don't let it eat Day 5.
