"""Day 4 hardware check — fires every nudge pattern on the real pins (Role 4).

    python scripts/test_actuators.py          # all patterns
    python scripts/test_actuators.py servo    # one of: leds, buzzer, servo, all
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fido.actuation.hardware import make_hardware  # noqa: E402
from fido.config import ActuationConfig  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "all"
hw = make_hardware(ActuationConfig())
print(f"Backend: {hw.name}")
try:
    if which in ("leds", "all"):
        print("LEDs: blink x5"); hw.blink(5, 0.2); time.sleep(0.5)
    if which in ("buzzer", "all"):
        print("Buzzer: chirp, then alarm"); hw.beep(1, 0.08); time.sleep(0.6); hw.beep(4, 0.25, 0.15)
    if which in ("servo", "all"):
        print("Servo: tilt x2"); hw.tilt(2)
    print("Done.")
finally:
    hw.close()
