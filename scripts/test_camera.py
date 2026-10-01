"""Day 2 camera check — live preview with the signals FIDO sees (Role 1).

    python scripts/test_camera.py

Shows face_present, EAR, P(eyes closed) and gaze_away on the video. Press q to quit.
Use it to tune ear_closed_threshold / eyes_closed_prob_threshold in fido/config.py.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fido.config import SensingConfig  # noqa: E402
from fido.sensing.camera import CameraSensor  # noqa: E402
from fido.sensing.fusion import classify  # noqa: E402
from fido.contracts import SensorReading  # noqa: E402

cfg = SensingConfig()
cam = CameraSensor(cfg).start()
print("Eye model:", "TFLite CNN" if cam.model.available else "EAR fallback")
try:
    while True:
        s = cam.latest()
        frame = s.get("frame")
        if frame is None:
            cv2.waitKey(50)
            continue
        state = classify(SensorReading(s["face_present"], s["eyes_closed_prob"], s["gaze_away"], None), cfg)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        lines = [f"state: {state.value}", f"face: {s['face_present']}",
                 f"EAR: {s['ear']:.3f}" if s["ear"] is not None else "EAR: -",
                 f"P(closed): {s['eyes_closed_prob']:.2f}" if s["eyes_closed_prob"] is not None else "P(closed): -",
                 f"gaze away: {s['gaze_away']}"]
        for i, txt in enumerate(lines):
            cv2.putText(bgr, txt, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        cv2.imshow("FIDO camera test", bgr)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
finally:
    cam.stop()
    cv2.destroyAllWindows()
