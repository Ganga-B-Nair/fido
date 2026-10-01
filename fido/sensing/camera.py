"""Camera sensing — Role 1 (Sensing & Perception).

Pipeline per frame:
    frame -> MediaPipe FaceMesh -> landmarks
          -> eye crops -> eye-state CNN (TFLite)  -> eyes_closed_prob
             (no model yet? falls back to Eye Aspect Ratio so the team is never blocked)
          -> nose vs. eye corners -> gaze_away

Runs in a background thread; call .latest() from the main loop.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from fido.config import SensingConfig

log = logging.getLogger(__name__)

# MediaPipe FaceMesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # p1..p6 for EAR
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 263, 33


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|). Drops toward 0 when the eye closes."""
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return float((v1 + v2) / (2.0 * h))


def _load_interpreter(path):
    """Try the Pi-friendly runtimes first, full TensorFlow last."""
    def litert():
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter

    def tflite_runtime():
        from tflite_runtime.interpreter import Interpreter
        return Interpreter

    def tensorflow():
        import tensorflow as tf
        return tf.lite.Interpreter

    for name, get in (("ai-edge-litert", litert), ("tflite-runtime", tflite_runtime),
                      ("tensorflow", tensorflow)):
        try:
            interp = get()(model_path=str(path))
            interp.allocate_tensors()
            log.info("Eye-state model loaded via %s", name)
            return interp
        except ImportError:
            continue
    log.warning("Found %s but no TFLite runtime — pip install ai-edge-litert", path)
    return None


class EyeStateModel:
    """Wraps the TFLite CNN trained by ml/train_eye_cnn.py. Output = P(eye closed)."""

    def __init__(self, cfg: SensingConfig):
        self.size = cfg.eye_input_size
        self.interp = _load_interpreter(cfg.eye_model_path) if cfg.eye_model_path.exists() else None
        if self.interp is None:
            log.warning("Eye-state CNN not loaded (%s) — using EAR fallback", cfg.eye_model_path)
        else:
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]

    @property
    def available(self) -> bool:
        return self.interp is not None

    def predict(self, eye_gray: np.ndarray) -> float:
        import cv2
        x = cv2.resize(eye_gray, (self.size, self.size)).astype(np.float32) / 255.0
        x = x.reshape(1, self.size, self.size, 1)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self.out["index"]).ravel()[0])


def _crop_eye(gray: np.ndarray, pts: np.ndarray, pad: float = 0.6) -> np.ndarray | None:
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w = x1 - x0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = w * (0.5 + pad) / 1.0
    xa, xb = int(max(cx - half, 0)), int(min(cx + half, gray.shape[1]))
    ya, yb = int(max(cy - half, 0)), int(min(cy + half, gray.shape[0]))
    if xb - xa < 8 or yb - ya < 8:
        return None
    return gray[ya:yb, xa:xb]


class FaceLandmarks:
    """MediaPipe face landmarks, whichever API the installed version offers.

    * mediapipe <= 0.10.21 : legacy mp.solutions.face_mesh (no model file needed)
    * newer mediapipe      : Tasks API FaceLandmarker, needs models/face_landmarker.task
                             (run: python scripts/download_models.py)
    Both return the same 468/478-point mesh, so the landmark indices above work for either.
    """

    def __init__(self, cfg: SensingConfig):
        import mediapipe as mp
        self.mp = mp
        self._t0 = time.time()
        self._last_ts = -1
        if hasattr(mp, "solutions"):
            self.kind = "solutions"
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            path = cfg.face_landmarker_path
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing — run `python scripts/download_models.py` once (needs internet)")
            self.kind = "tasks"
            self.mesh = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
        log.info("Face landmarks via MediaPipe %s API", self.kind)

    def detect(self, rgb: np.ndarray):
        """Returns a list of landmarks with .x/.y in 0..1, or None if no face."""
        if self.kind == "solutions":
            res = self.mesh.process(rgb)
            return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        ts = int((time.time() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)          # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.mesh.detect_for_video(img, ts)
        return res.face_landmarks[0] if res.face_landmarks else None


class _Frames:
    """Picks Picamera2 (Pi Camera Module) or OpenCV (USB webcam)."""

    def __init__(self, cfg: SensingConfig):
        self.cam = None
        self.cap = None
        backend = cfg.camera_backend
        if backend in ("auto", "picamera2"):
            try:
                from picamera2 import Picamera2
                self.cam = Picamera2()
                self.cam.configure(self.cam.create_preview_configuration(
                    main={"format": "RGB888", "size": (cfg.frame_width, cfg.frame_height)}))
                self.cam.start()
                log.info("Camera: Picamera2")
                return
            except Exception as e:  # noqa: BLE001
                if backend == "picamera2":
                    raise
                log.info("Picamera2 unavailable (%s), trying OpenCV", e)
        import cv2
        self.cap = cv2.VideoCapture(cfg.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {cfg.camera_index}")
        log.info("Camera: OpenCV index %d", cfg.camera_index)

    def read_rgb(self):
        import cv2
        if self.cam is not None:
            # Picamera2 "RGB888" is BGR byte order in memory, so convert for MediaPipe
            return cv2.cvtColor(self.cam.capture_array(), cv2.COLOR_BGR2RGB)
        ok, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

    def close(self):
        if self.cam is not None:
            self.cam.stop()
        if self.cap is not None:
            self.cap.release()


class CameraSensor:
    """Background thread producing the latest camera-derived signals."""

    def __init__(self, cfg: SensingConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = {"face_present": None, "eyes_closed_prob": None, "gaze_away": None,
                        "ear": None, "frame": None}
        self._stop = threading.Event()
        self._thread = None
        self.model = EyeStateModel(cfg)

    def start(self) -> "CameraSensor":
        self._mesh = FaceLandmarks(self.cfg)
        self._frames = _Frames(self.cfg)
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if hasattr(self, "_frames"):
            self._frames.close()

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def process(self, rgb: np.ndarray) -> dict:
        """Pure function of one frame -> signals. Exposed for scripts/test_camera.py."""
        import cv2
        h, w = rgb.shape[:2]
        lm = self._mesh.detect(rgb)
        if lm is None:
            return {"face_present": False, "eyes_closed_prob": None, "gaze_away": None, "ear": None}
        P = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        ear = (eye_aspect_ratio(P[LEFT_EYE]) + eye_aspect_ratio(P[RIGHT_EYE])) / 2

        if self.model.available:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            probs = [self.model.predict(c) for c in
                     (_crop_eye(gray, P[LEFT_EYE]), _crop_eye(gray, P[RIGHT_EYE])) if c is not None]
            closed = float(np.mean(probs)) if probs else None
        else:
            # EAR fallback mapped to a pseudo-probability around the threshold
            closed = float(np.clip((self.cfg.ear_closed_threshold + 0.06 - ear) / 0.12, 0, 1))

        # Head yaw proxy: nose should sit roughly midway between outer eye corners
        dl = abs(P[NOSE_TIP][0] - P[LEFT_EYE_OUTER][0])
        dr = abs(P[NOSE_TIP][0] - P[RIGHT_EYE_OUTER][0])
        yaw_ratio = min(dl, dr) / (max(dl, dr) + 1e-6)
        gaze_away = yaw_ratio < 0.35

        return {"face_present": True, "eyes_closed_prob": closed, "gaze_away": gaze_away, "ear": ear}

    def _run(self):
        period = 1.0 / self.cfg.fps
        while not self._stop.is_set():
            t0 = time.time()
            rgb = self._frames.read_rgb()
            if rgb is not None:
                try:
                    sig = self.process(rgb)
                except Exception:  # noqa: BLE001
                    log.exception("camera frame failed")
                    sig = None
                if sig is not None:
                    with self._lock:
                        self._latest.update(sig)
                        self._latest["frame"] = rgb
            time.sleep(max(0.0, period - (time.time() - t0)))
