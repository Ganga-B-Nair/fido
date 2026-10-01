"""Keyboard/mouse idle-time sensing — fallback signal (Role 2 owns fusion, Role 1 may help).

Backends, tried in order:
  1. evdev  — reads /dev/input directly; works on the Pi under Wayland.
              Needs the user in the `input` group:  sudo usermod -aG input $USER  (then re-login)
  2. pynput — works on laptops (X11 / macOS / Windows) for development.
  3. none   — reports None; fusion then relies on the camera alone.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class ActivitySensor:
    def __init__(self):
        self._last_input = time.time()
        self._lock = threading.Lock()
        self.backend = "none"
        self._stop = threading.Event()

    def _touch(self, *_args, **_kw):
        with self._lock:
            self._last_input = time.time()

    def seconds_since_input(self) -> float | None:
        if self.backend == "none":
            return None
        with self._lock:
            return time.time() - self._last_input

    def start(self) -> "ActivitySensor":
        if self._try_evdev() or self._try_pynput():
            log.info("Activity sensor backend: %s", self.backend)
        else:
            log.warning("No keyboard/mouse backend available — camera-only mode")
        return self

    def stop(self):
        self._stop.set()

    # --- backends -----------------------------------------------------------
    def _try_evdev(self) -> bool:
        try:
            import evdev
        except ImportError:
            return False
        devices = []
        for path in evdev.list_devices():
            try:
                d = evdev.InputDevice(path)
                caps = d.capabilities()
                if evdev.ecodes.EV_KEY in caps or evdev.ecodes.EV_REL in caps:
                    devices.append(d)
            except (PermissionError, OSError):
                continue
        if not devices:
            log.info("evdev: no readable input devices (are you in the 'input' group?)")
            return False

        def run():
            import select
            fds = {d.fd: d for d in devices}
            while not self._stop.is_set():
                r, _, _ = select.select(fds, [], [], 0.5)
                for fd in r:
                    try:
                        for _ev in fds[fd].read():
                            self._touch()
                    except OSError:
                        pass

        threading.Thread(target=run, daemon=True, name="evdev").start()
        self.backend = f"evdev({len(devices)} devices)"
        return True

    def _try_pynput(self) -> bool:
        try:
            from pynput import keyboard, mouse
            keyboard.Listener(on_press=self._touch).start()
            mouse.Listener(on_move=self._touch, on_click=self._touch, on_scroll=self._touch).start()
        except Exception as e:  # noqa: BLE001 - no display, Wayland, etc.
            log.info("pynput unavailable: %s", e)
            return False
        self.backend = "pynput"
        return True
