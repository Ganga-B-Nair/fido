"""Physical outputs — LEDs (eyes), buzzer, servo (head) — Role 4.

Uses gpiozero. On the Pi 5, gpiozero needs the lgpio pin factory (RPi.GPIO does NOT work on Pi 5).
It's preinstalled in Raspberry Pi OS; inside a venv create it with --system-site-packages.

Wiring (BCM numbering, change in config.py):
  LED left  : GPIO17 -> 220Ω -> LED -> GND
  LED right : GPIO27 -> 220Ω -> LED -> GND
  Buzzer    : GPIO22 -> buzzer + ; buzzer - -> GND   (active buzzer)
  Servo SG90: signal -> GPIO18, V+ -> 5V (pin 2), GND -> GND
              (for stable motion power the servo from an external 5V supply with common GND)
"""
from __future__ import annotations

import logging
import threading
import time

from fido.config import ActuationConfig
from fido.contracts import Nudge

log = logging.getLogger(__name__)


class MockHardware:
    """Prints instead of driving pins — used on laptops and in --sim mode."""
    name = "mock"

    def leds(self, on: bool):
        log.debug("LEDs %s", "on" if on else "off")

    def blink(self, times: int, period: float = 0.2):
        log.info("[mock] eyes blink x%d", times)

    def beep(self, times: int, length: float = 0.1, gap: float = 0.1):
        log.info("[mock] buzzer x%d (%.1fs)", times, length)

    def tilt(self, wiggles: int = 2):
        log.info("[mock] head tilt x%d", wiggles)

    def close(self):
        pass


class PiHardware:
    name = "gpio"

    def __init__(self, cfg: ActuationConfig):
        from gpiozero import LED, AngularServo, Buzzer
        self.led_l = LED(cfg.led_left_pin)
        self.led_r = LED(cfg.led_right_pin)
        self.buzzer = Buzzer(cfg.buzzer_pin)
        # SG90 pulse widths; tweak if your servo jitters or doesn't reach the ends
        self.servo = AngularServo(cfg.servo_pin, min_angle=-60, max_angle=60,
                                  min_pulse_width=0.0005, max_pulse_width=0.0025)
        self.servo.angle = 0
        time.sleep(0.3)
        self.servo.detach()  # stop jitter while idle
        self.leds(True)

    def leds(self, on: bool):
        for led in (self.led_l, self.led_r):
            led.on() if on else led.off()

    def blink(self, times: int, period: float = 0.2):
        for _ in range(times):
            self.leds(False); time.sleep(period)
            self.leds(True); time.sleep(period)

    def beep(self, times: int, length: float = 0.1, gap: float = 0.1):
        for _ in range(times):
            self.buzzer.on(); time.sleep(length)
            self.buzzer.off(); time.sleep(gap)

    def tilt(self, wiggles: int = 2):
        for _ in range(wiggles):
            self.servo.angle = 35; time.sleep(0.25)
            self.servo.angle = -35; time.sleep(0.25)
        self.servo.angle = 0
        time.sleep(0.25)
        self.servo.detach()

    def close(self):
        self.leds(False)
        for dev in (self.led_l, self.led_r, self.buzzer, self.servo):
            dev.close()


def make_hardware(cfg: ActuationConfig, force_mock: bool = False):
    if force_mock:
        return MockHardware()
    try:
        hw = PiHardware(cfg)
        log.info("Hardware: GPIO via gpiozero")
        return hw
    except Exception as e:  # noqa: BLE001 - not on a Pi, missing lgpio, etc.
        log.warning("GPIO unavailable (%s) — using mock hardware", e)
        return MockHardware()


class Actuator:
    """Maps a Nudge to an output pattern and runs it without blocking the main loop."""

    def __init__(self, hardware, face=None):
        self.hw = hardware
        self.face = face
        self._busy = threading.Lock()

    def perform(self, nudge: Nudge):
        if self.face is not None:
            self.face.react(nudge)
        threading.Thread(target=self._run, args=(nudge,), daemon=True).start()

    def _run(self, nudge: Nudge):
        if not self._busy.acquire(blocking=False):
            return  # previous pattern still playing
        try:
            if nudge == Nudge.CHIRP:
                self.hw.beep(1, 0.08)
            elif nudge == Nudge.ALARM:
                self.hw.beep(4, 0.25, 0.15)
            elif nudge == Nudge.HEAD_TILT:
                self.hw.tilt(2)
            elif nudge == Nudge.EYES_FLASH:
                self.hw.blink(5, 0.15)
            elif nudge == Nudge.FACE_SAD:
                self.hw.blink(1, 0.4)  # face does the heavy lifting on screen
        finally:
            self._busy.release()
