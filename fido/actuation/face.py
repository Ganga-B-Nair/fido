"""On-screen window: FIDO's animated face, plus an optional live camera panel — Role 4.

Must be updated from the main thread (pygame requirement): call face.update(state, status, camera)
many times a second. Expressions:
  ENGAGED    -> happy, slow blinks
  DISTRACTED -> worried, eyes look sideways
  AWAY       -> sleepy, half-closed eyes
  after a nudge -> a short reaction animation (alert / sad / wiggle)

Camera panel (right side, when show_camera=True): mirrored live video with the tracked eye
landmarks, a border coloured by attention state, and the live readings (EAR, P(closed), gaze).
Close the window or press Esc / q to quit FIDO.
"""
from __future__ import annotations

import math
import time

import numpy as np

from fido.contracts import AttentionState, Nudge

BG = (24, 26, 33)
FG = (240, 236, 226)
MUTED = (150, 150, 160)
ACCENT = (255, 184, 76)
SAD = (120, 170, 255)
STATE_COLOR = {AttentionState.ENGAGED: (90, 200, 120), AttentionState.DISTRACTED: (255, 184, 76),
               AttentionState.AWAY: (150, 150, 160), AttentionState.UNKNOWN: (110, 110, 120)}


class Face:
    def __init__(self, size=(480, 360), title="FIDO", show_camera: bool = False, cam_width: int = 480,
                 closed_threshold: float = 0.6):
        import pygame
        self.pg = pygame
        pygame.init()
        self.fw, self.h = size                       # face area
        self.cw = cam_width if show_camera else 0    # camera panel
        self.show_camera = show_camera
        self.closed_threshold = closed_threshold
        self.screen = pygame.display.set_mode((self.fw + self.cw, self.h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 22)
        self.big = pygame.font.SysFont(None, 30)
        self.reaction: Nudge | None = None
        self.reaction_until = 0.0
        self.quit_requested = False

    def react(self, nudge: Nudge, seconds: float = 2.5):
        self.reaction, self.reaction_until = nudge, time.time() + seconds

    # ------------------------------------------------------------------ face
    def _draw_face(self, state: AttentionState, status: str):
        pg = self.pg
        t = time.time()
        reacting = self.reaction if t < self.reaction_until else None
        pg.draw.rect(self.screen, BG, (0, 0, self.fw, self.h))

        cx, cy = self.fw // 2, self.h // 2 - 30
        eye_dx, eye_r = 80, 38
        color = SAD if reacting == Nudge.FACE_SAD else (ACCENT if reacting else FG)

        blink = (t % 4.0) < 0.12
        openness = {AttentionState.ENGAGED: 1.0, AttentionState.DISTRACTED: 0.8,
                    AttentionState.AWAY: 0.25}.get(state, 0.6)
        if reacting in (Nudge.ALARM, Nudge.EYES_FLASH, Nudge.CHIRP):
            openness = 1.25
        if blink and not reacting:
            openness = 0.08
        look_x = 18 * math.sin(t * 2) if state == AttentionState.DISTRACTED else 0
        bob = 10 * math.sin(t * 18) if reacting == Nudge.HEAD_TILT else 0

        for side in (-1, 1):
            ex, ey = cx + side * eye_dx + bob, cy
            rect = pg.Rect(0, 0, eye_r * 2, max(4, int(eye_r * 2 * openness)))
            rect.center = (ex, ey)
            pg.draw.ellipse(self.screen, color, rect)
            if openness > 0.3:
                pg.draw.circle(self.screen, BG, (int(ex + look_x), ey), 12)

        mouth_y = cy + 80
        if reacting == Nudge.FACE_SAD or state == AttentionState.DISTRACTED:
            pg.draw.arc(self.screen, color, (cx - 40, mouth_y, 80, 40), 0.2, math.pi - 0.2, 4)
        elif state == AttentionState.AWAY:
            pg.draw.line(self.screen, color, (cx - 25, mouth_y + 10), (cx + 25, mouth_y + 10), 4)
        else:
            pg.draw.arc(self.screen, color, (cx - 40, mouth_y - 30, 80, 40),
                        math.pi + 0.2, 2 * math.pi - 0.2, 4)

        label = state.value + (f"  ·  nudge: {reacting.value}" if reacting else "")
        self.screen.blit(self.font.render(label, True, FG), (12, self.h - 48))
        if status:
            self.screen.blit(self.font.render(status[:60], True, MUTED), (12, self.h - 26))

    # ---------------------------------------------------------------- camera
    def _draw_camera(self, state: AttentionState, cam: dict | None):
        pg = self.pg
        x0 = self.fw
        pg.draw.rect(self.screen, (12, 13, 17), (x0, 0, self.cw, self.h))
        frame = cam.get("frame") if cam else None
        if frame is None:
            msg = self.big.render("camera starting…", True, MUTED)
            self.screen.blit(msg, msg.get_rect(center=(x0 + self.cw // 2, self.h // 2)))
            return

        fh, fw = frame.shape[:2]
        mirrored = np.ascontiguousarray(frame[:, ::-1].swapaxes(0, 1))  # selfie view, (w, h, 3)
        surf = pg.transform.scale(pg.surfarray.make_surface(mirrored), (self.cw, self.h))
        self.screen.blit(surf, (x0, 0))
        sx, sy = self.cw / fw, self.h / fh

        closed_p = cam.get("eyes_closed_prob")
        eyes_closed = closed_p is not None and closed_p >= self.closed_threshold
        pts = cam.get("eye_pts")
        if pts is not None:
            dot = (255, 90, 90) if eyes_closed else (90, 230, 140)
            for px, py in pts:
                pg.draw.circle(self.screen, dot, (int(x0 + (fw - px) * sx), int(py * sy)), 3)

        sc = STATE_COLOR.get(state, MUTED)
        pg.draw.rect(self.screen, sc, (x0, 0, self.cw, self.h), 5)

        def fmt(v, spec):
            return "-" if v is None else format(v, spec)

        lines = [
            (state.value.upper(), sc),
            (f"face: {'yes' if cam.get('face_present') else 'no'}", FG),
            (f"EAR: {fmt(cam.get('ear'), '.3f')}", FG),
            (f"P(closed): {fmt(closed_p, '.2f')}  [{cam.get('eye_model', '?')}]",
             (255, 120, 120) if eyes_closed else FG),
            (f"gaze away: {'yes' if cam.get('gaze_away') else 'no'}",
             ACCENT if cam.get("gaze_away") else FG),
        ]
        panel = pg.Surface((210, 22 * len(lines) + 12), pg.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (x0 + 10, 10))
        for i, (txt, col) in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, col), (x0 + 18, 16 + 22 * i))

    # ---------------------------------------------------------------- public
    def update(self, state: AttentionState, status: str = "", camera: dict | None = None):
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key in (pg.K_ESCAPE, pg.K_q)):
                self.quit_requested = True
        self._draw_face(state, status)
        if self.show_camera:
            self._draw_camera(state, camera)
        pg.display.flip()

    def close(self):
        self.pg.quit()
