"""On-screen animated face (pygame) — Role 4.

Must be updated from the main thread (pygame requirement): call face.update(state, status)
once per loop tick. Expressions:
  ENGAGED    -> happy, slow blinks
  DISTRACTED -> worried, eyes look sideways
  AWAY       -> sleepy, half-closed eyes
  after a nudge -> a short reaction animation (alert / sad / wiggle)
Close the window or press Esc / q to quit FIDO.
"""
from __future__ import annotations

import math
import time

from fido.contracts import AttentionState, Nudge

BG = (24, 26, 33)
FG = (240, 236, 226)
ACCENT = (255, 184, 76)
SAD = (120, 170, 255)


class Face:
    def __init__(self, size=(480, 320), title="FIDO"):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 22)
        self.w, self.h = size
        self.reaction: Nudge | None = None
        self.reaction_until = 0.0
        self.quit_requested = False

    def react(self, nudge: Nudge, seconds: float = 2.5):
        # called from Actuator (main thread) — just records the reaction
        self.reaction, self.reaction_until = nudge, time.time() + seconds

    def update(self, state: AttentionState, status: str = ""):
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key in (pg.K_ESCAPE, pg.K_q)):
                self.quit_requested = True
        t = time.time()
        reacting = self.reaction if t < self.reaction_until else None

        self.screen.fill(BG)
        cx, cy = self.w // 2, self.h // 2 - 20
        eye_dx, eye_r = 80, 38
        color = SAD if reacting == Nudge.FACE_SAD else (ACCENT if reacting else FG)

        # eye openness and pupil offset by expression
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

        # mouth
        mouth_y = cy + 80
        if reacting == Nudge.FACE_SAD or state == AttentionState.DISTRACTED:
            pg.draw.arc(self.screen, color, (cx - 40, mouth_y, 80, 40), 0.2, math.pi - 0.2, 4)
        elif state == AttentionState.AWAY:
            pg.draw.line(self.screen, color, (cx - 25, mouth_y + 10), (cx + 25, mouth_y + 10), 4)
        else:
            pg.draw.arc(self.screen, color, (cx - 40, mouth_y - 30, 80, 40), math.pi + 0.2, 2 * math.pi - 0.2, 4)

        label = f"{state.value}" + (f"  ·  nudge: {reacting.value}" if reacting else "")
        self.screen.blit(self.font.render(label, True, FG), (12, self.h - 48))
        if status:
            self.screen.blit(self.font.render(status[:70], True, (150, 150, 160)), (12, self.h - 26))
        pg.display.flip()

    def close(self):
        self.pg.quit()
