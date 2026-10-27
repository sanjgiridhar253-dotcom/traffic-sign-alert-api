"""
sign_proposals.py

Lightweight sign-region proposals for the dashcam demo when a trained
YOLO detector is not available yet.

Uses color + contour heuristics (red/blue circular/rectangular blobs).
This is a DEMO detector, not production-grade.

When you have YOLO weights, prefer those in demo_e2e.py via --yolo-weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Proposal:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    def clip(self, w: int, h: int) -> "Proposal":
        return Proposal(
            max(0, self.x1), max(0, self.y1),
            min(w - 1, self.x2), min(h - 1, self.y2),
            self.score,
        )


def _mask_from_hsv(hsv: np.ndarray) -> np.ndarray:
    # Red wraps around hue=0
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 40), (140, 255, 255))
    yellow = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    return cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(blue, yellow))


def propose_signs(
    frame_bgr: np.ndarray,
    min_area_frac: float = 0.0015,
    max_area_frac: float = 0.25,
    max_proposals: int = 5,
) -> list[Proposal]:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = _mask_from_hsv(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(h * w)
    props: list[Proposal] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        frac = area / frame_area
        if frac < min_area_frac or frac > max_area_frac:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Prefer roughly square-ish sign boards
        ar = bw / max(bh, 1)
        if ar < 0.4 or ar > 2.5:
            continue
        # Expand box a little
        pad = int(0.08 * max(bw, bh))
        prop = Proposal(x - pad, y - pad, x + bw + pad, y + bh + pad, score=float(frac)).clip(w, h)
        if prop.x2 - prop.x1 < 16 or prop.y2 - prop.y1 < 16:
            continue
        props.append(prop)

    props.sort(key=lambda p: p.score, reverse=True)
    return props[:max_proposals]
