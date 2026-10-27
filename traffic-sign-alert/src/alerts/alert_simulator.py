"""
alert_simulator.py

Multi-modal feedback for the accessibility demo:
  - SCREEN: OpenCV flash overlay + console banner
  - AUDIO: text-to-speech when available (pyttsx3), else console
  - VIBRATION: console placeholder (swap for phone haptic later)

Usage:
    python src/alerts/alert_simulator.py
"""
from __future__ import annotations

import time
from dataclasses import asdict

import cv2
import numpy as np

from src.action_inference.action_map import AlertDecision, Modality, decide_action


class AlertSimulator:
    def __init__(
        self,
        log_path: str | None = None,
        enable_tts: bool = True,
        overlay_dir: str | None = "data/processed/alert_overlays",
        show_window: bool = True,
    ):
        self.log_path = log_path
        self.overlay_dir = overlay_dir
        self.show_window = show_window
        self._log_lines: list[str] = []
        self._tts = None
        self._gui_ok = None  # lazily probed
        if enable_tts:
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
            except Exception:
                self._tts = None  # MANUAL optional: pip install pyttsx3

    def _can_show(self) -> bool:
        if not self.show_window:
            return False
        if self._gui_ok is not None:
            return self._gui_ok
        try:
            # headless opencv-python-headless raises here
            cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__probe__")
            self._gui_ok = True
        except Exception:
            self._gui_ok = False
            print(
                "[SCREEN] GUI not available (headless OpenCV). "
                "Overlays will be saved to disk instead. "
                "MANUAL (optional): pip install opencv-python  OR run on a desktop session."
            )
        return self._gui_ok

    def fire(self, decision: AlertDecision, frame: np.ndarray | None = None) -> float:
        fired_at = time.time()
        overlay = frame.copy() if frame is not None else None

        for modality in decision.modalities:
            overlay = self._trigger_modality(modality, decision, overlay)

        if overlay is not None:
            if self._can_show():
                cv2.imshow("Traffic Sign Alert", overlay)
                cv2.waitKey(1)
            elif self.overlay_dir:
                import os
                os.makedirs(self.overlay_dir, exist_ok=True)
                path = os.path.join(self.overlay_dir, f"alert_{int(fired_at * 1000)}.jpg")
                cv2.imwrite(path, overlay)
                print(f"[SCREEN] saved overlay -> {path}")

        self._log(decision, fired_at)
        return fired_at

    def _trigger_modality(
        self,
        modality: Modality,
        decision: AlertDecision,
        frame: np.ndarray | None,
    ) -> np.ndarray | None:
        if modality == Modality.SCREEN:
            print(f"[SCREEN FLASH]  {decision.message}")
            if frame is not None:
                return self._draw_flash(frame, decision)
        elif modality == Modality.VIBRATION:
            print(f"[VIBRATE]       {decision.message}")
        elif modality == Modality.AUDIO:
            print(f"[AUDIO CUE]     \"{decision.message}\"")
            if self._tts is not None:
                try:
                    self._tts.say(decision.message)
                    self._tts.runAndWait()
                except Exception as e:
                    print(f"[AUDIO fallback] TTS failed ({e})")
        return frame

    @staticmethod
    def _draw_flash(frame: np.ndarray, decision: AlertDecision) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        # Full-frame tint flash
        color = (0, 0, 255) if decision.urgency.value == "high" else (0, 165, 255)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), color, thickness=-1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

        # Banner with sign + action
        banner_h = max(80, h // 6)
        cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), thickness=-1)
        line1 = f"SIGN: {decision.sign_name}"
        line2 = f"ACTION: {decision.action}"
        cv2.putText(out, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        # wrap long action text roughly
        cv2.putText(out, line2[:70], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _log(self, decision: AlertDecision, fired_at: float):
        line = (
            f"{fired_at:.3f}\t{decision.sign_class}\t{decision.urgency.value}\t"
            f"{','.join(m.value for m in decision.modalities)}\t{decision.action}"
        )
        self._log_lines.append(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    simulator = AlertSimulator(log_path="data/processed/alert_log.tsv", enable_tts=False)
    decision = decide_action("No entry", seconds_to_action=1.2, driver_profile="deaf")
    print("Decision:", asdict(decision))
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    simulator.fire(decision, blank)
    cv2.waitKey(1500)
    simulator.close()
