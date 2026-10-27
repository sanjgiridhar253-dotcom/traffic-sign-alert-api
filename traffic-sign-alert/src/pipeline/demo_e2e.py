"""
demo_e2e.py

End-to-end accessibility demo:

  dashcam video / webcam / frames
      -> detect sign regions (YOLO if weights given, else color proposals)
      -> classify crop with VGG16 (signs_general)
      -> map to action + modality (hearing / deaf / both)
      -> screen flash + optional TTS voice

MANUAL prerequisites:
  1. Trained VGG16 at models/vgg16_signs_general.pt  (already done)
  2. A dashcam mp4 at data/raw/sample_dashcam.mp4    (YOU must add)
     OR use --source webcam / --source frames-dir
  3. Optional: pip install pyttsx3 for real voice
  4. Optional: YOLO weights for better detection than color heuristics

Usage:
    # From video file (recommended first demo)
    python -m src.pipeline.demo_e2e \
        --source data/raw/sample_dashcam.mp4 \
        --weights models/vgg16_signs_general.pt \
        --driver-profile both \
        --max-seconds 5

    # From webcam for ~5 seconds
    python -m src.pipeline.demo_e2e --source webcam --max-seconds 5

    # From already-extracted frames
    python -m src.pipeline.demo_e2e --source data/processed/frames
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.action_inference.action_map import decide_action, estimate_time_to_action
from src.alerts.alert_simulator import AlertSimulator
from src.detection.sign_proposals import propose_signs
from src.models.predict_vgg16 import classify_pil, default_transform, load_vgg16
from src.pipeline.metrics import RunMetrics, timer_ms


def iter_frames(source: str, max_seconds: float | None, sample_fps: float):
    """Yield (frame_bgr, timestamp_s) from video path, 'webcam', or frames dir."""
    src_path = Path(source)
    if src_path.is_dir():
        files = sorted(
            [p for p in src_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not files:
            raise FileNotFoundError(f"No frames in {source}")
        for i, f in enumerate(files):
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            yield frame, i / max(sample_fps, 1e-6)
        return

    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. MANUAL: plug in a camera or use --source path/to.mp4"
            )
    else:
        if not src_path.exists():
            raise FileNotFoundError(
                f"Missing video: {source}\n"
                "MANUAL INTERVENTION: place a dashcam mp4 at data/raw/sample_dashcam.mp4"
            )
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(int(round(src_fps / sample_fps)), 1)
    start = time.time()
    idx = 0
    while True:
        if max_seconds is not None and (time.time() - start) > max_seconds:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if source == "webcam":
                ts = time.time() - start
            yield frame, float(ts)
        idx += 1
    cap.release()


def detect_boxes(frame, yolo_model, conf: float):
    """Return list of (x1,y1,x2,y2,score)."""
    if yolo_model is not None:
        results = yolo_model.predict(frame, conf=conf, verbose=False)[0]
        boxes = []
        if results.boxes is not None:
            for b in results.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2, float(b.conf.item())))
        return boxes
    props = propose_signs(frame)
    return [(p.x1, p.y1, p.x2, p.y2, p.score) for p in props]


def run(
    source: str,
    weights: str,
    driver_profile: str,
    max_seconds: float,
    sample_fps: float,
    conf_threshold: float,
    yolo_weights: str | None,
    alert_log: str,
    cooldown_s: float,
    show: bool,
):
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Missing VGG16 weights: {weights}\n"
            "MANUAL: train first with python src/models/train_vgg16.py"
        )

    model, ckpt, device = load_vgg16(weights)
    tfm = default_transform()
    simulator = AlertSimulator(log_path=alert_log, enable_tts=True, show_window=show)
    metrics = RunMetrics()

    yolo_model = None
    if yolo_weights:
        if not Path(yolo_weights).exists():
            print(f"YOLO weights not found at {yolo_weights}; using color proposals.")
        else:
            from ultralytics import YOLO
            yolo_model = YOLO(yolo_weights)
            print(f"Using YOLO detector: {yolo_weights}")
    else:
        print("No --yolo-weights given; using color/contour proposals (demo detector).")

    last_alert: dict[str, float] = {}
    print(f"Driver profile={driver_profile}  max_seconds={max_seconds}  source={source}")

    for frame, ts in iter_frames(source, max_seconds, sample_fps):
        h, w = frame.shape[:2]
        with timer_ms() as recog_t:
            boxes = detect_boxes(frame, yolo_model, conf_threshold)
            best = None
            for (x1, y1, x2, y2, score) in boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pred = classify_pil(model, ckpt, device, pil, tfm)
                if best is None or pred["confidence"] > best["confidence"]:
                    best = {**pred, "box": (x1, y1, x2, y2), "det_score": score}
        recog_ms = recog_t()

        vis = frame.copy()
        def _maybe_show(img):
            if not show:
                return False
            try:
                cv2.imshow("Traffic Sign Alert", img)
                return (cv2.waitKey(1) & 0xFF) == ord("q")
            except cv2.error:
                return False

        if best is None:
            if show:
                cv2.putText(vis, "No sign detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if _maybe_show(vis):
                    break
            continue

        x1, y1, x2, y2 = best["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{best['class_name']} {best['confidence']:.0%}"
        cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Deduplicate alerts for the same class within cooldown
        key = best["class_name"]
        if key in last_alert and (ts - last_alert[key]) < cooldown_s:
            if _maybe_show(vis):
                break
            continue

        with timer_ms() as dec_t:
            seconds = estimate_time_to_action(y2 - y1, h)
            decision = decide_action(
                best["class_name"],
                seconds_to_action=seconds,
                driver_profile=driver_profile,
            )
        decision_ms = dec_t()
        metrics.record_recognition(True, recog_ms)  # no GT in live demo
        metrics.record_decision(decision_ms)

        print(
            f"t={ts:5.2f}s  {best['class_name']} ({best['confidence']:.1%}) -> "
            f"{decision.urgency.value} | {decision.action} "
            f"[recog {recog_ms:.0f}ms, decide {decision_ms:.0f}ms]"
        )
        simulator.fire(decision, vis if show else None)
        last_alert[key] = ts

        if _maybe_show(vis):
            break

    print("\n=== Demo summary ===")
    for k, v in metrics.summary().items():
        print(f"{k}: {v}")
    simulator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/sample_dashcam.mp4",
                        help="mp4 path | 'webcam' | frames directory")
    parser.add_argument("--weights", default="models/vgg16_signs_general.pt")
    parser.add_argument("--yolo-weights", default=None,
                        help="Optional YOLO .pt for detection; else color proposals")
    parser.add_argument("--driver-profile", default="both",
                        choices=["hearing", "deaf", "both"])
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Seconds before re-alerting the same sign class")
    parser.add_argument("--alert-log", default="data/processed/alert_log.tsv")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    Path(args.alert_log).parent.mkdir(parents=True, exist_ok=True)
    run(
        source=args.source,
        weights=args.weights,
        driver_profile=args.driver_profile,
        max_seconds=args.max_seconds,
        sample_fps=args.sample_fps,
        conf_threshold=args.conf_threshold,
        yolo_weights=args.yolo_weights,
        alert_log=args.alert_log,
        cooldown_s=args.cooldown,
        show=not args.no_show,
    )
