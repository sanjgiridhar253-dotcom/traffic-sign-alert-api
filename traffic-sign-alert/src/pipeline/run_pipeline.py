"""
run_pipeline.py

The full loop from your architecture slide, wired end to end:

    dashcam frames -> YOLO recognition -> action/modality decision -> alert

and instrumented to produce exactly the three Evaluation Metrics numbers.
This is what you'd run for your Demo section.

Usage:
    python -m src.pipeline.run_pipeline \
        --frames data/processed/frames \
        --weights models/runs/sign_detector/weights/best.pt \
        --ground-truth data/processed/frames_ground_truth.json   # optional, enables accuracy

If --ground-truth is omitted, accuracy isn't computed (you can still demo
recognition + decision + alert latency without labeled ground truth).
"""
import argparse
import json
import os

from src.action_inference.action_map import decide_action, estimate_time_to_action
from src.alerts.alert_simulator import AlertSimulator
from src.pipeline.metrics import RunMetrics, timer_ms


def load_ground_truth(path: str | None) -> dict:
    """Expected format: {"frame_0000123ms.jpg": "stop", ...} - one label
    per frame, matching filenames produced by extract_frames.py."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def run(frames_dir: str, weights_path: str, ground_truth_path: str | None,
        conf_threshold: float, alert_log: str):
    from ultralytics import YOLO  # imported here so metrics.py stays torch-free

    model = YOLO(weights_path)
    ground_truth = load_ground_truth(ground_truth_path)
    simulator = AlertSimulator(log_path=alert_log)
    metrics = RunMetrics()

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png")))
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir} - run extract_frames.py first.")

    for frame_name in frame_files:
        frame_path = os.path.join(frames_dir, frame_name)

        # --- recognition stage (timed) ---
        with timer_ms() as recog_timer:
            results = model.predict(frame_path, conf=conf_threshold, verbose=False)[0]
        recognition_ms = recog_timer()

        if len(results.boxes) == 0:
            continue  # no sign in this frame

        # take the highest-confidence detection in this frame for the demo;
        # a production version would loop over every box
        box = results.boxes[results.boxes.conf.argmax()]
        cls_id = int(box.cls.item())
        predicted_class = model.names[cls_id]
        bbox_height = float(box.xywh[0][3])
        frame_height = results.orig_shape[0]

        if frame_name in ground_truth:
            correct = (predicted_class == ground_truth[frame_name])
            metrics.record_recognition(correct, recognition_ms)

        # --- decision stage (timed) ---
        with timer_ms() as decision_timer:
            seconds_to_action = estimate_time_to_action(bbox_height, frame_height)
            decision = decide_action(predicted_class, seconds_to_action)
        decision_ms = decision_timer()
        metrics.record_decision(decision_ms)

        # --- alert stage ---
        simulator.fire(decision)
        print(f"{frame_name}: {predicted_class} -> {decision.urgency.value} "
              f"(recognition {recognition_ms:.1f}ms, decision {decision_ms:.1f}ms)")

    print("\n=== Run summary ===")
    for key, value in metrics.summary().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--alert-log", default="data/processed/alert_log.tsv")
    args = parser.parse_args()

    run(args.frames, args.weights, args.ground_truth, args.conf_threshold, args.alert_log)
