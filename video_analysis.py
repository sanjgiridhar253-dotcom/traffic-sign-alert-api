"""Classify signs in a short dashcam clip with the trained VGG16 model.

Frames are read at 24 per second, up to 10 seconds. A sign is reported only
when the model keeps predicting that class above the confidence cutoff.
Actions are not chosen here; the caller looks them up with /predict.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "traffic-sign-alert"
MODEL_PATH = PROJECT_ROOT / "models" / "vgg16_signs_general.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.sign_proposals import propose_signs
from src.models.predict_vgg16 import classify_pil, default_transform, load_vgg16

TARGET_FPS = 24.0
MAX_SECONDS = 10.0
MAX_PROPOSALS = 3
# A sign must repeat: either on 3 frames inside half a second, or as two
# compact sign-shaped crops within about two seconds. One-off guesses are dropped.
BURST_CONFIDENCE = 0.70
BURST_COUNT = 3
BURST_WINDOW_S = 0.5
SHAPED_CONFIDENCE = 0.85
SHAPED_GAP_S = 2.2
MIN_ASPECT = 0.70
MAX_ASPECT = 1.25
BURST_MIN_ASPECT = 0.45
BURST_MAX_ASPECT = 1.45
MAX_AREA_FRAC = 0.03

# Model class index follows sign_actions.csv row order. These checks fail
# startup if either file is reordered, so a sign cannot be paired with the
# wrong action.
_INDEX_CHECKS = {
    0: "speed_limit_5kmph",
    7: "speed_limit_80kmph",
    18: "no_entry",
    30: "bicycles_crossing",
    36: "bicycles_crossing_2",
    51: "heavy_vehicle_accidents",
}

_lock = threading.Lock()
_runtime: dict | None = None


def _bind_sign_ids(class_folders: list[str], sign_ids: list[str]) -> list[str]:
    if len(class_folders) != len(sign_ids):
        raise RuntimeError(
            f"Model has {len(class_folders)} classes and the CSV has {len(sign_ids)} signs."
        )
    for index, folder in enumerate(class_folders):
        if not str(folder).startswith(f"{index:02d}_"):
            raise RuntimeError(f"Unexpected model class at index {index}: {folder}")
    for index, sign_id in _INDEX_CHECKS.items():
        if sign_ids[index] != sign_id:
            raise RuntimeError(
                f"CSV row {index} is {sign_ids[index]!r}; expected {sign_id!r}."
            )
    return list(sign_ids)


def load_runtime(sign_ids: list[str]) -> dict:
    """Load the checkpoint once. Later calls reuse it."""
    global _runtime
    with _lock:
        if _runtime is not None:
            return _runtime
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
        model, ckpt, device = load_vgg16(str(MODEL_PATH))
        folders = [str(name) for name in ckpt["classes"]]
        _runtime = {
            "model": model,
            "ckpt": ckpt,
            "device": device,
            "transform": default_transform(),
            "sign_ids": _bind_sign_ids(folders, sign_ids),
        }
        return _runtime


def analyze_video(path: Path, sign_ids: list[str]) -> dict:
    """Return confirmed sign ids in the order they first appear."""
    runtime = load_runtime(sign_ids)
    hits_by_sign: dict[str, list[dict]] = {}
    order: list[str] = []
    sampled = 0

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open that video. Choose an MP4 clip.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_s = frame_count / source_fps if source_fps > 0 and frame_count > 0 else None
    step = 1.0 / TARGET_FPS
    target = 0.0
    last_ts = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if timestamp_s > MAX_SECONDS + 1e-3:
                break
            last_ts = timestamp_s
            if timestamp_s + 1e-4 < target:
                continue
            sampled += 1
            target += step
            if target < timestamp_s:
                target = timestamp_s + step
            _collect_hits(frame, timestamp_s, runtime, hits_by_sign, order)
    finally:
        capture.release()

    if sampled == 0:
        raise ValueError("That video has no readable frames.")

    confirmed = _confirmed_signs(order, hits_by_sign)
    analyzed_s = min(last_ts, MAX_SECONDS)
    actual_fps = round(sampled / analyzed_s, 2) if analyzed_s > 0 else None
    return {
        "source_fps": round(source_fps, 3) if source_fps else None,
        "duration_s": None if duration_s is None else round(duration_s, 3),
        "analyzed_s": round(analyzed_s, 3),
        "frames_sampled": sampled,
        "sampled_fps": actual_fps,
        "target_fps": TARGET_FPS,
        "trimmed": bool(duration_s and duration_s > MAX_SECONDS + 0.05),
        "signs": [item["sign"] for item in confirmed],
        "detections": confirmed,
    }


def _collect_hits(frame: np.ndarray, timestamp_s: float, runtime: dict, hits_by_sign: dict, order: list[str]) -> None:
    frame_height, frame_width = frame.shape[:2]
    frame_area = float(frame_height * frame_width)
    best_by_sign: dict[str, dict] = {}
    for proposal in propose_signs(frame)[:MAX_PROPOSALS]:
        crop = frame[proposal.y1:proposal.y2, proposal.x1:proposal.x2]
        if crop.size == 0:
            continue
        box_width = proposal.x2 - proposal.x1
        box_height = proposal.y2 - proposal.y1
        if box_width < 16 or box_height < 16:
            continue
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        prediction = classify_pil(
            runtime["model"],
            runtime["ckpt"],
            runtime["device"],
            image,
            runtime["transform"],
        )
        if prediction["confidence"] < BURST_CONFIDENCE:
            continue
        sign_id = runtime["sign_ids"][prediction["class_index"]]
        hit = {
            "sign": sign_id,
            "model_label": prediction["class_name"],
            "confidence": float(prediction["confidence"]),
            "time_s": timestamp_s,
            "aspect": box_width / box_height,
            "area_frac": (box_width * box_height) / frame_area,
        }
        previous = best_by_sign.get(sign_id)
        if previous is None or hit["confidence"] > previous["confidence"]:
            best_by_sign[sign_id] = hit
    for sign_id, hit in best_by_sign.items():
        if sign_id not in hits_by_sign:
            order.append(sign_id)
            hits_by_sign[sign_id] = []
        hits_by_sign[sign_id].append(hit)


def _confirmed_signs(order: list[str], hits_by_sign: dict[str, list[dict]]) -> list[dict]:
    confirmed: list[dict] = []
    for sign_id in order:
        items = hits_by_sign[sign_id]
        kept: dict[float, dict] = {}
        for hit in _burst_hits(items):
            kept[hit["time_s"]] = hit
        for hit in _shaped_repeats(items):
            kept[hit["time_s"]] = hit
        if not kept:
            continue
        chosen = sorted(kept.values(), key=lambda hit: hit["time_s"])
        peak = max(chosen, key=lambda hit: hit["confidence"])
        confirmed.append({
            "sign": sign_id,
            "model_label": peak["model_label"],
            "max_confidence": round(peak["confidence"], 4),
            "frames": len(chosen),
            "first_seen_s": round(chosen[0]["time_s"], 3),
            "last_seen_s": round(chosen[-1]["time_s"], 3),
        })
    confirmed.sort(key=lambda item: item["first_seen_s"])
    return confirmed


def _compact_sign(hit: dict, min_aspect: float, max_aspect: float, min_confidence: float) -> bool:
    if hit["confidence"] < min_confidence:
        return False
    if not (min_aspect <= hit["aspect"] <= max_aspect):
        return False
    if hit["area_frac"] <= MAX_AREA_FRAC:
        return True
    # A sign held close to the camera fills more of the frame, but it stays square.
    return (
        hit["confidence"] >= 0.95
        and hit["area_frac"] <= 0.20
        and MIN_ASPECT <= hit["aspect"] <= MAX_ASPECT
    )


def _burst_hits(items: list[dict]) -> list[dict]:
    strong = sorted(
        (
            hit for hit in items
            if _compact_sign(hit, BURST_MIN_ASPECT, BURST_MAX_ASPECT, BURST_CONFIDENCE)
        ),
        key=lambda hit: hit["time_s"],
    )
    kept: list[dict] = []
    for index, start in enumerate(strong):
        window: list[dict] = []
        for hit in strong[index:]:
            if hit["time_s"] - start["time_s"] > BURST_WINDOW_S:
                break
            window.append(hit)
        if len(window) >= BURST_COUNT:
            kept.extend(window)
    return kept


def _shaped_repeats(items: list[dict]) -> list[dict]:
    shaped = [
        hit for hit in items
        if _compact_sign(hit, MIN_ASPECT, MAX_ASPECT, SHAPED_CONFIDENCE)
    ]
    shaped.sort(key=lambda hit: hit["time_s"])
    if len(shaped) < 2:
        return []
    clusters: list[list[dict]] = [[shaped[0]]]
    for hit in shaped[1:]:
        if hit["time_s"] - clusters[-1][-1]["time_s"] <= SHAPED_GAP_S:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    kept: list[dict] = []
    for cluster in clusters:
        if len(cluster) >= 2:
            kept.extend(cluster)
    return kept


def model_is_available() -> str | None:
    """Return an error string when video checking cannot run."""
    if not MODEL_PATH.is_file():
        return f"Trained model not found at {MODEL_PATH}."
    return None
