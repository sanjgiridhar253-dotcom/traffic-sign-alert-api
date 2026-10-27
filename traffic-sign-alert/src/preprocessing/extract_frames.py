"""
extract_frames.py

Pulls frames out of a dashcam mp4 at a fixed sampling rate and saves each
frame with its timestamp encoded in the filename. This timestamp is what
lets the downstream pipeline report "time taken to recognize a sign" and
"time taken to decide the action" against the original video clock,
not wall-clock processing time.

Usage:
    python src/preprocessing/extract_frames.py \
        --video data/raw/sample_dashcam.mp4 \
        --out data/processed/frames \
        --fps 5
"""
import argparse
import os
import cv2
from tqdm import tqdm


def extract_frames(video_path: str, out_dir: str, sample_fps: float) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # how many source frames to skip so we land close to `sample_fps`
    frame_interval = max(int(round(source_fps / sample_fps)), 1)

    saved_paths = []
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # zero-padded ms timestamp keeps filenames sortable
            fname = f"frame_{int(timestamp_ms):09d}ms.jpg"
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Saved {len(saved_paths)} frames to {out_dir} "
          f"(source ~{source_fps:.1f} fps -> sampled at ~{sample_fps} fps)")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to dashcam mp4")
    parser.add_argument("--out", required=True, help="Directory to write frames to")
    parser.add_argument("--fps", type=float, default=5.0,
                         help="Frames per second to sample (default 5 - enough "
                              "temporal resolution for sign tracking without "
                              "redundant near-duplicate frames)")
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.fps)
