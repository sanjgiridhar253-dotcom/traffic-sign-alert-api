"""
prepare_data.py

Downloads (optional) and prepares GTSRB with a TRACK-WISE train/val split
as required by the IEEE paper (Section 3), then writes ImageFolder-style
crops at 32x32 under data/processed/gtsrb_split/.

MANUAL STEP (if auto-download fails):
  1. Download GTSRB training zip from:
       https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/
       GTSRB_Final_Training_Images.zip
     and test zip:
       GTSRB_Final_Test_Images.zip
       GTSRB_Final_Test_GT.zip  (CSV ground truth)
  2. Place them in data/raw/gtsrb_zips/
  3. Re-run this script with --skip-download

Usage:
    python -m src.gtsrb.prepare_data \
        --raw data/raw/gtsrb \
        --out data/processed/gtsrb_split \
        --val-frac 0.2 \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

TRAIN_URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB_Final_Training_Images.zip"
)
TEST_IMG_URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB_Final_Test_Images.zip"
)
TEST_GT_URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB_Final_Test_GT.zip"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already present: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    print("MANUAL FALLBACK: if this fails, download the zip yourself into data/raw/gtsrb_zips/")
    urllib.request.urlretrieve(url, dest)


def _unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _parse_train_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def _save_roi(src: Path, roi: tuple[int, int, int, int], dest: Path, size: int = 32) -> None:
    x1, y1, x2, y2 = roi
    with Image.open(src) as im:
        crop = im.convert("RGB").crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dest)


def prepare_from_extracted(raw_root: Path, out_root: Path, val_frac: float, seed: int) -> None:
    """Expects GTSRB extracted under raw_root with Training/ and Final_Test/."""
    train_root = None
    for cand in [
        raw_root / "GTSRB" / "Final_Training" / "Images",
        raw_root / "Final_Training" / "Images",
        raw_root / "Training",
    ]:
        if cand.exists():
            train_root = cand
            break
    if train_root is None:
        raise FileNotFoundError(
            f"Could not find GTSRB training images under {raw_root}. "
            "MANUAL: extract GTSRB_Final_Training_Images.zip into data/raw/gtsrb/"
        )

    # Group by (class, track) for track-wise split
    tracks: dict[tuple[int, str], list[tuple[Path, tuple[int, int, int, int]]]] = defaultdict(list)
    class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        class_id = int(class_dir.name)
        csv_files = list(class_dir.glob("GT-*.csv"))
        if not csv_files:
            continue
        for row in _parse_train_csv(csv_files[0]):
            fname = row["Filename"]
            track = fname.split("_")[0]  # e.g. 00000_00001.ppm -> track 00000
            roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
            tracks[(class_id, track)].append((class_dir / fname, roi))

    rng = random.Random(seed)
    # Stratify: for each class, hold out val_frac of tracks
    by_class: dict[int, list[str]] = defaultdict(list)
    for (cid, track) in tracks:
        by_class[cid].append(track)

    train_keys: set[tuple[int, str]] = set()
    val_keys: set[tuple[int, str]] = set()
    for cid, track_list in by_class.items():
        uniq = sorted(set(track_list))
        rng.shuffle(uniq)
        n_val = max(1, int(round(len(uniq) * val_frac))) if len(uniq) > 1 else 0
        for t in uniq[:n_val]:
            val_keys.add((cid, t))
        for t in uniq[n_val:]:
            train_keys.add((cid, t))

    if out_root.exists():
        shutil.rmtree(out_root)
    counts = {"train": 0, "val": 0, "test": 0}

    def write_split(keys: set[tuple[int, str]], split: str) -> None:
        for key in tqdm(sorted(keys), desc=f"Writing {split}"):
            cid, track = key
            for i, (img_path, roi) in enumerate(tracks[key]):
                dest = out_root / split / f"{cid:02d}" / f"{track}_{i:04d}.png"
                _save_roi(img_path, roi, dest)
                counts[split] += 1

    write_split(train_keys, "train")
    write_split(val_keys, "val")

    # Official test set (optional)
    test_img_root = None
    for cand in [
        raw_root / "GTSRB" / "Final_Test" / "Images",
        raw_root / "Final_Test" / "Images",
    ]:
        if cand.exists():
            test_img_root = cand
            break
    gt_csv = None
    for cand in [
        raw_root / "GT-final_test.csv",
        raw_root / "GTSRB" / "GT-final_test.csv",
        raw_root / "Final_Test" / "GT-final_test.csv",
    ]:
        if cand.exists():
            gt_csv = cand
            break
    # Sometimes GT is only in a separate extract
    if gt_csv is None:
        found = list(raw_root.rglob("GT-final_test.csv"))
        if found:
            gt_csv = found[0]

    if test_img_root and gt_csv:
        for row in tqdm(_parse_train_csv(gt_csv), desc="Writing test"):
            fname = row["Filename"]
            cid = int(row["ClassId"])
            roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
            dest = out_root / "test" / f"{cid:02d}" / fname.replace(".ppm", ".png")
            _save_roi(test_img_root / fname, roi, dest)
            counts["test"] += 1
    else:
        print(
            "WARNING: official test images/GT not found. "
            "MANUAL: extract GTSRB_Final_Test_Images.zip and GTSRB_Final_Test_GT.zip "
            "into data/raw/gtsrb/ then re-run."
        )

    print("Split counts:", counts)
    print(f"Wrote ImageFolder split to {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/gtsrb")
    parser.add_argument("--zips", default="data/raw/gtsrb_zips")
    parser.add_argument("--out", default="data/processed/gtsrb_split")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw)
    zips = Path(args.zips)
    zips.mkdir(parents=True, exist_ok=True)

    train_zip = zips / "GTSRB_Final_Training_Images.zip"
    test_zip = zips / "GTSRB_Final_Test_Images.zip"
    gt_zip = zips / "GTSRB_Final_Test_GT.zip"

    if not args.skip_download:
        try:
            _download(TRAIN_URL, train_zip)
            _download(TEST_IMG_URL, test_zip)
            _download(TEST_GT_URL, gt_zip)
        except Exception as e:
            print(f"Download failed: {e}")
            print("MANUAL INTERVENTION REQUIRED: place the three GTSRB zips in", zips)
            raise

    raw.mkdir(parents=True, exist_ok=True)
    if train_zip.exists() and not (raw / "GTSRB" / "Final_Training").exists():
        print("Extracting training zip...")
        _unzip(train_zip, raw)
    if test_zip.exists() and not list(raw.rglob("Final_Test/Images")):
        print("Extracting test images zip...")
        _unzip(test_zip, raw)
    if gt_zip.exists() and not list(raw.rglob("GT-final_test.csv")):
        print("Extracting test GT zip...")
        _unzip(gt_zip, raw)

    prepare_from_extracted(raw, Path(args.out), args.val_frac, args.seed)


if __name__ == "__main__":
    main()
