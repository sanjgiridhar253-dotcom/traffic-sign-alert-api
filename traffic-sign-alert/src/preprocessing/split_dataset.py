"""
split_dataset.py

Takes the unified images/ + labels/ directories (same filename stems,
produced by unify_labels.py + your raw image folders) and splits them into
train/val/test, writing the YOLO-expected directory layout:

    data/processed/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt

Usage:
    python src/preprocessing/split_dataset.py \
        --images data/processed/all_images \
        --labels data/processed/all_labels \
        --out data/processed \
        --val-frac 0.15 --test-frac 0.1
"""
import argparse
import os
import shutil

from sklearn.model_selection import train_test_split


def collect_pairs(images_dir: str, labels_dir: str) -> list[tuple[str, str]]:
    pairs = []
    for fname in os.listdir(images_dir):
        stem, _ = os.path.splitext(fname)
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(label_path):
            pairs.append((os.path.join(images_dir, fname), label_path))
        # else: image has no detected signs / no label file - skipped here;
        # decide separately whether you want background-only frames included
    return pairs


def write_split(pairs, images_dir, labels_dir, split_name):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for img_path, label_path in pairs:
        shutil.copy(img_path, os.path.join(images_dir, os.path.basename(img_path)))
        shutil.copy(label_path, os.path.join(labels_dir, os.path.basename(label_path)))
    print(f"{split_name}: {len(pairs)} image/label pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = collect_pairs(args.images, args.labels)
    if not pairs:
        raise RuntimeError("No matching image/label pairs found - check your paths.")

    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=(args.val_frac + args.test_frac), random_state=args.seed
    )
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=rel_test, random_state=args.seed
    )

    write_split(train_pairs, f"{args.out}/images/train", f"{args.out}/labels/train", "train")
    write_split(val_pairs, f"{args.out}/images/val", f"{args.out}/labels/val", "val")
    write_split(test_pairs, f"{args.out}/images/test", f"{args.out}/labels/test", "test")
