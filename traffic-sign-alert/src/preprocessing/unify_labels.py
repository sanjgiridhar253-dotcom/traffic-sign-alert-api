"""
unify_labels.py

Your three Kaggle datasets don't share one label format:
  - signs_yolo already ships YOLO .txt labels (class cx cy w h, normalized 0-1)
  - car_detection and signs_general commonly ship Pascal-VOC XML or a flat
    CSV of bounding boxes (filename,xmin,ymin,xmax,ymax,class)

Rather than guess the exact schema (check the dataset's README after
downloading and adjust the loader you need), this script gives you one
converter per common format, all writing out the *same* YOLO .txt layout so
every downstream step (training, this repo's dataloaders) only has to know
one format.

Usage:
    python src/preprocessing/unify_labels.py \
        --format voc \
        --annotations data/raw/car_detection/annotations \
        --images data/raw/car_detection/images \
        --classes data/raw/car_detection/classes.txt \
        --out data/processed/labels

    python src/preprocessing/unify_labels.py \
        --format csv \
        --annotations data/raw/signs_general/labels.csv \
        --images data/raw/signs_general/images \
        --classes data/raw/signs_general/classes.txt \
        --out data/processed/labels
"""
import argparse
import csv
import os
import xml.etree.ElementTree as ET

from PIL import Image


def load_class_map(classes_path: str) -> dict[str, int]:
    with open(classes_path) as f:
        names = [line.strip() for line in f if line.strip()]
    return {name: idx for idx, name in enumerate(names)}


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def convert_voc(annotations_dir: str, images_dir: str, class_map: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

    for xml_name in xml_files:
        tree = ET.parse(os.path.join(annotations_dir, xml_name))
        root = tree.getroot()

        img_w = int(root.find("size/width").text)
        img_h = int(root.find("size/height").text)
        stem = os.path.splitext(xml_name)[0]

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in class_map:
                continue  # skip classes you're not training on
            cls_id = class_map[cls_name]

            box = obj.find("bndbox")
            xmin, ymin = float(box.find("xmin").text), float(box.find("ymin").text)
            xmax, ymax = float(box.find("xmax").text), float(box.find("ymax").text)
            cx, cy, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_path = os.path.join(out_dir, f"{stem}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Converted {len(xml_files)} VOC XML files -> YOLO labels in {out_dir}")


def convert_csv(csv_path: str, images_dir: str, class_map: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows_by_image: dict[str, list] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_image.setdefault(row["filename"], []).append(row)

    for filename, rows in rows_by_image.items():
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lines = []
        for row in rows:
            cls_name = row["class"].strip()
            if cls_name not in class_map:
                continue
            cls_id = class_map[cls_name]
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax, ymax = float(row["xmax"]), float(row["ymax"])
            cx, cy, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        stem = os.path.splitext(filename)[0]
        out_path = os.path.join(out_dir, f"{stem}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Converted {len(rows_by_image)} CSV-annotated images -> YOLO labels in {out_dir}")


def passthrough_yolo(annotations_dir: str, out_dir: str):
    """signs_yolo is already in YOLO format - just copy it into the unified
    labels directory so every dataset ends up in one place."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fname in os.listdir(annotations_dir):
        if fname.endswith(".txt"):
            shutil.copy(os.path.join(annotations_dir, fname), os.path.join(out_dir, fname))
            count += 1
    print(f"Copied {count} already-YOLO label files into {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["voc", "csv", "yolo"], required=True)
    parser.add_argument("--annotations", required=True,
                         help="XML/label dir for voc & yolo, or the CSV file path for csv")
    parser.add_argument("--images", help="Images dir (needed for voc/csv to read width/height)")
    parser.add_argument("--classes", help="Text file, one class name per line, in the order "
                                           "you want class IDs assigned (skip for --format yolo)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.format == "yolo":
        passthrough_yolo(args.annotations, args.out)
    else:
        class_map = load_class_map(args.classes)
        if args.format == "voc":
            convert_voc(args.annotations, args.images, class_map, args.out)
        elif args.format == "csv":
            convert_csv(args.annotations, args.images, class_map, args.out)
