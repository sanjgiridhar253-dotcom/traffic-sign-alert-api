"""
split_signs_general.py

Stratified per-class split for the signs_general classification dataset.
Original class identities are preserved (numeric ClassId + labels.csv names).
They are not remapped onto configs/classes.txt or any other dataset.

Supported input layouts:

    1. Official signs_general download:
        data/raw/signs_general/
            DATA/<class_id>/*.png
            TEST/<class_id>/*.png
            labels.csv                 # ClassId,Name

       DATA and TEST are pooled per class, then split 80/10/10.

    2. Flat class folders:
        data/raw/signs_general/
            stop/*.jpg
            yield/*.jpg

Output layout (ImageFolder, ready for VGG16):
    output/
        train/<00_Class_name>/...
        val/<00_Class_name>/...
        test/<00_Class_name>/...
        classes.txt                    # one original class name per ClassId
        classes.csv                    # class_id,name,folder
        manifest.csv                   # every image: split + class + path

Usage:
    python src/preprocessing/split_signs_general.py \
        --input data/raw/signs_general \
        --output data/processed/signs_general_split \
        --seed 42

    python src/preprocessing/split_signs_general.py \
        --input data/raw/signs_general \
        --output data/processed/signs_general_split \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class ClassImages:
    class_id: str
    display_name: str
    folder_name: str
    files: list[Path]


def list_image_files(class_dir: Path) -> list[Path]:
    """Return sorted image files for a single class directory."""
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def slugify(name: str) -> str:
    """Filesystem-safe slug that still keeps the original class wording."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "unnamed"


def class_folder_name(class_id: str, display_name: str) -> str:
    """Unique folder name: zero-padded id + slug, so duplicate names stay distinct."""
    prefix = f"{int(class_id):02d}" if class_id.isdigit() else class_id
    slug = slugify(display_name)
    return f"{prefix}_{slug}" if slug else prefix


def load_class_names(labels_csv: Path) -> dict[str, str]:
    """Read ClassId -> Name from signs_general/labels.csv."""
    mapping: dict[str, str] = {}
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ClassId" not in reader.fieldnames or "Name" not in reader.fieldnames:
            raise ValueError(f"Expected ClassId,Name columns in {labels_csv}")
        for row in reader:
            class_id = str(row["ClassId"]).strip()
            mapping[class_id] = str(row["Name"]).strip()
    return mapping


def collect_classes(input_dir: Path, labels_csv: Path | None) -> list[ClassImages]:
    """Discover classes from DATA/TEST + labels.csv, or from flat class folders."""
    names = load_class_names(labels_csv) if labels_csv and labels_csv.exists() else {}
    data_dir = input_dir / "DATA"
    test_dir = input_dir / "TEST"

    if data_dir.is_dir() and test_dir.is_dir():
        class_ids: set[str] = set()
        for root in (data_dir, test_dir):
            class_ids.update(path.name for path in root.iterdir() if path.is_dir())

        def sort_key(class_id: str) -> tuple[int, str]:
            return (int(class_id), class_id) if class_id.isdigit() else (10**9, class_id)

        collected: list[ClassImages] = []
        for class_id in sorted(class_ids, key=sort_key):
            files: list[Path] = []
            for root in (data_dir, test_dir):
                class_dir = root / class_id
                if class_dir.is_dir():
                    files.extend(list_image_files(class_dir))
            display_name = names.get(class_id, class_id)
            collected.append(
                ClassImages(
                    class_id=class_id,
                    display_name=display_name,
                    folder_name=class_folder_name(class_id, display_name),
                    files=files,
                )
            )
        return collected

    class_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    collected = []
    for class_dir in class_dirs:
        class_id = class_dir.name
        display_name = names.get(class_id, class_id)
        collected.append(
            ClassImages(
                class_id=class_id,
                display_name=display_name,
                folder_name=class_folder_name(class_id, display_name) if names else class_id,
                files=list_image_files(class_dir),
            )
        )
    return collected


def compute_split_counts(total: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    """Allocate counts with remainder assigned to test to keep exact totals."""
    if not (abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9):
        raise ValueError(
            f"Fractions must sum to 1.0; got train={train_frac}, val={val_frac}, test={test_frac}"
        )
    if total <= 0:
        return 0, 0, 0

    train_count = int(total * train_frac)
    val_count = int(total * val_frac)
    test_count = total - train_count - val_count
    if test_count < 0:
        test_count = 0
    return train_count, val_count, test_count


def stratified_split(
    files: list[Path],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Split a class's files into train/val/test using a deterministic shuffle."""
    if not files:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count, val_count, test_count = compute_split_counts(total, train_frac, val_frac, test_frac)
    if train_count + val_count + test_count != total:
        raise ValueError(
            f"Split counts do not sum to total for class of size {total}: "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    # Remainder (including rounding leftover) always goes to test.
    test_files = shuffled[train_count + val_count :]

    return {"train": train_files, "val": val_files, "test": test_files}


def validate_class_split(class_id: str, files: list[Path], split_map: dict[str, list[Path]]) -> None:
    """Ensure no overlap and exact total preservation within each class."""
    seen: set[str] = set()
    total = 0

    for split_name in ("train", "val", "test"):
        for path in split_map.get(split_name, []):
            key = str(path.resolve())
            if key in seen:
                raise ValueError(
                    f"Duplicate image detected in multiple splits: class={class_id}, file={path}"
                )
            seen.add(key)
            total += 1

    if total != len(files):
        raise ValueError(
            f"Class {class_id} total mismatch: expected {len(files)} images, found {total} assigned across splits"
        )


def unique_dest_name(source_path: Path, used_names: set[str]) -> str:
    """Avoid overwriting when DATA and TEST share a filename inside one class."""
    name = source_path.name
    if name not in used_names:
        used_names.add(name)
        return name

    origin = source_path.parent.parent.name
    alt = f"{origin}__{name}"
    if alt not in used_names:
        used_names.add(alt)
        return alt

    stem, suffix = source_path.stem, source_path.suffix
    index = 2
    while True:
        candidate = f"{stem}__{index}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def print_summary(class_totals: dict[str, dict[str, int]], total_images: int) -> None:
    """Print per-class counts and overall totals with percentages."""
    print("\nPer-class split summary")
    print(f"{'id':>4} {'class':<36} {'total':>6} {'train':>8} {'val':>8} {'test':>8}")
    print("-" * 78)

    def sort_key(class_id: str) -> tuple[int, str]:
        return (int(class_id), class_id) if class_id.isdigit() else (10**9, class_id)

    for class_id in sorted(class_totals, key=sort_key):
        counts = class_totals[class_id]
        print(
            f"{class_id:>4} {counts['name']:<36} {counts['total']:>6} {counts['train']:>8} "
            f"{counts['val']:>8} {counts['test']:>8}"
        )

    overall = {
        "train": sum(v["train"] for v in class_totals.values()),
        "val": sum(v["val"] for v in class_totals.values()),
        "test": sum(v["test"] for v in class_totals.values()),
        "total": total_images,
    }

    print("\nOverall totals")
    print(f"{'split':<8} {'count':>8} {'percent':>10}")
    print("-" * 30)
    for split_name in ("train", "val", "test"):
        count = overall[split_name]
        pct = (count / total_images * 100.0) if total_images else 0.0
        print(f"{split_name:<8} {count:>8} {pct:>9.2f}%")
    print(f"{'total':<8} {overall['total']:>8} {100.0 if total_images else 0.0:>9.2f}%")


def write_classifier_files(
    output_dir: Path,
    classes: list[ClassImages],
    class_totals: dict[str, dict[str, int]],
    manifest_rows: list[dict[str, str]],
    classes_out: Path,
    manifest_out: Path,
) -> None:
    """Write the class list and per-image classification table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    class_lines = "".join(f"{item.display_name}\n" for item in classes)
    local_classes = output_dir / "classes.txt"
    local_classes.write_text(class_lines, encoding="utf-8", newline="\n")
    if classes_out.resolve() != local_classes.resolve():
        classes_out.write_text(class_lines, encoding="utf-8", newline="\n")

    classes_csv = output_dir / "classes.csv"
    with classes_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "name", "folder", "total", "train", "val", "test"])
        writer.writeheader()
        for item in classes:
            counts = class_totals[item.class_id]
            writer.writerow(
                {
                    "class_id": item.class_id,
                    "name": item.display_name,
                    "folder": item.folder_name,
                    "total": counts["total"],
                    "train": counts["train"],
                    "val": counts["val"],
                    "test": counts["test"],
                }
            )

    with manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_id", "class_name", "folder", "filename", "relative_path", "source_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote class list: {classes_out}")
    print(f"Wrote class table: {classes_csv}")
    print(f"Wrote image manifest: {manifest_out}")


def copy_split(
    class_item: ClassImages,
    output_root: Path,
    split_map: dict[str, list[Path]],
) -> list[dict[str, str]]:
    """Copy one class into train/val/test and return manifest rows."""
    used_names: set[str] = set()
    rows: list[dict[str, str]] = []

    for split_name, file_list in split_map.items():
        target_dir = output_root / split_name / class_item.folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in file_list:
            dest_name = unique_dest_name(source_path, used_names)
            destination = target_dir / dest_name
            shutil.copy2(source_path, destination)
            relative_path = f"{split_name}/{class_item.folder_name}/{dest_name}"
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_item.class_id,
                    "class_name": class_item.display_name,
                    "folder": class_item.folder_name,
                    "filename": dest_name,
                    "relative_path": relative_path.replace("\\", "/"),
                    "source_path": str(source_path).replace("\\", "/"),
                }
            )
    return rows


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    dry_run: bool,
    labels_csv: Path | None = None,
    classes_out: Path | None = None,
    manifest_out: Path | None = None,
) -> None:
    """Create a stratified split for each class in the input tree."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")
    if abs((train_frac + val_frac + test_frac) - 1.0) >= 1e-9:
        raise ValueError(
            f"Fractions must sum to 1.0; got train={train_frac}, val={val_frac}, test={test_frac}"
        )

    if labels_csv is None:
        default_labels = input_dir / "labels.csv"
        labels_csv = default_labels if default_labels.exists() else None

    classes = collect_classes(input_dir, labels_csv)
    if not classes:
        raise ValueError(f"No class folders found under {input_dir}")

    class_totals: dict[str, dict[str, int]] = {}
    seen_sources: set[str] = set()
    overall_total = 0
    manifest_rows: list[dict[str, str]] = []

    for class_item in classes:
        split_map = stratified_split(class_item.files, train_frac, val_frac, test_frac, seed)
        validate_class_split(class_item.class_id, class_item.files, split_map)

        for paths in split_map.values():
            for path in paths:
                key = str(path.resolve())
                if key in seen_sources:
                    raise ValueError(f"Duplicate source file across splits: {path}")
                seen_sources.add(key)

        class_totals[class_item.class_id] = {
            "name": class_item.display_name,
            "total": len(class_item.files),
            "train": len(split_map["train"]),
            "val": len(split_map["val"]),
            "test": len(split_map["test"]),
        }
        overall_total += len(class_item.files)

        if not dry_run:
            manifest_rows.extend(copy_split(class_item, output_dir, split_map))

    if dry_run:
        print(f"Dry run only: no files copied to {output_dir}")
    else:
        print(f"Copied files into {output_dir}")
        write_classifier_files(
            output_dir=output_dir,
            classes=classes,
            class_totals=class_totals,
            manifest_rows=manifest_rows,
            classes_out=classes_out or (output_dir / "classes.txt"),
            manifest_out=manifest_out or (output_dir / "manifest.csv"),
        )

    print_summary(class_totals, overall_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified 80/10/10 split for per-class image datasets.")
    parser.add_argument("--input", required=True, help="Input root: signs_general (DATA/TEST) or flat class folders")
    parser.add_argument("--output", required=True, help="Output root for train/val/test copies")
    parser.add_argument("--labels", default=None, help="Optional labels.csv (ClassId,Name). Default: <input>/labels.csv")
    parser.add_argument("--classes-out", default=None, help="Where to write the class-name list (default: <output>/classes.txt)")
    parser.add_argument("--manifest-out", default=None, help="Where to write the image classification CSV (default: <output>/manifest.csv)")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction for the training split")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction for the validation split")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction for the test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split counts without copying any files",
    )
    args = parser.parse_args()

    split_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        dry_run=args.dry_run,
        labels_csv=Path(args.labels) if args.labels else None,
        classes_out=Path(args.classes_out) if args.classes_out else None,
        manifest_out=Path(args.manifest_out) if args.manifest_out else None,
    )
