"""
train_vgg16.py

Fine-tunes VGG16 as a pure classifier on CROPPED sign regions (not full
frames). This is the two-stage alternative your PDF listed alongside
YOLOv8: crop each sign using ground-truth or YOLO-predicted boxes, then
classify the crop. Useful as a comparison point in your results section -
does a dedicated classifier beat YOLO's built-in classification head?

Expects the signs_general ImageFolder split:
    data/processed/signs_general_split/train/<class_name>/*.png
    data/processed/signs_general_split/test/<class_name>/*.png
    data/processed/signs_general_split/val/<class_name>/*.png   # optional

Usage:
    python src/models/train_vgg16.py \
        --data data/processed/signs_general_split \
        --out models/vgg16_signs_general.pt \
        --epochs 20
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# VGG16 features: unfreeze from this index onward (block 4 + block 5).
# Earlier blocks stay ImageNet-frozen.
UNFREEZE_FEATURES_FROM = 17


def build_model(num_classes: int) -> nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last two conv blocks + the full classifier (not just the final Linear).
    for layer in model.features[UNFREEZE_FEATURES_FROM:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def trainable_param_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_class_names(data_dir: str, folder_names: list[str]) -> list[str]:
    """Prefer classes.txt (original signs_general names) when present."""
    names_path = Path(data_dir) / "classes.txt"
    if names_path.exists():
        names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) == len(folder_names):
            return names
    return folder_names


def try_image_folder(path: str, transform) -> datasets.ImageFolder | None:
    """Load ImageFolder if the directory exists and contains at least one image."""
    if not os.path.isdir(path):
        return None
    try:
        ds = datasets.ImageFolder(path, transform=transform)
    except FileNotFoundError:
        return None
    return ds if len(ds) > 0 else None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 0 if os.name == "nt" or device.type == "cpu" else 2

    # Mild aug only (no horizontal flip — that breaks left/right signs).
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    val_ds = try_image_folder(os.path.join(data_dir, "val"), eval_tfm)
    test_ds = try_image_folder(os.path.join(data_dir, "test"), eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        if test_ds is not None
        else None
    )

    class_names = load_class_names(data_dir, list(train_ds.classes))
    model = build_model(num_classes=len(train_ds.classes)).to(device)
    trainable, total = trainable_param_count(model)

    print(
        f"Device={device}  classes={len(train_ds.classes)}  "
        f"train={len(train_ds)}"
        + (f"  val={len(val_ds)}" if val_ds is not None else "  val=0")
        + (f"  test={len(test_ds)}" if test_ds is not None else "  test=0")
        + f"  batch={batch_size}  epochs={epochs}"
    )
    print(
        f"Fine-tuning: features[{UNFREEZE_FEATURES_FROM}:] + full classifier  "
        f"({trainable:,}/{total:,} trainable params)"
    )
    if val_loader is None:
        print("No val split found — saving last-epoch weights; test used only for final report.")

    # Lower LR on newly unfrozen conv blocks; higher on classifier head.
    backbone_params = [p for p in model.features[UNFREEZE_FEATURES_FROM:].parameters() if p.requires_grad]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": classifier_params, "lr": lr},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        train_loss = running_loss / n_seen if n_seen else 0.0

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_accuracy={val_acc:.4f}", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": train_ds.classes,
                        "class_names": class_names,
                        "val_accuracy": best_val_acc,
                    },
                    out_file,
                )
        else:
            # 90/10 train-test only: do not peek at test for checkpointing.
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}", flush=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "class_names": class_names,
                    "val_accuracy": None,
                    "train_loss": train_loss,
                    "epoch": epoch + 1,
                },
                out_file,
            )

    if val_loader is not None:
        print(f"Best val accuracy: {best_val_acc:.4f} -> saved to {out_file}")
    else:
        print(f"Saved last-epoch weights -> {out_file}")

    if test_loader is not None:
        if out_file.exists():
            ckpt = torch.load(out_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
        test_acc = evaluate(model, test_loader, device)
        print(f"Test accuracy (saved checkpoint): {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/signs_general_split")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/vgg16_signs_general.pt")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr, args.out)
