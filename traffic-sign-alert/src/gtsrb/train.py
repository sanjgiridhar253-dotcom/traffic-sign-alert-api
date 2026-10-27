"""
train.py

Trains the IEEE-paper compact CNN on the track-wise GTSRB split.

Usage:
    python -m src.gtsrb.train \
        --data data/processed/gtsrb_split \
        --out models/gtsrb_compact.pt \
        --epochs 50 --batch-size 64 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.gtsrb.model import CompactTrafficSignCNN, param_count


def build_loaders(data_dir: str, batch_size: int, workers: int):
    # Semantics-preserving aug: NO horizontal flip (IEEE Section 4.1)
    train_tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


def train(data_dir: str, out_path: str, epochs: int, batch_size: int, lr: float,
          patience: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 0 if device.type == "cpu" else 2

    train_ds, val_ds, train_loader, val_loader = build_loaders(data_dir, batch_size, workers)
    model = CompactTrafficSignCNN(num_classes=len(train_ds.classes)).to(device)
    print(f"Device={device}  classes={len(train_ds.classes)}  "
          f"train={len(train_ds)}  val={len(val_ds)}  params={param_count(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.CrossEntropyLoss()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = 0.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = running / n
        val_acc = evaluate(model, val_loader, device)
        sched.step(train_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_accuracy={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "val_accuracy": best_val,
                    "num_classes": len(train_ds.classes),
                },
                out,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    hist_path = out.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"Best val accuracy: {best_val:.4f} -> {out}")
    print(f"History saved: {hist_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/gtsrb_split")
    p.add_argument("--out", default="models/gtsrb_compact.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not Path(args.data, "train").exists():
        raise SystemExit(
            f"Missing {args.data}/train. MANUAL: run "
            "`python -m src.gtsrb.prepare_data` first (after placing GTSRB zips)."
        )
    train(args.data, args.out, args.epochs, args.batch_size, args.lr, args.patience, args.seed)
