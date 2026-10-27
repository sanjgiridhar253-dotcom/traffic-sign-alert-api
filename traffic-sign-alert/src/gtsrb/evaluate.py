"""
evaluate.py

IEEE Section 4.4 / 6: top-1 accuracy, macro precision/recall/F1, confusion matrix.

Usage:
    python -m src.gtsrb.evaluate \
        --data data/processed/gtsrb_split \
        --weights models/gtsrb_compact.pt \
        --split test \
        --out reports/gtsrb_eval
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.gtsrb.model import CompactTrafficSignCNN


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu()
        ys.append(y)
        ps.append(pred)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/gtsrb_split")
    p.add_argument("--weights", default="models/gtsrb_compact.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="reports/gtsrb_eval")
    args = p.parse_args()

    split_dir = Path(args.data) / args.split
    if not split_dir.exists():
        raise SystemExit(
            f"Missing {split_dir}. MANUAL: prepare GTSRB split / ensure test GT was extracted."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.ImageFolder(str(split_dir), transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = CompactTrafficSignCNN(num_classes=ckpt.get("num_classes", len(ds.classes))).to(device)
    model.load_state_dict(ckpt["model_state"])

    y_true, y_pred = collect_preds(model, loader, device)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": args.split,
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "val_accuracy_from_ckpt": ckpt.get("val_accuracy"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "classification_report.json").write_text(json.dumps(report, indent=2))
    np.save(out / "confusion_matrix.npy", cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion matrix ({args.split})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote reports to {out}")
    print("Fill IEEE Table 4 with accuracy / macro-F1 from metrics.json")


if __name__ == "__main__":
    main()
