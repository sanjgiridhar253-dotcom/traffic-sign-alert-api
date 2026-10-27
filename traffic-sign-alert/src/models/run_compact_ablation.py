"""Train the paper's compact CNN three ways on the signs_general split.

The official GTSRB archive was not reachable, so this uses the 52-class
images already in data/processed/signs_general_split. The test folder is
never used for training or early stopping. A stratified 20% of the training
images, seed 42, is the validation set.

Variants, matching Table 4:
  baseline: no augmentation, no batch norm, no dropout
  aug:      augmentation, no batch norm, no dropout
  final:    augmentation, batch norm, dropout 0.5
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/signs_general_split"
OUT = ROOT / "reports/section6"
SEED = 42
EPOCHS = 50
PATIENCE = 10
BATCH = 64
LR = 1e-3


class CompactCNN(nn.Module):
    def __init__(self, num_classes: int, use_bn: bool, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32, use_bn),
            self._block(32, 64, use_bn),
            self._block(64, 128, use_bn),
        )
        head = [nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True)]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @staticmethod
    def _block(in_ch: int, out_ch: int, use_bn: bool) -> nn.Sequential:
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        return nn.Sequential(*layers)


class ListDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def stratified_split(samples: list[tuple[str, int]], val_frac: float, seed: int):
    by_class: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        rows = by_class[label][:]
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * val_frac))) if len(rows) > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def run_variant(name: str, use_bn: bool, dropout: float, augment: bool, train_samples, val_samples, num_classes, device):
    torch.manual_seed(SEED)
    eval_tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    if augment:
        train_tfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        train_tfm = eval_tfm
    train_loader = DataLoader(ListDataset(train_samples, train_tfm), batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(ListDataset(val_samples, eval_tfm), batch_size=BATCH, shuffle=False, num_workers=2)
    model = CompactCNN(num_classes, use_bn, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_val = -1.0
    best_epoch = 0
    bad = 0
    weights = OUT / f"{name}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = n = 0
        for x, y in tqdm(train_loader, desc=f"{name} {epoch}/{EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = running / n
        val_acc = evaluate(model, val_loader, device)
        sched.step(train_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"{name} epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "val_accuracy": best_val, "epoch": epoch}, weights)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"{name} early stop at epoch {epoch}", flush=True)
                break
    return {
        "name": name,
        "best_val_accuracy": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"],
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
        "weights": str(weights),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def test_metrics(weights: Path, use_bn: bool, dropout: float, test_samples, num_classes, device) -> dict:
    model = CompactCNN(num_classes, use_bn, dropout).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    loader = DataLoader(ListDataset(test_samples, tfm), batch_size=BATCH, shuffle=False, num_workers=2)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            ys.append(y)
            ps.append(pred)
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    return {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = datasets.ImageFolder(DATA / "train")
    test_ds = datasets.ImageFolder(DATA / "test")
    train_samples, val_samples = stratified_split(train_ds.samples, 0.2, SEED)
    variants = [
        ("baseline", False, 0.0, False),
        ("aug", False, 0.0, True),
        ("final", True, 0.5, True),
    ]
    results = []
    for name, use_bn, dropout, augment in variants:
        summary = run_variant(name, use_bn, dropout, augment, train_samples, val_samples, len(train_ds.classes), device)
        summary.update(test_metrics(Path(summary["weights"]), use_bn, dropout, test_ds.samples, len(test_ds.classes), device))
        results.append(summary)
        print(
            f"{name} best_val={summary['best_val_accuracy']:.4f} "
            f"test={summary['test_accuracy']:.4f} macro_f1={summary['test_macro_f1']:.4f}",
            flush=True,
        )

    final = next(item for item in results if item["name"] == "final")
    epochs = [row["epoch"] for row in final["history"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["train_loss"] for row in final["history"]], label="train loss")
    ax.plot(epochs, [row["val_acc"] for row in final["history"]], label="val accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "final_curves.png", dpi=150)
    plt.close(fig)

    payload = {
        "dataset": "signs_general_split",
        "classes": len(train_ds.classes),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "test_images": len(test_ds.samples),
        "seed": SEED,
        "epochs_cap": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH,
        "lr": LR,
        "variants": [{k: v for k, v in item.items() if k != "history"} | {"history": item["history"]} for item in results],
    }
    (OUT / "ablation.json").write_text(json.dumps(payload, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
