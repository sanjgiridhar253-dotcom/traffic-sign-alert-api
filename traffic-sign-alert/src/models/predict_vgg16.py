"""
predict_vgg16.py

Load the trained signs_general VGG16 checkpoint and classify a crop / image.

Usage:
    python -m src.models.predict_vgg16 --image path/to/crop.png
    python -m src.models.predict_vgg16 --random-test
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_vgg16(weights_path: str, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    n_classes = len(ckpt["classes"])
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt, device


def default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def classify_pil(model, ckpt, device, image: Image.Image, tfm=None) -> dict:
    tfm = tfm or default_transform()
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "folder_class": ckpt["classes"][idx],
        "class_name": ckpt["class_names"][idx],
        "confidence": float(probs[idx]),
        "class_index": idx,
    }


def classify_path(weights_path: str, image_path: str) -> dict:
    model, ckpt, device = load_vgg16(weights_path)
    img = Image.open(image_path)
    out = classify_pil(model, ckpt, device, img)
    out["image"] = image_path
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/vgg16_signs_general.pt")
    p.add_argument("--image", default=None)
    p.add_argument("--random-test", action="store_true")
    args = p.parse_args()

    if args.random_test:
        root = Path("data/processed/signs_general_split/test")
        imgs = list(root.rglob("*.png")) + list(root.rglob("*.jpg"))
        if not imgs:
            raise SystemExit("No test images found.")
        args.image = str(random.choice(imgs))

    if not args.image:
        raise SystemExit("Provide --image or --random-test")

    result = classify_path(args.weights, args.image)
    print(result)


if __name__ == "__main__":
    main()
