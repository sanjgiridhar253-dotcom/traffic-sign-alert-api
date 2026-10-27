"""
Compact CNN from the IEEE paper (~0.63M parameters).

Architecture (Table 2):
  Conv32 -> pool -> Conv64 -> pool -> Conv128 -> pool -> FC256 -> FC43
"""
import torch
import torch.nn as nn


class CompactTrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = 43, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CompactTrafficSignCNN()
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print("out", tuple(y.shape), "params", param_count(m))
