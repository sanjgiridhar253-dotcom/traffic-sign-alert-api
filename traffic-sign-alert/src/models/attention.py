"""
attention.py

Implements CBAM (Convolutional Block Attention Module) - a lightweight
channel + spatial attention block you can drop into a CNN backbone. This is
what turns your "DL CNN model" into "DL CNN model WITH Attention" from the
architecture slide, and it's small enough to add without retraining from
scratch (fine-tune on top of your Step 3 YOLO weights).

Two ways to use this, cheapest first:

1. Standalone re-weighting (no architecture surgery): wrap CBAM around the
   feature map YOLO already extracts, before the detection head runs. Lower
   risk, faster to test, and what `apply_attention_to_features()` below does.

2. True backbone injection: splice CBAM blocks into specific YOLOv8 layers
   via Ultralytics' custom-yaml model config. More involved - only worth it
   once approach 1 shows attention helps.

Either way, evaluate with the SAME validation set and script your model
trained without attention (Step 3's plain YOLO run) so you have a real
before/after number for your results section, not just an assumption.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """"Which feature channels matter" - e.g. edge/shape channels that
    respond strongly to sign borders should be up-weighted."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # (B, C, 1, 1) gate


class SpatialAttention(nn.Module):
    """"Where in the frame matters" - pushes focus toward the sign's
    location and away from background clutter (trees, other vehicles)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(gate)  # (B, 1, H, W) gate


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


def apply_attention_to_features(feature_map: torch.Tensor, cbam: CBAM) -> torch.Tensor:
    """Re-weight a feature map extracted from YOLO's backbone before it
    reaches the detection head. `feature_map` shape: (B, C, H, W)."""
    return cbam(feature_map)


if __name__ == "__main__":
    # quick shape sanity check
    dummy = torch.randn(2, 256, 40, 40)  # batch=2, 256 channels, 40x40 feature map
    cbam = CBAM(channels=256)
    out = apply_attention_to_features(dummy, cbam)
    assert out.shape == dummy.shape
    print("CBAM output shape:", out.shape, "(matches input - attention re-weights, doesn't resize)")
