import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTUNet(nn.Module):
    def __init__(self):
        super(MNISTUNet, self).__init__()

        # Encoder
        self.down1 = self.conv_block(1, 16)  # → [16, 28, 28]
        self.pool = nn.MaxPool2d(2)  # → [16, 14, 14]

        # Bottleneck
        self.bottleneck = self.conv_block(16, 32)  # → [32, 14, 14]

        # Decoder
        self.up = nn.ConvTranspose2d(32, 16, 2, 2)  # → [16, 28, 28]
        self.dec = self.conv_block(32, 16)  # → [16, 28, 28]

        # Final output
        self.out = nn.Conv2d(16, 1, 1)  # → [1, 28, 28]

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # → same H, W
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # → same H, W
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [1, 28, 28]
        x1 = self.down1(x)  # → [16, 28, 28]
        x2 = self.pool(x1)  # → [16, 14, 14]

        x3 = self.bottleneck(x2)  # → [32, 14, 14]

        x4 = self.up(x3)  # → [16, 28, 28]
        x = torch.cat([x4, x1], dim=1)  # → [32, 28, 28]
        x = self.dec(x)  # → [16, 28, 28]

        out = self.out(x)  # → [1, 28, 28]
        return out
