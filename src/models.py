from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainCNN(nn.Module):
    """
    Simple CNN for IAM vs Emuru domain classification.

    Input:  (B, 1, 64, W)
    Output: (B, 2) logits, where:
        class 0 = IAM / genuine
        class 1 = Emuru / fake
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # no pooling here; global pooling later
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, W)
        feat = self.conv_block(x)     # (B, 128, H', W')
        feat = feat.mean(dim=[2, 3])  # global avg pool -> (B, 128)
        logits = self.fc(feat)        # (B, 2)
        return logits
