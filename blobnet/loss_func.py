from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedGaussianLoss(nn.Module):
    """
    Loss for normalized Gaussian heatmap regression.

    The network predicts one heatmap channel per image. This loss mixes global
    mean-squared error with an up-weighted peak-region MSE so the model learns
    both smooth backgrounds and precise atom centers.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        peak_weight: float = 0.5,
        threshold: float = 0.1,
        peak_boost: float = 5.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.peak_weight = float(peak_weight)
        self.threshold = float(threshold)
        self.peak_boost = float(peak_boost)
        self.from_logits = bool(from_logits)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            prediction = torch.sigmoid(prediction)

        mse = F.mse_loss(prediction, target)
        peak_mask = target > self.threshold
        weight_map = torch.ones_like(target)
        weight_map[peak_mask] = self.peak_boost
        weighted_mse = ((prediction - target) ** 2 * weight_map).mean()
        return self.mse_weight * mse + self.peak_weight * weighted_mse
