from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two-layer convolutional block used throughout the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for single-channel heatmap regression."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_filters: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(num_filters) < 2:
            raise ValueError("UNet expects at least two filter stages.")

        self.encoders = nn.ModuleList()
        for index, filters in enumerate(num_filters[:-1]):
            self.encoders.append(
                EncoderBlock(
                    input_channels if index == 0 else int(num_filters[index - 1]),
                    int(filters),
                )
            )

        self.bottleneck = ConvBlock(int(num_filters[-2]), int(num_filters[-1]))
        self.dropout = nn.Dropout(dropout)

        reversed_filters = list(reversed([int(value) for value in num_filters]))
        self.decoders = nn.ModuleList(
            DecoderBlock(reversed_filters[index], reversed_filters[index + 1])
            for index in range(len(reversed_filters) - 1)
        )
        self.final_conv = nn.Conv2d(int(num_filters[0]), num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.dropout(self.bottleneck(x))
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.final_conv(x)


Unet = UNet


def available_model_variants() -> list[str]:
    return ["unet"]


def build_unet(
    input_channels: int = 1,
    num_classes: int = 1,
    num_filters: Sequence[int] | None = None,
    dropout: float = 0.1,
) -> UNet:
    filters = list(num_filters) if num_filters is not None else [32, 64, 128, 256]
    return UNet(
        input_channels=input_channels,
        num_classes=num_classes,
        num_filters=filters,
        dropout=dropout,
    )


def build_model(
    variant: str,
    input_channels: int,
    num_classes: int,
    num_filters: Iterable[int] | None = None,
    dropout: float = 0.1,
    hourglass_depth: int | None = None,
):
    del hourglass_depth
    if variant.lower() != "unet":
        raise ValueError(
            "This cleaned pipeline supports only the U-Net workflow. "
            f"Requested variant '{variant}'."
        )
    return build_unet(
        input_channels=input_channels,
        num_classes=num_classes,
        num_filters=list(num_filters) if num_filters is not None else None,
        dropout=dropout,
    )
