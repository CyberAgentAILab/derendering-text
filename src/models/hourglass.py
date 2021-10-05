from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        **kwargs):
    blocks = []
    blocks.append(Residual(in_channels, out_channels))
    for _ in range(1, num_blocks):
        blocks.append(Residual(out_channels, out_channels, **kwargs))
    return nn.Sequential(*blocks)


def _make_layer_revr(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        **kwargs):
    blocks = []
    for _ in range(num_blocks - 1):
        blocks.append(Residual(in_channels, in_channels, **kwargs))
    blocks.append(Residual(in_channels, out_channels, **kwargs))
    return nn.Sequential(*blocks)


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, eps=1e-5),
            )
        else:
            self.skip = None
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        b1 = self.conv_2(self.conv_1(x))
        if self.skip is None:
            return self.out_relu(b1 + x)
        else:
            return self.out_relu(b1 + self.skip(x))


class HourGlassBlock(nn.Module):
    def __init__(self, n: int, channels: int, blocks: int):
        super().__init__()
        self.up_1 = _make_layer(channels[0], channels[0], blocks[0])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low_1 = _make_layer(channels[0], channels[1], blocks[0])
        if n <= 1:
            self.low_2 = _make_layer(channels[1], channels[1], blocks[1])
        else:
            self.low_2 = HourGlassBlock(n - 1, channels[1:], blocks[1:])
        self.low_3 = _make_layer_revr(channels[1], channels[0], blocks[0])

    def forward(self, x: torch.Tensor):
        def upsample(input):
            return F.interpolate(
                input, scale_factor=2, mode="bilinear", align_corners=True
            )

        up_1 = self.up_1(x)
        low = self.low_3(self.low_2(self.low_1(self.pool(x))))
        return upsample(low) + up_1


class HourGlassNet(nn.Module):
    def __init__(self, n: int, channels: List[int], blocks: List[int]):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128, eps=1e-5),
            nn.ReLU(),
            Residual(128, 256, stride=2),
        )
        hourglass_blocks = []
        for _ in range(2):
            hourglass_blocks.append(HourGlassBlock(n, channels, blocks))
        self.hourglass_blocks = nn.Sequential(*hourglass_blocks)

    def forward(self, x: torch.Tensor):
        return self.hourglass_blocks(self.pre(x))
