from typing import OrderedDict

import torch.nn as nn


def conv3x3_bn_relu(
    in_channels: int,
    out_channels: int,
    dilation: int = 1,
    kernel_size: int = 3,
    stride: int = 1,
    use_spectral_norm: bool = False,
):
    if dilation == 0:
        dilation = 1
        padding = 0
    else:
        padding = dilation
    if not use_spectral_norm:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("relu", nn.ReLU()),
                ]
            )
        )
    else:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.utils.spectral_norm(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=False,
                            )
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("relu", nn.ReLU()),
                ]
            )
        )
