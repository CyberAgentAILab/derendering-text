from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv3x3_bn_relu
from ..dto.dto_model import TextInfo
from .hourglass import HourGlassNet
from .textparser import TextParser


class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3_bn_relu(256, 128, stride=2)
        self.conv2 = conv3x3_bn_relu(128, 128, stride=2)
        self.conv3 = conv3x3_bn_relu(128, 128, stride=2)
        self.conv_upsample = conv3x3_bn_relu(256 + 128, 256)

    def forward(self, feat):
        featdown = self.conv1(feat)  # 80
        featdown = self.conv2(featdown)  # 40
        featdown = self.conv3(featdown)  # 20
        featup = F.interpolate(
            featdown, (feat.shape[2:4]), mode="bilinear", align_corners=False
        )
        featup = self.conv_upsample(torch.cat((feat, featup), 1))
        return featdown, featup


class Vectorization(nn.Module):
    def __init__(self, text_pool_num: int=10, dev: torch.device = None):
        super().__init__()
        self.backbone = HourGlassNet(3, [256, 256, 256, 512], [2, 2, 2, 2])
        self.down = Down()
        self.text_parser = TextParser(text_pool_num=text_pool_num, dev=dev)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_orig: torch.Tensor,
                text_instance_mask: Optional[torch.Tensor] = None) -> Tuple[TextInfo, int]:
        features = self.backbone(x)
        _, features = self.down(features)
        text_info = self.text_parser(features, x_orig, text_instance_mask)
        return text_info, features
