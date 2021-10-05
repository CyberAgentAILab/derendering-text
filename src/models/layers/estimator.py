import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import conv3x3_bn_relu


class AlphaEstimator(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.alpha_conv1 = conv3x3_bn_relu(in_channels, bottleneck_channels)
        self.alpha_conv2 = conv3x3_bn_relu(bottleneck_channels, 32)
        self.alpha_ref_conv1 = conv3x3_bn_relu(32, 32)
        self.alpha_ref_conv2 = conv3x3_bn_relu(32, 32)
        self.rgb_conv1 = conv3x3_bn_relu(3, 32)
        self.rgb_text = conv3x3_bn_relu(32 * 2, 32)
        self.alpha_estimator = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor):
        feat = self.alpha_conv1(feat)
        feat = self.alpha_conv2(feat)
        feat = F.interpolate(feat,
                             rgb.shape[2:4],
                             mode="bilinear",
                             align_corners=False)
        rgb_feat = self.rgb_conv1(rgb)
        feat = torch.cat((feat, rgb_feat), 1)
        feat = self.rgb_text(feat)
        alpha_ref = self.alpha_ref_conv1(feat)
        alpha_ref = self.alpha_ref_conv2(feat)
        alpha_ref = self.alpha_estimator(alpha_ref)
        return torch.sigmoid(alpha_ref)


class FontSizeEstimator(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv = conv3x3_bn_relu(in_channels, bottleneck_channels)
        self.font_size_estimator = nn.Conv2d(
            bottleneck_channels, 1, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat: torch.Tensor):
        feat = self.conv(feat)
        font_size_pred = self.font_size_estimator(feat)
        return font_size_pred


class FontEstimator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            bottleneck_channels: int,
            font_num: int = 100):
        super().__init__()
        self.conv = conv3x3_bn_relu(in_channels, bottleneck_channels)
        self.font_estimator = nn.Conv2d(
            bottleneck_channels, font_num, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat: torch.Tensor):
        feat = self.conv(feat)
        font_pred = self.font_estimator(feat)
        return font_pred


class EffectVisibilityEstimator(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv = conv3x3_bn_relu(in_channels, bottleneck_channels)
        self.stroke_visibility = nn.Conv2d(
            bottleneck_channels, 2, kernel_size=1)
        self.shadow_visibility = nn.Conv2d(
            bottleneck_channels, 2, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat: torch.Tensor):
        feat = self.conv(feat)
        shadow_visibility = self.shadow_visibility(feat)
        stroke_visibility = self.stroke_visibility(feat)
        return shadow_visibility, stroke_visibility


class EffectParamEstimator(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.stroke_param_division_num = 5
        self.shadow_param_num = 5
        self.conv = conv3x3_bn_relu(in_channels, bottleneck_channels)
        # classification for prerendered alpha
        self.stroke_param = nn.Conv2d(
            bottleneck_channels, self.stroke_param_division_num, kernel_size=1
        )
        # numerical value prediction
        self.shadow_param_sig = nn.Conv2d(
            bottleneck_channels, 2, kernel_size=1)
        self.shadow_param_tanh = nn.Conv2d(
            bottleneck_channels, 2, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat: torch.Tensor):
        feat = self.conv(feat)
        # mse
        shadow_param_sig = torch.sigmoid(self.shadow_param_sig(feat))
        shadow_param_tanh = torch.tanh(self.shadow_param_tanh(feat))
        stroke_param = self.stroke_param(feat)
        return shadow_param_sig, shadow_param_tanh, stroke_param
