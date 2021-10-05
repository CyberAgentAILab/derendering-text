import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import conv3x3_bn_relu


class WordDetector(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.word_det_conv_final = conv3x3_bn_relu(
            in_channels, bottleneck_channels)
        self.word_fg_feat = conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels)
        self.word_regression_feat = conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.word_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.word_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feat = self.word_det_conv_final(x)
        pred_word_fg = self.word_fg_pred(self.word_fg_feat(feat))
        word_regression_feat = self.word_regression_feat(feat)
        pred_word_tblr = F.relu(
            self.word_tblr_pred(word_regression_feat)) * 10.0
        pred_word_orient = self.orient_pred(word_regression_feat)
        return pred_word_fg, pred_word_tblr, pred_word_orient


class CharDetector(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.character_det_conv_final = conv3x3_bn_relu(
            in_channels, bottleneck_channels
        )
        self.char_fg_feat = conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels)
        self.char_hm_feat = conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels)
        self.char_regression_feat = conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.char_hm_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)
        self.char_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feat = self.character_det_conv_final(x.detach())

        pred_char_fg = self.char_fg_pred(self.char_fg_feat(feat))
        char_regression_feat = self.char_regression_feat(feat)
        pred_char_tblr = F.relu(
            self.char_tblr_pred(char_regression_feat)) * 10.0
        pred_char_orient = self.orient_pred(char_regression_feat)

        return pred_char_fg, pred_char_tblr, pred_char_orient


class CharRecognizer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            bottleneck_channels: int,
            num_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            conv3x3_bn_relu(in_channels, bottleneck_channels),
            conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
            conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
        )
        self.classifier = nn.Conv2d(
            bottleneck_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor):
        feat = self.body(feat)
        pred = self.classifier(feat)
        return pred


class InnerOCR(nn.Module):
    def __init__(
        self, e_channel: int = 256, t_channel: int = 128, num_classes: int = 94
    ):
        super().__init__()
        self.word_detector = WordDetector(e_channel, t_channel)
        self.char_detector = CharDetector(e_channel, t_channel)
        self.char_recognizer = CharRecognizer(
            e_channel, t_channel, num_classes)

    def forward(self, features_pix):
        word_out = self.word_detector(features_pix)
        char_out = self.char_detector(features_pix)
        recog_out = self.char_recognizer(features_pix)
        return (word_out, char_out, recog_out)
