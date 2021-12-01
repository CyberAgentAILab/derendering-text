import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..dto.dto_model import TextInfo
from .layers.estimator import (
    AlphaEstimator,
    EffectParamEstimator,
    EffectVisibilityEstimator,
    FontEstimator,
    FontSizeEstimator,
)
from .layers.geometry.bbox import get_bb_level_features, get_bbox
from .layers.geometry.shape import convert_shape
from .layers.inner_ocr import InnerOCR


class TextParser(nn.Module):
    def __init__(
            self,
            e_channel: int = 256,
            t_channel: int = 128,
            text_pool_num: int = 10,
            dev: torch.device = None):
        super().__init__()
        self.ocr = InnerOCR()
        self.effect_param = EffectParamEstimator(e_channel, t_channel)
        self.effect_visibility = EffectVisibilityEstimator(
            e_channel, t_channel)
        self.font = FontEstimator(e_channel, t_channel)
        #elf.font_size = FontSizeEstimator(e_channel, t_channel)
        self.alpha = AlphaEstimator(e_channel, t_channel)
        self.text_pool_num = text_pool_num
        self.dev = dev

    def forward(self, features_pix: torch.Tensor, img_org: torch.Tensor,
                text_instance_mask: Optional[torch.Tensor] = None) -> TextInfo:
        # OCR model
        ocr_outs = self.ocr(features_pix)
        if text_instance_mask is None:
            bbox_information = get_bbox(
                ocr_outs, (img_org.shape[2], img_org.shape[3]))
            text_instance_mask = bbox_information.get_text_instance_mask()
            text_instance_mask = torch.from_numpy(
                text_instance_mask).to(self.dev)
        else:
            bbox_information = None
        features_box, text_num = get_bb_level_features(
            features_pix, text_instance_mask, self.training, self.text_pool_num, self.dev
        )
        # Style Parse model
        effect_visibility_outs = self.effect_visibility(features_box)
        effect_param_outs = self.effect_param(features_box)
        font_outs = self.font(features_box)
        #font_size_outs = self.font_size(features_box)
        alpha_outs = self.alpha(features_pix, img_org)

        batch_num = features_pix.shape[0]
        effect_visibility_outs = convert_shape(
            effect_visibility_outs, batch_num, text_num, self.training, self.text_pool_num
        )
        effect_param_outs = convert_shape(
            effect_param_outs, batch_num, text_num, self.training, self.text_pool_num)
        font_outs = convert_shape([font_outs], batch_num, text_num, self.training, self.text_pool_num)[0]
        #font_size_outs = convert_shape([font_size_outs], batch_num, text_num, self.training, self.text_pool_num)[0]
        return TextInfo(
            ocr_outs,
            bbox_information,
            effect_visibility_outs,
            effect_param_outs,
            font_outs,
            None,
            alpha_outs,
        )
