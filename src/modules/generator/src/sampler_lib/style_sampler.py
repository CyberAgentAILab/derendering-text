import numpy as np
import random
import os
from src.dto.dto_generator import TextGeneratorInputHandler
from ..data_handler import DataHandler
from src.skia_lib import skia_util as sku
from src.skia_lib import skia_paintor as skp


class StyleSampler(object):
    def get_style_params(
            self,
            font_size: int,
            offset_x: int,
            offset_y: int,
            text_width: int,
            text_height: int):
        fill_param = skp.get_fill_param()
        shadow_param = skp.get_shadow_param(font_size)
        stroke_param = skp.get_stroke_param(font_size)
        grad_param = skp.get_gradation_param(
            offset_x, offset_y, text_width, text_height)
        return (shadow_param, fill_param, grad_param, stroke_param)

    def sample(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # get data for style sampler
        font_size, offset_y, offset_x, text_height, text_width = dh.tmp.get_data_for_style_sampler()
        # get visibility flag
        visibility_flags = skp.get_visibility_flag()
        effect_params = self.get_style_params(
            font_size, offset_x, offset_y, text_width, text_height)
        # set style sampler data
        dh.tmp.set_style_sampler_data(visibility_flags, effect_params)
        return visibility_flags, effect_params
