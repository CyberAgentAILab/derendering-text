import numpy as np
from src.dto.dto_generator import TextGeneratorInputHandler
from .data_handler import DataHandler
from .sampler_lib.text_font_sampler import TextFontSampler
from .sampler_lib.offset_sampler import OffsetSampler
from .sampler_lib.style_sampler import StyleSampler
from logzero import logger as log


class Sampler(object):
    def __init__(self, load_data_path: str, lang: str):
        self.text_font_sampler = TextFontSampler(load_data_path, lang)
        self.offset_sampler = OffsetSampler(lang)
        self.style_sampler = StyleSampler()
        self.max_shrink_trials = 50
        self.min_font_size = 10
        self.max_font_size = 120

    def sample_font_height_px(self, h_min: int, h_max: int):
        h_range = h_max - h_min
        font_size = np.floor(h_min + h_range * np.random.rand())
        return font_size

    def robust_HW(self, mask: np.ndarray):
        m = mask.copy()
        m = (~mask.astype(np.uint8)).astype('float') / 255
        rH = np.sum(m, axis=0)
        rW = np.sum(m, axis=1)
        rH = np.median(rH[rH != 0])
        rW = np.median(rW[rW != 0])
        return rH, rW

    def get_max_font_size(self, mask: np.ndarray):
        max_font_size, _ = self.robust_HW(mask)
        max_font_size = min(max_font_size, self.max_font_size)
        return max_font_size

    def sample(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # get blank region size from collision mask
        max_font_size = self.get_max_font_size(ih.collision_mask)
        mask_size = ih.collision_mask.shape[0:2]
        # store mask size data
        dh.tmp.set_mask_size(mask_size)
        i = 0
        # initialize sampling flag
        dh.sample_initialize()
        placed = False
        while i < self.max_shrink_trials and max_font_size > self.min_font_size:
            # sample font size
            font_size = self.sample_font_height_px(
                self.min_font_size, max_font_size)
            max_font_size = font_size  # update max font size
            # store font size
            dh.tmp.set_font_size(font_size)
            # sample text and font information
            text_font_data = self.text_font_sampler.sample(ih, dh)
            if text_font_data is None:
                i += 1
                continue
            # sample text offset information including character and text
            # bounding box
            offsets_data = self.offset_sampler.sample(ih, dh)
            if offsets_data is None:
                # re-sampling with shrinked fontsize if the sampler fails to
                # locate text in blank space
                i += 1  # The numer of the max re-sampling iteration  is 50
                continue
            else:
                # success locating text in blank space
                # sample text style information
                style_data = self.style_sampler.sample(ih, dh)
                # set sampled data in data handler
                dh.set_sampled_data(text_font_data, offsets_data, style_data)
                placed = True
                break
        return placed
