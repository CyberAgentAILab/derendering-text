import numpy as np
import random
import os
from typing import Tuple, List
import math
import scipy.signal as ssig
import traceback
import fontTools
from fontTools.ttLib import TTFont
import skia
from src.skia_lib import skia_util as sku
from src.dto.dto_generator import TextGeneratorInputHandler
from ..data_handler import DataHandler


def combert_emsize(target: float, emsize: float, fontSize: int):
    target = target / float(emsize) * fontSize
    return target


def get_ascender(ttfont: fontTools.ttLib.ttFont.TTFont, fontSize: int):
    emsize = ttfont['head'].unitsPerEm
    ascender = combert_emsize(ttfont['hhea'].ascender, emsize, fontSize)
    descender = combert_emsize(ttfont['hhea'].descender, emsize, fontSize)
    lineGap = combert_emsize(ttfont['hhea'].lineGap, emsize, fontSize)
    return ascender, descender, lineGap


class OffsetSampler(object):
    def __init__(self, lang: str):
        self.lang = lang

    def sample_text_type_flags(self, nline: int):
        if nline == 1 and random.random() < 0.5 and self.lang == 'jp':
            vertical_text_flag = 1
        else:
            vertical_text_flag = 0
        if nline == 1 and random.random() < 0.5:
            rotate_text_flag = 1
            angle = random.randint(-60, 60)
        else:
            rotate_text_flag = 0
            angle = 0
        text_type_flags = (vertical_text_flag, rotate_text_flag)
        return text_type_flags, 0

    def get_vertical_text_location_info(
            self,
            font_size: int,
            font_path: str,
            texts: List,
            font: skia.Font):
        height_margin_rate = random.random() * 0.1
        ttfont = TTFont(font_path)
        ascender, descender, lineGap = get_ascender(ttfont, font_size)
        bboxes = []
        text_offsets = []
        char_offsets = []
        for text_index, text in enumerate(texts):
            char_offset = []
            total_text_height = 0
            for char_index, char in enumerate(text):
                textblofb = skia.TextBlob(char, font)
                char_offset_x = font_size * 0.1
                char_offset_y = font_size * 0.1 + total_text_height
                text_spatial_info = sku.get_text_spatial_info(
                    char_offset_y, char_offset_x, font, char)
                (char_top, char_left), (char_height,
                                        char_width), box, _ = text_spatial_info
                total_text_height += char_height + lineGap
                char_offset.append((char_top, char_left))
                bboxes.append(box)
                if char_index == 0:
                    text_offsets.append((char_top, char_left))
            char_offsets.append(char_offset)
        return bboxes, text_offsets, char_offsets

    def get_horizontal_text_location_info(
            self,
            font_size: int,
            font_path: str,
            texts: List,
            font: skia.Font,
            angle: float):
        text_offsets = []
        char_offsets = []
        bboxes = []
        height_margin_rate = random.random() * 0.1
        total_text_height = 0
        for text_index, text in enumerate(texts):
            textblob = skia.TextBlob(text, font)
            text_offset_x = font_size * 0.1
            text_offset_y = font_size * 0.1 + total_text_height
            text_spatial_info = sku.get_text_spatial_info(
                text_offset_y, text_offset_x, font, text)
            (text_top, text_left), (text_height,
                                    text_width), box, char_offsets_x = text_spatial_info
            text_offsets.append((text_top, text_left))  # y,x
            bboxes.append(box)
            char_offsets.append([(text_top, char_offset_x)
                                 for char_offset_x in char_offsets_x])
            total_text_height += text_height + text_height * height_margin_rate
        return bboxes, text_offsets, char_offsets

    def rotate_bb(self, bboxes: np.ndarray, angle: float):
        sin = math.sin(2 * math.pi * (float(angle) / 360.))
        cos = math.cos(2 * math.pi * (float(angle) / 360.))
        rotated_bboxes = np.zeros_like(bboxes)
        for i in range(bboxes.shape[2]):
            for j in range(4):
                y, x = bboxes[:, j, i]
                ry = int(sin * x + 1 * cos * y)
                rx = int(cos * x - 1 * sin * y)
                rotated_bboxes[0, j, i] = ry
                rotated_bboxes[1, j, i] = rx
        return rotated_bboxes

    def catbboxes(self, bboxes: np.ndarray):
        bboxes_cat = np.concatenate(bboxes, axis=0)
        return bboxes_cat

    def postprocess_for_bboxes(self, bboxes: np.ndarray):
        bboxes = self.catbboxes(bboxes)
        bboxes = sku.bb_yxhw2coords(bboxes)
        #bboxes = self.rotate_bb(bboxes, angle)
        return bboxes

    def get_text_location_info(
            self,
            dh: DataHandler,
            text_type_flags: List,
            angle: float):
        font_size, font_path, texts, font = dh.tmp.get_data_for_text_location_info()
        vertical_text_flag, rotate_text_flag = text_type_flags
        if vertical_text_flag == 1:
            bboxes, text_offset, char_offset = self.get_vertical_text_location_info(
                font_size, font_path, texts, font)
        else:
            bboxes, text_offset, char_offset = self.get_horizontal_text_location_info(
                font_size, font_path, texts, font, angle)
        bboxes = self.postprocess_for_bboxes(bboxes)
        return bboxes, text_offset, char_offset

    def get_box_offset(self, mask: np.ndarray, text_size: int):
        text_height, text_width = text_size
        box_mask = np.zeros(
            (int(text_height),
             int(text_width)),
            dtype=np.uint8) + 255
        # position the text within the mask:
        mask = np.clip(mask.copy().astype(np.float), 0, 255)
        box_mask = np.clip(box_mask.copy().astype(np.float), 0, 255)
        box_mask[box_mask > 127] = 1e8
        intersect = ssig.fftconvolve(mask, box_mask[::-1, ::-1], mode='valid')
        safemask = intersect < 1e8
        if not np.any(safemask):  # no collision-free position:
            return None, box_mask
        minloc = np.transpose(np.nonzero(safemask))
        offset_y, offset_x = minloc[np.random.choice(minloc.shape[0]), :]
        return (offset_y, offset_x), box_mask

    def get_text_size_from_bb(self, bboxes: np.ndarray):
        text_height = np.max(bboxes[0, :, :]) - np.min(bboxes[0, :, :])
        text_width = np.max(bboxes[1, :, :]) - np.min(bboxes[1, :, :])
        return text_height, text_width

    def sample(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # mask for text placement
        mask = ih.collision_mask
        # load the number of text line
        text_line_num = dh.tmp.get_text_line_num()
        # get parameters for text types
        text_type_flags, angle = self.sample_text_type_flags(text_line_num)
        # get text location information
        bboxes, text_offsets, char_offsets = self.get_text_location_info(
            dh, text_type_flags, angle)
        # get text size from bbox
        text_size = self.get_text_size_from_bb(bboxes)
        try:
            box_offset, box_mask = self.get_box_offset(mask, text_size)
        except BaseException:
            box_offset = None
            # traceback.print_exc()
        if box_offset is None:
            return None
        else:
            # add offsets of the text location
            bboxes = sku.add_offset_bboxes(bboxes, box_offset)
            text_offsets = sku.add_offset_coords(text_offsets, box_offset)
            for char_index, char_offset in enumerate(char_offsets):
                char_offsets[char_index] = sku.add_offset_coords(
                    char_offset, box_offset)
            # set sampled data to data handler
            dh.tmp.set_offset_sampler_data(
                text_type_flags, angle, text_offsets, char_offsets, text_size)
        return text_type_flags, angle, bboxes, text_offsets, char_offsets
