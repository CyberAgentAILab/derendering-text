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
    def __init__(self, lang: str, rotate_text_option: bool = False):
        self.lang = lang
        self.rotate_text_option = rotate_text_option

    def sample_text_type_flags(self, nline: int):
        if nline == 1 and random.random() < 0.5 and self.lang == 'jp':
            vertical_text_flag = 1
        else:
            vertical_text_flag = 0
        if self.rotate_text_option is True and nline == 1 and random.random() < 0.5:
            rotate_text_flag = 1
            angle = random.randint(-45, 45)
        else:
            rotate_text_flag = 0
            angle = 0
        text_type_flags = (vertical_text_flag, rotate_text_flag)
        return text_type_flags, angle

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

    def rotate_bb(self, bboxes: np.ndarray, text_offset:List, angle: float):
        sin = math.sin(2 * math.pi * (float(angle) / 360.))
        cos = math.cos(2 * math.pi * (float(angle) / 360.))
        offset_y, offset_x = text_offset[0]
        rotated_bboxes = np.zeros_like(bboxes)
        for i in range(bboxes.shape[2]):
            for j in range(4):
                y, x = bboxes[:, j, i]
                ry = int(sin * x + 1 * cos * y + offset_y - offset_x*sin-offset_y*cos)
                rx = int(cos * x - 1 * sin * y + offset_x - offset_x*cos+offset_y*sin)
                rotated_bboxes[0, j, i] = ry
                rotated_bboxes[1, j, i] = rx
        return rotated_bboxes
    
    def rotate_offsets(self, offsets:List, baes_offset: Tuple, angle: float):
        sin = math.sin(2 * math.pi * (float(angle) / 360.))
        cos = math.cos(2 * math.pi * (float(angle) / 360.))
        base_offset_y, base_offset_x = baes_offset
        rotated_offsets = []
        for i, r in enumerate(offsets):
            y, x = r
            ry = int(sin * x + 1 * cos * y + base_offset_y - base_offset_x*sin-base_offset_y*cos)
            rx = int(cos * x - 1 * sin * y + base_offset_x - base_offset_x*cos+base_offset_y*sin)
            rotated_offsets.append((ry,rx))
        return rotated_offsets

    def catbboxes(self, bboxes: np.ndarray):
        bboxes_cat = np.concatenate(bboxes, axis=0)
        return bboxes_cat

    def postprocess_for_bboxes(self, bboxes: np.ndarray, angle: float):
        bboxes = self.catbboxes(bboxes)
        bboxes = sku.bb_yxhw2coords(bboxes)
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
        bboxes_topline = self.postprocess_for_bboxes([bboxes[0]], angle)
        bboxes_all = self.postprocess_for_bboxes(bboxes, angle)
        return bboxes_all, bboxes_topline, text_offset, char_offset

    def get_box_offset(self, mask: np.ndarray, text_size: Tuple, text_size_org: Tuple, bboxes_topline: np.array, angle: float):
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
        if angle != 0:
            text_height_org, text_width_org = text_size_org
            # This function supports the range of angle: [-90, 90]
            # A pivot of rotation is at the lower left of a text
            if angle > 0:
                add_offset_x, add_offset_y = self.get_add_offset(
                    text_height_org, 
                    text_width_org,
                    text_height, 
                    text_width,
                    bboxes_topline,
                    angle
                )
                offset_x += add_offset_x
                offset_y += add_offset_y
            elif angle < 0:
                offset_x += abs(text_width - text_width_org)
                offset_y += abs(text_height - text_height_org)
        return (offset_y, offset_x), box_mask
    
    def get_add_offset(
        self, 
        text_height_org: int, 
        text_width_org: int, 
        text_height: int, 
        text_width: int, 
        bboxes_topline: np.array, 
        angle: float
    ):
        text_size_topline = self.get_text_size_from_bb(bboxes_topline,angle)
        text_height_topline, text_width_topline = text_size_topline
        text_height_topline_rotate, text_width_topline_rotate = self.get_rotated_text_size(text_size_topline, angle)
        # there should be different processing in one line text and multiple line texts
        if text_height == text_height_topline_rotate:
            add_offset_x, add_offset_y = 0, 0
        else:
            total_diff_width = abs(text_width - text_width_org)
            total_diff_height = abs(text_height - text_height_org)
            topline_diff_width = abs(text_width_topline_rotate - text_width_topline)
            topline_diff_height = abs(text_height_topline_rotate - text_height_topline)
            add_offset_x = max(total_diff_width, topline_diff_width)
            add_offset_y = total_diff_height - topline_diff_height
        return add_offset_x, add_offset_y
        
    def get_text_size_from_bb(self, bboxes: np.ndarray, angle: float):
        text_height = np.max(bboxes[0, :, :]) - np.min(bboxes[0, :, :])
        text_width = np.max(bboxes[1, :, :]) - np.min(bboxes[1, :, :])
        return text_height, text_width

    def get_rotated_text_size(self, text_size: Tuple, angle: float):
        text_height_org, text_width_org = text_size
        if angle !=0 :
            sin = math.sin(2 * math.pi * (float(abs(angle)) / 360.))
            cos = math.cos(2 * math.pi * (float(abs(angle)) / 360.))
            text_height = text_width_org * sin + text_height_org * cos
            text_width = text_width_org * cos + text_height_org * sin
            return text_height, text_width
        else:
            return text_height_org, text_width_org

    def sample(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # mask for text placement
        mask = ih.collision_mask
        # load the number of text line
        text_line_num = dh.tmp.get_text_line_num()
        # get parameters for text types
        text_type_flags, angle = self.sample_text_type_flags(text_line_num)
        # get text location information
        bboxes, bboxes_topline, text_offsets, char_offsets = self.get_text_location_info(
            dh, text_type_flags, angle)
        # get text size from bbox
        text_size_org = self.get_text_size_from_bb(bboxes, angle)
        text_size_rotate = self.get_rotated_text_size(text_size_org, angle)
        try:
            box_offset, box_mask = self.get_box_offset(mask, text_size_rotate, text_size_org, bboxes_topline, angle)
        except BaseException:
            box_offset = None
        if box_offset is None:
            return None
        else:
            # add offsets of the text location
            bboxes = sku.add_offset_bboxes(bboxes, box_offset)
            text_offsets = sku.add_offset_coords(text_offsets, box_offset)
            for char_index, char_offset in enumerate(char_offsets):
                char_offsets[char_index] = sku.add_offset_coords(
                    char_offset, box_offset)
            if angle !=0:
                bboxes = self.rotate_bb(bboxes, text_offsets, angle)
                text_offsets = self.rotate_offsets(text_offsets, text_offsets[0], angle)
                #for char_index, char_offset in enumerate(char_offsets):
                #    char_offsets[char_index] = self.rotate_offsets(char_offset, text_offsets[0], angle)
            # set sampled data to data handler
            dh.tmp.set_offset_sampler_data(
                text_type_flags, angle, text_offsets, char_offsets, text_size_rotate)
        return text_type_flags, angle, bboxes, text_offsets, char_offsets
