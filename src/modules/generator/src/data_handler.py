import numpy as np
from itertools import chain
import skia
from typing import Tuple, List
from src.skia_lib import skia_util as sku
from src.skia_lib import skia_paintor as skp
from .synthtext_lib import synthtext_util as stu
from src.dto.dto_generator import (
    RenderingData,
    TrainingFormatData,
    FontData,
    TextFormData,
    EffectParams,
    ShadowParam,
    FillParam,
    GradParam,
    StrokeParam,
    EffectVisibility,
)


class DataHandler(object):
    def __init__(self, bg):
        self.bg = bg
        self.canvas_img = bg
        self.tmp = TmpDataHandler()
        self.mask_height = []
        self.mask_width = []

        self.texts = []
        self.font_id = []
        self.font_size = []
        self.font_path = []

        self.text_flags = []
        self.bboxes = []
        self.angle = []
        self.text_offsets = []
        self.char_offsets = []

        self.effect_params = []
        self.effect_visibility = []
        self.alpha_arr = np.zeros((bg.shape[0], bg.shape[1], 3))
        self.alpha_list_dict = {}
        self.index = 0
        self.effect_num = 3
    # flag management

    def sample_initialize(self):
        self.placed = False

    def set_sampling_sucess_flag(self):
        self.placed = True

    def check_result(self):
        return self.placed
    # set sampled data

    def set_sampled_data(self, text_font_data, offsets_data, style_data):
        self.set_text_and_font_data(text_font_data)
        self.set_offset_data(offsets_data)
        self.set_style_data(style_data)

    def set_text_and_font_data(self, text_font_data):
        texts, font_size, font_id, font_path = text_font_data
        self.texts.append(texts)
        self.font_size.append(font_size)
        self.font_id.append(font_id)
        self.font_path.append(font_path)

    def set_offset_data(self, offsets_data):
        text_flags, angle, bboxes, text_offsets, char_offsets = offsets_data
        self.text_flags.append(text_flags)
        self.angle.append(angle)
        self.bboxes.append(bboxes)
        self.text_offsets.append(text_offsets)
        self.char_offsets.append(char_offsets)

    def set_style_data(self, style_data):
        effect_visibility, effect_params = style_data
        self.effect_visibility.append(effect_visibility)
        self.effect_params.append(effect_params)

    def get_canvas_img(self):
        return self.canvas_img

    def set_canvas_img(self, canvas_img: np.ndarray):
        self.canvas_img = canvas_img

    def get_bboxes(self, text_index: int):
        return self.bboxes[text_index]

    def set_bboxes(self, bboxes, text_index: int):
        self.bboxes[text_index] = bboxes

    def box_alpha_mask_init(self):
        mask_size = self.tmp.get_mask_size()
        self.box_alpha_mask = np.zeros(mask_size, dtype=np.float32)

    def box_alpha_mask_update(self, rd: RenderingData):
        self.box_alpha_mask = np.maximum(self.box_alpha_mask, rd.shadow_alpha)
        self.box_alpha_mask = np.maximum(self.box_alpha_mask, rd.fill_alpha)
        self.box_alpha_mask = np.maximum(self.box_alpha_mask, rd.stroke_alpha)

    def get_text_rendering_data(self, text_index: int):
        mask_size = self.tmp.get_mask_size()
        textblob = self.tmp.get_text_blob(text_index)
        effect_params = self.tmp.get_effect_params()
        offsets = self.tmp.get_text_offsets(text_index)
        textset_offsets = self.tmp.get_text_offsets(0)
        effect_visibility = self.tmp.get_effect_visibility()
        angle = self.tmp.get_angle()
        paints = skp.get_paint(effect_params)
        alpha = skp.get_alpha(
            mask_size,
            textblob,
            offsets,
            effect_params,
            paints,
            pivot_offsets = None,
            angle = angle,
        )
        alpha = skp.alpha_with_visibility(alpha, effect_visibility)
        rd = RenderingData(
            textblob=textblob,
            textset_offsets=textset_offsets,
            offsets=offsets,
            effect_visibility=effect_visibility,
            effect_params=effect_params,
            paints=paints,
            alpha=alpha,
            angle=angle
        )
        rd.unpack()
        return rd

    def get_char_rendering_data(self, text_index: int, char_index: int):
        mask_size = self.tmp.get_mask_size()
        charblob = self.tmp.get_char_blob(text_index, char_index)
        effect_params = self.tmp.get_effect_params()
        offsets = self.tmp.get_char_offsets(text_index, char_index)
        effect_visibility = self.tmp.get_effect_visibility()
        angle = self.tmp.get_angle()
        paints = skp.get_paint(effect_params)
        alpha = skp.get_alpha(
            mask_size,
            charblob,
            offsets,
            effect_params,
            paints,
            pivot_offsets = None,
            angle = angle,
        )
        alpha = skp.alpha_with_visibility(alpha, effect_visibility)
        rd = RenderingData(
            textblob=charblob,
            offsets=offsets,
            effect_visibility=effect_visibility,
            effect_params=effect_params,
            paints=paints,
            alpha=alpha,
            angle=angle
        )
        rd.unpack()
        return rd

    def update_alpha(self, alpha_list: List):
        for i in range(self.effect_num):
            self.alpha_arr[:, :, i] = np.maximum(
                self.alpha_arr[:, :, i], alpha_list[i])
            
    def decompose_texts_and_bboxes(self):
        new_bboxes, new_texts, new_text_offsets, new_char_offsets, text_pivots, text_indexes = [], [], [], [], [], []
        for text_index, (box, texts, char_offsets) in enumerate(
                zip(self.bboxes, self.texts, self.char_offsets)):
            char_offsets = list(chain.from_iterable(char_offsets))
            tmp_box, tmp_char_offset, tmp_text = [], [], ''
            text_offset_index = 0
            char_index = 0
            for i, text in enumerate(texts):
                #_text = ''.join(text)
                for char in text:
                    if (char == ' ' or char == ' ') and len(tmp_text) > 0:
                        new_bboxes.append(np.array(tmp_box).transpose(1, 2, 0))
                        new_char_offsets.append(tmp_char_offset)
                        new_texts.append(tmp_text)
                        new_text_offsets.append(char_offsets[text_offset_index])
                        text_pivots.append(self.text_offsets[text_index][0])
                        text_indexes.append(text_index)
                        tmp_box, tmp_char_offset, tmp_text = [], [], ''
                        text_offset_index = char_index + 1
                    elif (char == ' ' or char == ' ') and len(tmp_text) == 0:
                        text_offset_index = char_index + 1
                    else:
                        tmp_box.append(box[:, :, char_index])
                        tmp_char_offset.append([char_index])
                        tmp_text += char
                    char_index += 1
                if i+1 < len(texts) and len(tmp_box) > 0:
                    new_bboxes.append(np.array(tmp_box).transpose(1, 2, 0))
                    new_char_offsets.append(tmp_char_offset)
                    new_texts.append(tmp_text)
                    new_text_offsets.append(char_offsets[text_offset_index])
                    text_pivots.append(self.text_offsets[text_index][0])
                    text_indexes.append(text_index)
                    tmp_box, tmp_char_offset, tmp_text = [], [], ''
                    text_offset_index = char_index
            if len(tmp_box) == 0:
                pass
            else:
                new_bboxes.append(np.array(tmp_box).transpose(1, 2, 0))
                new_char_offsets.append(tmp_char_offset)
                new_texts.append(tmp_text)
                new_text_offsets.append(char_offsets[text_offset_index])
                text_pivots.append(self.text_offsets[text_index][0])
                text_indexes.append(text_index)
        new_bboxes = np.concatenate(new_bboxes, axis=2)
        return new_bboxes, new_texts, new_text_offsets, new_char_offsets, text_pivots, text_indexes
    
    def sort_rectangle(self,poly):
        # sort the four coordinates of the polygon, points in poly should be sorted clockwise
        # First find the lowest point
        p_lowest = np.argmax(poly[:, 1])
        if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
            p0_index = np.argmin(np.sum(poly, axis = 1))
            p1_index = (p0_index + 1) % 4
            p2_index = (p0_index + 2) % 4
            p3_index = (p0_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]]#, 0.
        else:
            p_lowest_right = (p_lowest - 1) % 4
            p_lowest_left = (p_lowest + 1) % 4
            angle = np.arctan(
                -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
            if angle / np.pi * 180 > 45:
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]]#, -(np.pi / 2 - angle)
            else:
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]]#, angle
    
    def sort_clockwise(self, _rectangles):
        rectangles = np.zeros_like(_rectangles)
        for i in range(rectangles.shape[2]):
            pts = _rectangles[:,:,i].transpose((1,0))
            rect = self.sort_rectangle(pts)
            rect = np.array(rect)
            #             rect = np.zeros((4, 2), dtype="float32")
            #             s = pts.sum(axis=1)
            #             rect[0] = pts[np.argmin(s)]
            #             rect[2] = pts[np.argmax(s)]
            #             diff = np.diff(pts, axis=1)
            #             rect[3] = pts[np.argmin(diff)]
            #             rect[1] = pts[np.argmax(diff)]
            rectangles[:,:,i]=rect.transpose((1,0))
        return rectangles

    def get_training_format_data(self):
        # divide all texts by spaces
        charBB, texts, text_offsets, char_offsets, text_pivots, text_indexes = self.decompose_texts_and_bboxes()
        charBB = self.sort_clockwise(charBB)        
        wordBB = stu.charBB2wordBB(charBB, texts)
        wordBB = self.sort_clockwise(wordBB)        
        font_data, text_form_data, effect_params, effect_visibility = [], [], [], []
        for text_index in text_indexes:
            font_dto = FontData(
                font_size=self.font_size[text_index],
                font_id=self.font_id[text_index],
                font_path=self.font_path[text_index]
            )
            vertical_text_flag, rotate_text_flag = self.text_flags[text_index]
            text_form_dto = TextFormData(
                vertical_text_flag=vertical_text_flag,
                rotate_text_flag=rotate_text_flag,
                angle=self.angle[text_index],
                width_scale=1,
                text_index = text_index
            )
            shadow_param, fill_param, grad_param, stroke_param = self.effect_params[text_index]
            opacity, blur, dilation, angle, shift, offset_y, offset_x, shadow_color = shadow_param
            fill_color = fill_param
            grad_mode, blend_mode, points, gradation_colors, colorstop = grad_param
            border_weight, border_color = stroke_param
            effect_params_dto = EffectParams(
                shadow_param=ShadowParam(
                    opacity=opacity,
                    blur=blur,
                    dilation=dilation,
                    angle=angle,
                    shift=shift,
                    offset_y=offset_y,
                    offset_x=offset_x,
                    color=shadow_color
                ),
                fill_param=FillParam(
                    color=fill_color
                ),
                grad_param=GradParam(
                    grad_mode=grad_mode,
                    blend_mode=blend_mode,
                    points=points,
                    colors=gradation_colors,
                    colorstop=colorstop,
                ),
                stroke_param=StrokeParam(
                    border_weight=border_weight,
                    color=border_color,
                ),
            )
            shadow_visibility_flag, fill_visibility_flag, gard_visibility_flag, stroke_visibility_flag = self.effect_visibility[
                text_index]
            effect_visibility_dto = EffectVisibility(
                shadow_visibility_flag=shadow_visibility_flag,
                fill_visibility_flag=fill_visibility_flag,
                gard_visibility_flag=gard_visibility_flag,
                stroke_visibility_flag=stroke_visibility_flag,
            )
            font_data.append(font_dto)
            text_form_data.append(text_form_dto)
            effect_params.append(effect_params_dto)
            effect_visibility.append(effect_visibility_dto)
        return charBB, wordBB, texts, font_data, text_form_data, effect_params, effect_visibility, text_offsets, char_offsets, text_pivots

    def export_training_format_data(self):
        charBB, wordBB, texts, font_data, text_form_data, effect_params, effect_visibility, text_offsets, char_offsets, text_pivots = self.get_training_format_data()
        tfd = TrainingFormatData(
            bg=self.bg,  # color pixels for background
            img=self.canvas_img,
            alpha=self.alpha_arr,
            charBB=charBB,
            wordBB=wordBB,
            texts=texts,
            font_data=font_data,
            text_form_data=text_form_data,
            effect_params=effect_params,
            effect_visibility=effect_visibility,
            text_offsets=text_offsets,
            char_offsets=char_offsets,
            text_pivots=text_pivots
        )
        tfd.add_effect_merged_alphaBB()
        return tfd


class TmpDataHandler(object):
    def get_data_for_text_font_sampler(self):
        font_size = self.get_font_size()
        H, W = self.get_mask_size()
        return font_size, H, W

    def set_text_font_sampler_data(
            self,
            texts: List,
            font_path: str,
            nline: int,
            font: skia.Font):
        self.set_texts(texts)
        self.set_font_path(font_path)
        self.set_text_line_num(nline)
        self.set_font(font)

    def get_data_for_text_location_info(self):
        font_size = self.get_font_size()
        font_path = self.get_font_path()
        texts = self.get_texts()
        font = self.get_font()
        return font_size, font_path, texts, font

    def set_offset_sampler_data(
            self,
            text_type_flags: List,
            angle: float,
            text_offsets: Tuple,
            char_offsets: Tuple,
            text_size: int):
        self.set_text_type_flags(text_type_flags)
        self.set_angle(angle)
        self.set_text_offsets(text_offsets)
        self.set_char_offsets(char_offsets)
        self.set_text_size(text_size)

    def get_data_for_style_sampler(self):
        font_size = self.get_font_size()
        offset_x, offset_y = self.get_text_offsets(0)
        text_height, text_width = self.get_text_size()
        return font_size, offset_x, offset_y, text_height, text_width

    def set_style_sampler_data(
            self,
            visibility_flags: List,
            effect_params: List):
        self.set_effect_visibility(visibility_flags)
        self.set_effect_params(effect_params)

    def set_mask_size(self, mask_size: Tuple[int]):
        mask_height, mask_width = mask_size
        self.mask_height = int(mask_height)
        self.mask_width = int(mask_width)

    def get_mask_size(self):
        return (self.mask_height, self.mask_width)

    def set_text_size(self, text_size: int):
        self.text_size = text_size

    def get_text_size(self):
        return self.text_size

    def set_text_line_number(self, text_line_number: int):
        self.text_line_number = text_line_number

    def get_text_line_number(self):
        return self.text_line_number

    def set_font_size(self, font_size: int):
        self.font_size = font_size

    def get_font_size(self):
        return self.font_size

    def set_texts(self, texts: List[str]):
        self.texts = texts

    def get_texts(self):
        return self.texts

    def set_text_offsets(self, text_offsets: List):
        self.text_offsets = text_offsets

    def get_text_offsets(self, text_index: int):
        return self.text_offsets[text_index]

    def set_char_offsets(self, char_offsets: List):
        self.char_offsets = char_offsets

    def get_char_offsets(self, text_index: int, char_index: int):
        return self.char_offsets[text_index][char_index]

    def set_font_path(self, font_path: str):
        self.font_path = font_path

    def get_font_path(self):
        return self.font_path

    def set_text_line_num(self, text_line_num: int):
        self.text_line_num = text_line_num

    def get_text_line_num(self):
        return self.text_line_num

    def set_text_type_flags(self, text_type_flags: List):
        self.text_type_flags = text_type_flags

    def get_text_type_flags(self):
        return self.text_type_flags

    def set_angle(self, angle: float):
        self.angle = angle

    def get_angle(self):
        return self.angle

    def get_font(self):
        return self.font

    def set_font(self, font: skia.Font):
        self.font = font

    def set_effect_params(self, effect_params: List):
        self.effect_params = effect_params

    def get_effect_params(self):
        return self.effect_params

    def set_effect_visibility(self, effect_visibility: List):
        self.effect_visibility = effect_visibility

    def get_effect_visibility(self):
        return self.effect_visibility

    def get_text_blob(self, text_index: int):
        textblob = skia.TextBlob(self.texts[text_index], self.font)
        return textblob

    def get_char_blob(self, text_index: int, char_index: int):
        textblob = skia.TextBlob(self.texts[text_index][char_index], self.font)
        return textblob
