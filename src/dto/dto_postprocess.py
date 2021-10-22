from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from src.io import load_font_dicts
from src.skia_lib import skia_util as sku
from .dto_model import BatchWrapperBBI
from .dto_skia import(
    FontData,
    TextFormData,
    ShadowParam,
    FillParam,
    GradParam,
    StrokeParam,
    EffectParams,
    EffectVisibility
)


ColorPred = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class MetaDataPostprocessing:
    bbox_information: Any
    model_outs_size: Any
    img_org_size: Any


@dataclass
class InputData:
    """translated input data"""

    model_outs_size: Tuple[int, int]  # resized image size
    img_org_size: np.ndarray  # original image size


@dataclass
class TextBlobParameter:
    font_data: FontData
    text_form_data: TextFormData
    text_top: float  # text height from type face
    y_start: float  # text location parameter for drawing
    text: float  # scripts
    box: Any  # text bouding box information, (x0,y0,x1,y1)


@dataclass
class VectorData:
    """vectorized data for text rendering"""
    bg: np.ndarray  # color pixels for background
    tb_param: List[TextBlobParameter]
    effect_param: List[EffectParams]  # effect parameters
    effect_visibility: List[EffectVisibility]  # color parameters

    def get_texts(self):
        texts = {}
        for index, tb in enumerate(self.tb_param):
            texts[index] = tb.text
        return texts
    def set_text(self, index, text):
        self.tb_param[index].text=text
        self.update_wscale(index)

    def show_font(self, index):
        font_dict = load_font_dicts()
        font_id = self.tb_param[index].font_data.font_id
        print('font_path',font_dict[font_id])
    def set_font(self, index, font_id):
        self.tb_param[index].font_data.font_id=font_id
        self.update_wscale(index)

    def update_wscale(self, index):
        x0, y0, x1, y1 = self.tb_param[index].box
        text = self.tb_param[index].text
        font_id = self.tb_param[index].font_data.font_id
        font_size = self.tb_param[index].font_data.font_size
        font = sku.get_textfont(font_id, font_size)
        (_, _),(_, text_width), _, _ = sku.get_text_spatial_info(offset_y=0, offset_x=0, font=font, text=text)
        change_x_ratio = text_width/max(x1-x0,1e-5)
        new_wscale = (x1-x0)/max(text_width,1e-5)
        self.tb_param[index].text_form_data.width_scale = new_wscale

    def get_shadow_visibility(self):
        visibility = {}
        for index, ev in enumerate(self.effect_visibility):
            visibility[index] = ev.shadow_visibility_flag
        return visibility
    def set_shadow_visibility(self, index, flag):
        self.effect_visibility[index].shadow_visibility_flag=flag
    def set_shadow_param(self, index, param: ShadowParam):
        self.effect_param[index].shadow_param=param

    def set_fill_visibility(self, index, flag):
        self.effect_visibility[index].fill_visibility_flag=flag
    def set_fill_param(self, index, param: FillParam):
        self.effect_param[index].fill_param=param

    def set_stroke_visibility(self, index, flag):
        self.effect_visibility[index].stroke_visibility_flag=flag
    def set_stroke_param(self, index, param: StrokeParam):
        self.effect_param[index].stroke_param=param

    def get_offset(self, index):
        x0, _, _, _ = self.tb_param[index].box
        offset_x = int(
            x0 / max(float(self.tb_param[index].text_form_data.width_scale), 1e-5))
        offset_y = int(self.tb_param[index].y_start - self.tb_param[index].text_top)
        return (offset_y, offset_x)
    def set_offset(self, index, offset_y:int, offset_x:int):
        x0, y0, x1, y1 = self.tb_param[index].box
        new_x0 = offset_x * self.tb_param[index].text_form_data.width_scale
        new_y0 = offset_y
        dx = new_x0-x0
        dy = new_y0-y0
        self.tb_param[index].box = (new_x0, new_y0,x1+dx,y1+dy)
        self.tb_param[index].y_start = offset_y+self.tb_param[index].text_top

    def get_font_names(self):
        font_dict = load_font_dicts()
        font_names = []
        for tb in self.tb_param:
            font_path = font_dict[tb.font_id]
            font_file_name = font_path.split("/")[-1]
            # font name extract from a file name
            font_name = font_file_name.split(".")[0]
            font_names.append(font_name)
        return font_names

    def font_name(self, idx: int):
        return self.get_font_names()[idx]

    def get_stroke_params(self):
        stroke_params = []
        stroke_visibility = []
        for visibility_flag in self.effect_param.stroke_visibility:
            stroke_visibility.append(visibility_flag)
        for (stroke_width, _) in self.effect_param.effect_stroke_params:
            stroke_params.append(stroke_width)
        return stroke_visibility, stroke_params

    def get_shadow_params(self):
        shadow_params = []
        shadow_visibility = []
        for visibility_flag in self.effect_param.shadow_visibility:
            shadow_visibility.append(visibility_flag)
        for (
            op,
            bsz,
            dp,
            offsetds_y,
            offsetds_x,
        ) in self.effect_param.effect_shadow_params:
            # op: opacity parameter
            # bsz: blur parameter
            # dp: dilation parameter, but now not supported
            # offsetds_y: offset parameter for drop shadow
            # offsetds_x: offset parameter for drop shadow
            shadow_params.append((op, bsz, dp, offsetds_y, offsetds_x))
        return shadow_visibility, shadow_params

    def get_color_params(self):
        return (
            self.color_param.fill_color,
            self.color_param.stroke_color,
            self.color_param.shadow_color,
        )

    def get_background_pixels(self):
        return self.bg


@dataclass
class OptimizeParameter:
    """Optimization parameters"""

    font_outs: Any  # [N x B x F x 1 x 1] font prediction
    # font_size_outs: Any  # [N x B x 1 x 1 x 1] font size prediction
    affine_outs: Any  # [C x 6 x 1 x 1] affine transformation parameters
    char_vec: Any  # [C x 94 x H x W] character category information
    alpha_outs: Any  # [N x 3 x H x W] alpha prediction
    fill_color: Any  # [N x B x 3 x 1 x 1] fill color
    shadow_color: Any  # [N x B x 3 x 1 x 1] shadow color
    stroke_color: Any  # [N x B x 3 x 1 x 1] stroke color
    # [N x B x 2 x 1 x 1] visibility prediction for effect, 0 for w/o effect, 1 for w/ effect
    shadow_visibility_outs: Any
    # [N x B x 2 x 1 x 1] visibility prediction for effect, 0 for w/o effect, 1 for w/ effect
    stroke_visibility_outs: Any
    # [N x 2 x H x W] shadow parameter regression for dilation and blur, now dilation is not supported
    shadow_param_sig_outs: Any
    # [N x 2 x H x W] shadow parameter regression for offset of x and y
    shadow_param_tanh_outs: Any
    # [N x B x 5 x H x W] stroke parameter classification for weights
    stroke_param_outs: Any

    def get_shadow_params(self):
        return (
            self.shadow_param_sig_outs[0],
            self.shadow_param_tanh_outs[0],
            self.shadow_visibility_outs[0],
        )

    def get_stroke_params(self):
        return self.stroke_param_outs[0], self.stroke_visibility_outs[0]

    def get_shadow_visibility_outs(self):
        return self.shadow_visibility_outs[0]

    def get_stroke_visibility_outs(self):
        return self.stroke_visibility_outs[0]

    def get_color_params(self):
        return self.shadow_color, self.fill_color, self.stroke_color


@dataclass
class OutputData:
    """
    translated output data

    N: batch size
    B: number of text
    C: number of character
    F: number of channel or category
    H: height
    W: width
    """

    bg_pixels: np.ndarray  # [N x H x W x3] rgb background pixels
    # [N x B x F x 1 x 1] font prediction, F is the number of font's category
    font_outs: np.ndarray
    # font_size_outs: np.ndarray  # [N x B x 1 x 1 x 1] font size prediction
    affine_outs: np.ndarray  # [C x 6 x 1 x 1] affine parameter [2x3]
    char_rec_vec: np.ndarray  # [C x 94 x H x W] character prediction for OCR
    # [N x 3 x H x W] alpha prediction, 0:fill, 1:stroke, 2:shadow
    alpha_outs: np.ndarray
    color_pred: ColorPred  # [N x 3 x H x W] rgb background pixels
    # [N x H' x W' x 2] foreground pixels H' and W' are size of model's outputs
    text_fg_pred: np.ndarray
    # [N x B x 2 x 1 x 1] visibility prediction for effect, 0 for w/o effect, 1 for w/ effect
    shadow_visibility_outs: np.ndarray
    # [N x B x 2 x 1 x 1] visibility prediction for effect , 0 for w/o effect, 1 for w/ effect
    stroke_visibility_outs: np.ndarray
    # [N x 2 x H x W] shadow parameter regression for dilation and blur, now dilation is not supported
    shadow_param_sig_outs: np.ndarray
    # [N x 2 x H x W] shadow parameter regression for offset of x and y
    shadow_param_tanh_outs: np.ndarray
    # [N x B x 5 x H x W] stroke parameter classification for weights
    stroke_param_outs: np.ndarray
    bbox_information: BatchWrapperBBI  # OCR outputs
