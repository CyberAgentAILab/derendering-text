from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.functional import F

from src.dto.dto_skia import(
    ShadowParam,
    FillParam,
    GradParam,
    StrokeParam,
    EffectParams,
    EffectVisibility
)
from src.io import load_char_label_dicts
from src.dto.dto_postprocess import (
    MetaDataPostprocessing,
    OptimizeParameter,
    OutputData,
)
from .rpe import get_textblob_param_with_affine
from .tensor import arr_to_cuda, torch_to_numpy
from .vector_util import (
    extract_effect_visibility,
    extract_fonts,
    extract_shadow_params,
    extract_stroke_params,
    harmonization,
)


def resize_model_output(
    img: np.ndarray,
    bg_pixels: np.ndarray,
    text_fg_pred: np.ndarray,
    alpha_outs: np.ndarray,
    text_fg_threshold: float = 0.25,
    alpha_threshold: float = 0.1,
):
    img_size = (img.shape[1], img.shape[0])
    bg_pixel_resize = cv2.resize(bg_pixels, img_size)
    text_fg_pred_resize = cv2.resize(text_fg_pred, img_size)
    alpha_out_resize = cv2.resize(alpha_outs.transpose(1, 2, 0), img_size)
    for _ in range(4):
        bg_pixel_resize = cv2.bilateralFilter(bg_pixel_resize, 9, 75, 75)

    bg_pixel_harmonized = harmonization(
        img,
        bg_pixel_resize,
        text_fg_pred_resize,
        alpha_out_resize,
        text_fg_threshold,
        alpha_threshold,
    )
    return bg_pixel_harmonized


def get_postrefine_params(
        img: np.ndarray,
        bg: np.ndarray,
        model_outs_size: np.ndarray,
        od: OutputData,
        dev: torch.device = None):
    # convert model outputs
    # transform
    bg = (bg / 255.0 - 0.5) * 2
    bg_rec = cv2.resize(bg, model_outs_size)

    #  Numpy to Torch
    font_outs_rec = arr_to_cuda(od.font_outs[0:1], dev=dev)
    #font_size_outs_rec = arr_to_cuda(od.font_size_outs[0:1])
    alpha_outs_rec = arr_to_cuda(od.alpha_outs[0:1], dev=dev)
    shadow_visibility_outs_rec = arr_to_cuda(
        od.shadow_visibility_outs[0:1], dev=dev)
    shadow_param_sig_outs_rec = arr_to_cuda(
        od.shadow_param_sig_outs[0:1], dev=dev)
    shadow_param_tanh_outs_rec = arr_to_cuda(
        od.shadow_param_tanh_outs[0:1], dev=dev)
    stroke_visibility_outs_rec = arr_to_cuda(
        od.stroke_visibility_outs[0:1], dev=dev)
    stroke_param_outs_rec = arr_to_cuda(od.stroke_param_outs[0:1], dev=dev)
    affine_outs_rec = arr_to_cuda(od.affine_outs, dev=dev)
    char_rec_vec = arr_to_cuda(od.char_rec_vec, dev=dev)
    bg_rec = arr_to_cuda(bg_rec.transpose(2, 0, 1), dev=dev).unsqueeze(0)
    img_model_outs_size = cv2.resize(img, model_outs_size)
    img_rec = (img_model_outs_size / 255.0 - 0.5) * 2
    img_rec = arr_to_cuda(img_rec.transpose(2, 0, 1), dev=dev).unsqueeze(0)

    fill_color, shadow_color, stroke_color = od.color_pred
    fill_color_rec = arr_to_cuda(np.array(fill_color), dev=dev).float()
    shadow_color_rec = arr_to_cuda(
        np.array(shadow_color),
        dev=dev).float() / 255
    stroke_color_rec = arr_to_cuda(
        np.array(stroke_color),
        dev=dev).float() / 255

    colors = (
        fill_color_rec,
        shadow_color_rec,
        stroke_color_rec,
    )
    fix_params = img_rec, bg_rec, colors, od.bbox_information

    return (
        OptimizeParameter(
            font_outs_rec,
            # font_size_outs_rec,
            affine_outs_rec,
            char_rec_vec,
            alpha_outs_rec,
            fill_color_rec,
            shadow_color_rec,
            stroke_color_rec,
            shadow_visibility_outs_rec,
            stroke_visibility_outs_rec,
            shadow_param_sig_outs_rec,
            shadow_param_tanh_outs_rec,
            stroke_param_outs_rec,
        ),
        fix_params,
    )


def numpynize_optp(optp: OptimizeParameter) -> OptimizeParameter:
    optp.shadow_visibility_outs = F.softmax(
        optp.shadow_visibility_outs[0:1], 2)
    optp.shadow_visibility_outs = torch.reciprocal(
        1 + torch.exp(-50 * (optp.shadow_visibility_outs - 0.5))
    )
    optp.stroke_visibility_outs = F.softmax(
        optp.stroke_visibility_outs[0:1], 2)
    optp.stroke_visibility_outs = torch.reciprocal(
        1 + torch.exp(-50 * (optp.stroke_visibility_outs - 0.5))
    )

    optp.font_outs = torch_to_numpy(optp.font_outs)
    #optp.font_size_outs = torch_to_numpy(optp.font_size_outs)
    # affine_outs = affine_outs.data.cpu().numpy()
    optp.alpha_outs = torch_to_numpy(optp.alpha_outs).transpose(0, 2, 3, 1)
    optp.char_vec = torch_to_numpy(optp.char_vec)
    optp.shadow_visibility_outs = torch_to_numpy(optp.shadow_visibility_outs)
    optp.stroke_visibility_outs = torch_to_numpy(optp.stroke_visibility_outs)
    optp.shadow_param_sig_outs = torch_to_numpy(optp.shadow_param_sig_outs)
    optp.shadow_param_tanh_outs = torch_to_numpy(optp.shadow_param_tanh_outs)
    optp.stroke_param_outs = torch_to_numpy(optp.stroke_param_outs)

    optp.fill_color = (
        torch_to_numpy(
            F.relu(
                optp.fill_color).view(
                optp.fill_color.shape[1],
                3)) * 255)
    optp.fill_color = np.minimum(
        optp.fill_color, np.zeros_like(
            optp.fill_color) + 255)
    optp.shadow_color = (
        torch_to_numpy(
            F.relu(
                optp.shadow_color).view(
                optp.shadow_color.shape[1],
                3)) * 255)
    optp.shadow_color = np.minimum(
        optp.shadow_color, np.zeros_like(optp.fill_color) + 255
    )
    optp.stroke_color = (
        torch_to_numpy(
            F.relu(
                optp.stroke_color).view(
                optp.stroke_color.shape[1],
                3)) * 255)
    optp.stroke_color = np.minimum(
        optp.stroke_color, np.zeros_like(optp.fill_color) + 255
    )
    return optp


def get_texts(optp: OptimizeParameter):
    char_dict, _ = load_char_label_dicts()
    texts = ""
    for c in range(len(optp.char_vec)):
        char_id = np.argmax(optp.char_vec[c, :, 0, 0])
        texts += char_dict[char_id]
    return texts


def get_shadow_param_dto(shadow_params, shadow_colors):
    shadow_dto_list = []
    for shadow_param, shadow_color in zip(shadow_params, shadow_colors):
        (opacity, blur, dilation, offset_y, offset_x) = shadow_param
        shadow_dto = ShadowParam(
            opacity=opacity,
            blur=blur,
            dilation=dilation,
            angle=0,
            shift=None,
            offset_y=offset_y,
            offset_x=offset_x,
            color=shadow_color
        )
        shadow_dto_list.append(shadow_dto)
    return shadow_dto_list


def get_fill_param_dto(fill_colors):
    fill_dto_list = []
    for fill_color in fill_colors:
        fill_dto = FillParam(
            color=fill_color
        )
        fill_dto_list.append(fill_dto)
    return fill_dto_list


def get_stroke_param_dto(stroke_params, stroke_colors):
    stroke_dto_list = []
    for stroke_param, stroke_color in zip(stroke_params, stroke_colors):
        border_weight, _ = stroke_param
        stroke_dto = StrokeParam(
            border_weight=border_weight,
            color=stroke_color,
        )
        stroke_dto_list.append(stroke_dto)
    return stroke_dto_list


def get_effect_params_dto(shadow_param_dto, fill_param_dto, stroke_param_dto):
    effect_params_dto_list = []
    for shadow, fill, stroke in zip(
            shadow_param_dto, fill_param_dto, stroke_param_dto):
        effect_params_dto = EffectParams(
            shadow, fill, None, stroke
        )
        effect_params_dto_list.append(effect_params_dto)
    return effect_params_dto_list


def get_effect_params(
    mdp: MetaDataPostprocessing, optp: OptimizeParameter, tbp_list: List
) -> List[EffectParams]:
    text_rectangles = mdp.bbox_information.get_text_rectangle()[0]
    effect_shadow_params = extract_shadow_params(
        optp, text_rectangles, mdp.model_outs_size, mdp.img_org_size[0]
    )
    effect_stroke_params = extract_stroke_params(optp, tbp_list)
    shadow_color, fill_color, stroke_color = optp.get_color_params()
    shadow_param_dto = get_shadow_param_dto(effect_shadow_params, shadow_color)
    fill_param_dto = get_fill_param_dto(fill_color)
    stroke_param_dto = get_stroke_param_dto(effect_stroke_params, stroke_color)
    effect_params_dto = get_effect_params_dto(
        shadow_param_dto, fill_param_dto, stroke_param_dto)
    return effect_params_dto


def get_effect_visibility(optp: OptimizeParameter) -> List[EffectVisibility]:
    shadow_visibility = extract_effect_visibility(
        optp.get_shadow_visibility_outs())
    stroke_visibility = extract_effect_visibility(
        optp.get_stroke_visibility_outs())
    effect_visibility_dto_list = []
    for shadow_vis_flag, stroke_vis_flag in zip(
            shadow_visibility, stroke_visibility):
        effect_visibility_dto = EffectVisibility(
            shadow_visibility_flag=shadow_vis_flag,
            fill_visibility_flag=False,
            gard_visibility_flag=False,
            stroke_visibility_flag=stroke_vis_flag,
        )
        effect_visibility_dto_list.append(effect_visibility_dto)
    return effect_visibility_dto_list


def extract_rendering_params(
    mdp: MetaDataPostprocessing,
    optp: OptimizeParameter,
    rgb_rec: np.ndarray,
    img: np.ndarray,
    dev: torch.device = None
) -> Tuple:
    # extruct text data
    texts = get_texts(optp)
    # extruct font id
    font_ids = extract_fonts(optp.font_outs[0])
    # get font spatial params
    tb_param = get_textblob_param_with_affine(
        mdp,
        optp.affine_outs,
        texts,
        font_ids,
        rgb_rec,
        img,
        dev=dev
    )
    # get effect params
    effect_param = get_effect_params(mdp, optp, tb_param)
    effect_visibility = get_effect_visibility(optp)
    return tb_param, effect_param, effect_visibility
