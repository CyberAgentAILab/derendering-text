import os

import cv2
import numpy as np
import skia
from logzero import logger as log
from src.io import load_font_dicts


fontmgr = skia.FontMgr()


def harmonization(img, bg, text_fg, alpha, text_fg_th, alpha_th):
    mask = (text_fg[:, :, 1:2] > text_fg_th) & (
        np.max(alpha, 2)[:, :, np.newaxis] > alpha_th
    )
    prior = np.tile(mask.astype(np.float32), (1, 1, 3))
    prior = cv2.blur(prior, (10, 10))
    prior = np.maximum(prior, mask)
    prior = np.tanh(3 * prior)
    harmonized_bg = img.copy()
    for _ in range(5):
        harmonized_bg = harmonized_bg * (1 - prior) + bg * prior
    return harmonized_bg


def extract_effect_visibility(visibility_outs):
    visibility_flags = []
    for i in range(len(visibility_outs)):
        if visibility_outs[i, 1] > 0.1:
            visibility_flags.append(True)
        else:
            visibility_flags.append(False)
    return visibility_flags


def extract_shadow_params(
    optp,
    text_rectangles,
    model_outs_size,
    img_size,
):
    (
        shadow_param_sig_outs,
        shadow_param_tanh_outs,
        shadow_visibility_outs,
    ) = optp.get_shadow_params()
    shadow_params = []
    for i in range(len(text_rectangles)):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_rectangles[i]
        ys = int(min(y1, y2, y3, y4))
        ye = int(max(y1, y2, y3, y4))
        font_size = ye - ys
        op = shadow_visibility_outs[i, 1]  # shadow_param_sig_outs[i, 0]
        bsz = 0.5 * shadow_param_sig_outs[i, 1] * font_size
        # dp = 0.25 * param_shadow_outs[i, 2] * float(font_size_list[i])
        dp = 0
        offsetds_x = int(
            0.2
            * (shadow_param_tanh_outs[i, 0])
            * font_size
            * float(img_size[0])
            / float(model_outs_size[0])
        )
        offsetds_y = int(
            0.2
            * (shadow_param_tanh_outs[i, 1])
            * font_size
            * float(img_size[1])
            / float(model_outs_size[1])
        )
        shadow_param = (op, bsz, dp, offsetds_y, offsetds_x)
        shadow_params.append(shadow_param)
    return shadow_params


def extract_stroke_params(optp, tbp_list):
    param_stroke_outs, stroke_visibility_outs = optp.get_stroke_params()
    stroke_params = []
    for i in range(len(tbp_list)):
        param_index = np.argmax(param_stroke_outs[i, :])
        # weight = (param_stroke_outs[i, 0] * float(font_size_list[i]) / 25.)+0.05
        weight = (param_index * 0.2 *
                  float(tbp_list[i].font_data.font_size) / 25.0) + 0.05
        stroke_param = (weight, stroke_visibility_outs[i, 1])
        stroke_params.append(stroke_param)
    return stroke_params


def extract_fonts(font_outs):
    font_ids = []
    for i in range(len(font_outs)):
        font_id = np.argmax(font_outs[i, :, 0, 0])
        font_ids.append(font_id)
    return font_ids
