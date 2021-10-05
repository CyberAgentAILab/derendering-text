from typing import Tuple
import numpy as np
from PIL.Image import Image as PILImage
import torch
from torch.functional import F

from src.modules.postprocess.postref import post_refinement
from src.models.reconstructor import Reconstructor
from src.dto.dto_postprocess import InputData, MetaDataPostprocessing, OutputData, VectorData
from .manipulate import (
    extract_rendering_params,
    get_postrefine_params,
    numpynize_optp,
    resize_model_output,
)
from .tensor import torch_to_numpy


def transform(bg_pixels, text_fg_pred):
    bg_pixels = (bg_pixels.transpose(0, 2, 3, 1) * 0.5 + 0.5) * 255
    text_fg_pred = text_fg_pred.transpose(0, 2, 3, 1)
    return bg_pixels, text_fg_pred


def convert_output(outputs: Tuple) -> OutputData:
    # outputs
    # bg_pixels = inpaint of model output
    ti, bg_pixels, reconstructor_outs = outputs
    # ocr outs
    word_out, _, _ = ti.ocr_outs
    text_fg_pred, _, _ = word_out
    text_fg_pred = F.softmax(text_fg_pred, 1)
    # effect outs
    shadow_visibility_outs, stroke_visibility_outs = ti.effect_visibility_outs
    (
        shadow_param_sig_outs,
        shadow_param_tanh_outs,
        stroke_param_outs,
    ) = ti.effect_param_outs
    _, _, _, affine_outs, char_rec_vec, color_pred = reconstructor_outs
    fill_color_pred, shadow_color_pred, stroke_color_pred = color_pred

    # cuda to cpu and torch to numpy
    text_fg_pred = torch_to_numpy(text_fg_pred)
    bg_pixels = torch_to_numpy(bg_pixels)
    font_outs = torch_to_numpy(ti.font_outs)
    #font_size_outs = torch_to_numpy(ti.font_size_outs)
    affine_outs = torch_to_numpy(affine_outs)
    alpha_outs = torch_to_numpy(ti.alpha_outs)
    char_rec_vec = torch_to_numpy(char_rec_vec)
    shadow_visibility_outs = torch_to_numpy(shadow_visibility_outs)
    stroke_visibility_outs = torch_to_numpy(stroke_visibility_outs)
    shadow_param_sig_outs = torch_to_numpy(shadow_param_sig_outs)
    shadow_param_tanh_outs = torch_to_numpy(shadow_param_tanh_outs)
    stroke_param_outs = torch_to_numpy(stroke_param_outs)
    fill_color_pred = torch_to_numpy(fill_color_pred) / 255
    shadow_color_pred = torch_to_numpy(shadow_color_pred) / 255
    stroke_color_pred = torch_to_numpy(stroke_color_pred) / 255
    color_pred = (fill_color_pred, shadow_color_pred, stroke_color_pred)

    # transform
    bg_pixels, text_fg_pred = transform(bg_pixels, text_fg_pred)

    return OutputData(
        bg_pixels,
        font_outs,
        # font_size_outs,
        affine_outs,
        char_rec_vec,
        alpha_outs,
        color_pred,
        text_fg_pred,
        shadow_visibility_outs,
        stroke_visibility_outs,
        shadow_param_sig_outs,
        shadow_param_tanh_outs,
        stroke_param_outs,
        ti.bbox_information,
    )


def convert_input(inputs) -> InputData:
    # inputs
    img_norm, _, img_org_size = inputs
    img_org_size = img_org_size.numpy().astype(np.int32)[:, ::-1]
    # convert 2 vector representation
    _, _, height, width = img_norm.shape
    model_outs_size = (width, height)
    return InputData(model_outs_size, img_org_size)


def vectorize(img: PILImage, inps: Tuple, outs: Tuple,
              dev: torch.device = None) -> Tuple[VectorData, np.ndarray]:
    img = np.array(img)  # original image
    id = convert_input(inps)
    od = convert_output(outs)  # torch -> numpy

    # post refiment + convert to rendering engine parameter
    rgb_rec = outs[2][2]

    # resize model outputs
    bg = resize_model_output(
        img,
        od.bg_pixels[0],
        od.text_fg_pred[0],
        od.alpha_outs[0])
    opt_params, _ = get_postrefine_params(
        img, bg, id.model_outs_size, od, dev=dev)

    optp = numpynize_optp(opt_params)  # torch -> numpy
    mdp = MetaDataPostprocessing(
        od.bbox_information, id.model_outs_size, id.img_org_size
    )
    tb_param, effect_param, effect_visibility = extract_rendering_params(
        mdp, optp, rgb_rec, img
    )

    # vectorization data for rendering
    return (VectorData(bg, tb_param, effect_param, effect_visibility), rgb_rec)


def vectorize_postref(
    img: PILImage,
    inps: Tuple,
    outs: Tuple,
    reconstructor: Reconstructor,
    iter_count: int,
    dev: torch.device = None
) -> Tuple[VectorData, np.ndarray]:
    img = np.array(img)  # original image
    id = convert_input(inps)
    od = convert_output(outs)  # torch -> numpy

    # post refiment + convert to rederign engine parameter
    # resize model outputs
    bg = resize_model_output(
        img,
        od.bg_pixels[0],
        od.text_fg_pred[0],
        od.alpha_outs[0])
    opt_params, fix_params = get_postrefine_params(
        img, bg, id.model_outs_size, od, dev=dev)

    # refinement (too heavy)
    optp, rgb_rec = post_refinement(
        reconstructor, opt_params, fix_params, iter_count, dev=dev
    )

    # numpy
    optp = numpynize_optp(optp)
    mdp = MetaDataPostprocessing(
        od.bbox_information, id.model_outs_size, id.img_org_size
    )
    tb_param, effect_param, effect_visibility = extract_rendering_params(
        mdp, optp, rgb_rec, img, dev=dev
    )

    # vectorization data for rendering
    return (VectorData(bg, tb_param, effect_param, effect_visibility), rgb_rec)
