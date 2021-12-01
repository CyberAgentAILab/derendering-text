import math
import time
from typing import Generator, Tuple

import torch
import torch.optim as optim
from logzero import logger as log
from torch import nn
from torch.functional import F

from src.dto.dto_postprocess import OptimizeParameter
from .tensor import arr_to_cuda


def compute_gaussian_kernel(inp, sigma, dim=2, dev: torch.device = None):
    kernel_factor = 3.0 / 4.0 * math.sqrt(2 * math.pi)
    kernel_size = max(2, math.floor(sigma * kernel_factor + 0.5))
    if kernel_size % 2 == 0:
        kernel_size += 1
    psize = kernel_size // 2
    if inp.is_cuda:
        kernel = torch.zeros(kernel_size, kernel_size).float().to(dev) + 1
    else:
        kernel = torch.zeros(kernel_size, kernel_size).float() + 1
    kernel_size = [kernel_size] * dim
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )
    for mgrid in meshgrids:
        if inp.is_cuda:
            mgrid = mgrid.to(dev)
        mean = (kernel_size[0] - 1) / 2
        kernel *= (
            1
            / (sigma * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / (2 * sigma)) ** 2))
        )
    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))
    # kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))
    return kernel, psize


def gfilter(inp, sigma, dim=2, dev: torch.device = None):
    kernel, pad_size = compute_gaussian_kernel(inp, sigma, dim=2, dev=dev)
    pad_size *= 3
    inp = F.pad(inp, (pad_size, pad_size, pad_size, pad_size))
    out = F.conv2d(inp, kernel)
    out = F.conv2d(out, kernel)
    out = F.conv2d(out, kernel)
    return out


def rgb_gfilter(inp, sigma, dev: torch.device = None):
    inp_r = gfilter(inp[:, 0:1], sigma, dev=dev)
    inp_g = gfilter(inp[:, 1:2], sigma, dev=dev)
    inp_b = gfilter(inp[:, 2:3], sigma, dev=dev)
    out = torch.cat((inp_r, inp_g, inp_b), 1)
    return out


def post_refinement(
    reconstructor: nn.Module,
    optp: OptimizeParameter,
    fix_params: Tuple,
    iter_count: int,
    affine_identity: bool = True,
    dev: torch.device = None
) -> Tuple:
    img, _, _, bbox_information = fix_params
    text_instance_mask = bbox_information.get_text_instance_mask()
    char_instance_mask = bbox_information.get_char_instance_mask()

    if affine_identity:
        optp.affine_outs = torch.zeros_like(optp.affine_outs)
        optp.affine_outs[:, 0, 0, 0] = 0
        optp.affine_outs[:, 1, 0, 0] = 0
        optp.affine_outs[:, 2, 0, 0] = 0
        optp.affine_outs[:, 3, 0, 0] = 0
        optp.affine_outs[:, 4, 0, 0] = 0

    params_lr001 = []
    params_lr001 += [optp.font_outs.requires_grad_(True)]
    params_lr001 += [optp.shadow_visibility_outs.requires_grad_(True)]
    params_lr001 += [optp.shadow_param_sig_outs.requires_grad_(True)]
    params_lr001 += [optp.shadow_param_tanh_outs.requires_grad_(True)]
    params_lr001 += [optp.stroke_visibility_outs.requires_grad_(True)]
    params_lr001 += [optp.stroke_param_outs.requires_grad_(True)]
    params_lr001 += [optp.affine_outs.requires_grad_(True)]
    params_lr001 += [optp.fill_color.requires_grad_(True)]
    params_lr001 += [optp.shadow_color.requires_grad_(True)]
    params_lr001 += [optp.stroke_color.requires_grad_(True)]
    char_instance_mask = arr_to_cuda(char_instance_mask, dev=dev)
    text_instance_mask = arr_to_cuda(text_instance_mask, dev=dev)

    char_instance_mask = char_instance_mask.unsqueeze(
        1).float().repeat(1, 3, 1, 1)
    text_instance_mask = text_instance_mask.unsqueeze(
        1).float().repeat(1, 3, 1, 1)

    optimizer = optim.Adam(
        [
            {"params": params_lr001, "lr": 0.01},
        ],
        lr=0.01,
    )
    cnt = 0
    for k in range(iter_count):
        start = time.time()
        optimizer.zero_grad()
        rec_outs = reconstructor.reconstruction_with_vector_elements(
            optp, fix_params)
        rgb_reconstructed = rec_outs[1]
        rgb_reconstructed = (rgb_reconstructed.float() / 255.0 - 0.5) * 2
        img_float = img.float()
        if k < 100:
            # apply blur fuction in early optimization steps
            # this processing is a typical technique in fitting of character
            # appearance
            b_param = 10 * ((100 - k) / 100.0)
            img_float = rgb_gfilter(img_float, b_param, dev=dev)
            rgb_reconstructed = rgb_gfilter(
                rgb_reconstructed, b_param, dev=dev)
        loss = F.l1_loss(rgb_reconstructed, img_float.detach())
        loss.backward()
        optimizer.step()
        if k % 20 == 0:
            log.debug(f"time: {time.time() - start}")
            log.debug(f"{k}/{iter_count}, {cnt}, {loss.item()}")
            cnt += 1

    rec_outs = reconstructor.reconstruction_with_vector_elements(
        optp, fix_params)
    _, rgb_reconstructed = rec_outs
    return optp, rgb_reconstructed
