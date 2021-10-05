import math
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from util.path_list import get_prerendered_alpha_dir


def get_max_char_box_num(char_rectangles_array):
    max_num = 0
    for i in range(len(char_rectangles_array)):
        if max_num < len(char_rectangles_array[i]):
            max_num = len(char_rectangles_array[i])
    return max_num


def char_mask_pooling(
        x,
        char_rectangles_array,
        char_ins_mask,
        dev: torch.device = None):
    char_ins_mask_resize = F.interpolate(char_ins_mask, x.shape[2:4])
    max_char_box_num = get_max_char_box_num(char_rectangles_array)
    x_stack = (
        torch.zeros(
            x.shape[0] *
            max_char_box_num,
            x.shape[1],
            1,
            1).float().to(dev))
    for i in range(len(x)):
        char_box_num = len(char_rectangles_array[i])
        for k in range(char_box_num):
            loc = char_ins_mask_resize[i: i + 1] == (k + 1)
            if torch.sum(loc).item() == 0:
                pass
            else:
                x_loc = x[i: i + 1] * loc.float().detach()
                x_loc = F.adaptive_avg_pool2d(x_loc, (1, 1)) * (
                    x.shape[2] * x.shape[3] / max(torch.sum(loc).item(), 1)
                )
                x_stack[char_box_num * i + k] = x_loc[0]
    return x_stack


def get_global_alpha(
        alpha_list,
        char_rectangles,
        text_indexes,
        height,
        width,
        dev: torch.device = None):
    batch_size = 1
    rendered_alpha_global = torch.zeros(
        batch_size, 1, height, width).float().to(dev)
    loc_alpha = torch.zeros(batch_size, 1, height, width).float().to(dev)

    char_box_num = char_rectangles[0].shape[0]
    for c in range(char_box_num):
        text_id = int(text_indexes[0][c])
        x1, y1, x2, y2, x3, y3, x4, y4 = char_rectangles[0][c]
        ys = int(min(y1, y2, y3, y4))
        ye = int(max(y1, y2, y3, y4))
        xs = int(min(x1, x2, x3, x4))
        xe = int(max(x1, x2, x3, x4))
        h = ye - ys
        w = xe - xs
        ys = ys - h // 2
        ye = ye + h - h // 2
        xs = xs - w // 2
        xe = xe + w - w // 2
        if (ye - ys < 1) | (xe - xs < 1):
            continue
        dys = max(0, ys) - ys
        dxs = max(0, xs) - xs
        dye = max(height, ye) - height
        dxe = max(width, xe) - width
        ah, aw = alpha_list[c].shape[2:4]
        rendered_alpha_global[
            0, 0, ys + dys: ye - dye, xs + dxs: xe - dxe
        ] += alpha_list[c][0, 0, dys: ah - dye, dxs: aw - dxe]
        loc = alpha_list[c][0, 0, dys: ah - dye, dxs: aw - dxe] > 0
        loc_alpha[0, 0, ys + dys: ye - dye, xs +
                  dxs: xe - dxe][loc] = text_id + 1
    return rendered_alpha_global, loc_alpha


def compute_rgbmap_and_compositing(img, bg, alpha):
    alpha_tile = alpha.repeat(1, 3, 1, 1)
    alpha_div = alpha.clone()[alpha == 0] = 1
    # compute rgb map
    # print(alpha.shape, img.shape, bg.shape)
    rgb_map = (img - bg * (1 - alpha_tile)) / (alpha_div + 1e-5)
    # normalization
    rgb_map = torch.max(
        torch.min(rgb_map, torch.zeros_like(img) + 255), torch.zeros_like(img)
    )
    assert (torch.max(rgb_map) <= 256) | (
        torch.min(rgb_map) >= 0), "error rgb value"
    # composition
    img_composited = bg * (1 - alpha_tile) + rgb_map * (alpha_tile)
    return rgb_map, img_composited


def adaptive_threshold(mask, alpha):
    prior = mask.float().detach() * alpha
    u = torch.mean(prior[mask == 1]).item()
    std = torch.std(prior[mask == 1]).item()
    threshold = u + std * 1
    return threshold


def compositer(
    alpha_outs,
    rendered_alpha_outs,
    effect_visibility_outs,
    img,
    bg_img,
    colors,
    text_ins_mask,
    dev: torch.device = None
):
    (
        rendered_fill_alpha,
        rendered_stroke_alpha,
        rendered_shadow_alpha,
        fill_alpha_loc,
        stroke_alpha_loc,
        shadow_alpha_loc,
    ) = rendered_alpha_outs
    shadow_visibility, stroke_visibility = effect_visibility_outs
    shadow_visibility = F.softmax(shadow_visibility[0], 1)
    stroke_visibility = F.softmax(stroke_visibility[0], 1)
    shadow_visibility_db = torch.reciprocal(
        1 + torch.exp(-50 * (shadow_visibility - 0.5))
    )
    stroke_visibility_db = torch.reciprocal(
        1 + torch.exp(-50 * (stroke_visibility - 0.5))
    )
    if colors is not None:
        fill_colors, shadow_colors, stroke_colors = colors
    batch_size, _, height, width = alpha_outs.shape
    img = (img * 0.5 + 0.5) * 255
    bg_img = (bg_img * 0.5 + 0.5) * 255
    text_ins_mask[text_ins_mask == 255] = -1
    color_map_fill = torch.zeros(
        alpha_outs.shape[0],
        3,
        alpha_outs.shape[2],
        alpha_outs.shape[3]).to(dev)
    color_map_shadow = torch.zeros(
        alpha_outs.shape[0],
        3,
        alpha_outs.shape[2],
        alpha_outs.shape[3]).to(dev)
    color_map_stroke = torch.zeros(
        alpha_outs.shape[0],
        3,
        alpha_outs.shape[2],
        alpha_outs.shape[3]).to(dev)
    color_map_ones = torch.zeros(
        1,
        3,
        alpha_outs.shape[2],
        alpha_outs.shape[3]).to(dev) + 1
    # print(alpha_outs.shape)
    fill_alpha, shadow_alpha, stroke_alpha = torch.split(alpha_outs, 1, 1)

    fill_alpha = torch.max(
        fill_alpha - stroke_alpha,
        torch.zeros_like(fill_alpha))
    shadow_alpha = torch.max(
        shadow_alpha - fill_alpha - stroke_alpha,
        torch.zeros_like(shadow_alpha))

    rendered_fill_alpha_tile = rendered_fill_alpha.repeat(1, 3, 1, 1)
    rendered_shadow_alpha_tile = rendered_shadow_alpha.repeat(1, 3, 1, 1)
    rendered_stroke_alpha_tile = rendered_stroke_alpha.repeat(1, 3, 1, 1)
    rgb_map_shadow, img_shadow = compute_rgbmap_and_compositing(
        img, bg_img, shadow_alpha
    )
    rgb_map_fill, img_fill = compute_rgbmap_and_compositing(
        img, img_shadow, fill_alpha)
    rgb_map_stroke, _ = compute_rgbmap_and_compositing(
        img, img_fill, stroke_alpha)

    reconstruction = bg_img.clone()
    for i in range(batch_size):
        text_num = int(torch.max(text_ins_mask[i]).item())
        if text_ins_mask.is_cuda:
            fill_colors_pred = torch.zeros(1, text_num, 3).to(dev).float()
            shadow_colors_pred = torch.zeros(1, text_num, 3).to(dev).float()
            stroke_colors_pred = torch.zeros(1, text_num, 3).to(dev).float()
        else:
            fill_colors_pred = torch.zeros(1, text_num, 3).float()
            shadow_colors_pred = torch.zeros(1, text_num, 3).float()
            stroke_colors_pred = torch.zeros(1, text_num, 3).float()
        for k in range(text_num):
            loc_text = text_ins_mask[i: i + 1] == (k + 1)
            loc_fill = fill_alpha_loc[i: i + 1] == (k + 1)
            loc_shadow = shadow_alpha_loc[i: i + 1] == (k + 1)
            loc_stroke = stroke_alpha_loc[i: i + 1] == (k + 1)
            loc_all = loc_fill | loc_shadow | loc_stroke
            if torch.sum(loc_text).item() > 0:
                if colors is not None:
                    rgb_fill = (
                        fill_colors[i, k].view(1, 3, 1, 1) * 255
                    )
                    rgb_shadow = (
                        shadow_colors[i, k].view(1, 3, 1, 1) * 255
                    )
                    rgb_stroke = (
                        stroke_colors[i, k].view(1, 3, 1, 1) * 255
                    )
                else:
                    fill_color_loc = loc_text & (
                        fill_alpha[i: i + 1]
                        > adaptive_threshold(loc_text, fill_alpha[i: i + 1])
                    )
                    shadow_color_loc = loc_text & (
                        shadow_alpha[i: i + 1]
                        > adaptive_threshold(loc_text, shadow_alpha[i: i + 1])
                    )
                    stroke_color_loc = loc_text & (
                        stroke_alpha[i: i + 1]
                        > adaptive_threshold(loc_text, stroke_alpha[i: i + 1])
                    )

                    rgb_fill = rgb_map_fill[i: i + 1] * \
                        fill_color_loc.float().detach()
                    rgb_fill = F.adaptive_avg_pool2d(rgb_fill, (1, 1)) * (
                        height * width / max(torch.sum(fill_color_loc.detach()).item(), 1)
                    )
                    rgb_shadow = rgb_map_shadow[i: i + 1] * \
                        shadow_color_loc.float().detach()
                    rgb_shadow = F.adaptive_avg_pool2d(rgb_shadow, (1, 1)) * (
                        height * width / max(torch.sum(shadow_color_loc.detach()).item(), 1)
                    )
                    rgb_stroke = rgb_map_stroke[i: i + 1] * \
                        stroke_color_loc.float().detach()
                    rgb_stroke = F.adaptive_avg_pool2d(rgb_stroke, (1, 1)) * (
                        height * width / max(torch.sum(stroke_color_loc.detach()).item(), 1)
                    )
                    fill_colors_pred[i, k] = rgb_fill[0, :, 0, 0]
                    shadow_colors_pred[i, k] = rgb_shadow[0, :, 0, 0]
                    stroke_colors_pred[i, k] = rgb_stroke[0, :, 0, 0]
                color_map_fill[i: i + 1, 0:1][loc_text ==
                                              1] = rgb_fill[0, 0, 0, 0]
                color_map_fill[i: i + 1, 1:2][loc_text ==
                                              1] = rgb_fill[0, 1, 0, 0]
                color_map_fill[i: i + 1, 2:3][loc_text ==
                                              1] = rgb_fill[0, 2, 0, 0]
                color_map_shadow[i: i + 1, 0:1][loc_text ==
                                                1] = rgb_shadow[0, 0, 0, 0]
                color_map_shadow[i: i + 1, 1:2][loc_text ==
                                                1] = rgb_shadow[0, 1, 0, 0]
                color_map_shadow[i: i + 1, 2:3][loc_text ==
                                                1] = rgb_shadow[0, 2, 0, 0]
                color_map_stroke[i: i + 1, 0:1][loc_text ==
                                                1] = rgb_stroke[0, 0, 0, 0]
                color_map_stroke[i: i + 1, 1:2][loc_text ==
                                                1] = rgb_stroke[0, 1, 0, 0]
                color_map_stroke[i: i + 1, 2:3][loc_text ==
                                                1] = rgb_stroke[0, 2, 0, 0]

                color_fill = color_map_ones * rgb_fill
                color_shadow = color_map_ones * rgb_shadow
                color_stroke = color_map_ones * rgb_stroke
                shadow_prior = (
                    rendered_shadow_alpha_tile[i: i + 1]
                    * loc_shadow.float().detach()
                    * shadow_visibility_db[k, 1].view(1)
                )
                tmp_bg0 = (reconstruction[i: i +
                                          1].clone().detach() *
                           (1 -
                            shadow_prior) +
                           color_shadow *
                           shadow_prior)
                fill_prior = (
                    rendered_fill_alpha_tile[i: i + 1] * loc_fill.float().detach()
                )
                tmp_bg1 = tmp_bg0 * (1 - fill_prior) + color_fill * fill_prior
                stroke_prior = (
                    rendered_stroke_alpha_tile[i: i + 1]
                    * loc_stroke.float().detach()
                    * stroke_visibility_db[k, 1].view(1)
                )
                tmp_bg2 = tmp_bg1 * (1 - stroke_prior) + \
                    color_stroke * stroke_prior
                reconstruction[i: i + 1][loc_all.repeat(1, 3, 1, 1)] = tmp_bg2[
                    loc_all.repeat(1, 3, 1, 1)
                ]

    colors_pred = fill_colors_pred, shadow_colors_pred, stroke_colors_pred
    return (
        (color_map_fill, color_map_shadow, color_map_stroke),
        reconstruction,
        colors_pred,
    )


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
    return kernel, psize


def gfilter(inp, sigma, dim=2, dev: torch.device = None):
    kernel, pad_size = compute_gaussian_kernel(inp, sigma, dim=2, dev=dev)
    pad_size *= 3
    inp = F.pad(inp, (pad_size, pad_size, pad_size, pad_size))
    out = F.conv2d(inp, kernel)
    out = F.conv2d(out, kernel)
    out = F.conv2d(out, kernel)
    return out


class AlphaRenderer(nn.Module):
    def __init__(
            self,
            fontnum: int = 100,
            font_topk: int = 20,
            dev: torch.device = None):
        super().__init__()
        self.dev = dev
        self.fontnum = fontnum
        self.topk = font_topk
        self.topk_char = 1
        self.prerendered_alpha = np.load(
            os.path.join(
                get_prerendered_alpha_dir(),
                "prerendered_alpha_fill_100.npy"))

    def get_prerendered_alpha(self, topk_list, topk_char_list):
        alpha = self.prerendered_alpha[topk_list.cpu(), topk_char_list.cpu()]
        alpha = alpha.reshape((self.topk, self.topk_char, 64, 64))
        alpha = torch.from_numpy(alpha).to(self.dev).float() / 255.
        return alpha

    def forward(self, font_pred, char_labels, char_rec_vec, text_indexes):
        font_pred = font_pred.view(
            font_pred.shape[0] * font_pred.shape[1], font_pred.shape[2]
        )
        font_pred_sort = torch.sort(input=font_pred, dim=1, descending=True)[1]
        font_pred_sfm = torch.nn.Softmax(dim=1)(font_pred)
        if char_rec_vec.shape[0] == 0:
            return []
        char_rec_vec = char_rec_vec.view(
            char_rec_vec.shape[0], char_rec_vec.shape[1])
        char_pred_sort = torch.sort(
            input=char_rec_vec, dim=1, descending=True)[1]
        char_pred_sfm = torch.nn.Softmax(dim=1)(char_rec_vec)

        alpha_list = []
        for i in range(len(char_labels[0])):
            text_index = int(text_indexes[0][i])
            topk_list = font_pred_sort[text_index, 0: self.topk]
            font_pred_sfm_broad = font_pred_sfm[text_index, topk_list]
            font_pred_sfm_broad = font_pred_sfm_broad.view(
                self.topk, 1, 1, 1).repeat(
                1, self.topk_char, 64, 64)
            topk_char_list = char_pred_sort[i, 0: self.topk_char]
            # char_pred_sfm_broad = char_pred_sfm[i, topk_char_list]
            # char_pred_sfm_broad = char_pred_sfm_broad.view(
            #     1, self.topk_char, 1, 1
            # ).repeat(self.topk, 1, 64, 64)
            alpha = self.get_prerendered_alpha(topk_list, topk_char_list)
            alpha = font_pred_sfm_broad * alpha
            alpha = torch.sum(
                alpha.contiguous().view(self.topk * self.topk_char, 64, 64), 0
            )
            alpha_list.append(alpha.view(1, 1, 64, 64))
        return alpha_list


class StrokeAlphaRenderer(nn.Module):
    def __init__(self, dev: torch.device = None):
        super().__init__()
        self.dev = dev
        self.fontnum = 100
        self.topk = 1
        self.topk_char = 1
        self.topk_stroke = 5
        self.prerendered_alpha_stroke = np.load(
            os.path.join(
                get_prerendered_alpha_dir(),
                "prerendered_alpha_stroke_100.npy"))

    def get_prerendered_alpha(self, topk_list, topk_char_list):
        alpha = self.prerendered_alpha_stroke[topk_list.cpu(
        ), :, topk_char_list.cpu()]
        alpha = alpha.reshape(
            (self.topk, self.topk_stroke, self.topk_char, 64, 64))
        alpha = torch.from_numpy(alpha).to(self.dev).float() / 255.
        return alpha

    def forward(
            self,
            font_pred,
            stroke_param_pred,
            char_labels,
            char_rec_vec,
            text_indexes):
        font_pred = font_pred.view(
            font_pred.shape[0] * font_pred.shape[1], font_pred.shape[2]
        )
        font_pred_sort = torch.sort(input=font_pred, dim=1, descending=True)[1]
        font_pred_sfm = torch.nn.Softmax(dim=1)(font_pred)
        stroke_param_pred = stroke_param_pred.view(
            stroke_param_pred.shape[0] * stroke_param_pred.shape[1],
            stroke_param_pred.shape[2],
        )
        stroke_param_pred_sfm = torch.nn.Softmax(dim=1)(stroke_param_pred)
        if char_rec_vec.shape[0] == 0:
            return []
        char_rec_vec = char_rec_vec.view(
            char_rec_vec.shape[0], char_rec_vec.shape[1])
        char_pred_sort = torch.sort(
            input=char_rec_vec, dim=1, descending=True)[1]

        alpha_list = []
        for i in range(len(char_labels[0])):
            text_index = int(text_indexes[0][i])
            topk_char_list = char_pred_sort[i, 0: self.topk_char]
            topk_list = font_pred_sort[text_index, 0: self.topk]
            stroke_param_pred_sfm_broad = stroke_param_pred_sfm[text_index, :]
            stroke_param_pred_sfm_broad = stroke_param_pred_sfm_broad.view(
                1, self.topk_stroke, 1, 1, 1
            ).repeat(self.topk, 1, self.topk_char, 64, 64)

            alpha = self.get_prerendered_alpha(topk_list, topk_char_list)
            alpha = stroke_param_pred_sfm_broad * alpha

            alpha = torch.sum(
                alpha.view(
                    self.topk *
                    self.topk_char *
                    self.topk_stroke,
                    64,
                    64),
                0)
            alpha_list.append(alpha)
        return alpha_list


class AffineTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmp = 0
        self.pad_size = 32

    def forward(
            self,
            affine_outs,
            fill_alpha_list,
            stroke_alpha_list,
            targetsize_list):
        affine_00, affine_11, affine_01, affine_10, affine_02, affine_12 = torch.split(
            affine_outs, 1, 1)
        affine_00 = torch.sigmoid(affine_00) * 2
        affine_11 = torch.sigmoid(affine_11) * 2
        affine_01 = torch.tanh(affine_01) * 2
        affine_10 = torch.tanh(affine_10) * 2
        affine_02 = torch.tanh(affine_02) * 1
        affine_12 = torch.tanh(affine_12) * 1
        fill_alpha_affine_list = []
        stroke_alpha_affine_list = []
        for i in range(len(fill_alpha_list)):
            # affine_s = affine_sigmoid_pred[i].view(1,2,2)
            # affine_t = affine_tanh_pred[i].view(1,2,1)
            row_one = torch.cat(
                (
                    affine_00[i, 0].view(1, 1, 1),
                    affine_01[i, 0].view(1, 1, 1),
                    affine_02[i, 0].view(1, 1, 1),
                ),
                2,
            )
            row_two = torch.cat(
                (
                    affine_10[i, 0].view(1, 1, 1),
                    affine_11[i, 0].view(1, 1, 1),
                    affine_12[i, 0].view(1, 1, 1),
                ),
                2,
            )
            theta = torch.cat((row_one, row_two), 1)
            # theta = torch.cat((affine_s,affine_t),2)
            target_h = max(int(targetsize_list[0][i][0]) * 2, 1)
            target_w = max(int(targetsize_list[0][i][1]) * 2, 1)
            grid = F.affine_grid(
                theta, (1, 1, target_h, target_w), align_corners=False)
            fill_alpha = F.pad(
                fill_alpha_list[i].view(1, 1, 64, 64),
                (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            )
            stroke_alpha = F.pad(
                stroke_alpha_list[i].view(1, 1, 64, 64),
                (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            )
            fill_alpha_affine = torch.nn.functional.grid_sample(
                fill_alpha, grid)
            stroke_alpha_affine = torch.nn.functional.grid_sample(
                stroke_alpha, grid)
            fill_alpha_affine_list.append(fill_alpha_affine)
            stroke_alpha_affine_list.append(stroke_alpha_affine)
        return fill_alpha_affine_list, stroke_alpha_affine_list


class ShadowAlphaTransformer(nn.Module):
    def __init__(self, dev: torch.device = None):
        super().__init__()
        self.kernel_factor = 3.0 / 4.0 * math.sqrt(2 * math.pi)
        self.coefficient_blur = 0.5
        self.coefficient_offset = 0.2
        self.dev = dev

    def forward(
        self,
        alpha_global,
        font_size_pred,
        shadow_param_sig_pred,
        shadow_param_tanh_pred,
        text_array,
    ):
        shadow_param_sig_pred = shadow_param_sig_pred.view(
            shadow_param_sig_pred.shape[0] * shadow_param_sig_pred.shape[1],
            shadow_param_sig_pred.shape[2],
        )
        shadow_param_tanh_pred = shadow_param_tanh_pred.view(
            shadow_param_tanh_pred.shape[0] * shadow_param_tanh_pred.shape[1],
            shadow_param_tanh_pred.shape[2],
        )
        # font_size_pred = font_size_pred.view(
        #     font_size_pred.shape[0] * font_size_pred.shape[1], font_size_pred.shape[2]
        # )
        if shadow_param_sig_pred.is_cuda:
            zero_tensor = torch.Tensor([0.0]).float().to(self.dev)
            one_tensor = torch.Tensor([1.0]).float().to(self.dev)
        else:
            zero_tensor = torch.Tensor([0.0]).float()
            one_tensor = torch.Tensor([1.0]).float()
        shadow_alpha_global = torch.zeros_like(alpha_global)
        shadow_alpha_loc = torch.zeros_like(alpha_global)
        _, _, height, width = alpha_global.shape
        for i in range(len(alpha_global)):
            # print(len(alpha_global),len(text_array))
            for k in range(len(text_array[i])):
                text_index = k
                text_box = text_array[i][k]
                x1, y1, x2, y2, x3, y3, x4, y4 = text_box
                ys = int(min(y1, y2, y3, y4))
                ye = int(max(y1, y2, y3, y4))
                xs = int(min(x1, x2, x3, x4))
                xe = int(max(x1, x2, x3, x4))
                h = ye - ys
                w = xe - xs
                font_size = h
                op = 1.0  # shadow_param_sig_pred[text_index,0]
                blur = (
                    (shadow_param_sig_pred[text_index, 1] + 1e-5)
                    * font_size
                    * self.coefficient_blur
                )
                kernel_factor = 3.0 / 4.0 * math.sqrt(2 * math.pi)
                kernel_size = max(
                    2,
                    math.floor(
                        blur.item() *
                        kernel_factor +
                        0.5))
                # print(blur, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                psize = (kernel_size // 2) * 3
                dx_pixel_num = (
                    shadow_param_tanh_pred[text_index, 0]
                    * font_size
                    * self.coefficient_offset
                )
                dy_pixel_num = (
                    shadow_param_tanh_pred[text_index, 1]
                    * font_size
                    * self.coefficient_offset
                )
                if (h < 5) | (w < 5):
                    continue
                expand_h = psize + abs(dy_pixel_num.item())
                expand_w = psize + abs(dx_pixel_num.item())
                # print(x1, y1, x2, y2, x3, y3, x4, y4)
                ys = int(max(min(ys - expand_h, height), 0))
                ye = int(max(min(ye + expand_h, height), 0))
                xs = int(max(min(xs - expand_w, width), 0))
                xe = int(max(min(xe + expand_w, width), 0))
                alpha_shadow = alpha_global[i: i + 1, :, ys:ye, xs:xe]
                alpha_shadow = gfilter(
                    alpha_shadow.detach(), blur, dev=self.dev)
                alpha_shadow = alpha_shadow * op
                dy = dy_pixel_num / alpha_shadow.shape[2] * 2
                dx = dx_pixel_num / alpha_shadow.shape[3] * 2
                row_one = torch.cat(
                    (one_tensor, zero_tensor, -1 * dx.view(1)), 0).view(1, 3)
                row_two = torch.cat(
                    (zero_tensor, one_tensor, -1 * dy.view(1)), 0).view(1, 3)
                theta_shift = torch.cat((row_one, row_two), 0)
                grid = F.affine_grid(
                    theta_shift.unsqueeze(0),
                    (1, 1, alpha_shadow.shape[2], alpha_shadow.shape[3]),
                    align_corners=False,
                )
                alpha_shadow = torch.nn.functional.grid_sample(
                    alpha_shadow, grid)
                shadow_alpha_global[i, :, ys:ye, xs:xe] = alpha_shadow[0]
                shadow_alpha_loc[i, :, ys:ye,
                                 xs:xe][alpha_shadow[0] > 0] = k + 1
        return shadow_alpha_global, shadow_alpha_loc
