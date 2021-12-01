
from PIL import Image
import numpy as np
import random
import cv2
from shapely.geometry import Polygon
import torch.nn.functional as F
import os
import torch


def mask2png(saven, mask, get_mask=False):
    palette = get_palette(256)
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    mask.save(saven)
    if get_mask:
        return cv2.imread(saven)
    else:
        return 0


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class RandomResizeWithBB():
    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, inputs):
        input_imgs, input_rectangles = inputs
        img = input_imgs[0]
        target_long = random.randint(self.min_long, self.max_long)
        h, w, c = img.shape
        shorter = min(h, w)
        resize_rate = float(target_long) / shorter
        xsize = int(resize_rate * w)
        ysize = int(resize_rate * h)
        output_imgs = []
        for inp in input_imgs:
            if inp.dtype == np.int32:
                out = cv2.resize(
                    inp, (target_long, target_long), interpolation=cv2.INTER_NEAREST)
            else:
                out = cv2.resize(inp, (target_long, target_long))
            if len(out.shape) == 2:
                out = np.expand_dims(out, 2)
            output_imgs.append(out)
        out_rectangles = []
        for inp in input_rectangles:
            inp[:, [0, 2, 4, 6]] = inp[:, [0, 2, 4, 6]] * resize_rate
            inp[:, [1, 3, 5, 7]] = inp[:, [1, 3, 5, 7]] * resize_rate
            out_rectangles.append(inp)
        return output_imgs, out_rectangles


class RandomCropWithBB():
    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, inputs):
        input_imgs, input_rectangles = inputs
        imgarr = np.concatenate(input_imgs, axis=-1)
        h, w, c = imgarr.shape
        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)
        w_space = w - self.cropsize
        h_space = h - self.cropsize
        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space + 1)
        else:
            cont_left = random.randrange(-w_space + 1)
            img_left = 0
        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space + 1)
        else:
            cont_top = random.randrange(-h_space + 1)
            img_top = 0

        output_imgs = []
        for inp in input_imgs:
            container = np.zeros(
                (self.cropsize, self.cropsize, inp.shape[-1]), np.float32)
            container[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
                inp[img_top:img_top + ch, img_left:img_left + cw]
            output_imgs.append(container)
        out_rectangles = []
        for inp in input_rectangles:
            inp[:, [0, 2, 4, 6]] -= img_left
            inp[:, [1, 3, 5, 7]] -= img_top
            inp = np.minimum(inp, cw)
            inp = np.maximum(inp, 0)
            out_rectangles.append(inp)
        return output_imgs, out_rectangles


class RandomCropWithPolygon():
    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, input_imgs, input_rectangles, mask_id=6, max_try=50):
        imgarr = np.concatenate(input_imgs, axis=-1)
        textmask = input_imgs[mask_id]
        h, w, c = imgarr.shape
        full_size = np.array((w, h))
        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)
        w_space = w - self.cropsize
        h_space = h - self.cropsize
        for i in range(max_try):
            if w_space > 0:
                cont_left = 0
                img_left = random.randrange(w_space + 1)
            else:
                cont_left = random.randrange(-w_space + 1)
                img_left = 0
            if h_space > 0:
                cont_top = 0
                img_top = random.randrange(h_space + 1)
            else:
                cont_top = random.randrange(-h_space + 1)
                img_top = 0
            container = np.zeros((self.cropsize, self.cropsize, 1), np.float32)
            container[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
                textmask[img_top:img_top + ch, img_left:img_left + cw]
            if np.sum(container == 1) > 0:
                break
        crop_area = np.array(
            (img_top, img_top + ch, img_left, img_left + cw))  # y0,y1,x0,x1
        crop_size = np.array((cw, ch))  # y0,y1,x0,x1
        output_imgs = []
        for inp in input_imgs:
            container = np.zeros(
                (self.cropsize, self.cropsize, inp.shape[-1]), np.float32)
            container[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
                inp[img_top:img_top + ch, img_left:img_left + cw]
            output_imgs.append(container)
        out_rectangles = []
        for inp in input_rectangles:
            inp[:, [0, 2, 4, 6]] -= img_left
            inp[:, [0, 2, 4, 6]] += cont_left
            inp[:, [1, 3, 5, 7]] -= img_top
            inp[:, [1, 3, 5, 7]] += cont_top
            inp = np.minimum(inp, cw)
            inp = np.maximum(inp, 0)
            out_rectangles.append(inp)
        return output_imgs, out_rectangles


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)
        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
        return proc_img


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def adjust_learning_rate(
        lr,
        epoch,
        lr_rampdown_epochs,
        step_in_epoch,
        total_steps_in_epoch):
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    def cosine_rampdown(current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
    lr *= cosine_rampdown(epoch, lr_rampdown_epochs)
    return lr


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / \
        max(np.linalg.norm(p2 - p1), 1)


def shrink_poly(poly, r, R=0.3):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2(
            (poly[1][1] - poly[0][1]),
            (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2(
            (poly[2][1] - poly[3][1]),
            (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2(
            (poly[3][0] - poly[0][0]),
            (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2(
            (poly[2][0] - poly[1][0]),
            (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2(
            (poly[3][0] - poly[0][0]),
            (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2(
            (poly[2][0] - poly[1][0]),
            (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2(
            (poly[1][1] - poly[0][1]),
            (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2(
            (poly[2][1] - poly[3][1]),
            (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly
