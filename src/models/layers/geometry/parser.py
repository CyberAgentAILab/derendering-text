import math
from typing import List

import numpy as np
from logzero import logger as log
from shapely.geometry import Polygon

from src.dto.dto_model import WordInstance
from .nms import nms, nms_with_char_cls


def parse_word_bboxes(
    pred_word_fg,
    pred_word_tblr,
    pred_word_orient,
    W,
    H,
    nms_th=0.15,
    fg_th=0.95,
    scales=(1, 1),
):
    scale_w, scale_h = scales
    word_stride = 4
    word_keep_rows, word_keep_cols = np.where(pred_word_fg > fg_th)
    oriented_word_bboxes = np.zeros(
        (word_keep_rows.shape[0], 9), dtype=np.float32)
    for idx in range(oriented_word_bboxes.shape[0]):
        # foregrounds -> x,y
        y, x = word_keep_rows[idx], word_keep_cols[idx]
        # geomap
        t, b, l, r = pred_word_tblr[:, y, x]
        o = pred_word_orient[y, x]
        score = pred_word_fg[y, x]
        # 4 * left point * scale
        pointxl = scale_w * word_stride * (x - l)
        pointyt = scale_h * word_stride * (y - t)
        pointxr = scale_w * word_stride * (x + r)
        pointyb = scale_h * word_stride * (y + b)
        centerx = scale_w * word_stride * x
        centery = scale_h * word_stride * y
        four_points = rotate_rect(
            pointxl, pointyt, pointxr, pointyb, o, centerx, centery
        )
        oriented_word_bboxes[idx, :8] = np.array(
            four_points, dtype=np.float32).flat
        oriented_word_bboxes[idx, 8] = score
    keep, oriented_word_bboxes = nms(oriented_word_bboxes, nms_th, num_neig=1)
    oriented_word_bboxes = oriented_word_bboxes[keep]
    oriented_word_bboxes[:, :8] = oriented_word_bboxes[:, :8].round()
    oriented_word_bboxes[:, 0:8:2] = np.maximum(
        0, np.minimum(W - 1, oriented_word_bboxes[:, 0:8:2])
    )
    oriented_word_bboxes[:, 1:8:2] = np.maximum(
        0, np.minimum(H - 1, oriented_word_bboxes[:, 1:8:2])
    )
    return oriented_word_bboxes


def parse_char(
    pred_word_fg,
    pred_char_fg,
    pred_char_tblr,
    pred_char_orient,
    pred_char_cls,
    W,
    H,
    num_char_class=68,
    nms_th=0.3,
    fg_th=(0.95, 0.25),
    scales=(1, 1),
):
    scale_w, scale_h = scales
    char_stride = 4
    char_keep_rows, char_keep_cols = np.where(
        (pred_word_fg > fg_th[0]) & (pred_char_fg > fg_th[1])
    )

    oriented_char_bboxes = np.zeros(
        (char_keep_rows.shape[0], 9), dtype=np.float32)
    char_scores = np.zeros(
        (char_keep_rows.shape[0],
         num_char_class),
        dtype=np.float32)
    for idx in range(oriented_char_bboxes.shape[0]):
        y, x = char_keep_rows[idx], char_keep_cols[idx]
        t, b, l, r = pred_char_tblr[:, y, x]
        yr = scale_h * char_stride * y
        xr = scale_w * char_stride * x
        # o = 0.0  # pred_char_orient[y, x]
        o = pred_char_orient[y, x]
        score = pred_char_fg[y, x]
        four_points = rotate_rect(
            xr + char_stride * (-l),
            yr + char_stride * (-t),
            xr + char_stride * (r),
            yr + char_stride * (b),
            o,
            xr,
            yr,
        )
        oriented_char_bboxes[idx, :8] = np.array(
            four_points, dtype=np.float32).flat
        oriented_char_bboxes[idx, 8] = score
        char_scores[idx, :] = pred_char_cls[:, y, x]
    keep, oriented_char_bboxes, char_scores = nms_with_char_cls(
        oriented_char_bboxes, char_scores, nms_th, num_neig=0
    )
    oriented_char_bboxes = oriented_char_bboxes[keep]
    oriented_char_bboxes[:, :8] = oriented_char_bboxes[:, :8].round()
    oriented_char_bboxes[:, 0:8:2] = np.maximum(
        0, np.minimum(W - 1, oriented_char_bboxes[:, 0:8:2])
    )
    oriented_char_bboxes[:, 1:8:2] = np.maximum(
        0, np.minimum(H - 1, oriented_char_bboxes[:, 1:8:2])
    )
    char_scores = char_scores[keep]
    return oriented_char_bboxes, char_scores


def rotate_rect(x1, y1, x2, y2, degree, center_x, center_y):
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    new_points = list()
    for point in points:
        dx = point[0] - center_x
        dy = point[1] - center_y
        new_x = center_x + dx * math.cos(degree) - dy * math.sin(degree)
        new_y = center_y + dx * math.sin(degree) + dy * math.cos(degree)
        new_points.append([(new_x), (new_y)])
    return new_points


def extract_bboxes_from_word_instance(word_instance):
    bboxes = []
    for i in range(len(word_instance)):
        bbox = word_instance[i].word_bbox
        bboxes.append(bbox)
    return np.array(bboxes)


def parse_words(
        word_bboxes,
        char_bboxes,
        char_scores,
        char_dict) -> List[WordInstance]:
    def match(word_bbox, word_poly, char_bbox, char_poly):
        word_xs = word_bbox[0:8:2]
        word_ys = word_bbox[1:8:2]
        char_xs = char_bbox[0:8:2]
        char_ys = char_bbox[1:8:2]
        if (
            char_xs.min() >= word_xs.max()
            or char_xs.max() <= word_xs.min()
            or char_ys.min() >= word_ys.max()
            or char_ys.max() <= word_ys.min()
        ):
            return 0
        else:
            inter = char_poly.intersection(word_poly)
            return inter.area / \
                max((char_poly.area + word_poly.area - inter.area), 1)

    def rematch(word_bbox, word_poly, char_bbox, char_poly):
        word_xs = np.mean(word_bbox[0:8:2])
        word_ys = np.mean(word_bbox[1:8:2])
        char_xs = np.mean(char_bbox[0:8:2])
        char_ys = np.mean(char_bbox[1:8:2])

        dist = (word_xs - char_xs) ** 2 + (word_ys - char_ys) ** 2
        return float(-dist)

    def decode(char_scores):
        max_indices = char_scores.argmax(axis=1)
        text = [char_dict[int(idx)] for idx in max_indices]
        # log.debug(text)
        scores = [char_scores[idx, max_indices[idx]]
                  for idx in range(max_indices.shape[0])]
        return "".join(text), np.array(scores, dtype=np.float32).mean()

    def recog(word_bbox, char_bboxes, char_scores):
        word_vec = np.array([1, 0], dtype=np.float32)
        char_vecs = (char_bboxes.reshape((-1, 4, 2)) -
                     word_bbox[0:2]).mean(axis=1)
        proj = char_vecs.dot(word_vec)
        order = np.argsort(proj)
        text, score = decode(char_scores[order])
        return text, None, char_scores[order], char_bboxes[order]

    word_bbox_scores = word_bboxes[:, 8]
    word_bboxes = word_bboxes[:, :8]
    char_bboxes = char_bboxes[:, :8]
    word_polys = [
        Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])])
        for b in word_bboxes
    ]
    char_polys = [
        Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])])
        for b in char_bboxes
    ]
    num_word = word_bboxes.shape[0]
    num_char = char_bboxes.shape[0]
    word_instances = list()
    word_chars = [list() for _ in range(num_word)]
    if num_word > 0:
        for idx in range(num_char):
            char_bbox = char_bboxes[idx]
            char_poly = char_polys[idx]
            match_scores = np.zeros((num_word,), dtype=np.float32)
            for jdx in range(num_word):
                word_bbox = word_bboxes[jdx]
                word_poly = word_polys[jdx]
                match_scores[jdx] = match(
                    word_bbox, word_poly, char_bbox, char_poly)
            jdx = np.argmax(match_scores)
            if match_scores[jdx] > 0:
                word_chars[jdx].append(idx)
            else:
                match_scores = np.zeros((num_word,), dtype=np.float32)
                for jdx in range(num_word):
                    word_bbox = word_bboxes[jdx]
                    word_poly = word_polys[jdx]
                    match_scores[jdx] = rematch(
                        word_bbox, word_poly, char_bbox, char_poly
                    )
                jdx = np.argmax(match_scores)
                word_chars[jdx].append(idx)
    for idx in range(num_word):
        char_indices = word_chars[idx]
        if len(char_indices) > 0:
            text, text_score, tmp_char_scores, cbox = recog(
                word_bboxes[idx], char_bboxes[char_indices], char_scores[char_indices])
            word_instances.append(
                WordInstance(
                    word_bboxes[idx],
                    word_bbox_scores[idx],
                    text,
                    text_score,
                    tmp_char_scores,
                    char_bboxes=cbox,
                )
            )
    return word_instances
