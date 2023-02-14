from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import pyclipper
import skia
import torch
from logzero import logger as log
from shapely.geometry import Polygon
from torch.functional import F

from src.io import load_font_dicts
from src.dto.dto_postprocess import MetaDataPostprocessing, TextBlobParameter
from src.skia_lib.skia_util import get_font_path
from src.dto.dto_skia import(
    FontData,
    TextFormData
)


@dataclass
class AffineParameter:
    """Affine parameters matrix [2x3]"""

    # diagonal elements > 0
    affine_00: Any
    affine_01: Any
    affine_02: Any
    affine_10: Any
    affine_11: Any
    affine_12: Any

    def normalization(self):
        self.affine_00 = torch.sigmoid(self.affine_00) * 2
        self.affine_11 = torch.sigmoid(self.affine_11) * 2
        self.affine_01 = torch.tanh(self.affine_01) * 2
        self.affine_10 = torch.tanh(self.affine_10) * 2
        self.affine_02 = torch.tanh(self.affine_02) * 1
        self.affine_12 = torch.tanh(self.affine_12) * 1

    def get_theta(self):
        row_one = torch.cat(
            (
                self.affine_00[0, 0].view(1, 1, 1),
                self.affine_01[0, 0].view(1, 1, 1),
                self.affine_02[0, 0].view(1, 1, 1),
            ),
            2,
        )
        row_two = torch.cat(
            (
                self.affine_10[0, 0].view(1, 1, 1),
                self.affine_11[0, 0].view(1, 1, 1),
                self.affine_12[0, 0].view(1, 1, 1),
            ),
            2,
        )
        theta = torch.cat((row_one, row_two), 1)
        return theta


fontmgr = skia.FontMgr()
paint_text = skia.Paint(
    AntiAlias=True,
    Color=skia.Color(255, 0, 0),
)


def get_affine_transformed_box(
    affine_out: np.ndarray,
    target_h: float,
    target_w: float,
    org_box: Union[List, np.ndarray],
):
    xs, ys, _, _ = get_min_max_xy(org_box)
    locy, locx = np.where(affine_out == 1)
    if len(locx) == 0 | len(locy) == 0:
        return org_box
    pad_h = float(target_h) // 4  # 32
    pad_w = float(target_w) // 4
    xs_affine = np.min(locx)
    xe_affine = np.max(locx)
    ys_affine = np.min(locy)
    ye_affine = np.max(locy)
    x1 = xs_affine + xs - pad_w
    x2 = xs_affine + xs - pad_w
    x3 = xe_affine + xs - pad_w
    x4 = xe_affine + xs - pad_w
    y1 = ys_affine + ys - pad_h
    y2 = ye_affine + ys - pad_h
    y3 = ye_affine + ys - pad_h
    y4 = ys_affine + ys - pad_h
    affine_box = (x1, y1, x2, y2, x3, y3, x4, y4)
    return affine_box


def get_min_max_xy(box: Union[List, np.ndarray]):
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    ys = min(y1, y2, y3, y4)
    xs = min(x1, x2, x3, x4)
    ye = max(y1, y2, y3, y4)
    xe = max(x1, x2, x3, x4)
    return xs, ys, xe, ye


def compute_affine_transform(
    affine_params: AffineParameter,
    target_h: float,
    target_w: float,
    pad_size=32,
    dev: torch.device = None
):
    theta = affine_params.get_theta()
    grid = F.affine_grid(
        theta, (1, 1, target_h, target_w), align_corners=False)
    box = torch.zeros(1, 1, 64, 64).float().to(dev) + 1
    box_affine = F.pad(box, (pad_size, pad_size, pad_size, pad_size))
    output = torch.nn.functional.grid_sample(box_affine, grid, mode="nearest")
    output = output.data.cpu().numpy()[0, 0]
    return output


def get_affine_transformed_boxes(
    char_rectangles: Union[List, np.ndarray],
    char_sizes: List,
    affine_pred: torch.Tensor,
    dev: torch.device = None
):
    new_boxes = []
    affine_00, affine_11, affine_01, affine_10, affine_02, affine_12 = torch.split(
        affine_pred, 1, 1)
    affine_params = AffineParameter(
        affine_00, affine_01, affine_02, affine_10, affine_11, affine_12
    )
    affine_params.normalization()
    for i in range(len(char_rectangles)):
        org_box = char_rectangles[i]
        target_h = int(char_sizes[i][0]) * 2  # 128
        target_w = int(char_sizes[i][1]) * 2
        if (target_h < 5) | (target_w < 5):
            new_boxes.append(org_box)
            continue
        affine_out = compute_affine_transform(
            affine_params, target_h, target_w, dev=dev)
        box = get_affine_transformed_box(
            affine_out, target_h, target_w, org_box)
        new_boxes.append(box)
    return new_boxes


def get_box_size(font_size: float, ft, text: str):
    if font_size > 100:
        width = 1000
        height = 1000
    else:
        width = 400
        height = 400
    surface_tmp = skia.Surface(width, height)
    canvas_tmp = surface_tmp.getCanvas()
    canvas_tmp.clear(skia.ColorSetRGB(0, 0, 0))
    font = skia.Font(ft, font_size, 1.0, 0.0)
    textblob = skia.TextBlob(text, font)
    canvas_tmp.drawTextBlob(textblob, 10, height - 50, paint_text)
    alpha_tmp = surface_tmp.makeImageSnapshot().toarray()[:, :, 2]
    loc = np.where(alpha_tmp > 0)
    if len(loc[0]) == 0:
        return 0, 0
    ly1, lx1 = np.min(loc[0]), np.min(loc[1])
    my1, mx1 = np.max(loc[0]), np.max(loc[1])
    return my1 - ly1, mx1 - lx1 + 1e-5


def get_difference_and_scale(
    font_size_tmp: float,
    diff: float,
    font_size: float,
    wscale: float,
    boxh: float,
    boxw: float,
    ft,
    text: str,
):
    boxh_tmp, boxw_tmp = get_box_size(font_size_tmp, ft, text)
    diff_tmp = abs(boxh_tmp - boxh)
    if diff_tmp < diff:
        diff = diff_tmp
        font_size = font_size_tmp
        wscale = float(boxw / max(boxw_tmp, 1))
    flag = boxh - boxh_tmp
    return diff, font_size, wscale, flag


def search_font_size_and_wscale(ft, boxh: float, boxw: float, text: float):
    font_size = boxh
    diff = 10000
    wscale = 1
    # larger fontsize
    for i in range(1000):
        font_size_tmp = boxh + i
        diff, font_size, wscale, flag = get_difference_and_scale(
            font_size_tmp, diff, font_size, wscale, boxh, boxw, ft, text
        )
        if flag < 0:
            break
    # smaller font_size
    for i in range(1000):
        font_size_tmp = boxh - i
        diff, font_size, wscale, flag = get_difference_and_scale(
            font_size_tmp, diff, font_size, wscale, boxh, boxw, ft, text
        )
        if flag > 0:
            break
    return font_size, wscale


def get_font_type_face(font_id: int):
    font_dict = load_font_dicts()
    font_path = font_dict[font_id]
    ft = fontmgr.makeFromFile(font_path, 0)
    return ft


def get_font_info(ft, font_size: float, text: str):
    font = skia.Font(ft, font_size, 1.0, 0.0)
    glyphs = font.textToGlyphs(text)
    rects = font.getBounds(glyphs)
    positions = font.getPos(glyphs)
    return font, glyphs, rects, positions


def get_font_param(box: Union[List, np.ndarray], char: str, font_id: int):
    ft = get_font_type_face(font_id)
    xs, ys, xe, ye = get_min_max_xy(box)
    boxh = ye - ys
    boxw = xe - xs
    font_size, wscale = search_font_size_and_wscale(ft, boxh, boxw, char)
    _, _, rects, _ = get_font_info(ft, font_size, char)
    text_top = rects[0].top()
    return font_size, wscale, text_top


@dataclass
class BestCharacterSearch:
    """Optimization parameters"""

    loss: float
    index: int

    def update_index(self, loss, index):
        if loss < self.loss:
            self.loss = loss
            self.index = index


@dataclass
class TextLocationParameter:
    """Optimization parameters"""

    x0: int
    x1: int
    y0: int
    y1: int

    def update_by_character_box(self, x0, y0, x1, y1):
        if x0 < self.x0:
            self.x0 = x0
        if x1 > self.x1:
            self.x1 = x1
        if y0 < self.y0:
            self.y0 = y0
        if y1 > self.y1:
            self.y1 = y1

    def round(self) -> Tuple[int, int, int, int]:
        return (
            np.floor(self.x0).astype(int),
            np.floor(self.y0).astype(int),
            np.ceil(self.x1).astype(int),
            np.ceil(self.y1).astype(int),
        )


def compute_rgb_loss(
    rgb_rec: np.ndarray,
    img_model_outs_size: np.ndarray,
    xs: float,
    ys: float,
    xe: float,
    ye: float,
):
    rec_box = rgb_rec[int(ys): int(ye), int(xs): int(xe)] / 255
    org_box = img_model_outs_size[int(ys): int(ye), int(xs): int(xe)] / 255
    loss = np.mean((rec_box - org_box) ** 2)
    return loss


def search_bestchar(
    t: int,
    affine_boxes: Union[List, np.ndarray],
    charid2textid: Union[List, np.ndarray],
    texts: List,
    rgb_rec: np.ndarray,
    img: np.ndarray,
):
    best_char = BestCharacterSearch(loss=1000, index=-1)
    tlp = TextLocationParameter(x0=img.shape[1], x1=0, y0=img.shape[0], y1=0)
    one_text = ""
    char_boxes_in_text = []
    for c in range(len(affine_boxes)):
        if charid2textid[c] == t:
            # log.debug(c,texts[c])
            box = affine_boxes[c]
            (x1, y1, x2, y2, x3, y3, x4, y4) = box
            xs, ys, xe, ye = get_min_max_xy(box)
            loss = compute_rgb_loss(rgb_rec, img, xs, ys, xe, ye)
            # update
            one_text += texts[c]
            char_boxes_in_text.append(
                (x1, y1, x2, y2, x3, y3, x4, y4, -1 * loss))
            best_char.update_index(loss, c)
            tlp.update_by_character_box(xs, ys, xe, ye)
    char_boxes_in_text = np.array(char_boxes_in_text)
    return one_text, char_boxes_in_text, best_char, tlp


def nms_text(
    t: int,
    character_num: int,
    charid2textid: Union[List, np.ndarray],
    texts: List,
    keep: List,
):
    text_tmp = ""
    cnt = 0
    for c in range(character_num):
        if charid2textid[c] == t:
            if cnt not in keep:
                cnt += 1
                continue
            cnt += 1
            text_tmp += texts[c]
    return text_tmp


def get_wscale(
        font_id: int,
        font_size: float,
        text: str,
        x0: float,
        x1: float):
    ft = get_font_type_face(font_id)
    _, _, rects, positions = get_font_info(ft, font_size, text)
    text_width_tmp = (positions[-1].x() + rects[-1].right()) - (
        positions[0].x() - rects[0].left()
    )
    text_width = x1 - x0
    wscale = float(text_width / max(text_width_tmp, 1e-20))
    return wscale


def resize_yxlist(yxlist: List, model_outs_size: Tuple, img_size: Tuple):
    new_yxlist = []
    resize_rate_y = img_size[1] / model_outs_size[1]
    resize_rate_x = img_size[0] / model_outs_size[0]
    for i in range(len(yxlist)):
        y, x = yxlist[i]
        new_yxlist.append((int(resize_rate_y * y), int(resize_rate_x * x)))
    return new_yxlist


def resize_rectangles(
        inp: np.ndarray,
        model_outs_size: Tuple,
        img_size: Tuple):
    resize_rate_y = float(img_size[1]) / float(model_outs_size[1])
    resize_rate_x = float(img_size[0]) / float(model_outs_size[0])
    inp[:, [0, 2, 4, 6]] = inp[:, [0, 2, 4, 6]] * resize_rate_x
    inp[:, [1, 3, 5, 7]] = inp[:, [1, 3, 5, 7]] * resize_rate_y
    return inp


def resize_items(
    mdp: MetaDataPostprocessing,
    rgb_rec: np.ndarray,
    char_rectangles: np.ndarray,
    char_sizes: List,
):
    img_org_size = tuple(mdp.img_org_size[0])
    rgb_rec = rgb_rec.data.cpu().numpy()[0].transpose(1, 2, 0)
    rgb_rec = cv2.resize(rgb_rec, img_org_size)
    char_rectangles = resize_rectangles(
        char_rectangles.copy(), mdp.model_outs_size, img_org_size
    )
    char_sizes = resize_yxlist(char_sizes, mdp.model_outs_size, img_org_size)
    return rgb_rec, char_rectangles, char_sizes


def nms(
        boxes: np.ndarray,
        overlapThresh: float,
        neighbourThresh=0.5,
        minScore=0,
        num_neig=0):
    new_boxes = np.zeros_like(boxes)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [
        Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
        for b in boxes
    ]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))
    order = boxes[:, 8].argsort()[::-1]
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i + 1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])
                                )
                            )
                    except BaseException:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    if union > 0:
                        iou = inter / union if union > 0 else 0
                        iou_i = inter / areas[i] if areas[i] > 0 else 0
                        if iou_i > overlapThresh:
                            suppressed[j] = True
                        iou_j = inter / areas[j] if areas[j] > 0 else 0
                        if iou_j > overlapThresh:
                            suppressed[j] = True
                        if union > 0 and iou > overlapThresh:
                            suppressed[j] = True
                        if iou > neighbourThresh:
                            neighbours.append(j)
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] -
                               minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(
                    axis=0
                ) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick, new_boxes


def get_textblob_param_with_affine(
    mdp: MetaDataPostprocessing,
    affine_pred: torch.Tensor,
    texts: List,
    font_ids: List,
    rgb_rec: np.ndarray,
    img: np.ndarray,
    dev: torch.device = None
):
    # load data from extracted data
    char_rectangles = mdp.bbox_information.get_char_rectangle()[0]
    char_sizes = mdp.bbox_information.get_char_size()[0]
    charid2textid = mdp.bbox_information.get_charindex2textindex()[0]
    # charindex2charorder = ed.bbox_information.get_charindex2charorder()[0]
    # resize models
    rgb_rec, char_rectangles, char_sizes = resize_items(
        mdp, rgb_rec, char_rectangles, char_sizes
    )
    # get affine transformed boxes
    affine_boxes = get_affine_transformed_boxes(
        char_rectangles, char_sizes, affine_pred[0:1], dev=dev
    )

    # visualize_boxes(affine_boxes, img)
    # get font parameters from affine transformed boxes
    # font_sizes, scales, text_top_list = get_font_params(affine_boxes, font_ids, texts, charindex2textindex)
    # get Text blob parameters for external rendering engine
    text_num = len(font_ids)
    char_num = len(affine_boxes)
    text_blob_parameter_list = []
    for t in range(text_num):
        one_text, char_boxes_in_text, best_char, tlp = search_bestchar(
            t, affine_boxes, charid2textid, texts, rgb_rec, img
        )
        # non maximum suppression
        keep, char_boxes_in_text = nms(char_boxes_in_text, 0.3, num_neig=0)
        text = nms_text(t, char_num, charid2textid, texts, keep)
        target_id = best_char.index
        font_size, wscale, text_top = get_font_param(
            affine_boxes[target_id], texts[target_id], font_ids[t]
        )

        wscale = get_wscale(font_ids[t], font_size, text, tlp.x0, tlp.x1)
        _, ys, _, _ = get_min_max_xy(affine_boxes[target_id])

        if len(text) == 0:
            continue
        font_data_dto = FontData(
            font_size=font_size,
            font_id=font_ids[t],
            font_path=get_font_path(font_ids[t])
        )
        text_form_dto = TextFormData(
            vertical_text_flag=False,
            rotate_text_flag=False,
            angle=0,
            width_scale=wscale,
            text_index = t,
        )
        # size_param,scale,size_paramk,scalek = get_size_param(ft, boxh, boxw, text)
        # wscale = get_wscale(font_ids[t], font_size, text, tlp.x0,tlp.x1)
        # store to list
        text_blob_parameter = TextBlobParameter(
            font_data_dto,
            text_form_dto,
            text_top,
            ys,
            text,
            tlp.round(),
        )
        text_blob_parameter_list.append(text_blob_parameter)
    # fill_alpha = surfacek.makeImageSnapshot().toarray()[:, :, 2]
    # cv2.imwrite("tmp/skia_text.jpg", fill_alpha)
    return text_blob_parameter_list
