import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.io import load_char_label_dicts
from src.dto.dto_model import BatchWrapperBBI, BBoxInformation
from .parser import parse_char, parse_word_bboxes, parse_words


def get_bbox(ocr_outs, img_size) -> BBoxInformation:
    # configurated variables
    # img_size = cfg.INPUT_SIZE
    text_fg_th = 0.95  # cfg.WORD_MIN_SCORE
    char_fg_th = 0.5  # cfg.CHAR_MIN_SCORE
    char_dict, label_dict = load_char_label_dicts()
    # outputs
    word_out, char_out, recog_out = ocr_outs
    text_fg_pred, text_tblr_pred, text_orient_pred = word_out
    char_fg_pred, char_tblr_pred, char_orient_pred = char_out
    # normalization
    text_fg_pred = F.softmax(text_fg_pred, 1)
    char_fg_pred = F.softmax(char_fg_pred, 1)
    recog_out = F.softmax(recog_out, 1)
    # torch to numpy
    text_fg_pred = text_fg_pred.data.cpu().numpy()
    text_tblr_pred = text_tblr_pred.data.cpu().numpy()
    text_orient_pred = text_orient_pred.data.cpu().numpy()
    char_fg_pred = char_fg_pred.data.cpu().numpy()
    char_tblr_pred = char_tblr_pred.data.cpu().numpy()
    char_orient_pred = char_orient_pred.data.cpu().numpy()
    recog_out = recog_out.data.cpu().numpy()
    # convert outputs to bounding box
    batch_num = text_fg_pred.shape[0]
    bbox_information = []
    for i in range(batch_num):
        # word bounding boxes
        oriented_word_bboxes = parse_word_bboxes(
            text_fg_pred[i, 1],
            text_tblr_pred[i, 0:4],
            text_orient_pred[i, 0],
            img_size[1],
            img_size[0],
        )
        # character bounding boxes
        char_bboxes, char_scores = parse_char(
            text_fg_pred[i, 1],
            char_fg_pred[i, 1],
            char_tblr_pred[i, 0:4],
            char_orient_pred[i, 0],
            recog_out[0],
            img_size[1],
            img_size[0],
            num_char_class=recog_out.shape[1],
            fg_th=(text_fg_th, char_fg_th),
        )
        # merge word and character bounding boxes
        word_instances = parse_words(
            oriented_word_bboxes, char_bboxes, char_scores, char_dict
        )
        # extruct detailed bbox information from word instances
        bbi = extract_bbox_information(word_instances, img_size, label_dict)
        bbox_information.append(bbi)
    bbox_information = BatchWrapperBBI(bbox_information)
    return bbox_information


def get_character_height_and_width(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    ys = int(min(y1, y2, y3, y4))
    ye = int(max(y1, y2, y3, y4))
    xs = int(min(x1, x2, x3, x4))
    xe = int(max(x1, x2, x3, x4))
    height = ye - ys
    width = xe - xs
    return (height, width)


def update_instance_mask(instance_mask, polygon, char_cnt):
    cv2.fillPoly(
        instance_mask,
        polygon,
        char_cnt + 1,
    )


def extract_bbox_information(
        word_instances,
        img_size,
        label_dict,
        max_cnt=200):
    char_cnt = 0
    text_rectangle = []
    char_rectangle = []
    char_labels = []
    character_sizes = []
    charindex2textindex = []
    charindex2charorder = []
    wordlen = len(word_instances)
    text_instance_mask = np.zeros((img_size[0], img_size[1]))
    char_instance_mask = np.zeros((img_size[0], img_size[1]))
    for text_index in range(wordlen):
        char_bboxes = word_instances[text_index].char_bboxes
        char_polygons = char_bboxes[:, 0:8].reshape(len(char_bboxes), 4, 2)
        for char_index in range(len(char_polygons)):
            char_box = char_bboxes[char_index, 0:8]
            char_polygon = char_polygons[char_index: char_index +
                                         1].astype(np.int32)
            character_size = get_character_height_and_width(char_box)
            # update mask
            update_instance_mask(char_instance_mask, char_polygon, char_cnt)
            # update list
            charindex2textindex.append(text_index)
            char_rectangle.append(char_box)
            character_sizes.append(character_size)
            char_labels.append(
                label_dict[word_instances[text_index].text[char_index]])
            charindex2charorder.append(char_index)
            char_cnt += 1
            if char_cnt >= max_cnt:
                break
        text_box = word_instances[text_index].word_bbox
        text_polygon = text_box[0:8].reshape(1, 4, 2).astype(np.int32)
        # update mask
        update_instance_mask(text_instance_mask, text_polygon, text_index)
        # update list
        text_rectangle.append(text_box)
        if char_cnt >= max_cnt:
            break
    # list to numpy
    text_rectangle = np.array(text_rectangle)
    char_rectangle = np.array(char_rectangle)
    # bbox information constructor
    bbi = BBoxInformation(
        word_instances,
        text_instance_mask,
        text_rectangle,
        char_instance_mask,
        char_rectangle,
        char_labels,
        character_sizes,
        charindex2textindex,
        charindex2charorder,
    )
    return bbi


def get_bb_level_features(features_pix, text_instance_mask, training, text_pool_num:int=10, dev: torch.device=None):
    # resize text instance mask
    # print(text_instance_mask.shape)
    text_instance_mask = text_instance_mask.unsqueeze(1).float()
    text_instance_mask = F.interpolate(
        text_instance_mask, features_pix.shape[2:4])
    # mask pooling
    batch_num, channel_num = features_pix.shape[0:2]
    if training:
        bbox_num = [text_pool_num] * batch_num
        features_bb = torch.zeros(
            sum(bbox_num), channel_num, 1, 1).float().to(dev)
    else:
        bbox_num = count_bbox_number(text_instance_mask)
        features_bb = torch.zeros(
            sum(bbox_num), channel_num, 1, 1).float().to(dev)
    bbox_num_index = 0
    for i in range(batch_num):
        for b in range(bbox_num[i]):
            features_bb[bbox_num_index] = mask_pooling(
                features_pix, text_instance_mask, i, b
            )
            bbox_num_index += 1
    return features_bb, sum(bbox_num)


def count_bbox_number(text_instance_mask):
    bbox_num = []
    for i in range(len(text_instance_mask)):
        bbox_num.append(int(torch.max(text_instance_mask[i]).item()))
    return bbox_num


def mask_pooling(features, instance_mask, batch_index, mask_index):
    height, width = features.shape[2:4]
    mask_region = instance_mask[batch_index: batch_index +
                                1] == (mask_index + 1)
    features_mask_refion = (
        features[batch_index: batch_index + 1] * mask_region.float().detach()
    )
    mask_num = torch.sum(mask_region).item()
    features_mask_pooled = F.adaptive_avg_pool2d(features_mask_refion, (1, 1))
    features_mask_pooled_normed = (
        features_mask_pooled * (height * width) / max(mask_num, 1)
    )
    return features_mask_pooled_normed
