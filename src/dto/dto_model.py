from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class WordInstance:
    """
    単語ごとのBBoxおよびその情報
    """

    word_bbox: np.ndarray  # TODO: 確かめる
    word_bbox_score: np.float32
    text: str
    text_score: None  # TODO: 確かめる
    char_scores: np.ndarray  # TODO: 確かめる
    char_bboxes: np.ndarray  # TODO: 確かめる


@dataclass
class BBoxInformation:
    """
    BBox情報（各フィールドはバッチサイズ分のリスト）
    TODO: データ構造が若干チグハグなのでなんとかする
    """

    word_instances: List[WordInstance]
    text_instance_mask: np.ndarray  # text mask
    text_rectangle: List  # List for text bounding box array
    char_instance_mask: np.ndarray  # character mask
    char_rectangle: List  # List for character bounding box  array
    char_label: List  # character label for texts
    char_size: List  # each character height and width
    charindex2textindex: List  # character index -> text index for the character
    charindex2charorder: List  # character index -> character order in text


@dataclass
class BatchWrapperBBI:
    bbox_information: List[BBoxInformation]

    def get_word_instances(self):
        return [bbi.word_instances for bbi in self.bbox_information]

    def get_text_instance_mask(self):
        ti_masks = [bbi.text_instance_mask for bbi in self.bbox_information]
        return np.array(ti_masks)

    def get_text_rectangle(self):
        return [bbi.text_rectangle for bbi in self.bbox_information]

    def get_char_instance_mask(self):
        ci_masks = [bbi.char_instance_mask for bbi in self.bbox_information]
        return np.array(ci_masks)

    def get_char_rectangle(self):
        return [bbi.char_rectangle for bbi in self.bbox_information]

    def get_char_label(self):
        return [bbi.char_label for bbi in self.bbox_information]

    def get_char_size(self):
        return [bbi.char_size for bbi in self.bbox_information]

    def get_charindex2textindex(self):
        return [bbi.charindex2textindex for bbi in self.bbox_information]

    def get_charindex2charorder(self):
        return [bbi.charindex2charorder for bbi in self.bbox_information]


@dataclass
class TextInfo:
    """
    OCRにてパースされたテキスト情報
    """

    ocr_outs: Any
    bbox_information: BatchWrapperBBI
    effect_visibility_outs: Any
    effect_param_outs: Any
    font_outs: Any
    font_size_outs: Any
    alpha_outs: Any
