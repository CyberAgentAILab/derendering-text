from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
import numpy as np
import cv2
import pickle
import copy
import random
import skia
from src.modules.generator.src.synthtext_lib.synthtext_function import TextRegions, DepthCamera, filter_for_placement
from src.skia_lib import skia_util as sku
from src.skia_lib import skia_paintor as skp
from .dto_skia import(
    FontData,
    TextFormData,
    ShadowParam,
    FillParam,
    GradParam,
    StrokeParam,
    EffectParams,
    EffectVisibility
)


@dataclass
class GeneratorDataInfo:
    load_path: str
    save_path: str
    bg_dir: str
    img_dir: str
    alpha_dir: str
    metadata_dir: str
    prefixes: List

    def load_bg_and_masks(self, index):
        return self.loader.load_bg_and_masks(index)

    def set_loader(self, loader):
        self.loader = loader


@dataclass
class TextGeneratorInputHandler:
    bg: np.ndarray
    instance_per_img: int
    use_homography: bool
    NUM_REP = 1
    n = 0

    def set_synth_text_inputs(self, depth, seg, area, label):
        self.depth = depth
        self.seg = seg
        self.area = area
        self.label = label
        self.TR = TextRegions()
        xyz = DepthCamera.depth2xyz(depth)
        regions = self.TR.get_regions(xyz, seg, area, label)
        if self.use_homography:
            self.regions = filter_for_placement(xyz, seg, regions)
            self.nregions = len(regions['place_mask'])
        else:
            masks = []
            for idx, l in enumerate(regions['label']):
                mask = (seg == l).astype(np.float32)
                mask = mask * -1 + 1
                mask *= 255
                mask = mask.astype(np.uint8)
                masks.append(mask)
        self.set_mask(masks)

    def set_mask(self, seg):
        self.seg = seg

    def set_collision_mask(self, reg_idx, aug_idx, idx):
        ireg = reg_idx[aug_idx[idx]]
        self.collision_mask = self.place_masks[ireg].copy()
        if self.use_homography:
            self.H = self.regions['homography'][ireg]
            self.Hinv = self.regions['homography_inv'][ireg]

    def update_collision_mask(self, box_mask, reg_idx, aug_idx, idx):
        self.collision_mask += (255 *
                                (box_mask > 0).astype('float32')).astype('uint8')
        ireg = reg_idx[aug_idx[idx]]
        self.place_masks[ireg] = self.collision_mask.copy()
        self.n += 1

    def get_img_size(self):
        return self.bg.shape[0], self.bg.shape[1]

    def get_homography(self):
        return self.H, self.Hinv

    def get_loop_items(self):
        if self.use_homography:
            loop_items = self.get_loop_items_with_synth_text_rule()
        else:
            loop_items = self.get_loop_items_with_default_rule()
        return loop_items

    def get_loop_items_with_default_rule(self):
        self.place_masks = []
        for i in range(len(self.seg)):
            self.place_masks.append(self.seg[i].copy())
        num_txt_regions = len(self.seg)
        reg_idx = np.arange(num_txt_regions)
        np.random.shuffle(reg_idx)
        NUM_REP = random.randint(1, 5)
        aug_idx = np.arange(NUM_REP * num_txt_regions) % num_txt_regions
        return reg_idx, aug_idx

    def get_loop_items_with_synth_text_rule(self):
        self.place_masks = copy.deepcopy(self.regions['place_mask'])
        # np.arange(nregions)#min(nregions, 5*ninstance*self.max_text_regions))
        m = self.TR.get_num_text_regions(self.nregions)
        reg_idx = np.arange(min(2 * m, self.nregions))
        np.random.shuffle(reg_idx)
        reg_idx = reg_idx[:m]
        num_txt_regions = len(reg_idx)
        NUM_REP = 5
        aug_idx = np.arange(NUM_REP * num_txt_regions) % num_txt_regions
        return reg_idx, aug_idx


@dataclass
class RenderingData:
    textblob: skia.Font
    offsets: List[float]
    textset_offsets: List[float]
    effect_visibility: List[bool]
    effect_params: List[Tuple]
    paints: List[skia.Paint]
    alpha: List
    angle: float

    def unpack(self):
        self.unpack_offsets()
        self.unpack_textset_offsets()
        self.unpack_visibility_flags()
        self.unpack_effect_params()
        self.upack_shadow_params()
        self.upack_grad_params()
        self.unpack_paints()
        self.unpack_alpha()

    def unpack_offsets(self):
        self.offset_y, self.offset_x = self.offsets

    def unpack_textset_offsets(self):
        self.textset_offset_y, self.textset_offset_x = self.offsets

    def unpack_visibility_flags(self):
        self.shadow_visibility_flag, self.fill_visibility_flag, self.gardation_visibility_flag, self.stroke_visibility_flag = self.effect_visibility

    def unpack_effect_params(self):
        self.shadow_param, self.fill_param, self.grad_param, self.stroke_param = self.effect_params

    def upack_shadow_params(self):
        self.op, self.bsz, self.dp, self.theta, self.shift, self.shadow_offset_y, self.shadow_offset_x, self.shadow_color = self.shadow_param

    def upack_grad_params(self):
        self.grad_mode, self.grad_blend_mode, self.points, self.colors, self.colorstop = self.grad_param
        
    def unpack_paints(self):
        self.shadow_paint, self.fill_paint, self.stroke_paint, self.grad_paint = self.paints

    def unpack_alpha(self):
        self.shadow_alpha, self.fill_alpha, self.stroke_alpha, self.shadow_bitmap = self.alpha

    def get_alpha(self):
        return self.shadow_alpha, self.fill_alpha, self.stroke_alpha


@dataclass
class TrainingFormatData:
    """vectorized data for training model"""
    bg: np.ndarray  # color pixels for background
    img: np.ndarray
    alpha: np.ndarray
    charBB: np.ndarray
    wordBB: np.ndarray
    texts: List
    font_data: List[FontData]
    text_form_data: List[TextFormData]
    effect_params: List[EffectParams]
    effect_visibility: List[EffectVisibility]
    text_offsets: List[Tuple[float, float]]
    char_offsets: List[Tuple[float, float]]
    text_pivots: List[Tuple[float, float]]

    def del_large_volume_elements(self):
        self.bg = None
        self.img = None
        self.alpha = None

    def set_deleted_file_names(self, bg, img, alpha):
        self.bg = bg
        self.img = img
        self.alpha = alpha
    
    def add_effect_merged_alphaBB(self):
        mask_size = (self.bg.shape[0], self.bg.shape[1])
        text_num = len(self.texts)
        merged_alphaBB = []
        for t in range(text_num):
            # access to metadata for drawing alpha maps
            font_path = self.font_data[t].font_path
            font_size = self.font_data[t].font_size
            skia_font_object = sku.load_font_by_skia_format(font_size, font_path)
            textblob = skia.TextBlob(self.texts[t], skia_font_object)
            offsets = self.text_offsets[t]
            effect_params = self.effect_params[t].get_data()
            shadow_param, fill_param, _, stroke_param = effect_params
            fill_paint = skp.get_fill_paint(fill_param)
            shadow_paint = skp.get_shadow_paint(shadow_param)
            stroke_paint = skp.get_stroke_paint(stroke_param)
            grad_paint = None
            paints = shadow_paint, fill_paint, grad_paint, stroke_paint
            pivot_offsets = self.text_pivots[t]
            angle = self.text_form_data[t].angle
            
            # get alpha maps for each text boxes
            alpha = skp.get_alpha(mask_size, textblob, offsets, effect_params, paints, pivot_offsets, angle)
            alpha = skp.alpha_with_visibility(alpha, self.effect_visibility[t].get_data())
            shadow_alpha, fill_alpha, stroke_alpha, _ = alpha
            def get_box_from_alpha(alpha):
                imgEdge,_ = cv2.findContours(alpha.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                imgEdge =np.concatenate(imgEdge, 0)
                rect = cv2.minAreaRect(imgEdge)
                box = cv2.boxPoints(rect).reshape((8,))
                box_yx = np.zeros((2,4),dtype=np.float32)
                for j in range(4):
                    box_yx[0,j] = box[2*j+1]
                    box_yx[1,j] = box[2*j]
                return box_yx
            # convert alpha maps to a box
            merged_alpha = shadow_alpha+fill_alpha+stroke_alpha/3
            box = get_box_from_alpha(merged_alpha)
            merged_alphaBB.append(box)
            
        # set boxes to the key
        self.effect_merged_alphaBB = np.asarray(merged_alphaBB).transpose(1,2,0)
            
