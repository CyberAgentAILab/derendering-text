import os
import cv2
import numpy as np
import skia
from src.skia_lib import skia_util as sku
from util.path_list import get_prerendered_alpha_dir
from src.io import load_char_label_dicts
from logzero import logger as log


def crop_alpha_area(alpha):
    loc = np.where(alpha > 0)
    if len(loc[0]) == 0:
        return None
    ly1, lx1 = np.min(loc[0]), np.min(loc[1])
    my1, mx1 = np.max(loc[0]), np.max(loc[1])
    alpha_crop = alpha[ly1:my1, lx1:mx1]
    return alpha_crop


def gen_fill_pams(
        char_dict,
        font_num=100,
        font_size=100,
        width=400,
        height=400):

    paint_text = skia.Paint(
        AntiAlias=True,
        Color=skia.Color(255, 0, 0),
    )

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    character_num = len(char_dict)
    alpha_dic = np.zeros((font_num, character_num, 64, 64), dtype=np.float32)
    for f in range(font_num):
        for i in range(character_num):
            canvas.clear(skia.ColorSetRGB(0, 0, 0))
            char_target = char_dict[i].split("")[0]
            font = sku.get_textfont(f, font_size)
            textblob = sku.get_textblob(char_target, font)
            canvas.drawTextBlob(textblob, 100, 300, paint_text)
            alpha = surface.makeImageSnapshot().toarray()[
                :, :, 2]  # extract alpha of R
            alpha_crop = crop_alpha_area(alpha)
            if alpha_crop is None:
                continue
            if alpha_crop.shape[0]==0 or alpha_crop.shape[1]==0: 
                continue
            alpha_crop = cv2.resize(
                alpha_crop, (64, 64), interpolation=cv2.INTER_CUBIC)
            alpha_dic[f, i] = alpha_crop
        if f % 20 == 0:
            log.debug(f"progress: {f}/{font_num}")
    np.save(
        os.path.join(
            get_prerendered_alpha_dir(),
            "prerendered_alpha_fill_100.npy"),
        alpha_dic)


def gen_stroke_pams(
        char_dict,
        font_num=100,
        param_num=5,
        font_size=100,
        width=400,
        height=400):
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    character_num = len(char_dict)
    alpha_stroke_dic = np.zeros(
        (font_num, param_num, character_num, 64, 64), dtype=np.float32
    )
    for f in range(font_num):
        for p in range(param_num):
            for i in range(character_num):
                canvas.clear(skia.ColorSetRGB(0, 0, 0))
                char_target = char_dict[i].split("")[0]
                font = sku.get_textfont(f, font_size)
                textblob = sku.get_textblob(char_target, font)
                stroke_width = (p * 0.2 * float(font_size) / 25.0) + 0.05
                paint_text = skia.Paint(
                    AntiAlias=True,
                    Color=skia.ColorRED,
                    Style=skia.Paint.kStroke_Style,
                    StrokeWidth=stroke_width,
                )
                canvas.drawTextBlob(textblob, 100, 300, paint_text)
                alpha = surface.makeImageSnapshot().toarray()[:, :, 2]
                alpha_crop = crop_alpha_area(alpha)
                if alpha_crop is None:
                    continue
                if alpha_crop.shape[0]==0 or alpha_crop.shape[1]==0: 
                    continue
                alpha_crop = cv2.resize(
                    alpha_crop, (64, 64), interpolation=cv2.INTER_CUBIC)
                alpha_stroke_dic[f, p, i] = alpha_crop
        if f % 20 == 0:
            log.debug(f"progress: {f}/{font_num}")
    np.save(
        os.path.join(
            get_prerendered_alpha_dir(),
            "prerendered_alpha_stroke_100.npy"),
        alpha_stroke_dic)


def main():
    log.debug("load character config")
    char_dict, label_dict = load_char_label_dicts()
    os.makedirs(get_prerendered_alpha_dir(), exist_ok=True)
    log.debug("generate pre-rendered alpha maps for fill")
    gen_fill_pams(char_dict)
    log.debug("generate pre-rendered alpha maps for stroke")
    gen_stroke_pams(char_dict)
    log.debug("finish")


if __name__ == '__main__':
    main()
